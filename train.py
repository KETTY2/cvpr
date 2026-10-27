#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import csv
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui 
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def _camera_forward_world(camera):
    """Camera optical axis in world coordinates (camera looks along local +z)."""
    view_to_world = torch.linalg.inv(camera.world_view_transform)
    direction = view_to_world[2, :3]
    return direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)

def _closest_line_points(q1, d1, q2, d2):
    """Return closest points p1, p2 on two batches of 3D lines, their gap, and validity."""
    w = q1 - q2
    b = torch.dot(d1, d2)
    d = torch.sum(w * d1[None, :], dim=-1)
    e = torch.sum(w * d2[None, :], dim=-1)
    denom = 1.0 - b * b
    safe_denom = denom.clamp_min(1e-8)
    s = (b * e - d) / safe_denom
    t = (e - b * d) / safe_denom
    p1 = q1 + s[:, None] * d1[None, :]
    p2 = q2 + t[:, None] * d2[None, :]
    gap = torch.linalg.vector_norm(p1 - p2, dim=-1)
    valid = (
        (denom > 1e-4)
        & torch.isfinite(p1).all(dim=-1)
        & torch.isfinite(p2).all(dim=-1)
        & torch.isfinite(gap)
    )
    return p1, p2, gap, valid

def _view_loss(render_pkg, camera, opt, depth_l1_weight, iteration, dataset):
    image = render_pkg["render"]
    if camera.alpha_mask is not None:
        image = image * camera.alpha_mask.cuda()
    gt_image = camera.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

    depth_value = 0.0
    if depth_l1_weight(iteration) > 0 and camera.depth_reliable:
        inv_depth = render_pkg["depth"]
        mono_invdepth = camera.invdepthmap.cuda()
        depth_mask = camera.depth_mask.cuda()
        depth_pure = torch.abs((inv_depth - mono_invdepth) * depth_mask).mean()
        loss = loss + depth_l1_weight(iteration) * depth_pure
        depth_value = (depth_l1_weight(iteration) * depth_pure).item()
    return image, gt_image, Ll1, loss, depth_value

def _csv_vec(tensor, index):
    """Return a 3-vector as CSV scalars; blank fields when unavailable."""
    if tensor is None or index < 0 or index >= tensor.shape[0]:
        return ["", "", ""]
    return [float(v) for v in tensor[index].detach().reshape(-1)[:3].cpu().tolist()]

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    debug_csv_path = os.path.join(dataset.model_path, "gaussian_gradient_debug.csv")
    debug_csv_file = open(debug_csv_path, "w", newline="", buffering=1)
    debug_csv = csv.writer(debug_csv_file)
    debug_csv.writerow([
        "iteration", "gaussian_index", "flagged", "history_count", "camera_index",
        "x", "y", "z", "backward_grad_pre_redirect_x", "backward_grad_pre_redirect_y", "backward_grad_pre_redirect_z",
        "line_direction_x", "line_direction_y", "line_direction_z",
        "applied_grad_x", "applied_grad_y", "applied_grad_z",
        "candidate_x", "candidate_y", "candidate_z", "status",
    ])
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    num_cameras = len(viewpoint_stack)
    print(f"number of images: {num_cameras}")
    checkpoint_iterations.append(num_cameras * 101)
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    train_cameras = scene.getTrainCameras()
    camera_forwards = torch.stack([_camera_forward_world(cam).detach().cpu() for cam in train_cameras])

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        paired_mode = (
            iteration >= num_cameras * 30
        )
        if paired_mode:
            viewpoint_cam0=viewpoint_cam
            vind0=vind
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
        
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            camera_indices = [vind0, vind]
            viewpoint_cams = [viewpoint_cam0,viewpoint_cam]
        else: 
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
        
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)
            camera_indices = [vind]
            viewpoint_cams = [viewpoint_cam]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkgs = []
        view_losses = []

        for cam_idx, cam in zip(camera_indices, viewpoint_cams):
            if paired_mode:
                # Give each camera an independent output flag map, then union the flags.
                gaussians._gaussian_scene_num = torch.zeros_like(gaussians._gaussian_scene_num)
            pkg = render(
                cam, gaussians, pipe, bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                num_count=paired_mode,
            )
            image, gt_image, Ll1_view, loss_view, depth_view = _view_loss(
                pkg, cam, opt, depth_l1_weight, iteration, dataset
            )
            pixel_error = (image.detach() - gt_image).abs().mean(dim=0)
            threshold = torch.quantile(pixel_error, 0.95)
            pkg["high_error_pixels"].copy_(pixel_error >= threshold)
            render_pkgs.append(pkg)
            view_losses.append((Ll1_view, loss_view, depth_view))

        Ll1 = torch.stack([item[0] for item in view_losses]).mean()
        loss = torch.stack([item[1] for item in view_losses]).mean()
        Ll1depth = sum(item[2] for item in view_losses) / len(view_losses)
        redirect_params = [
            gaussians._xyz,
            gaussians._scaling,
            gaussians._rotation,
            gaussians._opacity,
            gaussians._features_dc,
            gaussians._features_rest,
        ]
        
        if paired_mode:
            view_losses[0][1].backward(retain_graph=True)
            grads1 = [
                None if p.grad is None else p.grad.detach().clone()
                for p in redirect_params
            ]

            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            view_losses[1][1].backward()
            grads2 = [
                None if p.grad is None else p.grad.detach().clone()
                for p in redirect_params
            ]
        else:
            view_losses[0][1].backward()


        if paired_mode:
            # 
            flag_union = render_pkgs[1]["gaussian_scene"].detach()
            gaussians._gaussian_scene_num = flag_union
            radii = render_pkgs[1]["radii"]
        else:    
            radii = render_pkgs[0]["radii"]


        # num_cameras * 60
        if paired_mode:
            with torch.no_grad():
                for parameter, grad_view1, grad_view2 in zip(redirect_params[1:], grads1[1:], grads2[1:]):
                    if grad_view1 is None and grad_view2 is None:
                        continue
                    if grad_view1 is None:
                        grad_view1 = torch.zeros_like(parameter)
                    if grad_view2 is None:
                        grad_view2 = torch.zeros_like(parameter)
                    parameter_grad =grad_view2
                    if parameter.grad is None:
                        parameter.grad = parameter_grad.detach().clone()
                    else:
                        parameter.grad.copy_(parameter_grad)

            with torch.no_grad():
                active = gaussians._gaussian_scene_num[:, 0] > 0.5
                current_xyz = gaussians._xyz.detach()
                grad1 = grads1[0]
                grad2 = grads2[0]
                g1 = grad1.detach() if grad1 is not None else torch.zeros_like(current_xyz)
                g2 = grad2.detach() if grad2 is not None else torch.zeros_like(current_xyz)
                q1 = current_xyz - g1  # eta = 1
                q2 = current_xyz - g2  # eta = 1
                d1 = camera_forwards[camera_indices[0]].to(device=current_xyz.device, dtype=current_xyz.dtype)
                d2 = camera_forwards[camera_indices[1]].to(device=current_xyz.device, dtype=current_xyz.dtype)
                grad1_norm = torch.linalg.vector_norm(g1, dim=-1)
                grad2_norm = torch.linalg.vector_norm(g2, dim=-1)
                p1, p2, line_gap, line_valid = _closest_line_points(q1, d1, q2, d2)
                target = 0.5 * (p1 + p2)

                visible_both = (render_pkgs[0]["radii"] > 0) & (render_pkgs[1]["radii"] > 0)
                mean_grad_norm = grad2_norm
                target_grad = current_xyz - target
                target_grad_norm = torch.linalg.vector_norm(target_grad, dim=-1)
                usable = (
                    active & visible_both & line_valid
                    & torch.isfinite(g1).all(dim=-1)
                    & torch.isfinite(g2).all(dim=-1)
                    & torch.isfinite(target_grad).all(dim=-1)
                    & (grad1_norm > 1e-12) & (grad2_norm > 1e-12)
                    & (mean_grad_norm > 1e-12) & (target_grad_norm > 1e-8)
                )
                combined_grad_before = gaussians._xyz.grad.detach().clone() if gaussians._xyz.grad is not None else None
                if gaussians._xyz.grad is not None and usable.any():
                    redirected = target_grad[usable] / target_grad_norm[usable].unsqueeze(-1)
                    redirected = redirected * mean_grad_norm[usable].unsqueeze(-1)
                    gaussians._xyz.grad[usable] = redirected

                # Log the first flagged Gaussian in the current pair.
                ids = torch.where(active)[0]
                debug_id = int(ids[0].item()) if ids.numel() else -1
                debug_id_valid = debug_id >= 0
                debug_flagged = int(debug_id_valid)
                debug_xyz = _csv_vec(current_xyz, debug_id) if debug_id_valid else ["", "", ""]
                debug_grad1 = _csv_vec(g1, debug_id) if debug_id_valid else ["", "", ""]
                debug_grad2 = _csv_vec(g2, debug_id) if debug_id_valid else ["", "", ""]
                debug_target = _csv_vec(target, debug_id) if debug_id_valid else ["", "", ""]
                debug_before = (
                    _csv_vec(combined_grad_before, debug_id)
                    if debug_id_valid and combined_grad_before is not None else ["", "", ""]
                )
                debug_status = "no_flagged_gaussian"
                if debug_id_valid:
                    debug_status = "not_usable_or_skipped"
                    if bool(usable[debug_id]):
                        debug_status = "gradient_redirected"
                    elif not bool(visible_both[debug_id]):
                        debug_status = "not_visible_in_both_views"
                    elif not bool(line_valid[debug_id]):
                        debug_status = "nearly_parallel_or_invalid_lines"

            debug_csv.writerow([
                iteration, debug_id, debug_flagged, camera_indices[0], camera_indices[1],
                *debug_xyz, *debug_grad1, *debug_grad2, *debug_target,
                *debug_before,
                *(_csv_vec(gaussians._xyz.grad, debug_id) if debug_id_valid and gaussians._xyz.grad is not None else ["", "", ""]),
                debug_status,
            ])
            debug_csv_file.flush()

#############
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                visible_union = radii > 0
                gaussians.max_radii2D[visible_union] = torch.maximum(
                    gaussians.max_radii2D[visible_union], radii[visible_union]
                )
                for pkg in render_pkgs:
                    gaussians.add_densification_stats(
                        pkg["viewspace_points"], pkg["radii"] > 0
                    )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # Split after the optimizer step so replacing Parameters cannot discard
            # the gradients that were just computed for this iteration.
            # This runs once per active densification iteration, with no seeding.
            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter:
                split_candidates = None
                if paired_mode:
                    # _gaussian_scene_num comes from render_pkgs[1] in paired mode.
                    split_candidates = gaussians._gaussian_scene_num[:, 0] > 0.5
                gaussians.random_split_explore(
                    num_explore=20,
                    candidate_mask=split_candidates,
                )

            # Save after densification and random splitting so the PLY includes
            # the Gaussians created during this iteration.
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    debug_csv_file.close()



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
