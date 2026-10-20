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

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    selected_debug_id = None
    
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

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration >= num_cameras*60:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE , num_count=True)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE , num_count=False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

###
        pixel_error = (image.detach() - gt_image).abs().mean(dim=0)
        threshold = torch.quantile(pixel_error, 0.95)
        render_pkg["high_error_pixels"].copy_(pixel_error >= threshold)
###

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

#########
        # Snapshot the unmodified autograd gradient before applying the fitted direction.
        # Select the first currently flagged Gaussian for this iteration.
        # The selected index is written to the CSV because it may change over time.

        # num_cameras * 60 
        if iteration > num_cameras * 60:
            if selected_debug_id is None:
                flagged_ids = torch.where(
                    gaussians._gaussian_scene_num[:, 0] > 0.5
                )[0]
                if flagged_ids.numel() > 0:
                    selected_debug_id = int(flagged_ids[0].item())

            debug_id = selected_debug_id if selected_debug_id is not None else -1
            debug_id_valid = 0 <= debug_id < gaussians._xyz.shape[0]

            debug_xyz = _csv_vec(gaussians._xyz, debug_id) if debug_id_valid else ["", "", ""]
            debug_raw_grad = (
                _csv_vec(gaussians._xyz.grad, debug_id)
                if debug_id_valid else ["", "", ""]
            )
            debug_quad_grad = ["", "", ""]
            debug_applied_grad = debug_raw_grad.copy()
            debug_candidate = ["", "", ""]

            debug_flagged = int(
                debug_id_valid
                and gaussians._gaussian_scene_num[debug_id, 0].item() > 0.5
            )
            debug_count = (
                int(gaussians.error_history_count[debug_id].item())
                if debug_id_valid else 0
            )
            debug_status = (
                "flagged_waiting_for_history" if debug_flagged
                else "waiting_for_flagged_gaussian" if selected_debug_id is None
                else "selected_gaussian_missing" if not debug_id_valid
                else "selected_gaussian_unflagged"
            )

            with torch.no_grad():
                active = gaussians._gaussian_scene_num[:, 0] > 0.5
                ids = torch.where(active & (radii > 0))[0]
                if ids.numel() > 0 and gaussians._xyz.grad is not None:
                    view = viewpoint_cam.world_view_transform.to(gaussians._xyz.device)
                    R = view[:3, :3]
                    t = view[3, :3]
                    R_inv = torch.linalg.inv(R)
                    xyz = gaussians._xyz.detach()[ids]
                    raw_grad = gaussians._xyz.grad.detach()[ids].clone()

                    # In this codebase points use row vectors: x_cam = x_world @ R + t.
                    xyz_cam = xyz @ R + t
                    grad_cam = raw_grad @ R
                    q_cam = xyz_cam - grad_cam  # requested -gradient candidate location
                    line_origin = (q_cam - t) @ R_inv
                    z_axis = torch.tensor([0., 0., 1.], device=xyz.device, dtype=xyz.dtype)
                    line_dir = (z_axis @ R_inv).expand_as(line_origin)
                    line_dir = torch.nn.functional.normalize(line_dir, dim=-1)

                    K = 5
                    decay = 0.7
                    for local, gid_tensor in enumerate(ids):
                        gid = int(gid_tensor.item())
                        count = int(gaussians.error_history_count[gid].item())
                        ptr = int(gaussians.error_history_ptr[gid].item())
                        target_sum = torch.zeros(3, device=xyz.device, dtype=xyz.dtype)
                        weight_sum = 0.0

                        # Compare the new line with stored lines. Ignore nearly parallel
                        # pairs because their closest points are numerically unstable.
                        for age in range(count):
                            hist_slot = (ptr - 1 - age) % K
                            p2 = gaussians.error_line_origin_history[gid, hist_slot]
                            d2 = gaussians.error_line_dir_history[gid, hist_slot]
                            if torch.linalg.vector_norm(d2) < 1e-8:
                                continue
                            d1 = line_dir[local]
                            w0 = line_origin[local] - p2
                            b = torch.dot(d1, d2)
                            d = torch.dot(d1, w0)
                            e = torch.dot(d2, w0)
                            denom = 1.0 - b * b
                            if denom < 1e-4:
                                continue
                            s_line = (b * e - d) / denom
                            t_line = (e - b * d) / denom
                            c1 = line_origin[local] + s_line * d1
                            c2 = p2 + t_line * d2
                            midpoint = 0.5 * (c1 + c2)
                            weight = decay ** age
                            target_sum += weight * midpoint
                            weight_sum += weight

                        if weight_sum > 0.0:
                            target = target_sum / weight_sum
                            desired_grad = xyz[local] - target
                            desired_norm = torch.linalg.vector_norm(desired_grad)
                            raw_norm = torch.linalg.vector_norm(raw_grad[local])
                            if torch.isfinite(desired_grad).all() and desired_norm > 1e-8 and raw_norm > 1e-12:
                                gaussians._xyz.grad[gid] = desired_grad / desired_norm * raw_norm
                                if gid == debug_id:
                                    debug_quad_grad = [float(v) for v in desired_grad.cpu().tolist()]
                                    debug_applied_grad = _csv_vec(gaussians._xyz.grad, gid)
                                    debug_candidate = [float(v) for v in target.cpu().tolist()]
                                    debug_status = "line_midpoint_gradient_applied"

                        # Insert newest line after comparison so it is not paired with itself.
                        slot = ptr % K
                        gaussians.error_line_origin_history[gid, slot] = line_origin[local]
                        gaussians.error_line_dir_history[gid, slot] = line_dir[local]
                        gaussians.error_camera_history[gid, slot] = vind
                        gaussians.error_history_ptr[gid] = (slot + 1) % K
                        gaussians.error_history_count[gid] = min(count + 1, K)

                if debug_flagged:
                    debug_count = int(gaussians.error_history_count[debug_id].item())

            # num_cameras * 60 
            debug_csv.writerow([
                iteration, debug_id, debug_flagged, debug_count, vind,
                *debug_xyz, *debug_raw_grad, *debug_quad_grad,
                *debug_applied_grad, *debug_candidate, debug_status,
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
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
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
