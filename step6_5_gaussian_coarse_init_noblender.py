#!/usr/bin/env python3
from __future__ import annotations
import shutil
import os
import argparse
from typing import Tuple
from pose_vis_common import draw_pose_axes_bgr

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from noblender_common import (
    Intrinsics,
    build_viewpoint_candidates,
    compose_affine_from_pose,
    compute_gaussian_coarse_loss,
    load_color_bgr,
    load_intrinsics_json_from_dict,
    load_json,
    load_mask,
    overlay_boundaries,
    parse_float_list,
    resize_color_bgr,
    resize_mask,
    rotation_matrix_to_rvec,
    save_depth,
    save_json,
    scale_intrinsics,
    keep_best_component,
    load_depth,
    compute_depth_agreement_score,

)
from render_backend_nvdiffrast import NvdiffrastRenderer

FINAL_DEPTH_WEIGHT = 0.15

def axis_angle_to_matrix_torch(axis_angle: torch.Tensor)->torch.Tensor:
    theta_sq = torch.sum(axis_angle ** 2, dim=-1)
    theta = torch.sqrt(theta_sq + 1e-8)
    half_theta = theta * 0.5
    w = torch.cos(half_theta)
    xyz = torch.sin(half_theta) * (axis_angle / theta)
    x, y, z = xyz[0], xyz[1], xyz[2]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
        2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)
    ]).reshape(3, 3)
    return R

def compose_affine_from_pose_torch(
        R_3x3:torch.Tensor,
        t_xyz_m:torch.Tensor,
        sxyz:torch.Tensor,
        U_3x3: torch.Tensor,
        center_raw_m: torch.Tensor,
        unit_scale:float,
    ) ->Tuple[torch.Tensor,torch.Tensor]:
    u0 = U_3x3[:, 0]
    u1 = U_3x3[:, 1]
    u2 = U_3x3[:, 2]
    M = (
        sxyz[0] * torch.outer(u0, u0) +
        sxyz[1] * torch.outer(u1, u1) +
        sxyz[2] * torch.outer(u2, u2)
    )
    m0 = M[0]
    m1 = M[1]
    m2 = M[2]

    r0 = R_3x3[0, 0] * m0 + R_3x3[0, 1] * m1 + R_3x3[0, 2] * m2
    r1 = R_3x3[1, 0] * m0 + R_3x3[1, 1] * m1 + R_3x3[1, 2] * m2
    r2 = R_3x3[2, 0] * m0 + R_3x3[2, 1] * m1 + R_3x3[2, 2] * m2
    RS = torch.stack([r0, r1, r2], dim=0)
    A_3x3 = RS * float(unit_scale)
    cx, cy, cz = center_raw_m[0], center_raw_m[1], center_raw_m[2]
    b_xyz = t_xyz_m - torch.stack([
        RS[0, 0] * cx + RS[0, 1] * cy + RS[0, 2] * cz,
        RS[1, 0] * cx + RS[1, 1] * cy + RS[1, 2] * cz,
        RS[2, 0] * cx + RS[2, 1] * cy + RS[2, 2] * cz,
    ])
    return A_3x3, b_xyz
def compute_loss_torch(
    render_alpha: torch.Tensor,
    target_mask: torch.Tensor,
    render_depth: torch.Tensor,
    query_depth: torch.Tensor,
    has_depth: bool,
    t: torch.Tensor,
    t_init: torch.Tensor,
    log_s: torch.Tensor,
    log_s_init: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy(render_alpha.clamp(1e-6, 1 - 1e-6), target_mask)
    inter = (render_alpha * target_mask).sum()
    union = render_alpha.sum() + target_mask.sum() - inter + 1e-6
    iou = 1.0 - inter / union
    area_render = render_alpha.sum()
    area_target = target_mask.sum()
    area_diff = torch.abs(area_render - area_target) / (area_target + 1e-6)

    mask_loss = bce + 0.5 * iou + 0.1 * area_diff

    depth_term = render_alpha.new_zeros(())
    if has_depth:
        sigma_depth_m = 0.02
        sigma2 = sigma_depth_m ** 2
        valid = (
            (query_depth > 0.0) &
            (render_depth > 0.0) &
            (target_mask > 0.5)
        ).float()
        weight = valid * render_alpha.clamp(0.0, 1.0)
        weight_sum = weight.sum().clamp_min(1.0)

        valid_count = valid.sum()
        diff2 = (render_depth - query_depth) ** 2
        gauss = torch.exp(-0.5 * diff2 / sigma2)
        gauss_mean = (gauss * weight).sum() / weight_sum

        depth_term = torch.where(
            valid_count >= 20.0,
            FINAL_DEPTH_WEIGHT * (1.0 - gauss_mean),
            render_alpha.new_tensor(FINAL_DEPTH_WEIGHT),
        )

    reg_t = ((t - t_init) ** 2).sum()
    reg_s = ((log_s - log_s_init) ** 2).sum()

    return mask_loss + depth_term + 0.01 * reg_t + 0.01 * reg_s

def compute_depth_penalty_numpy(
    query_depth: np.ndarray,
    render_depth: np.ndarray,
    query_mask: np.ndarray,
    render_mask: np.ndarray,
    sigma_depth_m: float = 0.02,
    min_valid: int = 20,
) -> float:
    valid = (
        query_mask &
        render_mask &
        np.isfinite(query_depth) &
        np.isfinite(render_depth) &
        (query_depth > 0.0) &
        (render_depth > 0.0)
    )
    if int(valid.sum()) < int(min_valid):
        return 1.0
    sigma2 = max(float(sigma_depth_m), 1e-6) ** 2
    mse = float(np.mean((query_depth[valid] - render_depth[valid]) ** 2))
    return float(mse / sigma2)
def compute_exact_final_score_numpy(
    query_mask: np.ndarray,
    query_depth: np.ndarray,
    render_mask: np.ndarray,
    render_depth: np.ndarray,
) -> Tuple[float, dict]:
    mask_loss, detail = compute_gaussian_coarse_loss(
        query_mask=query_mask,
        render_mask=render_mask,
    )

    depth_valid_px = int(((query_depth > 0.0) & query_mask).sum()) if query_depth is not None else 0
    mask_px = int(query_mask.sum())
    use_depth = (depth_valid_px >= 50) and ((depth_valid_px / max(mask_px, 1)) >= 0.15)

    depth_score = 0.0
    depth_penalty = 0.0
    if use_depth:
        depth_score = compute_depth_agreement_score(
            query_depth=query_depth,
            render_depth=render_depth,
            query_mask=query_mask,
            render_mask=render_mask,
            sigma_depth_m=0.02,
        )
        depth_penalty = compute_depth_penalty_numpy(
            query_depth=query_depth,
            render_depth=render_depth,
            query_mask=query_mask,
            render_mask=render_mask,
            sigma_depth_m=0.02,
        )

    depth_term = float(FINAL_DEPTH_WEIGHT * np.log1p(depth_penalty)) if use_depth else 0.0
    total = float(mask_loss + depth_term)

    detail.update({
        'used_query_depth': bool(use_depth),
        'depth_score': float(depth_score),
        'depth_penalty': float(depth_penalty),
        'loss_mask_only': float(mask_loss),
        'loss_depth_term': float(depth_term),
        'loss_total': float(total),
    })
    return total, detail

def evaluate_pose_numpy(
    renderer: NvdiffrastRenderer,
    query_mask: np.ndarray,
    query_color_bgr: np.ndarray,
    query_depth: np.ndarray,
    K: Intrinsics,
    R_3x3: np.ndarray,
    t_xyz_m: np.ndarray,
    sxyz: np.ndarray,
    U_3x3: np.ndarray,
    center_raw_m: np.ndarray,
    unit_scale: float,
) -> Tuple[float, dict, dict]:
    A, b, _ = compose_affine_from_pose(
        R_3x3=R_3x3, t_xyz_m=t_xyz_m, sxyz=sxyz,
        mesh_pca_basis_3x3=U_3x3, mesh_bbox_center_m=center_raw_m, unit_scale=unit_scale,
    )
    rendered = renderer.render(A, b, K) # Nvdiffrast에서 NumPy로 받아오도록 처리되어있다고 가정
    mask_raw = rendered['mask']
    mask = keep_best_component(mask_raw, query_mask)

    rgb = rendered['rgb'].copy()
    rgb[~mask] = 0
    alpha = rendered['alpha'].copy()
    alpha[~mask] = 0
    depth = rendered['depth'].copy()
    depth[~mask] = 0.0

    loss, detail = compute_exact_final_score_numpy(
        query_mask=query_mask,
        query_depth=query_depth,
        render_mask=mask,
        render_depth=depth,
    )
    detail.update({
        'R_3x3': R_3x3.tolist(),
        't_xyz_m': t_xyz_m.tolist(),
        'sx_sy_sz': sxyz.tolist(),
        'A_3x3': A.tolist(),
        'b_xyz': b.tolist(),
    })
    overlay = overlay_boundaries(query_color_bgr, query_mask, mask)
    overlay_raw = overlay_boundaries(query_color_bgr, query_mask, mask_raw)


    alpha_vis = np.clip(rendered['alpha'], 0.0, 1.0)[..., None] * 0.5
    blend = (
    query_color_bgr.astype(np.float32) * (1.0 - alpha_vis)
    + rendered['rgb'].astype(np.float32) * alpha_vis
            ).clip(0, 255).astype(np.uint8)

    iou_val = float(detail.get('iou', 0.0))
    area_val = float(detail.get('area_ratio', 0.0))
    cv2.putText(blend, f"IoU  : {iou_val:.3f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(blend, f"area : {area_val:.3f}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(blend, f"loss : {float(loss):.4f}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return float(loss), detail, {
        'rgb': rgb, 'alpha': alpha, 'mask': mask, 'depth': depth,
        'overlay': overlay, 'blend': blend,   'rgb_raw': rendered['rgb'].copy(),
        'alpha_raw': rendered['alpha'].copy(),
        'mask_raw': mask_raw.copy(),        'overlay_raw': overlay_raw,

    }

def gradient_descent_refine(
    renderer: NvdiffrastRenderer,
    query_mask: np.ndarray,
    query_color_bgr: np.ndarray,
    query_depth: np.ndarray,
    K: Intrinsics,
    R_init: np.ndarray,
    t_init: np.ndarray,
    s_init: np.ndarray,
    U_3x3: np.ndarray,
    center_raw_m: np.ndarray,
    unit_scale: float,
    device: str = "cuda",
    num_iterations: int = 150,
):
    query_depth_tensor = torch.tensor(query_depth, dtype=torch.float32, device=device)
    depth_valid_px = int(((query_depth > 0.0) & query_mask).sum())
    mask_px = int(query_mask.sum())
    has_depth = (depth_valid_px >= 50) and ((depth_valid_px / max(mask_px, 1)) >= 0.15)

    target_mask_tensor = torch.tensor(query_mask, dtype=torch.float32, device=device)
    U_3x3_t = torch.tensor(U_3x3, dtype=torch.float32, device=device)
    center_raw_t = torch.tensor(center_raw_m, dtype=torch.float32, device=device)
    t_init_tensor = torch.tensor(t_init, dtype=torch.float32, device=device)
    log_s_init_tensor = torch.log(torch.tensor(np.clip(s_init, 1e-8, None), dtype=torch.float32, device=device))
    rvec_init = rotation_matrix_to_rvec(R_init)
    rvec_param = torch.nn.Parameter(torch.tensor(rvec_init, dtype=torch.float32, device=device))
    t_param = torch.nn.Parameter(t_init_tensor.clone())
    log_s_param = torch.nn.Parameter(log_s_init_tensor.clone())
    optimizer = torch.optim.Adam([
         {'params': rvec_param, 'lr': 0.05},
         {'params': t_param, 'lr': 0.01},
         {'params': log_s_param, 'lr': 0.02}
    ])
    best_opt_loss = float('inf')
    best_train_loss = float('inf')
    best_state = {
        'R': R_init.copy(),
        't': t_init.copy(),
        's': s_init.copy(),
    }
    best_exact_state = {
        'R': R_init.copy(),
        't': t_init.copy(),
        's': s_init.copy(),
    }
    best_exact_score, _, _ = evaluate_pose_numpy(
        renderer=renderer,
        query_mask=query_mask,
        query_color_bgr=query_color_bgr,
        query_depth=query_depth,
        K=K,
        R_3x3=best_exact_state['R'],
        t_xyz_m=best_exact_state['t'],
        sxyz=best_exact_state['s'],
        U_3x3=U_3x3,
        center_raw_m=center_raw_m,
        unit_scale=unit_scale,
    )


    print(f"[INFO] PyTorch Adam Optimization 시작 (총 {num_iterations}회)")
    for i in range(num_iterations):
        optimizer.zero_grad()

        R_current = axis_angle_to_matrix_torch(rvec_param)
        s_current = torch.exp(log_s_param)

        A_torch, b_torch = compose_affine_from_pose_torch(
            R_3x3=R_current,
            t_xyz_m=t_param,
            sxyz=s_current,
            U_3x3=U_3x3_t,
            center_raw_m=center_raw_t,
            unit_scale=unit_scale
        )

        rendered_tensor = renderer.render_torch(A_torch, b_torch, K)
        render_alpha = rendered_tensor['alpha']
        render_depth = rendered_tensor['depth']

        loss = compute_loss_torch(
            render_alpha=render_alpha,
            target_mask=target_mask_tensor,
            render_depth=render_depth,
            query_depth=query_depth_tensor,
            has_depth=has_depth,
            t=t_param,
            t_init=t_init_tensor,
            log_s=log_s_param,
            log_s_init=log_s_init_tensor,
        )

        current_loss_val = float(loss.detach().item())
        best_opt_loss = min(best_opt_loss, current_loss_val)

        if current_loss_val < best_train_loss:
            best_train_loss = current_loss_val
            with torch.no_grad():
                best_state = {
                    'R': R_current.detach().cpu().numpy().copy(),
                    't': t_param.detach().cpu().numpy().copy(),
                    's': s_current.detach().cpu().numpy().copy(),
                }

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            R_eval = axis_angle_to_matrix_torch(rvec_param).detach().cpu().numpy().copy()
            t_eval = t_param.detach().cpu().numpy().copy()
            s_eval = torch.exp(log_s_param).detach().cpu().numpy().copy()

        current_exact_score, _, _ = evaluate_pose_numpy(
            renderer=renderer,
            query_mask=query_mask,
            query_color_bgr=query_color_bgr,
            query_depth=query_depth,
            K=K,
            R_3x3=R_eval,
            t_xyz_m=t_eval,
            sxyz=s_eval,
            U_3x3=U_3x3,
            center_raw_m=center_raw_m,
            unit_scale=unit_scale,
        )

        if current_exact_score < best_exact_score:
            best_exact_score = current_exact_score
            best_exact_state = {
                'R': R_eval.copy(),
                't': t_eval.copy(),
                's': s_eval.copy(),
            }
        if i % 50 == 0:
            print(
                f"  -> Iter {i:03d} | Train Loss: {current_loss_val:.6f} "
                f"| Best Train: {best_train_loss:.6f}"
            )


    print(f"[INFO] Optimization 완료 | Best Opt Loss: {best_opt_loss:.6f} | Best Train Loss: {best_train_loss:.6f}")


    train_loss_final, train_detail, train_pack = evaluate_pose_numpy(
        renderer=renderer,
        query_mask=query_mask,
        query_color_bgr=query_color_bgr,
        query_depth=query_depth,
        K=K,
        R_3x3=best_state['R'],
        t_xyz_m=best_state['t'],
        sxyz=best_state['s'],
        U_3x3=U_3x3,
        center_raw_m=center_raw_m,
        unit_scale=unit_scale,
    )

    exact_loss_final, exact_detail, exact_pack = evaluate_pose_numpy(
        renderer=renderer,
        query_mask=query_mask,
        query_color_bgr=query_color_bgr,
        query_depth=query_depth,
        K=K,
        R_3x3=best_exact_state['R'],
        t_xyz_m=best_exact_state['t'],
        sxyz=best_exact_state['s'],
        U_3x3=U_3x3,
        center_raw_m=center_raw_m,
        unit_scale=unit_scale,
    )

    print(
                f"  -> Iter {i:03d} | Train Loss: {current_loss_val:.6f} "
                f"| Best Train: {best_train_loss:.6f} "
                f"| Best Exact: {best_exact_score:.6f}"

    )


    return train_loss_final, train_detail, train_pack

def main():
    ap = argparse.ArgumentParser()


    ap.add_argument('--capture_dir', required=True)
    ap.add_argument('--config', default='pipeline_config.json')
    ap.add_argument('--top_k_mesh', type=int, default=3)
    ap.add_argument('--n_view_dirs', type=int, default=24)
    ap.add_argument('--roll_list', default='0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270')
    ap.add_argument('--max_scale_hyp', type=int, default=4)
    ap.add_argument('--render_scale', type=float, default=0.5)
    
    ap.add_argument('--num_iterations', type=int, default=400, help='Adam 최적화 반복 횟수')
    ap.add_argument('--device', default='cuda', help='사용할 디바이스 (cuda 또는 cpu)') # Device 인자 추가
    args = ap.parse_args()
    cfg = load_json(args.config)
    n_view_dirs = int(cfg.get('coarse_init_n_view_dirs', args.n_view_dirs))
    roll_list_raw = cfg.get('coarse_init_roll_list', args.roll_list)
    max_scale_hyp = int(cfg.get('coarse_init_max_scale_hyp', args.max_scale_hyp))
    render_scale = float(cfg.get('coarse_init_render_scale', args.render_scale))
    num_iterations = int(cfg.get('coarse_init_num_iterations', args.num_iterations))
    top_k_mesh = int(cfg.get('coarse_init_top_k_mesh', args.top_k_mesh))
    phase1_chunk_size = int(cfg.get('coarse_init_phase1_chunk_size', 64))


    calib_json_path = os.path.join(args.capture_dir, 'calib_prep', 'calib_input.json')
    scale_json_path = os.path.join(args.capture_dir, 'anisotropic_scale_hypothesis', 'summary.json')
    for p in [calib_json_path, scale_json_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    out_dir = os.path.join(args.capture_dir, 'gaussian_coarse_init')
    os.makedirs(out_dir, exist_ok=True)

    calib = load_json(calib_json_path)
    scale_init = load_json(scale_json_path)
    mesh_entries = scale_init.get('meshes', [])
    if not mesh_entries:
        raise RuntimeError('step6 summary.json의 meshes가 비어있음')

    query_crop_path = calib['saved']['query_crop_png']
    query_mask_path = calib['saved']['query_mask_png']
    query_depth_path = calib['saved'].get('query_depth_crop_npy')
    color_full_path = os.path.join(args.capture_dir, 'color.png')




    query_color_bgr = load_color_bgr(query_crop_path)
    color_full_bgr = load_color_bgr(color_full_path)
    query_mask = load_mask(query_mask_path)
    if query_depth_path and os.path.exists(query_depth_path):
        query_depth = load_depth(query_depth_path)
    else:
        query_depth = np.zeros(query_mask.shape, dtype=np.float32)

    depth_valid_px = int(((query_depth > 0.0) & query_mask).sum())
    mask_px = int(query_mask.sum())
    depth_valid_ratio = depth_valid_px / max(mask_px, 1)
    has_query_depth = (depth_valid_px >= 50) and (depth_valid_ratio >= 0.15)

    Kc = load_intrinsics_json_from_dict(calib['intrinsics_crop'])
    x0, y0, x1, y1 = map(int, calib['crop_xyxy'])

    if render_scale <= 0.0 or render_scale > 1.0:
        raise ValueError('--render_scale must be in (0,1]')
    if render_scale != 1.0:
        Hs = max(1, int(round(Kc.height * render_scale)))
        Ws = max(1, int(round(Kc.width * render_scale)))
        query_color_search = resize_color_bgr(query_color_bgr, (Hs, Ws))
        query_mask_search = resize_mask(query_mask, (Hs, Ws))
        query_depth_search = cv2.resize(query_depth, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        Ksearch = scale_intrinsics(Kc, render_scale)
    else:
        query_color_search = query_color_bgr
        query_mask_search = query_mask
        query_depth_search = query_depth
        Ksearch = Kc

    t0 = np.asarray(calib['init_translation_t0_xyz_m'], dtype=np.float64)


    requested_device = str(args.device).strip().lower()
    if not requested_device.startswith('cuda'):
        raise RuntimeError(f"This script requires CUDA. Requested device: {args.device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = requested_device



    roll_list = parse_float_list(str(roll_list_raw))
    rot_candidates = build_viewpoint_candidates(int(n_view_dirs), roll_list)
    print(f'[INFO] num meshes      : {len(mesh_entries)}')
    print(f'[INFO] seed rotations : {len(rot_candidates)}')

    phase1_results = []
    target_mask_tensor = torch.tensor(query_mask_search, dtype=torch.float32, device=device)

    query_mask_search_bool = query_mask_search.astype(bool)
    outside_u8 = (~query_mask_search_bool).astype(np.uint8)
    dist_out_np = cv2.distanceTransform(outside_u8, cv2.DIST_L2, 3).astype(np.float32)

    sigma_px = max(1.0, 0.05 * float(max(query_mask_search.shape)))
    dist_pen_np = np.clip(dist_out_np / sigma_px, 0.0, 1.0).astype(np.float32)
    dist_pen_t = torch.tensor(dist_pen_np, dtype=torch.float32, device=device)


    print("[INFO] 렌더링 배치를 구성하여 초기 시드를 탐색합니다...")
    for mesh_index, mesh_entry in enumerate(mesh_entries):
        mesh_path = str(mesh_entry['mesh_path'])
        mesh_unit = str(mesh_entry.get('mesh_unit', 'm'))
        unit_scale = float(mesh_entry.get('mesh_unit_scale_to_meter', 1.0))
        center_raw_m = np.asarray(
            mesh_entry.get('mesh_bbox_center_m', mesh_entry['mesh_pca_stats']['center_xyz']),
            dtype=np.float64
        )
        U_3x3 = np.asarray(mesh_entry['mesh_pca_stats']['pca_basis_3x3'], dtype=np.float64)
        scale_hyps = mesh_entry['scale_hypotheses'][: int(max_scale_hyp)]

        if not scale_hyps:
            print(f'[WARN] mesh_index={mesh_index} scale_hypotheses 비어있음, skip')
            continue

        renderer = NvdiffrastRenderer(mesh_path, device=device)
        candidates = []
        A_list = []
        b_list = []

        t0_t = torch.tensor(t0, dtype=torch.float32, device=device)
        U_3x3_t = torch.tensor(U_3x3, dtype=torch.float32, device=device)
        center_raw_t = torch.tensor(center_raw_m, dtype=torch.float32, device=device)

        for rc in rot_candidates:
            R = np.asarray(rc['R'], dtype=np.float64)
            for sh in scale_hyps:
                sxyz = np.asarray(sh['sx_sy_sz'], dtype=np.float64)

                A, b = compose_affine_from_pose_torch(
                    R_3x3=torch.tensor(R, dtype=torch.float32, device=device),
                    t_xyz_m=t0_t,
                    sxyz=torch.tensor(sxyz, dtype=torch.float32, device=device),
                    U_3x3=U_3x3_t,
                    center_raw_m=center_raw_t,
                    unit_scale=unit_scale
                )
                A_list.append(A)
                b_list.append(b)
                candidates.append({
                    'R_3x3': R.tolist(),
                    'view_dir': rc['view_dir'],
                    'roll_deg': float(rc['roll_deg']),
                    'sx_sy_sz': sxyz.tolist(),
                    'scale_kind': sh.get('kind', 'unknown'),
                })

        A_batch = torch.stack(A_list)
        b_batch = torch.stack(b_list)
        B = A_batch.shape[0]

        with torch.no_grad():
            loss_chunks = []
            for s in range(0, B, phase1_chunk_size):
                e = min(s + phase1_chunk_size, B)
                A_chunk = A_batch[s:e]
                b_chunk = b_batch[s:e]
                res = renderer.render_batch(A_chunk, b_chunk, Ksearch)
                alpha_batch = res['alpha']
                depth_batch = res['depth']
                Bc = alpha_batch.shape[0]
                target_exp = target_mask_tensor.unsqueeze(0).expand(Bc, -1, -1)

                bce = F.binary_cross_entropy(
                    alpha_batch.clamp(1e-6, 1 - 1e-6),
                    target_exp,
                    reduction='none'
                ).mean(dim=(1, 2))

                inter = (alpha_batch * target_exp).sum(dim=(1, 2))
                union = alpha_batch.sum(dim=(1, 2)) + target_exp.sum(dim=(1, 2)) - inter + 1e-6
                iou = 1.0 - inter / union

                query_mask_t = (target_exp > 0.5).float()
                render_mask_t = (alpha_batch > 0.5).float()

                kernel = torch.ones((1, 1, 7, 7), dtype=torch.float32, device=device)
                query_dil = F.conv2d(query_mask_t.unsqueeze(1), kernel, padding=3)
                query_dil = (query_dil > 0).squeeze(1)

                extra_mask = (render_mask_t > 0.5) & (~query_dil)
                query_area = query_mask_t.sum(dim=(1, 2)).clamp_min(1.0)
                extra_ratio = extra_mask.float().sum(dim=(1, 2)) / query_area
                alpha_area = alpha_batch.sum(dim=(1, 2)).clamp_min(1.0)
                spatial_pen = (
                    (alpha_batch * dist_pen_t.unsqueeze(0)).sum(dim=(1, 2))
                    / alpha_area
                )

                if has_query_depth:
                    sigma_depth_m = 0.02
                    sigma2 = sigma_depth_m ** 2

                    query_depth_t = torch.tensor(
                        query_depth_search, dtype=torch.float32, device=device
                    ).unsqueeze(0).expand(Bc, -1, -1)

                    valid = (
                        (query_depth_t > 0.0) &
                        (depth_batch > 0.0) &
                        (target_exp > 0.5) &
                        (alpha_batch > 0.5)
                    )
                    valid_f = valid.float()
                    valid_count = valid_f.sum(dim=(1, 2))

                    diff2 = (query_depth_t - depth_batch) ** 2
                    gauss = torch.exp(-0.5 * diff2 / sigma2)

                    gauss_sum = (gauss * valid_f).sum(dim=(1, 2))
                    gauss_mean = gauss_sum / valid_count.clamp_min(1.0)

                    depth_term = torch.where(
                        valid_count >= 20.0,
                        0.10 * (1.0 - gauss_mean),
                        torch.full_like(valid_count, 0.10, dtype=torch.float32),
                    )
                    loss_chunk = bce + 0.5 * iou + depth_term + 0.1 * extra_ratio + 0.2 * spatial_pen
                else:
                    loss_chunk = bce + 0.5 * iou + 0.1 * extra_ratio + 0.2 * spatial_pen


                loss_chunks.append(loss_chunk)


            loss_batch = torch.cat(loss_chunks, dim=0)

            rerank_k = min(4, B)
            rerank_idx = torch.argsort(loss_batch)[:rerank_k].tolist()

            reranked = []
            for idx in rerank_idx:
                exact_loss, exact_detail, _ = evaluate_pose_numpy(
                    renderer=renderer,
                    query_mask=query_mask_search,
                    query_color_bgr=query_color_search,
                    query_depth=query_depth_search,
                    K=Ksearch,
                    R_3x3=np.asarray(candidates[idx]['R_3x3'], dtype=np.float64),
                    t_xyz_m=t0,
                    sxyz=np.asarray(candidates[idx]['sx_sy_sz'], dtype=np.float64),
                    U_3x3=U_3x3,
                    center_raw_m=center_raw_m,
                    unit_scale=unit_scale,
                )
                reranked.append({
                    'idx': int(idx),
                    'exact_loss': float(exact_loss),
                    'surrogate_loss': float(loss_batch[idx].item()),
                    'exact_detail': exact_detail,
                })

            reranked.sort(key=lambda x: x['exact_loss'])

            topk = min(3, len(reranked))
            topk_items = reranked[:topk]
            topk_idx = [item['idx'] for item in topk_items]

            best_idx = int(topk_idx[0])
            best_seed = candidates[best_idx]
            best_seed_loss = float(topk_items[0]['exact_loss'])

            top_seed_losses = [float(item['exact_loss']) for item in topk_items]
            mean_top3_seed_loss = float(sum(top_seed_losses) / max(len(top_seed_losses), 1))
            mesh_rank_score = 0.7 * float(best_seed_loss) + 0.3 * mean_top3_seed_loss

            top_candidates = []
            for rank, item in enumerate(topk_items, start=1):
                idx = int(item['idx'])
                top_candidates.append({
                    'rank': int(rank),
                    'candidate_index': idx,
                    'loss_total': float(item['exact_loss']),
                    'surrogate_loss_total': float(item['surrogate_loss']),
                    **candidates[idx],
                })



        phase1_results.append({
            'mesh_index': int(mesh_index),
            'mesh_path': mesh_path,
            'mesh_unit': mesh_unit,
            'mean_top3_seed_loss': float(mean_top3_seed_loss),
            'mesh_rank_score': float(mesh_rank_score),
            'unit_scale': float(unit_scale),
            'center_raw_m': center_raw_m.tolist(),
            'U_3x3': U_3x3.tolist(),
            'best_seed': best_seed,
            'best_seed_loss': float(best_seed_loss),
            'top_candidates': top_candidates,
        })

        print(f'[INFO] phase1 best mesh_index={mesh_index} loss={best_seed_loss:.6f}')

        del renderer
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    if not phase1_results:
        raise RuntimeError('No valid mesh result found.')

    phase1_results.sort(key=lambda x: x['mesh_rank_score'])
    top_phase1_results = phase1_results[: int(top_k_mesh)]
    print('[INFO] phase1 overall best mesh_rank_score:', top_phase1_results[0]['mesh_rank_score'])
    print('[INFO] phase1 overall best best_seed_loss:', top_phase1_results[0]['best_seed_loss'])
    print(f'[INFO] refining top {len(top_phase1_results)} meshes from phase1')

    refined_results = []

    for phase1_rank, phase1_entry in enumerate(top_phase1_results, start=1):


        mesh_path_i = phase1_entry['mesh_path']
        mesh_unit_i = phase1_entry['mesh_unit']
        unit_scale_i = float(phase1_entry['unit_scale'])
        center_raw_m_i = np.asarray(phase1_entry['center_raw_m'], dtype=np.float64)
        U_3x3_i = np.asarray(phase1_entry['U_3x3'], dtype=np.float64)
        best_seed_i = phase1_entry['best_seed']
        best_seed_loss_i = float(phase1_entry['best_seed_loss'])

        renderer_i = NvdiffrastRenderer(mesh_path_i, device=device)

        phase1_top_packs_i = []
        for cand in phase1_entry['top_candidates']:
            _, cand_detail, cand_pack = evaluate_pose_numpy(
                renderer=renderer_i,
                query_mask=query_mask_search,
                query_color_bgr=query_color_search,
                query_depth=query_depth_search,
                K=Ksearch,
                R_3x3=np.asarray(cand['R_3x3']),
                t_xyz_m=t0,
                sxyz=np.asarray(cand['sx_sy_sz']),
                U_3x3=U_3x3_i,
                center_raw_m=center_raw_m_i,
                unit_scale=unit_scale_i
            )
            phase1_top_packs_i.append({
                'candidate': cand,
                'detail': cand_detail,
                'pack': cand_pack,
            })
        _, best_seed_detail_i, best_seed_pack_i = evaluate_pose_numpy(
            renderer=renderer_i,
            query_mask=query_mask_search,
            query_color_bgr=query_color_search,
            query_depth=query_depth_search,
            K=Ksearch,
            R_3x3=np.asarray(best_seed_i['R_3x3']),
            t_xyz_m=t0,
            sxyz=np.asarray(best_seed_i['sx_sy_sz']),
            U_3x3=U_3x3_i,
            center_raw_m=center_raw_m_i,
            unit_scale=unit_scale_i
        )

        refined_loss_i, refined_detail_i, refined_pack_i = gradient_descent_refine(
            renderer=renderer_i,
            query_mask=query_mask_search,
            query_color_bgr=query_color_search,
            query_depth=query_depth_search,
            K=Ksearch,
            R_init=np.asarray(best_seed_i['R_3x3'], dtype=np.float32),
            t_init=np.asarray(t0, dtype=np.float32),
            s_init=np.asarray(best_seed_i['sx_sy_sz'], dtype=np.float32),
            U_3x3=np.asarray(U_3x3_i, dtype=np.float32),
            center_raw_m=np.asarray(center_raw_m_i, dtype=np.float32),
            unit_scale=unit_scale_i,
            device=device,
            num_iterations=num_iterations
        )

        print(
            f'[INFO] refine phase1_rank={phase1_rank} '
            f'mesh_index={phase1_entry["mesh_index"]} '
            f'loss={refined_loss_i:.6f}'
        )

        refined_results.append({
            'phase1_rank': int(phase1_rank),
            'phase1_entry': phase1_entry,
            'mesh_path': mesh_path_i,
            'mesh_unit': mesh_unit_i,
            'unit_scale': unit_scale_i,
            'center_raw_m': center_raw_m_i,
            'U_3x3': U_3x3_i,
            'best_seed': best_seed_i,
            'best_seed_loss': best_seed_loss_i,
            'best_seed_detail': best_seed_detail_i,
            'best_seed_pack': best_seed_pack_i,
            'phase1_top_packs': phase1_top_packs_i,
            'best_loss': float(refined_loss_i),
            'best_detail': refined_detail_i,
            'best_pack': refined_pack_i,
        })

        del renderer_i
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    refined_results.sort(key=lambda x: x['best_loss'])
    for refine_rank, item in enumerate(refined_results, start=1):
        item['refine_rank'] = int(refine_rank)

    final_result = refined_results[0]

    best_phase1 = final_result['phase1_entry']
    mesh_path = final_result['mesh_path']
    mesh_unit = final_result['mesh_unit']
    unit_scale = float(final_result['unit_scale'])
    center_raw_m = final_result['center_raw_m']
    U_3x3 = final_result['U_3x3']
    best_seed = final_result['best_seed']
    best_seed_loss = float(final_result['best_seed_loss'])
    best_seed_detail = final_result['best_seed_detail']
    best_seed_pack = final_result['best_seed_pack']
    phase1_top_packs = final_result['phase1_top_packs']
    best_loss = float(final_result['best_loss'])
    best_detail = final_result['best_detail']
    best_pack = final_result['best_pack']

    print('[INFO] final selected mesh after top-k refine:', best_phase1['mesh_index'])

    # =========================================================
    # 3. 결과 저장
    # =========================================================
    cv2.imwrite(os.path.join(out_dir, 'query_crop.png'), query_color_search)
    cv2.imwrite(os.path.join(out_dir, 'query_mask.png'), (query_mask_search.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(out_dir, 'seed_overlay.png'), best_seed_pack['overlay'])
    cv2.imwrite(os.path.join(out_dir, 'seed_render.png'), best_seed_pack['rgb'])
    cv2.imwrite(os.path.join(out_dir, 'best_overlay.png'), best_pack['overlay'])
    cv2.imwrite(os.path.join(out_dir, 'best_render.png'), best_pack['rgb'])
    cv2.imwrite(os.path.join(out_dir, 'best_overlay_raw.png'), best_pack['overlay_raw'])

    alpha_u8 = np.clip(best_pack['alpha'] * 255.0, 0, 255).astype(np.uint8)
    best_rgb_raw_u8 = np.clip(best_pack['rgb_raw'], 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, 'best_render_raw.png'), best_rgb_raw_u8)
    cv2.imwrite(os.path.join(out_dir, 'best_mask_raw.png'), (best_pack['mask_raw'].astype(np.uint8) * 255))

    cv2.imwrite(os.path.join(out_dir, 'best_alpha.png'), alpha_u8)
    cv2.imwrite(os.path.join(out_dir, 'best_blend.png'), best_pack['blend'])
    
    cv2.imwrite(os.path.join(out_dir, 'best_mask.png'), (best_pack['mask'].astype(np.uint8) * 255))
    save_depth(os.path.join(out_dir, 'best_depth.npy'), best_pack['depth'])
    axis_len_vis_m = float(max(best_detail['sx_sy_sz']) * 2)

    best_axis_overlay = draw_pose_axes_bgr(
        image_bgr=best_pack['overlay'],
        intrinsics=Ksearch,
        A_3x3=np.asarray(best_detail['A_3x3'], dtype=np.float64),
        b_xyz=np.asarray(best_detail['b_xyz'], dtype=np.float64),
        center_raw_m=np.asarray(center_raw_m, dtype=np.float64),
        basis_3x3=np.eye(3, dtype=np.float64),
        axis_len_m=axis_len_vis_m,
    )
    best_axis_render = draw_pose_axes_bgr(
        image_bgr=best_pack['rgb'],
        intrinsics=Ksearch,
        A_3x3=np.asarray(best_detail['A_3x3'], dtype=np.float64),
        b_xyz=np.asarray(best_detail['b_xyz'], dtype=np.float64),
        center_raw_m=np.asarray(center_raw_m, dtype=np.float64),
        basis_3x3=np.eye(3, dtype=np.float64),
        axis_len_m=axis_len_vis_m,
    )

    cv2.imwrite(os.path.join(out_dir, 'best_axis_overlay.png'), best_axis_overlay)
    cv2.imwrite(os.path.join(out_dir, 'best_axis_render.png'), best_axis_render)
    crop_w = (x1 - x0) + 1
    crop_h = (y1 - y0) + 1
    ys = slice(y0, y1 + 1)
    xs = slice(x0, x1 + 1)

    best_axis_full = color_full_bgr.copy()
    best_axis_resized = cv2.resize(best_axis_overlay, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    best_axis_full[ys, xs] = best_axis_resized
    cv2.imwrite(os.path.join(out_dir, 'best_axis_full.png'), best_axis_full)

    best_overlay_full = color_full_bgr.copy()
    best_blend_full = color_full_bgr.copy()

    best_overlay_resized = cv2.resize(best_pack['overlay'], (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    best_blend_resized = cv2.resize(best_pack['blend'], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    best_overlay_full[ys, xs] = best_overlay_resized
    best_blend_full[ys, xs] = best_blend_resized

    cv2.imwrite(os.path.join(out_dir, 'best_overlay_full.png'), best_overlay_full)
    cv2.imwrite(os.path.join(out_dir, 'best_blend_full.png'), best_blend_full)

    best_blend_raw_full = color_full_bgr.copy()

    best_rgb_raw_resized = cv2.resize(best_pack['rgb_raw'], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    best_alpha_raw_resized = cv2.resize(
        best_pack['alpha_raw'].astype(np.float32),
        (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR,
)   
    best_alpha_raw_resized = np.clip(best_alpha_raw_resized, 0.0, 1.0)[..., None]

    roi = best_blend_raw_full[ys, xs].astype(np.float32)
    rgb_raw = best_rgb_raw_resized.astype(np.float32)

    blended_roi = roi * (1.0 - best_alpha_raw_resized) + rgb_raw * best_alpha_raw_resized
    best_blend_raw_full[ys, xs] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, 'best_blend_raw_full.png'), best_blend_raw_full)
    cv2.imwrite(os.path.join(out_dir, 'best_blend_raw_full.png'), best_blend_raw_full)


    for item in refined_results:
        rank = int(item['refine_rank'])
        cv2.imwrite(os.path.join(out_dir, f'refined_top{rank}_overlay.png'), item['best_pack']['overlay'])
        cv2.imwrite(os.path.join(out_dir, f'refined_top{rank}_render.png'), item['best_pack']['rgb'])

        refined_alpha_u8 = np.clip(item['best_pack']['alpha'] * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'refined_top{rank}_alpha.png'), refined_alpha_u8)

        cv2.imwrite(os.path.join(out_dir, f'refined_top{rank}_blend.png'), item['best_pack']['blend'])
        cv2.imwrite(
            os.path.join(out_dir, f'refined_top{rank}_mask.png'),
            (item['best_pack']['mask'].astype(np.uint8) * 255)
        )
        save_depth(os.path.join(out_dir, f'refined_top{rank}_depth.npy'), item['best_pack']['depth'])

    for item in phase1_top_packs:
        rank = int(item['candidate']['rank'])
        cv2.imwrite(os.path.join(out_dir, f'phase1_top{rank}_overlay.png'), item['pack']['overlay'])
        cv2.imwrite(os.path.join(out_dir, f'phase1_top{rank}_blend.png'), item['pack']['blend'])
        cv2.imwrite(os.path.join(out_dir, f'phase1_top{rank}_render.png'), item['pack']['rgb'])
        cv2.imwrite(
            os.path.join(out_dir, f'phase1_top{rank}_mask.png'),
            (item['pack']['mask'].astype(np.uint8) * 255)
            
        )
    step7_roots = [
        {
            'root_id': 0,
            'root_source': 'best_init',
            'root_rank': 1,
            'mesh_path': os.path.abspath(mesh_path),
            'mesh_unit': mesh_unit,
            'R_3x3': best_detail['R_3x3'],
            't_xyz_m': best_detail['t_xyz_m'],
            'sx_sy_sz': best_detail['sx_sy_sz'],
            'view_dir': best_seed.get('view_dir'),
            'roll_deg': best_seed.get('roll_deg'),
            'scale_kind': best_seed.get('scale_kind'),
            'loss_total': float(best_loss),
            'iou': float(best_detail.get('iou', 0.0)),
            'depth_score': float(best_detail.get('depth_score', 0.0)),
        }
    ]

    next_root_id = 1
    for cand in best_phase1['top_candidates']:
        if int(cand.get('rank', 0)) == 1:
            continue
        step7_roots.append({
            'root_id': next_root_id,
            'root_source': 'phase1_top_candidate',
            'root_rank': int(cand.get('rank', next_root_id + 1)),
            'mesh_path': os.path.abspath(mesh_path),
            'mesh_unit': mesh_unit,
            'R_3x3': cand['R_3x3'],
            't_xyz_m': list(map(float, t0)),
            'sx_sy_sz': cand['sx_sy_sz'],
            'view_dir': cand.get('view_dir'),
            'roll_deg': cand.get('roll_deg'),
            'scale_kind': cand.get('scale_kind'),
            'loss_total': float(cand.get('loss_total', 0.0)),
            'iou': 0.0,
            'depth_score': 0.0,
        })
        next_root_id += 1
        if len(step7_roots) >= 3:
            break

    init_json = {
        'capture_dir': os.path.abspath(args.capture_dir),
        'mesh_path': os.path.abspath(mesh_path),
        'mesh_unit': mesh_unit,
        'has_query_depth': bool(has_query_depth),
        'query_depth_valid_px': int(depth_valid_px),
        'query_depth_valid_ratio': float(depth_valid_ratio),
        'render_scale': float(render_scale),
        'intrinsics_search': {
            'fx': float(Ksearch.fx), 'fy': float(Ksearch.fy), 'cx': float(Ksearch.cx), 'cy': float(Ksearch.cy),
            'width': int(Ksearch.width), 'height': int(Ksearch.height),
        },
        'seed_best': {
            **best_seed,
            'loss_total': float(best_seed_loss),
        },
        'phase1_top_candidates': best_phase1['top_candidates'],
        'best_init': {
            
            'loss_total': float(best_loss),
            **best_detail,
        },
                'step7_roots': step7_roots,
                'refined_top_results': [
            {
                'refine_rank': int(item['refine_rank']),
                'phase1_rank': int(item['phase1_rank']),
                'mesh_path': os.path.abspath(item['mesh_path']),
                'mesh_unit': item['mesh_unit'],
                'best_loss': float(item['best_loss']),
                'best_seed_loss': float(item['best_seed_loss']),
                'phase1_mesh_index': int(item['phase1_entry']['mesh_index']),
                'phase1_best_seed_loss': float(item['phase1_entry']['best_seed_loss']),
                'phase1_mean_top3_seed_loss': float(item['phase1_entry'].get('mean_top3_seed_loss', item['phase1_entry']['best_seed_loss'])),
                'phase1_mesh_rank_score': float(item['phase1_entry'].get('mesh_rank_score', item['phase1_entry']['best_seed_loss'])),
                'best_init_summary': {
                    'iou': float(item['best_detail'].get('iou', 0.0)),
                    'bbox_iou': float(item['best_detail'].get('bbox_iou', 0.0)),
                    'area_ratio': float(item['best_detail'].get('area_ratio', 0.0)),
                    'center_dist_px': float(item['best_detail'].get('center_dist_px', 0.0)),
                    'depth_score': float(item['best_detail'].get('depth_score', 0.0)),
                    'depth_penalty': float(item['best_detail'].get('depth_penalty', 0.0)),
                    'loss_depth_term': float(item['best_detail'].get('loss_depth_term', 0.0)),
                    'extra_ratio': float(item['best_detail'].get('extra_ratio', 0.0)),
                },
                'saved': {
                    'overlay_png': os.path.abspath(os.path.join(out_dir, f'refined_top{int(item["refine_rank"])}_overlay.png')),
                    'render_png': os.path.abspath(os.path.join(out_dir, f'refined_top{int(item["refine_rank"])}_render.png')),
                    'blend_png': os.path.abspath(os.path.join(out_dir, f'refined_top{int(item["refine_rank"])}_blend.png')),
                    'mask_png': os.path.abspath(os.path.join(out_dir, f'refined_top{int(item["refine_rank"])}_mask.png')),
                    'depth_npy': os.path.abspath(os.path.join(out_dir, f'refined_top{int(item["refine_rank"])}_depth.npy')),
                },
            }
            for item in refined_results
        ],

        'saved': {
            'best_render_png': os.path.abspath(os.path.join(out_dir, 'best_render.png')),
            'best_alpha_png': os.path.abspath(os.path.join(out_dir, 'best_alpha.png')),
            'best_blend_png': os.path.abspath(os.path.join(out_dir, 'best_blend.png')),
            'best_mask_png': os.path.abspath(os.path.join(out_dir, 'best_mask.png')),
            'best_overlay_png': os.path.abspath(os.path.join(out_dir, 'best_overlay.png')),
            'best_depth_npy': os.path.abspath(os.path.join(out_dir, 'best_depth.npy')),
        },
    }
    save_json(os.path.join(out_dir, 'init.json'), init_json)

    print('[OK] gaussian coarse init saved (PyTorch Optimized)')
    print('  out_dir :', out_dir)
    print('  best    :', {
        'loss_total': best_loss,
        'iou': best_detail.get('iou', 0),
        'area_ratio': best_detail.get('area_ratio', 0),
        'center_dist_px': best_detail.get('center_dist_px', 0),
        'sx_sy_sz': best_detail.get('sx_sy_sz', []),
        't_xyz_m': best_detail.get('t_xyz_m', []),
    })

if __name__ == '__main__':
    main()