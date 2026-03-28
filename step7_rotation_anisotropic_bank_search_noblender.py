#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
import itertools
from typing import List

import cv2
import numpy as np
import torch

from noblender_common import (
    compose_affine_from_pose,
    load_color_bgr,
    load_intrinsics_json_from_dict,
    load_json,
    load_mask,
    mask_hw,
    overlay_boundaries,
    parse_float_list,
    resize_color_bgr,
    resize_mask,
    save_depth,
    save_json,
    scale_intrinsics,
    keep_best_component,
    iou,
    Rx,
    Ry,
    Rz,
)
from render_backend_nvdiffrast import NvdiffrastRenderer
def bbox_hw_from_mask_batch(mask: torch.Tensor):
    B, H, W = mask.shape
    rows = mask.any(dim=2)
    cols = mask.any(dim=1)

    y = torch.arange(H, device=mask.device, dtype=torch.float32).view(1, H)
    x = torch.arange(W, device=mask.device, dtype=torch.float32).view(1, W)

    y0 = torch.where(rows, y, float(H)).amin(dim=1)
    y1 = torch.where(rows, y, -1.0).amax(dim=1)
    x0 = torch.where(cols, x, float(W)).amin(dim=1)
    x1 = torch.where(cols, x, -1.0).amax(dim=1)

    h = (y1 - y0 + 1).clamp(min=1.0)
    w = (x1 - x0 + 1).clamp(min=1.0)

    empty = ~mask.any(dim=(1, 2))
    h[empty] = 1.0
    w[empty] = 1.0
    return w, h


def score_mask_on_gpu(query_mask: torch.Tensor, qw: torch.Tensor, qh: torch.Tensor, alpha_batch: torch.Tensor, alpha_thresh: float):
    B = alpha_batch.shape[0]
    q = query_mask.unsqueeze(0).expand(B, -1, -1)
    m = alpha_batch > float(alpha_thresh)

    inter = (m & q).sum(dim=(1, 2)).float()
    union = (m | q).sum(dim=(1, 2)).float().clamp_min(1.0)
    iou_raw = inter / union

    qarea = q[0].sum().float().clamp_min(1.0)
    marea = m.sum(dim=(1, 2)).float()
    area_ratio = marea / qarea

    rw, rh = bbox_hw_from_mask_batch(m)

    aspect_q = qh / qw.clamp_min(1e-8)
    aspect_r = rh / rw.clamp_min(1e-8)

    aspect_penalty = (torch.log(aspect_r.clamp_min(1e-8)) - torch.log(aspect_q.clamp_min(1e-8))).abs()
    area_penalty = torch.log(area_ratio.clamp_min(1e-8)).abs()

    score = iou_raw - 0.20 * area_penalty - 0.45 * aspect_penalty

    empty = (marea < 1.0)
    score = torch.where(empty, torch.full_like(score, -9999.0), score)

    return {
        "score": score,
        "iou_raw": iou_raw,
        "area_ratio": area_ratio,
        "aspect_q": aspect_q.expand(B),
        "aspect_r": aspect_r
    }

#
def score_mask_on_cpu_exact(query_mask: np.ndarray, alpha_raw: np.ndarray, alpha_thresh: float):
    mask_raw = (alpha_raw > alpha_thresh).astype(bool)
    mask_f = keep_best_component(mask_raw, query_mask)

    qarea = float(query_mask.sum())
    iou_raw_val = float(iou(query_mask, mask_raw))
    iou_f_val = float(iou(query_mask, mask_f))
    
    area_ratio_raw = float(mask_raw.sum()) / max(qarea, 1e-8)
    area_ratio_f = float(mask_f.sum()) / max(qarea, 1e-8)
    
    qw, qh = mask_hw(query_mask)
    rw, rh = mask_hw(mask_f)
    
    aspect_q = float(qh) / max(float(qw), 1e-8)
    aspect_r = float(rh) / max(float(rw), 1e-8)
    
    aspect_penalty = abs(math.log(max(aspect_r, 1e-8) / max(aspect_q, 1e-8)))
    area_penalty = abs(math.log(max(area_ratio_f, 1e-8)))
    score = float(iou_f_val - 0.20 * area_penalty - 0.45 * aspect_penalty)

    metrics = {
        'iou_raw': iou_raw_val,
        'iou_filtered': iou_f_val,
        'render_area_ratio_raw_to_query': area_ratio_raw,
        'render_area_ratio_filtered_to_query': area_ratio_f,
        'aspect_query': aspect_q,
        'aspect_render': aspect_r,
        'score': score,
    }
    return metrics



def legacy_best_init_to_root(init_info: dict) -> dict:
    best = init_info['best_init']
    seed = init_info.get('seed_best', {})
    return {
        'root_id': 0,
        'root_source': 'best_init_legacy',
        'root_rank': 1,
        'mesh_path': os.path.abspath(init_info['mesh_path']),
        'mesh_unit': init_info.get('mesh_unit', 'm'),
        'R_3x3': best['R_3x3'],
        't_xyz_m': best['t_xyz_m'],
        'sx_sy_sz': best['sx_sy_sz'],
        'view_dir': seed.get('view_dir'),
        'roll_deg': seed.get('roll_deg'),
        'scale_kind': seed.get('scale_kind'),
        'loss_total': float(best.get('loss_total', 0.0)),
        'iou': float(best.get('iou', 0.0)),
        'depth_score': float(best.get('depth_score', 0.0)),
    }
def rx_batch_torch(rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(rad)
    s = torch.sin(rad)
    z = torch.zeros_like(rad)
    o = torch.ones_like(rad)
    return torch.stack([
        o, z, z,
        z, c, -s,
        z, s, c,
    ], dim=-1).reshape(-1, 3, 3)

def ry_batch_torch(rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(rad)
    s = torch.sin(rad)
    z = torch.zeros_like(rad)
    o = torch.ones_like(rad)
    return torch.stack([
         c, z, s,
         z, o, z,
        -s, z, c,
    ], dim=-1).reshape(-1, 3, 3)

def rz_batch_torch(rad: torch.Tensor) -> torch.Tensor:
    c = torch.cos(rad)
    s = torch.sin(rad)
    z = torch.zeros_like(rad)
    o = torch.ones_like(rad)
    return torch.stack([
        c, -s, z,
        s,  c, z,
        z,  z, o,
    ], dim=-1).reshape(-1, 3, 3)
def compose_affine_from_pose_batch_torch(
    R_batch: torch.Tensor,
    t_batch: torch.Tensor,
    s_batch: torch.Tensor,
    U: torch.Tensor,
    center_raw_m: torch.Tensor,
    unit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    S_batch = (U.unsqueeze(0) * s_batch[:, None, :]) @ U.T
    RS_batch = torch.bmm(R_batch, S_batch)
    A_batch = RS_batch * float(unit_scale)

    center = center_raw_m.view(1, 3, 1).expand(R_batch.shape[0], -1, -1)
    b_batch = t_batch - torch.bmm(RS_batch, center).squeeze(-1)
    return A_batch, b_batch
def build_cartesian_bank_torch(
    R0: np.ndarray,
    t0: np.ndarray,
    s0: np.ndarray,
    U: np.ndarray,
    center_raw_m: np.ndarray,
    unit_scale: float,
    local_rot_deg_list,
    local_t_mm_list,
    local_gamma_list,
    tweak_list,
    base_view_dir,
    base_roll_deg,
    base_scale_kind,
    start_idx: int,
    device: str = 'cuda',
    rot_chunk_size: int = 128,
) -> dict:
    R0_t = torch.tensor(R0, dtype=torch.float32, device=device).unsqueeze(0)
    t0_t = torch.tensor(t0, dtype=torch.float32, device=device).unsqueeze(0)
    s0_t = torch.tensor(s0, dtype=torch.float32, device=device).unsqueeze(0)
    U_t = torch.tensor(U, dtype=torch.float32, device=device)
    center_t = torch.tensor(center_raw_m, dtype=torch.float32, device=device)

    rot_vals = torch.tensor(local_rot_deg_list, dtype=torch.float32, device=device) * (math.pi / 180.0)
    trans_vals = torch.tensor([x * 1e-3 for x in local_t_mm_list], dtype=torch.float32, device=device)
    gamma_vals = torch.tensor(local_gamma_list, dtype=torch.float32, device=device)
    tweak_t = torch.tensor(np.stack(tweak_list), dtype=torch.float32, device=device)

    rot_grid = torch.cartesian_prod(rot_vals, rot_vals, rot_vals)
    trans_grid = torch.cartesian_prod(trans_vals, trans_vals, trans_vals)
    scale_grid = (gamma_vals[:, None, None] * tweak_t[None, :, :]).reshape(-1, 3)

    trans_cpu = trans_grid.detach().cpu().tolist()
    scale_cpu = scale_grid.detach().cpu().tolist()

    meta = []
    R_parts = []
    t_parts = []
    s_parts = []
    A_parts = []
    b_parts = []

    idx = start_idx

    for rot_chunk in rot_grid.split(rot_chunk_size):
        rot_chunk_cpu_deg = torch.rad2deg(rot_chunk).detach().cpu().tolist()

        Rx_b = rx_batch_torch(rot_chunk[:, 0])
        Ry_b = ry_batch_torch(rot_chunk[:, 1])
        Rz_b = rz_batch_torch(rot_chunk[:, 2])
        R_delta = torch.bmm(torch.bmm(Rz_b, Ry_b), Rx_b)
        R_rot = torch.bmm(R_delta, R0_t.expand(R_delta.shape[0], -1, -1))

        Nr = R_rot.shape[0]
        Nt = trans_grid.shape[0]
        Ns = scale_grid.shape[0]

        R_expand = R_rot[:, None, None, :, :].expand(Nr, Nt, Ns, 3, 3)
        t_expand = (t0_t.view(1, 1, 1, 3) + trans_grid.view(1, Nt, 1, 3)).expand(Nr, Nt, Ns, 3)
        s_expand = (s0_t.view(1, 1, 1, 3) * scale_grid.view(1, 1, Ns, 3)).expand(Nr, Nt, Ns, 3)

        R_flat = R_expand.reshape(-1, 3, 3)
        t_flat = t_expand.reshape(-1, 3)
        s_flat = s_expand.reshape(-1, 3)

        A_flat, b_flat = compose_affine_from_pose_batch_torch(
            R_batch=R_flat,
            t_batch=t_flat,
            s_batch=s_flat,
            U=U_t,
            center_raw_m=center_t,
            unit_scale=unit_scale,
        )

        R_parts.append(R_flat)
        t_parts.append(t_flat)
        s_parts.append(s_flat)
        A_parts.append(A_flat)
        b_parts.append(b_flat)

        for dr_deg in rot_chunk_cpu_deg:
            for dt_xyz in trans_cpu:
                for scale_xyz in scale_cpu:
                    meta.append({
                        'id': idx,
                        'stem': f'cand_{idx:04d}',
                        'search_mode': 'local',
                        'source': 'cartesian_product',
                        'view_dir': base_view_dir,
                        'roll_deg': base_roll_deg,
                        'scale_kind': base_scale_kind,
                        'delta_deg_xyz': [float(x) for x in dr_deg],
                        'delta_t_xyz': [float(x) for x in dt_xyz],
                        'gamma_xyz': [float(x) for x in scale_xyz],
                    })
                    idx += 1

    return {
        'meta': meta,
        'R_bank': torch.cat(R_parts, dim=0),
        't_bank': torch.cat(t_parts, dim=0),
        's_bank': torch.cat(s_parts, dim=0),
        'A_bank': torch.cat(A_parts, dim=0),
        'b_bank': torch.cat(b_parts, dim=0),
    }


def build_local_candidate_bank(
    init_info: dict,
    U,
    center_raw_m,
    unit_scale,
    local_rot_deg_list,
    local_t_mm_list,
    local_gamma_list,
    tweak_list,
    device='cuda',
) -> dict:
    best = init_info['best_init']
    seed = init_info.get('seed_best', {})

    base_view_dir = seed.get('view_dir', None)
    base_roll_deg = seed.get('roll_deg', None)
    base_scale_kind = seed.get('scale_kind', 'local_refine')

    R0 = np.asarray(best['R_3x3'], dtype=np.float64)
    t0 = np.asarray(best['t_xyz_m'], dtype=np.float64)
    s0 = np.asarray(best['sx_sy_sz'], dtype=np.float64)

    seed_meta = []
    seed_R = []
    seed_t = []
    seed_s = []

    idx = 0
    seen = set()

    seed_meta.append({
        'id': idx,
        'stem': f'cand_{idx:04d}',
        'search_mode': 'local',
        'source': 'base',
        'view_dir': base_view_dir,
        'roll_deg': base_roll_deg,
        'scale_kind': base_scale_kind,
        'delta_deg_xyz': [0.0, 0.0, 0.0],
        'delta_t_xyz': [0.0, 0.0, 0.0],
        'gamma_xyz': [1.0, 1.0, 1.0],
    })
    seed_R.append(R0)
    seed_t.append(t0)
    seed_s.append(s0)
    idx += 1

    seen.add(tuple(np.round(np.concatenate([R0.reshape(-1), t0.reshape(-1), s0.reshape(-1)]), 6).tolist()))

    flip_roots = [
        ('flip180_local_x', R0 @ Rx(math.pi)),
        ('flip180_local_y', R0 @ Ry(math.pi)),
    ]
    for branch_type, R_flip in flip_roots:
        key = tuple(np.round(np.concatenate([R_flip.reshape(-1), t0.reshape(-1), s0.reshape(-1)]), 6).tolist())
        if key in seen:
            continue
        seen.add(key)

        seed_meta.append({
            'id': idx,
            'stem': f'cand_{idx:04d}',
            'search_mode': 'local',
            'source': 'flip_seed',
            'view_dir': base_view_dir,
            'roll_deg': base_roll_deg,
            'scale_kind': base_scale_kind,
            'delta_deg_xyz': [0.0, 0.0, 0.0],
            'delta_t_xyz': [0.0, 0.0, 0.0],
            'gamma_xyz': [1.0, 1.0, 1.0],
            'branch_type': branch_type,
            'flip_group_id': 0,
        })
        seed_R.append(R_flip)
        seed_t.append(t0)
        seed_s.append(s0)
        idx += 1

    R_seed_t = torch.tensor(np.stack(seed_R), dtype=torch.float32, device=device)
    t_seed_t = torch.tensor(np.stack(seed_t), dtype=torch.float32, device=device)
    s_seed_t = torch.tensor(np.stack(seed_s), dtype=torch.float32, device=device)
    U_t = torch.tensor(U, dtype=torch.float32, device=device)
    center_t = torch.tensor(center_raw_m, dtype=torch.float32, device=device)

    A_seed_t, b_seed_t = compose_affine_from_pose_batch_torch(
        R_batch=R_seed_t,
        t_batch=t_seed_t,
        s_batch=s_seed_t,
        U=U_t,
        center_raw_m=center_t,
        unit_scale=unit_scale,
    )

    def _tag_bank(bank: dict, source: str, branch_type: str, flip_group_id: int = 0) -> dict:
        tagged_meta = []
        for rec in bank['meta']:
            rec2 = dict(rec)
            rec2['source'] = source
            rec2['branch_type'] = branch_type
            rec2['flip_group_id'] = flip_group_id
            tagged_meta.append(rec2)
        return {
            'meta': tagged_meta,
            'R_bank': bank['R_bank'],
            't_bank': bank['t_bank'],
            's_bank': bank['s_bank'],
            'A_bank': bank['A_bank'],
            'b_bank': bank['b_bank'],
        }

    base_bank = _tag_bank(
        build_cartesian_bank_torch(
            R0=R0,
            t0=t0,
            s0=s0,
            U=U,
            center_raw_m=center_raw_m,
            unit_scale=unit_scale,
            local_rot_deg_list=local_rot_deg_list,
            local_t_mm_list=local_t_mm_list,
            local_gamma_list=local_gamma_list,
            tweak_list=tweak_list,
            base_view_dir=base_view_dir,
            base_roll_deg=base_roll_deg,
            base_scale_kind=base_scale_kind,
            start_idx=idx,
            device=device,
            rot_chunk_size=128,
        ),
        source='cartesian_product',
        branch_type='base',
        flip_group_id=0,
    )
    idx += len(base_bank['meta'])

    # flip bank는 계산량 폭증을 막기 위해 축소 grid로 시작
    flip_rot_deg_list = [x for x in local_rot_deg_list if abs(float(x)) <= 2.0]
    if not flip_rot_deg_list:
        flip_rot_deg_list = [0.0]

    flip_t_mm_list = [x for x in local_t_mm_list if abs(float(x)) <= 5.0]
    if 0.0 not in flip_t_mm_list:
        flip_t_mm_list = [0.0] + flip_t_mm_list
    if not flip_t_mm_list:
        flip_t_mm_list = [0.0]

    flip_gamma_list = [x for x in local_gamma_list if abs(float(x) - 1.0) <= 0.03]
    if not flip_gamma_list:
        flip_gamma_list = [1.0]

    flip_banks = []
    for branch_type, R_flip in flip_roots:
        flip_bank = build_cartesian_bank_torch(
            R0=R_flip,
            t0=t0,
            s0=s0,
            U=U,
            center_raw_m=center_raw_m,
            unit_scale=unit_scale,
            local_rot_deg_list=flip_rot_deg_list,
            local_t_mm_list=flip_t_mm_list,
            local_gamma_list=flip_gamma_list,
            tweak_list=tweak_list,
            base_view_dir=base_view_dir,
            base_roll_deg=base_roll_deg,
            base_scale_kind=base_scale_kind,
            start_idx=idx,
            device=device,
            rot_chunk_size=64,
        )
        flip_bank = _tag_bank(
            flip_bank,
            source='flip_cartesian',
            branch_type=branch_type,
            flip_group_id=0,
        )
        flip_banks.append(flip_bank)
        idx += len(flip_bank['meta'])

    banks = [base_bank] + flip_banks

    return {
        'meta': seed_meta + [m for bank in banks for m in bank['meta']],
        'R_bank': torch.cat([R_seed_t] + [bank['R_bank'] for bank in banks], dim=0),
        't_bank': torch.cat([t_seed_t] + [bank['t_bank'] for bank in banks], dim=0),
        's_bank': torch.cat([s_seed_t] + [bank['s_bank'] for bank in banks], dim=0),
        'A_bank': torch.cat([A_seed_t] + [bank['A_bank'] for bank in banks], dim=0),
        'b_bank': torch.cat([b_seed_t] + [bank['b_bank'] for bank in banks], dim=0),
    }






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dir', required=True)
    ap.add_argument('--alpha_thresh', type=float, default=0.03)
    ap.add_argument('--render_scale', type=float, default=1.0)
    
    ap.add_argument('--keep_top_vis', type=int, default=40)
    ap.add_argument('--expected_step8_topk', type=int, default=20)
    ap.add_argument('--exact_rerank_multiplier', type=int, default=4)
    
    ap.add_argument('--init_json', default=None)
    ap.add_argument('--local_rot_deg_list', default='-1,-2,-3,-4,0,1,2,3,4')
    ap.add_argument('--local_t_mm_list', default='-10,0,10')
    ap.add_argument('--local_gamma_list', default='0.97,1.0,1.03')
    
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if args.batch_size < 1:
        raise ValueError("[ERROR] batch_size must be >= 1")
    if args.exact_rerank_multiplier < 1:
        raise ValueError("[ERROR] exact_rerank_multiplier must be >= 1")
    if args.keep_top_vis < args.expected_step8_topk:
        raise ValueError(f"[ERROR] keep_top_vis({args.keep_top_vis}) >= expected_step8_topk({args.expected_step8_topk})")
    
    render_scale = float(args.render_scale)
    if render_scale <= 0.0 or render_scale > 1.0:
        raise ValueError("[ERROR] render_scale must be in (0, 1]")

    device = str(args.device).strip().lower()
    if not device.startswith("cuda"):
        raise ValueError("[ERROR] NvdiffrastRenderer requires a cuda device, e.g. 'cuda' or 'cuda:0'")
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] CUDA is not available")


    calib_json_path = os.path.join(args.capture_dir, 'calib_prep', 'calib_input.json')
    scale_json_path = os.path.join(args.capture_dir, 'anisotropic_scale_hypothesis', 'summary.json')
    if args.init_json is None:
        args.init_json = os.path.join(args.capture_dir, 'gaussian_coarse_init', 'init.json')

    check_paths = [calib_json_path, scale_json_path, args.init_json]

    for p in check_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] File not found: {p}")


    out_dir = os.path.join(args.capture_dir, 'rotation_scale_bank')
    os.makedirs(out_dir, exist_ok=True)

    calib = load_json(calib_json_path)
    scale_init = load_json(scale_json_path)
    init_info = load_json(args.init_json)

    query_crop_path = calib['saved']['query_crop_png']
    query_mask_path = calib['saved']['query_mask_png']

    color_c = load_color_bgr(query_crop_path)
    query_mask = load_mask(query_mask_path)
    Kc = load_intrinsics_json_from_dict(calib['intrinsics_crop'])

    mesh_path = os.path.abspath(init_info['mesh_path'])
    mesh_unit = str(init_info.get('mesh_unit', 'm'))
    roots = init_info.get('step7_roots')
    if not roots:
        roots = [legacy_best_init_to_root(init_info)]

    roots = [
        r for r in roots
        if os.path.abspath(r.get('mesh_path', mesh_path)) == mesh_path
    ]
    roots = roots[:3]
    mesh_entries = scale_init.get('meshes', [])
    if not roots:
        raise RuntimeError(f"[ERROR] no compatible roots found for mesh_path={mesh_path}")

    mesh_entry = next(
        (m for m in mesh_entries if os.path.abspath(m['mesh_path']) == mesh_path),
        None
    )
    if mesh_entry is None:
        raise RuntimeError(f"[ERROR] selected mesh_path not found in step6 summary: {mesh_path}")

    center_raw_m = np.asarray(
        mesh_entry.get('mesh_bbox_center_m', mesh_entry['mesh_pca_stats']['center_xyz']),
        dtype=np.float64
    )
    U = np.asarray(mesh_entry['mesh_pca_stats']['pca_basis_3x3'], dtype=np.float64)
    unit_scale = float(mesh_entry.get('mesh_unit_scale_to_meter', 1.0))





    if render_scale != 1.0:
        Hs = max(1, int(round(Kc.height * render_scale)))
        Ws = max(1, int(round(Kc.width * render_scale)))
        color_search = resize_color_bgr(color_c, (Hs, Ws))
        query_mask_search = resize_mask(query_mask, (Hs, Ws))
        Ksearch = scale_intrinsics(Kc, render_scale)
    else:
        color_search = color_c
        query_mask_search = query_mask
        Ksearch = Kc

    cv2.imwrite(os.path.join(out_dir, 'query_crop.png'), color_search)
    cv2.imwrite(os.path.join(out_dir, 'query_mask.png'), (query_mask_search.astype(np.uint8) * 255))

    renderer = NvdiffrastRenderer(mesh_path, device=device)

    tweak_list = [
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        np.array([1.03, 0.97, 1.0], dtype=np.float64),
        np.array([0.97, 1.03, 1.0], dtype=np.float64),
    ]

    num_scale_combinations = 0
    local_rot_deg = parse_float_list(args.local_rot_deg_list)
    local_t_mm = parse_float_list(args.local_t_mm_list)
    local_gamma = parse_float_list(args.local_gamma_list)

    meta_all = []
    A_parts = []
    b_parts = []
    R_parts = []
    t_parts = []
    s_parts = []

    seen_global = set()
    next_id = 0
    next_tensor_idx = 0

    for root in roots:
        root_init_info = {
            'best_init': {
                'R_3x3': root['R_3x3'],
                't_xyz_m': root['t_xyz_m'],
                'sx_sy_sz': root['sx_sy_sz'],
            },
            'seed_best': {
                'view_dir': root.get('view_dir'),
                'roll_deg': root.get('roll_deg'),
                'scale_kind': root.get('scale_kind'),
            }
        }

        bank = build_local_candidate_bank(
            init_info=root_init_info,
            U=U,
            center_raw_m=center_raw_m,
            unit_scale=unit_scale,
            local_rot_deg_list=local_rot_deg,
            local_t_mm_list=local_t_mm,
            local_gamma_list=local_gamma,
            tweak_list=tweak_list,
            device=device,
        )

        pose_key = torch.cat([
            bank['R_bank'].reshape(bank['R_bank'].shape[0], -1),
            bank['t_bank'],
            bank['s_bank'],
        ], dim=1)
        pose_key_cpu = torch.round(pose_key * 1e6).to(torch.int64).cpu().tolist()

        keep_local = []
        kept_meta = []

        for local_i, key in enumerate(pose_key_cpu):
            key_t = tuple(key)
            if key_t in seen_global:
                continue
            seen_global.add(key_t)

            rec = dict(bank['meta'][local_i])
            rec['id'] = next_id
            rec['stem'] = f'cand_{next_id:04d}'
            rec['root_id'] = int(root.get('root_id', 0))
            rec['root_source'] = root.get('root_source', 'unknown')
            rec['root_rank'] = int(root.get('root_rank', 1))
            rec['tensor_idx'] = next_tensor_idx + len(keep_local)
            kept_meta.append(rec)

            keep_local.append(local_i)
            next_id += 1

        if not keep_local:
            continue

        keep_idx = torch.tensor(keep_local, dtype=torch.long, device=device)

        A_sel = bank['A_bank'].index_select(0, keep_idx)
        b_sel = bank['b_bank'].index_select(0, keep_idx)
        R_sel = bank['R_bank'].index_select(0, keep_idx)
        t_sel = bank['t_bank'].index_select(0, keep_idx)
        s_sel = bank['s_bank'].index_select(0, keep_idx)

        meta_all.extend(kept_meta)
        A_parts.append(A_sel)
        b_parts.append(b_sel)
        R_parts.append(R_sel)
        t_parts.append(t_sel)
        s_parts.append(s_sel)

        next_tensor_idx += A_sel.shape[0]
    if not meta_all:
        raise RuntimeError("[ERROR] no candidates survived after dedup/filtering")

    A_bank = torch.cat(A_parts, dim=0)
    b_bank = torch.cat(b_parts, dim=0)
    R_bank = torch.cat(R_parts, dim=0)
    t_bank = torch.cat(t_parts, dim=0)
    s_bank = torch.cat(s_parts, dim=0)

    search_mode = 'local_from_gaussian_init'
    num_scale_combinations = len(local_gamma) * len(tweak_list)

    print(f'[INFO] search_mode            : {search_mode}')
    print(f'[INFO] total render candidates: {len(meta_all)}')
    if len(meta_all) > 10000:
        print("[WARN] Local candidate count is very large (>10,000). Optimization might take longer.")


    all_meta = []
    batch_size = args.batch_size


    print("[INFO] Phase 1: 100% GPU Bulk Scoring...")
    query_mask_t = torch.from_numpy(query_mask_search).to(device=device, dtype=torch.bool)
    qw, qh = bbox_hw_from_mask_batch(query_mask_t.unsqueeze(0))

    for st in range(0, len(meta_all), batch_size):
        ed = min(len(meta_all), st + batch_size)
        batch_meta = meta_all[st:ed]

        A_batch = A_bank[st:ed]
        b_batch = b_bank[st:ed]

        with torch.no_grad():
            res = renderer.render_batch(A_batch, b_batch, Ksearch)
            gpu_scores = score_mask_on_gpu(query_mask_t, qw, qh, res['alpha'], args.alpha_thresh)

        scores_cpu = gpu_scores['score'].cpu().numpy()
        iou_raw_cpu = gpu_scores['iou_raw'].cpu().numpy()
        area_ratio_cpu = gpu_scores['area_ratio'].cpu().numpy()
        aspect_q_cpu = gpu_scores['aspect_q'].cpu().numpy()
        aspect_r_cpu = gpu_scores['aspect_r'].cpu().numpy()

        for i, rec in enumerate(batch_meta):
            all_meta.append({
                **rec,
                "score_approx": float(scores_cpu[i]),
                "iou_raw": float(iou_raw_cpu[i]),
                "render_area_ratio_raw_to_query": float(area_ratio_cpu[i]),
                "aspect_query": float(aspect_q_cpu[i]),
                "aspect_render": float(aspect_r_cpu[i]),
            })

    all_meta.sort(key=lambda x: x['score_approx'], reverse=True)
    
    M = min(int(args.keep_top_vis * args.exact_rerank_multiplier), len(all_meta))
    print(f"\n[INFO] Phase 1.5: CPU Exact Re-ranking for top {M} candidates...")
    
    rerank_cands = all_meta[:M]
    
    # [수정] Phase 1.5 VRAM 스파이크 방지 (Batch Processing)
    for st in range(0, M, batch_size):
        ed = min(M, st + batch_size)
        chunk = rerank_cands[st:ed]
        idx_chunk = torch.tensor([c['tensor_idx'] for c in chunk], dtype=torch.long, device=device)
        A_chunk = A_bank.index_select(0, idx_chunk)
        b_chunk = b_bank.index_select(0, idx_chunk)

        
        with torch.no_grad():
            res_chunk = renderer.render_batch(A_chunk, b_chunk, Ksearch)
            alpha_chunk_cpu = res_chunk['alpha'].cpu().numpy()
            
        for i, cand in enumerate(chunk):
            exact_metrics = score_mask_on_cpu_exact(query_mask_search, alpha_chunk_cpu[i], args.alpha_thresh)
            cand.update(exact_metrics) # 정확한 'score', 'iou_filtered' 등 갱신

    for rec in all_meta:
        if 'score' not in rec:
            rec['score'] = rec['score_approx']
        if 'iou_filtered' not in rec:
            rec['iou_filtered'] = rec['iou_raw']
        if 'render_area_ratio_filtered_to_query' not in rec:
            rec['render_area_ratio_filtered_to_query'] = rec['render_area_ratio_raw_to_query']

    results_sorted = sorted(all_meta, key=lambda x: x['score'], reverse=True)

    # =========================================================
    # Phase 2: 최종 Top-K 디스크 저장 (VRAM 안전 Chunking)
    # =========================================================
    keep_top_vis = min(int(args.keep_top_vis), len(results_sorted))
    print(f"\n[INFO] Phase 2: Final rendering and saving for top {keep_top_vis} candidates...")
    
    top_k_cands = results_sorted[:keep_top_vis]
    saved_paths_dict = {}
    
    for st in range(0, keep_top_vis, batch_size):
        ed = min(keep_top_vis, st + batch_size)
        chunk = top_k_cands[st:ed]
        idx_chunk = torch.tensor([c['tensor_idx'] for c in chunk], dtype=torch.long, device=device)
        A_chunk = A_bank.index_select(0, idx_chunk)
        b_chunk = b_bank.index_select(0, idx_chunk)
        R_chunk = R_bank.index_select(0, idx_chunk).detach().cpu().tolist()
        t_chunk = t_bank.index_select(0, idx_chunk).detach().cpu().tolist()
        s_chunk = s_bank.index_select(0, idx_chunk).detach().cpu().tolist()
        A_chunk_cpu = A_chunk.detach().cpu().tolist()
        b_chunk_cpu = b_chunk.detach().cpu().tolist()


        with torch.no_grad():
            res_chunk = renderer.render_batch(A_chunk, b_chunk, Ksearch)
            alpha_chunk_raw = res_chunk['alpha'].cpu().numpy()
            depth_chunk_raw = res_chunk['depth'].cpu().numpy()
            rgb_chunk_raw = res_chunk['rgb'].cpu().numpy()
        
        for i, rec in enumerate(chunk):
            rec['R_3x3'] = R_chunk[i]
            rec['t_xyz_m'] = t_chunk[i]
            rec['sx_sy_sz'] = s_chunk[i]
            rec['A_3x3'] = A_chunk_cpu[i]
            rec['b_xyz'] = b_chunk_cpu[i]
            stem = rec['stem']
            
            alpha_raw = alpha_chunk_raw[i]
            mask_raw = (alpha_raw > args.alpha_thresh).astype(bool)
            mask_f = keep_best_component(mask_raw, query_mask_search)

            rgb = (np.clip(rgb_chunk_raw[i], 0.0, 1.0) * 255).astype(np.uint8)
            if rgb.shape[-1] == 3: rgb = rgb[..., ::-1].copy()
            rgb[~mask_f] = 0
            
            alpha = (np.clip(alpha_raw, 0.0, 1.0) * 255).astype(np.uint8)
            alpha[~mask_f] = 0
            
            depth = depth_chunk_raw[i].copy()
            depth[~mask_f] = 0.0
            
            overlay = overlay_boundaries(color_search, query_mask_search, mask_f)

            render_path = os.path.join(out_dir, stem + '_render.png')
            alpha_path = os.path.join(out_dir, stem + '_alpha.png')
            mask_path = os.path.join(out_dir, stem + '_mask.png')
            depth_path = os.path.join(out_dir, stem + '_depth.npy')
            overlay_path = os.path.join(out_dir, stem + '_overlay.png')

            cv2.imwrite(render_path, rgb)
            cv2.imwrite(alpha_path, alpha)
            cv2.imwrite(mask_path, (mask_f.astype(np.uint8) * 255))
            save_depth(depth_path, depth)
            cv2.imwrite(overlay_path, overlay)

            saved_paths_dict[stem] = {
                'render_png': os.path.abspath(render_path),
                'alpha_png': os.path.abspath(alpha_path),
                'mask_png': os.path.abspath(mask_path),
                'depth_npy': os.path.abspath(depth_path),
                'overlay_png': os.path.abspath(overlay_path),
            }

    # 완성된 경로 메타데이터 병합
    for rec in results_sorted:
        stem = rec['stem']
        if stem in saved_paths_dict:
            rec['paths'] = saved_paths_dict[stem]
    results_saved = results_sorted[:keep_top_vis]
    best = dict(results_saved[0])

    required_result_fields = [
    'id', 'R_3x3', 't_xyz_m', 'sx_sy_sz',
    'A_3x3', 'b_xyz',
    'score', 'iou_filtered']    
    required_path_fields = ['render_png', 'alpha_png', 'mask_png', 'depth_npy', 'overlay_png']

    for rec in results_sorted[:keep_top_vis]:
        for key in required_result_fields:
            if key not in rec:
                raise RuntimeError(f"[ERROR] step7 result missing required field: {key}")
        if 'paths' not in rec:
            raise RuntimeError('[ERROR] step7 top result missing paths metadata')
        for key in required_path_fields:
            if key not in rec['paths']:
                raise RuntimeError(f"[ERROR] step7 result paths missing required field: {key}")

    summary = {
        'capture_dir': os.path.abspath(args.capture_dir),
        'mesh_path': os.path.abspath(mesh_path),
        'mesh_unit': mesh_unit,
        'search': {
            'mode': search_mode,
            'alpha_thresh': float(args.alpha_thresh),
            'render_scale': float(args.render_scale),
            'num_scale_candidates': num_scale_combinations,
            'num_total_candidates': len(results_sorted),
            'init_json': os.path.abspath(args.init_json) if args.init_json else None,
            'num_saved_results': len(results_saved),
            'local_rot_deg_list': parse_float_list(args.local_rot_deg_list),
            'local_t_mm_list': parse_float_list(args.local_t_mm_list),
            'local_gamma_list': parse_float_list(args.local_gamma_list),
        },
        'query': {
            't0_xyz_m': calib['init_translation_t0_xyz_m'],
            'crop_xyxy': calib['crop_xyxy'],
        },
        'best_by_score': best,
        'results': results_saved, #
    }
    summary_path = os.path.join(out_dir, 'summary.json')
    save_json(summary_path, summary)

    print('\n[OK] 100% GPU-Scored rotation/scale bank saved')
    print('  out_dir :', out_dir)
    print('  summary :', summary_path)
    print('  best    :', {
        'id': best['id'],
        'score': best['score'],
        'sx_sy_sz': best['sx_sy_sz'],
        't_xyz_m': best['t_xyz_m'],
    })

if __name__ == '__main__':
    main()