#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from noblender_common import (
    Rx,
    Ry,
    Rz,
    build_uncert_target_map,
    compute_depth_agreement_score,
    iou,
    keep_best_component,
    load_color_rgb,
    load_depth,
    load_mask,
    mask_hw,
    overlay_boundaries,
    parse_float_list,
    relative_rvec,
    resize_mask,
    save_depth,
    save_json,
    load_json,
    load_intrinsics_json_from_dict,
    compose_affine_from_pose
)
# [최적화 1] CPU 렌더러 제거 및 GPU 렌더러(Nvdiffrast) 도입
from render_backend_nvdiffrast import NvdiffrastRenderer


class DinoV3Extractor:
    def __init__(self, model_name: str, device: str = 'cuda', dtype: str = 'bf16'):
        self.device = device
        if dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif dtype == 'fp16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device).eval()
        self.patch_size = int(self.model.config.patch_size)
        self.num_register_tokens = int(getattr(self.model.config, 'num_register_tokens', 0))

    def _prepare_image(self, rgb_uint8: np.ndarray) -> Image.Image:
        # 패치 사이즈 배수에 맞게 입력 보정 (Step 8 통일)
        h, w = rgb_uint8.shape[:2]
        new_h = max(self.patch_size, (h // self.patch_size) * self.patch_size)
        new_w = max(self.patch_size, (w // self.patch_size) * self.patch_size)
        if new_h != h or new_w != w:
            rgb_uint8 = cv2.resize(rgb_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(rgb_uint8.astype('uint8'))

    @torch.inference_mode()
    def extract(self, rgb_uint8: np.ndarray):
        img = self._prepare_image(rgb_uint8)
        # Processor 자체 Resize/Crop 강제 비활성화
        inputs = self.processor(images=img, return_tensors='pt', do_resize=False, do_center_crop=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        x = outputs.last_hidden_state
        global_feat = x[:, 0, :].clone()
        patch_tokens = x[:, 1 + self.num_register_tokens :, :]
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        global_feat = F.normalize(global_feat, dim=-1)
        H_in, W_in = inputs['pixel_values'].shape[-2:]
        ph = H_in // self.patch_size
        pw = W_in // self.patch_size
        feat_map = patch_tokens.view(1, ph, pw, -1)[0].permute(2, 0, 1).contiguous()
        return {
            'feat_map': feat_map,
            'global_feat': global_feat[0],
            'ph': int(ph),
            'pw': int(pw),
            'input_hw': [int(H_in), int(W_in)],
        }

    @torch.inference_mode()
    def extract_batch(self, rgb_uint8_list: List[np.ndarray]):
        if len(rgb_uint8_list) == 0:
            return {
                'feat_map': torch.empty(0),
                'global_feat': torch.empty(0),
                'ph': 0,
                'pw': 0,
                'input_hw': [0, 0],
            }
        imgs = [self._prepare_image(img) for img in rgb_uint8_list]
        sizes = [img.size for img in imgs]
        if len(set(sizes)) != 1:
            raise ValueError(f'mixed image sizes in DINO batch: {set(sizes)}')

        inputs = self.processor(images=imgs, return_tensors='pt', do_resize=False, do_center_crop=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        x = outputs.last_hidden_state
        global_feat = x[:, 0, :].clone()
        patch_tokens = x[:, 1 + self.num_register_tokens :, :]
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        global_feat = F.normalize(global_feat, dim=-1)
        B = patch_tokens.shape[0]
        H_in, W_in = inputs['pixel_values'].shape[-2:]
        ph = H_in // self.patch_size
        pw = W_in // self.patch_size
        feat_map = patch_tokens.view(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()
        return {
            'feat_map': feat_map,
            'global_feat': global_feat,
            'ph': int(ph),
            'pw': int(pw),
            'input_hw': [int(H_in), int(W_in)],
        }


def masked_patch_cosine_score_from_feats(q_dict: dict, query_mask: np.ndarray, a_feat_map: torch.Tensor, ph: int, pw: int, anchor_mask: np.ndarray):
    q_feat = q_dict['feat_map'].unsqueeze(0)
    a_feat = a_feat_map.unsqueeze(0)
    q_mask_s = resize_mask(query_mask, (ph, pw))
    a_mask_s = resize_mask(anchor_mask, (ph, pw))
    valid = q_mask_s & a_mask_s
    sim_map = (F.normalize(q_feat, dim=1) * F.normalize(a_feat, dim=1)).sum(dim=1)[0]
    sim_map_cpu = sim_map.detach().cpu().float().numpy()
    if int(valid.sum()) == 0:
        return 0.0, sim_map_cpu
    score = float(sim_map.detach().cpu().float()[torch.from_numpy(valid)].mean().item())
    return score, sim_map_cpu


def score_render_record(query_rgb_bgr: np.ndarray, query_mask: np.ndarray, query_depth: np.ndarray, q_dino: dict, render_rgb: np.ndarray, render_mask: np.ndarray, render_depth: np.ndarray, batch_feat: torch.Tensor, batch_global: torch.Tensor, idx: int):
    mask = keep_best_component(render_mask, query_mask)
    rgb = render_rgb.copy()
    rgb[~mask] = 0
    depth = render_depth.copy()
    depth[~mask] = 0.0

    qarea = float(query_mask.sum())
    iou_f = float(iou(query_mask, mask))
    area_ratio = float(mask.sum()) / max(qarea, 1e-8)
    qw, qh = mask_hw(query_mask)
    rw, rh = mask_hw(mask)
    qaspect = float(qh) / max(float(qw), 1e-8)
    raspect = float(rh) / max(float(rw), 1e-8)
    area_penalty = abs(math.log(max(area_ratio, 1e-8)))
    aspect_penalty = abs(math.log(max(raspect, 1e-8) / max(qaspect, 1e-8)))

    feat_score, sim_map = masked_patch_cosine_score_from_feats(
        q_dict=q_dino,
        query_mask=query_mask,
        a_feat_map=batch_feat[idx],
        ph=int(q_dino['ph']),
        pw=int(q_dino['pw']),
        anchor_mask=mask,
    )
    
    # [최적화 4] q_global을 반복해서 CPU 변환하지 않고 캐싱된 것 재사용
    q_global = q_dino['global_feat_cpu']
    a_global = batch_global[idx].detach().cpu().float().numpy()
    global_score = float(np.dot(q_global, a_global))

    qbbox = np.array(np.where(query_mask)).T
    bbox_overlap = 0.0
    if qbbox.size > 0 and mask.any():
        qx0, qy0, qx1, qy1 = int(qbbox[:, 1].min()), int(qbbox[:, 0].min()), int(qbbox[:, 1].max()), int(qbbox[:, 0].max())
        m_pts = np.array(np.where(mask)).T
        mx0, my0, mx1, my1 = int(m_pts[:, 1].min()), int(m_pts[:, 0].min()), int(m_pts[:, 1].max()), int(m_pts[:, 0].max())
        ix0, iy0 = max(qx0, mx0), max(qy0, my0)
        ix1, iy1 = min(qx1, mx1), min(qy1, my1)
        iw = max(0, ix1 - ix0 + 1)
        ih = max(0, iy1 - iy0 + 1)
        inter = float(iw * ih)
        qa = float((qx1 - qx0 + 1) * (qy1 - qy0 + 1))
        ma = float((mx1 - mx0 + 1) * (my1 - my0 + 1))
        bbox_overlap = inter / max(qa + ma - inter, 1e-8)

    depth_score = compute_depth_agreement_score(query_depth, depth, query_mask, mask, sigma_depth_m=0.02)
    score = float(
        0.35 * iou_f +
        0.25 * feat_score +
        0.10 * global_score +
        0.15 * depth_score +
        0.10 * bbox_overlap -
        0.10 * area_penalty -
        0.10 * aspect_penalty
    )

    overlay = overlay_boundaries(query_rgb_bgr, query_mask, mask)
    uncert_t = build_uncert_target_map(query_mask, mask, sim_map)
    return {
        'score': float(score),
        'iou_filtered': float(iou_f),
        'feat_score_masked_cosine': float(feat_score),
        'global_score_cosine': float(global_score),
        'bbox_overlap': float(bbox_overlap),
        'depth_score': float(depth_score),
        'render_area_ratio_to_query': float(area_ratio),
        'aspect_query': float(qaspect),
        'aspect_render': float(raspect),
        'mask': mask,
        'rgb': rgb,
        'depth': depth,
        'overlay': overlay,
        'uncert_target': uncert_t,
    }


def make_state(R: np.ndarray, t: np.ndarray, s: np.ndarray):
    return {
        'R': np.asarray(R, dtype=np.float64).copy(),
        't': np.asarray(t, dtype=np.float64).reshape(3).copy(),
        's': np.asarray(s, dtype=np.float64).reshape(3).copy(),
    }


def perturb_states(base_state: dict, rot_deg: float, trans_m: float, scale_log: float):
    states = [
        {'kind': 'base', 'delta_deg_xyz': [0.0, 0.0, 0.0], 'delta_t_xyz': [0.0, 0.0, 0.0], 'delta_log_sxyz': [0.0, 0.0, 0.0], 'state': make_state(base_state['R'], base_state['t'], base_state['s'])}
    ]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            r = [0.0, 0.0, 0.0]
            r[axis] = sign * float(rot_deg)
            dR = Rz(math.radians(r[2])) @ Ry(math.radians(r[1])) @ Rx(math.radians(r[0]))
            states.append({
                'kind': f'rot_{axis}_{sign:+.0f}',
                'delta_deg_xyz': r,
                'delta_t_xyz': [0.0, 0.0, 0.0],
                'delta_log_sxyz': [0.0, 0.0, 0.0],
                'state': make_state(dR @ base_state['R'], base_state['t'], base_state['s']),
            })
    for axis in range(3):
        for sign in (-1.0, 1.0):
            dt = np.zeros(3, dtype=np.float64)
            dt[axis] = sign * float(trans_m)
            states.append({
                'kind': f'trans_{axis}_{sign:+.0f}',
                'delta_deg_xyz': [0.0, 0.0, 0.0],
                'delta_t_xyz': dt.tolist(),
                'delta_log_sxyz': [0.0, 0.0, 0.0],
                'state': make_state(base_state['R'], base_state['t'] + dt, base_state['s']),
            })
    for axis in range(3):
        for sign in (-1.0, 1.0):
            dlog = np.zeros(3, dtype=np.float64)
            dlog[axis] = sign * float(scale_log)
            states.append({
                'kind': f'scale_{axis}_{sign:+.0f}',
                'delta_deg_xyz': [0.0, 0.0, 0.0],
                'delta_t_xyz': [0.0, 0.0, 0.0],
                'delta_log_sxyz': dlog.tolist(),
                'state': make_state(base_state['R'], base_state['t'], base_state['s'] * np.exp(dlog)),
            })
    return states


def _state_cache_key(state: dict) -> Tuple[float, ...]:
    return tuple(np.round(np.concatenate([state['R'].reshape(-1), state['t'], state['s']]), 5).tolist())


def _bad_score_record(st: dict, A: np.ndarray, b: np.ndarray, query_rgb_bgr: np.ndarray, query_mask: np.ndarray):
    return {
        'R_3x3': st['R'].tolist(),
        't_xyz_m': st['t'].tolist(),
        'sx_sy_sz': st['s'].tolist(),
        'A_3x3': A.tolist(),
        'b_xyz': b.tolist(),
        'score': -9999.0,
        'iou_filtered': 0.0,
        'feat_score_masked_cosine': 0.0,
        'global_score_cosine': 0.0,
        'bbox_overlap': 0.0,
        'depth_score': 0.0,
        'render_area_ratio_to_query': 0.0,
        'aspect_query': 1.0,
        'aspect_render': 1.0,
    }


def evaluate_state_batch(
    renderer,
    K,
    U,
    center_raw_m,
    unit_scale,
    query_rgb_bgr,
    query_mask,
    query_depth,
    extractor: DinoV3Extractor,
    q_dino: dict,
    candidate_records: List[dict],
    state_cache: dict,
    alpha_thresh: float,
    return_dense: bool = True,
):
    out = [None] * len(candidate_records)

    A_list = []
    b_list = []
    to_render_indices = []

    # 1. 캐시 확인 및 배치 렌더링용 텐서 수집
    for i, rec in enumerate(candidate_records):
        st = rec['state']
        key = _state_cache_key(st)

        if key in state_cache:
            out[i] = {**rec, **state_cache[key]}
            continue

        A, b, _ = compose_affine_from_pose(st['R'], st['t'], st['s'], U, center_raw_m, unit_scale)
        A_list.append(A)
        b_list.append(b)
        to_render_indices.append(i)

    if not to_render_indices:
        return out

    # [최적화 2] 렌더러 단건 호출 방식을 통배치(Batch) 렌더링으로 변경
    A_tensor = torch.tensor(np.stack(A_list), dtype=torch.float32, device=renderer.device)
    b_tensor = torch.tensor(np.stack(b_list), dtype=torch.float32, device=renderer.device)

    with torch.no_grad():
        render_res = renderer.render_batch(A_tensor, b_tensor, K)
        batch_alpha = render_res['alpha'].cpu().numpy()
        batch_depth = render_res['depth'].cpu().numpy()
        batch_rgb = render_res['rgb'].cpu().numpy()

    valid_dino_rgbs = []
    valid_indices = []
    interim = []

    # 3. 빈 마스크 탈락 및 DINO용 RGB 배열 구성
    for idx, orig_i in enumerate(to_render_indices):
        st = candidate_records[orig_i]['state']
        key = _state_cache_key(st)
        A = A_list[idx]
        b = b_list[idx]

        # [최적화 5] 0.03 하드코딩 제거
        mask_raw = batch_alpha[idx] > alpha_thresh
        
        if mask_raw.sum() < 1:
            bad_score = _bad_score_record(st, A, b, query_rgb_bgr, query_mask)
            state_cache[key] = bad_score
            out[orig_i] = {**candidate_records[orig_i], **bad_score}
            continue

        # [최적화 3] Nvdiffrast 출력을 확실한 uint8 RGB 포맷으로 강제 변환
        rgb_uint8 = (np.clip(batch_rgb[idx], 0.0, 1.0) * 255).astype(np.uint8)

        valid_dino_rgbs.append(rgb_uint8)
        valid_indices.append(orig_i)
        interim.append((A, b, rgb_uint8, mask_raw, batch_depth[idx], key))

    # 4. 유효한 후보들만 모아서 DINO Batch 추론 및 스코어링
    if len(valid_dino_rgbs) > 0:
        dino_batch = extractor.extract_batch(valid_dino_rgbs)
        for batch_idx, orig_i in enumerate(valid_indices):
            A, b, rgb_uint8, mask_raw, depth_raw, key = interim[batch_idx]
            st = candidate_records[orig_i]['state']
            
            scored = score_render_record(
                query_rgb_bgr=query_rgb_bgr,
                query_mask=query_mask,
                query_depth=query_depth,
                q_dino=q_dino,
                render_rgb=rgb_uint8,
                render_mask=mask_raw,
                render_depth=depth_raw,
                batch_feat=dino_batch['feat_map'],
                batch_global=dino_batch['global_feat'],
                idx=batch_idx,
            )
            light_score = {
                'R_3x3': st['R'].tolist(),
                't_xyz_m': st['t'].tolist(),
                'sx_sy_sz': st['s'].tolist(),
                'A_3x3': A.tolist(),
                'b_xyz': b.tolist(),
                'score': scored['score'],
                'iou_filtered': scored['iou_filtered'],
                'feat_score_masked_cosine': scored['feat_score_masked_cosine'],
                'global_score_cosine': scored['global_score_cosine'],
                'bbox_overlap': scored['bbox_overlap'],
                'depth_score': scored['depth_score'],
                'render_area_ratio_to_query': scored['render_area_ratio_to_query'],
                'aspect_query': scored['aspect_query'],
                'aspect_render': scored['aspect_render'],
            }

            state_cache[key] = light_score
            if return_dense:
                out[orig_i] = {
                    **candidate_records[orig_i],
                    **light_score,
                    'uncert_target': scored['uncert_target'],
                }
            else:
                out[orig_i] = {**candidate_records[orig_i], **light_score}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dir', required=True)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--mesh_unit', choices=['m', 'mm'], default='m')
    ap.add_argument('--top_from_step7', type=int, default=8)
    ap.add_argument('--model_name', default='facebook/dinov3-vits16-pretrain-lvd1689m')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--rot_steps_deg', default='4,2,1')
    ap.add_argument('--trans_steps_mm', default='4,2,1')
    ap.add_argument('--scale_steps_log', default='0.03,0.015,0.0075')
    ap.add_argument('--passes_per_stage', type=int, default=2)
    # [최적화 5] 죽은 인자 제거 및 alpha_thresh 추가
    ap.add_argument('--alpha_thresh', type=float, default=0.03)
    args = ap.parse_args()

    summary_path = os.path.join(args.capture_dir, 'rotation_scale_bank', 'summary.json')
    calib_json_path = os.path.join(args.capture_dir, 'calib_prep', 'calib_input.json')
    query_crop_path = os.path.join(args.capture_dir, 'calib_prep', 'query_crop.png')
    query_mask_path = os.path.join(args.capture_dir, 'calib_prep', 'query_mask.png')
    query_depth_path = os.path.join(args.capture_dir, 'calib_prep', 'query_depth_crop.npy')
    scale_json_path = os.path.join(args.capture_dir, 'anisotropic_scale_hypothesis', 'summary.json')
    for p in [summary_path, calib_json_path, query_crop_path, query_mask_path, query_depth_path, scale_json_path, args.mesh]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    out_dir = os.path.join(args.capture_dir, 'teacher_refine_v4')
    os.makedirs(out_dir, exist_ok=True)
    step7 = load_json(summary_path)
    calib = load_json(calib_json_path)
    scale_init = load_json(scale_json_path)

    mesh_path = os.path.abspath(args.mesh)
    mesh_entries = scale_init.get('meshes', [])
    mesh_entry = next(
        (m for m in mesh_entries if os.path.abspath(m['mesh_path']) == mesh_path),
        None,
    )
    if mesh_entry is None:
        raise RuntimeError(f"[ERROR] selected mesh_path not found in step6 summary: {mesh_path}")

    query_rgb = load_color_rgb(query_crop_path)
    query_rgb_bgr = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)
    query_mask = load_mask(query_mask_path)
    query_depth = load_depth(query_depth_path)
    Kc = load_intrinsics_json_from_dict(calib['intrinsics_crop'])

    center_raw_m = np.asarray(
        mesh_entry.get('mesh_bbox_center_m', mesh_entry['mesh_pca_stats']['center_xyz']),
        dtype=np.float64,
    )
    U = np.asarray(mesh_entry['mesh_pca_stats']['pca_basis_3x3'], dtype=np.float64)
    unit_scale = float(mesh_entry.get('mesh_unit_scale_to_meter', 1.0))


    renderer = NvdiffrastRenderer(args.mesh, device=args.device)
    extractor = DinoV3Extractor(model_name=args.model_name, device=args.device, dtype=args.dtype)
    print('[DINOv3] extracting query feature...')
    q_dino = extractor.extract(query_rgb)
    
    # [최적화 4] DINO 추출 후 q_global CPU 변환을 미리 수행하여 재사용
    q_dino['global_feat_cpu'] = q_dino['global_feat'].detach().cpu().float().numpy()

    step7_top = step7['results'][: min(int(args.top_from_step7), len(step7['results']))]
    rot_steps_deg = parse_float_list(args.rot_steps_deg)
    trans_steps_m = [x * 1e-3 for x in parse_float_list(args.trans_steps_mm)]
    scale_steps_log = parse_float_list(args.scale_steps_log)
    nstage = min(len(rot_steps_deg), len(trans_steps_m), len(scale_steps_log))

    # [최적화 7] 딕셔너리(Map) 방식의 Dedup 적용
    all_results_map = {}
    teacher_pairs = []
    teacher_proposals = []

    for base_rank, base in enumerate(step7_top):
        base_anchor_id = int(base['id'])
        R_base = np.asarray(base['R_3x3'], dtype=np.float64)
        t_base = np.asarray(base['t_xyz_m'], dtype=np.float64)
        s_base = np.asarray(base['sx_sy_sz'], dtype=np.float64)
        state = make_state(R_base, t_base, s_base)
        best_overall = None

        print(f'[REFINE] base anchor {base_anchor_id} ({base_rank + 1}/{len(step7_top)})')

        current_score = -1e18
        state_cache = {}
        for si in range(nstage):
            step_rot = rot_steps_deg[si]
            step_trans = trans_steps_m[si]
            step_scale = scale_steps_log[si]
            for pass_idx in range(int(args.passes_per_stage)):
                proposals = perturb_states(state, step_rot, step_trans, step_scale)
                scored = evaluate_state_batch(
                    renderer=renderer,
                    K=Kc,
                    U=U,
                    center_raw_m=center_raw_m,
                    unit_scale=unit_scale,
                    query_rgb_bgr=query_rgb_bgr,
                    query_mask=query_mask,
                    query_depth=query_depth,
                    extractor=extractor,
                    q_dino=q_dino,
                    candidate_records=proposals,
                    state_cache=state_cache,
                    alpha_thresh=float(args.alpha_thresh), # 인자 넘겨주기
                )
                for prop_idx, rec in enumerate(scored):
                    teacher_conf_prop = float(np.clip(0.5 + 0.5 * rec['score'], 0.0, 1.0))

                    prop_stem = (
                        f'base_{base_anchor_id:05d}'
                        f'_stage_{si:02d}'
                        f'_pass_{pass_idx:02d}'
                        f'_prop_{prop_idx:03d}'
                    )
                    prop_uncert_path = os.path.join(out_dir, prop_stem + '_uncert_target.npy')

                    if 'uncert_target' in rec:
                        np.save(prop_uncert_path, rec['uncert_target'].astype(np.float32))
                        prop_uncert_abs = os.path.abspath(prop_uncert_path)
                    else:
                        prop_uncert_abs = None

                    teacher_proposals.append({
                        'proposal_id': prop_stem,
                        'base_anchor_id': int(base_anchor_id),
                        'stage_idx': int(si),
                        'pass_idx': int(pass_idx),
                        'kind': rec['kind'],

                        'base_pose': {
                            'R_3x3': R_base.tolist(),
                            't_xyz_m': t_base.tolist(),
                            'sx_sy_sz': s_base.tolist(),
                        },

                        'proposal_delta': {
                            'delta_deg_xyz': [float(x) for x in rec['delta_deg_xyz']],
                            'delta_t_xyz': [float(x) for x in rec['delta_t_xyz']],
                            'delta_log_sxyz': [float(x) for x in rec['delta_log_sxyz']],
                        },

                        'proposal_pose': {
                            'R_3x3': rec['R_3x3'],
                            't_xyz_m': rec['t_xyz_m'],
                            'sx_sy_sz': rec['sx_sy_sz'],
                            'A_3x3': rec['A_3x3'],
                            'b_xyz': rec['b_xyz'],
                        },

                        'teacher_target_for_step10': {
                            'teacher_confidence': teacher_conf_prop,
                            'uncert_target_npy': prop_uncert_abs,
                        },

                        'score_detail': {
                            'score': float(rec['score']),
                            'iou_filtered': float(rec['iou_filtered']),
                            'feat_score_masked_cosine': float(rec['feat_score_masked_cosine']),
                            'global_score_cosine': float(rec['global_score_cosine']),
                            'bbox_overlap': float(rec['bbox_overlap']),
                            'depth_score': float(rec['depth_score']),
                            'render_area_ratio_to_query': float(rec['render_area_ratio_to_query']),
                            'aspect_query': float(rec['aspect_query']),
                            'aspect_render': float(rec['aspect_render']),
                        },
                    })

                # [최적화 7] 출처 확인용 base_anchor_id 포함 Key Dedup 저장
                for rec in scored:
                    k = (base_anchor_id, _state_cache_key(rec['state']))
                    if k not in all_results_map or rec['score'] > all_results_map[k]['score']:
                        lean_rec = {
                                    key: val for key, val in rec.items()
                                    if key not in ['state', 'uncert_target']
                                        }
                        lean_rec['base_anchor_id'] = base_anchor_id
                        all_results_map[k] = lean_rec

                scored_sorted = sorted(scored, key=lambda x: x['score'], reverse=True)
                best_stage = scored_sorted[0]

                if best_overall is None or best_stage['score'] > best_overall['score']:
                    best_overall = best_stage
                if best_stage['score'] > current_score + 1e-12:
                    current_score = best_stage['score']
                    state = make_state(np.asarray(best_stage['R_3x3']), np.asarray(best_stage['t_xyz_m']), np.asarray(best_stage['sx_sy_sz']))
                else:
                    break

        if best_overall is None:
            continue

        # -----------------------------------------------------------------------------------
        # [최적화 6 & 8] 지연 렌더링 및 메타데이터 일치화
        # -----------------------------------------------------------------------------------
        st_best = best_overall['state']
        A_best, b_best, _ = compose_affine_from_pose(st_best['R'], st_best['t'], st_best['s'], U, center_raw_m, unit_scale)
        
        A_tensor = torch.tensor([A_best], dtype=torch.float32, device=renderer.device)
        b_tensor = torch.tensor([b_best], dtype=torch.float32, device=renderer.device)
        
        with torch.no_grad():
            rendered_best = renderer.render_batch(A_tensor, b_tensor, Kc)
            best_rgb_uint8 = (np.clip(rendered_best['rgb'][0].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
            best_mask = rendered_best['alpha'][0].cpu().numpy() > float(args.alpha_thresh) # 인자 사용
            best_depth = rendered_best['depth'][0].cpu().numpy()
        
        dino_best = extractor.extract(best_rgb_uint8)
        
        final_scored = score_render_record(
            query_rgb_bgr=query_rgb_bgr,
            query_mask=query_mask,
            query_depth=query_depth,
            q_dino=q_dino,
            render_rgb=best_rgb_uint8,  # 올바른 RGB 포맷
            render_mask=best_mask,
            render_depth=best_depth,
            batch_feat=dino_best['feat_map'].unsqueeze(0),
            batch_global=dino_best['global_feat'].unsqueeze(0),
            idx=0
        )
        
        # 메타데이터 완전 일치 (최종 평가된 점수로 best_overall 덮어쓰기)
        best_final = {
            **best_overall,
            **{k: v for k, v in final_scored.items() if k not in ['mask', 'rgb', 'depth', 'overlay', 'uncert_target']}
        }

        delta_rvec_teacher = relative_rvec(R_base, np.asarray(best_final['R_3x3'], dtype=np.float64)).tolist()
        delta_log_sxyz_teacher = (
            np.log(np.clip(np.asarray(best_final['sx_sy_sz'], dtype=np.float64), 1e-8, None)) -
            np.log(np.clip(s_base, 1e-8, None))
        ).tolist()
        delta_t_teacher = (np.asarray(best_final['t_xyz_m'], dtype=np.float64) - t_base).tolist()
        teacher_conf = float(np.clip(0.5 + 0.5 * best_final['score'], 0.0, 1.0))

        stem = f'base_{base_anchor_id:05d}'
        render_path = os.path.join(out_dir, stem + '_render.png')
        mask_path = os.path.join(out_dir, stem + '_mask.png')
        depth_path = os.path.join(out_dir, stem + '_depth.npy')
        overlay_path = os.path.join(out_dir, stem + '_overlay.png')
        uncert_path = os.path.join(out_dir, stem + '_uncert_target.npy')

        # [최적화 3] 저장을 위해 RGB를 BGR로 변환
        cv2.imwrite(render_path, cv2.cvtColor(final_scored['rgb'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, (final_scored['mask'].astype(np.uint8) * 255))
        save_depth(depth_path, final_scored['depth'])
        cv2.imwrite(overlay_path, final_scored['overlay']) # overlay_boundaries는 자체적으로 BGR 반환
        np.save(uncert_path, final_scored['uncert_target'].astype(np.float32))

        teacher_pairs.append({
            'base_anchor_id': base_anchor_id,
            'base_rotation_info': base,
            'teacher_best_candidate': {
                k: v for k, v in best_final.items()
                if k != 'state'
            },
            'teacher_target_for_step10': {
                'delta_rvec': delta_rvec_teacher,
                'delta_t': delta_t_teacher,
                'delta_log_sxyz': delta_log_sxyz_teacher,
                'teacher_confidence': teacher_conf,
                'uncert_target_npy': os.path.abspath(uncert_path),
            },
            'saved': {
                'render_png': os.path.abspath(render_path),
                'mask_png': os.path.abspath(mask_path),
                'depth_npy': os.path.abspath(depth_path),
                'overlay_png': os.path.abspath(overlay_path),
            },
        })

    teacher_pairs_path = os.path.join(out_dir, 'teacher_pairs.json')
    save_json(teacher_pairs_path, {'pairs': teacher_pairs})
    teacher_proposals_path = os.path.join(out_dir, 'teacher_proposals.json')
    save_json(teacher_proposals_path, {'proposals': teacher_proposals})

    all_results_sorted = sorted(all_results_map.values(), key=lambda x: x['score'], reverse=True)
    summary_out = {
        'capture_dir': os.path.abspath(args.capture_dir),
        'mesh_path': os.path.abspath(args.mesh),
        'mesh_unit': args.mesh_unit,
        'dino': {
            'model_name': args.model_name,
            'device': args.device,
            'dtype': args.dtype,
        },
        'refine': {
            'top_from_step7': int(args.top_from_step7),
            'rot_steps_deg': rot_steps_deg[:nstage],
            'trans_steps_mm': parse_float_list(args.trans_steps_mm)[:nstage],
            'scale_steps_log': scale_steps_log[:nstage],
            'passes_per_stage': int(args.passes_per_stage),
        },
        'num_teacher_pairs': len(teacher_pairs),
        'teacher_pairs_path': os.path.abspath(teacher_pairs_path),
        'num_teacher_proposals': len(teacher_proposals),
        'teacher_proposals_path': os.path.abspath(teacher_proposals_path),
    }
    summary_path_out = os.path.join(out_dir, 'summary.json')
    save_json(summary_path_out, summary_out)

    print('[OK] noblender teacher refine saved')
    print('  out_dir       :', out_dir)
    print('  teacher_pairs :', teacher_pairs_path)
    print('  num_pairs     :', len(teacher_pairs))


if __name__ == '__main__':
    main()