#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from noblender_common import (
    load_alpha,
    load_color_rgb,
    load_depth,
    load_mask,
    mask_boundary_map,
    resize_mask,
    save_json,
    sobel_edge_map,
)
from rope3d_common import build_patch_xyz_map


# ---------------------------------------------------------
# [수정 4] 전경(Foreground) 중심 Depth 정규화 함수 새로 정의
# (배경 0 때문에 물체의 다이내믹 레인지가 압축되는 현상 방지)
# ---------------------------------------------------------
def normalize_depth_foreground(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    d_norm = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        mn = depth[valid].min()
        mx = depth[valid].max()
        d_norm[valid] = (depth[valid] - mn) / max(mx - mn, 1e-8)
    return d_norm


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

    def _prepare_image(self, rgb_uint8: np.ndarray):
        """ DINO processor의 임의 Resize/Crop을 막기 위해 직접 Patch Size 배수로 맞춤 """
        H, W = rgb_uint8.shape[:2]
        
        # [수정 2] 0-size 생성 잠복 버그 방어 (최소 patch_size 보장)
        new_H = max(self.patch_size, (H // self.patch_size) * self.patch_size)
        new_W = max(self.patch_size, (W // self.patch_size) * self.patch_size)
        
        if new_H != H or new_W != W:
            rgb_uint8 = cv2.resize(rgb_uint8, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            
        return Image.fromarray(rgb_uint8.astype('uint8'))

    @torch.inference_mode()
    def extract(self, rgb_uint8: np.ndarray):
        img = self._prepare_image(rgb_uint8)
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
        imgs = [self._prepare_image(img) for img in rgb_uint8_list]
        
        # [수정 3] extract_batch 혼합 해상도 텐서 스택 폭발 방어
        sizes = [img.size for img in imgs]
        if len(set(sizes)) != 1:
            raise ValueError(f"[ERROR] Anchor batch has mixed image sizes: {set(sizes)}")

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
            'input_hw': [int(H_in), int(W_in)]
        }


def save_tensor(path: str, tensor: torch.Tensor):
    torch.save(tensor.cpu(), path)


def save_npz(path: str, **kwargs):
    # 압축 해제 병목 제거를 위해 savez 사용
    np.savez(path, **kwargs)


def build_query_aux_maps(query_mask: np.ndarray, query_depth: np.ndarray, out_hw):
    ph, pw = out_hw
    q_mask_small = cv2.resize(query_mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    q_boundary = cv2.resize(mask_boundary_map(query_mask), (pw, ph), interpolation=cv2.INTER_NEAREST)

    depth_valid = (query_depth > 0).astype(np.float32)
    depth_valid_small = cv2.resize(depth_valid, (pw, ph), interpolation=cv2.INTER_NEAREST)
    depth_valid_edge = cv2.resize(mask_boundary_map(depth_valid > 0.5), (pw, ph), interpolation=cv2.INTER_NEAREST)

    # 전경 중심 Depth 정규화 후 NEAREST Resize 적용
    q_depth_norm = normalize_depth_foreground(query_depth)
    q_depth_small = cv2.resize(q_depth_norm, (pw, ph), interpolation=cv2.INTER_NEAREST)
    q_depth_edge = cv2.resize(sobel_edge_map(query_depth), (pw, ph), interpolation=cv2.INTER_NEAREST)

    return {
        'mask': q_mask_small,
        'boundary': q_boundary,
        'depth': q_depth_small,
        'depth_edge': q_depth_edge,
        'valid': depth_valid_small,
        'valid_edge': depth_valid_edge,
    }


def build_anchor_aux_maps(anchor_alpha: np.ndarray, anchor_mask: np.ndarray, anchor_depth: np.ndarray, out_hw):
    ph, pw = out_hw
    a_mask_small = cv2.resize(anchor_mask.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    a_boundary = cv2.resize(mask_boundary_map(anchor_mask), (pw, ph), interpolation=cv2.INTER_NEAREST)

    # 전경 중심 Depth 정규화 후 NEAREST Resize 적용
    depth_norm = normalize_depth_foreground(anchor_depth)
    depth_small = cv2.resize(depth_norm, (pw, ph), interpolation=cv2.INTER_NEAREST)
    depth_edge = cv2.resize(sobel_edge_map(anchor_depth), (pw, ph), interpolation=cv2.INTER_NEAREST)

    alpha01 = anchor_alpha.astype(np.float32) / 255.0
    alpha_small = cv2.resize(alpha01, (pw, ph), interpolation=cv2.INTER_LINEAR)
    alpha_edge = cv2.resize(sobel_edge_map(alpha01), (pw, ph), interpolation=cv2.INTER_LINEAR)

    return {
        'mask': a_mask_small,
        'boundary': a_boundary,
        'depth': depth_small,
        'depth_edge': depth_edge,
        'alpha': alpha_small,
        'alpha_edge': alpha_edge,
    }


def select_topk_from_summary(summary: dict, k: int):
    return summary['results'][: min(k, len(summary['results']))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dir', required=True)
    ap.add_argument('--model_name', default='facebook/dinov3-vits16-pretrain-lvd1689m')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--xyz_clip', type=float, default=4.0)
    args = ap.parse_args()

    query_crop_path = os.path.join(args.capture_dir, 'calib_prep', 'query_crop.png')
    query_mask_path = os.path.join(args.capture_dir, 'calib_prep', 'query_mask.png')
    query_depth_path = os.path.join(args.capture_dir, 'calib_prep', 'query_depth_crop.npy')
    calib_json_path = os.path.join(args.capture_dir, 'calib_input.json')
    summary_path = os.path.join(args.capture_dir, 'rotation_scale_bank', 'summary.json')
    rotation_bank_dir = os.path.join(args.capture_dir, 'rotation_scale_bank')

    for p in [query_crop_path, query_mask_path, query_depth_path, calib_json_path, summary_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    out_dir = os.path.join(args.capture_dir, 'feature_pack')
    os.makedirs(out_dir, exist_ok=True)

    query_rgb = load_color_rgb(query_crop_path)
    query_mask = load_mask(query_mask_path)
    query_depth = load_depth(query_depth_path)
    
    with open(calib_json_path, 'r') as f:
        calib = json.load(f)
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    topk = select_topk_from_summary(summary, int(args.top_k))
    if not topk:
        raise RuntimeError("[ERROR] no saved step7 results found in rotation_scale_bank/summary.json")

    intrinsics_crop = calib['intrinsics_crop']
    t0_xyz_m = calib['init_translation_t0_xyz_m']
    mesh_diag_m = float(calib['mesh_bbox_diag_m'])

    print(f'[INFO] selected top-{len(topk)} anchors from Step 7')
    print('[DINOv3] loading model...')
    extractor = DinoV3Extractor(model_name=args.model_name, device=args.device, dtype=args.dtype)

    print('[DINOv3] extracting query feature...')
    q = extractor.extract(query_rgb)
    query_aux = build_query_aux_maps(query_mask=query_mask, query_depth=query_depth, out_hw=(q['ph'], q['pw']))
    query_xyz = build_patch_xyz_map(
        depth_m=query_depth,
        valid_mask=(query_mask > 0) & (query_depth > 0),
        intrinsics_crop=intrinsics_crop,
        out_hw=(q['ph'], q['pw']),
        center_xyz_m=t0_xyz_m,
        scale_ref_m=mesh_diag_m,
        clip_value=float(args.xyz_clip),
    )

    save_tensor(os.path.join(out_dir, 'query_feat_map.pt'), q['feat_map'])
    save_tensor(os.path.join(out_dir, 'query_global_feat.pt'), q['global_feat'])
    save_npz(
        os.path.join(out_dir, 'query_aux_maps.npz'),
        mask=query_aux['mask'],
        boundary=query_aux['boundary'],
        depth=query_aux['depth'],
        depth_edge=query_aux['depth_edge'],
        valid=query_aux['valid'],
        valid_edge=query_aux['valid_edge'],
    )
    query_xyz_path = os.path.join(out_dir, 'query_xyz_map.npy')
    np.save(query_xyz_path, query_xyz.astype(np.float32))

    query_meta = {
        'input_path': os.path.abspath(query_crop_path),
        'mask_path': os.path.abspath(query_mask_path),
        'depth_path': os.path.abspath(query_depth_path),
        'xyz_path': os.path.abspath(query_xyz_path),
        'xyz_space': 'camera_local_centered',
        'xyz_center_xyz_m': list(map(float, t0_xyz_m)),
        'xyz_scale_ref_m': float(mesh_diag_m),
        'feat_shape_chw': list(map(int, q['feat_map'].shape)),
        'ph': q['ph'],
        'pw': q['pw'],
        'input_hw_after_processor': q['input_hw'],
    }
    save_json(os.path.join(out_dir, 'query_meta.json'), query_meta)

    # -------------------------------------------------------------
    # Anchor Batch Inference
    # -------------------------------------------------------------
    valid_anchors = []
    
    for rank, rec in enumerate(topk):
        required_step7_fields = [
            'id', 'score', 'iou_filtered',
            'R_3x3', 't_xyz_m', 'sx_sy_sz',
            'A_3x3', 'b_xyz',
        ]
        for key in required_step7_fields:
            if key not in rec:
                raise RuntimeError(f"[ERROR] step8 anchor missing step7 field: {key}")

        if 'paths' not in rec:
            raise RuntimeError("[ERROR] step8 anchor missing step7 paths")

        required_step7_paths = ['render_png', 'alpha_png', 'mask_png', 'depth_npy']
        for key in required_step7_paths:
            if key not in rec['paths']:
                raise RuntimeError(f"[ERROR] step8 anchor missing step7 path field: {key}")

        anchor_id = int(rec['id'])
        stem = f'cand_{anchor_id:04d}'
        anchor_render_path = rec.get('paths', {}).get('render_png', os.path.join(rotation_bank_dir, stem + '_render.png'))
        anchor_alpha_path = rec.get('paths', {}).get('alpha_png', os.path.join(rotation_bank_dir, stem + '_alpha.png'))
        anchor_mask_path = rec.get('paths', {}).get('mask_png', os.path.join(rotation_bank_dir, stem + '_mask.png'))
        anchor_depth_path = rec.get('paths', {}).get('depth_npy', os.path.join(rotation_bank_dir, stem + '_depth.npy'))

        if not (os.path.exists(anchor_render_path) and os.path.exists(anchor_alpha_path) and os.path.exists(anchor_mask_path)):
            print(f'[WARN] missing anchor files for {stem}, skip')
            continue

        if not os.path.exists(anchor_depth_path):
            print(f'[WARN] missing anchor depth for {stem}, skip')
            continue
            
        anchor_depth = load_depth(anchor_depth_path)
        anchor_rgb = load_color_rgb(anchor_render_path)
        anchor_alpha = load_alpha(anchor_alpha_path)
        anchor_mask = load_mask(anchor_mask_path)
        
        valid_anchors.append({
            'rank': rank,
            'rec': rec,
            'stem': stem,
            'rgb': anchor_rgb,
            'alpha': anchor_alpha,
            'mask': anchor_mask,
            'depth': anchor_depth,
            'paths': {
                'render': anchor_render_path,
                'alpha': anchor_alpha_path,
                'mask': anchor_mask_path,
                'depth': anchor_depth_path
            }
        })

    if not valid_anchors:
        print("[WARN] No valid anchors found to process!")
        return

    print(f'[DINOv3] extracting anchor features in batch (N={len(valid_anchors)})...')
    
    # 이미지 리스트를 Batch로 한 번에 추론 (내부에서 혼합 사이즈 예외처리됨)
    batch_rgb_list = [v['rgb'] for v in valid_anchors]
    batch_features = extractor.extract_batch(batch_rgb_list)
    
    anchor_records = []
    
    for i, v in enumerate(valid_anchors):
        rec = v['rec']
        stem = v['stem']
        rank = v['rank']
        
        a_feat_map = batch_features['feat_map'][i]
        a_global_feat = batch_features['global_feat'][i]
        ph = batch_features['ph']
        pw = batch_features['pw']
        
        anchor_aux = build_anchor_aux_maps(
            anchor_alpha=v['alpha'], 
            anchor_mask=v['mask'], 
            anchor_depth=v['depth'], 
            out_hw=(ph, pw)
        )

        # 다운스트림 호환용으로 t_xyz_m 우선 사용, 없으면 t0 사용
        anchor_center_xyz = rec.get('t_xyz_m', t0_xyz_m)
        
        anchor_xyz = build_patch_xyz_map(
            depth_m=v['depth'],
            valid_mask=(v['mask'] > 0) & (v['depth'] > 0),
            intrinsics_crop=intrinsics_crop,
            out_hw=(ph, pw),
            center_xyz_m=anchor_center_xyz,
            scale_ref_m=mesh_diag_m,
            clip_value=float(args.xyz_clip),
        )

        feat_map_path = os.path.join(out_dir, f'{stem}_feat_map.pt')
        global_feat_path = os.path.join(out_dir, f'{stem}_global_feat.pt')
        aux_map_path = os.path.join(out_dir, f'{stem}_aux_maps.npz')
        xyz_path = os.path.join(out_dir, f'{stem}_xyz_map.npy')
        meta_path = os.path.join(out_dir, f'{stem}_meta.json')

        save_tensor(feat_map_path, a_feat_map)
        save_tensor(global_feat_path, a_global_feat)
        save_npz(
            aux_map_path,
            mask=anchor_aux['mask'],
            boundary=anchor_aux['boundary'],
            depth=anchor_aux['depth'],
            depth_edge=anchor_aux['depth_edge'],
            alpha=anchor_aux['alpha'],
            alpha_edge=anchor_aux['alpha_edge'],
        )
        np.save(xyz_path, anchor_xyz.astype(np.float32))

        meta = {
            'root_id': rec.get('root_id'),
            'root_source': rec.get('root_source'),
            'root_rank': rec.get('root_rank'),
            'branch_type': rec.get('branch_type'),
            'flip_group_id': rec.get('flip_group_id'),
            'rank_in_topk': int(rank),
            'anchor_id': int(rec['id']),
            'rotation_info_from_step7': {
                'search_mode': rec.get('search_mode'),
                'view_dir': rec.get('view_dir'),
                'roll_deg': rec.get('roll_deg'),
                'scale_kind': rec.get('scale_kind'),
                'sx_sy_sz': rec.get('sx_sy_sz'),
                'score': rec.get('score'),
                'iou_filtered': rec.get('iou_filtered'),
                # Step 11 호환성을 보장하는 핵심 메타데이터 유지
                'R_3x3': rec.get('R_3x3'),
                't_xyz_m': rec.get('t_xyz_m'),
                'A_3x3': rec.get('A_3x3'),
                'b_xyz': rec.get('b_xyz'),
            },
            'render_path': os.path.abspath(v['paths']['render']),
            'alpha_path': os.path.abspath(v['paths']['alpha']),
            'mask_path': os.path.abspath(v['paths']['mask']),
            'depth_path': os.path.abspath(v['paths']['depth']),
            'xyz_path': os.path.abspath(xyz_path),
            'xyz_space': 'camera_local_centered',
            'xyz_center_xyz_m': list(map(float, anchor_center_xyz)),
            'xyz_scale_ref_m': float(mesh_diag_m),
            'feat_shape_chw': list(map(int, a_feat_map.shape)),
            'ph': ph,
            'pw': pw,
            'input_hw_after_processor': batch_features['input_hw'],
        }
        save_json(meta_path, meta)
        anchor_records.append(meta)

    summary_out = {
        'capture_dir': os.path.abspath(args.capture_dir),
        'model_name': args.model_name,
        'device': args.device,
        'dtype': args.dtype,
        'top_k_requested': int(args.top_k),
        'top_k_saved': int(len(anchor_records)),
        'query_meta': os.path.abspath(os.path.join(out_dir, 'query_meta.json')),
        'anchors': anchor_records,
    }
    summary_out_path = os.path.join(out_dir, 'summary.json')
    save_json(summary_out_path, summary_out)

    print('\n[OK] noblender + 3D RoPE feature pack saved')
    print('  out_dir :', out_dir)
    print('  summary :', summary_out_path)


if __name__ == '__main__':
    main()