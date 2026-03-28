#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import Dict

import cv2
import numpy as np
import torch

from rope3d_alignment_model import (
    CrossAttentionAlignmentModel3DRoPE,
    build_anchor_aux_tensor,
    build_query_aux_tensor,
    load_npz,
    load_pt,
    masked_patch_cosine_score,
)


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_xyz_npy(path: str) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f'xyz map must be [H,W,3], got {arr.shape} @ {path}')
    return torch.from_numpy(arr)
def sigmoid_np(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def compute_hybrid_score(rec: dict, w_score: float, w_pre: float, w_iou: float, w_step7: float) -> float:
    score_prob = sigmoid_np(float(rec['score_logit']))
    pre_score = float(rec['pre_score_masked_cosine'])

    step7 = rec.get('rotation_info_from_step7', {})
    iou_filtered = float(step7.get('iou_filtered', 0.0))
    step7_score = float(step7.get('score', 0.0))

    hybrid = (
        w_score * score_prob +
        w_pre   * pre_score +
        w_iou   * iou_filtered +
        w_step7 * step7_score
    )
    return float(hybrid)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dir', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--embed_dim', type=int, default=256)
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--rope_base', type=float, default=1000.0)
    ap.add_argument('--post_self_layers', type=int, default=1)
    ap.add_argument('--post_self_dropout', type=float, default=0.0)
    ap.add_argument('--rank_mode', choices=['pre', 'logit', 'hybrid'], default='hybrid')
    ap.add_argument('--w_score', type=float, default=0.45)
    ap.add_argument('--w_pre', type=float, default=0.20)
    ap.add_argument('--w_iou', type=float, default=0.25)
    ap.add_argument('--w_step7', type=float, default=0.10)
    args = ap.parse_args()

    refined_dir = os.path.join(args.capture_dir, 'refined_feature_pack')
    feature_pack_dir = os.path.join(args.capture_dir, 'feature_pack')

    refined_summary_path = os.path.join(refined_dir, 'summary.json')
    feature_summary_path = os.path.join(feature_pack_dir, 'summary.json')
    query_feat_path = os.path.join(refined_dir, 'query_refined_feat.pt')
    query_aux_path = os.path.join(feature_pack_dir, 'query_aux_maps.npz')
    query_xyz_path = os.path.join(feature_pack_dir, 'query_xyz_map.npy')

    for p in [refined_summary_path, feature_summary_path, query_feat_path, query_aux_path, query_xyz_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    refined_summary = load_json(refined_summary_path)
    feature_summary = load_json(feature_summary_path)

    feature_anchor_meta: Dict[int, dict] = {}
    for rec in feature_summary['anchors']:
        feature_anchor_meta[int(rec['anchor_id'])] = rec

    # ---------------------------------------------------------
    # [최적화 1] Query 텐서 루프 바깥에서 단 한 번만 GPU 적재
    # ---------------------------------------------------------
    query_feat = load_pt(query_feat_path).float().unsqueeze(0).to(args.device)
    query_aux = build_query_aux_tensor(load_npz(query_aux_path)).float().unsqueeze(0).to(args.device)
    query_xyz = load_xyz_npy(query_xyz_path).float().unsqueeze(0).to(args.device)

    model = CrossAttentionAlignmentModel3DRoPE(
        feat_ch=int(query_feat.shape[1]),
        aux_ch=int(query_aux.shape[1]),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        rope_base=float(args.rope_base),
        post_self_layers=int(args.post_self_layers),
        post_self_dropout=float(args.post_self_dropout),
    ).to(args.device)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f'[INFO] loaded checkpoint: {args.ckpt}')
    else:
        print('[INFO] no checkpoint provided -> zero/small initialized 3D RoPE alignment head')

    model.eval()

    out_dir = os.path.join(args.capture_dir, 'alignment_head_forward_3drope')
    os.makedirs(out_dir, exist_ok=True)

    pair_records = []
    for rec in refined_summary['anchors']:
        anchor_id = int(rec['anchor_id'])
        stem = f'cand_{anchor_id:04d}'
        anchor_feat_path = os.path.join(refined_dir, f'{stem}_refined_feat.pt')
        anchor_aux_path = rec['input_aux_path']

        if anchor_id not in feature_anchor_meta:
            print(f'[WARN] missing feature_pack meta for {stem}, skip')
            continue
        anchor_xyz_path = feature_anchor_meta[anchor_id].get('xyz_path', os.path.join(feature_pack_dir, f'{stem}_xyz_map.npy'))

        if not (os.path.exists(anchor_feat_path) and os.path.exists(anchor_aux_path) and os.path.exists(anchor_xyz_path)):
            print(f'[WARN] missing anchor input for {stem}, skip')
            continue

        # Anchor 텐서 로드 즉시 GPU로 적재하여 연산 병목 제거
        anchor_feat = load_pt(anchor_feat_path).float().unsqueeze(0).to(args.device)
        anchor_aux = build_anchor_aux_tensor(load_npz(anchor_aux_path)).float().unsqueeze(0).to(args.device)
        anchor_xyz = load_xyz_npy(anchor_xyz_path).float().unsqueeze(0).to(args.device)

        pre_score = masked_patch_cosine_score(query_feat, anchor_feat, query_aux, anchor_aux)

        with torch.inference_mode():
            out = model(
                query_feat,
                query_aux,
                query_xyz,
                anchor_feat,
                anchor_aux,
                anchor_xyz,
            )

        score_logit = float(out['score_logit'].cpu()[0, 0].item())
        delta_rvec = out['delta_rvec'].cpu()[0].numpy().tolist()
        delta_t = out['delta_t'].cpu()[0].numpy().tolist()
        delta_log_sxyz = out['delta_log_sxyz'].cpu()[0].numpy().tolist()
        
        # ---------------------------------------------------------
        # [최적화 2] Logit을 Sigmoid 적용하여 0~1 사이의 확률값으로 저장
        # ---------------------------------------------------------
        uncert_logit = out['uncert_logit_map'][0, 0] # GPU에서 직접 접근
        uncert_prob = torch.sigmoid(uncert_logit).cpu().numpy().astype(np.float32)

        # 직관성을 위해 파일명 _logit -> _prob 로 변경
        uncert_png_path = os.path.join(out_dir, f'{stem}_uncert_prob.png')
        cv2.imwrite(uncert_png_path, (uncert_prob * 255.0).clip(0, 255).astype(np.uint8))
        
        uncert_npy_path = os.path.join(out_dir, f'{stem}_uncert_prob.npy')
        np.save(uncert_npy_path, uncert_prob)

        pair_meta = {
            'anchor_id': anchor_id,
            'rotation_info_from_step7': rec['rotation_info_from_step7'],
            'pre_score_masked_cosine': float(pre_score),
            'score_logit': float(score_logit),
            'delta_rvec': delta_rvec,
            'delta_t': delta_t,
            'delta_log_sxyz': delta_log_sxyz,
            'query_xyz_path': os.path.abspath(query_xyz_path),
            'anchor_xyz_path': os.path.abspath(anchor_xyz_path),
            'uncert_prob_map_path': os.path.abspath(uncert_npy_path),
            'uncert_prob_vis_path': os.path.abspath(uncert_png_path),
        }
        pair_meta['score_prob'] = float(sigmoid_np(pair_meta['score_logit']))

        step7_info = pair_meta.get('rotation_info_from_step7', {})
        pair_meta['step7_iou_filtered'] = float(step7_info.get('iou_filtered', 0.0))
        pair_meta['step7_score'] = float(step7_info.get('score', 0.0))

        pair_meta['hybrid_score'] = compute_hybrid_score(
            pair_meta,
            w_score=float(args.w_score),
            w_pre=float(args.w_pre),
            w_iou=float(args.w_iou),
            w_step7=float(args.w_step7),
)
        pair_meta_path = os.path.join(out_dir, f'{stem}_forward.json')
        save_json(pair_meta_path, pair_meta)
        pair_records.append(pair_meta)
    if len(pair_records) == 0:
        raise RuntimeError('No anchor forward results produced.')

    if args.rank_mode == 'pre':
        pair_records = sorted(pair_records, key=lambda x: x['pre_score_masked_cosine'], reverse=True)
        best_by = 'pre_score_masked_cosine'
    elif args.rank_mode == 'logit':
        pair_records = sorted(pair_records, key=lambda x: x['score_logit'], reverse=True)
        best_by = 'score_logit'
    else:
        pair_records = sorted(pair_records, key=lambda x: x['hybrid_score'], reverse=True)
        best_by = 'hybrid_score'

    best = pair_records[0]
    summary_out = {
        'capture_dir': os.path.abspath(args.capture_dir),
        'checkpoint': os.path.abspath(args.ckpt) if args.ckpt is not None else None,
        'rope_base': float(args.rope_base),
        'best_by': best_by,
        'best': best,
        'pairs': pair_records,
    }
    summary_out_path = os.path.join(out_dir, 'summary.json')
    save_json(summary_out_path, summary_out)

    print('\n[OK] 3D RoPE cross-attention alignment forward saved')
    print('  out_dir :', out_dir)
    print('  summary :', summary_out_path)
    print('  best    :', {
        'anchor_id': best['anchor_id'],
        'best_by': best_by,
        'pre_score_masked_cosine': best['pre_score_masked_cosine'],
        'score_logit': best['score_logit'],
        'delta_rvec': best['delta_rvec'],
        'delta_t': best['delta_t'],
        'delta_log_sxyz': best['delta_log_sxyz'],
    })


if __name__ == '__main__':
    main()