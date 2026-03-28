#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math  # [수정 1] math 임포트 추가
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

def load_pt(path: str) -> torch.Tensor:
    return torch.load(path, map_location='cpu')


def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def save_pt(path: str, tensor: torch.Tensor):
    torch.save(tensor.cpu(), path)


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_imagenet_tensor(path: str, device: str) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(device)


def scaled_hw(h: int, w: int, num: int = 3, den: int = 2) -> tuple[int, int]:
    return max(1, (int(h) * int(num)) // int(den)), max(1, (int(w) * int(num)) // int(den))


class AnyUpRefiner(nn.Module):
    def __init__(
        self,
        repo: str = 'wimmerth/anyup',
        model_name: str = 'anyup_multi_backbone',
        use_natten: bool = True,
        q_chunk_size: int = 256,
        device: str = 'cuda',
    ):
        super().__init__()
        hub_kwargs = {'verbose': False}
        if model_name != 'anyup':
            hub_kwargs['use_natten'] = bool(use_natten)

        self.q_chunk_size = int(q_chunk_size)
        self.upsampler = torch.hub.load(repo, model_name, **hub_kwargs).to(device).eval()

        for p in self.upsampler.parameters():
            p.requires_grad_(False)

    def forward(self, hr_image: torch.Tensor, lr_features: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        return self.upsampler(
            hr_image,
            lr_features,
            output_size=output_hw,
            q_chunk_size=self.q_chunk_size,
        )


def build_query_aux_tensor(aux_npz: Dict[str, np.ndarray]) -> torch.Tensor:
    arr = np.stack([
        aux_npz['mask'],
        aux_npz['boundary'],
        aux_npz['depth'],
        aux_npz['depth_edge'],
        aux_npz['valid'],
        aux_npz['valid_edge'],
    ], axis=0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def build_anchor_aux_tensor(aux_npz: Dict[str, np.ndarray]) -> torch.Tensor:
    arr = np.stack([
        aux_npz['mask'],
        aux_npz['boundary'],
        aux_npz['depth'],
        aux_npz['depth_edge'],
        aux_npz['alpha'],
        aux_npz['alpha_edge'],
    ], axis=0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--anyup_repo', default='wimmerth/anyup')
    ap.add_argument('--anyup_model', default='anyup_multi_backbone')
    ap.add_argument('--anyup_use_natten', type=int, default=1)
    ap.add_argument('--q_chunk_size', type=int, default=256)
    ap.add_argument('--upsample_num', type=int, default=3)
    ap.add_argument('--upsample_den', type=int, default=2)


    args = ap.parse_args()

    feature_pack_dir = os.path.join(args.capture_dir, 'feature_pack')
    if not os.path.exists(feature_pack_dir):
        raise FileNotFoundError(feature_pack_dir)

    out_dir = os.path.join(args.capture_dir, 'refined_feature_pack')
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(feature_pack_dir, 'summary.json')
    summary = load_json(summary_path)

    query_feat_path = os.path.join(feature_pack_dir, 'query_feat_map.pt')
    query_aux_path = os.path.join(feature_pack_dir, 'query_aux_maps.npz')

    query_feat = load_pt(query_feat_path).float().unsqueeze(0)
    query_aux = build_query_aux_tensor(load_npz(query_aux_path)).float()

    feat_ch = int(query_feat.shape[1])
    aux_ch = int(query_aux.shape[1])

    query_meta_in = load_json(summary['query_meta'])
    query_image_path = query_meta_in['input_path']

    refiner = AnyUpRefiner(
        repo=args.anyup_repo,
        model_name=args.anyup_model,
        use_natten=bool(args.anyup_use_natten),
        q_chunk_size=int(args.q_chunk_size),
        device=args.device,
    )
    print(
        f"[INFO] loaded AnyUp refiner: repo={args.anyup_repo}, "
        f"model={args.anyup_model}, use_natten={bool(args.anyup_use_natten)}"
    )

    query_target_hw = scaled_hw(
        query_feat.shape[-2],
        query_feat.shape[-1],
        num=int(args.upsample_num),
        den=int(args.upsample_den),
    )

    with torch.inference_mode():
        query_hr_image = load_rgb_imagenet_tensor(query_image_path, device=args.device)
        query_refined = refiner(
            query_hr_image,
            query_feat.to(args.device),
            output_hw=query_target_hw,
        ).cpu()[0]

    query_refined_path = os.path.join(out_dir, 'query_refined_feat.pt')
    save_pt(query_refined_path, query_refined)

    query_meta = {
        'input_query_feat_path': os.path.abspath(query_feat_path),
        'input_query_aux_path': os.path.abspath(query_aux_path),
        'refined_feat_path': os.path.abspath(query_refined_path),
        'shape_chw': list(map(int, query_refined.shape)),
        'aux_ch': aux_ch,
        'feat_ch': feat_ch,
        'refiner': 'anyup',
        'target_hw': list(map(int, query_refined.shape[-2:])),
    }
    query_refined_meta_path = os.path.join(out_dir, 'query_refined_meta.json')
    with open(query_refined_meta_path, 'w') as f:
        json.dump(query_meta, f, indent=2)

    anchor_records = []

    for rec in summary['anchors']:
        anchor_id = int(rec['anchor_id'])
        stem = f'cand_{anchor_id:04d}'
        feat_path = os.path.join(feature_pack_dir, f'{stem}_feat_map.pt')
        aux_path = os.path.join(feature_pack_dir, f'{stem}_aux_maps.npz')
        if not (os.path.exists(feat_path) and os.path.exists(aux_path)):
            print(f'[WARN] missing anchor feature files for {stem}, skip')
            continue
        anchor_feat = load_pt(feat_path).float().unsqueeze(0)
        anchor_aux = build_anchor_aux_tensor(load_npz(aux_path)).float()

        if anchor_feat.shape[-2:] != query_feat.shape[-2:]:
            raise RuntimeError(
                f"[ERROR] anchor/query feature grid mismatch before AnyUp: "
                f"query={tuple(query_feat.shape[-2:])}, anchor={tuple(anchor_feat.shape[-2:])}"
            )

        with torch.inference_mode():
            anchor_hr_image = load_rgb_imagenet_tensor(rec['render_path'], device=args.device)
            anchor_refined = refiner(
                anchor_hr_image,
                anchor_feat.to(args.device),
                output_hw=query_target_hw,
            ).cpu()[0]

        refined_path = os.path.join(out_dir, f'{stem}_refined_feat.pt')
        save_pt(refined_path, anchor_refined)
        meta = {
            'anchor_id': anchor_id,
            'input_feat_path': os.path.abspath(feat_path),
            'input_aux_path': os.path.abspath(aux_path),
            'refined_feat_path': os.path.abspath(refined_path),
            'shape_chw': list(map(int, anchor_refined.shape)),
            'rotation_info_from_step7': rec['rotation_info_from_step7'],
            'refiner': 'anyup'
        }
        meta_path = os.path.join(out_dir, f'{stem}_refined_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        anchor_records.append(meta)

    summary_out = {
        'capture_dir': os.path.abspath(args.capture_dir),
        'refiner': 'anyup',
        'anyup_repo': args.anyup_repo,
        'anyup_model': args.anyup_model,
        'anyup_use_natten': bool(args.anyup_use_natten),
        'q_chunk_size': int(args.q_chunk_size),
        'upsample_num': int(args.upsample_num),
        'upsample_den': int(args.upsample_den),
        'query_refined_meta': os.path.abspath(os.path.join(out_dir, 'query_refined_meta.json')),
        'num_anchors': len(anchor_records),
        'anchors': anchor_records,
    }
    summary_out_path = os.path.join(out_dir, 'summary.json')
    with open(summary_out_path, 'w') as f:
        json.dump(summary_out, f, indent=2)

    print('\n[OK] noblender refined feature pack saved')
    print('  out_dir :', out_dir)
    print('  summary :', summary_out_path)


if __name__ == '__main__':
    main()