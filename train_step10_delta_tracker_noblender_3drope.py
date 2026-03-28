#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rope3d_alignment_model import (
    CrossAttentionAlignmentModel3DRoPE,
    build_anchor_aux_tensor,
    build_query_aux_tensor,
    load_npz,
    load_pt,
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


def axis_angle_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    # rvec: [B,3]
    theta = torch.linalg.norm(rvec, dim=1, keepdim=True).clamp_min(1e-8)
    axis = rvec / theta
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    K = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=1).view(-1, 3, 3)
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).unsqueeze(0).expand(rvec.shape[0], -1, -1)
    th = theta.view(-1, 1, 1)
    R = I + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)
    return R


def geodesic_angle_from_mats(R_pred: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    R_rel = R_pred.transpose(1, 2) @ R_tgt
    tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def weighted_geodesic_loss(pred_rvec: torch.Tensor, target_rvec: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    R_pred = axis_angle_to_matrix(pred_rvec)
    R_tgt = axis_angle_to_matrix(target_rvec)
    ang = geodesic_angle_from_mats(R_pred, R_tgt)
    return (ang * weight).mean()


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction='none').mean(dim=1)
    return (loss * weight).mean()


def confidence_loss(score_logit: torch.Tensor, teacher_conf: torch.Tensor) -> torch.Tensor:
    target = teacher_conf.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(score_logit[:, 0], target)


def uncertainty_supervision_loss(uncert_logit_map: torch.Tensor, uncert_target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(uncert_logit_map, uncert_target)


def uncertainty_regularization(uncert_logit_map: torch.Tensor) -> torch.Tensor:
    return 1e-3 * (uncert_logit_map ** 2).mean()


@dataclass
class PairRecord:
    capture_dir: str
    query_feat_path: str
    query_aux_path: str
    query_xyz_path: str
    anchor_feat_path: str
    anchor_aux_path: str
    anchor_xyz_path: str
    teacher_confidence: float
    delta_rvec: List[float]
    delta_t: List[float]
    delta_log_sxyz: List[float]
    uncert_target_path: str
    base_anchor_id: int


class TrackingPairDataset(Dataset):
    def __init__(self, capture_dirs: List[str]):
        self.records: List[PairRecord] = []
        for capture_dir in capture_dirs:
            tp_path = os.path.join(capture_dir, 'tracking_pairs', 'tracking_pairs.json')
            if not os.path.exists(tp_path):
                raise FileNotFoundError(tp_path)
            pairs = load_json(tp_path)['pairs']
            for pair in pairs:
                self.records.append(PairRecord(
                    capture_dir=os.path.abspath(capture_dir),
                    query_feat_path=os.path.abspath(pair['query_feat_path']),
                    query_aux_path=os.path.abspath(pair['query_aux_path']),
                    query_xyz_path=os.path.abspath(pair['query_xyz_path']),
                    anchor_feat_path=os.path.abspath(pair['anchor_feat_path']),
                    anchor_aux_path=os.path.abspath(pair['anchor_aux_path']),
                    anchor_xyz_path=os.path.abspath(pair['anchor_xyz_path']),
                    teacher_confidence=float(np.clip(pair['teacher_confidence'], 0.0, 1.0)),
                    delta_rvec=[float(x) for x in pair['delta_rvec']],
                    delta_t=[float(x) for x in pair['delta_t']],
                    delta_log_sxyz=[float(x) for x in pair['delta_log_sxyz']],
                    uncert_target_path=os.path.abspath(pair['uncert_target_path']),
                    base_anchor_id=int(pair.get('base_anchor_id', -1)),
                ))
        if not self.records:
            raise RuntimeError('No tracking pairs found')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        q_feat = load_pt(rec.query_feat_path).float()
        q_aux = build_query_aux_tensor(load_npz(rec.query_aux_path)).float()
        q_xyz = load_xyz_npy(rec.query_xyz_path).float()
        a_feat = load_pt(rec.anchor_feat_path).float()
        a_aux = build_anchor_aux_tensor(load_npz(rec.anchor_aux_path)).float()
        a_xyz = load_xyz_npy(rec.anchor_xyz_path).float()
        delta_rvec = torch.tensor(rec.delta_rvec, dtype=torch.float32)
        delta_t = torch.tensor(rec.delta_t, dtype=torch.float32)
        delta_log_sxyz = torch.tensor(rec.delta_log_sxyz, dtype=torch.float32)
        teacher_conf = torch.tensor(rec.teacher_confidence, dtype=torch.float32)
        uncert_target = torch.from_numpy(np.load(rec.uncert_target_path).astype(np.float32)).unsqueeze(0)
        return {
            'query_feat': q_feat,
            'query_aux': q_aux,
            'query_xyz': q_xyz,
            'anchor_feat': a_feat,
            'anchor_aux': a_aux,
            'anchor_xyz': a_xyz,
            'delta_rvec': delta_rvec,
            'delta_t': delta_t,
            'delta_log_sxyz': delta_log_sxyz,
            'teacher_conf': teacher_conf,
            'uncert_target': uncert_target,
            'capture_dir': rec.capture_dir,
            'base_anchor_id': rec.base_anchor_id,
        }


def move_batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def run_epoch(model, loader, optimizer, device, grad_clip, w_conf, w_rot, w_trans, w_scale, w_unc, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    logs = {k: [] for k in ['total', 'conf', 'rot', 'trans', 'scale', 'unc']}
    ctx = torch.enable_grad if train else torch.no_grad
    with ctx():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(
                batch['query_feat'], batch['query_aux'], batch['query_xyz'],
                batch['anchor_feat'], batch['anchor_aux'], batch['anchor_xyz']
            )
            pose_weight = 0.10 + batch['teacher_conf'].clamp(0.0, 1.0)
            loss_conf = confidence_loss(out['score_logit'], batch['teacher_conf'])
            loss_rot = weighted_geodesic_loss(out['delta_rvec'], batch['delta_rvec'], pose_weight)
            loss_trans = weighted_smooth_l1(out['delta_t'], batch['delta_t'], pose_weight)
            loss_scale = weighted_smooth_l1(out['delta_log_sxyz'], batch['delta_log_sxyz'], pose_weight)
            loss_unc = uncertainty_supervision_loss(out['uncert_logit_map'], batch['uncert_target']) + uncertainty_regularization(out['uncert_logit_map'])
            loss = w_conf * loss_conf + w_rot * loss_rot + w_trans * loss_trans + w_scale * loss_scale + w_unc * loss_unc
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            logs['total'].append(float(loss.item()))
            logs['conf'].append(float(loss_conf.item()))
            logs['rot'].append(float(loss_rot.item()))
            logs['trans'].append(float(loss_trans.item()))
            logs['scale'].append(float(loss_scale.item()))
            logs['unc'].append(float(loss_unc.item()))
    return {k: float(np.mean(v)) for k, v in logs.items()}


def split_capture_dirs(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dirs', required=True)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--embed_dim', type=int, default=256)
    ap.add_argument('--num_heads', type=int, default=8)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--rope_base', type=float, default=1000.0)
    ap.add_argument('--w_conf', type=float, default=1.0)
    ap.add_argument('--w_rot', type=float, default=1.0)
    ap.add_argument('--w_trans', type=float, default=1.0)
    ap.add_argument('--w_scale', type=float, default=0.0)
    ap.add_argument('--w_unc', type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    capture_dirs = split_capture_dirs(args.capture_dirs)
    dataset = TrackingPairDataset(capture_dirs)
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * float(args.val_ratio)))) if n_total > 1 else 0
    n_train = n_total - n_val
    if n_val > 0:
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(int(args.seed))
        )
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=str(args.device).startswith('cuda'))
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False,
                                num_workers=int(args.num_workers), pin_memory=str(args.device).startswith('cuda'))

    probe = dataset[0]
    feat_ch = int(probe['query_feat'].shape[0])
    aux_ch = int(probe['query_aux'].shape[0])
    model = CrossAttentionAlignmentModel3DRoPE(
        feat_ch=feat_ch,
        aux_ch=aux_ch,
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        rope_base=float(args.rope_base),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = 1e18
    history = []
    for epoch in range(int(args.epochs)):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device, float(args.grad_clip),
                                  float(args.w_conf), float(args.w_rot), float(args.w_trans), float(args.w_scale), float(args.w_unc), True)
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, optimizer=None, device=args.device, grad_clip=0,
                                    w_conf=float(args.w_conf), w_rot=float(args.w_rot), w_trans=float(args.w_trans),
                                    w_scale=float(args.w_scale), w_unc=float(args.w_unc), train=False)
            score_to_track = val_metrics['total']
        else:
            val_metrics = None
            score_to_track = train_metrics['total']
        history.append({'epoch': int(epoch), 'train': train_metrics, 'val': val_metrics})
        print(f"[Epoch {epoch:03d}] train_total={train_metrics['total']:.6f}", end='')
        if val_metrics is not None:
            print(f"  val_total={val_metrics['total']:.6f}")
        else:
            print()
        ckpt = {
            'model': model.state_dict(),
            'config': {
                'feat_ch': feat_ch,
                'aux_ch': aux_ch,
                'embed_dim': int(args.embed_dim),
                'num_heads': int(args.num_heads),
                'num_layers': int(args.num_layers),
                'rope_base': float(args.rope_base),
            },
        }
        if score_to_track < best_val:
            best_val = score_to_track
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
        save_json(os.path.join(args.save_dir, 'train_history.json'), {
            'capture_dirs': capture_dirs,
            'num_total_samples': int(n_total),
            'num_train': int(n_train),
            'num_val': int(n_val),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'best_val': float(best_val),
            'rope_base': float(args.rope_base),
            'history': history,
        })
    print('\n[OK] delta tracker training finished')
    print('  save_dir:', args.save_dir)
    print('  best.pt :', os.path.join(args.save_dir, 'best.pt'))


if __name__ == '__main__':
    main()
