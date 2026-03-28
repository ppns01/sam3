#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List,Optional

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

def rotmat_to_r6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)

def build_base_pose_vec(base_pose: dict) -> List[float]:
    R = np.asarray(base_pose['R_3x3'], dtype=np.float32).reshape(3, 3)
    t = np.asarray(base_pose['t_xyz_m'], dtype=np.float32).reshape(3)
    s = np.asarray(base_pose['sx_sy_sz'], dtype=np.float32).reshape(3)
    log_s = np.log(np.clip(s, 1e-8, None))
    return np.concatenate([rotmat_to_r6d(R), t, log_s], axis=0).astype(np.float32).tolist()

def build_delta_vec(proposal_delta: dict) -> List[float]:
    delta_deg = np.asarray(proposal_delta['delta_deg_xyz'], dtype=np.float32).reshape(3)
    delta_rad = np.deg2rad(delta_deg)
    delta_t = np.asarray(proposal_delta['delta_t_xyz'], dtype=np.float32).reshape(3)
    delta_log_s = np.asarray(proposal_delta['delta_log_sxyz'], dtype=np.float32).reshape(3)
    return np.concatenate([delta_rad, delta_t, delta_log_s], axis=0).astype(np.float32).tolist()
def axis_angle_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
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
    delta_vec: List[float]
    base_pose_vec: List[float]
    uncert_target_path: Optional[str]
    base_anchor_id: int


class AlignmentTeacherDataset3DRoPE(Dataset):
    def __init__(self, capture_dirs: List[str]):
        self.records: List[PairRecord] = []

        for capture_dir in capture_dirs:
            feature_pack_dir = os.path.join(capture_dir, 'feature_pack')
            refined_dir = os.path.join(capture_dir, 'refined_feature_pack')
            teacher_dir = os.path.join(capture_dir, 'teacher_refine_v4')

            teacher_proposals_path = os.path.join(teacher_dir, 'teacher_proposals.json')
            refined_summary_path = os.path.join(refined_dir, 'summary.json')
            feature_summary_path = os.path.join(feature_pack_dir, 'summary.json')
            query_xyz_path = os.path.join(feature_pack_dir, 'query_xyz_map.npy')

            for p in [teacher_proposals_path, refined_summary_path, feature_summary_path, query_xyz_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(p)

            teacher_proposals = load_json(teacher_proposals_path)['proposals']
            refined_summary = load_json(refined_summary_path)
            feature_summary = load_json(feature_summary_path)

            anchor_meta_map = {int(rec['anchor_id']): rec for rec in refined_summary['anchors']}
            feature_anchor_map = {int(rec['anchor_id']): rec for rec in feature_summary['anchors']}

            query_feat_path = os.path.join(refined_dir, 'query_refined_feat.pt')
            query_aux_path = os.path.join(feature_pack_dir, 'query_aux_maps.npz')

            for proposal in teacher_proposals:
                base_anchor_id = int(proposal['base_anchor_id'])
                if base_anchor_id not in anchor_meta_map or base_anchor_id not in feature_anchor_map:
                    continue

                anchor_meta = anchor_meta_map[base_anchor_id]
                feature_meta = feature_anchor_map[base_anchor_id]
                tgt = proposal['teacher_target_for_step10']

                anchor_feat_path = anchor_meta['refined_feat_path']
                anchor_aux_path = anchor_meta['input_aux_path']
                anchor_xyz_path = feature_meta.get(
                    'xyz_path',
                    os.path.join(feature_pack_dir, f'cand_{base_anchor_id:04d}_xyz_map.npy')
                )

                uncert_target_path = tgt.get('uncert_target_npy')
                if uncert_target_path is not None:
                    uncert_target_path = os.path.abspath(uncert_target_path)

                rec = PairRecord(
                    capture_dir=os.path.abspath(capture_dir),
                    query_feat_path=os.path.abspath(query_feat_path),
                    query_aux_path=os.path.abspath(query_aux_path),
                    query_xyz_path=os.path.abspath(query_xyz_path),
                    anchor_feat_path=os.path.abspath(anchor_feat_path),
                    anchor_aux_path=os.path.abspath(anchor_aux_path),
                    anchor_xyz_path=os.path.abspath(anchor_xyz_path),
                    teacher_confidence=float(np.clip(tgt['teacher_confidence'], 0.0, 1.0)),
                    delta_vec=build_delta_vec(proposal['proposal_delta']),
                    base_pose_vec=build_base_pose_vec(proposal['base_pose']),
                    uncert_target_path=uncert_target_path,
                    base_anchor_id=base_anchor_id,
                )
                self.records.append(rec)

        if len(self.records) == 0:
            raise RuntimeError('No teacher proposals found.')

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

        delta_vec = torch.tensor(rec.delta_vec, dtype=torch.float32)
        base_pose_vec = torch.tensor(rec.base_pose_vec, dtype=torch.float32)
        teacher_conf = torch.tensor(rec.teacher_confidence, dtype=torch.float32)

        if rec.uncert_target_path is not None and os.path.exists(rec.uncert_target_path):
            uncert_target = torch.from_numpy(
                np.load(rec.uncert_target_path).astype(np.float32)
            ).unsqueeze(0)
            uncert_valid = torch.tensor(1.0, dtype=torch.float32)
        else:
            Hf, Wf = q_feat.shape[-2], q_feat.shape[-1]
            uncert_target = torch.zeros((1, Hf, Wf), dtype=torch.float32)
            uncert_valid = torch.tensor(0.0, dtype=torch.float32)

        return {
            'query_feat': q_feat,
            'query_aux': q_aux,
            'query_xyz': q_xyz,
            'anchor_feat': a_feat,
            'anchor_aux': a_aux,
            'anchor_xyz': a_xyz,
            'delta_vec': delta_vec,
            'base_pose_vec': base_pose_vec,
            'teacher_conf': teacher_conf,
            'uncert_target': uncert_target,
            'uncert_valid': uncert_valid,
            'base_anchor_id': rec.base_anchor_id,
            'capture_dir': rec.capture_dir,
        }



def confidence_loss(score_logit: torch.Tensor, teacher_conf: torch.Tensor) -> torch.Tensor:
    target = teacher_conf.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(score_logit[:, 0], target)


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction='none').mean(dim=1)
    return (loss * weight).mean()


def masked_uncertainty_loss(
    uncert_logit_map: torch.Tensor,
    uncert_target: torch.Tensor,
    uncert_valid: torch.Tensor,
) -> torch.Tensor:
    per_sample = F.binary_cross_entropy_with_logits(
        uncert_logit_map,
        uncert_target,
        reduction='none',
    ).mean(dim=(1, 2, 3))
    weight = uncert_valid.float().view(-1)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


def uncertainty_regularization(uncert_logit_map: torch.Tensor) -> torch.Tensor:
    return 1e-3 * (uncert_logit_map ** 2).mean()


def move_batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def train_one_epoch(model, loader, optimizer, device, grad_clip, w_conf, w_unc):
    model.train()
    loss_sums = {k: 0.0 for k in ['total', 'conf', 'unc']}
    num_seen = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            batch['query_feat'],
            batch['query_aux'],
            batch['query_xyz'],
            batch['anchor_feat'],
            batch['anchor_aux'],
            batch['anchor_xyz'],
            batch['delta_vec'],
            batch['base_pose_vec'],
        )

        loss_conf = confidence_loss(out['score_logit'], batch['teacher_conf'])
        loss_unc = masked_uncertainty_loss(
            out['uncert_logit_map'],
            batch['uncert_target'],
            batch['uncert_valid'],
        )

        loss = w_conf * loss_conf + w_unc * loss_unc
        batch_size = int(batch['query_feat'].shape[0])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sums['total'] += float(loss.item()) * batch_size
        loss_sums['conf'] += float(loss_conf.item()) * batch_size
        loss_sums['unc'] += float(loss_unc.item()) * batch_size
        num_seen += batch_size

    return {k: loss_sums[k] / max(num_seen, 1) for k in loss_sums}



@torch.no_grad()
def eval_one_epoch(model, loader, device, w_conf, w_unc):
    model.eval()
    loss_sums = {k: 0.0 for k in ['total', 'conf', 'unc']}
    num_seen = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(
            batch['query_feat'],
            batch['query_aux'],
            batch['query_xyz'],
            batch['anchor_feat'],
            batch['anchor_aux'],
            batch['anchor_xyz'],
            batch['delta_vec'],
            batch['base_pose_vec'],
        )

        loss_conf = confidence_loss(out['score_logit'], batch['teacher_conf'])
        loss_unc = masked_uncertainty_loss(
            out['uncert_logit_map'],
            batch['uncert_target'],
            batch['uncert_valid'],
        )

        loss = w_conf * loss_conf + w_unc * loss_unc
        batch_size = int(batch['query_feat'].shape[0])

        loss_sums['total'] += float(loss.item()) * batch_size
        loss_sums['conf'] += float(loss_conf.item()) * batch_size
        loss_sums['unc'] += float(loss_unc.item()) * batch_size
        num_seen += batch_size

    return {k: loss_sums[k] / max(num_seen, 1) for k in loss_sums}





def split_capture_dirs(capture_dirs_str: str) -> List[str]:
    return [x.strip() for x in capture_dirs_str.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture_dirs', required=True, help='쉼표로 구분된 capture_dir 목록')
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=2)
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
    ap.add_argument('--w_trans', type=float, default=0.5)
    ap.add_argument('--w_scale', type=float, default=1.0)
    ap.add_argument('--w_unc', type=float, default=0.5)
    ap.add_argument('--post_self_layers', type=int, default=1)
    ap.add_argument('--post_self_dropout', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    capture_dirs = split_capture_dirs(args.capture_dirs)
    dataset = AlignmentTeacherDataset3DRoPE(capture_dirs)

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * float(args.val_ratio)))) if n_total > 1 else 0
    n_train = n_total - n_val

    if n_val > 0:
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(int(args.seed)),
        )
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(str(args.device).startswith('cuda')),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(str(args.device).startswith('cuda')),
        )

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
        post_self_layers=int(args.post_self_layers),
        post_self_dropout=float(args.post_self_dropout),
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = 1e18
    history = []
    for epoch in range(int(args.epochs)):
        train_metrics = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=args.device,
        grad_clip=float(args.grad_clip),
        w_conf=float(args.w_conf),
        w_unc=float(args.w_unc),
            )

        if val_loader is not None:
            val_metrics = eval_one_epoch(
            model=model,
            loader=val_loader,
            device=args.device,
            w_conf=float(args.w_conf),
            w_unc=float(args.w_unc),
            )
            score_to_track = val_metrics['total']
        else:
            val_metrics = None
            score_to_track = train_metrics['total']

        row = {'epoch': int(epoch), 'train': train_metrics, 'val': val_metrics}
        history.append(row)

        print(f"[Epoch {epoch:03d}] train_total={train_metrics['total']:.6f}", end='')
        if val_metrics is not None:
            print(f"  val_total={val_metrics['total']:.6f}")
        else:
            print()

        if score_to_track < best_val:
            best_val = score_to_track
            torch.save(
                {
                    'model': model.state_dict(),
                    'config': {
                        'feat_ch': feat_ch,
                        'aux_ch': aux_ch,
                        'embed_dim': int(args.embed_dim),
                        'num_heads': int(args.num_heads),
                        'num_layers': int(args.num_layers),
                        'rope_base': float(args.rope_base),
                        'post_self_layers': int(args.post_self_layers),
                        'post_self_dropout': float(args.post_self_dropout),
                    },
                },
                os.path.join(args.save_dir, 'best.pt'),
            )

        torch.save(
            {
                'model': model.state_dict(),
                'config': {
                    'feat_ch': feat_ch,
                    'aux_ch': aux_ch,
                    'embed_dim': int(args.embed_dim),
                    'num_heads': int(args.num_heads),
                    'num_layers': int(args.num_layers),
                    'rope_base': float(args.rope_base),
                    'post_self_layers': int(args.post_self_layers),
                    'post_self_dropout': float(args.post_self_dropout),
                },
            },
            os.path.join(args.save_dir, 'last.pt'),
        )

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
            'post_self_layers': int(args.post_self_layers),
            'post_self_dropout': float(args.post_self_dropout),
        })

    print('\n[OK] 3D RoPE training finished')
    print('  save_dir:', args.save_dir)
    print('  best.pt :', os.path.join(args.save_dir, 'best.pt'))
    print('  last.pt :', os.path.join(args.save_dir, 'last.pt'))
    print('  history :', os.path.join(args.save_dir, 'train_history.json'))


if __name__ == '__main__':
    main()