#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader

from rope3d_alignment_model import CrossAttentionAlignmentModel3DRoPE


def confidence_loss(score_logit: torch.Tensor, teacher_conf: torch.Tensor) -> torch.Tensor:
    target = teacher_conf.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(score_logit[:, 0], target)


def masked_uncertainty_loss(
    uncert_logit_map: torch.Tensor,
    uncert_target: torch.Tensor,
    uncert_valid: torch.Tensor,
) -> torch.Tensor:
    per_sample = F.binary_cross_entropy_with_logits(
        uncert_logit_map,
        uncert_target,
        reduction="none",
    ).mean(dim=(1, 2, 3))
    weight = uncert_valid.float().view(-1)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


def move_batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def decode_pair_sample(sample: Dict[str, bytes]) -> Dict[str, torch.Tensor]:
    with np.load(io.BytesIO(sample["npz"])) as z:
        q_feat = torch.from_numpy(z["q_feat"].astype(np.float32))
        q_aux = torch.from_numpy(z["q_aux"].astype(np.float32))
        q_xyz = torch.from_numpy(z["q_xyz"].astype(np.float32))
        a_feat = torch.from_numpy(z["a_feat"].astype(np.float32))
        a_aux = torch.from_numpy(z["a_aux"].astype(np.float32))
        a_xyz = torch.from_numpy(z["a_xyz"].astype(np.float32))
        delta_bank = torch.from_numpy(z["delta_bank"].astype(np.float32))
        base_pose_bank = torch.from_numpy(z["base_pose_bank"].astype(np.float32))
        teacher_conf_bank = torch.from_numpy(z["teacher_conf_bank"].astype(np.float32))

        if "uncert_bank" in z:
            uncert_bank = torch.from_numpy(z["uncert_bank"].astype(np.float32))
            uncert_valid_bank = torch.ones((uncert_bank.shape[0],), dtype=torch.float32)
        else:
            P = delta_bank.shape[0]
            Hf, Wf = q_feat.shape[-2], q_feat.shape[-1]
            uncert_bank = torch.zeros((P, 1, Hf, Wf), dtype=torch.float32)
            uncert_valid_bank = torch.zeros((P,), dtype=torch.float32)

    meta = json.loads(sample["json"].decode("utf-8"))
    target_idx = torch.tensor(int(meta["target_idx"]), dtype=torch.long)

    return {
        "query_feat": q_feat,
        "query_aux": q_aux,
        "query_xyz": q_xyz,
        "anchor_feat": a_feat,
        "anchor_aux": a_aux,
        "anchor_xyz": a_xyz,
        "delta_bank": delta_bank,
        "base_pose_bank": base_pose_bank,
        "teacher_conf_bank": teacher_conf_bank,
        "uncert_bank": uncert_bank,
        "uncert_valid_bank": uncert_valid_bank,
        "target_idx": target_idx,
        "num_props": int(meta.get("num_props", int(delta_bank.shape[0]))),
        "label": meta.get("label", "unknown"),
        "scene_id": str(meta.get("scene_id", "gt_wds")),
    }


def make_loader(url_pattern: str, batch_size: int, shuffle_buffer: int, num_workers: int,shardshuffle=False,) -> DataLoader:
    dataset = (
        wds.WebDataset(url_pattern, shardshuffle=shardshuffle)
        .shuffle(shuffle_buffer)
        .map(decode_pair_sample)
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def expand_pair_batch(batch: Dict[str, torch.Tensor]):
    B = batch["query_feat"].shape[0]
    P = batch["delta_bank"].shape[1]

    def repeat_flat(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1).expand(-1, P, *([-1] * (x.ndim - 1))).reshape(B * P, *x.shape[1:])

    expanded = {
        "query_feat": repeat_flat(batch["query_feat"]),
        "query_aux": repeat_flat(batch["query_aux"]),
        "query_xyz": repeat_flat(batch["query_xyz"]),
        "anchor_feat": repeat_flat(batch["anchor_feat"]),
        "anchor_aux": repeat_flat(batch["anchor_aux"]),
        "anchor_xyz": repeat_flat(batch["anchor_xyz"]),
        "delta_vec": batch["delta_bank"].reshape(B * P, -1),
        "base_pose_vec": batch["base_pose_bank"].reshape(B * P, -1),
    }
    return expanded, B, P

def train_one_epoch(model, loader, optimizer, device, grad_clip, w_conf):
    model.train()
    seen = 0
    sums = {k: 0.0 for k in ["total", "cls", "top1"]}

    for pair_batch in loader:
        pair_batch = move_batch_to_device(pair_batch, device)
        expanded, B, P = expand_pair_batch(pair_batch)

        out = model(
            expanded["query_feat"], expanded["query_aux"], expanded["query_xyz"],
            expanded["anchor_feat"], expanded["anchor_aux"], expanded["anchor_xyz"],
            expanded["delta_vec"], expanded["base_pose_vec"],
        )

        score_logits = out["score_logit"].view(B, P)
        target_idx = pair_batch["target_idx"]

        # Padding 마스킹
        num_props = pair_batch["num_props"]
        if isinstance(num_props, torch.Tensor):
            n = num_props.to(score_logits.device)
        else:
            n = torch.tensor(num_props, device=score_logits.device)        
        invalid = torch.arange(P, device=score_logits.device)[None, :] >= n[:, None]
        score_logits = score_logits.masked_fill(invalid, -1e9)

        # Soft CE (KL Divergence 꼴) 로직 적용
        teacher = pair_batch["teacher_conf_bank"].to(score_logits.device).float()
        teacher = teacher.masked_fill(invalid, 0.0)

        teacher_sum = teacher.sum(dim=1, keepdim=True)
        # 명시적 디바이스 할당으로 충돌 방지
        fallback = F.one_hot(target_idx.to(score_logits.device), num_classes=P).float().to(score_logits.device)
        teacher = torch.where(
            teacher_sum > 0,
            teacher / teacher_sum.clamp_min(1e-6),
            fallback,
        )

        loss_cls = F.cross_entropy(score_logits, target_idx)
        loss_soft = -(teacher * F.log_softmax(score_logits, dim=1)).sum(dim=1).mean()
        
        # w_conf 파라미터 실제 연결
        loss = loss_cls + w_conf * loss_soft

        optimizer.zero_grad(set_to_none=True)
        
        # Top-1 정확도 계산
        with torch.no_grad():
            pred_idx = torch.argmax(score_logits.detach(), dim=1)
            top1 = (pred_idx == target_idx.to(device)).float().mean()
            sums["top1"] += float(top1.item()) * B
            
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        sums["total"] += float(loss.item()) * B
        sums["cls"] += float(loss_cls.item()) * B
        seen += B

    return {k: sums[k] / max(seen, 1) for k in sums}


@torch.no_grad()
def eval_one_epoch(model, loader, device, w_conf):
    model.eval()
    seen = 0
    sums = {k: 0.0 for k in ["total", "cls", "top1"]}

    for pair_batch in loader:
        pair_batch = move_batch_to_device(pair_batch, device)
        expanded, B, P = expand_pair_batch(pair_batch)

        out = model(
            expanded["query_feat"], expanded["query_aux"], expanded["query_xyz"],
            expanded["anchor_feat"], expanded["anchor_aux"], expanded["anchor_xyz"],
            expanded["delta_vec"], expanded["base_pose_vec"],
        )

        score_logits = out["score_logit"].view(B, P)
        target_idx = pair_batch["target_idx"]

        # Padding 마스킹
        num_props = pair_batch["num_props"]
        if isinstance(num_props, torch.Tensor):
            n = num_props.to(score_logits.device)
        else:
            n = torch.tensor(num_props, device=score_logits.device)        
        invalid = torch.arange(P, device=score_logits.device)[None, :] >= n[:, None]
        score_logits = score_logits.masked_fill(invalid, -1e9)

        # Soft CE (KL Divergence 꼴) 로직 적용
        teacher = pair_batch["teacher_conf_bank"].to(score_logits.device).float()
        teacher = teacher.masked_fill(invalid, 0.0)

        teacher_sum = teacher.sum(dim=1, keepdim=True)
        fallback = F.one_hot(target_idx.to(score_logits.device), num_classes=P).float().to(score_logits.device)
        teacher = torch.where(
            teacher_sum > 0,
            teacher / teacher_sum.clamp_min(1e-6),
            fallback,
        )

        loss_cls = F.cross_entropy(score_logits, target_idx)
        loss_soft = -(teacher * F.log_softmax(score_logits, dim=1)).sum(dim=1).mean()
        loss = loss_cls + w_conf * loss_soft

        pred_idx = torch.argmax(score_logits, dim=1)
        top1 = (pred_idx == target_idx.to(device)).float().mean()

        sums["total"] += float(loss.item()) * B
        sums["cls"] += float(loss_cls.item()) * B
        sums["top1"] += float(top1.item()) * B
        seen += B

    return {k: sums[k] / max(seen, 1) for k in sums}
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds_train", required=True)
    ap.add_argument("--wds_val", default=None)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2, help="pair-level batch size")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--shuffle_buffer", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--rope_base", type=float, default=1000.0)
    ap.add_argument("--w_conf", type=float, default=1.0)
    ap.add_argument("--w_unc", type=float, default=0.5)
    ap.add_argument("--post_self_layers", type=int, default=1)
    ap.add_argument("--post_self_dropout", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_loader = make_loader(args.wds_train, args.batch_size, args.shuffle_buffer, args.num_workers,shardshuffle=100,)
    val_loader = make_loader(args.wds_val, args.batch_size, max(32, args.shuffle_buffer // 4), args.num_workers,shardshuffle=False,) if args.wds_val else None

    probe_pair = next(iter(make_loader(args.wds_train, batch_size=1, shuffle_buffer=4, num_workers=0)))
    feat_ch = int(probe_pair["query_feat"].shape[1])
    aux_ch = int(probe_pair["query_aux"].shape[1])

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
            w_conf=float(args.w_conf), # 인자 추가
        )

        if val_loader is not None:
            val_metrics = eval_one_epoch(
                model=model,
                loader=val_loader,
                device=args.device,
                w_conf=float(args.w_conf), # 인자 추가
            )
            # 최고 Top-1 정확도 저장을 위해 음수로 변환하여 추적
            score_to_track = -val_metrics["top1"]
        else:
            val_metrics = None
            score_to_track = -train_metrics["top1"]

        row = {"epoch": int(epoch), "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(f"[Epoch {epoch:03d}] train_total={train_metrics['total']:.4f} train_cls={train_metrics['cls']:.4f} train_top1={train_metrics['top1']:.4f}")
        if val_metrics is not None:
            print(f"  val_total={val_metrics['total']:.6f} val_cls={val_metrics['cls']:.6f} val_top1={val_metrics['top1']:.4f}")
        else:
            print()

        ckpt = {
            "model": model.state_dict(),
            "config": {
                "feat_ch": feat_ch,
                "aux_ch": aux_ch,
                "embed_dim": int(args.embed_dim),
                "num_heads": int(args.num_heads),
                "num_layers": int(args.num_layers),
                "rope_base": float(args.rope_base),
                "post_self_layers": int(args.post_self_layers),
                "post_self_dropout": float(args.post_self_dropout),
            },
        }

        if score_to_track < best_val:
            best_val = score_to_track
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        with open(os.path.join(args.save_dir, "train_history.json"), "w") as f:
            json.dump(
                {
                    "wds_train": args.wds_train,
                    "wds_val": args.wds_val,
                    "epochs": int(args.epochs),
                    "batch_size_pairs": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "best_val": float(best_val),
                    "history": history,
                },
                f,
                indent=2,
            )

    print("\n[OK] pair-level WDS training finished")
    print("  save_dir:", args.save_dir)
    print("  best.pt :", os.path.join(args.save_dir, "best.pt"))
    print("  last.pt :", os.path.join(args.save_dir, "last.pt"))


if __name__ == "__main__":
    main()
