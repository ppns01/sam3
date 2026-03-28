#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image as PILImage
import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _to_prob(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mn = float(arr.min())
    mx = float(arr.max())
    if mn < -0.5 or mx > 1.5:
        arr = _sigmoid(arr)
    return np.clip(arr, 0.0, 1.0)


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _overlay_boundary(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m8 = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.morphologyEx(m8, cv2.MORPH_GRADIENT, k)
    out = bgr.copy()
    out[boundary > 0] = (0, 0, 255)
    return out


def _collect_mask_candidates(state: Any, H: int, W: int, max_nodes: int = 60000) -> List[np.ndarray]:
    cand: List[np.ndarray] = []
    visited = set()
    n_nodes = 0

    def _push_2d(a2: np.ndarray):
        if a2.ndim != 2:
            return
        h, w = int(a2.shape[0]), int(a2.shape[1])
        if h < 16 or w < 16:
            return
        if a2.size > (H * W * 64):
            return
        cand.append(a2)

    def _maybe_is_color_image(a3: np.ndarray) -> bool:
        if a3.ndim != 3:
            return False
        h, w, c = a3.shape
        return (c in (3, 4)) and (a3.dtype == np.uint8 or a3.dtype == np.uint16)

    def _push_any(a: np.ndarray):
        if not isinstance(a, np.ndarray):
            try:
                a = np.asarray(a)
            except Exception:
                return

        if a.ndim == 2:
            _push_2d(a)
            return

        if a.ndim == 3:
            if _maybe_is_color_image(a):
                return
            if a.shape[0] <= 64 and a.shape[1] >= 16 and a.shape[2] >= 16:
                for i in range(int(a.shape[0])):
                    _push_2d(a[i])
                return
            if a.shape[2] <= 64 and a.shape[0] >= 16 and a.shape[1] >= 16:
                for i in range(int(a.shape[2])):
                    _push_2d(a[:, :, i])
                return
            return

        if a.ndim == 4:
            if a.shape[0] <= 8 and a.shape[1] <= 64 and a.shape[2] >= 16 and a.shape[3] >= 16:
                for b in range(int(a.shape[0])):
                    for i in range(int(a.shape[1])):
                        _push_2d(a[b, i])
                return
            if a.shape[0] <= 8 and a.shape[3] <= 64 and a.shape[1] >= 16 and a.shape[2] >= 16:
                for b in range(int(a.shape[0])):
                    for i in range(int(a.shape[3])):
                        _push_2d(a[b, :, :, i])
                return
            return

    def visit(obj: Any, depth: int = 0):
        nonlocal n_nodes
        if obj is None:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)
        n_nodes += 1
        if n_nodes > max_nodes:
            return

        try:
            if isinstance(obj, torch.Tensor):
                a = obj.detach().to("cpu")
                if a.numel() <= H * W * 64:
                    _push_any(a.float().numpy())
                return
            if isinstance(obj, np.ndarray):
                if obj.size <= H * W * 64:
                    _push_any(obj)
                return
        except Exception:
            pass

        if isinstance(obj, dict):
            for v in obj.values():
                visit(v, depth + 1)
            return

        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                visit(v, depth + 1)
            return

        if depth < 8:
            try:
                if hasattr(obj, "__dict__"):
                    for k, v in vars(obj).items():
                        if str(k).startswith("_"):
                            continue
                        visit(v, depth + 1)
                    return
            except Exception:
                pass

        if depth < 6:
            try:
                keys = [k for k in dir(obj) if not str(k).startswith("_")]
                for k in keys[:60]:
                    try:
                        v = getattr(obj, k)
                    except Exception:
                        continue
                    if callable(v):
                        continue
                    visit(v, depth + 1)
            except Exception:
                pass

    visit(state, 0)
    return cand


def run_sam3_and_save_masks(
    rgb: np.ndarray,
    prompt: str,
    sam3_root: str,
    out_dir: str,
    confidence: float = 0.3,
    device: str = "cuda",
    thresh: float = 0.5,
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.95,
    choose: str = "largest",
    index: int = 0,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    img_pil = PILImage.fromarray(rgb).convert("RGB")
    H, W = rgb.shape[:2]
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

    print("[SAM3] Building model...")
    model = build_sam3_image_model(bpe_path=bpe_path)
    model.to(device)
    model.eval()

    processor = Sam3Processor(model, confidence_threshold=float(confidence))

    print("[SAM3] Inference...")
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                state = processor.set_image(img_pil)
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(state=state, prompt=prompt)
        else:
            state = processor.set_image(img_pil)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)

    cands = _collect_mask_candidates(state, H=H, W=W)
    if len(cands) == 0:
        raise RuntimeError("SAM3 내부에서 mask 후보를 찾지 못했어.")

    print(f"[SAM3] Raw candidates: {len(cands)}")

    masks: List[np.ndarray] = []
    infos: List[Dict[str, Any]] = []

    min_area = int(min_area_ratio * H * W)
    max_area = int(max_area_ratio * H * W)
    seen = set()

    for a in cands:
        p = _to_prob(a)

        if p.shape != (H, W):
            try:
                p = cv2.resize(p.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            except Exception:
                continue

        std = float(p.std())
        if std < 1e-6:
            continue

        binaryness = float(((p < 0.1) | (p > 0.9)).mean())
        if binaryness < 0.15:
            continue

        m = (p >= float(thresh))
        area = int(m.sum())
        if area < min_area or area > max_area:
            continue

        hkey = hash((m.tobytes(), m.shape))
        if hkey in seen:
            continue
        seen.add(hkey)

        x0, y0, x1, y1 = _bbox_from_mask(m)
        masks.append(m)
        infos.append({
            "area": area,
            "area_ratio": float(area) / float(H * W),
            "binaryness": float(binaryness),
            "std": float(std),
            "bbox_xyxy": [x0, y0, x1, y1],
        })

    print(f"[SAM3] Kept masks after filtering: {len(masks)}")
    if len(masks) == 0:
        raise RuntimeError("mask 후보는 있었는데 전부 필터링됐어. thresh 나 min_area_ratio를 낮춰봐.")

    if choose == "largest":
        sel = int(np.argmax([inf["area"] for inf in infos]))
    elif choose == "center":
        cx_img, cy_img = W * 0.5, H * 0.5
        d2 = []
        for inf in infos:
            x0, y0, x1, y1 = inf["bbox_xyxy"]
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
            d2.append((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
        sel = int(np.argmin(d2))
    elif choose == "index":
        sel = int(np.clip(index, 0, len(masks) - 1))
    else:
        sel = int(np.argmax([inf["area"] for inf in infos]))

    meta = {
        "prompt": prompt,
        "H": H,
        "W": W,
        "thresh": float(thresh),
        "choose": choose,
        "selected_index": sel,
        "masks": infos,
    }
    with open(os.path.join(out_dir, "mask_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for i, m in enumerate(masks):
        cv2.imwrite(os.path.join(out_dir, f"mask_{i}.png"), (m.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(out_dir, f"overlay_{i}.png"), _overlay_boundary(rgb, m))

    sel_mask = masks[sel]
    mask_path = os.path.join(out_dir, "mask.png")
    cv2.imwrite(mask_path, (sel_mask.astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), _overlay_boundary(rgb, sel_mask))

    print("\n[SAM3] Saved:", os.path.abspath(out_dir))
    for i, inf in enumerate(infos):
        tag = "<-- SELECTED" if i == sel else ""
        ar = inf["area_ratio"] * 100.0
        print(f"  [{i}] area={inf['area']} ({ar:.2f}%) bbox={inf['bbox_xyxy']} {tag}")

    return mask_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True, help="step1 결과 폴더")
    ap.add_argument("--prompt", required=True, help='예: "bottle"')
    ap.add_argument("--sam3_root", default="/home/park/sam3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--min_area_ratio", type=float, default=0.0005)
    ap.add_argument("--max_area_ratio", type=float, default=0.95)
    ap.add_argument("--choose", choices=["largest", "center", "index"], default="largest")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    color_path = os.path.join(args.capture_dir, "color.png")
    if not os.path.exists(color_path):
        raise FileNotFoundError(f"color.png not found: {color_path}")

    out_dir = os.path.join(args.capture_dir, "sam3")
    rgb = np.array(PILImage.open(color_path).convert("RGB"))

    run_sam3_and_save_masks(
        rgb=rgb,
        prompt=args.prompt,
        sam3_root=args.sam3_root,
        out_dir=out_dir,
        confidence=args.confidence,
        device=args.device,
        thresh=args.thresh,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        choose=args.choose,
        index=args.index,
    )


if __name__ == "__main__":
    main()
