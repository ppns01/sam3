#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def feather_alpha(mask_u8: np.ndarray, blur_ksize: int = 7, blur_sigma: float = 2.0) -> np.ndarray:
    alpha = (mask_u8.astype(np.float32) * 255.0)
    if blur_ksize > 1:
        alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), blur_sigma)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)
    return alpha


def bbox_from_alpha(alpha: np.ndarray, thr: int = 204) -> Tuple[int, int, int, int]:
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        raise ValueError("alpha foreground is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def square_crop_rgba(
    rgba: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    pad_ratio: float = 0.10,
) -> np.ndarray:
    H, W = rgba.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    size = max(bw, bh)
    size = int(np.ceil(size * (1.0 + pad_ratio)))

    sx0 = int(np.floor(cx - size / 2))
    sy0 = int(np.floor(cy - size / 2))
    sx1 = sx0 + size
    sy1 = sy0 + size

    out = np.zeros((size, size, 4), dtype=np.uint8)

    src_x0 = max(0, sx0)
    src_y0 = max(0, sy0)
    src_x1 = min(W, sx1)
    src_y1 = min(H, sy1)

    dst_x0 = src_x0 - sx0
    dst_y0 = src_y0 - sy0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = rgba[src_y0:src_y1, src_x0:src_x1]
    return out


def resize_max_side(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def rgba_to_black_rgb(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = rgb * alpha
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def save_rgba_png(path: str, rgba: np.ndarray):
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_rgb_png(path: str, rgb: np.ndarray):
    Image.fromarray(rgb, mode="RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--pad_ratio", type=float, default=0.10)
    ap.add_argument("--alpha_blur_ksize", type=int, default=7)
    ap.add_argument("--alpha_blur_sigma", type=float, default=2.0)
    ap.add_argument("--max_side", type=int, default=1024)
    args = ap.parse_args()

    color_path = os.path.join(args.capture_dir, "color.png")
    mask_path = os.path.join(args.capture_dir, "sam3", "mask.png")

    if not os.path.exists(color_path):
        raise FileNotFoundError(color_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    out_dir = os.path.join(args.capture_dir, "trellis2")
    os.makedirs(out_dir, exist_ok=True)

    rgb = load_rgb(color_path)
    mask = load_mask(mask_path)

    alpha = feather_alpha(
        mask,
        blur_ksize=int(args.alpha_blur_ksize),
        blur_sigma=float(args.alpha_blur_sigma),
    )

    rgba = np.dstack([rgb, alpha])
    bbox = bbox_from_alpha(alpha, thr=204)

    rgba_crop = square_crop_rgba(
        rgba,
        bbox_xyxy=bbox,
        pad_ratio=float(args.pad_ratio),
    )

    rgba_ready = resize_max_side(rgba_crop, max_side=int(args.max_side))
    rgb_black = rgba_to_black_rgb(rgba_ready)

    rgba_path = os.path.join(out_dir, "input_rgba.png")
    rgb_black_path = os.path.join(out_dir, "input_rgb_black.png")
    preview_path = os.path.join(out_dir, "preview_alpha_on_white.png")
    meta_path = os.path.join(out_dir, "meta.json")

    save_rgba_png(rgba_path, rgba_ready)
    save_rgb_png(rgb_black_path, rgb_black)

    white = np.ones_like(rgb_black, dtype=np.uint8) * 255
    alpha_f = rgba_ready[:, :, 3:4].astype(np.float32) / 255.0
    preview = (
        rgba_ready[:, :, :3].astype(np.float32) * alpha_f
        + white.astype(np.float32) * (1.0 - alpha_f)
    )
    preview = np.clip(preview, 0, 255).astype(np.uint8)
    save_rgb_png(preview_path, preview)

    meta = {
        "capture_dir": os.path.abspath(args.capture_dir),
        "input_color": os.path.abspath(color_path),
        "input_mask": os.path.abspath(mask_path),
        "bbox_xyxy_alpha204": list(map(int, bbox)),
        "pad_ratio": float(args.pad_ratio),
        "alpha_blur_ksize": int(args.alpha_blur_ksize),
        "alpha_blur_sigma": float(args.alpha_blur_sigma),
        "max_side": int(args.max_side),
        "output_rgba": os.path.abspath(rgba_path),
        "output_rgb_black": os.path.abspath(rgb_black_path),
        "output_preview": os.path.abspath(preview_path),
        "shape_rgba": list(map(int, rgba_ready.shape)),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] TRELLIS.2 input prepared")
    print("  rgba   :", rgba_path)
    print("  rgbblk :", rgb_black_path)
    print("  preview:", preview_path)
    print("  meta   :", meta_path)


if __name__ == "__main__":
    main()