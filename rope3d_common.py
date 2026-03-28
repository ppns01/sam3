#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import cv2
import numpy as np


def _as_hw(hw_like) -> Tuple[int, int]:
    if isinstance(hw_like, dict):
        return int(hw_like['height']), int(hw_like['width'])
    if isinstance(hw_like, (tuple, list)) and len(hw_like) == 2:
        return int(hw_like[0]), int(hw_like[1])
    raise ValueError(f'cannot parse hw from: {hw_like}')


def scale_intrinsics_for_hw(K: Dict[str, float], src_hw, dst_hw) -> Dict[str, float]:
    """
    K의 기준 해상도 src_hw에서 dst_hw로 intrinsics를 선형 스케일링한다.
    """
    src_h, src_w = _as_hw(src_hw)
    dst_h, dst_w = _as_hw(dst_hw)
    sx = float(dst_w) / max(float(src_w), 1e-6)
    sy = float(dst_h) / max(float(src_h), 1e-6)
    return {
        'fx': float(K['fx']) * sx,
        'fy': float(K['fy']) * sy,
        'cx': float(K['cx']) * sx,
        'cy': float(K['cy']) * sy,
        'width': int(dst_w),
        'height': int(dst_h),
    }


def _resize_valid_mask(valid_mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    ph, pw = int(out_hw[0]), int(out_hw[1])
    valid_u8 = (valid_mask.astype(np.uint8) * 255)
    valid_small = cv2.resize(valid_u8, (pw, ph), interpolation=cv2.INTER_NEAREST)
    return valid_small > 127


def build_patch_xyz_map(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    intrinsics_crop: Dict[str, float],
    out_hw: Tuple[int, int],
    center_xyz_m: Iterable[float],
    scale_ref_m: float,
    clip_value: float = 4.0,
) -> np.ndarray:
    """
    metric depth + valid mask + crop intrinsics에서 patch-grid용 local XYZ map 생성.

    반환 shape: [ph, pw, 3], float32
    값 정의:
      xyz_local = (xyz_cam - center_xyz_m) / scale_ref_m
    """
    if depth_m.ndim != 2:
        raise ValueError(f'depth_m must be [H,W], got {depth_m.shape}')
    if valid_mask.shape != depth_m.shape:
        raise ValueError(f'valid_mask shape mismatch: {valid_mask.shape} vs {depth_m.shape}')

    src_hw = (int(depth_m.shape[0]), int(depth_m.shape[1]))
    ph, pw = int(out_hw[0]), int(out_hw[1])

    base_hw = (
        int(intrinsics_crop.get('height', src_hw[0])),
        int(intrinsics_crop.get('width', src_hw[1])),
    )
    K_depth = scale_intrinsics_for_hw(intrinsics_crop, base_hw, src_hw)
    K_small = scale_intrinsics_for_hw(K_depth, src_hw, (ph, pw))

    # [핵심 수정 1] Linear 보간에 의한 Ghost Depth 방지를 위해 무조건 NEAREST 적용
    depth_small = cv2.resize(depth_m.astype(np.float32), (pw, ph), interpolation=cv2.INTER_NEAREST)
    valid_small = _resize_valid_mask(valid_mask.astype(bool), (ph, pw))
    
    # [핵심 수정 2] Mask-aware downsampling: 유효하지 않은 픽셀의 깊이는 확실하게 0으로 날림
    depth_small[~valid_small] = 0.0

    ys, xs = np.meshgrid(
        np.arange(ph, dtype=np.float32),
        np.arange(pw, dtype=np.float32),
        indexing='ij',
    )

    z = depth_small.astype(np.float32)
    fx = max(float(K_small['fx']), 1e-6)
    fy = max(float(K_small['fy']), 1e-6)
    cx = float(K_small['cx'])
    cy = float(K_small['cy'])

    x = ((xs + 0.5) - cx) / fx * z
    y = ((ys + 0.5) - cy) / fy * z
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    finite = np.isfinite(z) & (z > 0)
    valid = valid_small & finite
    xyz[~valid] = 0.0

    center = np.asarray(list(center_xyz_m), dtype=np.float32).reshape(1, 1, 3)
    scale_ref = max(float(scale_ref_m), 1e-6)
    xyz = (xyz - center) / scale_ref
    xyz[~valid] = 0.0
    xyz = np.clip(xyz, -float(clip_value), float(clip_value)).astype(np.float32)
    return xyz
