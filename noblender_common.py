from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def asdict_intrinsics(K: Intrinsics) -> dict:
    return asdict(K)


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_intrinsics_json_from_dict(d: dict) -> Intrinsics:
    return Intrinsics(
        fx=float(d['fx']),
        fy=float(d['fy']),
        cx=float(d['cx']),
        cy=float(d['cy']),
        width=int(d['width']),
        height=int(d['height']),
    )


def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def load_alpha(path: str) -> np.ndarray:
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(path)
    return a.astype(np.uint8)


def load_color_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_color_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path: str) -> np.ndarray:
    d = np.load(path).astype(np.float32)
    if d.ndim != 2:
        raise ValueError(f'depth must be HxW, got {d.shape}')
    d[~np.isfinite(d)] = 0.0
    d[d < 0.0] = 0.0
    return d


def save_depth(path: str, depth: np.ndarray):
    np.save(path, depth.astype(np.float32))


def parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for x in s.split(','):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n


def Rx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def Ry(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def Rz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def look_at_rotation_from_direction(view_dir_world, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float64)) -> np.ndarray:
    z = normalize(np.asarray(view_dir_world, dtype=np.float64))
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-8:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = np.cross(up_hint, z)
    x = normalize(x)
    y = normalize(np.cross(z, x))
    return np.stack([x, y, z], axis=1)


def rotation_matrix_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec[:, 0]


def rvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def relative_rvec(R_base: np.ndarray, R_new: np.ndarray) -> np.ndarray:
    R_rel = np.asarray(R_new, dtype=np.float64) @ np.asarray(R_base, dtype=np.float64).T
    return rotation_matrix_to_rvec(R_rel)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou_xyxy(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    inter = float(iw * ih)
    area_a = float(max(0, ax1 - ax0 + 1) * max(0, ay1 - ay0 + 1))
    area_b = float(max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1))
    union = area_a + area_b - inter
    if union < 1e-8:
        return 0.0
    return inter / union


def mask_hw(mask: np.ndarray) -> Tuple[int, int]:
    x0, y0, x1, y1 = bbox_from_mask(mask)
    return max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union < 1e-8:
        return 0.0
    return inter / union


def contour_map(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    c = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return c > 0


def overlay_boundaries(color_bgr: np.ndarray, mask_obs: np.ndarray, mask_r: np.ndarray) -> np.ndarray:
    vis = color_bgr.copy()
    vis[contour_map(mask_obs)] = (0, 0, 255)
    vis[contour_map(mask_r)] = (255, 0, 0)
    return vis


def resize_mask(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def resize_color_bgr(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_color_rgb(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    h, w = out_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def scale_intrinsics(K: Intrinsics, scale: float) -> Intrinsics:
    return Intrinsics(
        fx=float(K.fx * scale),
        fy=float(K.fy * scale),
        cx=float(K.cx * scale),
        cy=float(K.cy * scale),
        width=int(round(K.width * scale)),
        height=int(round(K.height * scale)),
    )


def keep_best_component(mask_raw: np.ndarray, query_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    rm = (mask_raw.astype(np.uint8) > 0).astype(np.uint8)
    qm = (query_mask.astype(np.uint8) > 0).astype(np.uint8)
    if rm.sum() == 0:
        return mask_raw.copy()
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(rm, connectivity=8)
    if n <= 1:
        return rm.astype(bool)

    qbbox = bbox_from_mask(qm > 0)
    qarea = float(qm.sum())
    ys, xs = np.where(qm > 0)
    if xs.size > 0:
        qcx = float(xs.mean())
        qcy = float(ys.mean())
    else:
        h, w = qm.shape
        qcx = w * 0.5
        qcy = h * 0.5

    best_id = -1
    best_score = -1e18
    for cid in range(1, n):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == cid
        cbbox = bbox_from_mask(comp)
        biou = bbox_iou_xyxy(cbbox, qbbox)
        inter = float(np.logical_and(comp, qm > 0).sum())
        overlap_ratio = inter / max(float(area), 1e-8)
        cx, cy = centroids[cid]
        dist = float(np.hypot(cx - qcx, cy - qcy))
        area_ratio = float(area) / max(qarea, 1e-8)
        area_penalty = abs(math.log(max(area_ratio, 1e-8)))
        score = 4.0 * overlap_ratio + 2.0 * biou - 0.001 * dist - 0.2 * area_penalty
        if score > best_score:
            best_score = score
            best_id = cid
    if best_id < 0:
        best_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best_id


def normalize01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def sobel_edge_map(gray_or_depth: np.ndarray) -> np.ndarray:
    x = gray_or_depth.astype(np.float32)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return normalize01(mag)


def mask_boundary_map(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    b = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return (b > 0).astype(np.float32)


def spherical_fibonacci_dirs(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError('n must be positive')
    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    z = 1.0 - 2.0 * (i + 0.5) / float(n)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = 2.0 * math.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)
    return np.asarray([normalize(v) for v in dirs], dtype=np.float64)


def build_viewpoint_candidates(n_view_dirs: int, roll_list_deg: Sequence[float], round_decimals: int = 6) -> List[dict]:
    dirs = spherical_fibonacci_dirs(int(n_view_dirs))
    candidates: List[dict] = []
    seen = set()
    idx = 0
    for d in dirs:
        R_look = look_at_rotation_from_direction(d)
        for roll_deg in roll_list_deg:
            R = Rz(math.radians(float(roll_deg))) @ R_look
            key = tuple(np.round(R.reshape(-1), round_decimals).tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                'id': idx,
                'stem': f'cand_{idx:04d}',
                'R': R.tolist(),
                'view_dir': d.tolist(),
                'roll_deg': float(roll_deg),
            })
            idx += 1
    return candidates


def compose_affine_from_pose(
    R_3x3: np.ndarray,
    t_xyz_m: np.ndarray,
    sxyz: np.ndarray,
    mesh_pca_basis_3x3: np.ndarray,
    mesh_bbox_center_m: np.ndarray,
    unit_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(R_3x3, dtype=np.float64)
    t = np.asarray(t_xyz_m, dtype=np.float64).reshape(3)
    sxyz = np.asarray(sxyz, dtype=np.float64).reshape(3)
    U = np.asarray(mesh_pca_basis_3x3, dtype=np.float64)
    center = np.asarray(mesh_bbox_center_m, dtype=np.float64).reshape(3)

    D = np.diag(sxyz)
    S = U @ D @ U.T
    A = (R @ S) * float(unit_scale)
    b = t - (R @ S @ center)
    return A, b, S


def decompose_mask_metrics(query_mask: np.ndarray, render_mask: np.ndarray) -> dict:
    qbbox = bbox_from_mask(query_mask)
    rbbox = bbox_from_mask(render_mask)
    qarea = float(query_mask.sum())
    rarea = float(render_mask.sum())
    qw, qh = mask_hw(query_mask)
    rw, rh = mask_hw(render_mask)
    aspect_q = float(qh) / max(float(qw), 1e-8)
    aspect_r = float(rh) / max(float(rw), 1e-8)
    area_ratio = rarea / max(qarea, 1e-8)

    qys, qxs = np.where(query_mask)
    rys, rxs = np.where(render_mask)
    if qxs.size > 0:
        qcx = float(qxs.mean())
        qcy = float(qys.mean())
    else:
        h, w = query_mask.shape
        qcx, qcy = w * 0.5, h * 0.5
    if rxs.size > 0:
        rcx = float(rxs.mean())
        rcy = float(rys.mean())
    else:
        h, w = render_mask.shape
        rcx, rcy = w * 0.5, h * 0.5

    return {
        'iou': float(iou(query_mask, render_mask)),
        'query_bbox': list(qbbox),
        'render_bbox': list(rbbox),
        'bbox_iou': float(bbox_iou_xyxy(qbbox, rbbox)),
        'query_area': qarea,
        'render_area': rarea,
        'area_ratio': float(area_ratio),
        'aspect_query': float(aspect_q),
        'aspect_render': float(aspect_r),
        'center_query_xy': [qcx, qcy],
        'center_render_xy': [rcx, rcy],
        'center_dist_px': float(np.hypot(qcx - rcx, qcy - rcy)),
    }


def gaussianized_scalar_error(delta: float, sigma: float, square: bool = True) -> float:
    sigma = max(float(sigma), 1e-8)
    x = float(delta)
    if square:
        val = x * x
    else:
        val = abs(x)
    return float(1.0 - math.exp(-val / (2.0 * sigma * sigma)))


def gaussian_blur_mask(mask: np.ndarray, ksize: int = 15, sigma: float = 0.0) -> np.ndarray:
    x = mask.astype(np.float32)
    return cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def compute_gaussian_coarse_loss(
    query_mask: np.ndarray,
    render_mask: np.ndarray,
    sigma_log_area: float = 0.30,
    sigma_center_px: float = 24.0,
    sigma_log_bbox: float = 0.25,
    blur_ksize: int = 15,
) -> Tuple[float, dict]:
    m = decompose_mask_metrics(query_mask, render_mask)

    qbbox = m['query_bbox']
    rbbox = m['render_bbox']
    q_w = max(1.0, float(qbbox[2] - qbbox[0] + 1))
    q_h = max(1.0, float(qbbox[3] - qbbox[1] + 1))
    r_w = max(1.0, float(rbbox[2] - rbbox[0] + 1))
    r_h = max(1.0, float(rbbox[3] - rbbox[1] + 1))

    d_log_area = math.log(max(m['area_ratio'], 1e-8))
    d_log_w = math.log(max(r_w / max(q_w, 1e-8), 1e-8))
    d_log_h = math.log(max(r_h / max(q_h, 1e-8), 1e-8))

    loss_area = gaussianized_scalar_error(d_log_area, sigma_log_area)
    loss_ctr = gaussianized_scalar_error(m['center_dist_px'], sigma_center_px)
    loss_bbox = 0.5 * (
        gaussianized_scalar_error(d_log_w, sigma_log_bbox) +
        gaussianized_scalar_error(d_log_h, sigma_log_bbox)
    )

    q_blur = gaussian_blur_mask(query_mask, ksize=blur_ksize)
    r_blur = gaussian_blur_mask(render_mask, ksize=blur_ksize)
    loss_blur = float(np.mean(np.abs(q_blur - r_blur)))
    q_dilate = cv2.dilate(query_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    extra_mask = render_mask & (~q_dilate)
    extra_ratio = float(extra_mask.sum()) / max(float(m['query_area']), 1.0)

    total = (
        0.35 * loss_area +
        0.25 * loss_ctr +
        0.20 * loss_bbox +
        0.20 * loss_blur +
        0.25 * extra_ratio
    )    
    detail = {
        **m,
        'loss_area': float(loss_area),
        'loss_center': float(loss_ctr),
        'loss_bbox': float(loss_bbox),
        'loss_blur': float(loss_blur),
        'loss_total': float(total),
        'extra_ratio': float(extra_ratio),

    }
    return float(total), detail


def compute_depth_agreement_score(query_depth: np.ndarray, render_depth: np.ndarray, query_mask: np.ndarray, render_mask: np.ndarray, sigma_depth_m: float = 0.02) -> float:
    valid = query_mask & render_mask & np.isfinite(query_depth) & np.isfinite(render_depth) & (query_depth > 0.0) & (render_depth > 0.0)
    n = int(valid.sum())
    if n < 20:
        return 0.0
    mse = float(np.mean((query_depth[valid] - render_depth[valid]) ** 2))
    sigma2 = max(float(sigma_depth_m), 1e-6) ** 2
    return float(math.exp(-mse / (2.0 * sigma2)))


def build_uncert_target_map(query_mask: np.ndarray, render_mask: np.ndarray, sim_map: np.ndarray) -> np.ndarray:
    ph, pw = sim_map.shape
    q_small = resize_mask(query_mask, (ph, pw)).astype(np.float32)
    r_small = resize_mask(render_mask, (ph, pw)).astype(np.float32)
    overlap = q_small * r_small
    miss = np.clip(q_small - r_small, 0.0, 1.0)
    extra = np.clip(r_small - q_small, 0.0, 1.0)
    sim_norm = np.clip((sim_map.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    geom_uncert = np.maximum(miss, extra)
    feat_uncert = overlap * (1.0 - sim_norm)
    return np.clip(0.65 * geom_uncert + 0.35 * feat_uncert, 0.0, 1.0).astype(np.float32)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
