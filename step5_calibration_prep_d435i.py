#!/usr/bin/env python3
# 이 파일은 Step 5:
# D435i 기준 calibration 준비 단계다.
# RGB / depth / intrinsics / SAM3 mask / chosen GLB를 읽어서
# 이후 단계가 공통으로 쓸 입력 묶음을 만든다.

from __future__ import annotations  # 타입 힌트를 조금 더 편하게 쓰기 위한 선언

import os                           # 파일/폴더 경로 처리
import json                         # json 저장/로드
import argparse                     # 커맨드라인 인자 처리
from dataclasses import dataclass, asdict  # 간단한 구조체 만들기

import cv2                          # 이미지 저장/로드 및 간단한 시각화
import numpy as np                  # 수치 계산
import trimesh                      # GLB / mesh 로드


# -------------------------------------------------
# 1) 카메라 내부파라미터 구조체
# -------------------------------------------------
@dataclass
class Intrinsics:
    fx: float       # x축 초점거리 (pixel)
    fy: float       # y축 초점거리 (pixel)
    cx: float       # principal point x
    cy: float       # principal point y
    width: int      # 이미지 너비
    height: int     # 이미지 높이


# -------------------------------------------------
# 2) 파일 로더들
# -------------------------------------------------
def load_pipeline_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config 파일이 없어요: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)
def load_intrinsics_json(path: str) -> Intrinsics:
    # intrinsics.json을 읽어서 Intrinsics 구조체로 변환
    with open(path, "r") as f:
        d = json.load(f)

    return Intrinsics(
        fx=float(d["fx"]),
        fy=float(d["fy"]),
        cx=float(d["cx"]),
        cy=float(d["cy"]),
        width=int(d["width"]),
        height=int(d["height"]),
    )


def load_mask(path: str) -> np.ndarray:
    # mask.png를 그레이스케일로 읽음
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)

    # 흰색 부분만 True로 바꿈
    return (m > 127)


def load_depth(path: str) -> np.ndarray:
    # depth_m.npy를 float32 depth map으로 읽음
    d = np.load(path).astype(np.float32)

    # 깊이맵은 반드시 2차원이어야 함
    if d.ndim != 2:
        raise ValueError(f"depth must be HxW, got {d.shape}")

    # NaN, Inf, 음수 depth 정리
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d


def load_color_bgr(path: str) -> np.ndarray:
    # OpenCV는 기본이 BGR로 읽음
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mesh_as_trimesh(mesh_path: str) -> trimesh.Trimesh:
    # GLB는 scene으로 들어오는 경우가 많아서 force="scene" 사용
    loaded = trimesh.load(mesh_path, force="scene")

    # Scene이면 하나의 mesh로 합침
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()

    # 이미 단일 Trimesh면 그대로 사용
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()

    else:
        raise TypeError(f"unsupported mesh type: {type(loaded)}")

    # 기본 sanity check
    if len(mesh.vertices) == 0:
        raise ValueError("mesh has no vertices")
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")

    # 쓰지 않는 vertex 정리
    mesh.remove_unreferenced_vertices()
    return mesh


# -------------------------------------------------
# 3) D435i aligned depth sanity check
# -------------------------------------------------
def check_color_depth_shape(color_bgr: np.ndarray, depth_m: np.ndarray, K: Intrinsics):
    # color와 depth shape가 같은지 확인
    Hc, Wc = color_bgr.shape[:2]
    Hd, Wd = depth_m.shape[:2]

    if (Hc != Hd) or (Wc != Wd):
        raise ValueError(
            "Color and depth shape mismatch.\n"
            "D435i에서는 aligned_depth_to_color를 써야 한다.\n"
            f"color={color_bgr.shape[:2]}, depth={depth_m.shape[:2]}"
        )

    # intrinsics 크기도 color 기준과 같아야 함
    if (K.width != Wc) or (K.height != Hc):
        raise ValueError(
            "intrinsics size mismatch.\n"
            "aligned depth to color + color intrinsics 조합인지 확인해라.\n"
            f"intrinsics=({K.width},{K.height}) color=({Wc},{Hc})"
        )


# -------------------------------------------------
# 4) mask를 기준으로 crop 만들기
# -------------------------------------------------
def crop_by_mask(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: Intrinsics,
    pad: int = 30,
):
    # mask 안의 픽셀 좌표
    ys, xs = np.where(mask)

    # mask가 비어 있으면 에러
    if xs.size == 0:
        raise ValueError("mask is empty")

    H, W = mask.shape

    # bbox + 여유 pad
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(W - 1, int(xs.max()) + pad)
    y1 = min(H - 1, int(ys.max()) + pad)

    # crop 이미지들
    color_c = color_bgr[y0:y1 + 1, x0:x1 + 1].copy()
    depth_c = depth_m[y0:y1 + 1, x0:x1 + 1].copy()
    mask_c = mask[y0:y1 + 1, x0:x1 + 1].copy()

    # crop된 이미지 기준 intrinsics로 변환
    Kc = Intrinsics(
        fx=K.fx,
        fy=K.fy,
        cx=K.cx - x0,
        cy=K.cy - y0,
        width=(x1 - x0 + 1),
        height=(y1 - y0 + 1),
    )

    crop_xyxy = [x0, y0, x1, y1]
    return color_c, depth_c, mask_c, Kc, crop_xyxy


# -------------------------------------------------
# 5) 관측 물체 기본 정보
# -------------------------------------------------
def observed_mask_metrics(depth_m: np.ndarray, mask: np.ndarray, K: Intrinsics):
    # mask가 너무 작으면 잘못된 입력일 가능성이 큼
    if mask.sum() < 50:
        raise ValueError("mask is too small")

    # mask 전체 bbox
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    w_px = float(x1 - x0 + 1)
    h_px = float(y1 - y0 + 1)

    # mask 안에서 유효한 depth만 사용
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    if int(valid.sum()) < 50:
        raise ValueError("valid depth inside mask is too small")

    # 대표 거리: median depth
    z_med = float(np.median(depth_m[valid]))

    # mask 중심
    u0 = float(xs.mean())
    v0 = float(ys.mean())

    # 카메라 좌표계 기준 초기 translation
    tx = (u0 - K.cx) / max(K.fx, 1e-6) * z_med
    ty = (v0 - K.cy) / max(K.fy, 1e-6) * z_med
    tz = z_med

    # depth가 얼마나 잘 들어왔는지 비율
    valid_ratio = float(valid.sum()) / max(float(mask.sum()), 1.0)

    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_width_px": float(w_px),
        "bbox_height_px": float(h_px),
        "center_uv": [u0, v0],
        "z_med_m": z_med,
        "t0_xyz_m": [float(tx), float(ty), float(tz)],
        "valid_depth_pixels": int(valid.sum()),
        "mask_pixels": int(mask.sum()),
        "valid_depth_ratio_in_mask": float(valid_ratio),
    }


# -------------------------------------------------
# 6) depth + K + mask -> 관측 point cloud
# -------------------------------------------------
def backproject_masked_depth_to_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: Intrinsics,
) -> np.ndarray:
    # mask 안의 유효 depth만 선택
    ys_m, xs_m = np.where(mask)
    mask_diag  = float(np.hypot(xs_m.max()-xs_m.min(), ys_m.max()-ys_m.min())) if xs_m.size > 0 else 100.0

    for erode_px in [max(2, int(mask_diag * 0.02)), 2, 0]:
        if erode_px == 0:
            mask_try = mask
        else:
            kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
            mask_try = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)

        valid  = mask_try & np.isfinite(depth_m) & (depth_m > 0)
        ys, xs = np.where(valid)

        if xs.size >= 50:
            print(f'[INFO] backproject: erode_px={erode_px}  pts={xs.size}')
            break

    if xs.size < 50:
        raise ValueError("too few valid depth points")

    # depth
    z = depth_m[ys, xs].astype(np.float64)

    # pinhole camera 모델로 backprojection
    x = (xs.astype(np.float64) - K.cx) / max(K.fx, 1e-8) * z
    y = (ys.astype(np.float64) - K.cy) / max(K.fy, 1e-8) * z

    # [N,3] point cloud
    pts = np.stack([x, y, z], axis=1)
    return pts


def robust_filter_points(pts: np.ndarray) -> np.ndarray:
    # 너무 적으면 그냥 반환
    if len(pts) < 50:
        return pts

    # 1) z outlier 제거
    z = pts[:, 2]
    z_med = float(np.median(z))
    z_std = float(np.std(z))
    keep  = np.abs(z - z_med) < 2.0 * max(z_std, 1e-6)
    pts = pts[keep]

    if len(pts) < 50:
        return pts

    # 2) 중심 기준 radius outlier 제거
    c = np.median(pts, axis=0)
    r = np.linalg.norm(pts - c[None, :], axis=1)
    r_hi = np.percentile(r, 97)
    pts = pts[r <= r_hi]

    return pts


# -------------------------------------------------
# 7) mesh 기본 정보
# -------------------------------------------------
def mesh_bbox_center(mesh: trimesh.Trimesh, unit_scale: float = 1.0) -> np.ndarray:
    # mesh vertex를 meter 기준으로 변환
    v = np.asarray(mesh.vertices, dtype=np.float64) * float(unit_scale)

    # axis-aligned bbox 중심
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    return 0.5 * (mn + mx)


def mesh_bbox_diag(mesh: trimesh.Trimesh, unit_scale: float = 1.0) -> float:
    # mesh vertex를 meter 기준으로 변환
    v = np.asarray(mesh.vertices, dtype=np.float64) * float(unit_scale)

    # bbox 대각선 길이
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    return float(np.linalg.norm(mx - mn))


# -------------------------------------------------
# 8) preview 이미지 저장
# -------------------------------------------------
def save_preview(path: str, color_bgr: np.ndarray, obs: dict):
    vis = color_bgr.copy()

    x0, y0, x1, y1 = obs["bbox_xyxy"]
    u0, v0 = obs["center_uv"]

    # bbox 그리기
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # 중심점 그리기
    cv2.circle(vis, (int(round(u0)), int(round(v0))), 4, (0, 0, 255), -1)

    txt1 = f"h_px={obs['bbox_height_px']:.1f}  w_px={obs['bbox_width_px']:.1f}"
    txt2 = f"z_med={obs['z_med_m']:.4f} m  depth_valid={obs['valid_depth_ratio_in_mask']:.3f}"

    cv2.putText(vis, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(path, vis)


# -------------------------------------------------
# 9) main
# -------------------------------------------------
def main():
    # 커맨드라인 인자 정의
    ap = argparse.ArgumentParser()

    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--config", default="pipeline_config.json")
    args = ap.parse_args()
    cfg = load_pipeline_config(args.config)
    pad = cfg.get("crop_pad", 30)

    # 필요한 파일 경로
    color_path = os.path.join(args.capture_dir, "color.png")
    depth_path = os.path.join(args.capture_dir, "depth_m.npy")
    mask_path = os.path.join(args.capture_dir, "sam3", "mask.png")
    intr_path = os.path.join(args.capture_dir, "intrinsics.json")

    # 파일 존재 확인
    for p in [color_path, depth_path, mask_path, intr_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # 출력 폴더
    out_dir = os.path.join(args.capture_dir, "calib_prep")
    os.makedirs(out_dir, exist_ok=True)

    # 입력 로드
    color_bgr = load_color_bgr(color_path)
    depth_m = load_depth(depth_path)
    mask = load_mask(mask_path)
    K = load_intrinsics_json(intr_path)

    # mesh 단위 변환 비율

    # D435i aligned depth 가정 체크
    check_color_depth_shape(color_bgr, depth_m, K)

    # 관측 기하 정보 추출
    obs = observed_mask_metrics(depth_m, mask, K)

    # crop 생성
    color_c, depth_c, mask_c, Kc, crop_xyxy = crop_by_mask(
        color_bgr, depth_m, mask, K, pad=int(pad)
    )

    # 관측 point cloud 생성
    obs_pts = backproject_masked_depth_to_points(depth_m, mask, K)

    # 노이즈 제거
    obs_pts = robust_filter_points(obs_pts)


    # 파일 저장
    np.save(os.path.join(out_dir, "obs_points.npy"), obs_pts.astype(np.float32))
    np.save(os.path.join(out_dir, "query_depth_crop.npy"), depth_c.astype(np.float32))
    cv2.imwrite(os.path.join(out_dir, "query_crop.png"), color_c)
    cv2.imwrite(os.path.join(out_dir, "query_mask.png"), (mask_c.astype(np.uint8) * 255))

    preview_path = os.path.join(out_dir, "preview_bbox.png")
    save_preview(preview_path, color_bgr, obs)

    # json 결과 저장
    result = {
        "capture_dir": os.path.abspath(args.capture_dir),


        "intrinsics_full": asdict(K),
        "intrinsics_crop": asdict(Kc),
        "crop_xyxy": crop_xyxy,

        "observed": obs,
        "init_translation_t0_xyz_m": obs["t0_xyz_m"],



        "saved": {
            "query_crop_png": os.path.abspath(os.path.join(out_dir, "query_crop.png")),
            "query_mask_png": os.path.abspath(os.path.join(out_dir, "query_mask.png")),
            "query_depth_crop_npy": os.path.abspath(os.path.join(out_dir, "query_depth_crop.npy")),
            "obs_points_npy": os.path.abspath(os.path.join(out_dir, "obs_points.npy")),
            "preview_bbox_png": os.path.abspath(preview_path),
        },
    }

    out_json = os.path.join(out_dir, "calib_input.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print("\n[OK] calibration prep saved")
    print("  json    :", out_json)
    print("  preview :", preview_path)
    print("  t0      :", obs["t0_xyz_m"])
    print("  obs pts :", len(obs_pts))


if __name__ == "__main__":
    main()