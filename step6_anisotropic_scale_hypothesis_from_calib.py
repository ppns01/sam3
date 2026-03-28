#!/usr/bin/env python3
# 이 파일은 Step 6:
# Step 5에서 만든 calibration 입력을 이용해서
# 일반 물체용 anisotropic scale 후보(sx, sy, sz)를 만든다.

from __future__ import annotations  # 타입 힌트를 조금 더 편하게 쓰기 위한 선언

import os                           # 파일 경로 처리
import json                         # json 읽고 저장
import math                         # 로그/지수/제곱근
import argparse                     # 커맨드라인 인자 처리

import numpy as np                  # 수치 계산
import trimesh                      # mesh 로드


# -------------------------------------------------
# 1) Step 5 결과 로더
# -------------------------------------------------
def load_calib_input(path: str) -> dict:
    # Step 5에서 만든 calib_input.json 읽기
    with open(path, "r") as f:
        return json.load(f)


def load_points(path: str) -> np.ndarray:
    # 관측 point cloud 불러오기
    pts = np.load(path).astype(np.float64)

    # [N,3] 형식이어야 함
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {pts.shape}")

    return pts


def load_mesh_as_trimesh(mesh_path: str) -> trimesh.Trimesh:
    # GLB는 Scene으로 들어오는 경우가 많아서 force="scene" 사용
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(f"unsupported mesh type: {type(loaded)}")

    if len(mesh.vertices) == 0:
        raise ValueError("mesh has no vertices")
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")

    mesh.remove_unreferenced_vertices()
    return mesh


# -------------------------------------------------
# 2) mesh -> surface point cloud
# -------------------------------------------------
def sample_mesh_points(mesh: trimesh.Trimesh, n_points: int, unit_scale: float) -> np.ndarray:
    # mesh 표면에서 균일하게 점 샘플링
    pts = mesh.sample(n_points).astype(np.float64)

    # 단위 보정 (mm면 m로 바꿈)
    pts *= float(unit_scale)

    return pts


# -------------------------------------------------
# 3) point cloud 정리
# -------------------------------------------------
def robust_filter_points(pts: np.ndarray) -> np.ndarray:
    # 너무 적으면 그냥 반환
    if len(pts) < 50:
        return pts

    # 1) 중심 계산
    c = np.median(pts, axis=0)

    # 2) 중심에서의 거리
    r = np.linalg.norm(pts - c[None, :], axis=1)

    # 3) 너무 멀리 있는 outlier 제거
    r_hi = np.percentile(r, 99)
    pts = pts[r <= r_hi]

    return pts


# -------------------------------------------------
# 4) PCA / 회전 불변 통계 계산
# -------------------------------------------------
def compute_pca_stats(pts: np.ndarray) -> dict:
    # 점이 너무 적으면 PCA가 불안정
    if len(pts) < 10:
        raise ValueError("too few points")

    # 평균 중심
    center = pts.mean(axis=0)

    # 중심 제거
    xc = pts - center[None, :]

    # covariance 계산
    cov = np.cov(xc.T)

    # eigenvalue/eigenvector 계산
    evals, evecs = np.linalg.eigh(cov)

    # 큰 순서대로 정렬
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]

    # PCA 좌표계로 투영
    local = xc @ evecs

    # PCA frame에서 extents 계산
    extents = local.max(axis=0) - local.min(axis=0)

    # rotation-invariant 통계량들
    rms_radius = float(np.sqrt(np.mean(np.sum(xc * xc, axis=1))))
    median_radius = float(np.median(np.linalg.norm(xc, axis=1)))
    p90_radius = float(np.percentile(np.linalg.norm(xc, axis=1), 90))
    trace_sqrt = float(np.sqrt(np.sum(evals)))

    return {
        "center_xyz": center.tolist(),
        "pca_basis_3x3": evecs.tolist(),                  # PCA basis
        "cov_eigvals_desc": evals.tolist(),              # eigenvalues
        "cov_sqrt_eigvals_desc": np.sqrt(evals).tolist(),# sqrt eigenvalues
        "pca_extents_desc": extents.tolist(),            # PCA extents
        "rms_radius": rms_radius,
        "median_radius": median_radius,
        "p90_radius": p90_radius,
        "trace_sqrt": trace_sqrt,
        "num_points": int(len(pts)),
    }


# -------------------------------------------------
# 5) scale hypothesis 생성
# -------------------------------------------------
def safe_ratio(a: float, b: float) -> float:
    # 0으로 나누는 상황 방지
    return float(a) / max(float(b), 1e-8)


def clip_scale_triplet(sxyz, s_min=1e-4, s_max=1e3):
    # 비정상적으로 작은/큰 스케일 방지
    sxyz = np.asarray(sxyz, dtype=np.float64)
    sxyz = np.clip(sxyz, s_min, s_max)
    return sxyz


def regularize_toward_isotropic(sxyz, s_iso, alpha=0.35):
    """
    anisotropic 후보가 너무 과하게 벌어지지 않도록
    isotropic scale 쪽으로 조금 당겨준다.

    alpha=0   -> 완전 isotropic
    alpha=1   -> raw anisotropic 그대로
    """
    sxyz = np.asarray(sxyz, dtype=np.float64)
    s_iso = float(s_iso)

    out = []
    for s in sxyz:
        v = (1.0 - alpha) * math.log(max(s_iso, 1e-8)) + alpha * math.log(max(float(s), 1e-8))
        out.append(math.exp(v))

    return np.asarray(out, dtype=np.float64)


def add_hyp(hyps, sxyz, kind, info):
    # 후보 하나 추가
    sxyz = clip_scale_triplet(sxyz)

    if not np.all(np.isfinite(sxyz)):
        return

    hyps.append({
        "sx_sy_sz": [float(sxyz[0]), float(sxyz[1]), float(sxyz[2])],
        "kind": str(kind),
        "info": info,
    })


def deduplicate_hypotheses(hyps, log_tol=0.03):
    """
    비슷한 scale 후보는 하나로 합친다.
    log space에서 3% 이내면 같은 후보로 본다.
    """
    def key_of(h):
        v = np.log(np.clip(np.asarray(h["sx_sy_sz"], dtype=np.float64), 1e-8, None))
        return v

    out = []
    keys = []

    for h in hyps:
        k = key_of(h)

        ok = True
        for p in keys:
            if np.max(np.abs(k - p)) < log_tol:
                ok = False
                break

        if ok:
            out.append(h)
            keys.append(k)

    return out


def generate_anisotropic_scale_hypotheses(obs_stats: dict, mesh_stats: dict):
    hyps = []

    # -------------------------------------------------
    # 5-1) 먼저 isotropic 기준 scale 하나 구함
    # -------------------------------------------------
    s_iso_rms = safe_ratio(obs_stats["rms_radius"], mesh_stats["rms_radius"])
    s_iso_med = safe_ratio(obs_stats["median_radius"], mesh_stats["median_radius"])
    s_iso_p90 = safe_ratio(obs_stats["p90_radius"], mesh_stats["p90_radius"])
    s_iso_trace = safe_ratio(obs_stats["trace_sqrt"], mesh_stats["trace_sqrt"])

    # 여러 isotropic 추정값의 중앙값을 baseline으로 사용
    s_iso = float(np.median([s_iso_rms, s_iso_med, s_iso_p90, s_iso_trace]))

    # isotropic baseline 후보 추가
    add_hyp(
        hyps,
        [s_iso, s_iso, s_iso],
        "isotropic_median",
        {
            "sources": {
                "rms": s_iso_rms,
                "median": s_iso_med,
                "p90": s_iso_p90,
                "trace": s_iso_trace,
            }
        }
    )

    # -------------------------------------------------
    # 5-2) PCA extents ratio 기반 anisotropic 후보
    # -------------------------------------------------
    obs_ext = np.asarray(obs_stats["pca_extents_desc"], dtype=np.float64)
    mesh_ext = np.asarray(mesh_stats["pca_extents_desc"], dtype=np.float64)

    s_ext = obs_ext / np.clip(mesh_ext, 1e-8, None)

    add_hyp(
        hyps,
        regularize_toward_isotropic(s_ext, s_iso, alpha=0.35),
        "pca_extents_reg",
        {"raw": s_ext.tolist(), "s_iso": s_iso, "alpha": 0.35},
    )

    # -------------------------------------------------
    # 5-3) covariance eigen sqrt ratio 기반 anisotropic 후보
    # -------------------------------------------------
    obs_sv = np.asarray(obs_stats["cov_sqrt_eigvals_desc"], dtype=np.float64)
    mesh_sv = np.asarray(mesh_stats["cov_sqrt_eigvals_desc"], dtype=np.float64)

    s_cov = obs_sv / np.clip(mesh_sv, 1e-8, None)

    add_hyp(
        hyps,
        regularize_toward_isotropic(s_cov, s_iso, alpha=0.35),
        "pca_cov_reg",
        {"raw": s_cov.tolist(), "s_iso": s_iso, "alpha": 0.35},
    )

    # -------------------------------------------------
    # 5-4) extents/cov의 평균 후보
    # -------------------------------------------------
    s_blend = np.exp(0.5 * (
        np.log(np.clip(s_ext, 1e-8, None)) +
        np.log(np.clip(s_cov, 1e-8, None))
    ))

    add_hyp(
        hyps,
        regularize_toward_isotropic(s_blend, s_iso, alpha=0.50),
        "pca_blend_reg",
        {"raw": s_blend.tolist(), "s_iso": s_iso, "alpha": 0.50},
    )

    # -------------------------------------------------
    # 5-5) isotropic 주변 안전 후보도 같이 둔다
    # -------------------------------------------------
    for sf in [0.85, 1.0, 1.15]:
        add_hyp(
            hyps,
            [s_iso * sf, s_iso * sf, s_iso * sf],
            "isotropic_around",
            {"base": s_iso, "factor": sf},
        )

    # 중복 제거
    hyps = deduplicate_hypotheses(hyps, log_tol=0.03)
    return hyps


# -------------------------------------------------
# 6) main
# -------------------------------------------------
def main():
    # 인자 정의
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline_config.json")

    # Step 5 결과 폴더
    ap.add_argument("--capture_dir", required=True)
    # 선택한 GLB
    args = ap.parse_args()

    # 단위

    # mesh sampling 개수
    cfg = load_calib_input(args.config)
    mesh_unit = str(cfg.get("mesh_unit", "m"))
    mesh_sample_points = int(cfg.get("mesh_sample_points", 50000))
    max_meshes = int(cfg.get("num_seeds", 10))
    unit_scale = 1.0 if mesh_unit == "m" else 0.001


    # Step 5 결과 경로
    calib_json_path = os.path.join(args.capture_dir, "calib_prep", "calib_input.json")
    obs_points_path = os.path.join(args.capture_dir, "calib_prep", "obs_points.npy")

    for p in [calib_json_path, obs_points_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # 출력 폴더
    trellis_out_dir = os.path.join(args.capture_dir, "trellis2", "outputs")
    if not os.path.isdir(trellis_out_dir):
        raise FileNotFoundError(f"step4 outputs 폴더 없음: {trellis_out_dir}")
    summary_files = sorted(
        f for f in os.listdir(trellis_out_dir)
        if f.startswith("summary_") and f.endswith(".json")
    )
    if not summary_files:
        raise FileNotFoundError(f"step4 summary 없음: {trellis_out_dir}")

    step4_summary_path = os.path.join(trellis_out_dir, summary_files[-1])
    step4_summary = load_calib_input(step4_summary_path)
    mesh_records = step4_summary.get("records", [])[:max_meshes]
    if not mesh_records:
        raise RuntimeError("step4 summary records가 비어있음")

    out_dir = os.path.join(args.capture_dir, "anisotropic_scale_hypothesis")
    os.makedirs(out_dir, exist_ok=True)

    # Step 5 입력 읽기
    calib = load_calib_input(calib_json_path)
    obs_pts = load_points(obs_points_path)
    obs_pts = robust_filter_points(obs_pts)
    obs_stats = compute_pca_stats(obs_pts)
    all_mesh_results = []

    # mesh 읽기
    for rec in mesh_records:
        mesh_path = rec.get("glb")
        seed = int(rec.get("seed", -1))
        if not mesh_path or not os.path.exists(mesh_path):
            print(f"[WARN] mesh 파일 없음, skip: seed={seed} path={mesh_path}")
            continue

        try:
            mesh = load_mesh_as_trimesh(mesh_path)
            mesh_pts = sample_mesh_points(mesh, n_points=mesh_sample_points, unit_scale=unit_scale)
            mesh_pts = robust_filter_points(mesh_pts)
            mesh_stats = compute_pca_stats(mesh_pts)
            hyps = generate_anisotropic_scale_hypotheses(obs_stats, mesh_stats)

            v = np.asarray(mesh.vertices, dtype=np.float64) * float(unit_scale)
            mesh_center = 0.5 * (v.min(axis=0) + v.max(axis=0))
            mesh_diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))

            mesh_result = {
                "seed": seed,
                "mesh_path": os.path.abspath(mesh_path),
                "mesh_quality": rec.get("mesh_quality", {}),
                "mesh_unit": mesh_unit,
                "mesh_unit_scale_to_meter": float(unit_scale),
                "init_translation_t0_xyz_m": calib["init_translation_t0_xyz_m"],
                "observed_pca_stats": obs_stats,
                "mesh_pca_stats": mesh_stats,
                "mesh_bbox_center_m": mesh_center.tolist(),
                "mesh_bbox_diag_m": mesh_diag,
                "scale_hypotheses": hyps,
            }
            all_mesh_results.append(mesh_result)
            print(f"[OK] seed={seed} hyps={len(hyps)}")

        except Exception as e:
            print(f"[WARN] seed={seed} 처리 실패: {e}")
            continue

    if not all_mesh_results:
        raise RuntimeError("처리 가능한 mesh가 없음")

    summary = {
        "capture_dir": os.path.abspath(args.capture_dir),
        "source_step4_summary": os.path.abspath(step4_summary_path),
        "mesh_unit": mesh_unit,
        "mesh_unit_scale_to_meter": float(unit_scale),
        "init_translation_t0_xyz_m": calib["init_translation_t0_xyz_m"],
        "observed_pca_stats": obs_stats,
        "num_meshes": len(all_mesh_results),
        "meshes": all_mesh_results,
    }

    # json 저장
    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] anisotropic scale hypothesis init saved")
    print("  json    :", out_json)
    print("  num_meshes:", len(all_mesh_results))
    print("  t0      :", calib["init_translation_t0_xyz_m"])
    print("  num hyps:", len(hyps))
    for i, h in enumerate(hyps[:12]):
        print(f"  [{i:02d}] sxyz={h['sx_sy_sz']}  kind={h['kind']}")


if __name__ == "__main__":
    main()