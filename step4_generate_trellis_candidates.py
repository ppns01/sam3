#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import traceback
from datetime import datetime, timezone

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PIL import Image
import torch
import trimesh
import o_voxel
import gc; gc.collect()

from trellis2.pipelines import Trellis2ImageTo3DPipeline


# =============================================================
# Config 로더
# =============================================================
def load_pipeline_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config 파일이 없어요: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


# =============================================================
# 유틸 함수들
# =============================================================
def is_memory_like_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, MemoryError)
        or "out of memory" in msg
        or "cuda out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "cuda error: out of memory" in msg
    )


def _extract_trellis_mesh(obj, visited=None):
    """
    dict / list / tuple 형태의 중첩 컨테이너 구조를 재귀 탐색하여
    필수 속성을 모두 가진 TRELLIS Mesh 객체를 찾아 반환합니다.
    순환 참조로 인한 무한 루프를 방지합니다.
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return None
    visited.add(obj_id)

    required_attrs = ["vertices", "faces", "attrs", "coords", "layout", "voxel_size"]

    if all(hasattr(obj, attr) for attr in required_attrs):
        return obj

    if isinstance(obj, dict):
        for v in obj.values():
            res = _extract_trellis_mesh(v, visited)
            if res is not None:
                return res
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            res = _extract_trellis_mesh(item, visited)
            if res is not None:
                return res

    return None


def save_raw_geometry(mesh, out_dir: str, stem: str):
    ply_path = os.path.join(out_dir, f"{stem}.ply")
    obj_path = os.path.join(out_dir, f"{stem}.obj")

    try:
        v = mesh.vertices.detach().cpu().numpy().astype("float32")
        f = mesh.faces.detach().cpu().numpy().astype("int32")

        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        tm.remove_unreferenced_vertices()

        tm.export(ply_path)
        tm.export(obj_path)

        return ply_path, obj_path
    except Exception:
        for p in [ply_path, obj_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise


def export_visual_glb(mesh, out_dir: str, stem: str, decimation_target: int, texture_size: int):
    glb_path = os.path.join(out_dir, f"{stem}.glb")

    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(decimation_target),
            texture_size=int(texture_size),
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        glb.export(glb_path, extension_webp=False)
        return glb_path
    except Exception:
        if glb_path and os.path.exists(glb_path):
            try:
                os.remove(glb_path)
            except OSError:
                pass
        raise


def compute_mesh_quality_score(mesh: trimesh.Trimesh) -> dict:
    """GLB를 trimesh로 로드한 뒤 품질 점수를 계산한다. 높을수록 좋음."""
    v = len(mesh.vertices)
    f = len(mesh.faces)
    is_watertight = bool(mesh.is_watertight)

    areas = mesh.area_faces
    degenerate_ratio = float((areas < 1e-10).sum()) / max(f, 1)

    extents = mesh.bounding_box.extents  # [dx, dy, dz]
    aspect_max = float(extents.max() / max(extents.min(), 1e-8))
    aspect_ok = aspect_max < 5.0

    score = (
        (1.0 if is_watertight else 0.5)  # watertight → +1.0, 아니면 +0.5
        - degenerate_ratio               # 깨진 face 많으면 감점
        + (1.0 if aspect_ok else 0.0)    # bbox 비율 정상이면 +1.0
    )

    return {
        'score':             float(score),
        'is_watertight':     is_watertight,
        'num_vertices':      int(v),
        'num_faces':         int(f),
        'degenerate_ratio':  float(degenerate_ratio),
        'aspect_max':        float(aspect_max),
    }


# =============================================================
# main
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True)
    ap.add_argument("--config", default="pipeline_config.json")
    args = ap.parse_args()

    # config 읽기
    cfg = load_pipeline_config(args.config)
    model_id          = cfg["model_id"]
    input_mode        = cfg["input_mode"]
    preprocess_image  = cfg["preprocess_image"]
    num_seeds         = cfg["num_seeds"]
    decimation_target = cfg["decimation_target"]
    texture_size      = cfg["texture_size"]
    save_raw          = cfg["save_raw"]
    device_str        = cfg["device"]

    # 기본값 검증
    if decimation_target <= 0:
        raise ValueError("decimation_target must be > 0")
    if texture_size <= 0:
        raise ValueError("texture_size must be > 0")

    trellis_dir = os.path.join(args.capture_dir, "trellis2")
    input_path  = os.path.join(trellis_dir, f"input_{input_mode}.png")
    conv_mode   = "RGB" if input_mode == "rgb_black" else "RGBA"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    out_dir = os.path.join(trellis_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # seed 목록 자동 생성 (0 ~ num_seeds-1)
    seeds = list(range(num_seeds))

    # CUDA 검증
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")

    device = torch.device(device_str)
    if device.type != "cuda" or device.index is None:
        raise RuntimeError(
            f"An explicit CUDA device with index is required (e.g., cuda:0). Got: {device_str}"
        )
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested device {device} does not exist. "
            f"Available CUDA device count: {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(device)
    weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[TRELLIS] loading pipeline to {device} ({weight_dtype})...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)

    try:
        if hasattr(pipeline, "to"):
            moved = pipeline.to(device)
            if moved is not None:
                pipeline = moved
        elif hasattr(pipeline, "cuda"):
            pipeline.cuda()
        else:
            raise RuntimeError("Pipeline does not support device transfer methods.")
    except Exception as e:
        print(f"[WARN] pipeline.to({device}) failed: {e}. Falling back to .cuda() ...")
        if hasattr(pipeline, "cuda"):
            pipeline.cuda()
        else:
            raise

    print(f"[TRELLIS] input image : {input_path}")
    print(f"[TRELLIS] seeds       : {seeds}")

    all_records    = []
    failed_records = []
    run_timestamp  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    with Image.open(input_path) as img:
        image = img.convert(conv_mode)

    for i, seed in enumerate(seeds):
        print(f"\n[TRELLIS] generating seed={seed} ({i+1}/{len(seeds)})")
        stem = f"sample_seed_{seed:03d}_{run_timestamp}"

        outputs   = None
        mesh      = None
        ply_path  = None
        obj_path  = None
        glb_path  = None
        meta_path = None
        had_error = False

        used_decimation = int(decimation_target)
        used_texture    = int(texture_size)

        try:
            with torch.inference_mode():
                outputs = pipeline.run(
                    image,
                    seed=int(seed),
                    preprocess_image=bool(preprocess_image),
                )

            if outputs is None:
                raise ValueError("Pipeline returned None.")
            if isinstance(outputs, (list, tuple)) and len(outputs) == 0:
                raise ValueError("Pipeline returned empty outputs.")

            mesh = _extract_trellis_mesh(outputs)
            if mesh is None:
                raise TypeError("Could not find a valid TRELLIS mesh in pipeline outputs.")

            if save_raw:
                ply_path, obj_path = save_raw_geometry(mesh, out_dir, stem + "_raw")

            try:
                glb_path = export_visual_glb(mesh, out_dir, stem, used_decimation, used_texture)
            except Exception as export_e:
                if not is_memory_like_error(export_e):
                    raise
                print(f"[WARN] GLB export OOM ({export_e}). Retrying with lower settings...")
                torch.cuda.empty_cache()
                used_decimation = max(1000, used_decimation // 2)
                used_texture    = max(128,  used_texture    // 2)
                glb_path = export_visual_glb(mesh, out_dir, stem, used_decimation, used_texture)

            # GLB 로드해서 품질 점수 계산
            tm      = trimesh.load(glb_path, force='mesh')
            quality = compute_mesh_quality_score(tm)

            meta_path = os.path.join(out_dir, f"{stem}.json")
            meta = {
                "capture_dir":               os.path.abspath(args.capture_dir),
                "input_path":                os.path.abspath(input_path),
                "model_id":                  model_id,
                "seed":                      int(seed),
                "input_mode":                input_mode,
                "preprocess_image":          bool(preprocess_image),
                "save_raw":                  bool(save_raw),
                "device":                    str(device),
                "weight_dtype":              str(weight_dtype),
                "decimation_target_applied": used_decimation,
                "texture_size_applied":      used_texture,
                "raw_ply":  os.path.abspath(ply_path) if ply_path else None,
                "raw_obj":  os.path.abspath(obj_path) if obj_path else None,
                "glb":      os.path.abspath(glb_path),
                "mesh_quality": quality,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            all_records.append(meta)

            print("[OK] saved")
            print(f"  glb     : {glb_path}")
            print(f"  quality : {quality['score']:.4f}  "
                  f"(watertight={quality['is_watertight']}, "
                  f"degen={quality['degenerate_ratio']:.4f})")
            if save_raw:
                print(f"  raw     : {ply_path}")
            print(f"  meta    : {meta_path}")

        except Exception as e:
            had_error = True
            print(f"[FAIL] seed={seed} failed.")
            traceback.print_exc()

            failed_records.append({
                "seed":       int(seed),
                "error_type": type(e).__name__,
                "error":      str(e),
                "traceback":  traceback.format_exc(),
            })

            for p in [ply_path, obj_path, glb_path, meta_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        print(f"[CLEANUP] Removed orphan file: {p}")
                    except OSError:
                        pass
        finally:
            outputs = None
            mesh    = None
            torch.cuda.empty_cache()

    # 품질 점수 높은 순으로 정렬
    all_records_sorted = sorted(
        all_records,
        key=lambda x: x["mesh_quality"]["score"],
        reverse=True,
    )

    summary = {
        "capture_dir":            os.path.abspath(args.capture_dir),
        "input_path":             os.path.abspath(input_path),
        "model_id":               model_id,
        "input_mode":             input_mode,
        "preprocess_image":       bool(preprocess_image),
        "save_raw":               bool(save_raw),
        "device":                 str(device),
        "weight_dtype":           str(weight_dtype),
        "base_decimation_target": decimation_target,
        "base_texture_size":      texture_size,
        "num_seeds":              num_seeds,
        "seeds_attempted":        seeds,
        "num_candidates_success": len(all_records),
        "num_candidates_failed":  len(failed_records),
        "run_timestamp":          run_timestamp,
        "records":                all_records_sorted,  # 품질순 정렬!
        "failed_records":         failed_records,
    }

    summary_path = os.path.join(out_dir, f"summary_{run_timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] TRELLIS generation finished.")
    print(f"  Success : {len(all_records)} / {num_seeds}")
    print(f"  Failed  : {len(failed_records)}")
    print(f"  summary : {summary_path}")


if __name__ == "__main__":
    main()