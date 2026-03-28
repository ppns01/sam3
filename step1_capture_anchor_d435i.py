#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import datetime
import select
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo


# -------------------------------------------------
# 1) 기본 자료구조
# -------------------------------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


# -------------------------------------------------
# 2) 저장 함수
# -------------------------------------------------
def save_intrinsics_json(path: str, intr: Intrinsics):
    data = {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "width": int(intr.width),
        "height": int(intr.height),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_capture(out_dir: str, rgb: np.ndarray, depth_m: np.ndarray, intr: Intrinsics, meta: dict):
    os.makedirs(out_dir, exist_ok=True)

    color_path = os.path.join(out_dir, "color.png")
    depth_path = os.path.join(out_dir, "depth_m.npy")
    intr_path = os.path.join(out_dir, "intrinsics.json")
    meta_path = os.path.join(out_dir, "frame_meta.json")

    PILImage.fromarray(rgb).save(color_path)
    np.save(depth_path, depth_m.astype(np.float32))
    save_intrinsics_json(intr_path, intr)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[OK] Saved anchor frame")
    print("  color :", color_path)
    print("  depth :", depth_path)
    print("  intr  :", intr_path)
    print("  meta  :", meta_path)


# -------------------------------------------------
# 3) ROS raw bytes -> numpy
# -------------------------------------------------
def _reshape_by_step(
    buf: np.ndarray,
    h: int,
    w: int,
    bytes_per_pixel: int,
    step_bytes: int,
) -> np.ndarray:
    """
    ROS Image는 row마다 step 바이트를 가진다.
    padding이 있을 수 있으므로 step 기준으로 먼저 reshape 후 잘라낸다.
    """
    expected = h * step_bytes
    if buf.size < expected:
        raise ValueError(f"buffer too small: have={buf.size}, expected={expected}")

    row = buf[:expected].reshape(h, step_bytes)
    row = row[:, : w * bytes_per_pixel]
    return row.reshape(h, w, bytes_per_pixel)


def decode_color(msg: RosImage) -> np.ndarray:
    """
    ROS color image -> RGB uint8 (H, W, 3)

    Supports:
      rgb8, bgr8, rgba8, bgra8, mono8, 8UC1
    """
    enc = str(msg.encoding).lower()
    h = int(msg.height)
    w = int(msg.width)
    step_bytes = int(msg.step)

    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        img = _reshape_by_step(buf, h, w, 3, step_bytes)
        return img.copy()

    elif enc == "bgr8":
        img = _reshape_by_step(buf, h, w, 3, step_bytes)
        return img[:, :, ::-1].copy()

    elif enc == "rgba8":
        img4 = _reshape_by_step(buf, h, w, 4, step_bytes)
        return img4[:, :, :3].copy()

    elif enc == "bgra8":
        img4 = _reshape_by_step(buf, h, w, 4, step_bytes)
        return img4[:, :, :3][:, :, ::-1].copy()

    elif enc in ("mono8", "8uc1"):
        img1 = _reshape_by_step(buf, h, w, 1, step_bytes)[:, :, 0]
        return np.stack([img1, img1, img1], axis=-1).copy()

    else:
        raise ValueError(
            f"unsupported color encoding: enc={msg.encoding}, "
            f"size=({msg.height},{msg.width}), step={msg.step}, data_len={len(msg.data)}"
        )


def decode_depth_to_meters(msg: RosImage) -> np.ndarray:
    """
    ROS depth image -> float32 meters (H, W)

    Supports:
      16UC1 / mono16 : mm
      32FC1          : meters
    """
    enc = str(msg.encoding).lower()
    h = int(msg.height)
    w = int(msg.width)
    step_bytes = int(msg.step)

    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("16uc1", "mono16"):
        row = buf[: h * step_bytes].reshape(h, step_bytes)
        row = row[:, : w * 2]
        d = row.view(np.uint16).reshape(h, w).astype(np.float32) * 0.001
        d[~np.isfinite(d)] = 0.0
        d[d < 0] = 0.0
        return d

    elif enc == "32fc1":
        row = buf[: h * step_bytes].reshape(h, step_bytes)
        row = row[:, : w * 4]
        d = row.view(np.float32).reshape(h, w).astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0] = 0.0
        return d

    else:
        raise ValueError(
            f"unsupported depth encoding: enc={msg.encoding}, "
            f"size=({msg.height},{msg.width}), step={msg.step}, data_len={len(msg.data)}"
        )


# -------------------------------------------------
# 4) ROS Node
# -------------------------------------------------
class D435iAnchorCaptureNode(Node):
    def __init__(self, color_topic: str, depth_topic: str, info_topic: str):
        super().__init__("d435i_anchor_capture_node")

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.intr: Optional[Intrinsics] = None

        self._last_log_time = 0.0
        self.last_color_stamp_ns: Optional[int] = None
        self.last_depth_stamp_ns: Optional[int] = None
        self.last_info_stamp_ns: Optional[int] = None

        self.create_subscription(RosImage, color_topic, self._cb_color, qos_profile_sensor_data)
        self.create_subscription(RosImage, depth_topic, self._cb_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self._cb_info, qos_profile_sensor_data)

        self.get_logger().info(f"Subscribed color : {color_topic}")
        self.get_logger().info(f"Subscribed depth : {depth_topic}")
        self.get_logger().info(f"Subscribed info  : {info_topic}")

    def _cb_color(self, msg: RosImage):
        try:
            rgb = decode_color(msg)
            if rgb is not None and rgb.size > 0:
                self.latest_rgb = rgb
                self.last_color_stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        except Exception as e:
            now = time.time()
            if now - self._last_log_time > 1.0:
                self.get_logger().error(
                    f"decode_color failed: {e} | "
                    f"enc={msg.encoding} h={msg.height} w={msg.width} "
                    f"step={msg.step} data_len={len(msg.data)}"
                )
                self._last_log_time = now

    def _cb_depth(self, msg: RosImage):
        try:
            depth_m = decode_depth_to_meters(msg)
            if depth_m is not None and depth_m.size > 0:
                self.latest_depth_m = depth_m
                self.last_depth_stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        except Exception as e:
            now = time.time()
            if now - self._last_log_time > 1.0:
                self.get_logger().error(
                    f"decode_depth failed: {e} | "
                    f"enc={msg.encoding} h={msg.height} w={msg.width} "
                    f"step={msg.step} data_len={len(msg.data)}"
                )
                self._last_log_time = now

    def _cb_info(self, msg: CameraInfo):
        try:
            k = msg.k
            self.intr = Intrinsics(
                fx=float(k[0]),
                fy=float(k[4]),
                cx=float(k[2]),
                cy=float(k[5]),
                width=int(msg.width),
                height=int(msg.height),
            )
            self.last_info_stamp_ns = int(msg.header.stamp.sec) * 10**9 + int(msg.header.stamp.nanosec)
        except Exception as e:
            now = time.time()
            if now - self._last_log_time > 1.0:
                self.get_logger().error(f"camera info parse failed: {e}")
                self._last_log_time = now


# -------------------------------------------------
# 5) main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="./captures")
    ap.add_argument("--run_name", default=None)

    # D435i + realsense2_camera 기본 토픽
    ap.add_argument("--color_topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--depth_topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--info_topic", default="/camera/camera/aligned_depth_to_color/camera_info")

    ap.add_argument("--warmup_frames", type=int, default=15)
    args = ap.parse_args()

    if args.run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{ts}"
    else:
        run_name = args.run_name

    out_dir = os.path.join(args.out_root, run_name)

    print("\n--- D435i Anchor Capture ---")
    print("Expected setup:")
    print("  - color image")
    print("  - aligned depth to color")
    print("  - aligned_depth_to_color camera_info")
    print("\nControls:")
    print("  ENTER or s+ENTER : capture anchor frame")
    print("  q+ENTER          : quit")
    print("-----------------------------\n")

    rclpy.init()
    node = D435iAnchorCaptureNode(
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        info_topic=args.info_topic,
    )

    warm = 0
    last_status = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            now = time.time()
            if now - last_status > 1.0:
                have_c = node.latest_rgb is not None
                have_d = node.latest_depth_m is not None
                have_i = node.intr is not None
                print(f"[STATUS] color={have_c} depth={have_d} intr={have_i} warmup={warm}/{args.warmup_frames}")
                last_status = now

            if node.latest_rgb is None or node.latest_depth_m is None or node.intr is None:
                continue

            # aligned depth to color 전제 확인
            if node.latest_rgb.shape[:2] != node.latest_depth_m.shape[:2]:
                print(
                    f"[WARN] color/depth shape mismatch: "
                    f"color={node.latest_rgb.shape[:2]}, depth={node.latest_depth_m.shape[:2]}\n"
                    f"Check align_depth.enable:=true"
                )
                continue

            if warm < args.warmup_frames:
                warm += 1
                continue

            if select.select([sys.stdin], [], [], 0.0)[0]:
                cmd = sys.stdin.readline().strip().lower()

                if cmd == "q":
                    print("[QUIT]")
                    return

                if cmd in ("", "s"):
                    rgb = node.latest_rgb.copy()
                    depth_m = node.latest_depth_m.copy()
                    intr = node.intr

                    meta = {
                        "camera_model_assumption": "Intel RealSense D435i",
                        "topics": {
                            "color_topic": args.color_topic,
                            "depth_topic": args.depth_topic,
                            "info_topic": args.info_topic,
                        },
                        "image_shape": {
                            "color_hw": list(map(int, rgb.shape[:2])),
                            "depth_hw": list(map(int, depth_m.shape[:2])),
                        },
                        "timestamps_ns": {
                            "color": node.last_color_stamp_ns,
                            "depth": node.last_depth_stamp_ns,
                            "info": node.last_info_stamp_ns,
                        },
                        "note": (
                            "This capture assumes aligned depth to color. "
                            "Use realsense2_camera with align_depth.enable:=true"
                        ),
                    }

                    save_capture(out_dir, rgb, depth_m, intr, meta)
                    return

    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        return

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()