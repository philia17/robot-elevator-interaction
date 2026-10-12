#!/usr/bin/env python3
"""
capture_rgbd.py — Keyboard-triggered RGB-D capture from Nuwa-HP60C.

Press Enter (OpenCV window focused) or Space to capture a time-synchronized pair.
Press 'q' to quit. Use --auto for headless interval capture over SSH.

Saves under {output_dir}/{building}/:
  {panel_type}_{shot_num:03d}_rgb.jpg
  {panel_type}_{shot_num:03d}_depth.png
  {panel_type}_{shot_num:03d}_meta.json
"""

import argparse
import json
import os
import re
import time
import zlib
from datetime import datetime

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

DEFAULT_OUTPUT_DIR = "/data/datasets"
PANEL_TYPES = ("hall_call", "inside_panel", "inside_close", "accessible")

RGB_TOPIC = "/ascamera_hp60c/camera_publisher/rgb0/image"
DEPTH_TOPIC = "/ascamera_hp60c/camera_publisher/depth0/image_raw"

# Verified on AGX smoke test (DEVELOPMENT_PLAN.md); fy/cy assumed symmetric/default.
CAMERA_INFO = {
    "camera": "Nuwa-HP60C",
    "width": 640,
    "height": 480,
    "fx": 564.44,
    "fy": 564.44,
    "cx": 334.58,
    "cy": 240.0,
    "depth_unit": "mm",
    "depth_encoding": "16UC1",
    "rgb_encoding": "bgr8",
    "aligned_depth_to_color": True,
    "notes": "fy/cy not independently verified; symmetric assumption from fx/cx.",
}


class RGBDCapture(Node):
    def __init__(self, output_dir: str, building: str, panel_type: str):
        super().__init__("rgbd_capture")
        self.bridge = CvBridge()
        self.rgb_frame = None
        self.depth_frame = None
        self.last_capture_stamp = None
        self.output_dir = output_dir
        self.building = building
        self.panel_type = panel_type
        self.shot_num = self._next_shot_num()
        self.last_saved = None
        self.last_capture_time = 0.0

        # Freeze detection
        self._last_frame_hash = None
        self._last_frame_time = 0.0
        self._frozen = False
        self._frozen_start_time = 0.0
        self._duplicate_frame_since = None
        self._duplicate_freeze_threshold_sec = 3.0

        # Overlay data (set externally for bbox / notes / 3D coords display)
        self.overlay_bbox = None      # (x, y, w, h) — legacy single box
        self.overlay_note = None      # str
        self.overlay_coords_3d = None  # (x_mm, y_mm, z_mm)
        self.overlay_detections = []   # list of dicts for multi-button overlay
        self.overlay_detection_banner = None  # str — shown when no buttons / not a panel
        self.overlay_tape_reference_mm = None  # optional tape measure reading
        self.show_optical_center = True
        self.show_key_help = False
        self.key_help_lines = []
        self.vlm_model_label = ""
        self.live_status = None
        self.snapshot_dir = os.path.join(self.output_dir, "snapshots")
        # Live dashboard: 2.0 = 1280x960 panels (readable on external monitor).
        self.display_scale = 1.0
        self.panel_armed = False

        self.intrinsics = self._load_intrinsics()

        rgb_sub = Subscriber(self, Image, RGB_TOPIC)
        depth_sub = Subscriber(self, Image, DEPTH_TOPIC)
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self._sync_cb)

        # Timer to check topic liveness every 2s
        self._health_timer = self.create_timer(2.0, self._health_check)

        os.makedirs(output_dir, exist_ok=True)
        self._ensure_camera_info()

    def _next_shot_num(self):
        pattern = re.compile(
            rf"^{re.escape(self.panel_type)}_(\d{{3}})_rgb\.jpg$")
        max_num = 0
        if os.path.isdir(self.output_dir):
            for name in os.listdir(self.output_dir):
                match = pattern.match(name)
                if match:
                    max_num = max(max_num, int(match.group(1)))
        return max_num + 1

    def _dataset_root(self):
        return os.path.dirname(self.output_dir.rstrip("/"))

    def _load_intrinsics(self):
        info = dict(CAMERA_INFO)
        info_path = os.path.join(self._dataset_root(), "camera_info.yaml")
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
                info.update(loaded)
        return {
            "fx": float(info["fx"]),
            "fy": float(info["fy"]),
            "cx": float(info["cx"]),
            "cy": float(info["cy"]),
            "width": int(info["width"]),
            "height": int(info["height"]),
        }

    def _ensure_camera_info(self):
        info_path = os.path.join(self._dataset_root(), "camera_info.yaml")
        if os.path.exists(info_path):
            return
        os.makedirs(self._dataset_root(), exist_ok=True)
        with open(info_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(CAMERA_INFO, handle, sort_keys=False)

    def _sync_cb(self, rgb_msg, depth_msg):
        self.rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        self.depth_frame = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough")
        self.last_capture_stamp = rgb_msg.header.stamp
        now = time.monotonic()
        self._last_frame_time = now

        frame_hash = self._frame_signature(self.rgb_frame, self.depth_frame)
        if frame_hash == self._last_frame_hash:
            if self._duplicate_frame_since is None:
                self._duplicate_frame_since = now
            duplicate_for = now - self._duplicate_frame_since
            if duplicate_for >= self._duplicate_freeze_threshold_sec and not self._frozen:
                self._frozen = True
                self._frozen_start_time = self._duplicate_frame_since
                self.get_logger().error(
                    f"STREAM FROZEN — duplicate RGB-D frames for {duplicate_for:.1f}s. "
                    "Camera may have stalled. Restart capture script.")
        else:
            self._duplicate_frame_since = None
            self._frozen = False

        self._last_frame_hash = frame_hash

    @staticmethod
    def _frame_signature(rgb_frame, depth_frame):
        """Cheap sampled checksum for detecting repeated camera frames."""
        rgb_sample = np.ascontiguousarray(rgb_frame[::16, ::16])
        depth_sample = np.ascontiguousarray(depth_frame[::16, ::16])
        checksum = zlib.crc32(rgb_sample.tobytes())
        return zlib.crc32(depth_sample.tobytes(), checksum)

    def _health_check(self):
        """Periodic check: warn if camera stream has stopped."""
        if not self.has_frames():
            return
        elapsed = time.monotonic() - self._last_frame_time
        if elapsed > 3.0 and not self._frozen:
            self._frozen = True
            self._frozen_start_time = time.monotonic()
            self.get_logger().error(
                f"STREAM FROZEN — no new frames for {elapsed:.1f}s. "
                "Camera may have stalled. Restart capture script.")

    def is_frozen(self):
        return self._frozen

    def has_frames(self):
        return self.rgb_frame is not None and self.depth_frame is not None

    def capture(self, delay_sec: float):
        if not self.has_frames():
            self.get_logger().error("No synchronized frames available yet.")
            return False

        if self._frozen:
            frozen_for = time.monotonic() - self._frozen_start_time
            self.get_logger().error(
                f"REJECTED — stream frozen for {frozen_for:.0f}s. "
                "Restart capture script to recover.")
            return False

        now = time.monotonic()
        if self.last_capture_time and (now - self.last_capture_time) < delay_sec:
            remaining = delay_sec - (now - self.last_capture_time)
            self.get_logger().warn(f"Cooldown active — wait {remaining:.1f}s")
            return False

        shot_id = f"{self.panel_type}_{self.shot_num:03d}"
        rgb_path = os.path.join(self.output_dir, f"{shot_id}_rgb.jpg")
        depth_path = os.path.join(self.output_dir, f"{shot_id}_depth.png")
        meta_path = os.path.join(self.output_dir, f"{shot_id}_meta.json")

        cv2.imwrite(rgb_path, self.rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(depth_path, self.depth_frame)

        stamp = self.last_capture_stamp
        meta = {
            "building": self.building,
            "panel_type": self.panel_type,
            "shot_num": self.shot_num,
            "timestamp_unix": datetime.now().isoformat(),
            "ros_stamp_sec": stamp.sec if stamp else None,
            "ros_stamp_nanosec": stamp.nanosec if stamp else None,
            "rgb_path": os.path.basename(rgb_path),
            "depth_path": os.path.basename(depth_path),
            "rgb_shape": list(self.rgb_frame.shape),
            "depth_shape": list(self.depth_frame.shape),
            "depth_dtype": str(self.depth_frame.dtype),
            "valid_depth_pct": float(np.count_nonzero(self.depth_frame) * 100
                                     / self.depth_frame.size),
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
            handle.write("\n")

        self.get_logger().info(
            f"Saved: {os.path.basename(rgb_path)} | "
            f"{os.path.basename(depth_path)}")
        self.last_saved = shot_id
        self.shot_num += 1
        self.last_capture_time = now
        return True

    def capture_snapshot(self, cooldown_sec: float = 0.5):
        """Save timestamped rgb + colormap depth + meta + overlay preview."""
        if not self.has_frames():
            self.get_logger().error("No synchronized frames available yet.")
            return False

        if self._frozen:
            frozen_for = time.monotonic() - self._frozen_start_time
            self.get_logger().error(
                f"REJECTED — stream frozen for {frozen_for:.0f}s. "
                "Restart capture script to recover.")
            return False

        now = time.monotonic()
        if self.last_capture_time and (now - self.last_capture_time) < cooldown_sec:
            remaining = cooldown_sec - (now - self.last_capture_time)
            self.get_logger().warn(f"Cooldown active — wait {remaining:.1f}s")
            return False

        os.makedirs(self.snapshot_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.snapshot_dir, stamp)
        rgb_path = f"{base}_rgb.jpg"
        depth_path = f"{base}_depth.png"
        overlay_path = f"{base}_overlay.jpg"
        meta_path = f"{base}_meta.json"

        cv2.imwrite(rgb_path, self.rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(depth_path, self._depth_colormap())

        overlay = self.get_preview_frame()
        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        center_z = self.center_distance_mm()
        detections_meta = []
        for det in self.overlay_detections:
            item = {
                "label": det.get("label"),
                "description": det.get("description"),
                "bbox": list(det.get("bbox") or ()),
                "coords_3d_mm": list(det["coords_3d"]) if det.get("coords_3d") else None,
                "depth_ok": det.get("depth_ok"),
            }
            detections_meta.append(item)

        meta = {
            "building": self.building,
            "panel_type": self.panel_type,
            "timestamp": stamp,
            "timestamp_iso": datetime.now().isoformat(),
            "vlm_model": self.vlm_model_label,
            "center_distance_mm": center_z,
            "detection_banner": self.overlay_detection_banner,
            "detections": detections_meta,
            "rgb_path": os.path.basename(rgb_path),
            "depth_path": os.path.basename(depth_path),
            "overlay_path": os.path.basename(overlay_path),
            "intrinsics": self.intrinsics,
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        self.get_logger().info(
            f"Snapshot: {os.path.basename(rgb_path)} | {os.path.basename(depth_path)}")
        self.last_saved = stamp
        self.last_capture_time = now
        return True

    def _depth_colormap(self):
        """Convert raw 16-bit depth to JET colormap for visualization."""
        if self.depth_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        d = self.depth_frame.astype(np.float32)
        mask = d > 0
        if mask.any():
            d_min, d_max = d[mask].min(), d[mask].max()
            if d_max > d_min:
                d = np.clip((d - d_min) / (d_max - d_min) * 255, 0, 255)
            else:
                d[:] = 0
        d = d.astype(np.uint8)
        colored = cv2.applyColorMap(d, cv2.COLORMAP_JET)
        colored[~mask] = 0  # black out invalid pixels
        return colored

    @staticmethod
    def median_depth_mm(depth, u, v, window=5):
        """Median depth in a square window around pixel (u, v). Returns mm or None."""
        if depth is None:
            return None
        h, w = depth.shape[:2]
        half = window // 2
        u0 = max(0, int(u) - half)
        u1 = min(w, int(u) + half + 1)
        v0 = max(0, int(v) - half)
        v1 = min(h, int(v) + half + 1)
        patch = depth[v0:v1, u0:u1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def panel_distance_mm(self, frac=0.4):
        """Median depth over central ROI — legacy helper for batch capture."""
        if self.depth_frame is None:
            return None
        h, w = self.depth_frame.shape[:2]
        margin_y = int(h * (1 - frac) / 2)
        margin_x = int(w * (1 - frac) / 2)
        roi = self.depth_frame[margin_y:h - margin_y, margin_x:w - margin_x]
        valid = roi[roi > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def center_distance_mm(self):
        """Depth at optical center (cx, cy) — panel standoff for live demo."""
        if self.depth_frame is None:
            return None
        u = int(round(self.intrinsics["cx"]))
        v = int(round(self.intrinsics["cy"]))
        return self.median_depth_mm(self.depth_frame, u, v, window=7)

    @staticmethod
    def backproject_mm(u, v, z_mm, fx, fy, cx, cy):
        z_m = z_mm / 1000.0
        x_mm = (u - cx) * z_m / fx * 1000.0
        y_mm = (v - cy) * z_m / fy * 1000.0
        return x_mm, y_mm, z_mm

    @property
    def _scale(self):
        return float(getattr(self, "display_scale", 1.0))

    def _px(self, value):
        return int(round(float(value) * self._scale))

    def _font(self, base_size):
        return max(0.35, base_size * self._scale)

    def _thickness(self, base=1):
        return max(1, int(round(base * self._scale)))

    def _panel_size(self):
        if self.rgb_frame is not None:
            h, w = self.rgb_frame.shape[:2]
        else:
            w, h = 640, 480
        return self._px(w), self._px(h)

    def _draw_optical_center(self, panel):
        if not self.show_optical_center:
            return
        cx = self._px(self.intrinsics["cx"])
        cy = self._px(self.intrinsics["cy"])
        color = (255, 255, 0)  # cyan in BGR
        arm = self._px(20)
        cv2.drawMarker(panel, (cx, cy), color, cv2.MARKER_CROSS, arm, self._thickness(2))
        cv2.circle(panel, (cx, cy), self._px(4), color, self._thickness(1))
        cv2.putText(panel, "O", (cx + self._px(6), cy - self._px(6)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.5), color, self._thickness(2),
                    cv2.LINE_AA)

    @staticmethod
    def _fit_text_lines(panel_w, panel_h, x, y, w, h, lines, line_h=14):
        """Place annotation lines beside the box without leaving the panel."""
        placed = []
        right_x = x + w + 6
        if right_x + 80 < panel_w:
            tx, ty = right_x, max(14, y + 12)
            for line in lines:
                placed.append((line, tx, ty))
                ty += line_h
            return placed

        tx = max(2, min(x, panel_w - 80))
        ty = y + h + 14
        if ty + line_h * len(lines) > panel_h - 8:
            ty = max(30, y - 6 - line_h * len(lines))
        for line in lines:
            placed.append((line, tx, ty))
            ty += line_h
        return placed

    def _draw_detection_overlays(self, panel):
        cx0 = self._px(self.intrinsics["cx"])
        cy0 = self._px(self.intrinsics["cy"])
        panel_h, panel_w = panel.shape[:2]
        line_h = self._px(16)
        for det in self.overlay_detections:
            bbox = det.get("bbox")
            if not bbox:
                continue
            x, y, w, h = (self._px(bbox[0]), self._px(bbox[1]),
                          self._px(bbox[2]), self._px(bbox[3]))
            color = (0, 255, 255)
            cv2.rectangle(panel, (x, y), (x + w, y + h), color, self._thickness(2))
            u, v = x + w // 2, y + h // 2
            if self.show_optical_center:
                cv2.line(panel, (cx0, cy0), (u, v), color, self._thickness(1), cv2.LINE_AA)

            desc = det.get("description") or det.get("label") or "?"
            lines = [str(desc)[:48]]
            coords = det.get("coords_3d")
            if coords:
                tx, ty, tz = coords
                lines.append(f"({tx:.0f}, {ty:.0f}, {tz:.0f}) mm")
            elif det.get("depth_ok") is False:
                lines.append("(depth N/A)")

            for text, tx_pos, ty_pos in self._fit_text_lines(
                    panel_w, panel_h, x, y, w, h, lines, line_h=line_h):
                cv2.putText(panel, text, (tx_pos, ty_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.52), color,
                            self._thickness(2), cv2.LINE_AA)

    def _draw_detection_banner(self, panel):
        if not self.overlay_detection_banner:
            return
        msg = str(self.overlay_detection_banner)
        color = (0, 165, 255)  # orange
        pw = panel.shape[1]
        y0, y1 = self._px(78), self._px(128)
        cv2.rectangle(panel, (0, y0), (pw - 1, y1), (0, 0, 0), -1)
        cv2.putText(panel, msg, (self._px(10), self._px(112)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.72), color,
                    self._thickness(2), cv2.LINE_AA)

    def _draw_key_help(self):
        width = self._px(220)
        height = self._panel_size()[1]
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (width - 1, self._px(36)), (40, 40, 40), -1)
        cv2.putText(panel, "KEYS", (self._px(10), self._px(26)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.58), (220, 220, 220),
                    self._thickness(2), cv2.LINE_AA)
        y = self._px(52)
        for line in self.key_help_lines:
            cv2.putText(panel, line, (self._px(8), y),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.48), (200, 200, 200),
                        self._thickness(2), cv2.LINE_AA)
            y += self._px(22)
        if self.vlm_model_label:
            cv2.putText(panel, "Model:", (self._px(8), height - self._px(28)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.44), (140, 140, 140),
                        self._thickness(1), cv2.LINE_AA)
            cv2.putText(panel, self.vlm_model_label[:22], (self._px(8), height - self._px(8)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.44), (0, 220, 255),
                        self._thickness(2), cv2.LINE_AA)
        return panel

    def get_preview_frame(self):
        pw, ph = self._panel_size()
        if self.rgb_frame is None:
            help_w = self._px(220) if self.show_key_help else 0
            return np.zeros((ph, pw * 2 + help_w, 3), dtype=np.uint8)

        interp = cv2.INTER_CUBIC if self._scale > 1.0 else cv2.INTER_LINEAR
        rgb_panel = cv2.resize(self.rgb_frame, (pw, ph), interpolation=interp)

        bar_h = self._px(36)
        cv2.rectangle(rgb_panel, (0, 0), (pw - 1, bar_h), (0, 0, 0), -1)
        cv2.putText(
            rgb_panel,
            f"RGB  |  {self.building} / {self.panel_type}  |  Shots: {self.shot_num - 1}",
            (self._px(8), self._px(26)),
            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55), (255, 255, 255),
            self._thickness(2), cv2.LINE_AA)
        arm_tag = "ARMED" if getattr(self, "panel_armed", False) else "STANDBY"
        arm_color = (0, 220, 0) if getattr(self, "panel_armed", False) else (140, 140, 140)
        cv2.putText(rgb_panel, arm_tag, (self._px(300), self._px(26)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.52), arm_color,
                    self._thickness(2), cv2.LINE_AA)
        if self.last_saved:
            cv2.putText(rgb_panel, f"Last: {self.last_saved}", (self._px(420), self._px(26)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.48), (255, 255, 0),
                        self._thickness(2), cv2.LINE_AA)

        center_z = self.center_distance_mm()
        if center_z is not None:
            dist_text = f"{center_z:.0f}"
            cv2.putText(rgb_panel, dist_text, (self._px(10), self._px(72)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(1.15), (0, 255, 255),
                        self._thickness(3), cv2.LINE_AA)
            cv2.putText(
                rgb_panel, "mm",
                (self._px(28) + self._px(22 * len(dist_text)), self._px(72)),
                cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55), (0, 255, 255),
                self._thickness(2), cv2.LINE_AA)
            if self.overlay_tape_reference_mm is not None:
                delta = center_z - self.overlay_tape_reference_mm
                ok = abs(delta) < 30
                tag = "PASS" if ok else "FAIL"
                color = (0, 255, 0) if ok else (0, 0, 255)
                cv2.putText(
                    rgb_panel,
                    f"tape {self.overlay_tape_reference_mm:.0f} d={delta:+.0f} [{tag}]",
                    (self._px(10), self._px(96)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.48), color,
                    self._thickness(2), cv2.LINE_AA)

        self._draw_optical_center(rgb_panel)
        self._draw_detection_overlays(rgb_panel)
        self._draw_detection_banner(rgb_panel)

        bar_y = ph - self._px(36)
        if self.depth_frame is None:
            status, color = "WAITING FOR SYNCED FRAMES...", (0, 0, 255)
        elif getattr(self, "live_status", None):
            status = self.live_status
            color = (0, 255, 255)
        elif self._frozen:
            frozen_sec = time.monotonic() - self._frozen_start_time
            status = f"STREAM FROZEN ({frozen_sec:.0f}s) — RESTART SCRIPT"
            color = (0, 0, 255)
        else:
            status = "READY  |  Space: snapshot  |  q: quit"
            color = (0, 255, 0)
        cv2.rectangle(rgb_panel, (0, bar_y), (pw - 1, ph - 1), (0, 0, 0), -1)
        cv2.putText(rgb_panel, status, (self._px(8), bar_y + self._px(26)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.5), color,
                    self._thickness(2), cv2.LINE_AA)

        if self.overlay_bbox is not None:
            x, y, w, h = [self._px(v) for v in self.overlay_bbox]
            cv2.rectangle(rgb_panel, (x, y), (x + w, y + h), (0, 255, 255), self._thickness(2))

        if self.overlay_note is not None:
            cv2.putText(rgb_panel, str(self.overlay_note), (self._px(12), self._px(140)),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font(0.58), (0, 255, 255),
                        self._thickness(2), cv2.LINE_AA)

        if self.overlay_coords_3d is not None:
            cx, cy, cz = self.overlay_coords_3d
            cv2.putText(
                rgb_panel, f"3D: ({cx:.0f}, {cy:.0f}, {cz:.0f}) mm",
                (self._px(12), self._px(168)),
                cv2.FONT_HERSHEY_SIMPLEX, self._font(0.52), (0, 255, 255),
                self._thickness(2), cv2.LINE_AA)

        if self._frozen:
            cv2.rectangle(rgb_panel, (0, 0), (pw - 1, ph - 1), (0, 0, 255), self._thickness(4))

        depth_panel = cv2.resize(self._depth_colormap(), (pw, ph), interpolation=interp)
        cv2.rectangle(depth_panel, (0, 0), (pw - 1, bar_h), (0, 0, 0), -1)
        cv2.putText(depth_panel, "DEPTH (JET colormap, mm)",
                    (self._px(8), self._px(26)),
                    cv2.FONT_HERSHEY_SIMPLEX, self._font(0.55), (255, 255, 255),
                    self._thickness(2), cv2.LINE_AA)
        if self.depth_frame is not None:
            valid = self.depth_frame[self.depth_frame > 0]
            if len(valid) > 0:
                d_range = f"{valid.min()} – {valid.max()} mm"
                cv2.putText(depth_panel, d_range, (self._px(360), self._px(26)),
                            cv2.FONT_HERSHEY_SIMPLEX, self._font(0.48), (255, 255, 255),
                            self._thickness(2), cv2.LINE_AA)

        bar_w = self._px(24)
        bar_x = pw - bar_w
        bar = np.tile(np.linspace(255, 0, ph, dtype=np.uint8), (bar_w, 1)).T
        bar = cv2.applyColorMap(bar, cv2.COLORMAP_JET)
        if self.depth_frame is not None and (self.depth_frame > 0).any():
            bar_h_px = bar.shape[0] - 2
            cv2.putText(
                bar, f"{int(self.depth_frame[self.depth_frame > 0].max())}",
                (1, self._px(14)), cv2.FONT_HERSHEY_SIMPLEX, self._font(0.42),
                (255, 255, 255), self._thickness(1), cv2.LINE_AA)
            cv2.putText(
                bar, f"{int(self.depth_frame[self.depth_frame > 0].min())}",
                (1, bar_h_px), cv2.FONT_HERSHEY_SIMPLEX, self._font(0.42),
                (255, 255, 255), self._thickness(1), cv2.LINE_AA)
        depth_panel[:, bar_x:pw] = bar

        panels = [rgb_panel, depth_panel]
        if self.show_key_help:
            panels.append(self._draw_key_help())
        return np.hstack(panels)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keyboard-triggered RGB-D capture from Nuwa-HP60C")
    parser.add_argument("--building", required=True,
                        help="Building name (output subdir under --output-dir)")
    parser.add_argument("--panel-type", choices=PANEL_TYPES,
                        default="inside_panel",
                        help="Panel type prefix for filenames")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Dataset root (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Cooldown between captures in seconds")
    parser.add_argument("--auto", action="store_true",
                        help="Headless mode: capture every --delay seconds (no GUI)")
    return parser.parse_args()


def wait_for_frames(node, timeout_sec=5.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.has_frames():
            return True
    return False


def run_interactive(node, delay_sec):
    cv2.namedWindow("RGB-D Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RGB-D Capture", 1280, 520)
    print("\nFocus the preview window. Enter/Space: capture | q: quit\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            cv2.imshow("RGB-D Capture", node.get_preview_frame())

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                print(f"\nQuit. Total shots saved: {node.shot_num - 1}")
                break
            if key in (13, ord(" "), ord("c")):
                node.capture(delay_sec)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Total shots saved: {node.shot_num - 1}")
    finally:
        cv2.destroyAllWindows()


def run_auto(node, delay_sec):
    print(f"\nAuto mode: capturing every {delay_sec}s. Ctrl+C to stop.\n")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.has_frames():
                node.capture(delay_sec)
            time.sleep(max(delay_sec, 0.1))
    except KeyboardInterrupt:
        print(f"\nStopped. Total shots saved: {node.shot_num - 1}")


def main():
    args = parse_args()
    output_dir = os.path.join(
        os.path.expanduser(args.output_dir), args.building)

    rclpy.init(args=["capture_rgbd"])
    node = RGBDCapture(output_dir, args.building, args.panel_type)

    print(f"Saving to: {output_dir}")
    print(f"Building:  {args.building}")
    print(f"Panel:     {args.panel_type}")
    print(f"Next shot: {args.panel_type}_{node.shot_num:03d}")
    print("Waiting for synchronized camera frames...")

    if not wait_for_frames(node):
        print("ERROR: No synchronized frames received. Is start_camera.sh running?")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print(f"RGB: {node.rgb_frame.shape}  |  "
          f"Depth: {node.depth_frame.shape} ({node.depth_frame.dtype})")

    if args.auto:
        run_auto(node, args.delay)
    else:
        run_interactive(node, args.delay)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
