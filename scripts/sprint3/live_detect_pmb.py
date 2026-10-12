#!/usr/bin/env python3
"""
PMB live detection dashboard (Phase 1 — Qwen3-VL single-model).

Shows synced RGB-D preview with:
  - Optical center crosshair + center depth (mm)
  - Qwen3-VL bbox + per-box description + 3D coords
  - Explicit banner when no elevator buttons detected
  - Key help panel on the right

Requires ascamera_node running. Example:
  python3 ~/scripts/sprint3/live_detect_pmb.py --building pmb --panel-type inside_panel

Keys: Space snapshot | q quit | o optical | m tape | 1/2 switch VLM model
"""

import argparse
import os
import sys
import threading
import time

import cv2
import rclpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from capture_rgbd import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    PANEL_TYPES,
    RGBDCapture,
    wait_for_frames,
)
from qwen3_backend import MODEL_PROFILES, profile, run_vlm_detect  # noqa: E402

KEY_HELP = [
    "e      arm panel",
    "Space  snapshot",
    "q/Esc  quit",
    "o      optical O",
    "m      tape ref",
    "1      Qwen3-2B",
    "2      Qwen3-8B",
]


def attach_3d_coords(node, detections):
    """Fill coords_3d from depth at each bbox center; Z is raw depth mm."""
    K = node.intrinsics
    out = []
    for det in detections:
        det = dict(det)
        x, y, w, h = det["bbox"]
        u, v = x + w // 2, y + h // 2
        z_mm = node.median_depth_mm(node.depth_frame, u, v)
        if z_mm is None:
            det["depth_ok"] = False
            det["coords_3d"] = None
        else:
            det["depth_ok"] = True
            det["coords_3d"] = node.backproject_mm(
                u, v, z_mm, K["fx"], K["fy"], K["cx"], K["cy"])
        out.append(det)
    return out


class VLMInferenceWorker:
    """Tracks whether a background VLM subprocess is running."""

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False

    def is_busy(self):
        with self._lock:
            return self._busy

    def run_async(self, frame, model_key, on_done, panel_armed=False):
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _target():
            try:
                result = run_vlm_detect(frame, model_key, panel_armed=panel_armed)
                on_done(result)
            except Exception as exc:
                on_done({
                    "model_key": model_key,
                    "model_label": profile(model_key)["label"],
                    "detections": [],
                    "status_message": f"VLM error: {exc}",
                    "parse_error": str(exc),
                    "error": str(exc),
                    "elapsed": 0.0,
                    "response": "",
                    "returncode": -1,
                })
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_target, daemon=True).start()
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="PMB live detection dashboard")
    parser.add_argument("--building", default="pmb")
    parser.add_argument("--panel-type", choices=PANEL_TYPES, default="inside_panel")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vlm-model",
        choices=list(MODEL_PROFILES.keys()),
        default="qwen3-2b",
        help="Single VLM model for bbox + per-box description",
    )
    parser.add_argument(
        "--vlm-interval",
        type=float,
        default=None,
        help="Seconds between VLM passes (default: model-specific)",
    )
    parser.add_argument("--tape-mm", type=float, default=None,
                        help="Optional tape-measure reference (mm)")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=2.0,
        help="UI scale for live preview (2.0 = 1280x960 panels, larger text)",
    )
    return parser.parse_args()


def apply_vlm_result(node, result, model_key):
    prof = profile(model_key)
    node.vlm_model_label = prof["label"]
    dets = attach_3d_coords(node, result.get("detections") or [])
    node.overlay_detections = dets
    node.overlay_detection_banner = result.get("status_message")
    if result.get("parse_error") and not dets:
        node.overlay_detection_banner = (
            result.get("status_message") or "VLM output could not be parsed")
    elapsed = result.get("elapsed") or 0.0
    err = result.get("error")
    if err:
        node.live_status = f"VLM error ({elapsed:.1f}s): {err}"
    else:
        node.live_status = (
            f"{prof['label']} | {len(dets)} boxes | {elapsed:.1f}s | "
            f"{node.overlay_detection_banner}")


def run_live(node, model_key, vlm_interval, tape_mm=None, vlm_interval_override=None):
    window = "PMB Live Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    scale = max(1.0, float(getattr(node, "display_scale", 1.0)))
    cv2.resizeWindow(window, int(1480 * scale), int(520 * scale))

    node.show_key_help = True
    node.key_help_lines = KEY_HELP
    node.vlm_model_label = profile(model_key)["label"]

    worker = VLMInferenceWorker()
    last_vlm = 0.0
    inference_pending = False
    detection_age = None
    vlm_generation = 0
    active_vlm_generation = None

    if tape_mm is not None:
        node.overlay_tape_reference_mm = tape_mm

    def on_vlm_done(result, request_generation, request_model_key):
        nonlocal inference_pending, detection_age, last_vlm, active_vlm_generation
        if active_vlm_generation == request_generation:
            inference_pending = False
            active_vlm_generation = None

        if (
            request_generation != vlm_generation
            or request_model_key != model_key
            or not node.panel_armed
        ):
            print(
                f"Ignored stale VLM result "
                f"[{result.get('model_label') or request_model_key}]")
            return

        apply_vlm_result(node, result, request_model_key)
        detection_age = time.monotonic()
        last_vlm = time.monotonic()
        print(
            f"VLM [{result.get('model_label')}]: "
            f"{len(result.get('detections') or [])} boxes in "
            f"{result.get('elapsed', 0):.1f}s — {result.get('status_message')}")

    print("\nPMB Live Detection (Phase 1 — Qwen3-VL)")
    print(f"  Model: {profile(model_key)['label']}  interval: {vlm_interval:.0f}s")
    print("  e=arm panel  Space=snapshot  q=quit  o=optical  m=tape  1=2B  2=8B\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            now = time.monotonic()
            if (
                node.panel_armed
                and node.has_frames()
                and not node.is_frozen()
                and not inference_pending
                and not worker.is_busy()
                and now - last_vlm >= vlm_interval
            ):
                inference_pending = True
                node.live_status = f"INFERRING ({profile(model_key)['label']})..."
                frame = node.rgb_frame.copy()
                request_generation = vlm_generation
                request_model_key = model_key
                active_vlm_generation = request_generation
                callback = (
                    lambda result, gen=request_generation, key=request_model_key:
                    on_vlm_done(result, gen, key)
                )
                if not worker.run_async(frame, request_model_key, callback, panel_armed=True):
                    inference_pending = False
                    active_vlm_generation = None
            elif not node.panel_armed:
                node.live_status = "STANDBY — press E at elevator panel"
                node.overlay_detection_banner = "STANDBY — press E at elevator panel"
                if node.overlay_detections:
                    node.overlay_detections = []

            if node.is_frozen():
                node.live_status = "STREAM FROZEN — restart launcher"
            elif not inference_pending and detection_age is not None:
                age = now - detection_age
                if node.overlay_detections:
                    node.live_status = (
                        f"{profile(model_key)['label']} | "
                        f"{len(node.overlay_detections)} boxes | stale {age:.0f}s")
                elif node.overlay_detection_banner:
                    node.live_status = (
                        f"{node.overlay_detection_banner} | stale {age:.0f}s")

            cv2.imshow(window, node.get_preview_frame())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                node.show_optical_center = not node.show_optical_center
            if key == ord("m"):
                try:
                    text = input("Tape measure lens→panel (mm): ").strip()
                    node.overlay_tape_reference_mm = float(text)
                    print(f"Tape reference: {node.overlay_tape_reference_mm:.0f} mm")
                except (ValueError, EOFError):
                    print("Invalid tape value")
            if key in (ord(" "), ord("s")):
                if node.capture_snapshot(0.5):
                    print(f"Saved snapshot: {node.last_saved}")
            if key == ord("e"):
                node.panel_armed = not node.panel_armed
                node.overlay_detections = []
                vlm_generation += 1
                active_vlm_generation = None
                last_vlm = 0.0
                detection_age = None
                inference_pending = False
                if node.panel_armed:
                    node.overlay_detection_banner = "ARMED — detecting..."
                    print("Panel ARMED — VLM detection enabled")
                else:
                    node.overlay_detection_banner = "STANDBY — press E at elevator panel"
                    print("Panel STANDBY — VLM detection paused")
            if key == ord("1"):
                model_key = "qwen3-2b"
                vlm_interval = vlm_interval_override or profile(model_key)["default_interval"]
                node.overlay_detections = []
                node.overlay_detection_banner = "Switching to Qwen3-VL-2B..."
                node.vlm_model_label = profile(model_key)["label"]
                vlm_generation += 1
                active_vlm_generation = None
                inference_pending = False
                last_vlm = 0.0
                detection_age = None
                print("Switched to Qwen3-VL-2B")
            if key == ord("2"):
                model_key = "qwen3-8b"
                vlm_interval = vlm_interval_override or profile(model_key)["default_interval"]
                node.overlay_detections = []
                node.overlay_detection_banner = "Switching to Qwen3-VL-8B..."
                node.vlm_model_label = profile(model_key)["label"]
                vlm_generation += 1
                active_vlm_generation = None
                inference_pending = False
                last_vlm = 0.0
                detection_age = None
                print("Switched to Qwen3-VL-8B")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def main():
    global args
    args = parse_args()
    model_key = args.vlm_model
    vlm_interval = args.vlm_interval or profile(model_key)["default_interval"]
    output_dir = os.path.join(os.path.expanduser(args.output_dir), args.building)

    rclpy.init(args=["live_detect_pmb"])
    node = RGBDCapture(output_dir, args.building, args.panel_type)
    node.display_scale = max(1.0, args.display_scale)
    node.live_status = "Waiting for camera frames..."
    node.overlay_detection_banner = "STANDBY — press E at elevator panel"

    print(f"Building: {args.building}  Panel: {args.panel_type}")
    print(f"Snapshots: {node.snapshot_dir}")
    print(f"VLM: {profile(model_key)['label']} every {vlm_interval:.0f}s")
    print(f"Display scale: {node.display_scale}x")

    if not wait_for_frames(node, timeout_sec=20.0):
        print("ERROR: No synchronized frames. Is ascamera_node running?")
        print("Tip: wait for camera LED/stream, then re-run the launcher.")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    run_live(
        node, model_key, vlm_interval,
        tape_mm=args.tape_mm,
        vlm_interval_override=args.vlm_interval,
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
