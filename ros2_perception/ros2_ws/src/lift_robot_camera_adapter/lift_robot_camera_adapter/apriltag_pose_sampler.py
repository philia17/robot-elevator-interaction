"""Capture timestamp-paired AprilTag detections and TF poses without arm control."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

from .apriltag_sample_core import (
    CAMERA_FRAME,
    SamplingDeadline,
    summarize_samples,
    validate_sample,
)


class AprilTagPoseSampler(Node):
    def __init__(
        self,
        *,
        count,
        timeout_s,
        startup_timeout_s,
        stall_timeout_s,
        min_decision_margin,
    ):
        super().__init__("apriltag_pose_sampler")
        self.target_count = count
        self.timeout_s = timeout_s
        self.started_monotonic = time.monotonic()
        self.min_decision_margin = min_decision_margin
        self.deadline = SamplingDeadline(
            self.started_monotonic,
            startup_timeout_s=startup_timeout_s,
            stall_timeout_s=stall_timeout_s,
            timeout_s=timeout_s,
        )
        self.samples = []
        self.failure = None
        self._detections = {}
        self._transforms = {}
        self._consumed_stamps = set()
        self._last_accepted_stamp_ns = 0
        self._last_rejection_log = 0.0
        self.create_subscription(
            AprilTagDetectionArray,
            "/calibration/apriltag/detections",
            self._detection_callback,
            10,
        )
        self.create_subscription(TFMessage, "/tf", self._tf_callback, 100)
        self.create_timer(0.1, self._timer_callback)

    @staticmethod
    def _stamp_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def _detection_callback(self, msg):
        if msg.header.frame_id != CAMERA_FRAME:
            self._log_rejection("detection header is not camera_optical_frame")
            return
        matches = [
            detection
            for detection in msg.detections
            if detection.family == "tag36h11" and detection.id == 0
        ]
        if len(matches) != 1:
            return
        detection = matches[0]
        if detection.hamming != 0:
            return
        if not math.isfinite(detection.decision_margin) or detection.decision_margin < self.min_decision_margin:
            self._log_rejection("detection margin is below threshold")
            return
        stamp_ns = self._stamp_ns(msg.header.stamp)
        self._detections[stamp_ns] = (msg.header.frame_id, detection)
        self._trim_cache(self._detections)
        self._try_pair(stamp_ns)

    def _tf_callback(self, msg):
        for transform in msg.transforms:
            if (
                transform.header.frame_id == "camera_optical_frame"
                and transform.child_frame_id == "calibration_tag"
            ):
                stamp_ns = self._stamp_ns(transform.header.stamp)
                self._transforms[stamp_ns] = transform
                self._trim_cache(self._transforms)
                self._try_pair(stamp_ns)

    @staticmethod
    def _trim_cache(cache, limit=256):
        if len(cache) <= limit:
            return
        for stamp_ns in sorted(cache)[:-limit]:
            del cache[stamp_ns]

    def _log_rejection(self, reason):
        now = time.monotonic()
        if now - self._last_rejection_log >= 2.0:
            self.get_logger().warning(f"sample rejected: {reason}")
            self._last_rejection_log = now

    def _try_pair(self, stamp_ns):
        if (
            stamp_ns in self._consumed_stamps
            or stamp_ns not in self._detections
            or stamp_ns not in self._transforms
            or len(self.samples) >= self.target_count
        ):
            return
        frame_id, detection = self._detections.pop(stamp_ns)
        transform = self._transforms.pop(stamp_ns)
        self._consumed_stamps.add(stamp_ns)
        if stamp_ns <= self._last_accepted_stamp_ns:
            self._log_rejection("timestamp is not strictly increasing")
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        sample = {
                "stamp_ns": stamp_ns,
                "camera_frame": frame_id,
                "tag_frame": transform.child_frame_id,
                "translation_m": [translation.x, translation.y, translation.z],
                "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
                "centre_px": [detection.centre.x, detection.centre.y],
                "corners_px": [[corner.x, corner.y] for corner in detection.corners],
                "decision_margin": float(detection.decision_margin),
                "hamming": int(detection.hamming),
            }
        try:
            validate_sample(sample, min_decision_margin=self.min_decision_margin)
        except ValueError as exc:
            self._log_rejection(str(exc))
            return
        self.samples.append(sample)
        self._last_accepted_stamp_ns = stamp_ns
        self.deadline.record_progress(time.monotonic())
        if len(self.samples) % 10 == 0 or len(self.samples) == self.target_count:
            self.get_logger().info(f"captured {len(self.samples)}/{self.target_count}")

    def _timer_callback(self):
        if len(self.samples) >= self.target_count:
            return
        self.failure = self.deadline.failure(
            time.monotonic(), len(self.samples), self.target_count
        )

    @property
    def finished(self):
        return len(self.samples) >= self.target_count or self.failure is not None


def _write_json_atomic(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--startup-timeout-s", type=float, default=20.0)
    parser.add_argument("--stall-timeout-s", type=float, default=15.0)
    parser.add_argument("--min-decision-margin", type=float, default=50.0)
    parsed = parser.parse_args(args)
    output = parsed.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {output}")
    if parsed.count < 1:
        raise SystemExit("--count must be positive")
    for name in ("timeout_s", "startup_timeout_s", "stall_timeout_s"):
        value = getattr(parsed, name)
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(parsed.min_decision_margin) or parsed.min_decision_margin < 0.0:
        raise SystemExit("--min-decision-margin must be finite and non-negative")

    rclpy.init(args=None)
    node = AprilTagPoseSampler(
        count=parsed.count,
        timeout_s=parsed.timeout_s,
        startup_timeout_s=parsed.startup_timeout_s,
        stall_timeout_s=parsed.stall_timeout_s,
        min_decision_margin=parsed.min_decision_margin,
    )
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.failure:
            raise SystemExit(node.failure)
        document = {
            "schema_version": 2,
            "capture_status": "complete",
            "label": parsed.label,
            "target": {"family": "tag36h11", "id": 0, "edge_m": 0.064},
            "summary": summarize_samples(node.samples),
            "quality_contract": {
                "camera_frame": CAMERA_FRAME,
                "tag_frame": "calibration_tag",
                "min_decision_margin": parsed.min_decision_margin,
                "hamming": 0,
            },
            "samples": sorted(node.samples, key=lambda sample: sample["stamp_ns"]),
        }
        _write_json_atomic(output, document)
        print(f"APRILTAG_CAPTURE COMPLETE label={parsed.label} output={output}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
