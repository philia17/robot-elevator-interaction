#!/usr/bin/env python3
"""Deterministic, offline pixel-survey records for wrist-camera review.

This module reads an existing image and explicit coordinates only.  It has no
ROS, camera, serial, process, motion, or control interfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "offline_wrist_pixel_survey_v1"
NO_CONTROL = "NO_CONTROL_NO_CONTACT"
ALLOWED_METHODS = {"MANUAL_APPROXIMATE_SINGLE_FRAME", "MANUAL_REVIEWED"}
ALLOWED_CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}


class PixelSurveyError(ValueError):
    """Invalid or unsafe offline survey input."""


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise PixelSurveyError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PixelSurveyError(f"{field} must be numeric") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise PixelSurveyError(f"{field} must be finite")
    return result


def _point(value: Any, field: str, width: int, height: int) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != {"u", "v"}:
        raise PixelSurveyError(f"{field} must contain exactly u and v")
    u, v = _number(value["u"], f"{field}.u"), _number(value["v"], f"{field}.v")
    if not (0 <= u < width and 0 <= v < height):
        raise PixelSurveyError(f"{field} is outside image bounds")
    return {"u": u, "v": v}


def validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema", "control_status", "image", "image_size_px", "roi_px",
        "timestamp_utc", "metadata_state", "target_identity", "final_q_pose_identity",
        "button_center_px", "tool_center_px", "signed_error_px",
        "annotation", "canonical_sha256",
    }
    if set(record) != required:
        raise PixelSurveyError("record fields must be exact")
    if record["schema"] != SCHEMA or record["control_status"] != NO_CONTROL:
        raise PixelSurveyError("record is not explicitly no-control/no-contact")
    size = record["image_size_px"]
    if not isinstance(size, Mapping) or set(size) != {"width", "height"}:
        raise PixelSurveyError("image_size_px must contain width and height")
    width, height = int(size["width"]), int(size["height"])
    if width <= 0 or height <= 0:
        raise PixelSurveyError("image dimensions must be positive")
    _point(record["button_center_px"], "button_center_px", width, height)
    _point(record["tool_center_px"], "tool_center_px", width, height)
    roi = record["roi_px"]
    if not isinstance(roi, Mapping) or set(roi) != {"x", "y", "width", "height"}:
        raise PixelSurveyError("roi_px fields must be exact")
    if int(roi["x"]) < 0 or int(roi["y"]) < 0 or int(roi["width"]) <= 0 or int(roi["height"]) <= 0:
        raise PixelSurveyError("roi_px must be positive and nonnegative")
    try:
        datetime.fromisoformat(str(record["timestamp_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PixelSurveyError("timestamp_utc must be an ISO-8601 timestamp") from exc
    if record["metadata_state"] != "COMPLETE_OFFLINE_METADATA":
        raise PixelSurveyError("metadata is missing or stale")
    if not str(record["target_identity"]).strip() or not isinstance(record["final_q_pose_identity"], Mapping):
        raise PixelSurveyError("target and final q/pose identity metadata are required")
    annotation = record["annotation"]
    if not isinstance(annotation, Mapping) or set(annotation) != {"method", "confidence", "single_frame_uncertainty"}:
        raise PixelSurveyError("annotation metadata is incomplete or stale")
    if annotation["method"] not in ALLOWED_METHODS or annotation["confidence"] not in ALLOWED_CONFIDENCE:
        raise PixelSurveyError("unsupported annotation metadata")
    if not isinstance(annotation["single_frame_uncertainty"], bool):
        raise PixelSurveyError("single_frame_uncertainty must be boolean")
    error = record["signed_error_px"]
    if not isinstance(error, Mapping) or set(error) != {"du", "dv"}:
        raise PixelSurveyError("signed_error_px fields must be exact")
    expected = {
        "du": record["button_center_px"]["u"] - record["tool_center_px"]["u"],
        "dv": record["button_center_px"]["v"] - record["tool_center_px"]["v"],
    }
    if error != expected:
        raise PixelSurveyError("signed error does not follow button minus tool convention")
    unsigned = dict(record)
    unsigned.pop("canonical_sha256")
    if record["canonical_sha256"] != canonical_sha256(unsigned):
        raise PixelSurveyError("canonical hash mismatch")


def build_record(
    image_path: Path,
    *,
    width: int,
    height: int,
    button_u: float,
    button_v: float,
    tool_u: float,
    tool_v: float,
    timestamp_utc: str,
    target_identity: str,
    final_q_identity: str,
    final_pose_identity: str,
    annotation_method: str,
    confidence: str,
    roi: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    button = _point({"u": button_u, "v": button_v}, "button_center_px", width, height)
    tool = _point({"u": tool_u, "v": tool_v}, "tool_center_px", width, height)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "control_status": NO_CONTROL,
        "image": {"path": str(image_path), "sha256": image_sha256(image_path)},
        "image_size_px": {"width": width, "height": height},
        "roi_px": dict(roi or {"x": 0, "y": 0, "width": width, "height": height}),
        "timestamp_utc": timestamp_utc,
        "metadata_state": "COMPLETE_OFFLINE_METADATA",
        "target_identity": target_identity,
        "final_q_pose_identity": {"final_q_identity": final_q_identity, "final_pose_identity": final_pose_identity},
        "button_center_px": button,
        "tool_center_px": tool,
        "signed_error_px": {"du": button["u"] - tool["u"], "dv": button["v"] - tool["v"]},
        "annotation": {"method": annotation_method, "confidence": confidence, "single_frame_uncertainty": True},
    }
    record["canonical_sha256"] = canonical_sha256(record)
    validate_record(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--button-u", type=float, required=True)
    parser.add_argument("--button-v", type=float, required=True)
    parser.add_argument("--tool-u", type=float, required=True)
    parser.add_argument("--tool-v", type=float, required=True)
    parser.add_argument("--timestamp-utc", required=True)
    parser.add_argument("--target-identity", required=True)
    parser.add_argument("--final-q-identity", required=True)
    parser.add_argument("--final-pose-identity", required=True)
    parser.add_argument("--annotation-method", default="MANUAL_APPROXIMATE_SINGLE_FRAME")
    parser.add_argument("--confidence", default="LOW")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    record = build_record(args.image, width=args.width, height=args.height, button_u=args.button_u, button_v=args.button_v,
                          tool_u=args.tool_u, tool_v=args.tool_v, timestamp_utc=args.timestamp_utc,
                          target_identity=args.target_identity, final_q_identity=args.final_q_identity,
                          final_pose_identity=args.final_pose_identity, annotation_method=args.annotation_method,
                          confidence=args.confidence)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
