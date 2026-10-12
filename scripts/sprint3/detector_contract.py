"""Shared detector output contract for offline JSON and visualization.

Pure Python — no torch, ultralytics, or Grounding DINO imports.
See docs/detector_interface_contract.md for field semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# cv2 and numpy are imported lazily inside draw_contract_overlay()
# to keep the module importable without OpenCV on Mac

SCHEMA_VERSION = "1.0"

BACKEND_TRADITIONAL_YOLO = "traditional_yolo"
BACKEND_YOLO_WORLD = "yolo_world"
BACKEND_GROUNDING_DINO = "grounding_dino"

BACKENDS = frozenset({
    BACKEND_TRADITIONAL_YOLO,
    BACKEND_YOLO_WORLD,
    BACKEND_GROUNDING_DINO,
})

COARSE_BUTTON_CANDIDATE = "button_candidate"
COARSE_FLOOR_BUTTON_CANDIDATE = "floor_button_candidate"
COARSE_DOOR_BUTTON_CANDIDATE = "door_button_candidate"
COARSE_CALL_BUTTON_CANDIDATE = "call_button_candidate"
COARSE_PANEL_CANDIDATE = "panel_candidate"
COARSE_UNKNOWN_CANDIDATE = "unknown_candidate"

COARSE_TYPES = frozenset({
    COARSE_BUTTON_CANDIDATE,
    COARSE_FLOOR_BUTTON_CANDIDATE,
    COARSE_DOOR_BUTTON_CANDIDATE,
    COARSE_CALL_BUTTON_CANDIDATE,
    COARSE_PANEL_CANDIDATE,
    COARSE_UNKNOWN_CANDIDATE,
})

_OVERLAY_COLOR = (0, 255, 255)  # yellow in BGR
_CENTER_EPS = 1e-3


@dataclass
class Detection:
    detection_id: str
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    score: float
    coarse_type: str
    detector_class: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionArray:
    schema_version: str
    source_detector: str
    image_path: str
    image_width: int
    image_height: int
    detector_latency_ms: float
    detections: list[Detection] = field(default_factory=list)


def xywh_to_xyxy(xywh: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in xywh)
    return (x, y, x + w, y + h)


def clip_xyxy(
    xyxy: tuple[float, float, float, float] | list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = (float(v) for v in xyxy)
    w = float(image_width)
    h = float(image_height)
    return (
        max(0.0, min(x_min, w)),
        max(0.0, min(y_min, h)),
        max(0.0, min(x_max, w)),
        max(0.0, min(y_max, h)),
    )


def compute_center(xyxy: tuple[float, float, float, float] | list[float]) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = (float(v) for v in xyxy)
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def is_valid_box(
    xyxy: tuple[float, float, float, float] | list[float],
    image_width: int,
    image_height: int,
) -> bool:
    x_min, y_min, x_max, y_max = (float(v) for v in xyxy)
    if not (x_min < x_max and y_min < y_max):
        return False
    if image_width <= 0 or image_height <= 0:
        return False
    return (
        0.0 <= x_min < float(image_width)
        and 0.0 < x_max <= float(image_width)
        and 0.0 <= y_min < float(image_height)
        and 0.0 < y_max <= float(image_height)
    )


def detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "id": det.detection_id,
        "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
        "center_xy": [float(v) for v in det.center_xy],
        "score": float(det.score),
        "coarse_type": det.coarse_type,
        "detector_class": det.detector_class,
        "raw": dict(det.raw),
    }


def detection_array_to_dict(det_array: DetectionArray) -> dict[str, Any]:
    return {
        "schema_version": det_array.schema_version,
        "source_detector": det_array.source_detector,
        "image": {
            "path": det_array.image_path,
            "width": int(det_array.image_width),
            "height": int(det_array.image_height),
        },
        "detector_latency_ms": float(det_array.detector_latency_ms),
        "detections": [detection_to_dict(d) for d in det_array.detections],
    }


def write_contract_json(path: str | Path, det_array: DetectionArray, *, indent: int = 2) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = detection_array_to_dict(det_array)
    out.write_text(json.dumps(payload, indent=indent) + "\n", encoding="utf-8")
    return out


def validate_detection_array(det_array: DetectionArray) -> list[str]:
    errors: list[str] = []

    if det_array.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}, got {det_array.schema_version!r}")

    if det_array.source_detector not in BACKENDS:
        errors.append(
            f"source_detector must be one of {sorted(BACKENDS)}, got {det_array.source_detector!r}"
        )

    if det_array.image_width <= 0:
        errors.append(f"image_width must be positive, got {det_array.image_width}")
    if det_array.image_height <= 0:
        errors.append(f"image_height must be positive, got {det_array.image_height}")

    if not isinstance(det_array.detector_latency_ms, (int, float)):
        errors.append("detector_latency_ms must be numeric")
    elif float(det_array.detector_latency_ms) < 0.0 and float(det_array.detector_latency_ms) != -1.0:
        errors.append("detector_latency_ms must be >= 0 or -1.0")

    for idx, det in enumerate(det_array.detections):
        prefix = f"detections[{idx}]"
        if not det.detection_id:
            errors.append(f"{prefix}: detection_id must not be empty")
        if not is_valid_box(det.bbox_xyxy, det_array.image_width, det_array.image_height):
            errors.append(f"{prefix}: invalid bbox_xyxy {list(det.bbox_xyxy)}")
        expected_center = compute_center(det.bbox_xyxy)
        if (
            abs(det.center_xy[0] - expected_center[0]) > _CENTER_EPS
            or abs(det.center_xy[1] - expected_center[1]) > _CENTER_EPS
        ):
            errors.append(
                f"{prefix}: center_xy {list(det.center_xy)} "
                f"does not match bbox center {list(expected_center)}"
            )
        if det.score != -1.0 and not (0.0 <= float(det.score) <= 1.0):
            errors.append(f"{prefix}: score must be in [0, 1] or -1.0, got {det.score}")
        if det.coarse_type not in COARSE_TYPES:
            errors.append(
                f"{prefix}: coarse_type must be one of {sorted(COARSE_TYPES)}, "
                f"got {det.coarse_type!r}"
            )
        if not isinstance(det.raw, dict):
            errors.append(f"{prefix}: raw must be a dict")

    return errors


def draw_contract_overlay(
    img,
    det_array: DetectionArray,
    title: str,
):
    import cv2
    import numpy as np

    out = img.copy()
    for det in det_array.detections:
        x_min, y_min, x_max, y_max = (int(round(v)) for v in det.bbox_xyxy)
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), _OVERLAY_COLOR, 2)
        cx, cy = (int(round(v)) for v in det.center_xy)
        cv2.circle(out, (cx, cy), 3, _OVERLAY_COLOR, -1)
        label_parts = []
        if det.detector_class:
            label_parts.append(det.detector_class[:20])
        if det.score >= 0.0:
            label_parts.append(f"{det.score:.2f}")
        label = " ".join(label_parts) if label_parts else det.detection_id
        cv2.putText(
            out,
            label,
            (x_min, max(y_min - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            _OVERLAY_COLOR,
            1,
            cv2.LINE_AA,
        )
    if not det_array.detections:
        cv2.putText(
            out,
            "NO DETECTIONS",
            (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _OVERLAY_COLOR,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        title,
        (10, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _OVERLAY_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out
