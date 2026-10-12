"""Traditional YOLO Ultralytics Results.boxes → shared detector contract.

Pure Python adapter — no torch or ultralytics imports. Callers pass already-
inferred ``Results.boxes`` objects; this module only reads ``.xyxy``, ``.conf``,
and ``.cls``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from detector_contract import (
    BACKEND_TRADITIONAL_YOLO,
    COARSE_BUTTON_CANDIDATE,
    SCHEMA_VERSION,
    Detection,
    DetectionArray,
    clip_xyxy,
    compute_center,
    is_valid_box,
)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _class_name(names: Mapping[int, str] | Sequence[str] | Any, cls_id: int) -> str:
    if isinstance(names, Mapping):
        if cls_id in names:
            return str(names[cls_id])
        return str(names.get(str(cls_id), ""))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return ""


def adapt_traditional_yolo(
    boxes: Any,
    names: Mapping[int, str] | Sequence[str] | Any,
    image_path: str,
    image_width: int,
    image_height: int,
    detector_latency_ms: float,
    *,
    model_path: str = "",
) -> DetectionArray:
    """Convert Ultralytics ``Results.boxes`` into a ``DetectionArray``."""
    detections: list[Detection] = []
    next_id = 0

    if boxes is not None:
        xyxy_arr = _tensor_to_numpy(boxes.xyxy)
        conf_arr = _tensor_to_numpy(boxes.conf)
        cls_arr = _tensor_to_numpy(boxes.cls)

        if xyxy_arr.ndim == 1 and xyxy_arr.size == 4:
            xyxy_arr = xyxy_arr.reshape(1, 4)
        if conf_arr.ndim == 0:
            conf_arr = conf_arr.reshape(1)
        if cls_arr.ndim == 0:
            cls_arr = cls_arr.reshape(1)

        count = int(xyxy_arr.shape[0]) if xyxy_arr.ndim == 2 else 0
        for idx in range(count):
            original_xyxy = tuple(float(v) for v in xyxy_arr[idx, :4])
            clipped_xyxy = clip_xyxy(original_xyxy, image_width, image_height)
            if not is_valid_box(clipped_xyxy, image_width, image_height):
                continue

            cls_id = int(cls_arr[idx]) if idx < len(cls_arr) else -1
            class_name = _class_name(names, cls_id)
            score = float(conf_arr[idx]) if idx < len(conf_arr) else -1.0

            raw: dict[str, Any] = {
                "class_id": cls_id,
                "class_name": class_name,
                "native_bbox_xyxy": [float(v) for v in original_xyxy],
            }
            if model_path:
                raw["model_path"] = model_path

            detections.append(Detection(
                detection_id=f"traditional_yolo_{next_id:03d}",
                bbox_xyxy=clipped_xyxy,
                center_xy=compute_center(clipped_xyxy),
                score=score,
                coarse_type=COARSE_BUTTON_CANDIDATE,
                detector_class=class_name,
                raw=raw,
            ))
            next_id += 1

    return DetectionArray(
        schema_version=SCHEMA_VERSION,
        source_detector=BACKEND_TRADITIONAL_YOLO,
        image_path=image_path,
        image_width=int(image_width),
        image_height=int(image_height),
        detector_latency_ms=float(detector_latency_ms),
        detections=detections,
    )
