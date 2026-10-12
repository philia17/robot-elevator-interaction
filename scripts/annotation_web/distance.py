#!/usr/bin/env python3
"""Estimate camera-to-panel distance from an aligned uint16 depth image."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


METHOD = "panel_surface_median"


class DistanceEstimationError(ValueError):
    """Raised when a reliable panel distance cannot be calculated."""


def _bbox_xyxy(box: Any) -> tuple[float, float, float, float]:
    if isinstance(box, Mapping):
        box = box.get("bbox_xyxy", box.get("bbox"))
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        raise DistanceEstimationError(f"invalid bbox: {box!r}")
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise DistanceEstimationError(f"invalid bbox: {box!r}") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise DistanceEstimationError(f"invalid bbox: {box!r}")
    return x1, y1, x2, y2


def _scale_and_clip_boxes(
    bboxes: Sequence[Any],
    depth_shape: tuple[int, int],
    source_size: tuple[int, int] | None,
) -> list[tuple[int, int, int, int]]:
    height, width = depth_shape
    sx = width / source_size[0] if source_size and source_size[0] > 0 else 1.0
    sy = height / source_size[1] if source_size and source_size[1] > 0 else 1.0
    result: list[tuple[int, int, int, int]] = []
    for raw_box in bboxes:
        x1, y1, x2, y2 = _bbox_xyxy(raw_box)
        left = max(0, min(width, int(math.floor(x1 * sx))))
        top = max(0, min(height, int(math.floor(y1 * sy))))
        right = max(0, min(width, int(math.ceil(x2 * sx))))
        bottom = max(0, min(height, int(math.ceil(y2 * sy))))
        if right > left and bottom > top:
            result.append((left, top, right, bottom))
    if bboxes and not result:
        raise DistanceEstimationError("all bboxes fall outside the depth image")
    return result


def _panel_mask(
    shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if boxes:
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        span_x = max(1, x2 - x1)
        span_y = max(1, y2 - y1)
        margin_x = max(8, int(round(span_x * 0.75)))
        margin_y = max(8, int(round(span_y * 0.75)))
        roi_x1 = max(0, x1 - margin_x)
        roi_y1 = max(0, y1 - margin_y)
        roi_x2 = min(width, x2 + margin_x)
        roi_y2 = min(height, y2 + margin_y)
    else:
        # Conservative centre crop for images that have not been annotated yet.
        roi_x1, roi_x2 = int(width * 0.25), int(width * 0.75)
        roi_y1, roi_y2 = int(height * 0.20), int(height * 0.80)
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = True
    for left, top, right, bottom in boxes:
        # One-pixel expansion also removes mixed pixels on a button edge.
        mask[max(0, top - 1) : min(height, bottom + 1), max(0, left - 1) : min(width, right + 1)] = False
    return mask


def calculate_panel_distance(
    depth: np.ndarray,
    depth_scale_m_per_unit: float,
    bboxes: Sequence[Any] | None = None,
    *,
    source_size: tuple[int, int] | None = None,
    min_valid_pixels: int = 16,
) -> dict[str, float | str]:
    """Calculate robust panel distance from a uint16 depth array.

    ``bboxes`` use RGB ``[x1, y1, x2, y2]`` coordinates.  Pass ``source_size``
    as ``(rgb_width, rgb_height)`` when RGB and depth dimensions differ.
    """
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.dtype != np.uint16:
        raise DistanceEstimationError("depth must be a two-dimensional uint16 array")
    try:
        scale = float(depth_scale_m_per_unit)
    except (TypeError, ValueError) as exc:
        raise DistanceEstimationError("depth scale must be a positive finite number") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise DistanceEstimationError("depth scale must be a positive finite number")
    if min_valid_pixels < 1:
        raise DistanceEstimationError("min_valid_pixels must be positive")

    raw_boxes = list(bboxes or [])
    boxes = _scale_and_clip_boxes(raw_boxes, depth.shape, source_size)
    candidate_mask = _panel_mask(depth.shape, boxes)
    candidate_count = int(candidate_mask.sum())
    if candidate_count == 0:
        raise DistanceEstimationError("panel sampling region is empty")
    valid_values = depth[candidate_mask & (depth > 0)].astype(np.float64)
    valid_ratio = float(valid_values.size / candidate_count)
    if valid_values.size < min_valid_pixels:
        raise DistanceEstimationError(
            f"not enough valid panel depth pixels: {valid_values.size} < {min_valid_pixels}"
        )

    # MAD filtering removes isolated foreground/background measurements while
    # preserving a flat panel (MAD == 0) exactly.
    median = float(np.median(valid_values))
    deviations = np.abs(valid_values - median)
    mad = float(np.median(deviations))
    if mad > 0:
        robust_sigma = 1.4826 * mad
        filtered = valid_values[deviations <= 3.5 * robust_sigma]
        if filtered.size >= min_valid_pixels:
            valid_values = filtered
            median = float(np.median(valid_values))

    return {
        "distance_auto_cm": median * scale * 100.0,
        "distance_method": METHOD,
        "distance_valid_ratio": valid_ratio,
    }


def _metadata_scale(metadata: Mapping[str, Any]) -> float:
    for key in ("depth_scale_m_per_unit", "depth_scale_metres_per_stored_unit"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    raise DistanceEstimationError("metadata does not contain a depth scale")


def estimate_panel_distance(
    depth_path: str | Path,
    metadata: Mapping[str, Any],
    bboxes: Sequence[Any] | None = None,
    *,
    min_valid_pixels: int = 16,
) -> dict[str, float | str]:
    """Load a uint16 depth PNG and return the automatic distance fields."""
    path = Path(depth_path)
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise DistanceEstimationError(f"cannot read depth image: {path}")
    source_size: tuple[int, int] | None = None
    width, height = metadata.get("width"), metadata.get("height")
    if width is not None or height is not None:
        try:
            source_width, source_height = int(width), int(height)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DistanceEstimationError("metadata RGB dimensions are invalid") from exc
        if source_width <= 0 or source_height <= 0:
            raise DistanceEstimationError("metadata RGB dimensions are invalid")
        source_size = (source_width, source_height)
    try:
        scale = _metadata_scale(metadata)
    except (TypeError, ValueError) as exc:
        raise DistanceEstimationError("metadata depth scale is invalid") from exc
    return calculate_panel_distance(
        depth,
        scale,
        bboxes,
        source_size=source_size,
        min_valid_pixels=min_valid_pixels,
    )
