"""Shared fixtures for sprint3 contract and adapter tests."""

from __future__ import annotations

import pytest

# Normalized bbox specs from MockDetector (elevator_vision/detectors.py).
_MOCK_SPECS = (
    ("button_01", 0.20, 0.18, 0.36, 0.36, 0.96),
    ("button_02", 0.64, 0.18, 0.80, 0.36, 0.95),
    ("button_03", 0.20, 0.58, 0.36, 0.76, 0.94),
    ("button_04", 0.64, 0.58, 0.80, 0.76, 0.93),
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def mock_xyxy_boxes(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> list[tuple[str, float, float, float, float, float]]:
    """Return MockDetector-style pixel xyxy boxes (id, x_min, y_min, x_max, y_max, score)."""
    boxes: list[tuple[str, float, float, float, float, float]] = []
    for det_id, left, top, right, bottom, confidence in _MOCK_SPECS:
        x_min = float(round(left * image_width, 3))
        y_min = float(round(top * image_height, 3))
        x_max = float(round(right * image_width, 3))
        y_max = float(round(bottom * image_height, 3))
        boxes.append((det_id, x_min, y_min, x_max, y_max, confidence))
    return boxes


def mock_xywh_buttons(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
) -> list[dict]:
    """YOLO-World native button dicts with xywh bbox from MockDetector geometry."""
    buttons: list[dict] = []
    for det_id, x_min, y_min, x_max, y_max, confidence in mock_xyxy_boxes(
        image_width, image_height
    ):
        w = x_max - x_min
        h = y_max - y_min
        buttons.append({
            "bbox": [x_min, y_min, w, h],
            "confidence": confidence,
            "class_name": "button",
            "rel_area": (w * h) / float(image_width * image_height),
        })
    return buttons


@pytest.fixture
def image_size() -> tuple[int, int]:
    return IMAGE_WIDTH, IMAGE_HEIGHT
