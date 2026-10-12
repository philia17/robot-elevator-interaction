"""Unit tests for sprint3 detector adapters — no model dependencies."""

from __future__ import annotations

import numpy as np
import pytest

from adapter_traditional_yolo import adapt_traditional_yolo
from detector_contract import (
    BACKEND_TRADITIONAL_YOLO,
    COARSE_BUTTON_CANDIDATE,
    compute_center,
    validate_detection_array,
)
from tests.conftest import IMAGE_HEIGHT, IMAGE_WIDTH, mock_xyxy_boxes


IMAGE_PATH = "mock_capture.jpg"
LATENCY_MS = 25.0


class _MockYoloBoxes:
    """Minimal Ultralytics Results.boxes stand-in for adapter tests."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class TestAdaptTraditionalYolo:
    def test_empty_boxes(self):
        result = adapt_traditional_yolo(
            None, {0: "button"}, IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, LATENCY_MS
        )
        assert result.detections == []
        assert result.source_detector == BACKEND_TRADITIONAL_YOLO
        assert validate_detection_array(result) == []

    def test_xyxy_pass_through_and_cls_mapping(self):
        boxes = _MockYoloBoxes(
            xyxy=np.array([[100.0, 80.0, 200.0, 160.0]], dtype=np.float32),
            conf=np.array([0.91], dtype=np.float32),
            cls=np.array([0], dtype=np.float32),
        )
        names = {0: "elevator_button"}
        result = adapt_traditional_yolo(
            boxes, names, IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, LATENCY_MS
        )
        det = result.detections[0]
        assert det.bbox_xyxy == (100.0, 80.0, 200.0, 160.0)
        assert det.center_xy == compute_center(det.bbox_xyxy)
        assert det.score == pytest.approx(0.91)
        assert det.detector_class == "elevator_button"
        assert det.coarse_type == COARSE_BUTTON_CANDIDATE
        assert det.raw["class_id"] == 0

    def test_mock_detector_geometry(self):
        rows = []
        confs = []
        clss = []
        for idx, (_, x_min, y_min, x_max, y_max, score) in enumerate(mock_xyxy_boxes()):
            rows.append([x_min, y_min, x_max, y_max])
            confs.append(score)
            clss.append(idx % 2)
        boxes = _MockYoloBoxes(
            xyxy=np.array(rows, dtype=np.float32),
            conf=np.array(confs, dtype=np.float32),
            cls=np.array(clss, dtype=np.float32),
        )
        names = {0: "button_a", 1: "button_b"}
        result = adapt_traditional_yolo(
            boxes,
            names,
            IMAGE_PATH,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            LATENCY_MS,
            model_path="/models/yolo11n.onnx",
        )
        assert len(result.detections) == 4
        assert all(det.coarse_type == COARSE_BUTTON_CANDIDATE for det in result.detections)
        assert result.detections[0].raw["model_path"] == "/models/yolo11n.onnx"
        assert validate_detection_array(result) == []

    def test_invalid_box_dropped(self):
        boxes = _MockYoloBoxes(
            xyxy=np.array([
                [100.0, 100.0, 100.0, 150.0],
                [120.0, 120.0, 180.0, 180.0],
            ], dtype=np.float32),
            conf=np.array([0.5, 0.6], dtype=np.float32),
            cls=np.array([0, 0], dtype=np.float32),
        )
        result = adapt_traditional_yolo(
            boxes, {0: "button"}, IMAGE_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, LATENCY_MS
        )
        assert len(result.detections) == 1
