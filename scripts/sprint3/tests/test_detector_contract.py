"""Unit tests for detector_contract.py — no model dependencies."""

from __future__ import annotations

import json

import numpy as np
import pytest

from detector_contract import (
    BACKEND_YOLO_WORLD,
    COARSE_BUTTON_CANDIDATE,
    SCHEMA_VERSION,
    Detection,
    DetectionArray,
    clip_xyxy,
    compute_center,
    detection_array_to_dict,
    detection_to_dict,
    draw_contract_overlay,
    is_valid_box,
    validate_detection_array,
    write_contract_json,
    xywh_to_xyxy,
)
from tests.conftest import IMAGE_HEIGHT, IMAGE_WIDTH, mock_xyxy_boxes


def _detection_from_dict(d: dict) -> Detection:
    return Detection(
        detection_id=d["id"],
        bbox_xyxy=tuple(d["bbox_xyxy"]),
        center_xy=tuple(d["center_xy"]),
        score=float(d["score"]),
        coarse_type=d["coarse_type"],
        detector_class=d.get("detector_class", ""),
        raw=dict(d.get("raw", {})),
    )


def _detection_array_from_dict(d: dict) -> DetectionArray:
    return DetectionArray(
        schema_version=d["schema_version"],
        source_detector=d["source_detector"],
        image_path=d["image"]["path"],
        image_width=int(d["image"]["width"]),
        image_height=int(d["image"]["height"]),
        detector_latency_ms=float(d["detector_latency_ms"]),
        detections=[_detection_from_dict(det) for det in d["detections"]],
    )


def _make_detection(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    detection_id: str = "test_000",
    score: float = 0.9,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        bbox_xyxy=bbox_xyxy,
        center_xy=compute_center(bbox_xyxy),
        score=score,
        coarse_type=COARSE_BUTTON_CANDIDATE,
        detector_class="button",
        raw={"source": "test"},
    )


def _make_array(detections: list[Detection] | None = None) -> DetectionArray:
    return DetectionArray(
        schema_version=SCHEMA_VERSION,
        source_detector=BACKEND_YOLO_WORLD,
        image_path="mock.jpg",
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        detector_latency_ms=12.5,
        detections=detections or [],
    )


class TestXywhToXyxy:
    def test_converts_correctly(self):
        assert xywh_to_xyxy((10.0, 20.0, 30.0, 40.0)) == (10.0, 20.0, 40.0, 60.0)

    def test_accepts_list(self):
        assert xywh_to_xyxy([0.0, 0.0, 100.0, 50.0]) == (0.0, 0.0, 100.0, 50.0)


class TestClipXyxy:
    def test_in_bounds_unchanged(self):
        box = (100.0, 80.0, 200.0, 160.0)
        assert clip_xyxy(box, IMAGE_WIDTH, IMAGE_HEIGHT) == box

    def test_partial_out_of_bounds_clipped(self):
        clipped = clip_xyxy((-10.0, 50.0, 700.0, 500.0), IMAGE_WIDTH, IMAGE_HEIGHT)
        assert clipped == (0.0, 50.0, float(IMAGE_WIDTH), float(IMAGE_HEIGHT))

    def test_fully_out_of_bounds_becomes_invalid(self):
        """clip_xyxy always returns a tuple; fully-OOB boxes fail is_valid_box after clip."""
        clipped = clip_xyxy((-200.0, -200.0, -100.0, -100.0), IMAGE_WIDTH, IMAGE_HEIGHT)
        assert clipped == (0.0, 0.0, 0.0, 0.0)
        assert is_valid_box(clipped, IMAGE_WIDTH, IMAGE_HEIGHT) is False


class TestIsValidBox:
    def test_valid_box(self):
        assert is_valid_box((10.0, 10.0, 50.0, 50.0), IMAGE_WIDTH, IMAGE_HEIGHT) is True

    def test_zero_area_rejected(self):
        assert is_valid_box((100.0, 100.0, 100.0, 150.0), IMAGE_WIDTH, IMAGE_HEIGHT) is False

    def test_x_min_ge_x_max_rejected(self):
        assert is_valid_box((200.0, 50.0, 100.0, 150.0), IMAGE_WIDTH, IMAGE_HEIGHT) is False

    def test_mock_detector_boxes_are_valid(self):
        for _, x_min, y_min, x_max, y_max, _ in mock_xyxy_boxes():
            assert is_valid_box((x_min, y_min, x_max, y_max), IMAGE_WIDTH, IMAGE_HEIGHT) is True


class TestComputeCenter:
    def test_center_of_box(self):
        assert compute_center((100.0, 80.0, 200.0, 160.0)) == (150.0, 120.0)

    def test_mock_detector_centers(self):
        for _, x_min, y_min, x_max, y_max, _ in mock_xyxy_boxes():
            assert compute_center((x_min, y_min, x_max, y_max)) == (
                (x_min + x_max) / 2.0,
                (y_min + y_max) / 2.0,
            )


class TestDetectionRoundTrip:
    def test_detection_dict_round_trip(self):
        original = _make_detection((128.0, 86.4, 230.4, 172.8), detection_id="yolo_world_000")
        rebuilt = _detection_from_dict(detection_to_dict(original))
        assert rebuilt.detection_id == original.detection_id
        assert rebuilt.bbox_xyxy == original.bbox_xyxy
        assert rebuilt.center_xy == original.center_xy
        assert rebuilt.score == original.score
        assert rebuilt.coarse_type == original.coarse_type
        assert rebuilt.detector_class == original.detector_class
        assert rebuilt.raw == original.raw


class TestDetectionArraySerialization:
    def test_round_trip_with_detections(self):
        dets = [
            _make_detection(
                (x_min, y_min, x_max, y_max),
                detection_id=f"yolo_world_{idx:03d}",
                score=score,
            )
            for idx, (_, x_min, y_min, x_max, y_max, score) in enumerate(mock_xyxy_boxes())
        ]
        original = _make_array(dets)
        payload = detection_array_to_dict(original)
        json_text = json.dumps(payload)
        rebuilt = _detection_array_from_dict(json.loads(json_text))

        assert rebuilt.schema_version == original.schema_version
        assert rebuilt.source_detector == original.source_detector
        assert rebuilt.image_path == original.image_path
        assert rebuilt.image_width == original.image_width
        assert rebuilt.image_height == original.image_height
        assert rebuilt.detector_latency_ms == original.detector_latency_ms
        assert len(rebuilt.detections) == len(original.detections)
        for orig_det, rebuilt_det in zip(original.detections, rebuilt.detections):
            assert detection_to_dict(rebuilt_det) == detection_to_dict(orig_det)

    def test_empty_detections_round_trip(self, tmp_path):
        original = _make_array([])
        out_path = tmp_path / "empty.json"
        write_contract_json(out_path, original)
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        rebuilt = _detection_array_from_dict(payload)
        assert rebuilt.detections == []
        assert validate_detection_array(rebuilt) == []


class TestValidateDetectionArray:
    def test_empty_list_is_valid(self):
        assert validate_detection_array(_make_array([])) == []

    def test_valid_mock_detections_pass(self):
        dets = [
            _make_detection(
                (x_min, y_min, x_max, y_max),
                detection_id=f"yolo_world_{idx:03d}",
                score=score,
            )
            for idx, (_, x_min, y_min, x_max, y_max, score) in enumerate(mock_xyxy_boxes())
        ]
        assert validate_detection_array(_make_array(dets)) == []

    def test_missing_detection_id_reported(self):
        det = _make_detection((50.0, 50.0, 100.0, 100.0), detection_id="")
        errors = validate_detection_array(_make_array([det]))
        assert any("detection_id must not be empty" in err for err in errors)

    def test_invalid_bbox_reported(self):
        det = _make_detection((0.0, 0.0, 0.0, 0.0))
        errors = validate_detection_array(_make_array([det]))
        assert any("invalid bbox_xyxy" in err for err in errors)

    def test_mismatched_center_reported(self):
        det = _make_detection((50.0, 50.0, 100.0, 100.0))
        det.center_xy = (999.0, 999.0)
        errors = validate_detection_array(_make_array([det]))
        assert any("center_xy" in err and "does not match bbox center" in err for err in errors)

    def test_score_out_of_range_reported(self):
        det = _make_detection((50.0, 50.0, 100.0, 100.0), score=1.5)
        errors = validate_detection_array(_make_array([det]))
        assert any("score must be in [0, 1] or -1.0" in err for err in errors)

    def test_score_minus_one_allowed(self):
        det = _make_detection((50.0, 50.0, 100.0, 100.0), score=-1.0)
        assert validate_detection_array(_make_array([det])) == []

    def test_wrong_schema_version_reported(self):
        arr = _make_array([])
        arr.schema_version = "0.9"
        errors = validate_detection_array(arr)
        assert any("schema_version must be" in err for err in errors)


class TestDrawContractOverlay:
    def test_blank_image_does_not_raise(self):
        img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        det_array = _make_array([
            _make_detection((128.0, 86.4, 230.4, 172.8)),
        ])
        overlay = draw_contract_overlay(img, det_array, title="test overlay")
        assert overlay.shape == img.shape

    def test_empty_detections_on_blank_image(self):
        img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        overlay = draw_contract_overlay(img, _make_array([]), title="empty")
        assert overlay.shape == img.shape
