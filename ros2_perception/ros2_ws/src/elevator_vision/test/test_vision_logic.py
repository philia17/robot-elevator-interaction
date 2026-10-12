"""CPU-only tests for frame metrics, Mock detections, and SoM mapping."""

import numpy as np

from elevator_vision.detectors import MockDetector
from elevator_vision.frame_metrics import FrameGate, frame_motion, mean_brightness
from elevator_vision.models import Detection
from elevator_vision.som import generate_som, stable_detection_order


def _image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[40:440, 80:560] = 150
    image[100:180, 130:230] = 245
    return image


def test_mock_detector_is_deterministic_and_has_stable_ids():
    detector = MockDetector()
    first = detector.detect(_image())
    second = detector.detect(_image())
    assert first == second
    assert [item.detection_id for item in first] == [
        'button_01', 'button_02', 'button_03', 'button_04'
    ]


def test_mock_detector_geometry_and_confidence_are_valid():
    detections = MockDetector().detect(_image())
    for item in detections:
        assert 0.0 <= item.x_min < item.x_max <= 640.0
        assert 0.0 <= item.y_min < item.y_max <= 480.0
        assert item.center_x == (item.x_min + item.x_max) / 2.0
        assert item.center_y == (item.y_min + item.y_max) / 2.0
        assert 0.0 <= item.detector_confidence <= 1.0
        assert item.coarse_type == 'button_candidate'
        assert item.coarse_type not in {'1', '2', '3B', 'door_open'}


def test_som_order_is_center_y_then_center_x_then_id():
    detections = [
        Detection('z', 0, 0, 10, 10, 5, 20, 0.9, 'button'),
        Detection('b', 0, 0, 10, 10, 5, 10, 0.9, 'button'),
        Detection('a', 0, 0, 10, 10, 5, 10, 0.9, 'button'),
        Detection('x', 0, 0, 10, 10, 2, 10, 0.9, 'button'),
    ]
    assert [item.detection_id for item in stable_detection_order(detections)] == [
        'x', 'a', 'b', 'z'
    ]


def test_som_marker_mapping_and_geometry_preservation():
    detections = MockDetector().detect(_image())
    original_boxes = [item.bbox for item in detections]
    result = generate_som(_image(), list(reversed(detections)))
    assert result.marker_to_detection_id == {
        'A': 'button_01',
        'B': 'button_02',
        'C': 'button_03',
        'D': 'button_04',
    }
    assert [item.bbox for item in detections] == original_boxes
    assert result.annotated_image.shape == _image().shape
    assert not np.array_equal(result.annotated_image, _image())
    assert [item.detection_id for item in result.crops] == [
        'button_01', 'button_02', 'button_03', 'button_04'
    ]


def test_frame_metrics_and_minimum_interval():
    image = _image()
    assert mean_brightness(image) > 0.0
    assert frame_motion(image, image) == 0.0
    gate = FrameGate(
        output_width=320,
        output_height=240,
        min_sharpness=0.0,
        min_brightness=0.0,
        max_brightness=255.0,
        max_motion=255.0,
        min_interval_sec=1.0,
    )
    first = gate.evaluate(image, 10.0)
    second = gate.evaluate(image, 10.5)
    assert first.accepted
    assert first.resized_image.shape == (240, 320, 3)
    assert not second.accepted
    assert second.reason == 'minimum_interval'


def test_frame_gate_can_measure_low_resolution_and_publish_high_resolution():
    image = _image()
    gate = FrameGate(
        output_width=640,
        output_height=360,
        metric_width=320,
        metric_height=180,
        min_sharpness=0.0,
        min_brightness=0.0,
        max_brightness=255.0,
        max_motion=255.0,
        min_interval_sec=0.0,
    )

    result = gate.evaluate(image, 10.0)

    assert result.accepted
    assert result.resized_image.shape == (360, 640, 3)
    assert gate._previous_frame.shape == (180, 320, 3)
