"""CPU-only tests for fusion, floor mapping, resolution, and tracking."""

from dataclasses import replace
from pathlib import Path

import pytest

from elevator_pipeline.floor_map import (
    FloorMap,
    MissingBuildingError,
    MissingFloorError,
    normalize_label,
)
from elevator_pipeline.fusion import (
    FusionValidationError,
    Observation,
    fuse_detections_and_semantics,
    validate_bbox,
    validate_image_bounds,
)
from elevator_pipeline.resolver import Candidate, TaskSpec, resolve_target
from elevator_pipeline.temporal import TemporalTracker, bbox_iou, center_distance
from elevator_semantics.vlm import MockVlm, SemanticResult
from elevator_vision.detectors import MockDetector


class FakeImage:
    shape = (480, 640, 3)


def _map_path():
    return (
        Path(__file__).resolve().parents[2]
        / 'elevator_bringup'
        / 'config'
        / 'building_floor_map.yaml'
    )


def _mock_data():
    detections = MockDetector().detect(FakeImage())
    semantics = MockVlm().label([item.detection_id for item in detections])
    observations = fuse_detections_and_semantics(
        detections, semantics, 640, 480
    )
    return detections, semantics, observations


def _task(request_id='request-1'):
    return TaskSpec(True, request_id, 'building_a', '4')


def _target_observation():
    return _mock_data()[2][2]


def test_bbox_validation():
    detection = _mock_data()[0][0]
    validate_bbox(detection)
    with pytest.raises(FusionValidationError, match='positive'):
        validate_bbox(replace(detection, x_max=detection.x_min))


def test_image_bound_validation():
    detection = _mock_data()[0][0]
    validate_image_bounds(detection, 640, 480)
    with pytest.raises(FusionValidationError, match='bounds'):
        validate_image_bounds(replace(detection, x_max=641.0), 640, 480)


def test_confidence_validation():
    detections, semantics, _ = _mock_data()
    invalid = replace(detections[0], detector_confidence=1.1)
    with pytest.raises(FusionValidationError, match='detector_confidence'):
        fuse_detections_and_semantics(
            [invalid] + detections[1:], semantics, 640, 480
        )


def test_detection_id_uniqueness():
    detections, semantics, _ = _mock_data()
    with pytest.raises(FusionValidationError, match='duplicate detector'):
        fuse_detections_and_semantics(
            detections + [detections[0]], semantics, 640, 480
        )


def test_unknown_semantic_id_is_rejected_by_fusion():
    detections, semantics, _ = _mock_data()
    unknown = SemanticResult('unknown', '9', 'floor_button', True, 0.8)
    with pytest.raises(FusionValidationError, match='unknown semantic'):
        fuse_detections_and_semantics(
            detections, semantics + [unknown], 640, 480
        )


def test_missing_semantic_marks_observation_invalid():
    detections, semantics, _ = _mock_data()
    observations = fuse_detections_and_semantics(
        detections, semantics[:-1], 640, 480
    )
    assert not observations[-1].valid
    assert observations[-1].validation_reason == 'missing_semantic'


def test_fusion_association_geometry_and_confidence():
    detections, _, observations = _mock_data()
    by_id = {item.detection_id: item for item in observations}
    target = by_id['button_03']
    detection = detections[2]
    assert target.label == '3B'
    assert target.bbox == detection.bbox
    assert target.center_x == detection.center_x
    assert target.center_y == detection.center_y
    assert target.overall_confidence == pytest.approx((0.94 + 0.90) / 2.0)


@pytest.mark.parametrize('raw, expected', [
    ('G', 'G'),
    ('GF', 'G'),
    ('GROUND', 'G'),
    ('B1', 'B1'),
    ('B-1', 'B1'),
    ('3B', '3B'),
    ('03B', '3B'),
    ('PH', 'PH'),
    ('PENTHOUSE', 'PH'),
    ('door_open', 'DOOR_OPEN'),
    ('door_close', 'DOOR_CLOSE'),
])
def test_label_normalization(raw, expected):
    assert normalize_label(raw) == expected


def test_floor_map_loading_and_floor_4_mapping():
    floor_map = FloorMap.load(_map_path())
    entry = floor_map.get_floor('building_a', '4')
    assert entry.displayed_label == '3B'
    assert floor_map.matches('Floor 4', entry)


def test_missing_building_error():
    with pytest.raises(MissingBuildingError, match='missing building'):
        FloorMap.load(_map_path()).get_floor('unknown', '4')


def test_missing_floor_error():
    with pytest.raises(MissingFloorError, match='missing floor'):
        FloorMap.load(_map_path()).get_floor('building_a', '99')


def test_target_selection_picks_button_03():
    candidate = resolve_target(
        _task(),
        _mock_data()[2],
        FloorMap.load(_map_path()),
        minimum_confidence=0.5,
    )
    assert candidate.found
    assert candidate.effective_label == '3B'
    assert candidate.button.detection_id == 'button_03'


@pytest.mark.parametrize('replacement', [
    {'pressable': False},
    {'button_type': 'door_open'},
    {'overall_confidence': 0.2},
])
def test_target_exclusions(replacement):
    target = replace(_target_observation(), **replacement)
    candidate = resolve_target(
        _task(),
        [target],
        FloorMap.load(_map_path()),
        minimum_confidence=0.5,
    )
    assert not candidate.found


def test_iou_and_center_distance():
    target = _target_observation()
    assert bbox_iou(target, target) == 1.0
    assert center_distance(target, target) == 0.0
    shifted = replace(
        target,
        x_min=target.x_min + 10.0,
        x_max=target.x_max + 10.0,
        center_x=target.center_x + 10.0,
    )
    assert 0.0 < bbox_iou(target, shifted) < 1.0
    assert center_distance(target, shifted) == 10.0


def test_temporal_consistency_and_confirmation_count():
    candidate = resolve_target(
        _task(), _mock_data()[2], FloorMap.load(_map_path()), 0.5
    )
    tracker = TemporalTracker(required_confirmations=2)
    first = tracker.update(candidate)
    second = tracker.update(candidate)
    assert not first.valid
    assert first.confirmation_count == 1
    assert second.valid
    assert second.confirmation_count == 2
    assert second.status == 'STABLE'
    assert second.effective_label == '3B'
    assert second.button.detection_id == 'button_03'


def test_task_change_resets_confirmation_history():
    floor_map = FloorMap.load(_map_path())
    observations = _mock_data()[2]
    first_candidate = resolve_target(_task('first'), observations, floor_map, 0.5)
    second_candidate = resolve_target(_task('second'), observations, floor_map, 0.5)
    tracker = TemporalTracker(required_confirmations=2)
    tracker.update(first_candidate)
    after_reset = tracker.update(second_candidate)
    assert not after_reset.valid
    assert after_reset.confirmation_count == 1


def test_direct_phase_b_logic_proof():
    floor_map = FloorMap.load(_map_path())
    entry = floor_map.get_floor('building_a', '4')
    assert entry.displayed_label == '3B'
    candidate = resolve_target(_task(), _mock_data()[2], floor_map, 0.5)
    assert candidate.button.detection_id == 'button_03'
    tracker = TemporalTracker(required_confirmations=2)
    assert not tracker.update(candidate).valid
    target = tracker.update(candidate)
    assert target.valid
    assert target.status == 'STABLE'
