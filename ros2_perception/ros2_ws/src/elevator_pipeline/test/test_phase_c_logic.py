"""Pure-Python Phase C tests; no ROS graph is required."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from elevator_pipeline.diagnostics_core import DiagnosticsTracker
from elevator_pipeline.floor_map import FloorMap
from elevator_pipeline.manager_core import (
    CANCELLED,
    DETECTING,
    FAILED,
    FUSING,
    IDLE,
    LABELING,
    SELECTING,
    STABILIZING,
    SUCCESS,
    TIMEOUT,
    WAITING_FRAME,
    GoalValidationError,
    ManagerStateMachine,
)
from elevator_pipeline.visualization_core import build_overlay_plan
from elevator_vision.detectors import MockDetector
from elevator_vision.qos import parse_image_qos


class FakeImage:
    shape = (480, 640, 3)


def _project_root():
    return Path(__file__).resolve().parents[4]


def _floor_map():
    return FloorMap.load(
        _project_root()
        / 'ros2_ws/src/elevator_bringup/config/building_floor_map.yaml'
    )


def _manager(ids=None):
    values = iter(ids or ['request-a', 'request-b'])
    return ManagerStateMachine(_floor_map(), id_factory=lambda: next(values))


def _candidate(request_id='request-a', found=True):
    return SimpleNamespace(
        request_id=request_id,
        found=found,
        button=SimpleNamespace(detection_id='button_03'),
    )


def _target(request_id='request-a', valid=True):
    return SimpleNamespace(
        request_id=request_id,
        valid=valid,
        effective_label='3B',
        button=SimpleNamespace(detection_id='button_03'),
    )


@pytest.mark.parametrize('building,floor,timeout', [
    ('', '4', 20.0),
    ('building_a', '', 20.0),
    ('missing', '4', 20.0),
    ('building_a', '99', 20.0),
    ('building_a', '4', 0.0),
    ('building_a', '4', -1.0),
])
def test_goal_validation_rejects_invalid_input(building, floor, timeout):
    with pytest.raises(GoalValidationError):
        _manager().validate_goal(building, floor, timeout)


def test_action_state_transitions_and_successful_completion():
    manager = _manager()
    request_id = manager.start_goal('building_a', '4', 20.0, 100.0)
    assert request_id == 'request-a'
    assert manager.state == WAITING_FRAME
    manager.record_frame(True)
    assert manager.state == DETECTING
    manager.record_detections(4)
    assert manager.state == LABELING
    manager.record_semantics(4)
    assert manager.state == FUSING
    manager.record_observations()
    assert manager.state == SELECTING
    assert manager.record_candidate(_candidate())
    assert manager.state == STABILIZING
    assert manager.record_target(_target())
    snapshot = manager.snapshot()
    assert snapshot.state == SUCCESS
    assert snapshot.target.effective_label == '3B'
    assert snapshot.target.button.detection_id == 'button_03'
    assert snapshot.progress == 1.0


def test_timeout_uses_supplied_monotonic_values():
    manager = _manager()
    manager.start_goal('building_a', '4', 5.0, 10.0)
    assert not manager.tick(14.999)
    assert manager.tick(15.0)
    assert manager.state == TIMEOUT
    assert 'timed out' in manager.snapshot().failure_reason


def test_cancellation_and_task_reset_isolate_future_goal():
    manager = _manager()
    manager.start_goal('building_a', '4', 20.0, 0.0)
    assert manager.cancel()
    assert manager.state == CANCELLED
    manager.reset()
    assert manager.state == IDLE
    next_id = manager.start_goal('building_a', '4', 20.0, 1.0)
    assert next_id == 'request-b'
    assert manager.snapshot().detected_count == 0


def test_stale_candidate_and_target_are_rejected():
    manager = _manager()
    manager.start_goal('building_a', '4', 20.0, 0.0)
    assert not manager.record_candidate(_candidate('old-request'))
    assert not manager.record_target(_target('old-request'))
    assert manager.state == WAITING_FRAME


def test_failure_completion_records_reason():
    manager = _manager()
    manager.start_goal('building_a', '4', 20.0, 0.0)
    assert manager.fail('mapping unavailable')
    assert manager.state == FAILED
    assert manager.snapshot().failure_reason == 'mapping unavailable'


def _observation(detection):
    return SimpleNamespace(
        detection_id=detection.detection_id,
        label='3B' if detection.detection_id == 'button_03' else '1',
        button_type='floor_button',
        pressable=True,
        semantic_confidence=0.90,
        overall_confidence=0.92,
    )


def test_visualization_plan_preserves_detector_geometry_and_marks_target():
    detections = MockDetector().detect(FakeImage())
    observations = [_observation(item) for item in detections]
    plan = build_overlay_plan(
        detections,
        observations,
        _candidate(),
        _target(),
    )
    assert [item.marker for item in plan] == ['A', 'B', 'C', 'D']
    target_item = [item for item in plan if item.detection_id == 'button_03'][0]
    assert target_item.bbox == detections[2].bbox
    assert target_item.color == (0, 255, 0)
    assert '3B' in target_item.lines[0]


def test_visualization_handles_missing_semantics_and_no_target():
    detections = MockDetector().detect(FakeImage())
    plan = build_overlay_plan(detections, [], None, None)
    assert len(plan) == 4
    assert all('?' in item.lines[0] for item in plan)


def test_visualization_handles_empty_detections():
    assert build_overlay_plan([], [], None, None) == ()


def test_diagnostic_health_and_warning_state():
    tracker = DiagnosticsTracker(history_size=4, stale_frame_sec=2.0)
    empty = tracker.snapshot(5.0)
    assert empty.overall_status == 'WARN'
    assert 'no_frame_received' in empty.warnings
    tracker.record_frame(False, 5.0)
    rejected = tracker.snapshot(5.1)
    assert rejected.overall_status == 'WARN'
    assert 'last_frame_rejected' in rejected.warnings
    tracker.record_state('TIMEOUT')
    timeout = tracker.snapshot(5.2)
    assert timeout.overall_status == 'ERROR'
    assert timeout.timeout_count == 1


def test_diagnostic_history_is_bounded_and_calculates_fps():
    tracker = DiagnosticsTracker(history_size=3)
    for timestamp in (0.0, 1.0, 2.0, 3.0, 4.0):
        tracker.record_frame(True, timestamp)
    snapshot = tracker.snapshot(4.0)
    assert tracker.frame_history_size == 3
    assert snapshot.input_fps == pytest.approx(1.0)
    assert snapshot.detector_latency_ms == -1.0
    assert snapshot.fusion_latency_ms == -1.0


def test_qos_configuration_parsing():
    settings = parse_image_qos('reliable', 'transient_local', 7)
    assert settings.reliability == 'reliable'
    assert settings.durability == 'transient_local'
    assert settings.depth == 7
    with pytest.raises(ValueError):
        parse_image_qos('sometimes', 'volatile', 10)
    with pytest.raises(ValueError):
        parse_image_qos('best_effort', 'volatile', 0)


def test_smoke_cleanup_uses_isolated_process_group():
    script = (_project_root() / 'scripts/smoke_test.sh').read_text(encoding='utf-8')
    assert 'setsid ros2 launch' in script
    assert 'LAUNCH_PGID' in script
    assert 'kill -INT -- "-${LAUNCH_PGID}"' in script
    assert 'pkill' not in script
    assert 'killall' not in script
