import copy
import math
from pathlib import Path

import pytest
import yaml

from roarm_press_sequencer.core import (
    ConfigError,
    ControlState,
    HighLevelState,
    JOINT_NAMES,
    PressMode,
    Sequencer,
    SequencerError,
    TeachCapture,
    load_artifact,
)


def artifact_dict(*, press_qualified=True):
    fingerprint = "fixture-measured-a"
    pose = {
        "qualified": True,
        "measured_at": "2026-07-15T12:00:00+08:00",
        "positions_rad": [0.0] * 6,
        "sample_range_rad": [0.0] * 6,
        "fixture_fingerprint": fingerprint,
        "operator_confirmation_id": "operator-capture-1",
    }
    profile_names = ("base", "shoulder", "elbow", "wrist", "roll", "gripper")
    edges = (
        ("monitor_disarmed", "park"), ("park", "ready"),
        ("ready", "pre_press"), ("pre_press", "visual_5mm_gate"),
        ("visual_5mm_gate", "contact"), ("contact", "press"),
        ("press", "retract"), ("retract", "pre_press"),
        ("pre_press", "ready"), ("ready", "park"),
    )
    return {
        "schema_version": 1,
        "artifact_id": "measured-panel-a-v1",
        "created_at": "2026-07-15T12:00:00+08:00",
        "robot": {
            "model": "roarm_m3", "controller": "t101_single_joint_only",
            "joint_names": list(JOINT_NAMES), "encoder_step_rad": 0.001,
        },
        "fixture": {
            "fingerprint": fingerprint, "arm_mount_marks_mm": [0.0, 0.0],
            "panel_stop_marks_mm": [0.0, 0.0], "panel_plane_x_mm": 320.0,
            "panel_yaw_deg": 0.0, "max_shift_at_2n_mm": 0.5,
        },
        "button": {
            "id": "button-a", "center_base_mm": [320.0, 0.0, 160.0],
            "visible_size_mm": [20.0, 20.0], "travel_mm": 1.5,
            "actuation_force_n": 0.8, "activation_signal": "continuity-a",
        },
        "contact": {
            "frame": "stock_contact", "source": "stock_end_effector",
            "patch_size_mm": [10.0, 10.0], "offset_from_hand_tcp_mm": [0.0, 0.0, 0.0],
            "normal_in_hand_tcp": [1.0, 0.0, 0.0], "gripper_locked_ros_rad": 0.0,
        },
        "driver_profiles": {
            name: {"firmware_joint": index + 1, "qualified": True}
            for index, name in enumerate(profile_names)
        },
        "poses": {name: copy.deepcopy(pose) for name in ("park", "ready", "pre_press", "contact", "press", "retract")},
        "segments": [{
            "from": source, "to": target, "require_human_confirm": True,
            "steps": [{"joint": JOINT_NAMES[0], "target_rad": 0.1,
                       "expected_contact_delta_mm": 0.2, "max_joint_error_deg": 0.5}],
        } for source, target in edges],
        "qualification": {
            "measurements_qualified": press_qualified,
            "contact_qualified": press_qualified,
            "activation_qualified": press_qualified,
            "physical_press_enabled": press_qualified,
            "required_consecutive_press_successes": 10,
        },
    }


def write_artifact(tmp_path, data=None):
    path = tmp_path / "artifact.yaml"
    path.write_text(yaml.safe_dump(data or artifact_dict(), sort_keys=False))
    return path


def load(tmp_path, data=None):
    return load_artifact(write_artifact(tmp_path, data))


def start_transition(seq, now=10.0):
    seq.update_measured(JOINT_NAMES, [0.0] * 6)
    seq.request_transition(HighLevelState.PARKED)
    token = seq.expected_confirmation(now, "confirm-1", "sweep clear")
    return seq.approve_next(token, now)


def test_template_placeholders_fail_closed():
    template = Path(__file__).parents[1] / "config" / "press_artifact.template.yaml"
    with pytest.raises(ConfigError, match="cannot be null"):
        load_artifact(template)


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["robot"].update(joint_names=list(reversed(JOINT_NAMES))), "exact six"),
    (lambda d: d["poses"]["ready"].update(positions_rad=[0.0] * 5), "exactly 6"),
    (lambda d: d["poses"]["ready"].update(positions_rad=[0, 0, math.nan, 0, 0, 0]), "finite"),
    (lambda d: d["poses"]["ready"].update(fixture_fingerprint="wrong"), "metadata/fingerprint"),
    (lambda d: d["driver_profiles"]["base"].update(qualified=False), "not physically qualified"),
])
def test_config_rejections(tmp_path, mutation, match):
    data = artifact_dict()
    mutation(data)
    with pytest.raises(ConfigError, match=match):
        load(tmp_path, data)


def test_forbids_vendor_home_and_t102_text(tmp_path):
    for value in ("vendor home", "T:102"):
        data = artifact_dict()
        data["artifact_id"] = value
        with pytest.raises(ConfigError, match="prohibited"):
            load(tmp_path, data)


def test_state_transition_requires_gate_and_inspection(tmp_path):
    seq = Sequencer(load(tmp_path), dry_run=True)
    start_transition(seq)
    assert seq.control is ControlState.WAIT_DRIVER
    seq.driver_status("CONVERGED_DISARMED", 11.0)
    assert seq.state is HighLevelState.PARKED
    assert seq.control is ControlState.HUMAN_INSPECTION
    with pytest.raises(SequencerError):
        seq.complete_inspection("")
    seq.complete_inspection("pose and sweep normal")
    assert seq.control is ControlState.STABLE_DISARMED


def test_rejects_skip_transition(tmp_path):
    seq = Sequencer(load(tmp_path))
    with pytest.raises(SequencerError, match="not allowed"):
        seq.request_transition(HighLevelState.PRE_PRESS)


def test_confirmation_expired_reused_and_feedback_bound(tmp_path):
    seq = Sequencer(load(tmp_path), dry_run=True)
    seq.update_measured(JOINT_NAMES, [0.0] * 6)
    seq.request_transition(HighLevelState.PARKED)
    expired = seq.expected_confirmation(10.0, "old", "clear", ttl_sec=1.0)
    with pytest.raises(SequencerError, match="expired"):
        seq.approve_next(expired, 12.0)
    token = seq.expected_confirmation(12.0, "fresh", "clear")
    seq.update_measured(JOINT_NAMES, [0.01, 0, 0, 0, 0, 0])
    with pytest.raises(SequencerError, match="bound"):
        seq.approve_next(token, 12.1)


def test_one_joint_target_uses_latest_measured_snapshot(tmp_path):
    seq = Sequencer(load(tmp_path), dry_run=False, execution_enabled=True)
    measured = [0.0, -0.2, 0.3, -0.4, 0.5, 0.6]
    seq.update_measured(JOINT_NAMES, measured)
    seq.request_transition(HighLevelState.PARKED)
    token = seq.expected_confirmation(2.0, "one", "clear")
    target = seq.approve_next(token, 2.0)
    assert target.changed_joint == JOINT_NAMES[0]
    assert target.positions == pytest.approx([0.1, *measured[1:]])
    assert sum(abs(a - b) > 0.001 for a, b in zip(measured, target.positions)) == 1


def test_explicit_ordered_steps_and_fresh_gate_each_step(tmp_path):
    data = artifact_dict()
    data["segments"][0]["steps"] = [
        {"joint": JOINT_NAMES[0], "target_rad": 0.1, "expected_contact_delta_mm": 0.2, "max_joint_error_deg": 0.5},
        {"joint": JOINT_NAMES[1], "target_rad": -0.1, "expected_contact_delta_mm": 0.2, "max_joint_error_deg": 0.5},
    ]
    seq = Sequencer(load(tmp_path, data), dry_run=False, execution_enabled=True)
    first = start_transition(seq)
    assert first.changed_joint == JOINT_NAMES[0]
    seq.update_measured(JOINT_NAMES, [0.1, 0, 0, 0, 0, 0])
    seq.driver_status("CONVERGED_DISARMED", 11.0)
    assert seq.control is ControlState.AWAIT_HUMAN_CONFIRM
    second_token = seq.expected_confirmation(11.0, "confirm-2", "second clear")
    second = seq.approve_next(second_token, 11.0)
    assert second.changed_joint == JOINT_NAMES[1]
    assert second.positions == pytest.approx([0.1, -0.1, 0, 0, 0, 0])


def test_fault_aborts_without_retry_or_compensation(tmp_path):
    seq = Sequencer(load(tmp_path), dry_run=True)
    start_transition(seq)
    seq.driver_status("FAULT_NON_TARGET_DRIFT", 11.0)
    assert seq.state is HighLevelState.ABORT_LATCHED_DISARMED
    assert seq.pending is None
    record = seq.audit.records[-1]
    assert record["retry_count"] == record["compensation_count"] == 0


def test_timeout_aborts_and_latches(tmp_path):
    seq = Sequencer(load(tmp_path), dry_run=True, timeout_sec=2.0)
    start_transition(seq, now=10.0)
    seq.poll(12.1)
    assert seq.state is HighLevelState.ABORT_LATCHED_DISARMED
    with pytest.raises(SequencerError, match="latched"):
        seq.request_transition(HighLevelState.PARKED)


def test_operator_abort_resets_success_counter(tmp_path):
    seq = Sequencer(load(tmp_path))
    seq.state = HighLevelState.PARKED
    seq.record_cycle(True, activation_evidence="continuity-1")
    seq.abort("operator doubt")
    assert seq.consecutive_press_successes == 0


def test_press_mode_disabled_until_all_flags_qualified(tmp_path):
    seq = Sequencer(load(tmp_path, artifact_dict(press_qualified=False)))
    with pytest.raises(SequencerError, match="PRESS disabled"):
        seq.set_mode(PressMode.PRESS)


def test_ten_consecutive_cycle_accounting_and_reset(tmp_path):
    seq = Sequencer(load(tmp_path))
    seq.state = HighLevelState.PARKED
    for index in range(9):
        assert seq.record_cycle(True, activation_evidence=f"signal-{index}") == index + 1
    assert not seq.qualified_for_repeatability
    assert seq.record_cycle(False) == 0
    for index in range(10):
        seq.record_cycle(True, activation_evidence=f"signal-b-{index}")
    assert seq.qualified_for_repeatability


def test_teach_capture_measured_only_and_stability_gate():
    capture = TeachCapture(0.001)
    samples = [{name: index * 0.00005 for name in JOINT_NAMES} for index in range(10)]
    result = capture.capture("park", samples, "human-1")
    assert result["source"] == "measured_joint_states_only"
    bad = copy.deepcopy(samples)
    bad[-1][JOINT_NAMES[2]] = 0.01
    with pytest.raises(SequencerError, match="not stable"):
        capture.capture("park", bad, "human-2")


def test_gripper_change_locked_after_ready(tmp_path):
    data = artifact_dict()
    data["segments"][2]["steps"][0]["joint"] = JOINT_NAMES[5]
    with pytest.raises(ConfigError, match="gripper is locked"):
        load(tmp_path, data)
