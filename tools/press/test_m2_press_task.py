"""Tests for the pure M2 multi-joint press task state machine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.press.m1_continuous_return_task import (
    TaskAuthorization,
    TaskError,
    measured_digest,
)
from tools.press.m2_press_task import (
    CONFIRMATION_REQUIRED_BEFORE,
    MultiJointPressTask,
    POST_VERIFY_REQUIRED_OUTCOME,
    PressTaskState,
    WAYPOINT_TOLERANCE_RAD,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTE = REPO_ROOT / "config" / "press" / "pilots" / "m2_multi_joint_route_v2.json"


def _healthy_status(publisher_count: int = 0) -> dict:
    return {
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "fault": "",
        "torque_enabled": [True] * 6,
        "voltage": 12.12,
        "joint_target_publisher_count": publisher_count,
    }


def _standby(task: MultiJointPressTask) -> tuple[float, ...]:
    return tuple(task.artifact["endpoints"]["press_standby_start_rad"])


def _authorized_task(**kwargs) -> MultiJointPressTask:
    task = MultiJointPressTask(ROUTE, **kwargs)
    start = _standby(task)
    task.authorize(
        TaskAuthorization(
            token_id="test-token",
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(start),
            expires_monotonic=9999.0,
            confirmation_text="offline dry run",
        ),
        start,
        100.0,
    )
    return task


def _run_motion(task: MultiJointPressTask, publisher_count: int = 0) -> int:
    """Drive the task through all waypoints with synthetic convergence."""
    count = 0
    now = 101.0
    confirmations = []
    while task.state in (
        PressTaskState.READY_NEXT_STEP,
        PressTaskState.AWAIT_PHASE_CONFIRMATION,
    ):
        if task.state is PressTaskState.AWAIT_PHASE_CONFIRMATION:
            phase = task.next_phase
            task.confirm_phase(phase, f"operator confirms {phase}")
            confirmations.append(phase)
            continue
        candidate = task.issue_next(now)
        if candidate is None:
            continue
        task.observe(
            candidate.target_positions_rad,
            _healthy_status(publisher_count),
            now + 0.5,
        )
        count += 1
        now += 1.0
    assert confirmations == list(CONFIRMATION_REQUIRED_BEFORE)
    return count


# ------------------------------------------------------------------
# happy path
# ------------------------------------------------------------------


def test_full_cycle_reaches_post_verify_then_completes():
    task = _authorized_task()
    count = _run_motion(task)
    assert count == len(task.waypoints) == task.artifact["totals"]["waypoint_count"]
    assert task.state is PressTaskState.AWAIT_POST_VERIFY

    task.record_post_verify(
        {
            "outcome": POST_VERIFY_REQUIRED_OUTCOME,
            "trial_id": "m0b_trial_test_0001",
            "target_label": "1",
        }
    )
    assert task.state is PressTaskState.COMPLETE_VERIFIED
    assert task.post_verify_record["trial_id"] == "m0b_trial_test_0001"


def test_phase_confirmation_gate_blocks_press_and_retract():
    task = _authorized_task()
    # Drive through outbound only
    now = 101.0
    for _ in range(task.artifact["totals"]["outbound_count"]):
        candidate = task.issue_next(now)
        assert candidate is not None
        assert candidate.phase == "outbound"
        task.observe(candidate.target_positions_rad, _healthy_status(), now + 0.5)
        now += 1.0
    # Next phase is press — issue_next must gate on confirmation
    assert task.next_phase == "press"
    result = task.issue_next(now)
    assert result is None
    assert task.state is PressTaskState.AWAIT_PHASE_CONFIRMATION
    task.confirm_phase("press", "operator confirms press")
    candidate = task.issue_next(now)
    assert candidate is not None
    assert candidate.phase == "press"


def test_candidates_are_multi_joint():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    assert candidate is not None
    assert len(candidate.moving_joint_indices) >= 2


# ------------------------------------------------------------------
# artifact validation
# ------------------------------------------------------------------


def test_artifact_flag_tamper_rejected(tmp_path):
    tampered = json.loads(ROUTE.read_text())
    tampered["motion_authorized"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(TaskError, match="motion_authorized"):
        MultiJointPressTask(path)


def test_artifact_schema_mismatch_rejected(tmp_path):
    tampered = json.loads(ROUTE.read_text())
    tampered["schema"] = "something_else"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(TaskError, match="schema"):
        MultiJointPressTask(path)


def test_artifact_phase_order_tamper_rejected(tmp_path):
    tampered = json.loads(ROUTE.read_text())
    # Swap a return waypoint before the press waypoint
    tampered["waypoints"][0]["phase"] = "return"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(TaskError):
        MultiJointPressTask(path)


def test_artifact_retry_policy_tamper_rejected(tmp_path):
    tampered = json.loads(ROUTE.read_text())
    tampered["route_policy"]["automatic_retry"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(TaskError, match="automatic_retry"):
        MultiJointPressTask(path)


# ------------------------------------------------------------------
# authorization
# ------------------------------------------------------------------


def test_authorization_binds_artifact_hash():
    task = MultiJointPressTask(ROUTE)
    start = _standby(task)
    bad = TaskAuthorization(
        "bad", "0" * 64, measured_digest(start), 200.0, "ok"
    )
    with pytest.raises(TaskError, match="different route"):
        task.authorize(bad, start, 100.0)


def test_authorization_requires_standby_start():
    task = MultiJointPressTask(ROUTE)
    off_start = list(_standby(task))
    off_start[1] += 5 * WAYPOINT_TOLERANCE_RAD
    auth = TaskAuthorization(
        "t", task.sha256, measured_digest(off_start), 200.0, "ok"
    )
    with pytest.raises(TaskError, match="press standby start"):
        task.authorize(auth, off_start, 100.0)


def test_authorization_binds_measured_digest():
    task = MultiJointPressTask(ROUTE)
    start = _standby(task)
    other = list(start)
    other[0] += 0.01
    auth = TaskAuthorization(
        "t", task.sha256, measured_digest(other), 200.0, "ok"
    )
    with pytest.raises(TaskError, match="measured start"):
        task.authorize(auth, start, 100.0)


def test_expired_authorization_rejected():
    task = MultiJointPressTask(ROUTE)
    start = _standby(task)
    auth = TaskAuthorization(
        "t", task.sha256, measured_digest(start), 50.0, "ok"
    )
    with pytest.raises(TaskError, match="expired"):
        task.authorize(auth, start, 100.0)


def test_empty_confirmation_text_rejected():
    task = MultiJointPressTask(ROUTE)
    start = _standby(task)
    auth = TaskAuthorization(
        "t", task.sha256, measured_digest(start), 200.0, "   "
    )
    with pytest.raises(TaskError, match="authorization text"):
        task.authorize(auth, start, 100.0)


# ------------------------------------------------------------------
# phase confirmation
# ------------------------------------------------------------------


def test_confirm_wrong_phase_rejected():
    task = _authorized_task()
    now = 101.0
    for _ in range(task.artifact["totals"]["outbound_count"]):
        candidate = task.issue_next(now)
        task.observe(candidate.target_positions_rad, _healthy_status(), now + 0.5)
        now += 1.0
    assert task.issue_next(now) is None  # enters AWAIT_PHASE_CONFIRMATION
    with pytest.raises(TaskError, match="pending phase"):
        task.confirm_phase("retract", "wrong phase")
    # state unchanged; the correct confirmation still works
    task.confirm_phase("press", "ok")
    assert task.state is PressTaskState.READY_NEXT_STEP


def test_confirm_empty_text_rejected():
    task = _authorized_task()
    now = 101.0
    for _ in range(task.artifact["totals"]["outbound_count"]):
        candidate = task.issue_next(now)
        task.observe(candidate.target_positions_rad, _healthy_status(), now + 0.5)
        now += 1.0
    assert task.issue_next(now) is None
    with pytest.raises(TaskError, match="confirmation text"):
        task.confirm_phase("press", "   ")


def test_confirm_outside_wait_state_rejected():
    task = _authorized_task()
    with pytest.raises(TaskError, match="not awaiting"):
        task.confirm_phase("press", "too early")


def test_confirm_unknown_phase_rejected():
    task = _authorized_task()
    with pytest.raises(TaskError, match="does not take"):
        task.confirm_phase("outbound", "nope")


# ------------------------------------------------------------------
# abort paths
# ------------------------------------------------------------------


def test_abort_on_fault_status():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    status = _healthy_status()
    status["fault"] = "test fault"
    task.observe(candidate.target_positions_rad, status, 101.5)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED
    assert "fault" in task.abort_reason


def test_abort_on_torque_flag():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    status = _healthy_status()
    status["torque_enabled"] = [True] * 5 + [False]
    task.observe(candidate.target_positions_rad, status, 101.5)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED


def test_abort_on_voltage():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    status = _healthy_status()
    status["voltage"] = 10.0
    task.observe(candidate.target_positions_rad, status, 101.5)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED


def test_abort_on_publisher_count():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    task.observe(candidate.target_positions_rad, _healthy_status(2), 101.5)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED
    assert "publisher" in task.abort_reason


def test_abort_on_convergence_timeout():
    task = _authorized_task(timeout_sec=5.0)
    candidate = task.issue_next(101.0)
    task.observe(candidate.measured_start_rad, _healthy_status(), 200.0)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED
    assert "timeout" in task.abort_reason


def test_abort_on_target_error():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    off_target = list(candidate.target_positions_rad)
    off_target[2] += 3 * WAYPOINT_TOLERANCE_RAD
    task.observe(off_target, _healthy_status(), 101.5)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED
    assert "converge" in task.abort_reason


def test_abort_on_unexpected_driver_state():
    task = _authorized_task()
    candidate = task.issue_next(101.0)
    status = _healthy_status()
    status["state"] = "SOMETHING_ELSE"
    task.observe(candidate.target_positions_rad, status, 101.5)
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED


def test_no_candidate_after_abort():
    task = _authorized_task()
    task.abort("test")
    with pytest.raises(TaskError):
        task.issue_next(101.0)


# ------------------------------------------------------------------
# post-verify
# ------------------------------------------------------------------


def test_post_verify_failure_latches_abort():
    task = _authorized_task()
    _run_motion(task)
    task.record_post_verify({"outcome": "FAILED", "trial_id": "m0b_x"})
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED
    assert "post-verify failed" in task.abort_reason


def test_post_verify_requires_trial_id():
    task = _authorized_task()
    _run_motion(task)
    with pytest.raises(TaskError, match="trial_id"):
        task.record_post_verify({"outcome": POST_VERIFY_REQUIRED_OUTCOME})


def test_post_verify_wrong_state_rejected():
    task = _authorized_task()
    with pytest.raises(TaskError, match="not awaiting"):
        task.record_post_verify(
            {"outcome": POST_VERIFY_REQUIRED_OUTCOME, "trial_id": "x"}
        )
