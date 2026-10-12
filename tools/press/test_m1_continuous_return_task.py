from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from tools.press.m1_continuous_return_task import (
    ContinuousReturnTask,
    TaskAuthorization,
    TaskError,
    TaskState,
    measured_digest,
)
from tools.press.m1_no_contact_route import REPO_ROOT


ARTIFACT = REPO_ROOT / "config/press/pilots/m1_no_contact_taught_route_v2.json"


def healthy(state="CONVERGED_DISARMED", **changes):
    value = {
        "state": state,
        "armed": False,
        "fault": "",
        "torque_enabled": [True] * 6,
        "voltage": 12.12,
        "joint_target_publisher_count": 0,
    }
    value.update(changes)
    return value


def authorize(task, start, now=100.0, ttl=300.0, token="task-1"):
    task.authorize(
        TaskAuthorization(
            token_id=token,
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(start),
            expires_monotonic=now + ttl,
            confirmation_text="现场安全，连续 return 无接触。",
        ),
        start,
        now,
    )


def physical_pre_contact_start(task):
    # Physical hand-teach record, not the nominal outbound HOLD endpoint.
    pre = task.artifact["endpoints"]["pre_contact_recorded_rad"]
    return tuple(pre)


def test_full_task_runs_51_candidates_and_holds_roll_gripper():
    task = ContinuousReturnTask(ARTIFACT)
    measured = physical_pre_contact_start(task)
    authorize(task, measured)
    assert task.state is TaskState.READY_NEXT_STEP
    count = 0
    while task.state is TaskState.READY_NEXT_STEP:
        candidate = task.issue_next(101.0 + count)
        changed = [
            index
            for index, (before, after) in enumerate(
                zip(candidate.measured_start_rad, candidate.target_positions_rad)
            )
            if abs(after - before) > 0.0015339807878856412 + 1e-12
        ]
        assert changed == [candidate.joint_index]
        assert candidate.target_positions_rad[4] == pytest.approx(measured[4])
        assert candidate.target_positions_rad[5] == pytest.approx(measured[5])
        measured = candidate.target_positions_rad
        task.observe(measured, healthy(), 101.5 + count)
        count += 1
    assert count == 51
    assert task.state is TaskState.COMPLETE_DISARMED
    assert measured[:5] == pytest.approx(
        task.artifact["endpoints"]["press_standby_start_rad"][:5], abs=0.0015339818
    )
    assert measured[4:] == pytest.approx(physical_pre_contact_start(task)[4:])


def test_authorization_is_start_hash_expiry_and_token_bound():
    task = ContinuousReturnTask(ARTIFACT)
    start = physical_pre_contact_start(task)
    bad = TaskAuthorization("bad", "0" * 64, measured_digest(start), 200.0, "ok")
    with pytest.raises(TaskError, match="different route"):
        task.authorize(bad, start, 100.0)
    expired = TaskAuthorization("old", task.sha256, measured_digest(start), 99.0, "ok")
    with pytest.raises(TaskError, match="expired"):
        task.authorize(expired, start, 100.0)
    mismatched = list(start); mismatched[2] += 0.01
    good = TaskAuthorization("good", task.sha256, measured_digest(start), 200.0, "ok")
    with pytest.raises(TaskError, match="different measured"):
        task.authorize(good, mismatched, 100.0)


@pytest.mark.parametrize(
    "status,reason",
    [
        (healthy("FAULT:bad"), "fault"),
        (healthy(armed=True), "did not disarm"),
        (healthy(torque_enabled=[True, True, False, True, True, True]), "torque"),
        (healthy(voltage=10.9), "voltage"),
        (healthy(joint_target_publisher_count=1), "publisher"),
    ],
)
def test_unhealthy_status_latches_abort(status, reason):
    task = ContinuousReturnTask(ARTIFACT)
    start = physical_pre_contact_start(task)
    authorize(task, start)
    candidate = task.issue_next(101.0)
    task.observe(candidate.target_positions_rad, status, 101.1)
    assert task.state is TaskState.ABORT_LATCHED_DISARMED
    assert reason in task.abort_reason


def test_target_or_non_target_error_and_timeout_latch_abort():
    task = ContinuousReturnTask(ARTIFACT, timeout_sec=1.0)
    start = physical_pre_contact_start(task)
    authorize(task, start)
    candidate = task.issue_next(101.0)
    wrong = list(candidate.target_positions_rad); wrong[candidate.joint_index] += 0.01
    task.observe(wrong, healthy(), 101.1)
    assert task.state is TaskState.ABORT_LATCHED_DISARMED
    task = ContinuousReturnTask(ARTIFACT, timeout_sec=1.0)
    authorize(task, start)
    task.issue_next(101.0)
    task.observe(start, healthy("COMMAND_ACTIVE"), 102.1)
    assert task.state is TaskState.ABORT_LATCHED_DISARMED
    assert "timeout" in task.abort_reason


def test_active_driver_may_be_armed_until_converged_and_production_binds_one_owner():
    task = ContinuousReturnTask(ARTIFACT, expected_motion_publisher_count=1)
    start = physical_pre_contact_start(task)
    authorize(task, start)
    candidate = task.issue_next(101.0)
    task.observe(
        start,
        healthy("COMMAND_ACTIVE", armed=True, joint_target_publisher_count=1),
        101.1,
    )
    assert task.state is TaskState.WAIT_DRIVER
    task.observe(
        candidate.target_positions_rad,
        healthy("CONVERGED_DISARMED", joint_target_publisher_count=1),
        101.5,
    )
    assert task.state is TaskState.READY_NEXT_STEP


def test_source_has_no_ros_serial_or_network_imports():
    source = REPO_ROOT / "tools/press/m1_continuous_return_task.py"
    tree = ast.parse(source.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"rclpy", "serial", "socket", "requests"})
    text = source.read_text().lower()
    assert "/roarm_safe/joint_target" not in text
    assert "set_armed" not in text
