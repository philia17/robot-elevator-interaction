"""Tests for the inert M1 return adapter — no hardware required.

These tests verify the adapter's fail-closed behavior without a live ROS
domain or serial device.  They import the node class directly and exercise
its validation logic.
"""

from __future__ import annotations

import json
import math

import pytest

# Verify critical invariants in node.py source.
from pathlib import Path


def _find_repo_root() -> Path:
    """Support both the Mac layout (repo/ros2_arm/<pkg>) and the AGX layout
    (roarm_ws/src/lift_robot/<pkg>)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "tools" / "press" / "m1_continuous_return_task.py").is_file():
            return parent
    raise RuntimeError("could not locate repo root containing tools/press")


REPO_ROOT = _find_repo_root()
ADAPTER_SOURCE = (
    Path(__file__).resolve().parents[1] / "roarm_m1_adapter" / "node.py"
)
TOOLS_SOURCE = (
    REPO_ROOT / "tools" / "press" / "m1_continuous_return_task.py"
)

ROUTE_V2 = REPO_ROOT / "config" / "press" / "pilots" / "m1_no_contact_taught_route_v2.json"

from tools.press.m1_continuous_return_task import (
    ContinuousReturnTask,
    TaskAuthorization,
    TaskState,
    measured_digest,
)


# ------------------------------------------------------------------
# Interface contract tests — verify the inert-by-default contract
# is enforced at source level
# ------------------------------------------------------------------


def test_adapter_source_has_no_root_level_target_pub():
    """The adapter must NOT create a publisher at module/class level."""
    text = ADAPTER_SOURCE.read_text()
    # Creating a publisher at module scope would require calling it outside __init__
    # or in __init__ before the auth gate.  Verify the create_publisher call
    # only appears inside the guarded _create_execution_resources method.
    assert "create_publisher" in text
    lines = text.split("\n")
    in_create_resources = False
    publisher_lines = []
    for i, line in enumerate(lines, start=1):
        if "def _create_execution_resources" in line:
            in_create_resources = True
        elif line.startswith("    def ") and in_create_resources:
            in_create_resources = False
        if "create_publisher" in line and not in_create_resources:
            publisher_lines.append(i)
    assert not publisher_lines, (
        f"create_publisher found outside _create_execution_resources at lines {publisher_lines}"
    )


def test_adapter_source_has_no_auto_arm_client():
    """Arm client must only be created inside _create_execution_resources."""
    text = ADAPTER_SOURCE.read_text()
    lines = text.split("\n")
    in_create_resources = False
    client_lines = []
    for i, line in enumerate(lines, start=1):
        if "def _create_execution_resources" in line:
            in_create_resources = True
        elif line.startswith("    def ") and in_create_resources:
            in_create_resources = False
        if "create_client" in line and not in_create_resources:
            client_lines.append(i)
    assert not client_lines, (
        f"create_client found outside _create_execution_resources at lines {client_lines}"
    )


def test_adapter_source_never_calls_torque_or_clear_fault():
    """The adapter must never call Torque, clear_fault, or emit T102."""
    import ast
    tree = ast.parse(ADAPTER_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            else:
                call_name = ""
            assert call_name not in {"torque_on", "torque_off", "clear_fault"}, (
                f"forbidden call '{call_name}' at line {node.lineno}"
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert '"t":102' not in node.value.lower().replace(" ", ""), (
                f"forbidden T102 reference at line {node.lineno}"
            )


def test_adapter_source_never_retries_or_compensates():
    """The adapter must never auto-retry, compensate, or auto-retract."""
    import ast
    tree = ast.parse(ADAPTER_SOURCE.read_text())
    # Check for while/for loops in _disarm or _abort (would indicate retry)
    for node in ast.walk(tree):
        if isinstance(node, (ast.While, ast.For)):
            # Find enclosing function
            for parent in ast.walk(tree):
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(parent, 'body') and any(
                        n is node
                        for stmt in (parent.body if isinstance(parent.body, list) else [parent.body])
                        for n in ast.walk(stmt) if n is node
                    ):
                        func_name = parent.name
                        if func_name in ("_disarm", "_abort"):
                            raise AssertionError(
                                f"retry loop found in {func_name} at line {node.lineno}"
                            )
        # Check for specific call patterns
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            else:
                call_name = ""
            assert "retry" not in call_name, (
                f"forbidden retry call '{call_name}' at line {node.lineno}"
            )
            assert "compensat" not in call_name, (
                f"forbidden compensation call '{call_name}' at line {node.lineno}"
            )
            assert "auto_retract" not in call_name, (
                f"forbidden auto-retract call '{call_name}' at line {node.lineno}"
            )


def test_node_class_has_no_default_publisher():
    """__init__ must not create _target_pub as a publisher object."""
    text = ADAPTER_SOURCE.read_text()
    # The init should set _target_pub = None, not create a publisher
    assert "self._target_pub = None" in text


def test_node_class_has_no_default_arm_client():
    """__init__ must not create _arm_client as a client object."""
    text = ADAPTER_SOURCE.read_text()
    assert "self._arm_client = None" in text


# ------------------------------------------------------------------
# Task state machine tests (already tested in tools/press; quick
# sanity check against v2 artifact)
# ------------------------------------------------------------------


def test_v2_task_runs_51_steps():
    task = ContinuousReturnTask(ROUTE_V2)
    pre = tuple(task.artifact["endpoints"]["pre_contact_recorded_rad"])
    task.authorize(
        TaskAuthorization(
            token_id="test-v2",
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(pre),
            expires_monotonic=9999.0,
            confirmation_text="offline dry run",
        ),
        pre,
        100.0,
    )
    assert task.state is TaskState.READY_NEXT_STEP
    count = 0
    measured = pre
    healthy = {
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "fault": "",
        "torque_enabled": [True] * 6,
        "voltage": 12.12,
        "joint_target_publisher_count": 0,
    }
    while task.state is TaskState.READY_NEXT_STEP:
        candidate = task.issue_next(101.0 + count)
        changed = [
            i
            for i, (a, b) in enumerate(
                zip(candidate.measured_start_rad, candidate.target_positions_rad)
            )
            if abs(b - a) > 0.0015339817878856412 + 1e-12
        ]
        assert changed == [candidate.joint_index]
        assert candidate.target_positions_rad[4] == pytest.approx(pre[4])
        assert candidate.target_positions_rad[5] == pytest.approx(pre[5])
        measured = candidate.target_positions_rad
        task.observe(measured, dict(healthy), 101.5 + count)
        count += 1
    assert count == 51
    assert task.state is TaskState.COMPLETE_DISARMED


def test_v2_authorization_binds_route_hash():
    task = ContinuousReturnTask(ROUTE_V2)
    start = tuple(task.artifact["endpoints"]["pre_contact_recorded_rad"])
    bad = TaskAuthorization("bad", "0" * 64, measured_digest(start), 200.0, "ok")
    with pytest.raises(Exception, match="different route"):
        task.authorize(bad, start, 100.0)


def test_v2_abort_on_fault():
    task = ContinuousReturnTask(ROUTE_V2)
    start = tuple(task.artifact["endpoints"]["pre_contact_recorded_rad"])
    task.authorize(
        TaskAuthorization("t", task.sha256, measured_digest(start), 200.0, "ok"),
        start, 100.0,
    )
    task.issue_next(101.0)
    task.observe(
        start,
        {"state": "FAULT:test", "armed": False, "fault": "test",
         "torque_enabled": [True] * 6, "voltage": 12.12,
         "joint_target_publisher_count": 0},
        101.1,
    )
    assert task.state is TaskState.ABORT_LATCHED_DISARMED


# ------------------------------------------------------------------
# Adapter pre-condition validation tests
# ------------------------------------------------------------------


def test_precondition_checks_require_matched_pair():
    """validate_status_payload rejects bad payloads."""
    from roarm_safe_driver.calibration_core import (
        validate_status_payload,
        SAFE_DRIVER_STATUS_SOURCE,
    )
    from roarm_safe_driver.core import SafetyError

    # Missing state
    with pytest.raises(SafetyError):
        validate_status_payload({}, SAFE_DRIVER_STATUS_SOURCE)

    # Armed=true is rejected
    with pytest.raises(SafetyError, match="disarmed"):
        validate_status_payload(
            {
                "state": "CONVERGED_DISARMED",
                "armed": True,
                "dry_run": False,
                "fault": "",
                "feedback_stamp_ns": 1000,
                "positions_rad": [0.0] * 6,
                "torque_enabled": [True] * 6,
                "voltage": 12.0,
            },
            SAFE_DRIVER_STATUS_SOURCE,
        )

    # Voltage out of range
    with pytest.raises(SafetyError, match="voltage"):
        validate_status_payload(
            {
                "state": "CONVERGED_DISARMED",
                "armed": False,
                "dry_run": False,
                "fault": "",
                "feedback_stamp_ns": 1000,
                "positions_rad": [0.0] * 6,
                "torque_enabled": [True] * 6,
                "voltage_v": 10.0,
            },
            SAFE_DRIVER_STATUS_SOURCE,
        )


def test_validate_status_alignment_checks_timestamp_and_positions():
    """validate_status_alignment requires matching stamp and positions."""
    from roarm_safe_driver.calibration_core import validate_status_alignment
    from roarm_safe_driver.core import SafetyError

    class Joint:
        stamp_ns = 1000
        positions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    evidence = {
        "feedback_stamp_ns": 1000,
        "positions_rad": [0.0] * 6,
    }
    # Should pass
    validate_status_alignment(evidence, Joint())

    # Stamp mismatch
    evidence_bad = dict(evidence, feedback_stamp_ns=999)
    with pytest.raises(SafetyError, match="timestamp"):
        validate_status_alignment(evidence_bad, Joint())

    # Position mismatch
    evidence_bad2 = dict(evidence, positions_rad=[0.0, 0.0, 0.0, 0.0, 0.0, 0.001])
    with pytest.raises(SafetyError, match="positions"):
        validate_status_alignment(evidence_bad2, Joint())


# ------------------------------------------------------------------
# Integration: task + fake-driver cycle simulation
# ------------------------------------------------------------------


def test_full_return_cycle_with_fake_single_publisher():
    """Simulate the full 51-step return with one fake publisher.

    Each step: arm → publish one JointState → fake driver converges →
    auto-disarm → advance.  This matches the adapter's execution model.
    """
    from roarm_safe_driver.core import SafeController

    task = ContinuousReturnTask(
        ROUTE_V2,
        expected_motion_publisher_count=1,
    )
    pre = tuple(task.artifact["endpoints"]["pre_contact_recorded_rad"])
    task.authorize(
        TaskAuthorization(
            token_id="fake-1",
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(pre),
            expires_monotonic=9999.0,
            confirmation_text="fake driver integration test",
        ),
        pre,
        100.0,
    )

    writes: list[bytes] = []
    controller = SafeController(writes.append, dry_run=False)
    measured = pre
    now = 100.0

    def feed_healthy(pos, t):
        from roarm_safe_driver.core import Feedback
        controller.accept_feedback(
            Feedback(
                positions=tuple(float(v) for v in pos),
                loads=(0, 0, 0, 0, 0, 0),
                torque_enabled=(True, True, True, True, True, True),
                voltage=12.1,
                received_at=t,
                raw={"source": "synthetic"},
            )
        )

    count = 0
    while task.state is TaskState.READY_NEXT_STEP:
        candidate = task.issue_next(now + count)

        # Arm
        feed_healthy(measured, now + count)
        controller.arm(now + count)
        assert controller.armed

        # Publish one JointState (this is what the adapter does)
        payload = controller.command(candidate.target_positions_rad, now + count + 0.01)
        decoded = json.loads(payload)
        assert decoded.get("T") == 101

        # Simulate settling: feed target position 21 times
        settle_start = now + count + 0.1
        target = tuple(float(v) for v in candidate.target_positions_rad)
        for si in range(21):
            feed_healthy(target, settle_start + si * 0.05)

        assert controller.last_result == "CONVERGED_DISARMED"
        assert not controller.armed

        measured = target
        healthy_status = {
            "state": "CONVERGED_DISARMED",
            "armed": False,
            "fault": "",
            "torque_enabled": [True] * 6,
            "voltage": 12.1,
            "joint_target_publisher_count": 1,
        }
        task.observe(measured, healthy_status, now + count + 1.5)
        count += 1

    assert count == 51
    assert task.state is TaskState.COMPLETE_DISARMED
    assert len(writes) == 51
    assert all(b"102" not in w for w in writes)


def test_abort_when_publisher_count_changes():
    """If publisher count is not 1, the task aborts."""
    task = ContinuousReturnTask(
        ROUTE_V2,
        expected_motion_publisher_count=0,
    )
    pre = tuple(task.artifact["endpoints"]["pre_contact_recorded_rad"])
    task.authorize(
        TaskAuthorization("t", task.sha256, measured_digest(pre), 200.0, "ok"),
        pre, 100.0,
    )
    task.issue_next(101.0)
    # Report publisher count 2 when task expects 0
    task.observe(
        pre,
        {"state": "CONVERGED_DISARMED", "armed": False, "fault": "",
         "torque_enabled": [True] * 6, "voltage": 12.0,
         "joint_target_publisher_count": 2},
        101.1,
    )
    assert task.state is TaskState.ABORT_LATCHED_DISARMED
    assert "publisher" in task.abort_reason


def test_disarm_is_single_attempt():
    """The adapter's _disarm method must be one-shot, no retry."""
    text = ADAPTER_SOURCE.read_text()
    # Verify _disarm has no loop/retry
    disarm_lines = []
    in_disarm = False
    for i, line in enumerate(text.split("\n"), start=1):
        if "def _disarm" in line:
            in_disarm = True
        elif line.startswith("    def ") and in_disarm:
            break
        elif in_disarm:
            disarm_lines.append(line)
    disarm_body = "\n".join(disarm_lines)
    assert "while" not in disarm_body, "_disarm must not contain a retry loop"
    assert "for " not in disarm_body, "_disarm must not contain a retry loop"
    # Only one call_async should exist
    call_count = disarm_body.count("call_async")
    assert call_count <= 1, f"_disarm should call arm service at most once, found {call_count}"
