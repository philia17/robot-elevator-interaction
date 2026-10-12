"""Tests for the inert M2 press adapter — no hardware required.

These tests verify the adapter's fail-closed behavior without a live ROS
domain or serial device, plus the safe-driver node's T101/T102 command
dispatch and a full fake-driver T102 press cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    """Support both the Mac layout (repo/ros2_arm/<pkg>) and the AGX layout
    (roarm_ws/src/lift_robot/<pkg>)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "tools" / "press" / "m2_press_task.py").is_file():
            return parent
    raise RuntimeError("could not locate repo root containing tools/press")


REPO_ROOT = _find_repo_root()
ADAPTER_SOURCE = (
    Path(__file__).resolve().parents[1] / "roarm_m2_adapter" / "node.py"
)

ROUTE = REPO_ROOT / "config" / "press" / "pilots" / "m2_multi_joint_route_v2.json"

from tools.press.m1_continuous_return_task import (
    TaskAuthorization,
    measured_digest,
)
from tools.press.m2_press_task import (
    MultiJointPressTask,
    POST_VERIFY_REQUIRED_OUTCOME,
    PressTaskState,
)

# Importable on both layouts once tools.press has extended sys.path.
import roarm_safe_driver as _roarm_safe_driver_pkg

DRIVER_NODE_SOURCE = (
    Path(_roarm_safe_driver_pkg.__file__).resolve().parent / "node.py"
)


# ------------------------------------------------------------------
# Interface contract tests — verify the inert-by-default contract
# is enforced at source level
# ------------------------------------------------------------------


def test_adapter_source_has_no_root_level_target_pub():
    """The adapter must NOT create a publisher outside the guarded method."""
    text = ADAPTER_SOURCE.read_text()
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
        f"create_publisher found outside _create_execution_resources "
        f"at lines {publisher_lines}"
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
        f"create_client found outside _create_execution_resources "
        f"at lines {client_lines}"
    )


def test_adapter_source_never_calls_torque_or_clear_fault():
    """The adapter must never call Torque or clear_fault."""
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


def test_adapter_source_never_retries_or_compensates():
    """The adapter must never auto-retry, compensate, or auto-retract."""
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
    assert "self._target_pub = None" in text


def test_node_class_has_no_default_arm_client():
    """__init__ must not create _arm_client as a client object."""
    text = ADAPTER_SOURCE.read_text()
    assert "self._arm_client = None" in text


def test_disarm_is_single_attempt():
    """The adapter's _disarm method must be one-shot, no retry."""
    text = ADAPTER_SOURCE.read_text()
    disarm_lines = []
    in_disarm = False
    for line in text.split("\n"):
        if "def _disarm" in line:
            in_disarm = True
        elif line.startswith("    def ") and in_disarm:
            break
        elif in_disarm:
            disarm_lines.append(line)
    disarm_body = "\n".join(disarm_lines)
    assert "while" not in disarm_body, "_disarm must not contain a retry loop"
    assert "for " not in disarm_body, "_disarm must not contain a retry loop"
    call_count = disarm_body.count("call_async")
    assert call_count <= 1, (
        f"_disarm should call arm service at most once, found {call_count}"
    )


def test_publisher_destroyed_before_post_verify():
    """Motion resources must be destroyed when entering AWAIT_POST_VERIFY."""
    text = ADAPTER_SOURCE.read_text()
    on_complete_lines = []
    in_method = False
    for line in text.split("\n"):
        if "def _on_motion_complete" in line:
            in_method = True
        elif line.startswith("    def ") and in_method:
            break
        elif in_method:
            on_complete_lines.append(line)
    body = "\n".join(on_complete_lines)
    assert "_destroy_publisher" in body
    assert "_disarm" in body


# ------------------------------------------------------------------
# Safe-driver node dispatch contract
# ------------------------------------------------------------------


def test_driver_node_dispatches_multi_joint_to_t102():
    """_command_callback must route multi-joint targets to command_multi."""
    text = DRIVER_NODE_SOURCE.read_text()
    lines = text.split("\n")
    in_callback = False
    callback_lines = []
    for line in lines:
        if "def _command_callback" in line:
            in_callback = True
        elif line.startswith("    def ") and in_callback:
            break
        elif in_callback:
            callback_lines.append(line)
    body = "\n".join(callback_lines)
    assert "count_moving_joints" in body, (
        "_command_callback must count moving joints against feedback"
    )
    assert "command_multi" in body, (
        "_command_callback must dispatch multi-joint targets to command_multi"
    )
    assert "self.controller.command(" in body, (
        "_command_callback must preserve the single-joint T101 path"
    )


def test_driver_node_dispatch_logic_matches_core():
    """The dispatch decision must be reproducible with count_moving_joints."""
    from roarm_safe_driver.core import ENCODER_STEP_RAD, count_moving_joints

    measured = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    single = list(measured)
    single[1] += 0.02
    assert count_moving_joints(single, measured) == 1

    multi = list(measured)
    multi[0] += 0.02
    multi[1] += 0.02
    assert count_moving_joints(multi, measured) == 2

    noise = list(measured)
    noise[0] += ENCODER_STEP_RAD * 0.5
    assert count_moving_joints(noise, measured) == 0


# ------------------------------------------------------------------
# Integration: task + fake-driver T102 cycle with node-style dispatch
# ------------------------------------------------------------------


def _feed_healthy(controller, positions, at):
    from roarm_safe_driver.core import Feedback

    controller.accept_feedback(
        Feedback(
            positions=tuple(float(v) for v in positions),
            loads=(0, 0, 0, 0, 0, 0),
            torque_enabled=(True, True, True, True, True, True),
            voltage=12.1,
            received_at=at,
            raw={"source": "synthetic"},
        )
    )


def test_full_press_cycle_with_fake_driver_dispatch():
    """Simulate the full 14-waypoint cycle through SafeController.

    Each waypoint: arm → dispatch by moving-joint count (the safe-driver
    node's rule) → converge → auto-disarm → advance.  Confirms every
    multi-joint waypoint produces a T102 payload and the task ends in
    AWAIT_POST_VERIFY, then completes on the M0-B pass outcome.
    """
    from roarm_safe_driver.core import SafeController, count_moving_joints

    task = MultiJointPressTask(ROUTE, expected_motion_publisher_count=1)
    start = tuple(task.artifact["endpoints"]["press_standby_start_rad"])
    task.authorize(
        TaskAuthorization(
            token_id="fake-m2",
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(start),
            expires_monotonic=99999.0,
            confirmation_text="fake driver integration test",
        ),
        start,
        100.0,
    )

    writes: list[bytes] = []
    controller = SafeController(writes.append, dry_run=False)
    measured = start
    now = 100.0
    count = 0
    t102_count = 0
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
        candidate = task.issue_next(now + count)
        if candidate is None:
            continue

        # Arm
        _feed_healthy(controller, measured, now + count)
        controller.arm(now + count)
        assert controller.armed

        # Node-style dispatch: multi-joint → T102, single-joint → T101
        target = candidate.target_positions_rad
        moving = count_moving_joints(target, controller.feedback.positions)
        if moving > 1:
            payload = controller.command_multi(target, now + count + 0.01)
            assert json.loads(payload).get("T") == 102
            t102_count += 1
        else:
            payload = controller.command(target, now + count + 0.01)
            assert json.loads(payload).get("T") == 101

        # Simulate settling: feed exact target repeatedly
        settle_start = now + count + 0.1
        exact = tuple(float(v) for v in target)
        for si in range(25):
            _feed_healthy(controller, exact, settle_start + si * 0.05)

        assert controller.last_result == "CONVERGED_DISARMED"
        assert not controller.armed

        measured = exact
        task.observe(
            measured,
            {
                "state": "CONVERGED_DISARMED",
                "armed": False,
                "fault": "",
                "torque_enabled": [True] * 6,
                "voltage": 12.1,
                "joint_target_publisher_count": 1,
            },
            now + count + 2.0,
        )
        count += 1

    total = task.artifact["totals"]["waypoint_count"]
    assert count == total
    assert confirmations == ["press", "retract"]
    assert len(writes) == total
    assert t102_count >= total - 2, "the M2 route must be predominantly multi-joint"
    assert task.state is PressTaskState.AWAIT_POST_VERIFY

    task.record_post_verify(
        {
            "outcome": POST_VERIFY_REQUIRED_OUTCOME,
            "trial_id": "m0b_fake_driver_cycle",
        }
    )
    assert task.state is PressTaskState.COMPLETE_VERIFIED


def test_abort_when_publisher_count_changes():
    """If publisher count is not the expected value, the task aborts."""
    task = MultiJointPressTask(ROUTE, expected_motion_publisher_count=0)
    start = tuple(task.artifact["endpoints"]["press_standby_start_rad"])
    task.authorize(
        TaskAuthorization(
            "t", task.sha256, measured_digest(start), 200.0, "ok"
        ),
        start,
        100.0,
    )
    task.issue_next(101.0)
    task.observe(
        start,
        {
            "state": "CONVERGED_DISARMED",
            "armed": False,
            "fault": "",
            "torque_enabled": [True] * 6,
            "voltage": 12.0,
            "joint_target_publisher_count": 2,
        },
        101.1,
    )
    assert task.state is PressTaskState.ABORT_LATCHED_DISARMED
    assert "publisher" in task.abort_reason
