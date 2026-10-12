#!/usr/bin/env python3
"""Pure task-authorized M1 return state machine; no ROS or hardware I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.press.m1_no_contact_route import (
    ENCODER_STEP_RAD,
    GRIPPER_INDEX,
    JOINT_NAMES,
    REPO_ROOT,
    T101_COMMAND_LIMITS,
    sha256_file,
)


class TaskError(RuntimeError):
    pass


class TaskState(str, Enum):
    AWAIT_AUTHORIZATION = "AWAIT_AUTHORIZATION"
    READY_NEXT_STEP = "READY_NEXT_STEP"
    WAIT_DRIVER = "WAIT_DRIVER"
    COMPLETE_DISARMED = "COMPLETE_DISARMED"
    ABORT_LATCHED_DISARMED = "ABORT_LATCHED_DISARMED"


@dataclass(frozen=True)
class TaskAuthorization:
    token_id: str
    artifact_sha256: str
    start_measured_digest: str
    expires_monotonic: float
    confirmation_text: str


@dataclass(frozen=True)
class CandidateStep:
    task_token_id: str
    artifact_sha256: str
    route_step_id: str
    route_index: int
    joint_index: int
    joint: str
    measured_start_rad: tuple[float, ...]
    target_positions_rad: tuple[float, ...]
    target_rad: float
    driver_speed_steps_s: int
    driver_acc_register: int


def measured_digest(positions: Sequence[float]) -> str:
    if len(positions) != 6:
        raise TaskError("measured positions must contain six values")
    values = []
    for index, value in enumerate(positions):
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise TaskError(f"measured[{index}] is not finite")
        values.append(round(float(value), 9))
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _vector(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise TaskError(f"{field} must contain exactly six values")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise TaskError(f"{field}[{index}] is not finite")
        number = float(item)
        if not math.isfinite(number):
            raise TaskError(f"{field}[{index}] is not finite")
        result.append(number)
    return tuple(result)


class ContinuousReturnTask:
    """Issues at most one next single-joint candidate after each convergence.

    This class is intentionally not a ROS node. An eventual adapter must arm,
    publish, and disarm via the already verified safe driver; it must feed this
    object observed status only. This class never opens a device or publishes.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        timeout_sec: float = 20.0,
        expected_motion_publisher_count: int = 0,
    ):
        self.path = Path(artifact_path).expanduser().resolve()
        self.sha256 = sha256_file(self.path)
        self.artifact = json.loads(self.path.read_text())
        self._validate_artifact()
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if expected_motion_publisher_count not in (0, 1):
            raise ValueError("expected_motion_publisher_count must be 0 or 1")
        self.timeout_sec = float(timeout_sec)
        self.expected_motion_publisher_count = expected_motion_publisher_count
        self.state = TaskState.AWAIT_AUTHORIZATION
        self.authorization: TaskAuthorization | None = None
        self.consumed_tokens: set[str] = set()
        self.latest_measured: tuple[float, ...] | None = None
        self.held_roll_gripper: tuple[float, float] | None = None
        self.next_index = 0
        self.pending: CandidateStep | None = None
        self.pending_deadline: float | None = None
        self.abort_reason = ""
        self.audit: list[dict[str, Any]] = []

    @property
    def steps(self) -> list[Mapping[str, Any]]:
        return self.artifact["return"]["steps"]

    def _record(self, event: str, **fields: Any) -> None:
        self.audit.append({"event": event, **fields})

    def _validate_artifact(self) -> None:
        value = self.artifact
        if value.get("schema") != "m1_no_contact_taught_route_v1":
            raise TaskError("unexpected route artifact schema")
        if value.get("status") != "offline_validated_candidate":
            raise TaskError("route artifact has not passed offline validation")
        for key in (
            "execution_ready",
            "motion_authorized",
            "contact_authorized",
            "press_authorized",
        ):
            if value.get(key) is not False:
                raise TaskError(f"route artifact {key} must be false")
        contract = value.get("joint_contract", {})
        if tuple(contract.get("joint_names", ())) != JOINT_NAMES:
            raise TaskError("route artifact joint contract is invalid")
        if contract.get("one_logical_joint_per_step") is not True:
            raise TaskError("route artifact must require one logical joint per step")
        steps = value.get("return", {}).get("steps")
        if not isinstance(steps, list) or not steps:
            raise TaskError("route artifact has no return steps")
        for index, step in enumerate(steps, start=1):
            if step.get("phase") != "return" or step.get("step_id") != f"R{index:03d}":
                raise TaskError("return steps must be ordered R001..RNN")
            joint_index = step.get("joint_index")
            if not isinstance(joint_index, int) or not 0 <= joint_index < 5:
                raise TaskError("return must only move task axes")
            if step.get("joint") != JOINT_NAMES[joint_index]:
                raise TaskError("return joint name/index mismatch")
            target = _vector(step.get("target_positions_rad"), "target_positions_rad")
            lower, upper = T101_COMMAND_LIMITS[joint_index]
            if not lower <= target[joint_index] <= upper:
                raise TaskError("return target is outside T101 limit")

    def authorize(
        self,
        authorization: TaskAuthorization,
        measured: Sequence[float],
        now_monotonic: float,
    ) -> None:
        if self.state is not TaskState.AWAIT_AUTHORIZATION:
            raise TaskError("task is not awaiting authorization")
        if not authorization.confirmation_text.strip():
            raise TaskError("task authorization text is required")
        if authorization.token_id in self.consumed_tokens:
            raise TaskError("task authorization token was already consumed")
        if authorization.artifact_sha256 != self.sha256:
            raise TaskError("task authorization binds a different route artifact")
        if now_monotonic > authorization.expires_monotonic:
            raise TaskError("task authorization is expired")
        current = _vector(measured, "measured")
        if measured_digest(current) != authorization.start_measured_digest:
            raise TaskError("task authorization binds a different measured start")
        expected = _vector(
            self.artifact["endpoints"]["pre_contact_task_target_with_gripper_hold_rad"],
            "pre_contact_task_target_with_gripper_hold_rad",
        )
        tolerance = float(self.artifact["joint_contract"]["hold_tolerance_rad"])
        for index in range(5):
            if abs(current[index] - expected[index]) > tolerance + 1e-12:
                raise TaskError(
                    f"current {JOINT_NAMES[index]} is not at the pre-contact task start"
                )
        self.authorization = authorization
        self.consumed_tokens.add(authorization.token_id)
        self.latest_measured = current
        # The physical pre_contact record has a one-step roll and four-step
        # gripper difference from the nominal outbound HOLD target. Preserve
        # both values throughout this return; never sneak in a second axis.
        self.held_roll_gripper = (current[4], current[GRIPPER_INDEX])
        self.state = TaskState.READY_NEXT_STEP
        self._record("authorized", artifact_sha256=self.sha256, token_id=authorization.token_id)

    def issue_next(self, now_monotonic: float) -> CandidateStep | None:
        if self.state is TaskState.COMPLETE_DISARMED:
            return None
        if self.state is not TaskState.READY_NEXT_STEP or self.latest_measured is None:
            raise TaskError("task is not ready to issue a next candidate")
        if self.authorization is None or now_monotonic > self.authorization.expires_monotonic:
            self.abort("task authorization expired")
            raise TaskError("task authorization expired")
        step = self.steps[self.next_index]
        joint_index = int(step["joint_index"])
        target = list(self.latest_measured)
        target[joint_index] = float(step["target_rad"])
        if self.held_roll_gripper is not None:
            target[4], target[GRIPPER_INDEX] = self.held_roll_gripper
        changed = [
            index
            for index, (before, after) in enumerate(zip(self.latest_measured, target))
            if abs(after - before) > ENCODER_STEP_RAD + 1e-12
        ]
        if changed != [joint_index]:
            self.abort("candidate would not be a single logical joint command")
            raise TaskError("candidate would not be a single logical joint command")
        lower, upper = T101_COMMAND_LIMITS[joint_index]
        if not lower <= target[joint_index] <= upper:
            self.abort("candidate target outside T101 limit")
            raise TaskError("candidate target outside T101 limit")
        candidate = CandidateStep(
            task_token_id=self.authorization.token_id,
            artifact_sha256=self.sha256,
            route_step_id=str(step["step_id"]),
            route_index=self.next_index,
            joint_index=joint_index,
            joint=str(step["joint"]),
            measured_start_rad=self.latest_measured,
            target_positions_rad=tuple(target),
            target_rad=target[joint_index],
            driver_speed_steps_s=int(step["speed_steps_s"]),
            driver_acc_register=int(step["acc_register"]),
        )
        self.pending = candidate
        self.pending_deadline = now_monotonic + self.timeout_sec
        self.state = TaskState.WAIT_DRIVER
        self._record("candidate_issued", step_id=candidate.route_step_id, joint=candidate.joint)
        return candidate

    def observe(
        self,
        measured: Sequence[float],
        driver_status: Mapping[str, Any],
        now_monotonic: float,
    ) -> None:
        current = _vector(measured, "measured")
        self.latest_measured = current
        if self.state in (TaskState.COMPLETE_DISARMED, TaskState.ABORT_LATCHED_DISARMED):
            return
        if self.authorization is None or now_monotonic > self.authorization.expires_monotonic:
            self.abort("task authorization expired")
            return
        if self.state is TaskState.WAIT_DRIVER and self.pending_deadline is not None:
            if now_monotonic > self.pending_deadline:
                self.abort("safe-driver convergence timeout")
                return
        self._require_healthy_status(driver_status)
        if self.state is not TaskState.WAIT_DRIVER or self.pending is None:
            return
        status = str(driver_status.get("state", "")).upper()
        if status in {"ARMED", "COMMAND_ACTIVE", "DRY_RUN_ACTIVE"}:
            if driver_status.get("armed") is not True:
                self.abort("safe-driver active state is not armed")
            return
        if status != "CONVERGED_DISARMED":
            self.abort(f"unexpected safe-driver state while waiting: {status}")
            return
        if driver_status.get("armed") is not False:
            self.abort("safe-driver convergence did not disarm")
            return
        tolerance = ENCODER_STEP_RAD
        target = self.pending.target_positions_rad
        selected = self.pending.joint_index
        if abs(current[selected] - target[selected]) > tolerance + 1e-12:
            self.abort("selected joint did not converge to candidate target")
            return
        for index in range(6):
            if index == selected:
                continue
            if abs(current[index] - target[index]) > tolerance + 1e-12:
                self.abort(f"non-target drift at {JOINT_NAMES[index]}")
                return
        self._record("candidate_converged", step_id=self.pending.route_step_id)
        self.next_index += 1
        self.pending = None
        self.pending_deadline = None
        if self.next_index == len(self.steps):
            self._validate_return_endpoint(current)
            self.state = TaskState.COMPLETE_DISARMED
            self._record("complete_disarmed")
        else:
            self.state = TaskState.READY_NEXT_STEP

    def _require_healthy_status(self, status: Mapping[str, Any]) -> None:
        fault = str(status.get("fault", ""))
        state = str(status.get("state", "")).upper()
        torque = status.get("torque_enabled")
        voltage = status.get("voltage")
        publishers = status.get("joint_target_publisher_count")
        if "FAULT" in state or fault:
            self.abort(f"safe-driver fault: {fault or state}")
            return
        if torque != [True] * 6:
            self.abort("one or more torque flags are not true")
            return
        if isinstance(voltage, bool) or not isinstance(voltage, (int, float)) or not 11.0 <= float(voltage) <= 13.0:
            self.abort("safe-driver voltage is outside 11-13 V")
            return
        if publishers != self.expected_motion_publisher_count:
            self.abort("unexpected motion publisher count")

    def _validate_return_endpoint(self, current: tuple[float, ...]) -> None:
        standby = _vector(
            self.artifact["endpoints"]["press_standby_start_rad"],
            "press_standby_start_rad",
        )
        tolerance = float(self.artifact["joint_contract"]["hold_tolerance_rad"])
        for index in range(5):
            if abs(current[index] - standby[index]) > tolerance + 1e-12:
                self.abort(f"return endpoint error at {JOINT_NAMES[index]}")
                raise TaskError("return endpoint does not match standby task axes")
        if self.held_roll_gripper is None:
            self.abort("held non-task axes missing")
            raise TaskError("held non-task axes missing")
        if abs(current[4] - self.held_roll_gripper[0]) > ENCODER_STEP_RAD + 1e-12:
            self.abort("wrist roll changed despite hold policy")
            raise TaskError("wrist roll changed despite hold policy")
        if abs(current[GRIPPER_INDEX] - self.held_roll_gripper[1]) > ENCODER_STEP_RAD + 1e-12:
            self.abort("gripper changed despite hold policy")
            raise TaskError("gripper changed despite hold policy")

    def abort(self, reason: str) -> None:
        if self.state is TaskState.ABORT_LATCHED_DISARMED:
            return
        self.abort_reason = reason
        self.pending = None
        self.pending_deadline = None
        self.state = TaskState.ABORT_LATCHED_DISARMED
        self._record("abort_latched", reason=reason)
