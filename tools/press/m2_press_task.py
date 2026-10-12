#!/usr/bin/env python3
"""Pure task-authorized M2 multi-joint press state machine; no ROS or I/O.

Executes the four-phase T102 route (outbound -> press -> retract -> return)
from an offline-validated `m2_multi_joint_route_v1` artifact.  Entering the
`press` and `retract` phases each requires a separate one-shot operator
confirmation.  After the return endpoint is validated, the task waits for an
M0-B post-verify result; only `FLOW_PASS_NO_LIGHT_EXPECTED` completes the
task.  Any fault latches an abort — no retry, no compensation, no automatic
retract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.press.m1_continuous_return_task import (
    TaskAuthorization,
    TaskError,
    _vector,
    measured_digest,
)
from tools.press.m1_no_contact_route import (
    ENCODER_STEP_RAD,
    JOINT_NAMES,
    T101_COMMAND_LIMITS,
    sha256_file,
)


SCHEMA = "m2_multi_joint_route_v1"
PHASE_ORDER = ("outbound", "press", "retract", "return")
CONFIRMATION_REQUIRED_BEFORE = ("press", "retract")
# The safe driver declares convergence within its 0.5 deg tolerance band and a
# <=2 encoder-step terminal settle; accept measured waypoints within the same
# band instead of the single-encoder-step strictness of the M1 task.
WAYPOINT_TOLERANCE_RAD = math.radians(0.5)
POST_VERIFY_REQUIRED_OUTCOME = "FLOW_PASS_NO_LIGHT_EXPECTED"


class PressTaskState(str, Enum):
    AWAIT_AUTHORIZATION = "AWAIT_AUTHORIZATION"
    READY_NEXT_STEP = "READY_NEXT_STEP"
    AWAIT_PHASE_CONFIRMATION = "AWAIT_PHASE_CONFIRMATION"
    WAIT_DRIVER = "WAIT_DRIVER"
    AWAIT_POST_VERIFY = "AWAIT_POST_VERIFY"
    COMPLETE_VERIFIED = "COMPLETE_VERIFIED"
    ABORT_LATCHED_DISARMED = "ABORT_LATCHED_DISARMED"


@dataclass(frozen=True)
class CandidateWaypoint:
    task_token_id: str
    artifact_sha256: str
    waypoint_id: str
    waypoint_index: int
    phase: str
    measured_start_rad: tuple[float, ...]
    target_positions_rad: tuple[float, ...]
    moving_joint_indices: tuple[int, ...]
    driver_speed_steps_s: int
    driver_acc_register: int


class MultiJointPressTask:
    """Issues at most one next T102 waypoint candidate after each convergence.

    This class is intentionally not a ROS node.  An adapter must arm, publish,
    and disarm via the already verified safe driver; it feeds this object
    observed status only.  This class never opens a device or publishes.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        timeout_sec: float = 30.0,
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
        self.state = PressTaskState.AWAIT_AUTHORIZATION
        self.authorization: TaskAuthorization | None = None
        self.consumed_tokens: set[str] = set()
        self.confirmed_phases: set[str] = set()
        self.latest_measured: tuple[float, ...] | None = None
        self.next_index = 0
        self.pending: CandidateWaypoint | None = None
        self.pending_deadline: float | None = None
        self.abort_reason = ""
        self.post_verify_record: dict[str, Any] | None = None
        self.audit: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # artifact validation
    # ------------------------------------------------------------------

    @property
    def waypoints(self) -> list[Mapping[str, Any]]:
        return self.artifact["waypoints"]

    @property
    def next_phase(self) -> str | None:
        if self.next_index >= len(self.waypoints):
            return None
        return str(self.waypoints[self.next_index]["phase"])

    def _record(self, event: str, **fields: Any) -> None:
        self.audit.append({"event": event, **fields})

    def _validate_artifact(self) -> None:
        value = self.artifact
        if value.get("schema") != SCHEMA:
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
        if contract.get("command_type") != "T102_multi_joint":
            raise TaskError("route artifact command_type must be T102_multi_joint")
        policy = value.get("route_policy", {})
        for key in (
            "automatic_retry",
            "automatic_compensation",
            "automatic_retract_after_fault",
        ):
            if policy.get(key) is not False:
                raise TaskError(f"route policy {key} must be false")
        waypoints = value.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise TaskError("route artifact has no waypoints")
        totals = value.get("totals", {})
        if totals.get("waypoint_count") != len(waypoints):
            raise TaskError("waypoint totals do not match waypoint list")
        seen_phase_index = 0
        phase_counts = {phase: 0 for phase in PHASE_ORDER}
        for index, waypoint in enumerate(waypoints, start=1):
            if waypoint.get("waypoint_id") != f"W{index:03d}":
                raise TaskError("waypoints must be ordered W001..WNN")
            phase = waypoint.get("phase")
            if phase not in PHASE_ORDER:
                raise TaskError(f"unknown waypoint phase: {phase}")
            phase_index = PHASE_ORDER.index(phase)
            if phase_index < seen_phase_index:
                raise TaskError("waypoint phases must follow the route order")
            seen_phase_index = phase_index
            phase_counts[phase] += 1
            target = _vector(waypoint.get("positions_rad"), "positions_rad")
            for joint_index, (lower, upper) in enumerate(T101_COMMAND_LIMITS):
                if not lower <= target[joint_index] <= upper:
                    raise TaskError(
                        f"waypoint {waypoint['waypoint_id']} "
                        f"{JOINT_NAMES[joint_index]} outside T101 limit"
                    )
        for phase in PHASE_ORDER:
            if phase_counts[phase] < 1:
                raise TaskError(f"route artifact is missing the {phase} phase")
            expected = totals.get(f"{phase}_count")
            if expected != phase_counts[phase]:
                raise TaskError(f"{phase} waypoint count does not match totals")
        endpoints = value.get("endpoints", {})
        final_target = _vector(
            waypoints[-1].get("positions_rad"), "positions_rad"
        )
        declared_return = _vector(
            endpoints.get("return_target_rad"), "return_target_rad"
        )
        declared_standby = _vector(
            endpoints.get("press_standby_start_rad"), "press_standby_start_rad"
        )
        for index in range(6):
            if abs(final_target[index] - declared_return[index]) > 1e-12:
                raise TaskError("final waypoint does not match return endpoint")
            if abs(declared_return[index] - declared_standby[index]) > 1e-12:
                raise TaskError("return endpoint does not match press standby")
        speed = policy.get("t102_shared_speed_steps_s")
        acc = policy.get("t102_shared_acc_register")
        if not isinstance(speed, int) or isinstance(speed, bool) or speed < 1:
            raise TaskError("route policy shared speed is invalid")
        if not isinstance(acc, int) or isinstance(acc, bool) or acc < 0:
            raise TaskError("route policy shared acc register is invalid")

    # ------------------------------------------------------------------
    # authorization + phase confirmation
    # ------------------------------------------------------------------

    def authorize(
        self,
        authorization: TaskAuthorization,
        measured: Sequence[float],
        now_monotonic: float,
    ) -> None:
        if self.state is not PressTaskState.AWAIT_AUTHORIZATION:
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
        standby = _vector(
            self.artifact["endpoints"]["press_standby_start_rad"],
            "press_standby_start_rad",
        )
        for index in range(6):
            if abs(current[index] - standby[index]) > WAYPOINT_TOLERANCE_RAD + 1e-12:
                raise TaskError(
                    f"current {JOINT_NAMES[index]} is not at the press standby start"
                )
        self.authorization = authorization
        self.consumed_tokens.add(authorization.token_id)
        self.latest_measured = current
        self.state = PressTaskState.READY_NEXT_STEP
        self._record(
            "authorized",
            artifact_sha256=self.sha256,
            token_id=authorization.token_id,
        )

    def confirm_phase(self, phase: str, confirmation_text: str) -> None:
        """One-shot operator confirmation to enter the press or retract phase."""
        if phase not in CONFIRMATION_REQUIRED_BEFORE:
            raise TaskError(f"phase {phase} does not take a confirmation")
        if not confirmation_text.strip():
            raise TaskError("phase confirmation text is required")
        if self.state is not PressTaskState.AWAIT_PHASE_CONFIRMATION:
            raise TaskError("task is not awaiting a phase confirmation")
        if self.next_phase != phase:
            raise TaskError(
                f"pending phase is {self.next_phase}, not {phase}"
            )
        if phase in self.confirmed_phases:
            raise TaskError(f"phase {phase} confirmation was already consumed")
        self.confirmed_phases.add(phase)
        self.state = PressTaskState.READY_NEXT_STEP
        self._record("phase_confirmed", phase=phase)

    # ------------------------------------------------------------------
    # candidate issuance
    # ------------------------------------------------------------------

    def issue_next(self, now_monotonic: float) -> CandidateWaypoint | None:
        if self.state in (
            PressTaskState.COMPLETE_VERIFIED,
            PressTaskState.AWAIT_POST_VERIFY,
        ):
            return None
        if (
            self.state is not PressTaskState.READY_NEXT_STEP
            or self.latest_measured is None
        ):
            raise TaskError("task is not ready to issue a next candidate")
        if (
            self.authorization is None
            or now_monotonic > self.authorization.expires_monotonic
        ):
            self.abort("task authorization expired")
            raise TaskError("task authorization expired")
        phase = self.next_phase
        if phase is None:
            raise TaskError("no waypoint remains to issue")
        if phase in CONFIRMATION_REQUIRED_BEFORE and phase not in self.confirmed_phases:
            self.state = PressTaskState.AWAIT_PHASE_CONFIRMATION
            self._record("phase_confirmation_required", phase=phase)
            return None
        waypoint = self.waypoints[self.next_index]
        target = _vector(waypoint["positions_rad"], "positions_rad")
        max_delta = _vector(
            self.artifact["joint_contract"]["default_joint_max_delta_rad"],
            "default_joint_max_delta_rad",
        )
        moving: list[int] = []
        for index in range(6):
            delta = abs(target[index] - self.latest_measured[index])
            if delta > ENCODER_STEP_RAD + 1e-12:
                moving.append(index)
            # Allow the driver convergence band on top of the per-joint
            # planning delta; the safe driver remains the strict gate.
            if delta > max_delta[index] + WAYPOINT_TOLERANCE_RAD + 1e-12:
                self.abort(
                    f"candidate delta at {JOINT_NAMES[index]} exceeds max delta"
                )
                raise TaskError("candidate delta exceeds per-joint max delta")
        if not moving:
            self.abort("candidate would not change any joint")
            raise TaskError("candidate would not change any joint")
        policy = self.artifact["route_policy"]
        candidate = CandidateWaypoint(
            task_token_id=self.authorization.token_id,
            artifact_sha256=self.sha256,
            waypoint_id=str(waypoint["waypoint_id"]),
            waypoint_index=self.next_index,
            phase=str(waypoint["phase"]),
            measured_start_rad=self.latest_measured,
            target_positions_rad=target,
            moving_joint_indices=tuple(moving),
            driver_speed_steps_s=int(policy["t102_shared_speed_steps_s"]),
            driver_acc_register=int(policy["t102_shared_acc_register"]),
        )
        self.pending = candidate
        self.pending_deadline = now_monotonic + self.timeout_sec
        self.state = PressTaskState.WAIT_DRIVER
        self._record(
            "candidate_issued",
            waypoint_id=candidate.waypoint_id,
            phase=candidate.phase,
            moving_joint_indices=list(moving),
        )
        return candidate

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def observe(
        self,
        measured: Sequence[float],
        driver_status: Mapping[str, Any],
        now_monotonic: float,
    ) -> None:
        current = _vector(measured, "measured")
        self.latest_measured = current
        if self.state in (
            PressTaskState.COMPLETE_VERIFIED,
            PressTaskState.ABORT_LATCHED_DISARMED,
        ):
            return
        if (
            self.authorization is None
            or now_monotonic > self.authorization.expires_monotonic
        ):
            self.abort("task authorization expired")
            return
        if (
            self.state is PressTaskState.WAIT_DRIVER
            and self.pending_deadline is not None
            and now_monotonic > self.pending_deadline
        ):
            self.abort("safe-driver convergence timeout")
            return
        self._require_healthy_status(driver_status)
        if self.state is not PressTaskState.WAIT_DRIVER or self.pending is None:
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
        target = self.pending.target_positions_rad
        for index in range(6):
            if abs(current[index] - target[index]) > WAYPOINT_TOLERANCE_RAD + 1e-12:
                self.abort(
                    f"{JOINT_NAMES[index]} did not converge to the waypoint target"
                )
                return
        self._record("waypoint_converged", waypoint_id=self.pending.waypoint_id)
        self.next_index += 1
        self.pending = None
        self.pending_deadline = None
        if self.next_index == len(self.waypoints):
            self._validate_return_endpoint(current)
            if self.state is PressTaskState.ABORT_LATCHED_DISARMED:
                return
            self.state = PressTaskState.AWAIT_POST_VERIFY
            self._record("motion_complete_awaiting_post_verify")
        else:
            self.state = PressTaskState.READY_NEXT_STEP

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
        if (
            isinstance(voltage, bool)
            or not isinstance(voltage, (int, float))
            or not 11.0 <= float(voltage) <= 13.0
        ):
            self.abort("safe-driver voltage is outside 11-13 V")
            return
        if publishers != self.expected_motion_publisher_count:
            self.abort("unexpected motion publisher count")

    def _validate_return_endpoint(self, current: tuple[float, ...]) -> None:
        standby = _vector(
            self.artifact["endpoints"]["press_standby_start_rad"],
            "press_standby_start_rad",
        )
        for index in range(6):
            if abs(current[index] - standby[index]) > WAYPOINT_TOLERANCE_RAD + 1e-12:
                self.abort(f"return endpoint error at {JOINT_NAMES[index]}")
                return
        self._record("return_endpoint_validated")

    # ------------------------------------------------------------------
    # post-verify (M0-B result)
    # ------------------------------------------------------------------

    def record_post_verify(self, result: Mapping[str, Any]) -> None:
        """Record the M0-B reacquisition outcome after the return phase.

        The task completes only on `FLOW_PASS_NO_LIGHT_EXPECTED`.  Any other
        outcome is a failed attempt and latches the abort state — it is never
        reported as a successful press.
        """
        if self.state is not PressTaskState.AWAIT_POST_VERIFY:
            raise TaskError("task is not awaiting a post-verify result")
        if not isinstance(result, Mapping):
            raise TaskError("post-verify result must be a mapping")
        outcome = str(result.get("outcome", "")).strip()
        trial_id = str(result.get("trial_id", "")).strip()
        if not trial_id:
            raise TaskError("post-verify result requires a trial_id")
        record = {"outcome": outcome, "trial_id": trial_id}
        for key in ("target_label", "evidence_sha256", "evidence_path"):
            if key in result:
                record[key] = result[key]
        self.post_verify_record = record
        if outcome != POST_VERIFY_REQUIRED_OUTCOME:
            self.abort(f"post-verify failed: outcome={outcome or 'missing'}")
            return
        self.state = PressTaskState.COMPLETE_VERIFIED
        self._record("post_verify_passed", trial_id=trial_id)

    # ------------------------------------------------------------------
    # abort latch
    # ------------------------------------------------------------------

    def abort(self, reason: str) -> None:
        if self.state is PressTaskState.ABORT_LATCHED_DISARMED:
            return
        self.abort_reason = reason
        self.pending = None
        self.pending_deadline = None
        self.state = PressTaskState.ABORT_LATCHED_DISARMED
        self._record("abort_latched", reason=reason)
