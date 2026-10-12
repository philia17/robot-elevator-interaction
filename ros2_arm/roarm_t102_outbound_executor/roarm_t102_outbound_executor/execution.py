"""Pure exact-plan Dynamic precontact execution state machine.

This module imports no ROS package and creates no resources.  The ROS node is
responsible only for translating the operations emitted here into one lease,
arm, publisher and fresh-feedback episode.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


MOVE_CONFIRMATION_TOKEN = "MOVE_FROZEN_DYNAMIC_PLAN_ONLY"
VALIDATE_CONFIRMATION_TOKEN = "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION"
DYNAMIC_LEASE_PURPOSE = "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE"
FAULT_CLEARED_DISARMED = "FAULT_CLEARED_DISARMED"
OPERATOR_DISARMED = "DISARMED:operator"


class MoveBlocked(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(self.code if not self.detail else f"{self.code}: {self.detail}")


def lease_reconciliation_decision(
    *, status_age_s: float, maximum_age_s: float, active: object,
    purpose: object, lease_id: object, expected_lease_id: str,
) -> str:
    """Classify fresh lease evidence after an enable call may have escaped."""

    if (
        not math.isfinite(float(status_age_s))
        or not 0.0 <= float(status_age_s) <= float(maximum_age_s)
    ):
        raise MoveBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN", "lease status stale")
    if type(active) is not bool:
        raise MoveBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN", "lease active type")
    if active is False:
        return "PROVEN_INACTIVE"
    if (
        type(purpose) is not str
        or type(lease_id) is not str
        or purpose != DYNAMIC_LEASE_PURPOSE
        or lease_id != expected_lease_id
    ):
        raise MoveBlocked(
            "MOVE_LEASE_RECONCILIATION_FOREIGN",
            "active lease identity does not match dispatched enable",
        )
    return "RELEASE_EXACT_ACTIVE"


@dataclass(frozen=True)
class MovePolicy:
    feedback_max_age_s: float
    waypoint_tolerance_rad: float
    waypoint_timeout_s: float
    min_voltage_v: float
    max_voltage_v: float
    encoder_step_rad: float
    per_command_caps_rad: tuple[float, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MovePolicy":
        keys = (
            "move_feedback_max_age_s", "move_waypoint_tolerance_rad",
            "move_waypoint_timeout_s", "move_min_voltage_v", "move_max_voltage_v",
            "encoder_step_rad",
        )
        try:
            numbers = [float(value[key]) for key in keys]
        except (KeyError, TypeError, ValueError) as exc:
            raise MoveBlocked("MOVE_POLICY_INVALID", str(exc)) from exc
        if any(not math.isfinite(number) for number in numbers):
            raise MoveBlocked("MOVE_POLICY_INVALID", "non-finite threshold")
        feedback_age, tolerance, timeout, low, high, encoder_step = numbers
        if not 0.0 < feedback_age <= 1.0:
            raise MoveBlocked("MOVE_POLICY_INVALID", "feedback age")
        if not 0.0 < tolerance <= math.radians(1.0):
            raise MoveBlocked("MOVE_POLICY_INVALID", "hard tolerance")
        if not 0.1 <= timeout <= 30.0:
            raise MoveBlocked("MOVE_POLICY_INVALID", "waypoint timeout")
        if not 0.0 < low < high <= 30.0:
            raise MoveBlocked("MOVE_POLICY_INVALID", "voltage range")
        if not 0.0 < encoder_step <= 0.01:
            raise MoveBlocked("MOVE_POLICY_INVALID", "encoder step")
        try:
            caps = tuple(float(x) for x in value["per_command_caps_rad"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MoveBlocked("MOVE_POLICY_INVALID", "per-command caps missing/invalid") from exc
        if len(caps) != 6 or any(not math.isfinite(x) or x <= encoder_step for x in caps):
            raise MoveBlocked("MOVE_POLICY_INVALID", "six finite per-command caps required")
        return cls(feedback_age, tolerance, timeout, low, high, encoder_step, caps)


@dataclass(frozen=True)
class SafetySample:
    positions_rad: tuple[float, ...]
    feedback_age_s: float
    state: str
    armed: bool
    dry_run: bool
    fault: str
    torque_enabled: tuple[bool, ...]
    voltage_v: float
    joint_target_publishers: int
    lease_active: bool
    lease_purpose: str = ""
    dynamic_exact_no_reissue: bool = False
    lease_status_age_s: float = math.inf
    feedback_received_monotonic: float = -math.inf
    feedback_stamp_age_s: float = math.inf
    dynamic_exact_accepted_command_sequence: int = 0
    dynamic_exact_accepted_target_positions: tuple[float, ...] = ()
    dynamic_exact_advance_ready: bool = False
    status_received_monotonic: float = -math.inf

    def validate(self, policy: MovePolicy, *, expected_publishers: int,
                 expected_lease: bool,
                 allow_fault_cleared_disarmed: bool = False,
                 allow_torque_on_verified: bool = False,
                 allow_operator_disarmed: bool = False) -> None:
        if len(self.positions_rad) != 6 or any(not math.isfinite(x) for x in self.positions_rad):
            raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", "six finite joints required")
        if not math.isfinite(self.feedback_age_s) or not 0.0 <= self.feedback_age_s <= policy.feedback_max_age_s:
            raise MoveBlocked("MOVE_FEEDBACK_STALE")
        if not math.isfinite(self.feedback_received_monotonic):
            raise MoveBlocked("MOVE_FEEDBACK_STALE", "feedback receive identity")
        safety_failure = runtime_safety_failure_reason(self)
        if safety_failure:
            raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", safety_failure)
        if (
            not math.isfinite(self.feedback_stamp_age_s)
            or not 0.0 <= self.feedback_stamp_age_s <= policy.feedback_max_age_s
        ):
            raise MoveBlocked("MOVE_FEEDBACK_STALE", "feedback source stamp")
        required_state = (
            OPERATOR_DISARMED
            if allow_operator_disarmed else FAULT_CLEARED_DISARMED
            if allow_fault_cleared_disarmed else "CONVERGED_DISARMED"
        )
        allowed_states = {required_state}
        if allow_torque_on_verified:
            allowed_states.add("TORQUE_ON_VERIFIED")
        if self.state not in allowed_states or self.armed:
            raise MoveBlocked(
                "MOVE_SAFETY_GATE_FAILED",
                f"{required_state} required",
            )
        if not math.isfinite(self.voltage_v) or not policy.min_voltage_v <= self.voltage_v <= policy.max_voltage_v:
            raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", "voltage")
        if self.joint_target_publishers != expected_publishers:
            raise MoveBlocked("MOVE_OWNER_DRIFT", "joint_target publisher count")
        if not math.isfinite(self.lease_status_age_s) or not 0.0 <= self.lease_status_age_s <= policy.feedback_max_age_s:
            raise MoveBlocked("MOVE_OWNER_DRIFT", "lease status is stale")
        if self.lease_active is not expected_lease:
            raise MoveBlocked("MOVE_OWNER_DRIFT", "lease state")
        if expected_lease and (
            self.lease_purpose != DYNAMIC_LEASE_PURPOSE
            or not self.dynamic_exact_no_reissue
        ):
            raise MoveBlocked("MOVE_OWNER_DRIFT", "dynamic no-reissue lease proof")


def runtime_safety_failure_reason(sample: SafetySample) -> str:
    """Preserve a freshly received SafeDriver fault/torque terminal cause."""

    fault = str(sample.fault or "").strip()
    if fault:
        return f"FAULT:{fault}"
    state = str(sample.state or "").strip()
    if state.upper().startswith("FAULT:"):
        return state
    if len(sample.torque_enabled) != 6 or not all(sample.torque_enabled):
        enabled = sum(flag is True for flag in sample.torque_enabled)
        return f"TORQUE_SWITCH_OFF:{enabled}/6 enabled"
    if sample.dry_run:
        return "driver dry-run"
    return ""


def validate_command_ack(
    sample: SafetySample,
    *,
    prior_sequence: int,
    expected_target: Sequence[float],
    pre_publish_positions: Sequence[float],
    encoder_step_rad: float,
) -> None:
    """Prove that SafeDriver accepted exactly this immutable waypoint once."""
    if sample.dynamic_exact_accepted_command_sequence != prior_sequence + 1:
        raise MoveBlocked(
            "MOVE_COMMAND_ACK_INVALID",
            "accepted command sequence did not advance exactly once",
        )
    accepted = sample.dynamic_exact_accepted_target_positions
    expected = tuple(float(value) for value in expected_target)
    measured = tuple(float(value) for value in pre_publish_positions)
    if len(accepted) != 6 or len(expected) != 6 or len(measured) != 6:
        raise MoveBlocked("MOVE_COMMAND_ACK_INVALID", "accepted target shape")
    if any(
        not math.isfinite(value)
        for vector in (accepted, expected, measured)
        for value in vector
    ):
        raise MoveBlocked("MOVE_COMMAND_ACK_INVALID", "accepted target non-finite")
    step = float(encoder_step_rad)
    if not math.isfinite(step) or step <= 0.0:
        raise MoveBlocked("MOVE_COMMAND_ACK_INVALID", "encoder step invalid")
    for accepted_value, expected_value, measured_value in zip(
        accepted, expected, measured
    ):
        held = abs(expected_value - measured_value) <= step + 1e-12
        reference = measured_value if held else expected_value
        tolerance = 1e-12 if held else step / 2.0 + 1e-12
        if abs(accepted_value - reference) > tolerance:
            raise MoveBlocked("MOVE_COMMAND_ACK_INVALID", "accepted target mismatch")


def validate_waypoints(plan: Mapping[str, object], policy: MovePolicy) -> tuple[tuple[float, ...], ...]:
    raw = plan.get("waypoints_rad")
    route_kind = str(plan.get("route_kind") or "PRECONTACT")
    bootstrap = plan.get("bootstrap_identity")
    initial_disarmed_state = plan.get("initial_disarmed_state")
    source_return_initial = bool(
        route_kind == "STANDBY"
        and initial_disarmed_state in {
            FAULT_CLEARED_DISARMED, "TORQUE_ON_VERIFIED", OPERATOR_DISARMED,
        }
        and isinstance(plan.get("source_precontact_plan_hash"), str)
        and bool(plan.get("source_precontact_plan_hash"))
    )
    admission = plan.get("start_state_admission")
    stationary_precontact_initial = bool(
        route_kind == "PRECONTACT"
        and initial_disarmed_state in {
            OPERATOR_DISARMED, FAULT_CLEARED_DISARMED, "TORQUE_ON_VERIFIED",
        }
        and plan.get("schema") == "t102_dynamic_precontact_plan_v1"
        and isinstance(admission, Mapping)
        and admission.get("mode") == (
            "FRESH_STATIONARY_OPERATOR_DISARMED_LEGAL_Q0"
            if initial_disarmed_state == OPERATOR_DISARMED
            else "FRESH_STATIONARY_RECOVERED_DISARMED_LEGAL_Q0"
        )
    )
    if initial_disarmed_state is not None and not (
        source_return_initial or stationary_precontact_initial
    ):
        raise MoveBlocked(
            "PLAN_INVALIDATED", "source-route initial disarmed state",
        )
    local_standby_bootstrap = bool(
        route_kind == "STANDBY"
        and isinstance(bootstrap, Mapping)
        and bootstrap.get("mode") == "LOCAL_STANDARD_STANDBY_CORRECTION"
        and bootstrap.get("admission_state") in {
            FAULT_CLEARED_DISARMED, "CONVERGED_DISARMED",
        }
    )
    minimum, maximum = (
        (1, 1) if local_standby_bootstrap
        else ((2, 64) if route_kind == "STANDBY" else (3, 6))
    )
    if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
        raise MoveBlocked(
            "PLAN_INVALIDATED",
            f"exactly {minimum}-{maximum} frozen waypoints required",
        )
    waypoints = tuple(tuple(float(x) for x in waypoint) for waypoint in raw)
    if any(len(wp) != 6 or any(not math.isfinite(x) for x in wp) for wp in waypoints):
        raise MoveBlocked("PLAN_INVALIDATED", "waypoints must be six finite joints")
    if plan.get("automatic_retry") is not False:
        raise MoveBlocked("PLAN_INVALIDATED", "retry contract")
    if route_kind == "STANDBY":
        if any(plan.get(key) is not False for key in ("automatic_compensation", "overshoot", "clear_fault")):
            raise MoveBlocked("PLAN_INVALIDATED", "standby recovery contract")
        try:
            previous = tuple(float(x) for x in plan["start_positions_rad"])
            exact_end = tuple(float(x) for x in plan["standard_standby_rad"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MoveBlocked("PLAN_INVALIDATED", "standby endpoints missing") from exc
        if waypoints[-1] != exact_end:
            raise MoveBlocked("PLAN_INVALIDATED", "standby endpoint is not exact q0")
    else:
        if plan.get("fixed_v2_fallback") is not False:
            raise MoveBlocked("PLAN_INVALIDATED", "fallback contract")
        if plan.get("no_contact") is not True or any(plan.get(key) is not False for key in ("press", "retract", "return")):
            raise MoveBlocked("PLAN_INVALIDATED", "no-contact contract")
        session = plan.get("session")
        if not isinstance(session, Mapping):
            raise MoveBlocked("PLAN_INVALIDATED", "session start missing")
        try:
            previous = tuple(float(x) for x in session["q0_rad"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MoveBlocked("PLAN_INVALIDATED", "session q0 missing") from exc
    if len(previous) != 6 or any(not math.isfinite(x) for x in previous):
        raise MoveBlocked("PLAN_INVALIDATED", "session q0 malformed")
    for index, waypoint in enumerate(waypoints):
        maximum_delta = max(
            abs(target-current) for target, current in zip(waypoint, previous)
        )
        if maximum_delta <= policy.waypoint_tolerance_rad + 1e-12:
            raise MoveBlocked(
                "PLAN_COMMAND_SHAPE_REJECTED",
                f"waypoint {index} delta does not exceed hard tolerance",
            )
        previous = waypoint
    return waypoints


class ExactPlanExecution:
    """Deterministic one-publish-per-waypoint execution protocol."""

    def __init__(
        self, plan: Mapping[str, object], policy: MovePolicy, *, now: float,
        initial_disarmed_state: str = "CONVERGED_DISARMED",
    ) -> None:
        self.plan = plan
        self.policy = policy
        bootstrap = plan.get("bootstrap_identity")
        self.bootstrap_admission_state = (
            str(bootstrap.get("admission_state") or "")
            if isinstance(bootstrap, Mapping)
            and str(plan.get("route_kind") or "") == "STANDBY"
            and bootstrap.get("mode") == "LOCAL_STANDARD_STANDBY_CORRECTION"
            and bootstrap.get("admission_state") in {
                FAULT_CLEARED_DISARMED, "CONVERGED_DISARMED",
            }
            else ""
        )
        self.local_standby_bootstrap = bool(
            str(plan.get("route_kind") or "") == "STANDBY"
            and self.bootstrap_admission_state
        )
        frozen_initial_disarmed_state = str(
            plan.get("initial_disarmed_state") or ""
        )
        if (
            frozen_initial_disarmed_state
            and str(initial_disarmed_state) != frozen_initial_disarmed_state
        ) or (
            not frozen_initial_disarmed_state
            and str(initial_disarmed_state) == OPERATOR_DISARMED
        ):
            raise MoveBlocked(
                "PLAN_INVALIDATED", "initial disarmed state identity drift",
            )
        self.allow_initial_operator_disarmed = bool(
            frozen_initial_disarmed_state == OPERATOR_DISARMED
        )
        self.allow_initial_torque_on_verified = bool(
            initial_disarmed_state == "TORQUE_ON_VERIFIED"
        )
        self.allow_initial_fault_cleared_disarmed = bool(
            initial_disarmed_state == FAULT_CLEARED_DISARMED
        )
        self.waypoints = validate_waypoints(plan, policy)
        self.state = "PRECHECK"
        self.waypoint_index = 0
        self.waypoint_deadline = float("inf")
        self.publish_count = 0
        self.published_targets: list[tuple[float, ...]] = []
        self.terminal_code = ""
        self.terminal_reason = ""
        self.endpoint_error_rad = math.inf
        self.terminal_feedback_age_s = math.inf
        self._last_now = float(now)
        self._published_at = -math.inf
        self._seen_command_active = False

    @property
    def terminal(self) -> bool:
        return self.state in {"SUCCEEDED", "FAILED", "CANCELLED"}

    @property
    def target(self) -> tuple[float, ...]:
        return self.waypoints[self.waypoint_index]

    def precheck(self, sample: SafetySample) -> None:
        if self.state != "PRECHECK":
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if self.plan.get("initial_disarmed_state") and (
            sample.state != self.plan["initial_disarmed_state"]
        ):
            raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", "source-route initial state drift")
        sample.validate(
            self.policy, expected_publishers=0, expected_lease=False,
            allow_fault_cleared_disarmed=(
                self.bootstrap_admission_state == FAULT_CLEARED_DISARMED
                or self.allow_initial_fault_cleared_disarmed
            ),
            allow_torque_on_verified=self.allow_initial_torque_on_verified,
            allow_operator_disarmed=self.allow_initial_operator_disarmed,
        )
        self.state = "ACQUIRE_LEASE"

    def lease_acquired(self, sample: SafetySample) -> None:
        if self.state != "ACQUIRE_LEASE":
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if self.plan.get("initial_disarmed_state") and (
            sample.state != self.plan["initial_disarmed_state"]
        ):
            raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", "source-route initial state drift")
        sample.validate(
            self.policy, expected_publishers=0, expected_lease=True,
            allow_fault_cleared_disarmed=(
                self.bootstrap_admission_state == FAULT_CLEARED_DISARMED
                or self.allow_initial_fault_cleared_disarmed
            ),
            allow_torque_on_verified=self.allow_initial_torque_on_verified,
            allow_operator_disarmed=self.allow_initial_operator_disarmed,
        )
        self.state = "CREATE_PUBLISHER"

    def publisher_created(self, sample: SafetySample) -> None:
        if self.state != "CREATE_PUBLISHER":
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if self.plan.get("initial_disarmed_state") and (
            sample.state != self.plan["initial_disarmed_state"]
        ):
            raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", "source-route initial state drift")
        sample.validate(
            self.policy, expected_publishers=1, expected_lease=True,
            allow_fault_cleared_disarmed=(
                self.bootstrap_admission_state == FAULT_CLEARED_DISARMED
                or self.allow_initial_fault_cleared_disarmed
            ),
            allow_torque_on_verified=self.allow_initial_torque_on_verified,
            allow_operator_disarmed=self.allow_initial_operator_disarmed,
        )
        self.state = "ARM_WAYPOINT"

    def armed(self, success: bool, reason: str = "") -> None:
        if self.state != "ARM_WAYPOINT":
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if not success:
            self.fail("MOVE_ARM_REJECTED", reason)
            return
        self.state = "PUBLISH_WAYPOINT"

    def mark_published(self, target: Sequence[float], *, now: float) -> None:
        if self.state != "PUBLISH_WAYPOINT":
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if not math.isfinite(float(now)) or float(now) < self._last_now:
            raise MoveBlocked("MOVE_CLOCK_INVALID", "non-finite/regressed monotonic time")
        exact = tuple(float(x) for x in target)
        if exact != self.target:
            raise MoveBlocked("PLAN_INVALIDATED", "published target differs from frozen waypoint")
        self.publish_count += 1
        if self.publish_count != self.waypoint_index + 1:
            raise MoveBlocked("MOVE_REISSUE_BLOCKED", "waypoint published more than once")
        self.published_targets.append(exact)
        self._last_now = float(now)
        self._published_at = float(now)
        self._seen_command_active = False
        self.waypoint_deadline = float(now) + self.policy.waypoint_timeout_s
        self.state = "WAIT_CONVERGENCE"

    def command_within_caps(self, target: Sequence[float], measured: Sequence[float]) -> bool:
        # Mirror the driver's pre/post-quantization delta checks against the
        # *measured* joints, not the previous planned waypoint. Held joints are
        # snapped to feedback by the driver and require no extra headroom.
        if len(target) != 6 or len(measured) != 6:
            return False
        step = self.policy.encoder_step_rad
        for q, actual, cap in zip(target, measured, self.policy.per_command_caps_rad):
            if not math.isfinite(q) or not math.isfinite(actual):
                return False
            delta = abs(q - actual)
            if delta <= step + 1e-12:
                continue
            quantized = round(q / step) * step
            if delta > cap + 1e-12 or abs(quantized - actual) > cap + 1e-12:
                return False
        return True

    def observe(self, sample: SafetySample, *, now: float) -> None:
        if self.state != "WAIT_CONVERGENCE":
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if not math.isfinite(float(now)) or float(now) < self._last_now:
            self.fail("MOVE_CLOCK_INVALID", "non-finite/regressed monotonic time", sample=sample)
            return
        self._last_now = float(now)
        if float(now) >= self.waypoint_deadline:
            self.fail("MOVE_WAYPOINT_TIMEOUT", "no retry permitted", sample=sample)
            return
        if (
            not math.isfinite(sample.feedback_received_monotonic)
            or sample.feedback_received_monotonic <= self._published_at
        ):
            self.fail(
                "MOVE_COMMAND_ACK_STALE",
                "feedback was not received after waypoint publish",
                sample=sample,
            )
            return
        # Active motion may report COMMAND_ACTIVE/armed.  Every other safety
        # predicate remains strict while waiting.
        if len(sample.positions_rad) != 6 or any(not math.isfinite(x) for x in sample.positions_rad):
            self.fail("MOVE_SAFETY_GATE_FAILED", "six finite joints required", sample=sample)
            return
        if not math.isfinite(sample.feedback_age_s) or not 0.0 <= sample.feedback_age_s <= self.policy.feedback_max_age_s:
            self.fail("MOVE_FEEDBACK_STALE", sample=sample)
            return
        safety_failure = runtime_safety_failure_reason(sample)
        if safety_failure:
            self.fail("MOVE_SAFETY_GATE_FAILED", safety_failure, sample=sample)
            return
        if not math.isfinite(sample.feedback_stamp_age_s) or not 0.0 <= sample.feedback_stamp_age_s <= self.policy.feedback_max_age_s:
            self.fail("MOVE_FEEDBACK_STALE", "feedback source stamp", sample=sample)
            return
        if (
            sample.state not in {"COMMAND_ACTIVE", "CONVERGED_DISARMED"}
            or (sample.state == "COMMAND_ACTIVE") is not sample.armed
        ):
            self.fail("MOVE_SAFETY_GATE_FAILED", "driver state/armed drift", sample=sample)
            return
        if not math.isfinite(sample.voltage_v) or not self.policy.min_voltage_v <= sample.voltage_v <= self.policy.max_voltage_v:
            self.fail("MOVE_SAFETY_GATE_FAILED", "voltage drift", sample=sample)
            return
        if sample.joint_target_publishers != 1 or not sample.lease_active or sample.lease_purpose != DYNAMIC_LEASE_PURPOSE or not sample.dynamic_exact_no_reissue:
            self.fail("MOVE_OWNER_DRIFT", sample=sample)
            return
        if not math.isfinite(sample.lease_status_age_s) or not 0.0 <= sample.lease_status_age_s <= self.policy.feedback_max_age_s:
            self.fail("MOVE_OWNER_DRIFT", "lease status is stale", sample=sample)
            return
        if sample.state == "COMMAND_ACTIVE" and sample.armed:
            self._seen_command_active = True
            if (
                self.waypoint_index + 1 < len(self.waypoints)
                and sample.dynamic_exact_advance_ready
            ):
                error = max(
                    abs(a - b) for a, b in zip(sample.positions_rad, self.target)
                )
                if error > self.policy.waypoint_tolerance_rad:
                    self.fail(
                        "MOVE_ENDPOINT_ERROR",
                        "SafeDriver advance-ready disagrees with measured endpoint",
                        sample=sample,
                    )
                    return
                if not self.command_within_caps(
                    self.waypoints[self.waypoint_index + 1], sample.positions_rad
                ):
                    # Still executing this single command. Let it converge a
                    # little further; do not send a rejected next waypoint and
                    # mistake its missing acknowledgement for a motion fault.
                    return
                self.endpoint_error_rad = error
                self.waypoint_index += 1
                self.state = "PUBLISH_WAYPOINT"
            return
        if not self._seen_command_active:
            self.fail(
                "MOVE_COMMAND_ACK_MISSING",
                "CONVERGED_DISARMED arrived before COMMAND_ACTIVE evidence",
                sample=sample,
            )
            return
        if sample.state != "CONVERGED_DISARMED" or sample.armed:
            return
        error = max(abs(a-b) for a, b in zip(sample.positions_rad, self.target))
        if error > self.policy.waypoint_tolerance_rad:
            self.fail("MOVE_ENDPOINT_ERROR", f"{error:.12g}", sample=sample)
            return
        self.endpoint_error_rad = error
        self.terminal_feedback_age_s = sample.feedback_age_s
        if self.waypoint_index + 1 == len(self.waypoints):
            self.state = "TEARDOWN_SUCCESS"
        else:
            if not self.command_within_caps(
                self.waypoints[self.waypoint_index + 1], sample.positions_rad
            ):
                self.fail("MOVE_COMMAND_CAP_EXCEEDED", "settled pose to next waypoint", sample=sample)
                return
            self.waypoint_index += 1
            self.state = "ARM_WAYPOINT"

    def cancel(self, reason: str = "operator cancellation requested") -> None:
        if not self.terminal:
            self.state = "TEARDOWN_CANCEL"
            self.terminal_code = "MOVE_CANCELLED"
            self.terminal_reason = reason

    def fail(self, code: str, reason: str = "", *, sample: SafetySample | None = None) -> None:
        if self.terminal:
            return
        self.state = "TEARDOWN_FAILURE"
        self.terminal_code = str(code)
        self.terminal_reason = str(reason)
        if sample is not None:
            self.terminal_feedback_age_s = sample.feedback_age_s

    def teardown_complete(self, *, success: bool) -> None:
        if self.state not in {"TEARDOWN_SUCCESS", "TEARDOWN_FAILURE", "TEARDOWN_CANCEL"}:
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", "teardown from non-terminal path")
        if success and self.state == "TEARDOWN_SUCCESS":
            self.state = "SUCCEEDED"
            if str(self.plan.get("route_kind") or "") == "STANDBY":
                self.terminal_code = "EXACT_STANDBY_COMPLETE_DISARMED"
                self.terminal_reason = "exact segmented Standby route converged; resources released"
            else:
                self.terminal_code = "DYNAMIC_PRECONTACT_STOPPED_DISARMED"
                self.terminal_reason = "exact frozen plan converged; resources released"
        elif self.state == "TEARDOWN_CANCEL":
            self.state = "CANCELLED"
        else:
            self.state = "FAILED"
            if not self.terminal_code:
                self.terminal_code = "MOVE_TEARDOWN_FAILED"


__all__ = [
    "DYNAMIC_LEASE_PURPOSE", "ExactPlanExecution", "FAULT_CLEARED_DISARMED",
    "MOVE_CONFIRMATION_TOKEN", "OPERATOR_DISARMED",
    "MoveBlocked", "MovePolicy", "SafetySample", "VALIDATE_CONFIRMATION_TOKEN",
    "lease_reconciliation_decision", "runtime_safety_failure_reason",
    "validate_command_ack", "validate_waypoints",
]
