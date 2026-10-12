"""Pure, fail-closed controller core for one outbound-only T102 route.

This module is deliberately *not* a ROS node and imports no robot libraries.
It is an offline contract plus fake-controller test seam for a future dedicated
motion owner.  The default configuration is disabled.  It cannot implement
press, retract, return, torque, teleop, fault-clear, Initialize, or Standby.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import secrets
from typing import Any, Mapping, Protocol, Sequence


MAX_TICKET_TTL_S = 2.0
FINAL_JOINT_TOLERANCE_RAD = 0.02
# Mirror of the safe driver's 4096-step encoder grid and its payload
# quantization: moving joints are rounded onto the grid and held joints are
# snapped to the measured pose before the T102 telemetry target is published.
# A status acknowledgement therefore binds to the emitted setpoint within one
# encoder step per joint, not by exact float equality.
ENCODER_STEP_RAD = 2.0 * math.pi / 4096.0
STATUS_TARGET_ACK_TOLERANCE_RAD = ENCODER_STEP_RAD


class T102ExecutorError(RuntimeError):
    """A validation or lifecycle failure that must leave the route aborted."""


class T102State(str, Enum):
    INERT = "INERT"
    EXECUTING = "EXECUTING_OUTBOUND_ONLY"
    COMPLETE = "COMPLETE_NO_CONTACT"
    ABORTED = "ABORTED_SAFE_FAILURE"


@dataclass(frozen=True)
class PreparedCertificateResult:
    """Read-only staging result; it is explicitly not an execution release."""

    accepted_for_inert_staging: bool
    reason: str
    route: "SealedOutboundRoute | None" = None


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise T102ExecutorError("non-canonical route value") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise T102ExecutorError(f"{name} must be a lowercase SHA-256")
    return value


def _vector(value: Any, name: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 6
    ):
        raise T102ExecutorError(f"{name} must have six finite joint values")
    output = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in output):
        raise T102ExecutorError(f"{name} must have six finite joint values")
    return output


def setpoint_sha256(positions_rad: Sequence[float]) -> str:
    """Hash the exact six-joint command the feedback acknowledgement must bind."""

    return canonical_sha256({"positions_rad": list(_vector(positions_rad, "setpoint"))})


def acknowledges_status_target(
    target_positions_rad: Sequence[float] | None,
    setpoint: Sequence[float],
    tolerance_rad: float = STATUS_TARGET_ACK_TOLERANCE_RAD,
) -> bool:
    """Bind a fresh safe-driver status telemetry target to the emitted setpoint.

    The driver stores the validated, encoder-quantized T102 payload target in
    its status telemetry, so the acknowledgement is a per-joint tolerance of
    one encoder step: tighter than every route waypoint separation, yet wide
    enough to describe the driver's real payload target.
    """

    if target_positions_rad is None:
        return False
    if not isinstance(setpoint, Sequence) or isinstance(setpoint, (str, bytes)):
        return False
    if len(target_positions_rad) != 6 or len(setpoint) != 6:
        return False
    if not math.isfinite(float(tolerance_rad)) or tolerance_rad < 0.0:
        return False
    try:
        target = tuple(float(value) for value in target_positions_rad)
        expected = tuple(float(value) for value in setpoint)
    except (TypeError, ValueError):
        return False
    return all(
        abs(actual - goal) <= tolerance_rad + 1e-12
        for actual, goal in zip(target, expected)
    )


@dataclass(frozen=True)
class FrozenBindings:
    """Hashes that bind route bytes to one target/Fn/r6/provenance freeze."""

    m0_target_sha256: str
    fn_sha256: str
    r6_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _sha(value, name)


@dataclass(frozen=True)
class SealedOutboundRoute:
    route_sha256: str
    bindings: FrozenBindings
    setpoints_rad: tuple[tuple[float, ...], ...]
    # Offline-diagnostic markers are carried through the route parser so the
    # execute admission boundary can fail closed on them.  Inert certificate
    # staging may still parse/cache such a route; execution must not.
    route_mode: str = ""
    command_interface: str = ""
    diagnostic_only: bool = False

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], supplied_sha256: str
    ) -> "SealedOutboundRoute":
        if not isinstance(value, Mapping):
            raise T102ExecutorError("route must be an object")
        supplied = _sha(supplied_sha256, "supplied route SHA-256")
        unsigned = dict(value)
        embedded_raw = unsigned.pop("route_sha256", None)
        # Older in-memory certificates carry the parameter hash only; current
        # route artifacts additionally carry a self-hash over bytes excluding
        # that field.  Both forms are accepted, but an embedded value must
        # agree with the separately supplied hash.
        if embedded_raw is None:
            actual = canonical_sha256(value)
        else:
            embedded = _sha(embedded_raw, "route_sha256")
            actual = canonical_sha256(unsigned)
            if not secrets.compare_digest(embedded, supplied):
                raise T102ExecutorError(
                    "embedded route SHA-256 differs from supplied route SHA-256"
                )
        if not secrets.compare_digest(actual, supplied):
            raise T102ExecutorError("route SHA-256 mismatch")
        schema = value.get("schema")
        if schema not in {
            "t102_outbound_only_route_v1",
            "t102_supervised_outbound_route_v1",
        }:
            raise T102ExecutorError("route violates outbound-only policy: schema")
        required = {
            "phase": "OUTBOUND_TO_PRECONTACT",
            "no_contact": True,
            "press": False,
            "retract": False,
            "return": False,
            "automatic_retry": False,
        }
        for key, expected in required.items():
            if value.get(key) != expected:
                raise T102ExecutorError(f"route violates outbound-only policy: {key}")
        raw_bindings = value.get("frozen_bindings")
        if not isinstance(raw_bindings, Mapping):
            raise T102ExecutorError("route frozen bindings are missing")
        bindings = FrozenBindings(
            m0_target_sha256=_sha(
                raw_bindings.get("m0_target_sha256"), "m0_target_sha256"
            ),
            fn_sha256=_sha(raw_bindings.get("fn_sha256"), "fn_sha256"),
            r6_sha256=_sha(raw_bindings.get("r6_sha256"), "r6_sha256"),
            provenance_sha256=_sha(
                raw_bindings.get("provenance_sha256"), "provenance_sha256"
            ),
        )
        raw_points = value.get("setpoints_rad")
        if (
            not isinstance(raw_points, Sequence)
            or isinstance(raw_points, (str, bytes))
            or not raw_points
        ):
            raise T102ExecutorError("route contains no setpoints")
        route_mode = value.get("route_mode")
        command_interface = value.get("command_interface")
        return cls(
            actual,
            bindings,
            tuple(_vector(point, "route setpoint") for point in raw_points),
            route_mode if isinstance(route_mode, str) else "",
            command_interface if isinstance(command_interface, str) else "",
            value.get("diagnostic_only") is True,
        )


@dataclass(frozen=True)
class ExecutionTicket:
    route_sha256: str
    bindings: FrozenBindings
    issued_monotonic: float
    expires_monotonic: float
    operator_confirmation: str


@dataclass(frozen=True)
class Feedback:
    observed_monotonic: float
    state: str
    fault: str = ""
    acknowledged_setpoint_sha256: str = ""
    measured_positions_rad: tuple[float, ...] | None = None


@dataclass(frozen=True)
class StatusTargetAck:
    """Fresh safe-driver status telemetry snapshot carrying the T102 target."""

    observed_monotonic: float
    target_positions_rad: tuple[float, ...] | None = None


class ControllerHooks(Protocol):
    """Deliberately small injection seam; production integration is absent."""

    def acquire_owner(self, owner_id: str) -> bool: ...
    def release_owner(self, owner_id: str) -> None: ...
    def foreign_publishers(self) -> tuple[str, ...]: ...
    def feedback(self) -> Feedback: ...
    def status_ack(self) -> StatusTargetAck: ...
    def cancelled(self) -> bool: ...
    def emit_t102_setpoint(
        self, owner_id: str, positions_rad: tuple[float, ...]
    ) -> None: ...


@dataclass(frozen=True)
class ExecutorConfig:
    enabled: bool = False
    owner_id: str = "t102-outbound-only-core"
    max_ticket_ttl_s: float = MAX_TICKET_TTL_S
    max_feedback_age_s: float = 0.25
    final_joint_tolerance_rad: float = FINAL_JOINT_TOLERANCE_RAD
    # Future ROS wiring must opt in independently.  This only permits a
    # verified certificate to be cached by this pure core; it never enables
    # motion, creates a publisher/client, or changes ``enabled``.
    accept_prepared_certificates: bool = False


@dataclass
class FakeController:
    """In-memory test double.  It does not contact ROS, serial, or hardware."""

    latest_feedback: Feedback
    latest_status_ack: StatusTargetAck = StatusTargetAck(10.05)
    foreign_owner_ids: tuple[str, ...] = ()
    cancel_requested: bool = False
    cancel_after_emits: int | None = None
    acquire_allowed: bool = True
    activate_foreign_after_emits: int | None = None
    foreign_owner_ids_after_emits: tuple[str, ...] = ()
    post_emit_feedback: list[Feedback] = field(default_factory=list)
    post_emit_status_acks: list[StatusTargetAck] = field(default_factory=list)
    emitted: list[tuple[str, tuple[float, ...]]] = field(default_factory=list)
    acquired: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    status_acks_read: list[StatusTargetAck] = field(default_factory=list)

    def acquire_owner(self, owner_id: str) -> bool:
        self.acquired.append(owner_id)
        return self.acquire_allowed

    def release_owner(self, owner_id: str) -> None:
        self.released.append(owner_id)

    def foreign_publishers(self) -> tuple[str, ...]:
        return self.foreign_owner_ids

    def feedback(self) -> Feedback:
        return self.latest_feedback

    def status_ack(self) -> StatusTargetAck:
        self.status_acks_read.append(self.latest_status_ack)
        return self.latest_status_ack

    def cancelled(self) -> bool:
        return self.cancel_requested

    def emit_t102_setpoint(
        self, owner_id: str, positions_rad: tuple[float, ...]
    ) -> None:
        self.emitted.append((owner_id, positions_rad))
        if self.post_emit_status_acks:
            self.latest_status_ack = self.post_emit_status_acks.pop(0)
        if self.post_emit_feedback:
            self.latest_feedback = self.post_emit_feedback.pop(0)
        if (
            self.cancel_after_emits is not None
            and len(self.emitted) >= self.cancel_after_emits
        ):
            self.cancel_requested = True
        if (
            self.activate_foreign_after_emits is not None
            and len(self.emitted) >= self.activate_foreign_after_emits
        ):
            self.foreign_owner_ids = self.foreign_owner_ids_after_emits


class T102OutboundExecutorCore:
    """One-shot route sequencer; every failure tears down ownership without retry."""

    def __init__(self, config: ExecutorConfig = ExecutorConfig()) -> None:
        self._config = config
        self.state = T102State.INERT
        self.staged_route: SealedOutboundRoute | None = None

    def accept_prepared_route_certificate(
        self,
        certificate: Mapping[str, Any],
        supplied_certificate_sha256: str,
    ) -> PreparedCertificateResult:
        """Verify and optionally cache an offline certificate, never execute it.

        This is the only intended no-hardware bridge for a future inert ROS
        shell.  A certificate cannot override ``enabled=False`` or introduce
        an execution capability.  Invalid input and default configuration both
        retain the existing staged value and leave the core ``INERT``.
        """

        if not self._config.accept_prepared_certificates:
            return PreparedCertificateResult(
                False, "prepared certificate staging disabled"
            )
        try:
            supplied = _sha(supplied_certificate_sha256, "prepared certificate SHA-256")
            if not isinstance(certificate, Mapping):
                raise T102ExecutorError("prepared certificate must be an object")
            actual = canonical_sha256(certificate)
            if not secrets.compare_digest(actual, supplied):
                raise T102ExecutorError("prepared certificate SHA-256 mismatch")
            required = {
                "schema": "t102_prepared_route_certificate_v1",
                "status": "OFFLINE_ROUTE_RELEASED",
                "execution_ready": False,
                "motion_authorized": False,
                "no_hardware_transition": True,
            }
            for field, expected in required.items():
                if certificate.get(field) != expected:
                    raise T102ExecutorError(
                        f"prepared certificate violates inert policy: {field}"
                    )
            raw_route = certificate.get("sealed_outbound_route")
            route_hash = _sha(
                certificate.get("sealed_outbound_route_sha256"), "sealed route SHA-256"
            )
            if not isinstance(raw_route, Mapping):
                raise T102ExecutorError("prepared certificate route is missing")
            route = SealedOutboundRoute.from_mapping(raw_route, route_hash)
            raw_bindings = certificate.get("frozen_bindings")
            if not isinstance(raw_bindings, Mapping):
                raise T102ExecutorError("prepared certificate bindings are missing")
            bindings = FrozenBindings(
                m0_target_sha256=_sha(
                    raw_bindings.get("m0_target_sha256"), "certificate m0_target_sha256"
                ),
                fn_sha256=_sha(raw_bindings.get("fn_sha256"), "certificate fn_sha256"),
                r6_sha256=_sha(raw_bindings.get("r6_sha256"), "certificate r6_sha256"),
                provenance_sha256=_sha(
                    raw_bindings.get("provenance_sha256"),
                    "certificate provenance_sha256",
                ),
            )
            if bindings != route.bindings:
                raise T102ExecutorError(
                    "prepared certificate bindings do not match sealed route"
                )
        except T102ExecutorError as exc:
            return PreparedCertificateResult(False, str(exc))
        self.staged_route = route
        return PreparedCertificateResult(
            True, "offline route staged; execution remains disabled", route
        )

    def execute(
        self,
        route: SealedOutboundRoute,
        ticket: ExecutionTicket,
        controller: ControllerHooks,
        *,
        now_monotonic: float,
    ) -> T102State:
        self._validate_execution_admissible(route)
        self._validate_ticket(route, ticket, now_monotonic)
        acquired = False
        try:
            if not controller.acquire_owner(self._config.owner_id):
                raise T102ExecutorError("exclusive outbound owner not acquired")
            acquired = True
            self.state = T102State.EXECUTING
            final_index = len(route.setpoints_rad) - 1
            for index, point in enumerate(route.setpoints_rad):
                before_feedback, before_status = self._gate_setpoint(
                    controller, now_monotonic
                )
                controller.emit_t102_setpoint(self._config.owner_id, point)
                self._require_post_emit_evidence(
                    controller,
                    point,
                    is_final=index == final_index,
                    strict_supervised=(
                        route.route_mode
                        == "SUPERVISED_RESEARCH_PROTOTYPE_EXECUTION"
                    ),
                    now=now_monotonic,
                    previous_feedback_observed_monotonic=before_feedback.observed_monotonic,
                    previous_status_observed_monotonic=before_status.observed_monotonic,
                )
            self.state = T102State.COMPLETE
            return self.state
        except Exception:
            self.state = T102State.ABORTED
            raise
        finally:
            if acquired:
                controller.release_owner(self._config.owner_id)

    def _validate_execution_admissible(self, route: SealedOutboundRoute) -> None:
        """Fail closed at the execute admission boundary on diagnostic markers.

        A route artifact explicitly marked OFFLINE_DIAGNOSTIC_ONLY, exposing
        command_interface NOT_IMPLEMENTED, or carrying diagnostic_only=true
        must never enter the execute path, even when ``enabled=True`` and a
        valid ticket are supplied.  This check runs before ticket validation,
        owner acquisition, and any setpoint emission.
        """

        if route.route_mode == "OFFLINE_DIAGNOSTIC_ONLY":
            raise T102ExecutorError(
                "route is OFFLINE_DIAGNOSTIC_ONLY and cannot execute"
            )
        if route.command_interface == "NOT_IMPLEMENTED":
            raise T102ExecutorError("route command interface is NOT_IMPLEMENTED")
        if route.diagnostic_only:
            raise T102ExecutorError("route is diagnostic_only and cannot execute")
        if route.route_mode and route.route_mode != "SUPERVISED_RESEARCH_PROTOTYPE_EXECUTION":
            raise T102ExecutorError("route lacks supervised execution mode")
        if route.command_interface and route.command_interface != "ROS2_ACTION_ROARM_T102_OUTBOUND_EXECUTOR":
            raise T102ExecutorError("route lacks the reviewed ROS Action interface")

    def _validate_ticket(
        self, route: SealedOutboundRoute, ticket: ExecutionTicket, now: float
    ) -> None:
        if not self._config.enabled:
            raise T102ExecutorError("executor is disabled by default")
        if ticket.operator_confirmation != "MOVE_TO_PRECONTACT_ONLY":
            raise T102ExecutorError("missing exact operator confirmation")
        if ticket.expires_monotonic <= ticket.issued_monotonic:
            raise T102ExecutorError("invalid ticket lifetime")
        if (
            ticket.expires_monotonic - ticket.issued_monotonic
            > self._config.max_ticket_ttl_s
        ):
            raise T102ExecutorError("ticket lifetime exceeds maximum")
        if not ticket.issued_monotonic <= now <= ticket.expires_monotonic:
            raise T102ExecutorError("ticket is not current")
        if not secrets.compare_digest(
            _sha(ticket.route_sha256, "ticket route SHA-256"), route.route_sha256
        ):
            raise T102ExecutorError("ticket route does not match sealed route")
        if ticket.bindings != route.bindings:
            raise T102ExecutorError("ticket frozen target/Fn/r6/provenance mismatch")

    def _gate_setpoint(
        self, controller: ControllerHooks, now: float
    ) -> tuple[Feedback, StatusTargetAck]:
        if controller.cancelled():
            raise T102ExecutorError("operator cancellation requested")
        foreign = controller.foreign_publishers()
        if foreign:
            raise T102ExecutorError("foreign command publisher present")
        feedback = controller.feedback()
        if now - feedback.observed_monotonic > self._config.max_feedback_age_s:
            raise T102ExecutorError("feedback is stale")
        if feedback.observed_monotonic > now + 1e-9:
            raise T102ExecutorError("feedback timestamp is in the future")
        if feedback.fault:
            raise T102ExecutorError("safe-driver fault present")
        if feedback.state not in {"CONVERGED_DISARMED", "TRACKING_OUTBOUND"}:
            raise T102ExecutorError("safe-driver state is not outbound-safe")
        status = controller.status_ack()
        if now - status.observed_monotonic > self._config.max_feedback_age_s:
            raise T102ExecutorError("status telemetry is stale")
        if status.observed_monotonic > now + 1e-9:
            raise T102ExecutorError("status telemetry timestamp is in the future")
        return feedback, status

    def _require_post_emit_evidence(
        self,
        controller: ControllerHooks,
        setpoint: tuple[float, ...],
        *,
        is_final: bool,
        strict_supervised: bool,
        now: float,
        previous_feedback_observed_monotonic: float,
        previous_status_observed_monotonic: float,
    ) -> None:
        """Require a fresh status target acknowledgement and feedback terminal.

        After every emission the executor needs a strictly newer, setpoint-bound
        status telemetry target from the safe driver *and* a strictly newer,
        command-bound feedback terminal.  Neither channel may be stale, from
        the future, faulted, or borrowed from a previous setpoint.
        """

        if controller.cancelled():
            raise T102ExecutorError("operator cancellation requested")
        if controller.foreign_publishers():
            raise T102ExecutorError("foreign command publisher present")
        status = controller.status_ack()
        if now - status.observed_monotonic > self._config.max_feedback_age_s:
            raise T102ExecutorError("post-setpoint status telemetry is stale")
        if status.observed_monotonic > now + 1e-9:
            raise T102ExecutorError(
                "post-setpoint status telemetry timestamp is in the future"
            )
        if status.observed_monotonic <= previous_status_observed_monotonic:
            raise T102ExecutorError(
                "status target acknowledgement did not transition after emission"
            )
        if not acknowledges_status_target(status.target_positions_rad, setpoint):
            raise T102ExecutorError(
                "status telemetry does not acknowledge emitted setpoint"
            )
        feedback = controller.feedback()
        if now - feedback.observed_monotonic > self._config.max_feedback_age_s:
            raise T102ExecutorError("post-setpoint feedback is stale")
        if feedback.observed_monotonic > now + 1e-9:
            raise T102ExecutorError("post-setpoint feedback timestamp is in the future")
        if feedback.observed_monotonic <= previous_feedback_observed_monotonic:
            raise T102ExecutorError(
                "post-setpoint feedback did not transition after emission"
            )
        if feedback.fault:
            raise T102ExecutorError("safe-driver fault present after setpoint")
        expected_hash = setpoint_sha256(setpoint)
        if not secrets.compare_digest(
            feedback.acknowledged_setpoint_sha256, expected_hash
        ):
            raise T102ExecutorError("feedback does not acknowledge emitted setpoint")
        if not strict_supervised and not is_final:
            if feedback.state not in {"TRACKING_OUTBOUND", "CONVERGED_DISARMED"}:
                raise T102ExecutorError(
                    "intermediate setpoint did not reach an allowed state"
                )
            return
        if feedback.state != "CONVERGED_DISARMED":
            raise T102ExecutorError("setpoint is not converged and disarmed")
        if feedback.measured_positions_rad is None:
            raise T102ExecutorError("setpoint measured joint vector is missing")
        measured = _vector(
            feedback.measured_positions_rad, "setpoint measured joint vector"
        )
        if any(
            abs(actual - expected) > self._config.final_joint_tolerance_rad
            for actual, expected in zip(measured, setpoint)
        ):
            raise T102ExecutorError("setpoint measured joint vector is outside tolerance")
