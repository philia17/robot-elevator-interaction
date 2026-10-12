"""Installable fail-closed core for one outbound pre-contact attempt.

This module is the single source of truth for target, release, route-artifact,
ticket, gate, and resource-lifecycle validation.  It has no ROS, serial,
planner, or hardware imports.  The production node injects no real executor,
so the shipped package remains inert even when parameters are misconfigured.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import secrets
from typing import Any, Mapping, Protocol, Sequence


MAX_TICKET_TTL_S = 2.0
PRECONTACT_DISTANCE_M = 0.050
REQUIRED_CONFIRMATION_TEXT = "MOVE_TO_PRECONTACT_ONLY"
TARGET_SCHEMA = "button1_frozen_target_v2"
RELEASE_SCHEMA = "precontact_execution_release_v2"
ROUTE_SCHEMA = "precontact_outbound_route_v1"
ROUTE_PHASE = "OUTBOUND_TO_PRECONTACT"
REQUIRED_EXECUTION_INTERFACE = "SAFE_ADAPTER_ACTION_SERVER"


class PrecontactError(RuntimeError):
    """A fail-closed validation or state-transition error."""


class PrecontactState(str, Enum):
    INERT = "INERT"
    PREPARED_NO_MOTION = "PREPARED_NO_MOTION"
    EXECUTING_OUTBOUND_TO_PRECONTACT = "EXECUTING_OUTBOUND_TO_PRECONTACT"
    COMPLETE_DISARMED = "COMPLETE_DISARMED"
    ABORT_LATCHED = "ABORT_LATCHED"


class TerminalCode(str, Enum):
    PREPARED = "PREPARED_NO_MOTION"
    REJECTED = "REJECTED_NO_MOTION"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED_INERT"
    EXECUTING = "EXECUTING_OUTBOUND_TO_PRECONTACT"
    COMPLETE = "COMPLETE_DISARMED"
    ABORTED = "ABORT_LATCHED"
    CANCELLED = "CANCELLED_SAFE_FAILURE"


def canonical_json(value: Any) -> str:
    """Return the sole canonical JSON representation used for safety hashes."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PrecontactError(f"canonical JSON failed: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_route_artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(_mapping(value, "outbound_route_artifact"))
    unsigned.pop("canonical_payload_sha256", None)
    return canonical_sha256(unsigned)


def canonical_setpoint_sha256(positions_rad: Sequence[float]) -> str:
    return canonical_sha256({"positions_rad": list(_vec(positions_rad, 6, "positions_rad"))})


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrecontactError(f"{field} must be an object")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PrecontactError(f"{field} must be a lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise PrecontactError(f"{field} must be a lowercase SHA-256")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise PrecontactError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PrecontactError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise PrecontactError(f"{field} must be finite")
    return result


def _vec(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise PrecontactError(f"{field} must contain exactly {length} finite values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, field: str) -> tuple[float, float, float]:
    vector = _vec(value, 3, field)
    magnitude = math.sqrt(sum(item * item for item in vector))
    if not 0.999 <= magnitude <= 1.001:
        raise PrecontactError(f"{field} must be a unit vector")
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _require_equal_vector(left: Sequence[float], right: Sequence[float], field: str) -> None:
    if _distance(left, right) > 1e-9:
        raise PrecontactError(f"{field} mismatch")


@dataclass(frozen=True)
class OutboundRouteArtifact:
    artifact_sha256: str
    canonical_payload_sha256: str
    route_hash: str
    manifest_sha256: str
    target_id: str
    detection_id: str
    source_stamp_ns: int
    source_frame_id: str
    m0_frozen_artifact_sha256: str
    calibration_release_id: str
    calibration_bundle_sha256: str
    a_base_m: tuple[float, float, float]
    n_out_base: tuple[float, float, float]
    p_pre_base_m: tuple[float, float, float]
    final_setpoint_sha256: str


def _route_content_from_artifact(route: Mapping[str, Any]) -> dict[str, Any]:
    bindings = _mapping(route.get("release_bindings"), "route.release_bindings")
    geometry = _mapping(route.get("geometry"), "route.geometry")
    quantization = _mapping(route.get("quantization"), "route.quantization")
    return {
        "manifest_sha256": bindings.get("manifest_sha256"),
        "phase": route.get("phase"),
        "target_binding": _mapping(route.get("target_binding"), "route.target_binding"),
        "geometry": {
            "A_base_m": geometry.get("A_base_m"),
            "N_out_base": geometry.get("N_out_base"),
            "P_pre_base_m": geometry.get("P_pre_base_m"),
        },
        "start_quantized_rad": quantization.get("start_quantized_rad"),
        "emitted_setpoints_rad": quantization.get("emitted_setpoints_rad"),
        "precontact_contact_point_base_m": geometry.get("precontact_contact_point_base_m"),
        "precontact_distance_m": geometry.get("exact_precontact_distance_m"),
        "contact_rotation_xyzw": geometry.get("contact_rotation_xyzw"),
        "quantized_endpoint_evidence": route.get("quantized_endpoint_evidence"),
        "collision_evidence": route.get("clearance_evidence"),
    }


def parse_outbound_route_artifact(
    value: Mapping[str, Any],
    supplied_artifact_sha256: str,
) -> OutboundRouteArtifact:
    """Validate the complete frozen route bytes and their internal hash chain."""

    route = _mapping(value, "outbound_route_artifact")
    supplied = _sha256(supplied_artifact_sha256, "outbound_route_artifact_sha256")
    artifact_hash = canonical_sha256(route)
    if not secrets.compare_digest(artifact_hash, supplied):
        raise PrecontactError("outbound route artifact SHA-256 mismatch")
    if route.get("schema") != ROUTE_SCHEMA:
        raise PrecontactError("unexpected outbound route artifact schema")
    if route.get("status") != "offline_released_route":
        raise PrecontactError("outbound route artifact is not released")
    if route.get("phase") != ROUTE_PHASE or route.get("phases") != [ROUTE_PHASE]:
        raise PrecontactError("outbound route artifact permits a forbidden phase")
    exact_safety = {
        "executable": True,
        "adapter_consumable": True,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "no_contact": True,
        "automatic_retry": False,
    }
    for field, expected in exact_safety.items():
        if route.get(field) is not expected:
            raise PrecontactError(f"outbound route artifact has unsafe {field}")
    if route.get("required_execution_interface") != REQUIRED_EXECUTION_INTERFACE:
        raise PrecontactError("outbound route artifact names another execution interface")

    payload_hash = _sha256(
        route.get("canonical_payload_sha256"),
        "route.canonical_payload_sha256",
    )
    if canonical_route_artifact_sha256(route) != payload_hash:
        raise PrecontactError("outbound route canonical payload hash mismatch")
    route_hash = _sha256(route.get("route_hash"), "route.route_hash")
    if canonical_sha256(_route_content_from_artifact(route)) != route_hash:
        raise PrecontactError("outbound route_hash does not bind route content")
    if route.get("route_id") != f"precontact-outbound-{route_hash[:16]}":
        raise PrecontactError("outbound route_id does not bind route_hash")

    binding = _mapping(route.get("target_binding"), "route.target_binding")
    target_id = str(binding.get("target_id") or "").strip()
    detection_id = str(binding.get("detection_id") or "").strip()
    source_frame_id = str(binding.get("source_frame_id") or "").strip()
    calibration_release_id = str(binding.get("calibration_release_id") or "").strip()
    source_stamp_ns = binding.get("source_stamp_ns")
    if not target_id or not detection_id or not source_frame_id or not calibration_release_id:
        raise PrecontactError("outbound route target binding is incomplete")
    if isinstance(source_stamp_ns, bool) or not isinstance(source_stamp_ns, int) or source_stamp_ns <= 0:
        raise PrecontactError("outbound route source stamp is invalid")
    m0_hash = _sha256(
        binding.get("m0_frozen_artifact_sha256"),
        "route.target_binding.m0_frozen_artifact_sha256",
    )
    calibration_hash = _sha256(
        binding.get("r6_bundle_artifact_sha256"),
        "route.target_binding.r6_bundle_artifact_sha256",
    )
    release_bindings = _mapping(route.get("release_bindings"), "route.release_bindings")
    manifest_sha256 = _sha256(
        release_bindings.get("manifest_sha256"),
        "route.release_bindings.manifest_sha256",
    )
    m0_release = _mapping(release_bindings.get("m0_frozen"), "route.release_bindings.m0_frozen")
    r6_release = _mapping(release_bindings.get("r6_bundle"), "route.release_bindings.r6_bundle")
    if m0_release.get("artifact_sha256") != m0_hash:
        raise PrecontactError("route target binding does not match M0 release")
    if r6_release.get("artifact_sha256") != calibration_hash:
        raise PrecontactError("route target binding does not match r6 release")
    if r6_release.get("release_id") != calibration_release_id:
        raise PrecontactError("route target binding does not match calibration release id")

    geometry = _mapping(route.get("geometry"), "route.geometry")
    a_base = _vec(geometry.get("A_base_m"), 3, "route.geometry.A_base_m")
    n_out = _unit(geometry.get("N_out_base"), "route.geometry.N_out_base")
    p_pre = _vec(geometry.get("P_pre_base_m"), 3, "route.geometry.P_pre_base_m")
    expected_p_pre = tuple(a + PRECONTACT_DISTANCE_M * n for a, n in zip(a_base, n_out))
    _require_equal_vector(p_pre, expected_p_pre, "route P_pre=A+0.050*N")
    _require_equal_vector(
        _vec(geometry.get("target_surface_base_m"), 3, "route.geometry.target_surface_base_m"),
        a_base,
        "route A alias",
    )
    _require_equal_vector(
        _unit(geometry.get("normal_out_base"), "route.geometry.normal_out_base"),
        n_out,
        "route N alias",
    )
    _require_equal_vector(
        _vec(
            geometry.get("precontact_contact_point_base_m"),
            3,
            "route.geometry.precontact_contact_point_base_m",
        ),
        p_pre,
        "route P_pre alias",
    )
    if abs(_finite(geometry.get("exact_precontact_distance_m"), "route exact distance") - PRECONTACT_DISTANCE_M) > 1e-12:
        raise PrecontactError("route pre-contact distance is not exactly 50 mm")

    quantization = _mapping(route.get("quantization"), "route.quantization")
    emitted_raw = quantization.get("emitted_setpoints_rad")
    if not isinstance(emitted_raw, Sequence) or isinstance(emitted_raw, (str, bytes)) or not emitted_raw:
        raise PrecontactError("route emitted setpoints are missing")
    emitted = tuple(_vec(point, 6, "route emitted setpoint") for point in emitted_raw)
    final_setpoint = emitted[-1]
    endpoint = _mapping(route.get("quantized_endpoint_evidence"), "route.quantized_endpoint_evidence")
    if endpoint.get("status") != "validated":
        raise PrecontactError("route quantized endpoint is not validated")
    _require_equal_vector(
        _vec(endpoint.get("checked_positions_rad"), 6, "route checked endpoint"),
        final_setpoint,
        "route final setpoint evidence",
    )
    clearance = _mapping(route.get("clearance_evidence"), "route.clearance_evidence")
    if clearance.get("status") != "clear":
        raise PrecontactError("route clearance evidence is not clear")
    path = clearance.get("quantized_path_setpoints_rad")
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or len(path) != len(emitted) + 1:
        raise PrecontactError("route clearance path does not include standby plus all setpoints")
    if int(clearance.get("checked_waypoint_count") or -1) != len(path):
        raise PrecontactError("route clearance does not cover every waypoint")
    if int(clearance.get("segments_checked") or -1) != len(path) - 1:
        raise PrecontactError("route clearance does not cover every swept segment")
    if clearance.get("continuous_corridor_status") != "clear" or clearance.get("swept_segments_status") != "clear":
        raise PrecontactError("route continuous swept corridor is not clear")
    parsed_path = tuple(_vec(point, 6, "route clearance path setpoint") for point in path)
    start = _vec(quantization.get("start_quantized_rad"), 6, "route quantized start")
    _require_equal_vector(parsed_path[0], start, "route clearance standby start")
    if len(parsed_path[1:]) != len(emitted) or any(
        _distance(actual, expected) > 1e-12
        for actual, expected in zip(parsed_path[1:], emitted)
    ):
        raise PrecontactError("route clearance path does not bind every emitted setpoint")

    return OutboundRouteArtifact(
        artifact_sha256=artifact_hash,
        canonical_payload_sha256=payload_hash,
        route_hash=route_hash,
        manifest_sha256=manifest_sha256,
        target_id=target_id,
        detection_id=detection_id,
        source_stamp_ns=source_stamp_ns,
        source_frame_id=source_frame_id,
        m0_frozen_artifact_sha256=m0_hash,
        calibration_release_id=calibration_release_id,
        calibration_bundle_sha256=calibration_hash,
        a_base_m=a_base,  # type: ignore[arg-type]
        n_out_base=n_out,
        p_pre_base_m=p_pre,  # type: ignore[arg-type]
        final_setpoint_sha256=canonical_setpoint_sha256(final_setpoint),
    )


@dataclass(frozen=True)
class FrozenTarget:
    snapshot_hash: str
    label: str
    target_id: str
    detection_id: str
    source_stamp_ns: int
    source_frame_id: str
    frozen_monotonic: float
    m0_frozen_artifact_sha256: str
    a_base_m: tuple[float, float, float]
    n_out_base: tuple[float, float, float]
    p_pre_base_m: tuple[float, float, float]
    outbound_release_manifest_sha256: str
    execution_release_manifest_sha256: str

    @property
    def precontact_fingertip_base_m(self) -> tuple[float, float, float]:
        return self.p_pre_base_m


def parse_frozen_target(canonical_snapshot_json: str, supplied_hash: str) -> FrozenTarget:
    supplied = _sha256(supplied_hash, "target_snapshot_sha256")
    if not isinstance(canonical_snapshot_json, str) or not canonical_snapshot_json:
        raise PrecontactError("canonical_target_snapshot_json is required")
    try:
        payload = json.loads(canonical_snapshot_json)
    except json.JSONDecodeError as exc:
        raise PrecontactError(f"target snapshot is not JSON: {exc}") from exc
    payload = _mapping(payload, "target snapshot")
    if canonical_json(payload) != canonical_snapshot_json:
        raise PrecontactError("target snapshot is not canonical JSON")
    calculated = canonical_sha256(payload)
    if not secrets.compare_digest(calculated, supplied):
        raise PrecontactError("target snapshot SHA-256 mismatch")
    if payload.get("schema") != TARGET_SCHEMA or str(payload.get("label") or "").strip() != "1":
        raise PrecontactError("target snapshot is not a Button 1 frozen target")
    if payload.get("normal_direction_status") != "VALID_OUTWARD":
        raise PrecontactError("target normal is not verified outward")
    target_id = str(payload.get("target_id") or "").strip()
    detection_id = str(payload.get("detection_id") or "").strip()
    source_frame_id = str(payload.get("source_frame_id") or "").strip()
    source_stamp_ns = payload.get("source_stamp_ns")
    if not target_id or not detection_id or not source_frame_id:
        raise PrecontactError("target snapshot identity is incomplete")
    if isinstance(source_stamp_ns, bool) or not isinstance(source_stamp_ns, int) or source_stamp_ns <= 0:
        raise PrecontactError("source_stamp_ns must be a positive integer")
    frozen_at = _finite(payload.get("frozen_monotonic"), "frozen_monotonic")
    a_base = _vec(payload.get("A_base_m"), 3, "A_base_m")
    n_out = _unit(payload.get("N_out_base"), "N_out_base")
    p_pre = _vec(payload.get("P_pre_base_m"), 3, "P_pre_base_m")
    expected_p_pre = tuple(a + PRECONTACT_DISTANCE_M * n for a, n in zip(a_base, n_out))
    _require_equal_vector(p_pre, expected_p_pre, "snapshot P_pre=A+0.050*N")
    return FrozenTarget(
        snapshot_hash=calculated,
        label="1",
        target_id=target_id,
        detection_id=detection_id,
        source_stamp_ns=source_stamp_ns,
        source_frame_id=source_frame_id,
        frozen_monotonic=frozen_at,
        m0_frozen_artifact_sha256=_sha256(
            payload.get("m0_frozen_artifact_sha256"),
            "m0_frozen_artifact_sha256",
        ),
        a_base_m=a_base,  # type: ignore[arg-type]
        n_out_base=n_out,
        p_pre_base_m=p_pre,  # type: ignore[arg-type]
        outbound_release_manifest_sha256=_sha256(
            payload.get("outbound_release_manifest_sha256"),
            "outbound_release_manifest_sha256",
        ),
        execution_release_manifest_sha256=_sha256(
            payload.get("execution_release_manifest_sha256"),
            "execution_release_manifest_sha256",
        ),
    )


@dataclass(frozen=True)
class ReleaseManifest:
    manifest_hash: str
    execution_enabled: bool
    execution_ready: bool
    command_mode: str
    outbound_release_manifest_sha256: str
    route: OutboundRouteArtifact | None


DEFAULT_RELEASE_MANIFEST = {
    "schema": RELEASE_SCHEMA,
    "release_id": "inert-default",
    "execution_enabled": False,
    "execution_ready": False,
    "command_mode": "T101_SINGLE_JOINT_ONLY",
    "outbound_release_manifest_sha256": "0" * 64,
    "outbound_route_artifact_sha256": "0" * 64,
    "outbound_route_artifact": None,
}


def parse_release_manifest(value: Mapping[str, Any]) -> ReleaseManifest:
    manifest = _mapping(value, "release manifest")
    if manifest.get("schema") != RELEASE_SCHEMA:
        raise PrecontactError("unexpected release manifest schema")
    if not str(manifest.get("release_id") or "").strip():
        raise PrecontactError("release_id is required")
    enabled = manifest.get("execution_enabled") is True
    ready = manifest.get("execution_ready") is True
    command_mode = str(manifest.get("command_mode") or "")
    if command_mode != "T101_SINGLE_JOINT_ONLY":
        raise PrecontactError("release manifest must be T101-only")
    route_value = manifest.get("outbound_route_artifact")
    route: OutboundRouteArtifact | None = None
    outbound_manifest_hash = _sha256(
        manifest.get("outbound_release_manifest_sha256"),
        "outbound_release_manifest_sha256",
    )
    if route_value is not None:
        route = parse_outbound_route_artifact(
            _mapping(route_value, "outbound_route_artifact"),
            manifest.get("outbound_route_artifact_sha256"),
        )
        if route.manifest_sha256 != outbound_manifest_hash:
            raise PrecontactError(
                "outer execution release does not bind the route outbound release manifest"
            )
    elif enabled or ready:
        raise PrecontactError("an executable/ready release requires a full outbound route artifact")
    return ReleaseManifest(
        manifest_hash=canonical_sha256(manifest),
        execution_enabled=enabled,
        execution_ready=ready,
        command_mode=command_mode,
        outbound_release_manifest_sha256=outbound_manifest_hash,
        route=route,
    )


@dataclass(frozen=True)
class AdmissionGates:
    execution_enabled: bool = False
    execution_ready: bool = False
    physical_authorized: bool = False
    feedback_fresh: bool = False
    feedback_aligned: bool = False
    safe_driver_single_source: bool = False
    dry_run_false: bool = False
    fault_clear: bool = False
    torque_all_enabled: bool = False
    voltage_safe: bool = False
    driver_disarmed: bool = False
    at_configured_standby: bool = False
    zero_motion_publishers: bool = False
    auto_torque_recovery_disabled: bool = False
    workspace_keepout_passed: bool = False
    route_outbound_only: bool = False
    route_t101_only: bool = False

    def failures(self) -> tuple[str, ...]:
        return tuple(field for field, value in self.__dict__.items() if value is not True)


@dataclass(frozen=True)
class CompletionEvidence:
    feedback_snapshot_sha256: str
    route_artifact_sha256: str
    final_setpoint_sha256: str
    feedback_fresh: bool
    feedback_aligned: bool
    fault_clear: bool
    armed: bool
    safe_driver_state: str


@dataclass(frozen=True)
class AuthorizationTicket:
    authorization_id: str
    server_nonce: str
    target_snapshot_hash: str
    execution_release_manifest_sha256: str
    outbound_release_manifest_sha256: str
    route_artifact_sha256: str
    m0_frozen_artifact_sha256: str
    expires_monotonic: float
    reserved: bool = False
    consumed: bool = False


@dataclass(frozen=True)
class PrepareResult:
    accepted: bool
    authorization_id: str
    ttl_ms: int
    code: TerminalCode
    reason: str
    release_manifest_sha256: str = ""
    route_artifact_sha256: str = ""
    m0_frozen_artifact_sha256: str = ""


@dataclass(frozen=True)
class ActionResult:
    success: bool
    code: TerminalCode
    reason: str
    state: PrecontactState


class ExecutionResources(Protocol):
    implemented: bool

    def create(self) -> None: ...

    def destroy(self) -> None: ...


class NotImplementedExecutor:
    implemented = False

    def create(self) -> None:
        raise PrecontactError("real motion executor is not implemented")

    def destroy(self) -> None:
        return


class FakeExecutionResources:
    implemented = True

    def __init__(self, *, fail_destroy: bool = False) -> None:
        self.publisher_count = 0
        self.arm_client_count = 0
        self.events: list[str] = []
        self.fail_destroy = fail_destroy

    def create(self) -> None:
        if self.publisher_count or self.arm_client_count:
            raise PrecontactError("execution resources already exist")
        self.publisher_count = 1
        self.arm_client_count = 1
        self.events.append("create")

    def destroy(self) -> None:
        self.publisher_count = 0
        self.arm_client_count = 0
        self.events.append("destroy")
        if self.fail_destroy:
            raise PrecontactError("injected destroy failure")


class PrecontactExecutionCore:
    """One immutable route, one target, one ticket, one outbound attempt."""

    def __init__(
        self,
        release_manifest: Mapping[str, Any] | None = None,
        *,
        ticket_ttl_s: float = MAX_TICKET_TTL_S,
        executor: ExecutionResources | None = None,
    ) -> None:
        if not 0.0 < float(ticket_ttl_s) <= MAX_TICKET_TTL_S:
            raise ValueError(f"ticket_ttl_s must be in (0, {MAX_TICKET_TTL_S:g}]")
        self.release = parse_release_manifest(release_manifest or DEFAULT_RELEASE_MANIFEST)
        self.ticket_ttl_s = float(ticket_ttl_s)
        self.executor: ExecutionResources = executor or NotImplementedExecutor()
        self.state = PrecontactState.INERT
        self._ticket: AuthorizationTicket | None = None
        self._target: FrozenTarget | None = None
        self._terminal_reason = ""
        self._ticket_counter = 0

    @property
    def target(self) -> FrozenTarget | None:
        return self._target

    @property
    def ticket(self) -> AuthorizationTicket | None:
        return self._ticket

    @property
    def terminal_reason(self) -> str:
        return self._terminal_reason

    def _identity_failures(self, target: FrozenTarget) -> tuple[str, ...]:
        route = self.release.route
        if route is None:
            return ("outbound_route_artifact",)
        failures: list[str] = []
        comparisons = (
            (
                target.execution_release_manifest_sha256,
                self.release.manifest_hash,
                "execution_release_manifest_hash",
            ),
            (
                target.outbound_release_manifest_sha256,
                self.release.outbound_release_manifest_sha256,
                "outbound_release_manifest_hash",
            ),
            (
                target.outbound_release_manifest_sha256,
                route.manifest_sha256,
                "route_outbound_release_manifest_hash",
            ),
            (target.m0_frozen_artifact_sha256, route.m0_frozen_artifact_sha256, "m0_frozen_artifact_hash"),
            (target.target_id, route.target_id, "target_id"),
            (target.detection_id, route.detection_id, "detection_id"),
            (target.source_stamp_ns, route.source_stamp_ns, "source_stamp_ns"),
            (target.source_frame_id, route.source_frame_id, "source_frame_id"),
        )
        failures.extend(name for actual, expected, name in comparisons if actual != expected)
        if _distance(target.a_base_m, route.a_base_m) > 1e-9:
            failures.append("A_base_m")
        if _distance(target.n_out_base, route.n_out_base) > 1e-9:
            failures.append("N_out_base")
        if _distance(target.p_pre_base_m, route.p_pre_base_m) > 1e-9:
            failures.append("P_pre_base_m")
        return tuple(failures)

    def _all_failures(self, target: FrozenTarget, gates: AdmissionGates, now: float) -> tuple[str, ...]:
        failures = list(self._identity_failures(target))
        if now < target.frozen_monotonic or now - target.frozen_monotonic > MAX_TICKET_TTL_S:
            failures.append("target_snapshot_ttl")
        if gates.execution_enabled is not True or self.release.execution_enabled is not True:
            failures.append("execution_enabled")
        if gates.execution_ready is not True or self.release.execution_ready is not True:
            failures.append("execution_ready")
        failures.extend(gates.failures())
        return tuple(dict.fromkeys(failures))

    def prepare(
        self,
        canonical_target_snapshot_json: str,
        target_snapshot_sha256: str,
        gates: AdmissionGates,
        *,
        now_monotonic: float,
    ) -> PrepareResult:
        if self.state is not PrecontactState.INERT:
            return PrepareResult(False, "", 0, TerminalCode.REJECTED, f"adapter is {self.state.value}")
        try:
            target = parse_frozen_target(canonical_target_snapshot_json, target_snapshot_sha256)
            failures = self._all_failures(target, gates, float(now_monotonic))
        except PrecontactError as exc:
            return PrepareResult(False, "", 0, TerminalCode.REJECTED, str(exc))
        if failures:
            return PrepareResult(False, "", 0, TerminalCode.REJECTED, "blocked: " + ",".join(failures))
        route = self.release.route
        if route is None:  # Defensive; identity failures already reject this.
            return PrepareResult(False, "", 0, TerminalCode.REJECTED, "missing route artifact")
        self._ticket_counter += 1
        server_nonce = secrets.token_hex(16)
        authorization_id = hashlib.sha256(
            canonical_json(
                {
                    "nonce": server_nonce,
                    "counter": self._ticket_counter,
                    "expires_monotonic": float(now_monotonic) + self.ticket_ttl_s,
                    "target_snapshot_sha256": target.snapshot_hash,
                    "execution_release_manifest_sha256": self.release.manifest_hash,
                    "outbound_release_manifest_sha256": route.manifest_sha256,
                    "route_artifact_sha256": route.artifact_sha256,
                    "m0_frozen_artifact_sha256": route.m0_frozen_artifact_sha256,
                }
            ).encode("utf-8")
        ).hexdigest()
        self._target = target
        self._ticket = AuthorizationTicket(
            authorization_id=authorization_id,
            server_nonce=server_nonce,
            target_snapshot_hash=target.snapshot_hash,
            execution_release_manifest_sha256=self.release.manifest_hash,
            outbound_release_manifest_sha256=route.manifest_sha256,
            route_artifact_sha256=route.artifact_sha256,
            m0_frozen_artifact_sha256=route.m0_frozen_artifact_sha256,
            expires_monotonic=float(now_monotonic) + self.ticket_ttl_s,
        )
        self.state = PrecontactState.PREPARED_NO_MOTION
        return PrepareResult(
            True,
            authorization_id,
            int(self.ticket_ttl_s * 1000),
            TerminalCode.PREPARED,
            "prepared; confirm exactly once",
            self.release.manifest_hash,
            route.artifact_sha256,
            route.m0_frozen_artifact_sha256,
        )

    def reserve_goal(
        self,
        authorization_id: str,
        target_snapshot_sha256: str,
        confirmation_text: str,
        *,
        now_monotonic: float,
    ) -> tuple[bool, str]:
        if self.state is not PrecontactState.PREPARED_NO_MOTION or self._ticket is None:
            return False, "no prepared authorization ticket"
        ticket = self._ticket
        if ticket.reserved or ticket.consumed:
            return False, "authorization ticket is already reserved or consumed"
        if not secrets.compare_digest(ticket.authorization_id, str(authorization_id)):
            return False, "authorization ticket mismatch"
        if not secrets.compare_digest(ticket.target_snapshot_hash, str(target_snapshot_sha256)):
            return False, "target snapshot hash mismatch"
        if str(confirmation_text) != REQUIRED_CONFIRMATION_TEXT:
            return False, "confirmation text mismatch"
        if float(now_monotonic) >= ticket.expires_monotonic:
            return False, "authorization ticket expired"
        self._ticket = AuthorizationTicket(**{**ticket.__dict__, "reserved": True})
        return True, "authorization ticket reserved"

    def admit(
        self,
        authorization_id: str,
        target_snapshot_sha256: str,
        confirmation_text: str,
        gates: AdmissionGates,
        *,
        now_monotonic: float,
    ) -> ActionResult:
        if self.state is not PrecontactState.PREPARED_NO_MOTION or self._ticket is None or self._target is None:
            return self._latch("no prepared authorization ticket", TerminalCode.REJECTED)
        ticket = self._ticket
        if ticket.consumed or not ticket.reserved:
            return self._latch("authorization ticket was not reserved by the Action goal", TerminalCode.REJECTED)
        if not secrets.compare_digest(ticket.authorization_id, str(authorization_id)):
            return self._latch("authorization ticket mismatch", TerminalCode.REJECTED)
        if not secrets.compare_digest(ticket.target_snapshot_hash, str(target_snapshot_sha256)):
            return self._latch("target snapshot hash mismatch", TerminalCode.REJECTED)
        if str(confirmation_text) != REQUIRED_CONFIRMATION_TEXT:
            return self._latch("confirmation text mismatch", TerminalCode.REJECTED)
        if float(now_monotonic) >= ticket.expires_monotonic:
            return self._latch("authorization ticket expired", TerminalCode.REJECTED)
        route = self.release.route
        if route is None or (
            ticket.execution_release_manifest_sha256 != self.release.manifest_hash
            or ticket.outbound_release_manifest_sha256 != route.manifest_sha256
            or ticket.outbound_release_manifest_sha256
            != self.release.outbound_release_manifest_sha256
            or ticket.route_artifact_sha256 != route.artifact_sha256
            or ticket.m0_frozen_artifact_sha256 != route.m0_frozen_artifact_sha256
        ):
            return self._latch("ticket release/route identity changed", TerminalCode.REJECTED)
        self._ticket = AuthorizationTicket(**{**ticket.__dict__, "consumed": True})
        failures = self._all_failures(self._target, gates, float(now_monotonic))
        if failures:
            return self._latch("admission blocked: " + ",".join(failures), TerminalCode.REJECTED)
        if not self.executor.implemented:
            return self._latch("real publisher path is NOT_IMPLEMENTED", TerminalCode.NOT_IMPLEMENTED)
        try:
            self.executor.create()
        except Exception as exc:  # pragma: no cover - defensive executor boundary
            return self._latch(f"execution resource creation failed: {exc}", TerminalCode.ABORTED)
        self.state = PrecontactState.EXECUTING_OUTBOUND_TO_PRECONTACT
        return ActionResult(True, TerminalCode.EXECUTING, "outbound-only execution admitted", self.state)

    def complete_disarmed(self, evidence: CompletionEvidence) -> ActionResult:
        """Complete only from explicit fresh route-bound driver evidence."""

        if self.state is not PrecontactState.EXECUTING_OUTBOUND_TO_PRECONTACT:
            return self._latch("completion reported outside execution", TerminalCode.ABORTED)
        route = self.release.route
        failures: list[str] = []
        try:
            _sha256(evidence.feedback_snapshot_sha256, "completion.feedback_snapshot_sha256")
            _sha256(evidence.route_artifact_sha256, "completion.route_artifact_sha256")
            _sha256(evidence.final_setpoint_sha256, "completion.final_setpoint_sha256")
        except PrecontactError as exc:
            failures.append(str(exc))
        if route is None or evidence.route_artifact_sha256 != route.artifact_sha256:
            failures.append("completion_route_artifact")
        if route is None or evidence.final_setpoint_sha256 != route.final_setpoint_sha256:
            failures.append("completion_final_setpoint")
        if evidence.feedback_fresh is not True:
            failures.append("completion_feedback_fresh")
        if evidence.feedback_aligned is not True:
            failures.append("completion_feedback_aligned")
        if evidence.fault_clear is not True:
            failures.append("completion_fault_clear")
        if evidence.armed is not False:
            failures.append("completion_armed_false")
        if evidence.safe_driver_state != "CONVERGED_DISARMED":
            failures.append("completion_driver_state")
        if failures:
            return self._latch("completion blocked: " + ",".join(failures), TerminalCode.ABORTED)
        destroy_error = self._destroy_resources()
        if destroy_error:
            self.state = PrecontactState.ABORT_LATCHED
            self._terminal_reason = f"completion resource destroy failed: {destroy_error}"
            return ActionResult(False, TerminalCode.ABORTED, self._terminal_reason, self.state)
        self.state = PrecontactState.COMPLETE_DISARMED
        self._terminal_reason = "verified fresh aligned CONVERGED_DISARMED; resources destroyed"
        return ActionResult(True, TerminalCode.COMPLETE, self._terminal_reason, self.state)

    def abort(self, reason: str, *, cancelled: bool = False) -> ActionResult:
        return self._latch(reason, TerminalCode.CANCELLED if cancelled else TerminalCode.ABORTED)

    def _destroy_resources(self) -> str:
        try:
            self.executor.destroy()
        except Exception as exc:  # noqa: BLE001 - destruction failures remain latched
            return str(exc)
        return ""

    def _latch(self, reason: str, code: TerminalCode) -> ActionResult:
        destroy_error = self._destroy_resources()
        self.state = PrecontactState.ABORT_LATCHED
        self._terminal_reason = str(reason)
        if destroy_error:
            self._terminal_reason += f"; resource destroy failed: {destroy_error}"
        return ActionResult(False, code, self._terminal_reason, self.state)


__all__ = [
    "ActionResult",
    "AdmissionGates",
    "AuthorizationTicket",
    "CompletionEvidence",
    "DEFAULT_RELEASE_MANIFEST",
    "FakeExecutionResources",
    "FrozenTarget",
    "MAX_TICKET_TTL_S",
    "NotImplementedExecutor",
    "OutboundRouteArtifact",
    "PRECONTACT_DISTANCE_M",
    "PrecontactError",
    "PrecontactExecutionCore",
    "PrecontactState",
    "PrepareResult",
    "REQUIRED_CONFIRMATION_TEXT",
    "RELEASE_SCHEMA",
    "ROUTE_PHASE",
    "ROUTE_SCHEMA",
    "TARGET_SCHEMA",
    "TerminalCode",
    "canonical_json",
    "canonical_route_artifact_sha256",
    "canonical_setpoint_sha256",
    "canonical_sha256",
    "parse_frozen_target",
    "parse_outbound_route_artifact",
    "parse_release_manifest",
]
