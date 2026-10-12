"""Build and validate a narrowly executable T102 research-prototype release.

The release is intentionally not motion authority.  It binds one positive
outbound-only route to the exact diagnostic route and continuous-clearance
result from which it was reviewed.  Runtime authority still comes from the
short-lived Prepare ticket and the operator's second Move confirmation.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from typing import Any, Mapping, Sequence


ROUTE_SCHEMA = "t102_supervised_outbound_route_v1"
RELEASE_SCHEMA = "t102_supervised_prototype_execution_release_v1"
ROUTE_MODE = "SUPERVISED_RESEARCH_PROTOTYPE_EXECUTION"
COMMAND_INTERFACE = "ROS2_ACTION_ROARM_T102_OUTBOUND_EXECUTOR"
ENDPOINT_SCOPE = "BUTTON1_120MM_NO_CONTACT_OUTBOUND_ONLY"
CLEARANCE_PASS = "PROVISIONAL_CONTINUOUS_CLEARANCE_PASS"
WAYPOINT_TOLERANCE_RAD = 0.005
HARD_TOLERANCE_CEILING_RAD = 0.02
MAX_INCREMENT_DEG = 1.10
MIN_EFFECTIVE_INCREMENT_DEG = 1.0
HISTORICAL_RESIDUAL_RAD = 0.036815538


class SupervisedReleaseError(ValueError):
    """A positive route/release binding is malformed or unsafe."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SupervisedReleaseError("value is not canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise SupervisedReleaseError(f"{field} must be a lowercase SHA-256")
    return result


def _vector(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise SupervisedReleaseError(f"{field} must contain six finite values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SupervisedReleaseError(f"{field} must contain six finite values") from exc
    if not all(math.isfinite(item) for item in result):
        raise SupervisedReleaseError(f"{field} must contain six finite values")
    return result


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    supplied = _sha(unsigned.pop(field, None), field)
    if not secrets.compare_digest(canonical_sha256(unsigned), supplied):
        raise SupervisedReleaseError(f"{field} mismatch")
    return supplied


def _route_interval_metrics(route: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float, float]:
    start = _vector(route.get("start_positions_rad"), "start_positions_rad")
    raw_points = route.get("setpoints_rad")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)) or not raw_points:
        raise SupervisedReleaseError("route contains no setpoints")
    previous = start
    metrics: list[dict[str, Any]] = []
    maxima: list[float] = []
    for index, raw_point in enumerate(raw_points):
        point = _vector(raw_point, f"setpoints_rad[{index}]")
        deltas_deg = [math.degrees(abs(after - before)) for before, after in zip(previous, point)]
        maximum = max(deltas_deg)
        maxima.append(maximum)
        metrics.append(
            {
                "interval_index": index,
                "joint_deltas_deg": deltas_deg,
                "maximum_increment_deg": maximum,
                "effective_motion_over_1deg": maximum > MIN_EFFECTIVE_INCREMENT_DEG,
                "worst_case_next_gap_deg": math.degrees(WAYPOINT_TOLERANCE_RAD) + maximum,
            }
        )
        previous = point
    return metrics, max(maxima), min(maxima)


def build_supervised_executable_route(
    diagnostic_route: Mapping[str, Any],
    *,
    diagnostic_route_sha256: str,
    route_identity: str,
) -> dict[str, Any]:
    """Derive a positive route from a separately validated diagnostic route.

    This does not flip the source object.  It copies only the exact start,
    endpoint, setpoints and frozen bindings after independently checking the
    source diagnostic self-hash and its inert markers.
    """

    if diagnostic_route.get("route_mode") != "OFFLINE_DIAGNOSTIC_ONLY":
        raise SupervisedReleaseError("source route is not diagnostic-only")
    if diagnostic_route.get("command_interface") != "NOT_IMPLEMENTED":
        raise SupervisedReleaseError("source route command interface drift")
    if diagnostic_route.get("diagnostic_only") is not True:
        raise SupervisedReleaseError("source route must retain diagnostic_only=true")
    embedded = _self_hash(diagnostic_route, "route_sha256")
    if not secrets.compare_digest(embedded, _sha(diagnostic_route_sha256, "diagnostic_route_sha256")):
        raise SupervisedReleaseError("source diagnostic route hash differs")
    bindings = diagnostic_route.get("frozen_bindings")
    if not isinstance(bindings, Mapping):
        raise SupervisedReleaseError("source diagnostic bindings are missing")
    frozen_bindings = {key: _sha(value, f"frozen_bindings.{key}") for key, value in bindings.items()}
    if set(frozen_bindings) != {"m0_target_sha256", "fn_sha256", "r6_sha256", "provenance_sha256"}:
        raise SupervisedReleaseError("unexpected frozen binding set")
    start = _vector(diagnostic_route.get("start_positions_rad"), "start_positions_rad")
    endpoint = _vector(
        diagnostic_route.get("static_ik_candidate_positions_rad"),
        "static_ik_candidate_positions_rad",
    )
    raw_points = diagnostic_route.get("setpoints_rad")
    points = [_vector(point, f"setpoints_rad[{index}]") for index, point in enumerate(raw_points or [])]
    if not points or points[-1] != endpoint:
        raise SupervisedReleaseError("diagnostic route does not end at its IK candidate")
    route: dict[str, Any] = {
        "schema": ROUTE_SCHEMA,
        "route_identity": str(route_identity),
        "source_diagnostic_route_sha256": embedded,
        "route_mode": ROUTE_MODE,
        "command_interface": COMMAND_INTERFACE,
        "endpoint_scope": ENDPOINT_SCOPE,
        "phase": "OUTBOUND_TO_PRECONTACT",
        "phases": ["OUTBOUND_TO_PRECONTACT"],
        "joint_path": "SIMULTANEOUS_T102_JOINT_SPACE_CHORD",
        "frozen_bindings": frozen_bindings,
        "start_positions_rad": list(start),
        "static_ik_candidate_positions_rad": list(endpoint),
        "setpoints_rad": [list(point) for point in points],
        "waypoint_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
        "hard_tolerance_ceiling_rad": HARD_TOLERANCE_CEILING_RAD,
        "max_increment_limit_deg": MAX_INCREMENT_DEG,
        "min_effective_increment_limit_deg": MIN_EFFECTIVE_INCREMENT_DEG,
        "historical_convergence_reference": {
            "joint": "link2_to_link3",
            "residual_rad": HISTORICAL_RESIDUAL_RAD,
            "residual_deg": math.degrees(HISTORICAL_RESIDUAL_RAD),
            "policy": "FIRST_UNDERTRAVEL_ABORTS_NO_RETRY",
        },
        "physical_convergence_guaranteed": False,
        "convergence_policy": "FIRST_UNDERTRAVEL_ABORTS_NO_RETRY",
        "no_contact": True,
        "contact_authorized": False,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "execution_ready": True,
        "motion_authorized": False,
        "trajectory_generated": True,
        "command_generated": True,
        "adapter_consumable": True,
        "diagnostic_only": False,
    }
    metrics, maximum, minimum = _route_interval_metrics(route)
    route["interval_convergence_bounds"] = metrics
    route["maximum_increment_deg"] = maximum
    route["minimum_effective_increment_deg"] = minimum
    route["route_sha256"] = canonical_sha256(route)
    validate_supervised_executable_route(route)
    return route


def validate_supervised_executable_route(route: Mapping[str, Any]) -> str:
    if not isinstance(route, Mapping):
        raise SupervisedReleaseError("route must be an object")
    required = {
        "schema": ROUTE_SCHEMA,
        "route_mode": ROUTE_MODE,
        "command_interface": COMMAND_INTERFACE,
        "endpoint_scope": ENDPOINT_SCOPE,
        "phase": "OUTBOUND_TO_PRECONTACT",
        "phases": ["OUTBOUND_TO_PRECONTACT"],
        "no_contact": True,
        "contact_authorized": False,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "execution_ready": True,
        "motion_authorized": False,
        "trajectory_generated": True,
        "command_generated": True,
        "adapter_consumable": True,
        "diagnostic_only": False,
        "physical_convergence_guaranteed": False,
        "convergence_policy": "FIRST_UNDERTRAVEL_ABORTS_NO_RETRY",
    }
    for field, expected in required.items():
        if route.get(field) != expected:
            raise SupervisedReleaseError(f"executable route field mismatch: {field}")
    _sha(route.get("source_diagnostic_route_sha256"), "source_diagnostic_route_sha256")
    bindings = route.get("frozen_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"m0_target_sha256", "fn_sha256", "r6_sha256", "provenance_sha256"}:
        raise SupervisedReleaseError("executable route bindings are missing")
    for key, value in bindings.items():
        _sha(value, f"frozen_bindings.{key}")
    tolerance = float(route.get("waypoint_tolerance_rad", float("nan")))
    if tolerance != WAYPOINT_TOLERANCE_RAD or not 0.0 < tolerance <= HARD_TOLERANCE_CEILING_RAD:
        raise SupervisedReleaseError("waypoint tolerance must be exactly 0.005 rad")
    if route.get("hard_tolerance_ceiling_rad") != HARD_TOLERANCE_CEILING_RAD:
        raise SupervisedReleaseError("hard tolerance ceiling drift")
    metrics, maximum, minimum = _route_interval_metrics(route)
    if maximum > MAX_INCREMENT_DEG + 1e-9:
        raise SupervisedReleaseError("route increment exceeds 1.10 deg")
    if minimum <= MIN_EFFECTIVE_INCREMENT_DEG:
        raise SupervisedReleaseError("route interval lacks effective motion over 1.0 deg")
    if any(item["worst_case_next_gap_deg"] >= 2.0 for item in metrics):
        raise SupervisedReleaseError("route interval can exceed the 2.0 deg soft band")
    if route.get("interval_convergence_bounds") != metrics:
        raise SupervisedReleaseError("interval convergence bounds drift")
    if abs(float(route.get("maximum_increment_deg", float("nan"))) - maximum) > 1e-9:
        raise SupervisedReleaseError("maximum increment declaration drift")
    if abs(float(route.get("minimum_effective_increment_deg", float("nan"))) - minimum) > 1e-9:
        raise SupervisedReleaseError("minimum effective increment declaration drift")
    if _vector(route.get("setpoints_rad")[-1], "final setpoint") != _vector(
        route.get("static_ik_candidate_positions_rad"), "static_ik_candidate_positions_rad"
    ):
        raise SupervisedReleaseError("route endpoint drift")
    historical = route.get("historical_convergence_reference")
    if not isinstance(historical, Mapping) or historical.get("policy") != "FIRST_UNDERTRAVEL_ABORTS_NO_RETRY":
        raise SupervisedReleaseError("historical convergence reference missing")
    if abs(float(historical.get("residual_rad", float("nan"))) - HISTORICAL_RESIDUAL_RAD) > 1e-12:
        raise SupervisedReleaseError("historical residual drift")
    return _self_hash(route, "route_sha256")


def build_supervised_release(
    route: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    issued_at_unix_s: float | None = None,
    expires_at_unix_s: float | None = None,
) -> dict[str, Any]:
    route_sha = validate_supervised_executable_route(route)
    now = time.time() if issued_at_unix_s is None else float(issued_at_unix_s)
    expiry = now + 4 * 3600 if expires_at_unix_s is None else float(expires_at_unix_s)
    if not math.isfinite(now) or not math.isfinite(expiry) or not now < expiry <= now + 24 * 3600:
        raise SupervisedReleaseError("release expiry must be within 24 hours")
    required_binding_keys = {
        "target_artifact_file_sha256",
        "q0_sha256",
        "fn_sha256",
        "r6_sha256",
        "camera_acceptance_sha256",
        "ik_fk_evidence_file_sha256",
        "geometry_profile_file_sha256",
        "continuous_clearance_file_sha256",
        "continuous_clearance_canonical_sha256",
        "source_diagnostic_route_sha256",
        "executor_source_bundle_sha256",
    }
    if set(bindings) != required_binding_keys:
        raise SupervisedReleaseError("release bindings set is incomplete or expanded")
    sealed_bindings = {key: _sha(value, f"bindings.{key}") for key, value in bindings.items()}
    if sealed_bindings["fn_sha256"] != route["frozen_bindings"]["fn_sha256"]:
        raise SupervisedReleaseError("Fn binding differs from route")
    if sealed_bindings["r6_sha256"] != route["frozen_bindings"]["r6_sha256"]:
        raise SupervisedReleaseError("r6 binding differs from route")
    if sealed_bindings["source_diagnostic_route_sha256"] != route["source_diagnostic_route_sha256"]:
        raise SupervisedReleaseError("diagnostic route binding differs from route")
    release: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "route_mode": ROUTE_MODE,
        "command_interface": COMMAND_INTERFACE,
        "endpoint_scope": ENDPOINT_SCOPE,
        "execution_ready": True,
        "adapter_consumable": True,
        "diagnostic_only": False,
        "trajectory_generated": True,
        "motion_authorized": False,
        "contact_authorized": False,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "single_use": True,
        "runtime_ticket_required": True,
        "issued_at_unix_s": now,
        "expires_at_unix_s": expiry,
        "route_sha256": route_sha,
        "target_snapshot_sha256": route["frozen_bindings"]["m0_target_sha256"],
        "clearance_result": CLEARANCE_PASS,
        "waypoint_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
        "hard_tolerance_ceiling_rad": HARD_TOLERANCE_CEILING_RAD,
        "maximum_increment_deg": route["maximum_increment_deg"],
        "minimum_effective_increment_deg": route["minimum_effective_increment_deg"],
        "convergence_policy": "FIRST_UNDERTRAVEL_ABORTS_NO_RETRY",
        "physical_convergence_guaranteed": False,
        "bindings": sealed_bindings,
    }
    release["release_sha256"] = canonical_sha256(release)
    validate_supervised_release(release, route, now_unix_s=now)
    return release


def validate_supervised_release(
    release: Mapping[str, Any],
    route: Mapping[str, Any],
    *,
    now_unix_s: float | None = None,
) -> str:
    route_sha = validate_supervised_executable_route(route)
    required = {
        "schema": RELEASE_SCHEMA,
        "route_mode": ROUTE_MODE,
        "command_interface": COMMAND_INTERFACE,
        "endpoint_scope": ENDPOINT_SCOPE,
        "execution_ready": True,
        "adapter_consumable": True,
        "diagnostic_only": False,
        "trajectory_generated": True,
        "motion_authorized": False,
        "contact_authorized": False,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "single_use": True,
        "runtime_ticket_required": True,
        "clearance_result": CLEARANCE_PASS,
        "waypoint_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
        "hard_tolerance_ceiling_rad": HARD_TOLERANCE_CEILING_RAD,
        "convergence_policy": "FIRST_UNDERTRAVEL_ABORTS_NO_RETRY",
        "physical_convergence_guaranteed": False,
    }
    for field, expected in required.items():
        if release.get(field) != expected:
            raise SupervisedReleaseError(f"release field mismatch: {field}")
    if not secrets.compare_digest(_sha(release.get("route_sha256"), "route_sha256"), route_sha):
        raise SupervisedReleaseError("release route hash mismatch")
    if release.get("target_snapshot_sha256") != route["frozen_bindings"]["m0_target_sha256"]:
        raise SupervisedReleaseError("release target binding mismatch")
    if abs(float(release.get("maximum_increment_deg", float("nan"))) - route["maximum_increment_deg"]) > 1e-9:
        raise SupervisedReleaseError("release maximum increment drift")
    if abs(float(release.get("minimum_effective_increment_deg", float("nan"))) - route["minimum_effective_increment_deg"]) > 1e-9:
        raise SupervisedReleaseError("release minimum increment drift")
    issued = float(release.get("issued_at_unix_s", float("nan")))
    expiry = float(release.get("expires_at_unix_s", float("nan")))
    if not math.isfinite(issued) or not math.isfinite(expiry) or not issued < expiry <= issued + 24 * 3600:
        raise SupervisedReleaseError("release lifetime invalid")
    now = time.time() if now_unix_s is None else float(now_unix_s)
    if not issued <= now <= expiry:
        raise SupervisedReleaseError("release is not current")
    bindings = release.get("bindings")
    if not isinstance(bindings, Mapping):
        raise SupervisedReleaseError("release bindings missing")
    for key, value in bindings.items():
        _sha(value, f"bindings.{key}")
    if bindings.get("source_diagnostic_route_sha256") != route["source_diagnostic_route_sha256"]:
        raise SupervisedReleaseError("source diagnostic binding mismatch")
    if bindings.get("fn_sha256") != route["frozen_bindings"]["fn_sha256"]:
        raise SupervisedReleaseError("Fn binding mismatch")
    if bindings.get("r6_sha256") != route["frozen_bindings"]["r6_sha256"]:
        raise SupervisedReleaseError("r6 binding mismatch")
    return _self_hash(release, "release_sha256")


__all__ = [
    "CLEARANCE_PASS", "COMMAND_INTERFACE", "ENDPOINT_SCOPE", "HARD_TOLERANCE_CEILING_RAD",
    "MAX_INCREMENT_DEG", "MIN_EFFECTIVE_INCREMENT_DEG", "RELEASE_SCHEMA", "ROUTE_MODE",
    "ROUTE_SCHEMA", "SupervisedReleaseError", "WAYPOINT_TOLERANCE_RAD",
    "build_supervised_executable_route", "build_supervised_release", "canonical_sha256",
    "validate_supervised_executable_route", "validate_supervised_release",
]
