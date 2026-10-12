"""Pure offline builder for a bounded, direct T102 pre-contact proposal.

This is deliberately a diagnostic artifact, not a ROS/serial command builder.
It turns a *previously verified* static IK start/end pair into simultaneous
joint-space chord setpoints.  The artifact is permanently non-executable:
the future dedicated motion owner must independently revalidate runtime
feedback, collision evidence and an operator ticket before it could use the
numbers here.

Bounding the commanded deltas (segmentation) is a route-planning restriction
only.  It does **not** reduce or eliminate physical T102 undertravel: the
first field waypoint stopped 0.036815538 rad (2.11 deg) short on the elbow
and the safe driver latched a fault after its single allowed reissue, just
beyond the driver's 2x-tolerance soft-converge band.  This artifact therefore
keeps every authority flag false and documents that no offline patch can
guarantee physical convergence without driver-side or hardware changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from press.t102_outbound_executor_core import FINAL_JOINT_TOLERANCE_RAD


SCHEMA = "t102_outbound_only_route_v1"
PHASE = "OUTBOUND_TO_PRECONTACT"
EXPECTED_DELTA_CAPS_DEG = (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
# Mirror of the safe driver's 4096-step encoder grid (read-only reference;
# this builder never sends a command).
ENCODER_STEP_RAD = 2.0 * math.pi / 4096.0
# A partial first field trial faulted with residual elbow error.  Keep the
# direct simultaneous chord, but make each command no larger than half of the
# reviewed profile limit.  This is a route-planning restriction only.
DEFAULT_MAX_PROFILE_DELTA_FRACTION = 0.5
DEFAULT_DRIVER_PROFILE = (
    Path(__file__).resolve().parents[2] / "config/agx_quick_control_cartesian_diagnostic.yaml"
)
# Observed first field waypoint (2026-08): the elbow stopped this far short of
# the absolute T102 target after one allowed reissue and the safe driver then
# latched a fault.  0.036815538 rad = 2.109... deg, just beyond the reviewed
# profile's 2x-tolerance soft band.  The value is kept as a read-only field
# reference so offline diagnostics can deterministically classify residuals
# against the driver policy without importing or editing the driver.
OBSERVED_FIELD_FAILURE_RESIDUAL_RAD = 0.036815538
OBSERVED_FIELD_FAILURE_JOINT = "link2_to_link3"


class T102RouteBlocked(ValueError):
    """The static candidate or its diagnostic-only contract was invalid."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise T102RouteBlocked("route value is not canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise T102RouteBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _vector(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise T102RouteBlocked(f"{field} must contain exactly six finite values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise T102RouteBlocked(f"{field} must contain exactly six finite values") from exc
    if not all(math.isfinite(item) for item in result):
        raise T102RouteBlocked(f"{field} must contain exactly six finite values")
    return result


def load_confirmed_t102_delta_caps_rad(
    profile_path: str | Path = DEFAULT_DRIVER_PROFILE,
) -> tuple[float, ...]:
    """Read the deployed diagnostic profile and fail if the known caps drift.

    The builder does not import the driver and cannot loosen its limits.  This
    source-bound check makes the six caps used in the diagnostic artifact
    explicit rather than silently inheriting a possibly different default.
    """

    path = Path(profile_path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise T102RouteBlocked(f"driver profile is unreadable: {exc}") from exc
    match = re.search(r"^\s*joint_max_delta_deg:\s*\[([^]]+)\]", text, flags=re.MULTILINE)
    if match is None:
        raise T102RouteBlocked("driver profile does not declare joint_max_delta_deg")
    try:
        caps = tuple(float(item.strip()) for item in match.group(1).split(","))
    except ValueError as exc:
        raise T102RouteBlocked("driver profile has invalid joint_max_delta_deg") from exc
    if caps != EXPECTED_DELTA_CAPS_DEG:
        raise T102RouteBlocked("driver profile caps differ from the reviewed diagnostic caps")
    return tuple(math.radians(value) for value in caps)


def load_declared_t102_driver_tolerance_deg(
    profile_path: str | Path = DEFAULT_DRIVER_PROFILE,
) -> float:
    """Read the deployed profile's convergence tolerance for diagnostics only.

    The route never loosens or tightens the driver's gate; it merely embeds
    the declared tolerance and the driver's documented 2x soft-converge band
    so a reviewer can compare a segment against what the driver can absorb
    after its single stalled-T102 reissue.
    """

    path = Path(profile_path).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise T102RouteBlocked(f"driver profile is unreadable: {exc}") from exc
    match = re.search(r"^\s*tolerance_deg:\s*([0-9.]+)", text, flags=re.MULTILINE)
    if match is None:
        raise T102RouteBlocked("driver profile does not declare tolerance_deg")
    try:
        tolerance = float(match.group(1))
    except ValueError as exc:
        raise T102RouteBlocked("driver profile has invalid tolerance_deg") from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise T102RouteBlocked("driver profile tolerance_deg must be positive")
    return tolerance


def classify_t102_terminal_residual(
    residual_rad: float,
    driver_tolerance_deg: float,
) -> dict[str, Any]:
    """Classify a measured T102 terminal residual against the driver policy.

    The deployed safe driver permits at most one stalled reissue and then
    soft-converges a stable residual up to ``2 * tolerance_deg``; anything
    larger latches a command convergence fault.  This classification is a
    read-only mirror of that existing policy.  It never widens the executor's
    own final joint gate (0.02 rad) and never grants execution authority.
    """

    try:
        residual = float(residual_rad)
        tolerance = float(driver_tolerance_deg)
    except (TypeError, ValueError) as exc:
        raise T102RouteBlocked("terminal residual classification needs finite numbers") from exc
    if not math.isfinite(residual) or residual < 0.0:
        raise T102RouteBlocked("terminal residual must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise T102RouteBlocked("driver tolerance must be finite and positive")
    band_deg = 2.0 * tolerance
    residual_deg = math.degrees(residual)
    within_band = residual_deg <= band_deg
    return {
        "residual_rad": residual,
        "residual_deg": residual_deg,
        "soft_converge_band_deg": band_deg,
        "within_soft_converge_band": within_band,
        "driver_terminal_policy": (
            "SOFT_CONVERGE_ELIGIBLE_AFTER_SINGLE_REISSUE"
            if within_band
            else "DRIVER_FAULTS_AFTER_SINGLE_REISSUE"
        ),
        "note": (
            "existing driver policy only; executor keeps its own 0.02 rad "
            "final joint gate and never invents convergence"
        ),
    }


def _bounded_chord(start: Sequence[float], end: Sequence[float], caps: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    """Return simultaneous T102 endpoints along one bounded joint-space chord."""

    segments = max(1, max(math.ceil(abs(target - origin) / cap) for origin, target, cap in zip(start, end, caps)))
    setpoints = tuple(
        tuple(end) if index == segments else tuple(
            origin + (index / segments) * (target - origin) for origin, target in zip(start, end)
        )
        for index in range(1, segments + 1)
    )
    previous = tuple(start)
    for setpoint in setpoints:
        if any(abs(value - prior) > cap + 1e-12 for value, prior, cap in zip(setpoint, previous, caps)):
            raise T102RouteBlocked("bounded chord exceeds a confirmed T102 delta cap")
        previous = setpoint
    if setpoints[-1] != tuple(end):
        raise T102RouteBlocked("bounded chord does not terminate at the static IK candidate")
    return setpoints


def _conservative_caps(caps: Sequence[float], fraction: float) -> tuple[float, ...]:
    try:
        fraction = float(fraction)
    except (TypeError, ValueError) as exc:
        raise T102RouteBlocked("max_profile_delta_fraction must be finite in (0, 1]") from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise T102RouteBlocked("max_profile_delta_fraction must be finite in (0, 1]")
    return tuple(cap * fraction for cap in caps)


def build_t102_outbound_diagnostic_route(
    *,
    verified_start_positions_rad: Sequence[float],
    verified_static_ik_candidate_rad: Sequence[float],
    frozen_bindings: Mapping[str, Any],
    profile_path: str | Path = DEFAULT_DRIVER_PROFILE,
    max_profile_delta_fraction: float = DEFAULT_MAX_PROFILE_DELTA_FRACTION,
) -> dict[str, Any]:
    """Build a sealed display/audit-only direct outbound candidate.

    ``frozen_bindings`` must carry hashes for the exact M0 target, active Fn,
    r6 calibration bundle and provenance/release package.  No legacy phases
    are represented, and every possible authority flag is false.
    """

    start = _vector(verified_start_positions_rad, "verified_start_positions_rad")
    candidate = _vector(verified_static_ik_candidate_rad, "verified_static_ik_candidate_rad")
    if start == candidate:
        raise T102RouteBlocked("static IK candidate must differ from the verified start")
    if not isinstance(frozen_bindings, Mapping):
        raise T102RouteBlocked("frozen_bindings must be an object")
    expected_binding_keys = {
        "m0_target_sha256", "fn_sha256", "r6_sha256", "provenance_sha256",
    }
    if set(frozen_bindings) != expected_binding_keys:
        raise T102RouteBlocked("frozen_bindings must contain exactly M0/Fn/r6/provenance hashes")
    bindings = {key: _sha256(frozen_bindings[key], key) for key in sorted(expected_binding_keys)}
    profile_caps = load_confirmed_t102_delta_caps_rad(profile_path)
    driver_tolerance_deg = load_declared_t102_driver_tolerance_deg(profile_path)
    caps = _conservative_caps(profile_caps, max_profile_delta_fraction)
    fraction = float(max_profile_delta_fraction)
    setpoints = _bounded_chord(start, candidate, caps)
    segment_deltas = [
        tuple(abs(after - before) for before, after in zip(previous, point))
        for previous, point in zip((start, *setpoints[:-1]), setpoints)
    ]
    first_segment_max_delta_rad = max(segment_deltas[0])
    driver_soft_converge_band_deg = 2.0 * driver_tolerance_deg

    route: dict[str, Any] = {
        "schema": SCHEMA,
        "route_mode": "OFFLINE_DIAGNOSTIC_ONLY",
        "phase": PHASE,
        "phases": [PHASE],
        "joint_path": "SIMULTANEOUS_T102_JOINT_SPACE_CHORD",
        "command_interface": "NOT_IMPLEMENTED",
        "frozen_bindings": bindings,
        "start_positions_rad": list(start),
        "static_ik_candidate_positions_rad": list(candidate),
        "setpoints_rad": [list(point) for point in setpoints],
        "driver_delta_caps_deg": list(EXPECTED_DELTA_CAPS_DEG),
        "max_profile_delta_fraction": fraction,
        "segment_delta_caps_deg": [fraction * cap for cap in EXPECTED_DELTA_CAPS_DEG],
        "encoder_step_rad": ENCODER_STEP_RAD,
        "driver_tolerance_deg": driver_tolerance_deg,
        "driver_soft_converge_band_deg": driver_soft_converge_band_deg,
        "segment_deltas_encoder_steps": [
            [delta / ENCODER_STEP_RAD for delta in segment] for segment in segment_deltas
        ],
        "first_segment_max_delta_rad": first_segment_max_delta_rad,
        "first_segment_max_delta_deg": math.degrees(first_segment_max_delta_rad),
        "first_segment_exceeds_soft_converge_band": (
            math.degrees(first_segment_max_delta_rad) > driver_soft_converge_band_deg
        ),
        "terminal_residual_policy": {
            "driver_tolerance_deg": driver_tolerance_deg,
            "soft_converge_band_deg": driver_soft_converge_band_deg,
            "single_stalled_reissue": True,
            "within_band_driver_outcome": "SOFT_CONVERGE_AFTER_ONE_REISSUE",
            "beyond_band_driver_outcome": "FAULT_LATCHED_AFTER_ONE_REISSUE",
            "executor_keeps_own_final_joint_gate_rad": FINAL_JOINT_TOLERANCE_RAD,
        },
        "observed_field_failure_reference": {
            "source": "first field outbound waypoint after one allowed reissue",
            "joint": OBSERVED_FIELD_FAILURE_JOINT,
            "max_residual_rad": OBSERVED_FIELD_FAILURE_RESIDUAL_RAD,
            "max_residual_deg": math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD),
            "soft_converge_band_deg": driver_soft_converge_band_deg,
            "residual_exceeded_soft_converge_band": (
                math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD) > driver_soft_converge_band_deg
            ),
            "driver_outcome": "FAULT_LATCHED_AFTER_ONE_REISSUE",
        },
        "physical_convergence_guaranteed": False,
        "segmentation_does_not_fix_physical_undertravel": True,
        "execution_blocked_reason": (
            "first field waypoint stopped 2.11 deg short and the safe driver "
            "faulted after its single allowed reissue; bounding commanded "
            "deltas into chord segments cannot guarantee physical convergence, "
            "so the route stays non-executable until a driver-side or hardware "
            "fix is reviewed"
        ),
        "no_contact": True,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "command_generated": False,
        "adapter_consumable": False,
        "diagnostic_only": True,
        "required_runtime_gates": [
            "INDEPENDENT_IK_FK_RELEASE",
            "JOINT_LIMIT_AND_CONTINUOUS_KEEPOUT_RELEASE",
            "FRESH_SAFE_DRIVER_FEEDBACK",
            "EXCLUSIVE_MOTION_OWNER",
            "EXPLICIT_OPERATOR_MOVE_TICKET",
        ],
    }
    route["route_sha256"] = canonical_sha256(route)
    validate_t102_outbound_diagnostic_route(route)
    return route


def validate_t102_outbound_diagnostic_route(route: Mapping[str, Any]) -> None:
    """Verify the artifact remains a bounded, no-authority proposal."""

    if not isinstance(route, Mapping):
        raise T102RouteBlocked("route must be an object")
    if route.get("schema") != SCHEMA or route.get("route_mode") != "OFFLINE_DIAGNOSTIC_ONLY":
        raise T102RouteBlocked("unexpected route schema or mode")
    if route.get("phase") != PHASE or route.get("phases") != [PHASE]:
        raise T102RouteBlocked("route must contain exactly the outbound pre-contact phase")
    if route.get("joint_path") != "SIMULTANEOUS_T102_JOINT_SPACE_CHORD":
        raise T102RouteBlocked("route is not a simultaneous joint-space chord")
    if route.get("command_interface") != "NOT_IMPLEMENTED":
        raise T102RouteBlocked("diagnostic route must not expose a command interface")
    for field in ("no_contact", "diagnostic_only"):
        if route.get(field) is not True:
            raise T102RouteBlocked(f"route must keep {field}=true")
    for field in ("press", "retract", "return", "automatic_retry", "execution_ready", "motion_authorized", "trajectory_generated", "command_generated", "adapter_consumable"):
        if route.get(field) is not False:
            raise T102RouteBlocked(f"route must keep {field}=false")
    bindings = route.get("frozen_bindings")
    if not isinstance(bindings, Mapping):
        raise T102RouteBlocked("route frozen bindings are missing")
    for key in ("m0_target_sha256", "fn_sha256", "r6_sha256", "provenance_sha256"):
        _sha256(bindings.get(key), f"frozen_bindings.{key}")
    start = _vector(route.get("start_positions_rad"), "start_positions_rad")
    target = _vector(route.get("static_ik_candidate_positions_rad"), "static_ik_candidate_positions_rad")
    points = route.get("setpoints_rad")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not points:
        raise T102RouteBlocked("route has no bounded setpoints")
    fraction = route.get("max_profile_delta_fraction")
    caps = _conservative_caps(tuple(math.radians(value) for value in EXPECTED_DELTA_CAPS_DEG), fraction)
    expected_segment_caps = [float(fraction) * cap for cap in EXPECTED_DELTA_CAPS_DEG]
    if route.get("driver_delta_caps_deg") != list(EXPECTED_DELTA_CAPS_DEG):
        raise T102RouteBlocked("route driver profile caps differ from reviewed caps")
    if route.get("segment_delta_caps_deg") != expected_segment_caps:
        raise T102RouteBlocked("route segment caps differ from conservative profile fraction")
    encoder_step = route.get("encoder_step_rad")
    if not isinstance(encoder_step, (int, float)) or not math.isfinite(float(encoder_step)) or float(encoder_step) <= 0.0:
        raise T102RouteBlocked("route encoder step is invalid")
    tolerance_deg = route.get("driver_tolerance_deg")
    if not isinstance(tolerance_deg, (int, float)) or not math.isfinite(float(tolerance_deg)) or float(tolerance_deg) <= 0.0:
        raise T102RouteBlocked("route driver tolerance is invalid")
    expected_band = 2.0 * float(tolerance_deg)
    if route.get("driver_soft_converge_band_deg") != expected_band:
        raise T102RouteBlocked("route soft-converge band differs from declared tolerance")
    previous = start
    segment_steps = route.get("segment_deltas_encoder_steps")
    if not isinstance(segment_steps, Sequence) or isinstance(segment_steps, (str, bytes)) or len(segment_steps) != len(points):
        raise T102RouteBlocked("route encoder-step segment deltas are missing or malformed")
    for index, raw_point in enumerate(points):
        point = _vector(raw_point, f"setpoints_rad[{index}]")
        raw_steps = segment_steps[index]
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)) or len(raw_steps) != 6:
            raise T102RouteBlocked(f"route segment deltas[{index}] are malformed")
        if any(abs(value - prior) > cap + 1e-12 for value, prior, cap in zip(point, previous, caps)):
            raise T102RouteBlocked("route setpoint exceeds a reviewed delta cap")
        for delta, expected_steps in zip((abs(value - prior) for value, prior in zip(point, previous)), raw_steps):
            try:
                steps = float(expected_steps)
            except (TypeError, ValueError) as exc:
                raise T102RouteBlocked(f"route segment deltas[{index}] are not numeric") from exc
            if not math.isfinite(steps) or abs(delta / float(encoder_step) - steps) > 1e-9:
                raise T102RouteBlocked(f"route segment deltas[{index}] mismatch setpoints")
        previous = point
    first_deltas = segment_steps[0]
    first_max_rad = max(float(step) for step in first_deltas) * float(encoder_step)
    declared_max_rad = route.get("first_segment_max_delta_rad")
    if (
        not isinstance(declared_max_rad, (int, float))
        or not math.isfinite(float(declared_max_rad))
        or abs(float(declared_max_rad) - first_max_rad) > 1e-12
    ):
        raise T102RouteBlocked("route first-segment max delta is inconsistent")
    declared_max_deg = route.get("first_segment_max_delta_deg")
    if (
        not isinstance(declared_max_deg, (int, float))
        or not math.isfinite(float(declared_max_deg))
        or abs(float(declared_max_deg) - math.degrees(first_max_rad)) > 1e-9
    ):
        raise T102RouteBlocked("route first-segment max delta degrees are inconsistent")
    if route.get("first_segment_exceeds_soft_converge_band") is not (
        math.degrees(first_max_rad) > expected_band
    ):
        raise T102RouteBlocked("route first-segment soft-band flag is inconsistent")
    policy = route.get("terminal_residual_policy")
    if not isinstance(policy, Mapping):
        raise T102RouteBlocked("route terminal residual policy is missing")
    if policy.get("driver_tolerance_deg") != tolerance_deg:
        raise T102RouteBlocked("route terminal policy tolerance differs from declared tolerance")
    if policy.get("soft_converge_band_deg") != expected_band:
        raise T102RouteBlocked("route terminal policy band differs from declared tolerance")
    if policy.get("single_stalled_reissue") is not True:
        raise T102RouteBlocked("route terminal policy must keep single_stalled_reissue=true")
    if policy.get("within_band_driver_outcome") != "SOFT_CONVERGE_AFTER_ONE_REISSUE":
        raise T102RouteBlocked("route terminal policy within-band outcome is inconsistent")
    if policy.get("beyond_band_driver_outcome") != "FAULT_LATCHED_AFTER_ONE_REISSUE":
        raise T102RouteBlocked("route terminal policy beyond-band outcome is inconsistent")
    if policy.get("executor_keeps_own_final_joint_gate_rad") != FINAL_JOINT_TOLERANCE_RAD:
        raise T102RouteBlocked("route terminal policy must not claim an executor gate change")
    failure = route.get("observed_field_failure_reference")
    if not isinstance(failure, Mapping):
        raise T102RouteBlocked("route observed field failure reference is missing")
    if failure.get("joint") != OBSERVED_FIELD_FAILURE_JOINT:
        raise T102RouteBlocked("route observed field failure joint is inconsistent")
    failure_residual = failure.get("max_residual_rad")
    if not isinstance(failure_residual, (int, float)) or abs(float(failure_residual) - OBSERVED_FIELD_FAILURE_RESIDUAL_RAD) > 1e-12:
        raise T102RouteBlocked("route observed field failure residual is inconsistent")
    failure_residual_deg = failure.get("max_residual_deg")
    if not isinstance(failure_residual_deg, (int, float)) or abs(float(failure_residual_deg) - math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD)) > 1e-9:
        raise T102RouteBlocked("route observed field failure residual degrees are inconsistent")
    if failure.get("soft_converge_band_deg") != expected_band:
        raise T102RouteBlocked("route observed field failure band differs from declared tolerance")
    if failure.get("residual_exceeded_soft_converge_band") is not (
        math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD) > expected_band
    ):
        raise T102RouteBlocked("route observed field failure soft-band flag is inconsistent")
    if failure.get("driver_outcome") != "FAULT_LATCHED_AFTER_ONE_REISSUE":
        raise T102RouteBlocked("route observed field failure outcome is inconsistent")
    if route.get("physical_convergence_guaranteed") is not False:
        raise T102RouteBlocked("route must not guarantee physical convergence")
    if route.get("segmentation_does_not_fix_physical_undertravel") is not True:
        raise T102RouteBlocked("route must keep segmentation-only claim true")
    if not isinstance(route.get("execution_blocked_reason"), str) or not route.get("execution_blocked_reason").strip():
        raise T102RouteBlocked("route execution-blocked reason is missing")
    if previous != target:
        raise T102RouteBlocked("route endpoint differs from static IK candidate")
    unsigned = dict(route)
    supplied = _sha256(unsigned.pop("route_sha256", None), "route_sha256")
    if canonical_sha256(unsigned) != supplied:
        raise T102RouteBlocked("route SHA-256 mismatch")


__all__ = [
    "DEFAULT_DRIVER_PROFILE", "DEFAULT_MAX_PROFILE_DELTA_FRACTION", "ENCODER_STEP_RAD", "EXPECTED_DELTA_CAPS_DEG",
    "OBSERVED_FIELD_FAILURE_JOINT", "OBSERVED_FIELD_FAILURE_RESIDUAL_RAD", "PHASE", "SCHEMA", "T102RouteBlocked",
    "build_t102_outbound_diagnostic_route", "canonical_sha256", "classify_t102_terminal_residual",
    "load_confirmed_t102_delta_caps_rad", "load_declared_t102_driver_tolerance_deg",
    "validate_t102_outbound_diagnostic_route",
]
