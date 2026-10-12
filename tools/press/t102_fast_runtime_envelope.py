#!/usr/bin/env python3
"""Standard-library-only command envelope for the fixed T102 v2 route.

This module deliberately knows nothing about ROS, kinematics, URDF, meshes,
obstacles, or clearance receipts.  It validates one reviewed package and the
cheap measured-feedback facts needed to hand off between its four outbound
coarse targets.  Passing this envelope is not motion or contact authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_SCHEMA = "t102_fast_precontact_route_package_v2"
SCENARIO = "standard_standby_to_button_1_70mm_coarse_precontact"
JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)
ENCODER_STEP_RAD = 2.0 * math.pi / 4096.0
STANDARD_STANDBY_RAD = (
    0.06289321230331128,
    -1.188835110611372,
    2.3316507975861747,
    0.46479617872934925,
    -1.5109710760673565,
    0.0015339807878856412,
)
HARD_JOINT_LIMITS_RAD = (
    (-math.pi, math.pi),
    (-math.pi / 2.0, math.pi / 2.0),
    (0.0, 2.95),
    (-math.pi / 2.0, math.pi / 2.0),
    (-math.pi, math.pi),
    (0.0, 1.5),
)
COMMAND_DELTA_CAPS_RAD = tuple(
    math.radians(value) for value in (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
)
MINIMUM_USEFUL_SEGMENT_MOTION_RAD = math.radians(3.0)
EXPECTED_PACKAGE_SHA256 = "b8d691535c97efd82113f2c51d721a6a617524e4c5f0951075e421079d47262c"


class EnvelopeBlocked(ValueError):
    """The package or one command-level runtime fact failed closed."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EnvelopeBlocked("value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EnvelopeBlocked(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise EnvelopeBlocked(f"non-finite JSON constant: {value}")


def load_json(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except EnvelopeBlocked:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EnvelopeBlocked(f"package JSON unreadable: {path}: {exc}") from exc


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvelopeBlocked(f"{field} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], field: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EnvelopeBlocked(
            f"{field} keys mismatch: missing={missing}, extra={extra}"
        )


def vector(value: Any, field: str, *, length: int = 6) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise EnvelopeBlocked(f"{field} must contain exactly {length} finite numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EnvelopeBlocked(f"{field}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise EnvelopeBlocked(f"{field}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvelopeBlocked(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EnvelopeBlocked(f"{field} must be finite")
    return number


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _within(value: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(value - expected) <= tolerance


def _validate_hard_limits(q: Sequence[float], field: str) -> None:
    for index, (item, (lower, upper)) in enumerate(zip(q, HARD_JOINT_LIMITS_RAD)):
        if not lower <= item <= upper:
            raise EnvelopeBlocked(
                f"{field}[{index}] outside existing hard joint limits"
            )


def _validate_transition(
    before: Sequence[float],
    after: Sequence[float],
    *,
    field: str,
    require_useful_motion: bool,
) -> tuple[float, ...]:
    _validate_hard_limits(after, field)
    deltas = tuple(abs(b - a) for a, b in zip(before, after))
    if any(
        delta > cap + 1e-12
        for delta, cap in zip(deltas, COMMAND_DELTA_CAPS_RAD)
    ):
        raise EnvelopeBlocked(f"{field} exceeds existing per-command delta caps")
    if require_useful_motion and max(deltas) < MINIMUM_USEFUL_SEGMENT_MOTION_RAD - 1e-12:
        raise EnvelopeBlocked(f"{field} is a tiny-step/deadband-risk segment")
    return deltas


def validate_package_document(
    raw: Any,
    *,
    expected_package_sha256: str | None = EXPECTED_PACKAGE_SHA256,
) -> dict[str, Any]:
    """Validate and return one v2 package without any non-stdlib dependency."""

    package = dict(_require_mapping(raw, "package"))
    _require_exact_keys(
        package,
        {
            "schema",
            "scenario",
            "joint_names",
            "encoder_step_rad",
            "start",
            "outbound_targets_rad",
            "endpoint",
            "command_limits",
            "handoff_policy",
            "command_contract",
            "authority",
            "package_sha256",
        },
        "package",
    )
    if package["schema"] != PACKAGE_SCHEMA:
        raise EnvelopeBlocked("package schema mismatch")
    if package["scenario"] != SCENARIO:
        raise EnvelopeBlocked("scenario mismatch")
    if package["joint_names"] != list(JOINT_NAMES):
        raise EnvelopeBlocked("joint-name/order mismatch")
    if not _within(_number(package["encoder_step_rad"], "encoder_step_rad"), ENCODER_STEP_RAD):
        raise EnvelopeBlocked("encoder-step constant mismatch")

    start = _require_mapping(package["start"], "start")
    _require_exact_keys(start, {"name", "positions_rad", "tolerance_rad", "tolerance_rule"}, "start")
    if start["name"] != "STANDARD_STANDBY":
        raise EnvelopeBlocked("start posture name mismatch")
    q_start = vector(start["positions_rad"], "start.positions_rad")
    if any(abs(a - b) > 1e-12 for a, b in zip(q_start, STANDARD_STANDBY_RAD)):
        raise EnvelopeBlocked("stored start is not the reviewed quantized Standard Standby")
    tolerance = _number(start["tolerance_rad"], "start.tolerance_rad")
    if not _within(tolerance, ENCODER_STEP_RAD):
        raise EnvelopeBlocked("start tolerance must be exactly one encoder step")
    if start["tolerance_rule"] != "MAX_ABS_ONE_ENCODER_STEP":
        raise EnvelopeBlocked("start tolerance rule mismatch")
    _validate_hard_limits(q_start, "start.positions_rad")
    if any(
        abs(value - round(value / ENCODER_STEP_RAD) * ENCODER_STEP_RAD) > 1e-12
        for value in q_start
    ):
        raise EnvelopeBlocked("stored start is not encoder-grid quantized")

    limits = _require_mapping(package["command_limits"], "command_limits")
    _require_exact_keys(
        limits,
        {
            "hard_joint_limits_rad",
            "per_command_delta_caps_rad",
            "minimum_useful_segment_motion_rad",
            "maximum_route_reverse_rad",
        },
        "command_limits",
    )
    hard_limits = limits["hard_joint_limits_rad"]
    if hard_limits != [list(item) for item in HARD_JOINT_LIMITS_RAD]:
        raise EnvelopeBlocked("hard joint limits differ from the existing driver limits")
    caps = vector(limits["per_command_delta_caps_rad"], "per_command_delta_caps_rad")
    if any(abs(a - b) > 1e-12 for a, b in zip(caps, COMMAND_DELTA_CAPS_RAD)):
        raise EnvelopeBlocked("per-command delta caps mismatch")
    minimum_segment = _number(
        limits["minimum_useful_segment_motion_rad"],
        "minimum_useful_segment_motion_rad",
    )
    if not _within(minimum_segment, MINIMUM_USEFUL_SEGMENT_MOTION_RAD):
        raise EnvelopeBlocked("minimum useful segment constant mismatch")
    maximum_route_reverse = _number(
        limits["maximum_route_reverse_rad"], "maximum_route_reverse_rad"
    )
    if not _within(maximum_route_reverse, 0.0087):
        raise EnvelopeBlocked("maximum route reverse constant mismatch")

    targets_raw = package["outbound_targets_rad"]
    if not isinstance(targets_raw, list) or len(targets_raw) != 4:
        raise EnvelopeBlocked("route must contain exactly four outbound targets")
    targets = [vector(item, f"outbound_targets_rad[{index}]") for index, item in enumerate(targets_raw)]
    if len(set(targets)) != 4:
        raise EnvelopeBlocked("outbound targets must be distinct")
    endpoint_q = targets[-1]
    total_direction = tuple(goal - start_value for start_value, goal in zip(q_start, endpoint_q))
    previous = q_start
    previous_distance = distance(previous, endpoint_q)
    for index, target in enumerate(targets):
        if any(
            abs(value - round(value / ENCODER_STEP_RAD) * ENCODER_STEP_RAD) > 1e-12
            for value in target
        ):
            raise EnvelopeBlocked(f"outbound target {index + 1} is not encoder-grid quantized")
        _validate_transition(
            previous,
            target,
            field=f"outbound_targets_rad[{index}]",
            require_useful_motion=True,
        )
        for joint, (before, after, overall) in enumerate(
            zip(previous, target, total_direction)
        ):
            if abs(overall) <= ENCODER_STEP_RAD:
                if abs(after - q_start[joint]) > ENCODER_STEP_RAD + 1e-12:
                    raise EnvelopeBlocked(
                        f"outbound target {index + 1} moves stationary joint {joint}"
                    )
            elif (after - before) * (1.0 if overall > 0.0 else -1.0) < -maximum_route_reverse - 1e-12:
                raise EnvelopeBlocked(
                    f"outbound target {index + 1} materially reverses joint {joint}"
                )
        current_distance = distance(target, endpoint_q)
        if index < len(targets) - 1 and not (
            current_distance < previous_distance - ENCODER_STEP_RAD + 1e-12
        ):
            raise EnvelopeBlocked(
                f"outbound target {index + 1} does not make endpoint progress"
            )
        previous = target
        previous_distance = current_distance

    endpoint = _require_mapping(package["endpoint"], "endpoint")
    _require_exact_keys(
        endpoint,
        {
            "name",
            "button",
            "intended_no_contact_distance_m",
            "joint_positions_rad",
            "target_fingertip_base_m",
            "terminal_max_joint_error_rad",
        },
        "endpoint",
    )
    if endpoint["name"] != "BUTTON_1_COARSE_PRECONTACT_70MM":
        raise EnvelopeBlocked("endpoint name mismatch")
    if type(endpoint["button"]) is not int or endpoint["button"] != 1:
        raise EnvelopeBlocked("endpoint button metadata mismatch")
    if not _within(
        _number(endpoint["intended_no_contact_distance_m"], "endpoint distance"),
        0.070,
    ):
        raise EnvelopeBlocked("endpoint no-contact metadata mismatch")
    if vector(endpoint["joint_positions_rad"], "endpoint.joint_positions_rad") != endpoint_q:
        raise EnvelopeBlocked("endpoint joint metadata does not match final target")
    vector(endpoint["target_fingertip_base_m"], "endpoint.target_fingertip_base_m", length=3)
    if not _within(
        _number(
            endpoint["terminal_max_joint_error_rad"],
            "endpoint.terminal_max_joint_error_rad",
        ),
        0.02,
    ):
        raise EnvelopeBlocked("terminal joint-error constant mismatch")

    policy = _require_mapping(package["handoff_policy"], "handoff_policy")
    _require_exact_keys(
        policy,
        {
            "minimum_measured_progress_rad",
            "maximum_wrong_direction_rad",
            "minimum_endpoint_distance_reduction_rad",
            "maximum_feedback_age_s",
            "minimum_replace_interval_s",
        },
        "handoff_policy",
    )
    expected_policy = {
        "minimum_measured_progress_rad": 0.035,
        "maximum_wrong_direction_rad": 0.0087,
        "minimum_endpoint_distance_reduction_rad": ENCODER_STEP_RAD,
        "maximum_feedback_age_s": 1.0,
        "minimum_replace_interval_s": 0.5,
    }
    for field, expected in expected_policy.items():
        actual = _number(policy[field], f"handoff_policy.{field}")
        if not _within(actual, expected):
            raise EnvelopeBlocked(f"handoff policy constant mismatch: {field}")

    contract = _require_mapping(package["command_contract"], "command_contract")
    _require_exact_keys(
        contract,
        {
            "command_interface",
            "direction",
            "terminal_stage",
            "press_included",
            "contact_included",
            "return_included",
        },
        "command_contract",
    )
    if contract["command_interface"] != "T102_MULTI_JOINT_ABSOLUTE":
        raise EnvelopeBlocked("command interface mismatch")
    if contract["direction"] != "OUTBOUND_ONLY":
        raise EnvelopeBlocked("route is not outbound-only")
    if contract["terminal_stage"] != "COARSE_PRECONTACT_STOP":
        raise EnvelopeBlocked("route does not stop at coarse precontact")
    for field in ("press_included", "contact_included", "return_included"):
        if contract[field] is not False:
            raise EnvelopeBlocked(f"{field} must remain false")

    authority = _require_mapping(package["authority"], "authority")
    _require_exact_keys(
        authority,
        {"execution_ready", "motion_authorized", "contact_authorized"},
        "authority",
    )
    for field in ("execution_ready", "motion_authorized", "contact_authorized"):
        if authority[field] is not False:
            raise EnvelopeBlocked(f"authority.{field} must remain false")

    supplied_hash = package["package_sha256"]
    if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
        raise EnvelopeBlocked("package_sha256 is malformed")
    unsigned = dict(package)
    unsigned.pop("package_sha256")
    calculated = canonical_sha256(unsigned)
    if supplied_hash != calculated:
        raise EnvelopeBlocked("package_sha256 self-hash mismatch")
    if expected_package_sha256 is not None and supplied_hash != expected_package_sha256:
        raise EnvelopeBlocked("package hash differs from the reviewed v2 package")
    return package


def load_and_validate_package(
    path: str | Path,
    *,
    expected_package_sha256: str | None = EXPECTED_PACKAGE_SHA256,
) -> dict[str, Any]:
    return validate_package_document(
        load_json(path), expected_package_sha256=expected_package_sha256
    )


def validate_start_posture(
    package: Mapping[str, Any], current_start_rad: Sequence[float]
) -> dict[str, Any]:
    current = vector(current_start_rad, "current_start_rad")
    expected = vector(package["start"]["positions_rad"], "stored start")
    tolerance = _number(package["start"]["tolerance_rad"], "start tolerance")
    errors = tuple(abs(a - b) for a, b in zip(current, expected))
    if max(errors) > tolerance + 1e-12:
        raise EnvelopeBlocked(
            "current start is outside the documented one-encoder-step tolerance"
        )
    _validate_hard_limits(current, "current_start_rad")
    return {
        "accepted": True,
        "maximum_start_error_rad": max(errors),
        "tolerance_rad": tolerance,
    }


def validate_command_transition(
    package: Mapping[str, Any],
    current_rad: Sequence[float],
    target_rad: Sequence[float],
    *,
    endpoint_rad: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Reusable command-level boundary for a later target-relative stage.

    This performs only finite-vector, hard-limit, delta-cap, and optional
    endpoint-progress checks.  A future Gemini/visual-servo layer must provide
    its own fresh target binding and authority; this function adds neither.
    """

    current = vector(current_rad, "current_rad")
    target = vector(target_rad, "target_rad")
    _validate_hard_limits(current, "current_rad")
    deltas = _validate_transition(
        current, target, field="target_rad", require_useful_motion=False
    )
    result = {
        "accepted": True,
        "deltas_rad": list(deltas),
        "maximum_delta_rad": max(deltas),
        "endpoint_distance_reduction_rad": None,
    }
    if endpoint_rad is not None:
        endpoint = vector(endpoint_rad, "endpoint_rad")
        reduction = distance(current, endpoint) - distance(target, endpoint)
        if reduction <= 0.0:
            raise EnvelopeBlocked("target does not reduce endpoint distance")
        result["endpoint_distance_reduction_rad"] = reduction
    return result


def assess_measured_handoff(
    package: Mapping[str, Any],
    transition_index: int,
    segment_start_rad: Sequence[float],
    actual_rad: Sequence[float],
    *,
    feedback_age_s: float,
) -> dict[str, Any]:
    """Apply the small v2 asynchronous measured-feedback handoff rule."""

    if type(transition_index) is not int or transition_index not in (0, 1, 2):
        raise EnvelopeBlocked("transition_index must be 0, 1, or 2")
    start = vector(segment_start_rad, "segment_start_rad")
    targets = package["outbound_targets_rad"]
    current_target = vector(
        targets[transition_index], "sealed current_target_rad"
    )
    next_target = vector(
        targets[transition_index + 1], "sealed next_target_rad"
    )
    actual = vector(actual_rad, "actual_rad")
    endpoint = vector(package["endpoint"]["joint_positions_rad"], "endpoint")
    policy = package["handoff_policy"]
    age = _number(feedback_age_s, "feedback_age_s")
    if age < 0.0 or age > float(policy["maximum_feedback_age_s"]) + 1e-12:
        raise EnvelopeBlocked("joint feedback is stale")

    signed_progress: list[float] = []
    for before, goal, measured in zip(start, current_target, actual):
        direction = goal - before
        if abs(direction) <= 0.5 * ENCODER_STEP_RAD:
            signed_progress.append(0.0)
        else:
            signed_progress.append(
                (measured - before) * (1.0 if direction > 0.0 else -1.0)
            )
    maximum_useful = max(signed_progress)
    maximum_wrong = max((-value for value in signed_progress), default=0.0)
    endpoint_reduction = distance(start, endpoint) - distance(actual, endpoint)
    _validate_hard_limits(actual, "actual_rad")
    next_deltas = tuple(abs(goal - measured) for measured, goal in zip(actual, next_target))
    next_caps_ok = all(
        delta <= cap + 1e-12
        for delta, cap in zip(next_deltas, COMMAND_DELTA_CAPS_RAD)
    )
    next_endpoint_reduction = distance(actual, endpoint) - distance(next_target, endpoint)
    next_signed_toward_endpoint = []
    for measured, target, goal in zip(actual, next_target, endpoint):
        remaining = goal - measured
        if abs(remaining) <= 0.5 * ENCODER_STEP_RAD:
            next_signed_toward_endpoint.append(0.0)
        else:
            next_signed_toward_endpoint.append(
                (target - measured) * (1.0 if remaining > 0.0 else -1.0)
            )
    next_maximum_wrong = max(
        (-value for value in next_signed_toward_endpoint), default=0.0
    )
    accepted = bool(
        maximum_useful
        >= float(policy["minimum_measured_progress_rad"]) - 1e-12
        and maximum_wrong
        <= float(policy["maximum_wrong_direction_rad"]) + 1e-12
        and endpoint_reduction
        >= float(policy["minimum_endpoint_distance_reduction_rad"]) - 1e-12
        and next_maximum_wrong
        <= float(policy["maximum_wrong_direction_rad"]) + 1e-12
        and next_caps_ok
        and next_endpoint_reduction > 0.0
    )
    result = {
        "accepted": accepted,
        "source": "measured_large_step_progress_v2",
        "positions_rad": list(actual),
        "feedback_age_s": age,
        "signed_progress_rad": signed_progress,
        "maximum_useful_motion_rad": maximum_useful,
        "maximum_wrong_direction_rad": maximum_wrong,
        "endpoint_distance_reduction_rad": endpoint_reduction,
        "next_command": {
            "deltas_rad": list(next_deltas),
            "within_delta_caps": next_caps_ok,
            "endpoint_distance_reduction_rad": next_endpoint_reduction,
        },
        "next_command_signed_toward_endpoint_rad": next_signed_toward_endpoint,
        "next_command_maximum_wrong_direction_rad": next_maximum_wrong,
    }
    return result
