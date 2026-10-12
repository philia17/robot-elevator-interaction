"""Pure builder for the exact, segmented return to Standard Standby.

The return is admitted only from the terminal endpoint of the executor's
consumed Dynamic-IK plan.  It follows that already checked corridor in reverse,
re-samples the actual measured start, and never creates a ROS resource.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from press.t102_dynamic_precontact_core import (
    DynamicPrepareBlocked,
    FrozenModel,
    PreparedPlan,
    _dot,
    _point_segment_distance,
    _segment_distance,
    _sub,
)


SCHEMA = "t102_exact_segmented_standby_route_v1"
CONFIRMATION_TOKEN = "EXECUTE_EXACT_SEGMENTED_STANDBY_ROUTE"
SUCCESS_CODE = "EXACT_SEGMENTED_STANDBY_ROUTE_PREPARED_NO_MOTION"
BOOTSTRAP_STATE = "FAULT_CLEARED_DISARMED"
BOOTSTRAP_STATES = frozenset({BOOTSTRAP_STATE, "CONVERGED_DISARMED"})
BOOTSTRAP_ENVELOPE_RAD = math.radians(3.0)
SOFT_STANDBY_LIMIT_RAD = math.radians(2.0)
SOFT_STANDBY_STATES = frozenset({
    "CONVERGED_DISARMED", "FAULT_CLEARED_DISARMED", "TORQUE_ON_VERIFIED",
})
SOURCE_ROUTE_INITIAL_STATE = "DISARMED:operator"
COMPLETED_PRE_RETURN_START_LIMIT_RAD = math.radians(2.0)


class StandbyRouteBlocked(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(self.code if not detail else f"{self.code}: {detail}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _vector(value: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise StandbyRouteBlocked("STANDBY_ROUTE_INVALID", label) from exc
    if len(result) != 6 or any(not math.isfinite(item) for item in result):
        raise StandbyRouteBlocked("STANDBY_ROUTE_INVALID", label)
    return result


def _vector3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise StandbyRouteBlocked("STANDBY_ROUTE_INVALID", label) from exc
    if len(result) != 3 or any(not math.isfinite(item) for item in result):
        raise StandbyRouteBlocked("STANDBY_ROUTE_INVALID", label)
    return result  # type: ignore[return-value]


def _quantize(value: Sequence[float], step: float) -> tuple[float, ...]:
    return tuple(round(float(item) / step) * step for item in value)


@dataclass(frozen=True)
class StandbyRoutePlan:
    payload: Mapping[str, Any]
    plan_hash: str
    expires_monotonic: float
    consumed: bool = False


def _refine_chord(
    start: Sequence[float], end: Sequence[float], caps: Sequence[float], step: float,
) -> list[tuple[float, ...]]:
    start_q = _vector(start, "segment start")
    end_q = _vector(end, "segment end")
    segment_count = max(
        1,
        max(math.ceil(abs(target - origin) / cap) for origin, target, cap in zip(start_q, end_q, caps)),
    )
    result: list[tuple[float, ...]] = []
    previous = start_q
    for index in range(1, segment_count + 1):
        raw = tuple(
            origin + (target - origin) * index / segment_count
            for origin, target in zip(start_q, end_q)
        )
        waypoint = end_q if index == segment_count else _quantize(raw, step)
        if waypoint == previous:
            continue
        if any(abs(a - b) > cap + 1e-12 for a, b, cap in zip(waypoint, previous, caps)):
            raise StandbyRouteBlocked("STANDBY_ROUTE_CAP_REJECTED")
        result.append(waypoint)
        previous = waypoint
    return result


def _validate_proxy_path(
    model: FrozenModel,
    start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    *,
    wall_point: Sequence[float],
    outward_normal: Sequence[float],
) -> int:
    """Re-run the boot-frozen outbound proxy checks on every return segment."""
    policy = model.policy
    cursor = _vector(start, "path start")
    wall = _vector3(wall_point, "wall point")
    normal = _vector3(outward_normal, "normal")
    sample_count = max(3, int(policy["segment_sample_count"]))
    checked = 0
    for target_value in waypoints:
        target = _vector(target_value, "path target")
        for sample_i in range(sample_count):
            ratio = sample_i / (sample_count - 1)
            q = tuple(a + ratio * (b - a) for a, b in zip(cursor, target))
            try:
                model._validate_limits(q)
            except DynamicPrepareBlocked as exc:
                raise StandbyRouteBlocked(exc.code, str(exc)) from exc
            state = model.fk(q)
            capsules: dict[str, tuple] = {}
            for item in policy["capsules"]:
                a = state.points[int(item["joint_a"])]
                b = state.points[int(item["joint_b"])]
                radius = float(item["radius_m"])
                capsules[str(item["name"])] = (a, b, radius)
                if min(_dot(normal, _sub(a, wall)), _dot(normal, _sub(b, wall))) - radius < float(policy["wall_margin_m"]):
                    raise StandbyRouteBlocked("WALL_HALFSPACE_REJECTED", str(item["name"]))
                for proxy_key, configured_key in (
                    ("base_proxy", "base_check_capsules"),
                    ("camera_bracket_proxy", "bracket_check_capsules"),
                ):
                    if item["name"] not in policy[configured_key]:
                        continue
                    proxy = policy[proxy_key]
                    if str(proxy.get("primitive") or "sphere") == "sphere":
                        distance = _point_segment_distance(proxy["center_m"], a, b)
                    else:
                        distance = _segment_distance(a, b, proxy["endpoint_a_m"], proxy["endpoint_b_m"])
                    if distance < radius + float(proxy["radius_m"]):
                        raise StandbyRouteBlocked("BASE_BRACKET_PROXY_REJECTED", proxy_key)
            for left, right in policy["self_pairs"]:
                a0, a1, ar = capsules[left]
                b0, b1, br = capsules[right]
                if _segment_distance(a0, a1, b0, b1) < ar + br:
                    raise StandbyRouteBlocked("SELF_PROXY_REJECTED", f"{left}/{right}")
            checked += 1
        cursor = target
    return checked


def prepare_standby_route(
    model: FrozenModel,
    source_plan: PreparedPlan | None,
    current_positions_rad: Sequence[float],
    *,
    ui_request_id: str,
    now_monotonic: float,
    bootstrap_state: str = "",
) -> StandbyRoutePlan:
    current = _vector(current_positions_rad, "current positions")
    if not str(ui_request_id).strip() or not math.isfinite(float(now_monotonic)):
        raise StandbyRouteBlocked("STANDBY_ROUTE_INVALID", "request identity/time")
    target = _vector(model.policy["standard_standby_rad"], "Standard Standby")
    step = float(model.policy["encoder_step_rad"])
    caps = _vector(model.policy["per_command_caps_rad"], "delta caps")
    model._validate_limits(current)
    model._validate_limits(target)
    tolerance = min(
        math.radians(1.0),
        float(model.policy["move_waypoint_tolerance_rad"]),
    )
    endpoint_error = max(abs(a - b) for a, b in zip(current, target))
    if endpoint_error <= tolerance and bootstrap_state not in {
        BOOTSTRAP_STATE, "TORQUE_ON_VERIFIED", SOURCE_ROUTE_INITIAL_STATE,
    }:
        payload = {
            "schema": SCHEMA,
            "route_kind": "STANDBY",
            "ui_request_id": str(ui_request_id),
            "executor_boot_epoch": model.boot_epoch,
            "model_hashes": model.hashes,
            "start_positions_rad": current,
            "standard_standby_rad": target,
            "waypoints_rad": [],
            "already_at_standby": True,
            "automatic_retry": False,
            "automatic_compensation": False,
            "overshoot": False,
            "clear_fault": False,
            "created_monotonic": float(now_monotonic),
        }
        return StandbyRoutePlan(payload, canonical_hash(payload), float(now_monotonic) + float(model.policy["ticket_ttl_s"]))
    if (
        bootstrap_state in SOFT_STANDBY_STATES
        and endpoint_error <= SOFT_STANDBY_LIMIT_RAD + 1e-12
    ):
        # This proof depends only on fresh measured Standby, not the retained
        # outbound route.  A previous PRE must not force reverse-route checks
        # after a manual/configured return has already reached Standby.
        soft_identity = {
            "mode": "SOURCELESS_SOFT_STANDBY_NO_MOTION",
            "admission_state": bootstrap_state,
            "measured_q_rad": current,
            "max_joint_error_rad": endpoint_error,
            "limit_rad": SOFT_STANDBY_LIMIT_RAD,
        }
        payload = {
            "schema": SCHEMA,
            "route_kind": "STANDBY",
            "ui_request_id": str(ui_request_id),
            "executor_boot_epoch": model.boot_epoch,
            "model_hashes": model.hashes,
            "soft_standby_identity": soft_identity,
            "start_positions_rad": current,
            "standard_standby_rad": target,
            "waypoints_rad": [],
            "already_at_standby": True,
            "automatic_retry": False,
            "automatic_compensation": False,
            "overshoot": False,
            "clear_fault": False,
            "created_monotonic": float(now_monotonic),
        }
        return StandbyRoutePlan(
            payload, canonical_hash(payload),
            float(now_monotonic) + float(model.policy["ticket_ttl_s"]),
        )
    if source_plan is None and bootstrap_state:
        if bootstrap_state not in BOOTSTRAP_STATES:
            raise StandbyRouteBlocked(
                "STANDBY_BOOTSTRAP_STATE_REJECTED", str(bootstrap_state)
            )
        if endpoint_error > BOOTSTRAP_ENVELOPE_RAD + 1e-12:
            raise StandbyRouteBlocked(
                "STANDBY_BOOTSTRAP_ENVELOPE_REJECTED",
                f"{endpoint_error:.12g} > {BOOTSTRAP_ENVELOPE_RAD:.12g}",
            )
        quantized_target = _quantize(target, step)
        if max(abs(a - b) for a, b in zip(quantized_target, target)) > 1e-9:
            raise StandbyRouteBlocked(
                "STANDBY_ROUTE_INVALID", "Standard Standby is not encoder-quantized"
            )
        route = [target]
        if any(abs(a - b) > cap + 1e-12 for a, b, cap in zip(target, current, caps)):
            raise StandbyRouteBlocked("STANDBY_ROUTE_CAP_REJECTED")
        bootstrap_identity = {
            "mode": "LOCAL_STANDARD_STANDBY_CORRECTION",
            "admission_state": bootstrap_state,
            "per_joint_envelope_rad": BOOTSTRAP_ENVELOPE_RAD,
            "start_max_joint_error_rad": endpoint_error,
        }
        payload = {
            "schema": SCHEMA,
            "route_kind": "STANDBY",
            "ui_request_id": str(ui_request_id),
            "executor_boot_epoch": model.boot_epoch,
            "model_hashes": model.hashes,
            "bootstrap_identity": bootstrap_identity,
            "start_positions_rad": current,
            "standard_standby_rad": target,
            "waypoints_rad": route,
            "per_command_caps_rad": caps,
            "encoder_step_rad": step,
            "path_validation": {
                "mode": "LOCAL_STANDARD_STANDBY_BOOTSTRAP_ENVELOPE",
                "limits": "PASS",
                "per_joint_envelope": "PASS",
            },
            "already_at_standby": False,
            "automatic_retry": False,
            "automatic_compensation": False,
            "overshoot": False,
            "clear_fault": False,
            "created_monotonic": float(now_monotonic),
        }
        return StandbyRoutePlan(
            payload,
            canonical_hash(payload),
            float(now_monotonic) + float(model.policy["ticket_ttl_s"]),
        )
    if source_plan is None or source_plan.plan.get("schema") != "t102_dynamic_precontact_plan_v1":
        raise StandbyRouteBlocked("STANDBY_ROUTE_SOURCE_REQUIRED", "consumed Dynamic-IK route absent")
    if bootstrap_state and bootstrap_state not in (
        SOFT_STANDBY_STATES | {SOURCE_ROUTE_INITIAL_STATE}
    ):
        raise StandbyRouteBlocked(
            "STANDBY_ROUTE_STATE_REJECTED", str(bootstrap_state)
        )
    if not source_plan.consumed or canonical_hash(source_plan.plan) != source_plan.plan_hash:
        raise StandbyRouteBlocked("STANDBY_ROUTE_SOURCE_INVALID", "source must be exact consumed plan")
    if canonical_hash(source_plan.plan.get("model_hashes")) != canonical_hash(model.hashes):
        raise StandbyRouteBlocked("STANDBY_ROUTE_SOURCE_INVALID", "model identity drift")
    outbound = tuple(_vector(item, "source waypoint") for item in source_plan.plan.get("waypoints_rad", ()))
    if len(outbound) < 2:
        raise StandbyRouteBlocked("STANDBY_ROUTE_SOURCE_INVALID", "source corridor too short")
    if (
        max(abs(a - b) for a, b in zip(current, outbound[-1]))
        > COMPLETED_PRE_RETURN_START_LIMIT_RAD + 1e-12
    ):
        raise StandbyRouteBlocked("STANDBY_ROUTE_ENDPOINT_DRIFT", "fresh q is not completed PRE")
    anchors = list(reversed(outbound[:-1])) + [target]
    route: list[tuple[float, ...]] = []
    cursor = current
    for anchor in anchors:
        additions = _refine_chord(cursor, anchor, caps, step)
        route.extend(additions)
        if additions:
            cursor = additions[-1]
    if len(route) < 2 or route[-1] != target:
        raise StandbyRouteBlocked("STANDBY_ROUTE_INVALID", "segmented exact endpoint")
    session = source_plan.plan.get("session")
    proof = source_plan.plan.get("normal_proof")
    if not isinstance(session, Mapping) or not isinstance(proof, Mapping):
        raise StandbyRouteBlocked("STANDBY_ROUTE_SOURCE_INVALID", "path geometry missing")
    checked = _validate_proxy_path(
        model,
        current,
        route,
        wall_point=session["wall_surface_point_base_m"],
        outward_normal=proof["n_out"],
    )
    payload = {
        "schema": SCHEMA,
        "route_kind": "STANDBY",
        "ui_request_id": str(ui_request_id),
        "executor_boot_epoch": model.boot_epoch,
        "model_hashes": model.hashes,
        "source_precontact_plan_hash": source_plan.plan_hash,
        "session_id": str(source_plan.plan["session"]["session_id"]),
        "start_positions_rad": current,
        "standard_standby_rad": target,
        "waypoints_rad": route,
        "per_command_caps_rad": caps,
        "encoder_step_rad": step,
        "path_validation": {
            "mode": "REVERSE_OF_FROZEN_OUTBOUND_WITH_ACTUAL_START_RESAMPLED",
            "sample_count": checked,
            "limits": "PASS",
            "wall": "PASS",
            "self": "PASS",
            "base_bracket": "PASS",
        },
        "already_at_standby": False,
        "automatic_retry": False,
        "automatic_compensation": False,
        "overshoot": False,
        "clear_fault": False,
        "created_monotonic": float(now_monotonic),
    }
    if bootstrap_state in {
        BOOTSTRAP_STATE, "TORQUE_ON_VERIFIED", SOURCE_ROUTE_INITIAL_STATE,
    }:
        payload["initial_disarmed_state"] = bootstrap_state
    return StandbyRoutePlan(payload, canonical_hash(payload), float(now_monotonic) + float(model.policy["ticket_ttl_s"]))


def consume_standby_route(
    ticket: StandbyRoutePlan,
    *,
    model: FrozenModel,
    current_positions_rad: Sequence[float],
    now_monotonic: float,
) -> StandbyRoutePlan:
    if ticket.consumed or float(now_monotonic) >= ticket.expires_monotonic:
        raise StandbyRouteBlocked("STANDBY_ROUTE_EXPIRED_OR_REPLAYED")
    if canonical_hash(ticket.payload) != ticket.plan_hash:
        raise StandbyRouteBlocked("STANDBY_ROUTE_IDENTITY_DRIFT")
    if ticket.payload.get("executor_boot_epoch") != model.boot_epoch:
        raise StandbyRouteBlocked("STANDBY_ROUTE_IDENTITY_DRIFT", "executor")
    current = _vector(current_positions_rad, "current positions")
    frozen = _vector(ticket.payload["start_positions_rad"], "frozen start")
    if max(abs(a - b) for a, b in zip(current, frozen)) > float(model.policy["q0_drift_limit_rad"]):
        raise StandbyRouteBlocked("STANDBY_ROUTE_START_DRIFT")
    return StandbyRoutePlan(ticket.payload, ticket.plan_hash, ticket.expires_monotonic, True)


__all__ = [
    "BOOTSTRAP_ENVELOPE_RAD", "BOOTSTRAP_STATE", "BOOTSTRAP_STATES",
    "CONFIRMATION_TOKEN", "SCHEMA", "SOFT_STANDBY_LIMIT_RAD",
    "SOFT_STANDBY_STATES", "SOURCE_ROUTE_INITIAL_STATE",
    "SUCCESS_CODE", "StandbyRouteBlocked",
    "StandbyRoutePlan", "canonical_hash", "consume_standby_route", "prepare_standby_route",
]
