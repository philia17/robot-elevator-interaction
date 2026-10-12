"""Pure Stage-B positional contact-approach ``OUT + BACK`` contract.

This module consumes one fresh current-panel PRESS receipt and freezes a
sampled, symmetric approach plan.  It deliberately
contains no ROS, serial, subprocess, or robot-command imports.  The default
execution policy is disabled; callers must opt into the build/runtime gates in
an independently reviewed integration.

The contract is intentionally narrow:

* one fresh PRE ``q`` and one frozen panel geometry;
* one measured current-panel gap plus exactly 4 mm positional overtravel from
  fresh actual FK along ``-normal_out_base``, with total travel capped at
  100 mm while retaining the accepted panel geometry identity;
* one immediate return to the byte-for-byte frozen start joint vector;
* positional contact authority only: no force control, activation claim,
  retry, compensation, dwell, or blind return after a failure.

``prepare_stage_b_plan`` is zero-resource.  ``StageBRehearsalCore.execute`` is
an offline lifecycle model and is kept separate from the ROS node so the
tests can exercise all admission and teardown paths without hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from itertools import combinations
import json
import math
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "t102_stage_b_contact_approach_out_back_plan_v1"
RECEIPT_SCHEMA = "current_panel_press_receipt_v1"
WRIST_CORRECTED_RECEIPT_SCHEMA = "current_panel_press_receipt_v2"
MAX_WRIST_TANGENT_CORRECTION_M = 0.035
PRESS_OVERTRAVEL_M = 0.004
APPROACH_DISTANCE_M = 0.070 + PRESS_OVERTRAVEL_M
APPROACH_GEOMETRY_TOLERANCE_M = 1e-6
FORWARD_DISTANCE_M = APPROACH_DISTANCE_M
MIN_CURRENT_PANEL_GAP_M = 0.001
MAX_CURRENT_PANEL_GAP_M = 0.100
REQUIRED_CONFIRMATION_TOKEN = "EXECUTE_STAGE_B_CONTACT_APPROACH_OUT_BACK"
LEASE_PURPOSE = "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE"
LOCKED_JOINT_INDICES = (4, 5)
POSITIONAL_CONTACT_CAPSULES = ("tool",)
SAFE_IDLE_EXECUTION_STATES = frozenset({"CONVERGED_DISARMED", "DISARMED:operator"})
DEFAULT_Q_DRIFT_TOLERANCE_RAD = 0.003
DEFAULT_FEEDBACK_MAX_AGE_S = 0.25
DEFAULT_POSITION_RESIDUAL_M = 0.0025
DEFAULT_AXIS_RESIDUAL_DEG = 8.0
DEFAULT_MAX_JOINT_JUMP_RAD = 0.45
DEFAULT_ENCODER_STEP_RAD = 2.0 * math.pi / 4096.0
DEFAULT_PLAN_TTL_S = 10.0
DEFAULT_WAYPOINT_TOLERANCE_RAD = math.radians(0.5)
RUNTIME_MIN_FORWARD_WAYPOINTS = 2
RUNTIME_MAX_FORWARD_WAYPOINTS = 4
RUNTIME_GEOMETRY_SAMPLE_COUNT = 101
# Frozen copy of the SafeDriver T102/T101 absolute-command limits.  The
# accepted-target audit below must reproduce the driver's clamp exactly.
SAFE_DRIVER_COMMAND_LIMITS_RAD = (
    (-math.pi, math.pi),
    (-math.pi / 2.0, math.pi / 2.0),
    (0.0, 2.95),
    (-math.pi / 2.0, math.pi / 2.0),
    (-math.pi, math.pi),
    (0.0, 1.5),
)


class StageBBlocked(RuntimeError):
    """A validation or lifecycle failure which must fail closed."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(self.code if not detail else f"{self.code}: {detail}")


class StageBState(str, Enum):
    IDLE = "IDLE"
    PREPARED_NO_MOTION = "PREPARED_NO_MOTION"
    EXECUTING_FORWARD = "EXECUTING_FORWARD"
    EXECUTING_RETURN = "EXECUTING_RETURN"
    COMPLETE_CONTACT_APPROACH_RETURNED_PRE = "COMPLETE_CONTACT_APPROACH_RETURNED_PRE"
    ABORTED_SAFE_FAILURE = "ABORTED_SAFE_FAILURE"


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StageBBlocked("CANONICAL_JSON_INVALID") from exc


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise StageBBlocked("IDENTITY_INVALID", field_name)
    return value


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise StageBBlocked("NONFINITE_INPUT", field_name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StageBBlocked("NONFINITE_INPUT", field_name) from exc
    if not math.isfinite(result):
        raise StageBBlocked("NONFINITE_INPUT", field_name)
    return result


def _vector(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise StageBBlocked("INPUT_INVALID", field_name)
    return tuple(_finite(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, field_name: str) -> tuple[float, float, float]:
    vector = _vector(value, 3, field_name)
    magnitude = math.sqrt(sum(item * item for item in vector))
    if magnitude <= 1e-12 or abs(magnitude - 1.0) > 0.02:
        raise StageBBlocked("NORMAL_DIRECTION_INVALID", field_name)
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _scale(value: Sequence[float], scalar: float) -> tuple[float, ...]:
    return tuple(float(item) * float(scalar) for item in value)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBBlocked("INPUT_INVALID", field_name)
    return value


def _policy(model: Any, key: str, default: Any) -> Any:
    value = getattr(model, "policy", {})
    if isinstance(value, Mapping):
        return value.get(key, default)
    return default


def _model_hashes(model: Any) -> Mapping[str, str]:
    hashes = getattr(model, "hashes", {})
    if not isinstance(hashes, Mapping):
        raise StageBBlocked("MODEL_IDENTITY_UNAVAILABLE")
    required = ("kinematic_model_hash", "collision_proxy_hash", "tool_model_hash", "calibration_hash")
    result = {}
    for key in required:
        result[key] = _digest(hashes.get(key), f"model.{key}")
    return result


def _validate_limits(model: Any, q: Sequence[float], *, field_name: str) -> None:
    values = _vector(q, 6, field_name)
    checker = getattr(model, "_validate_limits", None)
    if callable(checker):
        try:
            checker(values)
        except Exception as exc:  # noqa: BLE001 - normalize model failures
            raise StageBBlocked("JOINT_LIMIT_REJECTED", str(exc)) from exc
        return
    raw = _policy(model, "joint_hard_limits_rad", None)
    if raw is None:
        return
    if not isinstance(raw, Sequence) or len(raw) != 6:
        raise StageBBlocked("POLICY_INVALID", "joint_hard_limits_rad")
    for index, (value, bounds) in enumerate(zip(values, raw)):
        lower, upper = _vector(bounds, 2, f"joint_hard_limits_rad[{index}]")
        if lower > upper or not lower <= value <= upper:
            raise StageBBlocked("JOINT_LIMIT_REJECTED", str(index))


def _joint_caps(model: Any) -> tuple[float, ...]:
    raw = _policy(model, "per_command_caps_rad", None)
    if raw is None:
        raw = _policy(model, "per_move_joint_caps_rad", None)
    if raw is None:
        raw = (0.05,) * 6
    caps = _vector(raw, 6, "per_command_caps_rad")
    if any(value <= 0.0 for value in caps):
        raise StageBBlocked("POLICY_INVALID", "per_command_caps_rad")
    return caps


def _hard_limits(model: Any) -> tuple[tuple[float, float], ...] | None:
    raw = _policy(model, "joint_hard_limits_rad", None)
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or len(raw) != 6:
        raise StageBBlocked("POLICY_INVALID", "joint_hard_limits_rad")
    result = []
    for index, bounds in enumerate(raw):
        low, high = _vector(bounds, 2, f"joint_hard_limits_rad[{index}]")
        if low > high:
            raise StageBBlocked("POLICY_INVALID", f"joint_hard_limits_rad[{index}]")
        result.append((low, high))
    return tuple(result)


def _validate_fixed_joints(q: Sequence[float], q0: Sequence[float], *, field_name: str) -> None:
    for index in LOCKED_JOINT_INDICES:
        if abs(float(q[index]) - float(q0[index])) > 1e-12:
            raise StageBBlocked("FIXED_WRIST_JOINT_DRIFT", f"{field_name}[{index}]")


def _path_distance(a0: Sequence[float], a1: Sequence[float], b0: Sequence[float], b1: Sequence[float]) -> float:
    """Distance between two 3-D segments, used by the accepted proxy checks."""

    u = _sub(a1, a0)
    v = _sub(b1, b0)
    w = _sub(a0, b0)
    aa, bb, cc = _dot(u, u), _dot(u, v), _dot(v, v)
    dd, ee = _dot(u, w), _dot(v, w)
    epsilon = 1e-15
    if aa <= epsilon and cc <= epsilon:
        return _norm(w)
    if aa <= epsilon:
        return _segment_point_distance(a0, b0, b1)
    if cc <= epsilon:
        return _segment_point_distance(b0, a0, a1)
    denominator = aa * cc - bb * bb
    s = 0.0 if denominator <= epsilon else max(0.0, min(1.0, (bb * ee - cc * dd) / denominator))
    t = (bb * s + ee) / cc
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -dd / aa))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (bb - dd) / aa))
    return _norm(_sub(_add(a0, _scale(u, s)), _add(b0, _scale(v, t))))


def _segment_point_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    segment = _sub(end, start)
    denominator = _dot(segment, segment)
    ratio = 0.0 if denominator <= 1e-15 else max(
        0.0, min(1.0, _dot(_sub(point, start), segment) / denominator)
    )
    return _norm(_sub(point, _add(start, _scale(segment, ratio))))


def _contact_depth_from_endpoint(
    model: Any,
    endpoint_q: Sequence[float],
    *,
    wall_surface_point_base_m: Sequence[float],
    normal_out_base: Sequence[float],
    positional_contact_capsules: Sequence[str],
) -> float:
    """Lock contact allowance to the frozen endpoint's named centerlines."""

    contact_capsules = tuple(str(value) for value in positional_contact_capsules)
    if not contact_capsules or len(set(contact_capsules)) != len(contact_capsules):
        raise StageBBlocked("CONTACT_POLICY_INVALID", "contact capsule")
    wall = _vector(wall_surface_point_base_m, 3, "wall_surface_point_base_m")
    normal = _unit(normal_out_base, "normal_out_base")
    policy = getattr(model, "policy", {})
    capsules_policy = policy.get("capsules") if isinstance(policy, Mapping) else None
    if not isinstance(capsules_policy, Sequence):
        raise StageBBlocked("CONTACT_POLICY_INVALID", "capsules")
    try:
        state = model.fk(_vector(endpoint_q, 6, "contact endpoint q"))
    except StageBBlocked:
        raise
    except Exception as exc:  # noqa: BLE001 - model boundary is fail-closed
        raise StageBBlocked("COLLISION_MODEL_UNAVAILABLE", "contact endpoint") from exc
    points = getattr(state, "points", None)
    if not isinstance(points, Sequence):
        raise StageBBlocked("COLLISION_MODEL_UNAVAILABLE", "fk.points")
    depths: dict[str, float] = {}
    for item in capsules_policy:
        if not isinstance(item, Mapping):
            raise StageBBlocked("POLICY_INVALID", "capsules")
        name = str(item.get("name") or "")
        if name not in contact_capsules:
            continue
        if name in depths:
            raise StageBBlocked("CONTACT_POLICY_INVALID", "duplicate contact capsule")
        try:
            first = _vector(points[int(item["joint_a"])], 3, f"capsule {name}.endpoint_a")
            second = _vector(points[int(item["joint_b"])], 3, f"capsule {name}.endpoint_b")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StageBBlocked("POLICY_INVALID", f"capsule {name}") from exc
        centerline_wall_distance = min(
            _dot(normal, _sub(first, wall)),
            _dot(normal, _sub(second, wall)),
        )
        depths[name] = max(0.0, -centerline_wall_distance)
    if set(depths) != set(contact_capsules):
        raise StageBBlocked("CONTACT_POLICY_INVALID", "contact capsule unavailable")
    return max(depths.values())


def validate_model_path(
    model: Any,
    start_q: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    *,
    wall_surface_point_base_m: Sequence[float],
    normal_out_base: Sequence[float],
    positional_contact_capsules: Sequence[str] = (),
    positional_contact_depth_limit_m: float = 0.0,
) -> None:
    """Sample every command segment with the accepted dynamic collision model."""

    wall = _vector(wall_surface_point_base_m, 3, "wall_surface_point_base_m")
    normal = _unit(normal_out_base, "normal_out_base")
    # Reuse the accepted Stage-A-equivalent sampler when the loaded model
    # exposes its complete dynamic policy.  The small local fallback below is
    # retained for pure test doubles that intentionally omit optional proxy
    # sections; it follows the same wall/self/base/bracket gates.
    policy_value = getattr(model, "policy", {})
    contact_capsules = tuple(str(value) for value in positional_contact_capsules)
    contact_depth_limit = _finite(
        positional_contact_depth_limit_m, "positional_contact_depth_limit_m"
    )
    if contact_depth_limit < 0.0:
        raise StageBBlocked("CONTACT_POLICY_INVALID", "negative contact depth")
    if len(set(contact_capsules)) != len(contact_capsules):
        raise StageBBlocked("CONTACT_POLICY_INVALID", "duplicate contact capsule")
    if (
        not contact_capsules
        and isinstance(policy_value, Mapping)
        and {"capsules", "self_pairs"}.issubset(policy_value)
    ):
        try:
            from press.t102_post_precontact_xy_core import model_path_validator as accepted_path_validator
            accepted = accepted_path_validator(
                model,
                wall_surface_point_base_m=wall,
                panel_normal_base=normal,
            )
            previous_q = _vector(start_q, 6, "path start")
            for target_q in waypoints:
                accepted(previous_q, _vector(target_q, 6, "path waypoint"))
                previous_q = _vector(target_q, 6, "path waypoint")
            return
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - normalize accepted helper failures
            code = str(getattr(exc, "code", "COLLISION_PATH_REJECTED"))
            raise StageBBlocked(code, str(exc)) from exc
    sample_count = int(_policy(model, "segment_sample_count", 3))
    if sample_count < 2:
        raise StageBBlocked("POLICY_INVALID", "segment_sample_count")
    policy = getattr(model, "policy", {})
    if not isinstance(policy, Mapping):
        policy = {}
    capsules_policy = policy.get("capsules", ())
    configured_capsule_names = {
        str(item.get("name") or "") for item in capsules_policy if isinstance(item, Mapping)
    }
    if any(name not in configured_capsule_names for name in contact_capsules):
        raise StageBBlocked("CONTACT_POLICY_INVALID", "contact capsule unavailable")
    self_pairs = policy.get("self_pairs", ())
    wall_margin = _finite(policy.get("wall_margin_m", 0.0), "wall_margin_m")
    base_checks = set(policy.get("base_check_capsules", ()))
    bracket_checks = set(policy.get("bracket_check_capsules", ()))

    previous = _vector(start_q, 6, "path start")
    all_segments = tuple(waypoints)
    for segment_index, target_raw in enumerate(all_segments):
        target = _vector(target_raw, 6, f"path waypoint {segment_index}")
        for sample_index in range(sample_count):
            ratio = sample_index / (sample_count - 1)
            q = _add(previous, _scale(_sub(target, previous), ratio))
            try:
                state = model.fk(q)
            except Exception as exc:  # noqa: BLE001 - model boundary is fail-closed
                raise StageBBlocked("COLLISION_MODEL_UNAVAILABLE", str(exc)) from exc
            points = getattr(state, "points", None)
            if not isinstance(points, Sequence):
                raise StageBBlocked("COLLISION_MODEL_UNAVAILABLE", "fk.points")
            capsules: dict[str, tuple[Sequence[float], Sequence[float], float]] = {}
            for item in capsules_policy:
                if not isinstance(item, Mapping):
                    raise StageBBlocked("POLICY_INVALID", "capsules")
                name = str(item.get("name") or "")
                try:
                    first = points[int(item["joint_a"])]
                    second = points[int(item["joint_b"])]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise StageBBlocked("POLICY_INVALID", f"capsule {name}") from exc
                radius = _finite(item.get("radius_m"), f"capsule {name}.radius_m")
                if radius < 0.0:
                    raise StageBBlocked("POLICY_INVALID", f"capsule {name}.radius_m")
                capsules[name] = (first, second, radius)
                centerline_wall_distance = min(
                    _dot(normal, _sub(first, wall)),
                    _dot(normal, _sub(second, wall)),
                )
                if name in contact_capsules:
                    # Positional contact authority permits only the named
                    # tool centerline to reach the frozen surface, never pass
                    # through it. Capsule radius is not treated as forbidden
                    # wall overlap for that one contact body.
                    if centerline_wall_distance < -contact_depth_limit - APPROACH_GEOMETRY_TOLERANCE_M:
                        raise StageBBlocked("CONTACT_WALL_OVERSHOOT", name)
                elif centerline_wall_distance - radius < wall_margin:
                    raise StageBBlocked("WALL_HALFSPACE_REJECTED", name)
                for proxy_key, configured in (
                    ("base_proxy", base_checks),
                    ("camera_bracket_proxy", bracket_checks),
                ):
                    if name not in configured:
                        continue
                    proxy = policy.get(proxy_key)
                    if not isinstance(proxy, Mapping):
                        raise StageBBlocked("POLICY_INVALID", proxy_key)
                    primitive = str(proxy.get("primitive") or "sphere")
                    if primitive == "sphere":
                        center = _vector(proxy.get("center_m"), 3, f"{proxy_key}.center_m")
                        distance = _segment_point_distance(center, first, second)
                    elif primitive == "capsule":
                        distance = _path_distance(
                            first,
                            second,
                            _vector(proxy.get("endpoint_a_m"), 3, f"{proxy_key}.endpoint_a_m"),
                            _vector(proxy.get("endpoint_b_m"), 3, f"{proxy_key}.endpoint_b_m"),
                        )
                    else:
                        raise StageBBlocked("POLICY_INVALID", proxy_key)
                    proxy_radius = _finite(proxy.get("radius_m"), f"{proxy_key}.radius_m")
                    if distance < radius + proxy_radius:
                        raise StageBBlocked("BASE_BRACKET_PROXY_REJECTED", proxy_key)
            for pair in self_pairs:
                if not isinstance(pair, Sequence) or len(pair) != 2:
                    raise StageBBlocked("POLICY_INVALID", "self_pairs")
                try:
                    left = capsules[str(pair[0])]
                    right = capsules[str(pair[1])]
                except KeyError as exc:
                    raise StageBBlocked("POLICY_INVALID", "self_pairs") from exc
                if _path_distance(left[0], left[1], right[0], right[1]) < left[2] + right[2]:
                    raise StageBBlocked("SELF_PROXY_REJECTED", f"{pair[0]}/{pair[1]}")
        previous = target


@dataclass(frozen=True)
class CurrentPanelPressReceipt:
    """Fresh actual-q, accepted panel geometry, and click-scoped wrist lock."""

    schema: str
    receipt_id: str
    receipt_sha256: str
    panel_geometry_sha256: str
    terminal_q_rad: tuple[float, ...]
    p_button_base_m: tuple[float, ...]
    normal_out_base: tuple[float, ...]
    wall_surface_point_base_m: tuple[float, ...]
    kinematic_model_hash: str
    collision_proxy_hash: str
    tool_model_hash: str
    calibration_hash: str
    feedback_stamp_ns: int
    feedback_fresh: bool
    stationary_sample_count: int
    stationary_span_rad: float
    cleanup_proven: bool
    joint_target_publishers: int
    lease_active: bool
    armed: bool
    teardown_complete: bool
    terminal_resources: Mapping[str, int]
    wrist_acquisition_id: str
    wrist_lock_identity: tuple[str, ...]
    wrist_lock_stage: str
    wrist_lock_center_px: tuple[float, ...]
    wrist_lock_observation_count: int
    gemini_p_button_base_m: tuple[float, ...]
    gemini_wall_surface_point_base_m: tuple[float, ...]
    wrist_tangent_delta_base_m: tuple[float, ...]
    wrist_tangent_calibration_sha256: str
    wrist_image_error_px: tuple[float, ...]
    wrist_correction_applied: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], supplied_sha256: str | None = None) -> "CurrentPanelPressReceipt":
        value = _mapping(raw, "panel_press_receipt")
        schema = str(value.get("schema") or "")
        if schema not in {RECEIPT_SCHEMA, WRIST_CORRECTED_RECEIPT_SCHEMA}:
            raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", "schema")

        def text(name: str) -> str:
            result = str(value.get(name) or "").strip()
            if not result:
                raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", name)
            return result

        def number(name: str) -> Any:
            if name not in value:
                raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", name)
            return value[name]

        lock_identity = value.get("wrist_lock_identity")
        if not isinstance(lock_identity, Sequence) or isinstance(lock_identity, (str, bytes)):
            raise StageBBlocked("WRIST_LOCK_INVALID", "identity")
        resources = value.get("terminal_resources", {})
        p_button = _vector(value.get("p_button_base_m"), 3, "p_button_base_m")
        wall_surface = _vector(
            value.get("wall_surface_point_base_m"), 3, "wall_surface_point_base_m"
        )
        corrected = schema == WRIST_CORRECTED_RECEIPT_SCHEMA
        receipt = cls(
            schema=schema,
            receipt_id=text("receipt_id"), receipt_sha256=text("receipt_sha256"),
            panel_geometry_sha256=text("panel_geometry_sha256"),
            terminal_q_rad=_vector(value.get("terminal_q_rad"), 6, "terminal_q_rad"),
            p_button_base_m=p_button,
            normal_out_base=_unit(value.get("normal_out_base"), "normal_out_base"),
            wall_surface_point_base_m=wall_surface,
            kinematic_model_hash=text("kinematic_model_hash"),
            collision_proxy_hash=text("collision_proxy_hash"),
            tool_model_hash=text("tool_model_hash"), calibration_hash=text("calibration_hash"),
            feedback_stamp_ns=int(number("feedback_stamp_ns")),
            feedback_fresh=value.get("feedback_fresh") is True,
            stationary_sample_count=int(number("stationary_sample_count")),
            stationary_span_rad=_finite(number("stationary_span_rad"), "stationary_span_rad"),
            cleanup_proven=value.get("cleanup_proven") is True,
            joint_target_publishers=int(number("joint_target_publishers")),
            lease_active=value.get("lease_active") is True, armed=value.get("armed") is True,
            teardown_complete=value.get("teardown_complete") is True,
            terminal_resources=resources if isinstance(resources, Mapping) else {},
            wrist_acquisition_id=text("wrist_acquisition_id"),
            wrist_lock_identity=tuple(str(item) for item in lock_identity),
            wrist_lock_stage=text("wrist_lock_stage"),
            wrist_lock_center_px=_vector(value.get("wrist_lock_center_px"), 2, "wrist_lock_center_px"),
            wrist_lock_observation_count=int(number("wrist_lock_observation_count")),
            gemini_p_button_base_m=(
                _vector(value.get("gemini_p_button_base_m"), 3, "gemini_p_button_base_m")
                if corrected else p_button
            ),
            gemini_wall_surface_point_base_m=(
                _vector(
                    value.get("gemini_wall_surface_point_base_m"),
                    3,
                    "gemini_wall_surface_point_base_m",
                )
                if corrected else wall_surface
            ),
            wrist_tangent_delta_base_m=(
                _vector(value.get("wrist_tangent_delta_base_m"), 3, "wrist_tangent_delta_base_m")
                if corrected else (0.0, 0.0, 0.0)
            ),
            wrist_tangent_calibration_sha256=(
                text("wrist_tangent_calibration_sha256") if corrected else ""
            ),
            wrist_image_error_px=(
                _vector(value.get("wrist_image_error_px"), 2, "wrist_image_error_px")
                if corrected else (0.0, 0.0)
            ),
            wrist_correction_applied=(
                value.get("wrist_correction_applied") is True if corrected else False
            ),
        )
        if supplied_sha256 is not None and receipt.receipt_sha256 != supplied_sha256:
            raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", "supplied hash")
        receipt.validate(value)
        return receipt

    def unsigned_mapping(self) -> Mapping[str, Any]:
        value = asdict(self)
        value.pop("receipt_id", None)
        value.pop("receipt_sha256", None)
        if self.schema == RECEIPT_SCHEMA:
            for field in (
                "gemini_p_button_base_m",
                "gemini_wall_surface_point_base_m",
                "wrist_tangent_delta_base_m",
                "wrist_tangent_calibration_sha256",
                "wrist_image_error_px",
                "wrist_correction_applied",
            ):
                value.pop(field, None)
        return value

    def validate(self, raw: Mapping[str, Any] | None = None) -> None:
        if self.receipt_id != self.receipt_sha256:
            raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", "receipt id/hash mismatch")
        for name in ("receipt_sha256", "panel_geometry_sha256", "kinematic_model_hash", "collision_proxy_hash", "tool_model_hash", "calibration_hash"):
            _digest(getattr(self, name), name)
        if raw is not None:
            unsigned = dict(raw); unsigned.pop("receipt_id", None); unsigned.pop("receipt_sha256", None)
            if canonical_hash(unsigned) != self.receipt_sha256:
                raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", "canonical digest")
        _vector(self.terminal_q_rad, 6, "terminal_q_rad")
        _vector(self.p_button_base_m, 3, "p_button_base_m")
        _unit(self.normal_out_base, "normal_out_base")
        _vector(self.wall_surface_point_base_m, 3, "wall_surface_point_base_m")
        if self.schema == WRIST_CORRECTED_RECEIPT_SCHEMA:
            _digest(
                self.wrist_tangent_calibration_sha256,
                "wrist_tangent_calibration_sha256",
            )
            delta = _vector(
                self.wrist_tangent_delta_base_m, 3, "wrist_tangent_delta_base_m"
            )
            gemini_button = _vector(
                self.gemini_p_button_base_m, 3, "gemini_p_button_base_m"
            )
            gemini_wall = _vector(
                self.gemini_wall_surface_point_base_m,
                3,
                "gemini_wall_surface_point_base_m",
            )
            if self.wrist_correction_applied is not True:
                raise StageBBlocked("WRIST_CORRECTION_REQUIRED")
            if _norm(delta) > MAX_WRIST_TANGENT_CORRECTION_M + 1e-12:
                raise StageBBlocked("WRIST_CORRECTION_TOO_LARGE")
            if abs(_dot(delta, self.normal_out_base)) > 1e-9:
                raise StageBBlocked("WRIST_CORRECTION_NORMAL_COMPONENT")
            if any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(
                    self.p_button_base_m, _add(gemini_button, delta)
                )
            ) or any(
                abs(actual - expected) > 1e-12
                for actual, expected in zip(
                    self.wall_surface_point_base_m, _add(gemini_wall, delta)
                )
            ):
                raise StageBBlocked("WRIST_CORRECTION_GEOMETRY_MISMATCH")
        if self.feedback_stamp_ns <= 0 or not self.feedback_fresh or self.stationary_sample_count < 3 or not math.isfinite(self.stationary_span_rad) or self.stationary_span_rad < 0.0:
            raise StageBBlocked("ARRIVAL_RECEIPT_POSTURE_UNPROVEN")
        if not self.cleanup_proven or self.joint_target_publishers != 0 or self.lease_active or self.armed or not self.teardown_complete:
            raise StageBBlocked("ARRIVAL_RECEIPT_UNSAFE", "current resources")
        if not isinstance(self.terminal_resources, Mapping) or not self.terminal_resources or any(type(item) is not int or item != 0 for item in self.terminal_resources.values()):
            raise StageBBlocked("ARRIVAL_RECEIPT_UNSAFE", "terminal resources")
        if not self.wrist_acquisition_id or not self.wrist_lock_identity or self.wrist_lock_stage != "PRE-LOCK" or self.wrist_lock_observation_count < 3:
            raise StageBBlocked("WRIST_LOCK_INVALID")
        _vector(self.wrist_lock_center_px, 2, "wrist_lock_center_px")


PreTerminalReceipt = CurrentPanelPressReceipt
StageAPreTerminalReceipt = CurrentPanelPressReceipt


@dataclass(frozen=True)
class FrozenStageBPlan:
    plan: Mapping[str, Any]
    plan_sha256: str
    receipt: PreTerminalReceipt

    @property
    def forward_waypoints_rad(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(point) for point in self.plan["forward_waypoints_rad"])

    @property
    def return_waypoints_rad(self) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(point) for point in self.plan["return_waypoints_rad"])


def _solve(model: Any, seed: Sequence[float], target_tcp: Sequence[float], desired_axis: Sequence[float]) -> tuple[float, ...]:
    solver = getattr(model, "solve", None)
    if not callable(solver):
        raise StageBBlocked("IK_UNAVAILABLE")
    try:
        audit = solver(seed, target_tcp, desired_axis)
    except Exception as exc:  # noqa: BLE001
        raise StageBBlocked("IK_FAILED", str(exc)) from exc
    q = getattr(audit, "q", audit)
    result = _vector(q, 6, "ik.q")
    expected_seed_hash = canonical_hash(list(seed))
    actual_seed_hash = getattr(audit, "seed_hash", None)
    if actual_seed_hash is not None and actual_seed_hash != expected_seed_hash:
        raise StageBBlocked("IK_BRANCH_DISCONTINUITY", "seed hash")
    residual = getattr(audit, "position_residual_m", None)
    if residual is not None and _finite(residual, "ik.position_residual_m") > _finite(
        _policy(model, "position_residual_limit_m", DEFAULT_POSITION_RESIDUAL_M), "position_residual_limit_m"
    ):
        raise StageBBlocked("IK_RESIDUAL_EXCEEDED")
    axis_residual = getattr(audit, "axis_residual_deg", None)
    if axis_residual is not None and _finite(axis_residual, "ik.axis_residual_deg") > _finite(
        _policy(model, "axis_residual_limit_deg", DEFAULT_AXIS_RESIDUAL_DEG), "axis_residual_limit_deg"
    ):
        raise StageBBlocked("ORIENTATION_WINDOW_MISS")
    condition = getattr(audit, "condition", None)
    manipulability = getattr(audit, "manipulability", None)
    if condition is not None and not math.isfinite(float(condition)):
        raise StageBBlocked("IK_NEAR_SINGULAR")
    if manipulability is not None and not math.isfinite(float(manipulability)):
        raise StageBBlocked("IK_NEAR_SINGULAR")
    return result


def _partition(start: Sequence[float], endpoint: Sequence[float], caps: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    delta = _sub(endpoint, start)
    if max(abs(value) for value in delta) <= 1e-12:
        raise StageBBlocked("NO_FORWARD_MOTION")
    steps = max(1, max(int(math.ceil(abs(value) / cap)) for value, cap in zip(delta, caps)))
    return tuple(_add(start, _scale(delta, index / steps)) for index in range(1, steps + 1))


def _validate_adjacent_caps(
    start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    caps: Sequence[float],
    *,
    detail_prefix: str,
) -> None:
    """Enforce the unchanged per-command cap on a frozen waypoint path."""

    previous = _vector(start, 6, f"{detail_prefix} start")
    for index, raw_point in enumerate(waypoints):
        point = _vector(raw_point, 6, f"{detail_prefix} {index}")
        if any(
            abs(value - prior) > cap + 1e-12
            for value, prior, cap in zip(point, previous, caps)
        ):
            raise StageBBlocked("WAYPOINT_CAP_REJECTED", f"{detail_prefix} {index}")
        previous = point


def _quantize_runtime_waypoint(
    waypoint: Sequence[float],
    q0: Sequence[float],
    encoder_step_rad: float,
) -> tuple[float, ...]:
    """Reproduce SafeDriver's Python-round target quantization before freeze."""

    point = _vector(waypoint, 6, "runtime quantization waypoint")
    start = _vector(q0, 6, "runtime quantization q0")
    step = _finite(encoder_step_rad, "runtime encoder step")
    if step <= 0.0:
        raise StageBBlocked("POLICY_INVALID", "encoder_step_rad")
    result = []
    for index, value in enumerate(point):
        if index in LOCKED_JOINT_INDICES:
            result.append(start[index])
            continue
        quantized = round(value / step) * step
        lower, upper = SAFE_DRIVER_COMMAND_LIMITS_RAD[index]
        result.append(min(upper, max(lower, quantized)))
    return tuple(result)


def _buffered_cap_utilization(
    start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    caps: Sequence[float],
    *,
    first_segment_buffer_rad: float,
    subsequent_segment_buffer_rad: float,
    minimum_command_delta_rad: float,
) -> float | None:
    """Return worst buffered cap use, or ``None`` for an unsafe command path."""

    previous = _vector(start, 6, "runtime cap start")
    cap_values = _vector(caps, 6, "runtime caps")
    first_buffer = _finite(first_segment_buffer_rad, "first segment cap buffer")
    later_buffer = _finite(subsequent_segment_buffer_rad, "subsequent segment cap buffer")
    minimum_delta = _finite(minimum_command_delta_rad, "minimum command delta")
    if first_buffer < 0.0 or later_buffer < 0.0 or minimum_delta < 0.0:
        raise StageBBlocked("POLICY_INVALID", "runtime cap buffer")
    worst = 0.0
    for segment_index, raw_point in enumerate(waypoints):
        point = _vector(raw_point, 6, f"runtime cap waypoint {segment_index}")
        deltas = tuple(abs(value - prior) for value, prior in zip(point, previous))
        # A frozen command must remain meaningful even at the allowed
        # convergence edge; otherwise SafeDriver can correctly reject it as a
        # no-op rather than acknowledge a new command.
        if max(deltas) <= minimum_delta + 1e-12:
            return None
        buffer = first_buffer if segment_index == 0 else later_buffer
        if any(delta + buffer > cap + 1e-12 for delta, cap in zip(deltas, cap_values)):
            return None
        worst = max(
            worst,
            max((delta + buffer) / cap for delta, cap in zip(deltas, cap_values)),
        )
        previous = point
    return worst


def _audit_runtime_normal_path(
    model: Any,
    start_q: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    *,
    reference_tcp_base_m: Sequence[float],
    forward_direction_base: Sequence[float],
    current_panel_gap_m: float,
    residual_limit_m: float,
    retreat: bool,
    tangent_offset_base_m: Sequence[float] = (0.0, 0.0, 0.0),
    sample_count: int = RUNTIME_GEOMETRY_SAMPLE_COUNT,
) -> Mapping[str, float | int]:
    """Densely audit the actual joint-linear runtime chords in FK space."""

    count = max(RUNTIME_GEOMETRY_SAMPLE_COUNT, int(sample_count))
    reference = _vector(reference_tcp_base_m, 3, "runtime geometry reference TCP")
    direction = _unit(forward_direction_base, "runtime geometry direction")
    gap = _finite(current_panel_gap_m, "runtime geometry gap")
    tangent_offset = _vector(
        tangent_offset_base_m, 3, "runtime geometry tangent offset"
    )
    if abs(_dot(tangent_offset, direction)) > 1e-9:
        raise StageBBlocked("WRIST_CORRECTION_NORMAL_COMPONENT")
    residual = _finite(residual_limit_m, "runtime geometry residual")
    if gap <= 0.0 or residual <= 0.0:
        raise StageBBlocked("POLICY_INVALID", "runtime geometry bounds")
    previous_q = _vector(start_q, 6, "runtime geometry start")
    previous_projection = math.inf if retreat else -math.inf
    maximum_lateral = 0.0
    reverse_steps = 0
    for segment_index, raw_target in enumerate(waypoints):
        target = _vector(raw_target, 6, f"runtime geometry waypoint {segment_index}")
        for sample_index in range(count):
            if segment_index and sample_index == 0:
                continue
            ratio = sample_index / (count - 1)
            q = _add(previous_q, _scale(_sub(target, previous_q), ratio))
            tcp = _fk_tcp(model, q, f"runtime geometry TCP {segment_index}/{sample_index}")
            delta = _sub(tcp, reference)
            projection = _dot(delta, direction)
            path_ratio = max(0.0, min(1.0, projection / gap))
            expected_tangent = _scale(tangent_offset, path_ratio)
            lateral = _norm(
                _sub(
                    _sub(delta, _scale(direction, projection)),
                    expected_tangent,
                )
            )
            if retreat:
                reversed_step = projection > previous_projection + APPROACH_GEOMETRY_TOLERANCE_M
            else:
                reversed_step = projection + APPROACH_GEOMETRY_TOLERANCE_M < previous_projection
            if reversed_step:
                reverse_steps += 1
            if (
                reversed_step
                or projection < -residual
                or projection > gap + residual
                or lateral > residual
            ):
                raise StageBBlocked("APPROACH_PATH_NON_MONOTONIC", f"runtime segment {segment_index}")
            previous_projection = projection
            maximum_lateral = max(maximum_lateral, lateral)
        previous_q = target
    return {
        "sample_count_per_segment": count,
        "reverse_step_count": reverse_steps,
        "maximum_lateral_m": maximum_lateral,
        "terminal_projection_m": previous_projection,
    }


def _select_runtime_waypoints(
    quantized_dense_waypoints: Sequence[Sequence[float]],
    q0: Sequence[float],
    caps: Sequence[float],
    *,
    first_segment_buffer_rad: float,
    subsequent_segment_buffer_rad: float,
    minimum_command_delta_rad: float,
    validate_candidate: Callable[[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]], None],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[int, ...], float]:
    """Choose up to four valid ordered dense-path points, deterministically.

    Prefer the largest allowed waypoint count so each runtime joint command
    follows a shorter Cartesian chord.  If a four-point candidate cannot pass
    the existing cap, collision, and normal-path audits, fall back to three and
    then two points without weakening any admission gate.
    """

    dense = tuple(
        _vector(point, 6, f"quantized dense waypoint {index}")
        for index, point in enumerate(quantized_dense_waypoints)
    )
    start = _vector(q0, 6, "runtime selector q0")
    if len(dense) < RUNTIME_MIN_FORWARD_WAYPOINTS:
        raise StageBBlocked("RUNTIME_WAYPOINT_SELECTION_FAILED", "dense path too short")
    last_failure = "no cap-valid candidate"
    maximum_count = min(RUNTIME_MAX_FORWARD_WAYPOINTS, len(dense))
    for count in range(maximum_count, RUNTIME_MIN_FORWARD_WAYPOINTS - 1, -1):
        ranked: list[
            tuple[float, tuple[int, ...], tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]
        ] = []
        for internal in combinations(range(len(dense) - 1), count - 1):
            zero_based_indices = tuple(internal) + (len(dense) - 1,)
            forward = tuple(dense[index] for index in zero_based_indices)
            back = tuple(reversed(forward[:-1])) + (start,)
            forward_score = _buffered_cap_utilization(
                start,
                forward,
                caps,
                first_segment_buffer_rad=first_segment_buffer_rad,
                subsequent_segment_buffer_rad=subsequent_segment_buffer_rad,
                minimum_command_delta_rad=minimum_command_delta_rad,
            )
            back_score = _buffered_cap_utilization(
                forward[-1],
                back,
                caps,
                first_segment_buffer_rad=subsequent_segment_buffer_rad,
                subsequent_segment_buffer_rad=subsequent_segment_buffer_rad,
                minimum_command_delta_rad=minimum_command_delta_rad,
            )
            if forward_score is None or back_score is None:
                continue
            source_indices = tuple(index + 1 for index in zero_based_indices)
            ranked.append((max(forward_score, back_score), source_indices, forward, back))
        for score, source_indices, forward, back in sorted(ranked, key=lambda item: (item[0], item[1])):
            try:
                validate_candidate(forward, back)
            except StageBBlocked as exc:
                last_failure = exc.code
                continue
            return forward, back, source_indices, score
    raise StageBBlocked("RUNTIME_WAYPOINT_SELECTION_FAILED", last_failure)


def _fk_tcp(model: Any, q: Sequence[float], field_name: str) -> tuple[float, float, float]:
    try:
        state = model.fk(q)
        return _vector(getattr(state, "tcp"), 3, field_name)  # type: ignore[return-value]
    except StageBBlocked:
        raise
    except Exception as exc:  # noqa: BLE001
        raise StageBBlocked("KINEMATIC_MODEL_UNAVAILABLE", field_name) from exc


def prepare_stage_b_plan(
    receipt: PreTerminalReceipt | Mapping[str, Any],
    model: Any,
    *,
    current_q_rad: Sequence[float] | None = None,
    now_ns: int | None = None,
    now_monotonic: float = 0.0,
    q_drift_tolerance_rad: float | None = None,
    contact_authorized: bool = False,
    press_authorized: bool = False,
    force_control: bool = False,
    zero_dwell: bool = True,
    require_wrist_tangent_correction: bool = False,
) -> FrozenStageBPlan:
    """Freeze one zero-resource Stage-B plan from a fresh PRE receipt."""

    parsed = receipt if isinstance(receipt, PreTerminalReceipt) else PreTerminalReceipt.from_mapping(receipt)
    parsed.validate()
    if require_wrist_tangent_correction and (
        parsed.schema != WRIST_CORRECTED_RECEIPT_SCHEMA
        or not parsed.wrist_correction_applied
    ):
        raise StageBBlocked("WRIST_CORRECTION_REQUIRED")
    if type(contact_authorized) is not bool or not contact_authorized:
        raise StageBBlocked("CONTACT_AUTHORIZATION_DISABLED")
    if type(press_authorized) is not bool:
        raise StageBBlocked("CONTACT_POLICY_INVALID", "press_authorized")
    if force_control is not False:
        raise StageBBlocked("FORCE_CONTROL_OUT_OF_SCOPE")
    if zero_dwell is not True:
        raise StageBBlocked("DWELL_NOT_PERMITTED")
    if now_ns is not None:
        max_age_ns = int(_finite(_policy(model, "maximum_q0_age_s", 1.0), "maximum_q0_age_s") * 1e9)
        if now_ns < parsed.feedback_stamp_ns or now_ns - parsed.feedback_stamp_ns > max_age_ns:
            raise StageBBlocked("ARRIVAL_RECEIPT_STALE")
    q0 = _vector(parsed.terminal_q_rad, 6, "terminal_q_rad")
    current = q0 if current_q_rad is None else _vector(current_q_rad, 6, "current_q_rad")
    tolerance = (
        float(q_drift_tolerance_rad)
        if q_drift_tolerance_rad is not None
        else float(_policy(model, "q0_drift_limit_rad", DEFAULT_Q_DRIFT_TOLERANCE_RAD))
    )
    if not math.isfinite(tolerance) or tolerance < 0.0 or max(abs(a - b) for a, b in zip(current, q0)) > tolerance:
        raise StageBBlocked("START_Q_DRIFT")
    hashes = _model_hashes(model)
    for key, expected in hashes.items():
        if getattr(parsed, key) != expected:
            raise StageBBlocked("MODEL_IDENTITY_MISMATCH", key)
    _validate_limits(model, q0, field_name="terminal_q_rad")
    normal = _unit(parsed.normal_out_base, "normal_out_base")
    forward_direction = tuple(-value for value in normal)
    start_tcp = _fk_tcp(model, q0, "start_tcp_base_m")
    # Receipt correction is relative to Gemini's target, not the starting
    # fingertip. Aim at the corrected absolute point from the measured PRE.
    to_button = _sub(parsed.p_button_base_m, start_tcp)
    tangent_delta = _sub(to_button, _scale(normal, _dot(normal, to_button)))
    residual_limit = _finite(
        _policy(model, "position_residual_limit_m", DEFAULT_POSITION_RESIDUAL_M),
        "position_residual_limit_m",
    )
    if residual_limit <= 0.0:
        raise StageBBlocked("POLICY_INVALID", "position_residual_limit_m")
    current_panel_gap = _dot(normal, _sub(start_tcp, parsed.p_button_base_m))
    if not math.isfinite(current_panel_gap) or current_panel_gap < MIN_CURRENT_PANEL_GAP_M:
        raise StageBBlocked("CURRENT_PANEL_GAP_INVALID", f"{current_panel_gap:.12g}")
    commanded_forward_distance = current_panel_gap + PRESS_OVERTRAVEL_M
    if commanded_forward_distance > MAX_CURRENT_PANEL_GAP_M:
        raise StageBBlocked(
            "CURRENT_PANEL_GAP_TOO_LARGE",
            f"commanded {commanded_forward_distance:.12g}",
        )
    target_tcp = _add(parsed.p_button_base_m, _scale(forward_direction, PRESS_OVERTRAVEL_M))
    button_plane_offset = _dot(normal, _sub(target_tcp, parsed.p_button_base_m))
    if abs(button_plane_offset + PRESS_OVERTRAVEL_M) > APPROACH_GEOMETRY_TOLERANCE_M:
        raise StageBBlocked("CONTACT_GEOMETRY_MISMATCH", "fixed button-plane overtravel")
    sample_count = max(1, int(_policy(model, "cartesian_sample_count", 3)))
    seed = q0
    audits: list[tuple[float, ...]] = []
    for index in range(1, sample_count + 1):
        ratio = index / sample_count
        desired = _add(
            _add(start_tcp, _scale(tangent_delta, ratio)),
            _scale(forward_direction, commanded_forward_distance * ratio),
        )
        q = _solve(model, seed, desired, forward_direction)
        _validate_limits(model, q, field_name=f"ik_waypoint[{index}]")
        _validate_fixed_joints(q, q0, field_name=f"ik_waypoint[{index}]")
        jump = max(abs(a - b) for a, b in zip(q, seed))
        if jump > float(_policy(model, "maximum_joint_jump_rad", DEFAULT_MAX_JOINT_JUMP_RAD)):
            raise StageBBlocked("IK_BRANCH_DISCONTINUITY", f"sample {index}")
        audits.append(q)
        seed = q
    raw_endpoint_q = audits[-1]
    caps = _joint_caps(model)
    # Every interpolation is completed here, before authorization, and then
    # frozen into the plan.  Runtime never inserts or edits a waypoint.
    forward_list: list[tuple[float, ...]] = []
    previous_q = q0
    for audit_q in audits:
        if max(abs(a - b) for a, b in zip(audit_q, previous_q)) <= 1e-12:
            continue
        forward_list.extend(_partition(previous_q, audit_q, caps))
        previous_q = audit_q
    if not forward_list or forward_list[-1] != raw_endpoint_q:
        raise StageBBlocked("NO_FORWARD_MOTION")
    offline_contact_depth_limit = _contact_depth_from_endpoint(
        model,
        raw_endpoint_q,
        wall_surface_point_base_m=parsed.wall_surface_point_base_m,
        normal_out_base=normal,
        positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
    )
    maximum_contact_depth = PRESS_OVERTRAVEL_M + residual_limit
    if offline_contact_depth_limit > maximum_contact_depth:
        raise StageBBlocked("CONTACT_DEPTH_LIMIT_EXCEEDED", "offline endpoint")
    offline_forward_waypoints = tuple(forward_list)
    offline_return_waypoints = tuple(reversed(offline_forward_waypoints[:-1])) + (q0,)
    _validate_adjacent_caps(
        q0,
        offline_forward_waypoints,
        caps,
        detail_prefix="offline forward",
    )
    _validate_adjacent_caps(
        offline_forward_waypoints[-1],
        offline_return_waypoints,
        caps,
        detail_prefix="offline return",
    )
    # The dense raw IK path remains the authoritative offline audit. Runtime
    # executes only a SafeDriver-quantized, cap-buffered shortcut selected from
    # this exact ordered path.
    validate_model_path(
        model,
        q0,
        offline_forward_waypoints,
        wall_surface_point_base_m=parsed.wall_surface_point_base_m,
        normal_out_base=normal,
        positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
        positional_contact_depth_limit_m=offline_contact_depth_limit,
    )
    validate_model_path(
        model,
        offline_forward_waypoints[-1],
        offline_return_waypoints,
        wall_surface_point_base_m=parsed.wall_surface_point_base_m,
        normal_out_base=normal,
        positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
        positional_contact_depth_limit_m=offline_contact_depth_limit,
    )

    previous_projection = -math.inf
    for index, waypoint in enumerate(offline_forward_waypoints):
        tcp = _fk_tcp(model, waypoint, f"offline_forward_tcp[{index}]")
        delta = _sub(tcp, start_tcp)
        projection = _dot(delta, forward_direction)
        expected_tangent = _scale(
            tangent_delta,
            max(0.0, min(1.0, projection / commanded_forward_distance)),
        )
        cross_axis = _norm(
            _sub(
                _sub(delta, _scale(forward_direction, projection)),
                expected_tangent,
            )
        )
        if (
            projection + APPROACH_GEOMETRY_TOLERANCE_M < previous_projection
            or projection < -APPROACH_GEOMETRY_TOLERANCE_M
            or projection > commanded_forward_distance + residual_limit
            or cross_axis > residual_limit
        ):
            raise StageBBlocked("APPROACH_PATH_NON_MONOTONIC", f"offline {index}")
        previous_projection = projection

    encoder_step = _finite(
        _policy(model, "encoder_step_rad", DEFAULT_ENCODER_STEP_RAD),
        "encoder_step_rad",
    )
    waypoint_tolerance = _finite(
        _policy(model, "move_waypoint_tolerance_rad", DEFAULT_WAYPOINT_TOLERANCE_RAD),
        "move_waypoint_tolerance_rad",
    )
    q0_drift_limit = _finite(
        _policy(model, "q0_drift_limit_rad", DEFAULT_Q_DRIFT_TOLERANCE_RAD),
        "q0_drift_limit_rad",
    )
    if encoder_step <= 0.0 or waypoint_tolerance <= 0.0 or q0_drift_limit < 0.0:
        raise StageBBlocked("POLICY_INVALID", "runtime command margins")
    first_segment_buffer = q0_drift_limit + waypoint_tolerance + encoder_step
    subsequent_segment_buffer = waypoint_tolerance + encoder_step
    quantized_dense_waypoints = tuple(
        _quantize_runtime_waypoint(point, q0, encoder_step)
        for point in offline_forward_waypoints
    )
    quantized_endpoint_q = quantized_dense_waypoints[-1]
    positional_contact_depth_limit = _contact_depth_from_endpoint(
        model,
        quantized_endpoint_q,
        wall_surface_point_base_m=parsed.wall_surface_point_base_m,
        normal_out_base=normal,
        positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
    )
    if positional_contact_depth_limit > maximum_contact_depth:
        raise StageBBlocked("CONTACT_DEPTH_LIMIT_EXCEEDED", "quantized endpoint")
    endpoint_tcp = _fk_tcp(model, quantized_endpoint_q, "forward_endpoint_tcp_base_m")
    actual_delta = _sub(endpoint_tcp, start_tcp)
    commanded_along = _dot(actual_delta, forward_direction)
    lateral = _norm(
        _sub(
            _sub(actual_delta, _scale(forward_direction, commanded_along)),
            tangent_delta,
        )
    )
    endpoint_target_residual = _norm(_sub(endpoint_tcp, target_tcp))
    endpoint_projection_error = abs(commanded_along - commanded_forward_distance)
    if any(
        metric > residual_limit
        for metric in (
            endpoint_target_residual,
            endpoint_projection_error,
            lateral,
        )
    ):
        raise StageBBlocked("IK_RESIDUAL_EXCEEDED", "measured actual-start endpoint")

    def validate_runtime_candidate(
        candidate_forward: tuple[tuple[float, ...], ...],
        candidate_back: tuple[tuple[float, ...], ...],
    ) -> None:
        for index, point in enumerate(candidate_forward + candidate_back):
            _validate_limits(model, point, field_name=f"runtime waypoint {index}")
            _validate_fixed_joints(point, q0, field_name=f"runtime waypoint {index}")
        validate_model_path(
            model,
            q0,
            candidate_forward,
            wall_surface_point_base_m=parsed.wall_surface_point_base_m,
            normal_out_base=normal,
            positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
            positional_contact_depth_limit_m=positional_contact_depth_limit,
        )
        validate_model_path(
            model,
            candidate_forward[-1],
            candidate_back,
            wall_surface_point_base_m=parsed.wall_surface_point_base_m,
            normal_out_base=normal,
            positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
            positional_contact_depth_limit_m=positional_contact_depth_limit,
        )
        _audit_runtime_normal_path(
            model,
            q0,
            candidate_forward,
            reference_tcp_base_m=start_tcp,
            forward_direction_base=forward_direction,
            current_panel_gap_m=commanded_forward_distance,
            residual_limit_m=residual_limit,
            retreat=False,
            tangent_offset_base_m=tangent_delta,
        )
        _audit_runtime_normal_path(
            model,
            candidate_forward[-1],
            candidate_back,
            reference_tcp_base_m=start_tcp,
            forward_direction_base=forward_direction,
            current_panel_gap_m=commanded_forward_distance,
            residual_limit_m=residual_limit,
            retreat=True,
            tangent_offset_base_m=tangent_delta,
        )

    forward_waypoints, return_waypoints, runtime_source_indices, runtime_cap_utilization = (
        _select_runtime_waypoints(
            quantized_dense_waypoints,
            q0,
            caps,
            first_segment_buffer_rad=first_segment_buffer,
            subsequent_segment_buffer_rad=subsequent_segment_buffer,
            minimum_command_delta_rad=encoder_step,
            validate_candidate=validate_runtime_candidate,
        )
    )
    sample_tcp = [
        list(_fk_tcp(model, waypoint, f"forward_sample_tcp[{index}]"))
        for index, waypoint in enumerate(forward_waypoints)
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "operation": "CONTACT_APPROACH_OUT_BACK",
        "panel_press_receipt_id": parsed.receipt_id,
        "panel_press_receipt_sha256": parsed.receipt_sha256,
        "panel_geometry_sha256": parsed.panel_geometry_sha256,
        "wrist_acquisition_id": parsed.wrist_acquisition_id,
        "wrist_lock_identity": list(parsed.wrist_lock_identity),
        "receipt_schema": parsed.schema,
        "gemini_p_button_base_m": list(parsed.gemini_p_button_base_m),
        "gemini_wall_surface_point_base_m": list(
            parsed.gemini_wall_surface_point_base_m
        ),
        "wrist_tangent_delta_base_m": list(parsed.wrist_tangent_delta_base_m),
        "wrist_tangent_calibration_sha256": parsed.wrist_tangent_calibration_sha256,
        "wrist_image_error_px": list(parsed.wrist_image_error_px),
        "wrist_correction_applied": parsed.wrist_correction_applied,
        "model_hashes": dict(hashes),
        "start_q_rad": list(q0),
        "forward_endpoint_q_rad": list(quantized_endpoint_q),
        "forward_waypoints_rad": [list(point) for point in forward_waypoints],
        "return_waypoints_rad": [list(point) for point in return_waypoints],
        "offline_forward_validation_waypoints_rad": [
            list(point) for point in offline_forward_waypoints
        ],
        "locked_joint_indices": list(LOCKED_JOINT_INDICES),
        "start_tcp_base_m": list(start_tcp),
        "p_button_base_m": list(parsed.p_button_base_m),
        "forward_target_tcp_base_m": list(target_tcp),
        "forward_sample_tcp_base_m": sample_tcp,
        "forward_direction_base": list(forward_direction),
        "normal_out_base": list(normal),
        "press_overtravel_m": PRESS_OVERTRAVEL_M,
        "requested_forward_distance_m": commanded_forward_distance,
        "commanded_forward_distance_m": commanded_forward_distance,
        "approach_distance_m": commanded_forward_distance,
        "current_panel_gap_m": current_panel_gap,
        "minimum_current_panel_gap_m": MIN_CURRENT_PANEL_GAP_M,
        "maximum_current_panel_gap_m": MAX_CURRENT_PANEL_GAP_M,
        "approach_geometry_tolerance_m": APPROACH_GEOMETRY_TOLERANCE_M,
        "endpoint_projection_m": commanded_along,
        "endpoint_projection_error_m": endpoint_projection_error,
        "endpoint_lateral_residual_m": lateral,
        "endpoint_target_residual_m": endpoint_target_residual,
        "per_command_caps_rad": list(caps),
        "runtime_source_waypoint_indices": list(runtime_source_indices),
        "runtime_quantization_step_rad": encoder_step,
        "runtime_first_segment_cap_buffer_rad": first_segment_buffer,
        "runtime_subsequent_segment_cap_buffer_rad": subsequent_segment_buffer,
        "runtime_max_buffered_cap_utilization": runtime_cap_utilization,
        "runtime_geometry_sample_count": RUNTIME_GEOMETRY_SAMPLE_COUNT,
        "no_contact": False,
        "contact_authorized": True,
        "position_contact_approach": True,
        "positional_contact_capsules": list(POSITIONAL_CONTACT_CAPSULES),
        "positional_contact_depth_limit_m": positional_contact_depth_limit,
        "press_authorized": press_authorized,
        "press": False,
        "force_control": False,
        "zero_dwell": True,
        "button_activation_claimed": False,
        "retract": False,
        "automatic_retry": False,
        "automatic_compensation": False,
        "forward_once_only": True,
        "command_count": len(forward_waypoints) + len(return_waypoints),
        "lease_purpose": LEASE_PURPOSE,
        "continuous_exact_advance_required": True,
        "return_endpoint_exact_frozen_start_q": True,
        "zero_resource_prepare": True,
        "confirmation_token": REQUIRED_CONFIRMATION_TOKEN,
        "issued_monotonic": _finite(now_monotonic, "issued_monotonic"),
        "plan_ttl_s": _finite(_policy(model, "stage_b_plan_ttl_s", DEFAULT_PLAN_TTL_S), "stage_b_plan_ttl_s"),
    }
    if payload["plan_ttl_s"] <= 0.0:
        raise StageBBlocked("PLAN_TTL_INVALID")
    payload["expires_monotonic"] = payload["issued_monotonic"] + payload["plan_ttl_s"]
    return FrozenStageBPlan(payload, canonical_hash(payload), parsed)


def verify_stage_b_plan(
    plan_json: str,
    plan_sha256: str,
    *,
    expected_receipt: PreTerminalReceipt | Mapping[str, Any],
    model: Any,
    current_q_rad: Sequence[float],
    now_ns: int | None = None,
    now_monotonic: float | None = None,
    expected_press_authorized: bool | None = None,
) -> Mapping[str, Any]:
    """Recheck immutable plan identity and fresh PRE posture before execution."""

    try:
        value = json.loads(plan_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StageBBlocked("PLAN_INVALIDATED", "JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise StageBBlocked("PLAN_INVALIDATED", "schema")
    if canonical_json(value) != plan_json or canonical_hash(value) != plan_sha256:
        raise StageBBlocked("PLAN_INVALIDATED", "hash")
    receipt = expected_receipt if isinstance(expected_receipt, PreTerminalReceipt) else PreTerminalReceipt.from_mapping(expected_receipt)
    receipt.validate()
    if (
        value.get("panel_press_receipt_id") != receipt.receipt_id
        or value.get("panel_press_receipt_sha256") != receipt.receipt_sha256
        or value.get("panel_geometry_sha256") != receipt.panel_geometry_sha256
        or value.get("wrist_acquisition_id") != receipt.wrist_acquisition_id
        or value.get("wrist_lock_identity") != list(receipt.wrist_lock_identity)
        or value.get("receipt_schema") != receipt.schema
        or value.get("gemini_p_button_base_m")
        != list(receipt.gemini_p_button_base_m)
        or value.get("gemini_wall_surface_point_base_m")
        != list(receipt.gemini_wall_surface_point_base_m)
        or value.get("wrist_tangent_delta_base_m")
        != list(receipt.wrist_tangent_delta_base_m)
        or value.get("wrist_tangent_calibration_sha256")
        != receipt.wrist_tangent_calibration_sha256
        or value.get("wrist_image_error_px") != list(receipt.wrist_image_error_px)
        or value.get("wrist_correction_applied") is not receipt.wrist_correction_applied
    ):
        raise StageBBlocked("PLAN_IDENTITY_MISMATCH", "panel PRESS receipt")
    hashes = _model_hashes(model)
    if value.get("model_hashes") != dict(hashes):
        raise StageBBlocked("MODEL_IDENTITY_MISMATCH")
    exact_flags = {
        "operation": "CONTACT_APPROACH_OUT_BACK",
        "no_contact": False,
        "contact_authorized": True,
        "position_contact_approach": True,
        "positional_contact_capsules": list(POSITIONAL_CONTACT_CAPSULES),
        "press": False,
        "force_control": False,
        "zero_dwell": True,
        "button_activation_claimed": False,
        "retract": False,
        "automatic_retry": False,
        "automatic_compensation": False,
        "forward_once_only": True,
        "lease_purpose": LEASE_PURPOSE,
        "continuous_exact_advance_required": True,
        "return_endpoint_exact_frozen_start_q": True,
        "zero_resource_prepare": True,
        "confirmation_token": REQUIRED_CONFIRMATION_TOKEN,
    }
    for key, expected in exact_flags.items():
        if value.get(key) != expected:
            raise StageBBlocked("CONTACT_CONTRACT_REJECTED", key)
    if type(value.get("press_authorized")) is not bool or (
        expected_press_authorized is not None
        and value.get("press_authorized") is not expected_press_authorized
    ):
        raise StageBBlocked("CONTACT_CONTRACT_REJECTED", "press_authorized")
    current_panel_gap = _finite(value.get("current_panel_gap_m"), "plan.current_panel_gap_m")
    press_overtravel = _finite(value.get("press_overtravel_m"), "plan.press_overtravel_m")
    commanded_forward_distance = current_panel_gap + PRESS_OVERTRAVEL_M
    if (
        current_panel_gap < MIN_CURRENT_PANEL_GAP_M
        or commanded_forward_distance > MAX_CURRENT_PANEL_GAP_M
        or abs(press_overtravel - PRESS_OVERTRAVEL_M) > 1e-12
        or value.get("minimum_current_panel_gap_m") != MIN_CURRENT_PANEL_GAP_M
        or value.get("maximum_current_panel_gap_m") != MAX_CURRENT_PANEL_GAP_M
    ):
        raise StageBBlocked("CURRENT_PANEL_GAP_INVALID")
    for field in ("requested_forward_distance_m", "commanded_forward_distance_m", "approach_distance_m"):
        if abs(_finite(value.get(field), f"plan.{field}") - commanded_forward_distance) > 1e-12:
            raise StageBBlocked("STROKE_INVALID", field)
    if value.get("approach_geometry_tolerance_m") != APPROACH_GEOMETRY_TOLERANCE_M:
        raise StageBBlocked("STROKE_INVALID")
    issued = _finite(value.get("issued_monotonic"), "plan.issued_monotonic")
    expires = _finite(value.get("expires_monotonic"), "plan.expires_monotonic")
    ttl = _finite(value.get("plan_ttl_s"), "plan.plan_ttl_s")
    if ttl <= 0.0 or expires <= issued or abs((expires - issued) - ttl) > 1e-9:
        raise StageBBlocked("PLAN_TTL_INVALID")
    if now_monotonic is not None and (not math.isfinite(now_monotonic) or now_monotonic < issued or now_monotonic > expires):
        raise StageBBlocked("PLAN_EXPIRED")
    q0 = _vector(value.get("start_q_rad"), 6, "plan.start_q_rad")
    if any(abs(a - b) > 1e-12 for a, b in zip(q0, receipt.terminal_q_rad)):
        raise StageBBlocked("PLAN_IDENTITY_MISMATCH", "PRE q")
    current = _vector(current_q_rad, 6, "current_q_rad")
    tolerance = float(_policy(model, "q0_drift_limit_rad", DEFAULT_Q_DRIFT_TOLERANCE_RAD))
    if max(abs(a - b) for a, b in zip(current, q0)) > tolerance:
        raise StageBBlocked("START_Q_DRIFT")
    if now_ns is not None:
        max_age_ns = int(float(_policy(model, "maximum_q0_age_s", 1.0)) * 1e9)
        if now_ns < receipt.feedback_stamp_ns or now_ns - receipt.feedback_stamp_ns > max_age_ns:
            raise StageBBlocked("ARRIVAL_RECEIPT_STALE")
    forward = tuple(_vector(point, 6, "plan.forward_waypoint") for point in value.get("forward_waypoints_rad", ()))
    back = tuple(_vector(point, 6, "plan.return_waypoint") for point in value.get("return_waypoints_rad", ()))
    offline_forward = tuple(
        _vector(point, 6, "plan.offline_forward_validation_waypoint")
        for point in value.get("offline_forward_validation_waypoints_rad", ())
    )
    expected_back = tuple(reversed(forward[:-1])) + (q0,)
    if (
        not forward
        or not offline_forward
        or back != expected_back
        or _vector(value.get("forward_endpoint_q_rad"), 6, "plan.forward_endpoint_q_rad") != forward[-1]
        or value.get("command_count") != len(forward) + len(back)
        or value.get("locked_joint_indices") != list(LOCKED_JOINT_INDICES)
    ):
        raise StageBBlocked("PLAN_INVALIDATED", "waypoint shape")
    caps = _vector(value.get("per_command_caps_rad"), 6, "plan.per_command_caps_rad")
    if caps != _joint_caps(model):
        raise StageBBlocked("PLAN_INVALIDATED", "per-command caps")
    for index, point in enumerate(forward + back):
        _validate_limits(model, point, field_name=f"plan waypoint {index}")
        _validate_fixed_joints(point, q0, field_name=f"plan waypoint {index}")
    offline_back = tuple(reversed(offline_forward[:-1])) + (q0,)
    for index, point in enumerate(offline_forward + offline_back):
        _validate_limits(model, point, field_name=f"offline validation waypoint {index}")
        _validate_fixed_joints(point, q0, field_name=f"offline validation waypoint {index}")
    _validate_adjacent_caps(
        q0,
        offline_forward,
        caps,
        detail_prefix="offline forward",
    )
    _validate_adjacent_caps(
        offline_forward[-1],
        offline_back,
        caps,
        detail_prefix="offline return",
    )
    normal = _unit(value.get("normal_out_base"), "plan.normal_out_base")
    direction = _unit(value.get("forward_direction_base"), "plan.forward_direction_base")
    if max(abs(direction[index] + normal[index]) for index in range(3)) > 1e-12:
        raise StageBBlocked("NORMAL_DIRECTION_INVALID", "forward is not -normal_out_base")
    p_button = _vector(value.get("p_button_base_m"), 3, "plan.p_button_base_m")
    if p_button != receipt.p_button_base_m:
        raise StageBBlocked("PLAN_IDENTITY_MISMATCH", "contact geometry")
    start_tcp = _fk_tcp(model, q0, "verified_start_tcp_base_m")
    if _vector(value.get("start_tcp_base_m"), 3, "plan.start_tcp_base_m") != start_tcp:
        raise StageBBlocked("PLAN_INVALIDATED", "actual start TCP")
    recomputed_gap = _dot(normal, _sub(start_tcp, p_button))
    if (
        not math.isfinite(recomputed_gap)
        or recomputed_gap < MIN_CURRENT_PANEL_GAP_M
        or recomputed_gap > MAX_CURRENT_PANEL_GAP_M
        or abs(recomputed_gap - current_panel_gap) > 1e-12
    ):
        raise StageBBlocked("CURRENT_PANEL_GAP_INVALID")
    recomputed_commanded_distance = recomputed_gap + PRESS_OVERTRAVEL_M
    if recomputed_commanded_distance > MAX_CURRENT_PANEL_GAP_M:
        raise StageBBlocked("CURRENT_PANEL_GAP_TOO_LARGE", "verified commanded stroke")
    to_button = _sub(p_button, start_tcp)
    tangent_delta = _sub(to_button, _scale(normal, _dot(normal, to_button)))
    target_tcp = _add(p_button, _scale(direction, PRESS_OVERTRAVEL_M))
    if _vector(value.get("forward_target_tcp_base_m"), 3, "forward target") != target_tcp:
        raise StageBBlocked("PLAN_INVALIDATED", "endpoint is not measured gap plus overtravel")
    if (
        abs(_dot(normal, _sub(target_tcp, p_button)) + PRESS_OVERTRAVEL_M)
        > APPROACH_GEOMETRY_TOLERANCE_M
    ):
        raise StageBBlocked("PLAN_INVALIDATED", "endpoint is not fixed overtravel beyond button plane")
    residual_limit = _finite(
        _policy(model, "position_residual_limit_m", DEFAULT_POSITION_RESIDUAL_M),
        "position_residual_limit_m",
    )
    if residual_limit <= 0.0:
        raise StageBBlocked("POLICY_INVALID", "position_residual_limit_m")
    encoder_step = _finite(
        _policy(model, "encoder_step_rad", DEFAULT_ENCODER_STEP_RAD),
        "encoder_step_rad",
    )
    waypoint_tolerance = _finite(
        _policy(model, "move_waypoint_tolerance_rad", DEFAULT_WAYPOINT_TOLERANCE_RAD),
        "move_waypoint_tolerance_rad",
    )
    q0_drift_limit = _finite(
        _policy(model, "q0_drift_limit_rad", DEFAULT_Q_DRIFT_TOLERANCE_RAD),
        "q0_drift_limit_rad",
    )
    first_segment_buffer = q0_drift_limit + waypoint_tolerance + encoder_step
    subsequent_segment_buffer = waypoint_tolerance + encoder_step
    if (
        encoder_step <= 0.0
        or waypoint_tolerance <= 0.0
        or q0_drift_limit < 0.0
        or _finite(value.get("runtime_quantization_step_rad"), "plan.runtime_quantization_step_rad") != encoder_step
        or _finite(value.get("runtime_first_segment_cap_buffer_rad"), "plan.runtime_first_segment_cap_buffer_rad") != first_segment_buffer
        or _finite(value.get("runtime_subsequent_segment_cap_buffer_rad"), "plan.runtime_subsequent_segment_cap_buffer_rad") != subsequent_segment_buffer
        or value.get("runtime_geometry_sample_count") != RUNTIME_GEOMETRY_SAMPLE_COUNT
    ):
        raise StageBBlocked("PLAN_INVALIDATED", "runtime policy")

    quantized_dense = tuple(
        _quantize_runtime_waypoint(point, q0, encoder_step)
        for point in offline_forward
    )
    quantized_endpoint = quantized_dense[-1]
    positional_contact_depth_limit = _finite(
        value.get("positional_contact_depth_limit_m"),
        "plan.positional_contact_depth_limit_m",
    )
    maximum_contact_depth = PRESS_OVERTRAVEL_M + residual_limit
    if (
        positional_contact_depth_limit < 0.0
        or positional_contact_depth_limit > maximum_contact_depth
    ):
        raise StageBBlocked("PLAN_INVALIDATED", "contact depth limit")
    endpoint_contact_depth = _contact_depth_from_endpoint(
        model,
        quantized_endpoint,
        wall_surface_point_base_m=receipt.wall_surface_point_base_m,
        normal_out_base=normal,
        positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
    )
    if abs(positional_contact_depth_limit - endpoint_contact_depth) > 1e-12:
        raise StageBBlocked("PLAN_INVALIDATED", "endpoint contact depth")
    endpoint_tcp = _fk_tcp(model, quantized_endpoint, "verified forward endpoint TCP")
    endpoint_delta = _sub(endpoint_tcp, start_tcp)
    endpoint_projection = _dot(endpoint_delta, direction)
    endpoint_lateral = _norm(
        _sub(
            _sub(endpoint_delta, _scale(direction, endpoint_projection)),
            tangent_delta,
        )
    )
    endpoint_projection_error = abs(endpoint_projection - recomputed_commanded_distance)
    endpoint_target_residual = _norm(_sub(endpoint_tcp, target_tcp))
    endpoint_metrics = {
        "endpoint_projection_m": endpoint_projection,
        "endpoint_projection_error_m": endpoint_projection_error,
        "endpoint_lateral_residual_m": endpoint_lateral,
        "endpoint_target_residual_m": endpoint_target_residual,
    }
    if any(
        metric > residual_limit
        for metric in (endpoint_projection_error, endpoint_lateral, endpoint_target_residual)
    ):
        raise StageBBlocked("IK_RESIDUAL_EXCEEDED", "quantized endpoint")
    for field, expected in endpoint_metrics.items():
        if abs(_finite(value.get(field), f"plan.{field}") - expected) > 1e-12:
            raise StageBBlocked("PLAN_INVALIDATED", field)

    previous_projection = -math.inf
    for index, point in enumerate(offline_forward):
        tcp = _fk_tcp(model, point, f"verified_offline_forward_tcp[{index}]")
        delta = _sub(tcp, start_tcp)
        projection = _dot(delta, direction)
        expected_tangent = _scale(
            tangent_delta,
            max(0.0, min(1.0, projection / recomputed_commanded_distance)),
        )
        lateral = _norm(
            _sub(
                _sub(delta, _scale(direction, projection)),
                expected_tangent,
            )
        )
        if (
            projection + APPROACH_GEOMETRY_TOLERANCE_M < previous_projection
            or projection < -APPROACH_GEOMETRY_TOLERANCE_M
            or projection > recomputed_commanded_distance + residual_limit
            or lateral > residual_limit
        ):
            raise StageBBlocked("APPROACH_PATH_NON_MONOTONIC", f"offline {index}")
        previous_projection = projection

    offline_contact_depth = _contact_depth_from_endpoint(
        model,
        offline_forward[-1],
        wall_surface_point_base_m=receipt.wall_surface_point_base_m,
        normal_out_base=normal,
        positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES,
    )
    if offline_contact_depth > maximum_contact_depth:
        raise StageBBlocked("CONTACT_DEPTH_LIMIT_EXCEEDED", "verified offline endpoint")
    validate_model_path(model, q0, offline_forward, wall_surface_point_base_m=receipt.wall_surface_point_base_m, normal_out_base=normal, positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES, positional_contact_depth_limit_m=offline_contact_depth)
    validate_model_path(model, offline_forward[-1], offline_back, wall_surface_point_base_m=receipt.wall_surface_point_base_m, normal_out_base=normal, positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES, positional_contact_depth_limit_m=offline_contact_depth)

    def validate_runtime_candidate(
        candidate_forward: tuple[tuple[float, ...], ...],
        candidate_back: tuple[tuple[float, ...], ...],
    ) -> None:
        for index, point in enumerate(candidate_forward + candidate_back):
            _validate_limits(model, point, field_name=f"verified runtime waypoint {index}")
            _validate_fixed_joints(point, q0, field_name=f"verified runtime waypoint {index}")
        validate_model_path(model, q0, candidate_forward, wall_surface_point_base_m=receipt.wall_surface_point_base_m, normal_out_base=normal, positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES, positional_contact_depth_limit_m=positional_contact_depth_limit)
        validate_model_path(model, candidate_forward[-1], candidate_back, wall_surface_point_base_m=receipt.wall_surface_point_base_m, normal_out_base=normal, positional_contact_capsules=POSITIONAL_CONTACT_CAPSULES, positional_contact_depth_limit_m=positional_contact_depth_limit)
        _audit_runtime_normal_path(model, q0, candidate_forward, reference_tcp_base_m=start_tcp, forward_direction_base=direction, current_panel_gap_m=recomputed_commanded_distance, residual_limit_m=residual_limit, retreat=False, tangent_offset_base_m=tangent_delta)
        _audit_runtime_normal_path(model, candidate_forward[-1], candidate_back, reference_tcp_base_m=start_tcp, forward_direction_base=direction, current_panel_gap_m=recomputed_commanded_distance, residual_limit_m=residual_limit, retreat=True, tangent_offset_base_m=tangent_delta)

    expected_forward, expected_return, expected_indices, expected_utilization = _select_runtime_waypoints(
        quantized_dense,
        q0,
        caps,
        first_segment_buffer_rad=first_segment_buffer,
        subsequent_segment_buffer_rad=subsequent_segment_buffer,
        minimum_command_delta_rad=encoder_step,
        validate_candidate=validate_runtime_candidate,
    )
    raw_indices = value.get("runtime_source_waypoint_indices")
    if (
        forward != expected_forward
        or back != expected_return
        or _vector(value.get("forward_endpoint_q_rad"), 6, "plan.forward_endpoint_q_rad") != quantized_endpoint
        or not isinstance(raw_indices, list)
        or any(type(index) is not int for index in raw_indices)
        or tuple(raw_indices) != expected_indices
        or abs(_finite(value.get("runtime_max_buffered_cap_utilization"), "plan.runtime_max_buffered_cap_utilization") - expected_utilization) > 1e-15
        or value.get("command_count") != len(expected_forward) + len(expected_return)
    ):
        raise StageBBlocked("PLAN_INVALIDATED", "runtime waypoint selection")
    recomputed_samples = [
        list(_fk_tcp(model, point, f"verified_forward_tcp[{index}]"))
        for index, point in enumerate(expected_forward)
    ]
    if value.get("forward_sample_tcp_base_m") != recomputed_samples:
        raise StageBBlocked("PLAN_INVALIDATED", "sample TCP audit")
    return value


@dataclass(frozen=True)
class StageBGoal:
    """Typed action-goal equivalent; no arbitrary q, geometry, or JSON."""

    plan_sha256: str
    stage_a_receipt_sha256: str
    authorization_id: str
    confirmation_token: str


@dataclass(frozen=True)
class StageBExecutionConfig:
    """Pure runtime thresholds; ROS/build/contact gates remain node-owned."""

    max_feedback_age_s: float = 0.5
    endpoint_tolerance_rad: float = math.radians(0.5)
    command_ack_timeout_s: float = 0.75
    waypoint_timeout_s: float = 12.0
    min_voltage_v: float = 11.0
    max_voltage_v: float = 13.0
    encoder_step_rad: float = DEFAULT_ENCODER_STEP_RAD

    def validate(self) -> None:
        values = (
            self.max_feedback_age_s,
            self.endpoint_tolerance_rad,
            self.command_ack_timeout_s,
            self.waypoint_timeout_s,
            self.min_voltage_v,
            self.max_voltage_v,
            self.encoder_step_rad,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "non-finite")
        if not 0.0 < self.max_feedback_age_s <= 1.0:
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "feedback age")
        if not 0.0 < self.endpoint_tolerance_rad <= math.radians(1.0):
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "endpoint tolerance")
        if not 0.0 < self.command_ack_timeout_s <= 2.0:
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "command ACK timeout")
        if not 0.1 <= self.waypoint_timeout_s <= 30.0:
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "waypoint timeout")
        if not 0.0 < self.min_voltage_v < self.max_voltage_v <= 30.0:
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "voltage")
        if not 0.0 < self.encoder_step_rad <= 0.01:
            raise StageBBlocked("EXECUTION_POLICY_INVALID", "encoder step")


@dataclass(frozen=True)
class StageBSafetySample:
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
    accepted_command_sequence: int = 0
    accepted_target_positions: tuple[float, ...] = ()
    dynamic_exact_advance_ready: bool = False
    status_received_monotonic: float = -math.inf

    def validate_owner(
        self,
        config: StageBExecutionConfig,
        *,
        expected_publishers: int,
        expected_lease: bool,
        require_disarmed: bool,
    ) -> None:
        config.validate()
        _vector(self.positions_rad, 6, "feedback.positions_rad")
        if (
            not math.isfinite(self.feedback_age_s)
            or not 0.0 <= self.feedback_age_s <= config.max_feedback_age_s
            or not math.isfinite(self.feedback_stamp_age_s)
            or not 0.0 <= self.feedback_stamp_age_s <= config.max_feedback_age_s
            or not math.isfinite(self.feedback_received_monotonic)
        ):
            raise StageBBlocked("MOVE_FEEDBACK_STALE")
        if self.dry_run or self.fault:
            raise StageBBlocked("MOVE_SAFETY_GATE_FAILED", "driver dry-run/fault")
        if len(self.torque_enabled) != 6 or not all(type(value) is bool and value for value in self.torque_enabled):
            raise StageBBlocked("MOVE_SAFETY_GATE_FAILED", "torque6/6")
        if not math.isfinite(self.voltage_v) or not config.min_voltage_v <= self.voltage_v <= config.max_voltage_v:
            raise StageBBlocked("MOVE_SAFETY_GATE_FAILED", "voltage")
        if self.joint_target_publishers != expected_publishers:
            raise StageBBlocked("MOVE_OWNER_DRIFT", "joint_target publisher count")
        if (
            not math.isfinite(self.lease_status_age_s)
            or not 0.0 <= self.lease_status_age_s <= config.max_feedback_age_s
            or self.lease_active is not expected_lease
        ):
            raise StageBBlocked("MOVE_OWNER_DRIFT", "lease state")
        if expected_lease and (
            self.lease_purpose != LEASE_PURPOSE or not self.dynamic_exact_no_reissue
        ):
            raise StageBBlocked("MOVE_OWNER_DRIFT", "exact no-reissue lease")
        if require_disarmed and (self.state not in SAFE_IDLE_EXECUTION_STATES or self.armed):
            raise StageBBlocked(
                "MOVE_SAFETY_GATE_FAILED",
                "CONVERGED_DISARMED or DISARMED:operator required",
            )


def validate_command_acceptance(
    sample: StageBSafetySample,
    *,
    prior_sequence: int,
    requested_target: Sequence[float],
    publish_boundary_positions: Sequence[float],
    encoder_step_rad: float,
) -> tuple[float, ...]:
    """Bind one fresh SafeDriver acceptance event to one frozen command."""

    if sample.accepted_command_sequence != int(prior_sequence) + 1:
        raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", "accepted sequence")
    requested = _vector(requested_target, 6, "requested_target")
    boundary = _vector(publish_boundary_positions, 6, "publish_boundary_positions")
    accepted = _vector(sample.accepted_target_positions, 6, "accepted_target")
    step = _finite(encoder_step_rad, "encoder_step_rad")
    if step <= 0.0:
        raise StageBBlocked("EXECUTION_POLICY_INVALID", "encoder step")
    for index, (value, measured, returned, limits) in enumerate(
        zip(requested, boundary, accepted, SAFE_DRIVER_COMMAND_LIMITS_RAD)
    ):
        lower, upper = limits
        if not lower <= returned <= upper:
            raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", "accepted target limit")
        if abs(value - measured) > step + 1e-12:
            effective = round(value / step) * step
            expected = min(upper, max(lower, effective))
            if abs(returned - expected) > 1e-12:
                raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", f"accepted mover target {index}")
        elif abs(returned - measured) > step + 1e-12:
            # Feedback may advance by one encoder step between our publish
            # boundary sample and SafeDriver's receive boundary.  Bind the
            # first typed ACK value, then require it to remain byte-stable.
            raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", f"accepted held target {index}")
    return accepted


class StageBExactExecution:
    """Frozen N-waypoint approach followed by its exact mirrored return."""

    def __init__(
        self,
        plan: FrozenStageBPlan | Mapping[str, Any],
        config: StageBExecutionConfig = StageBExecutionConfig(),
        *,
        now_monotonic: float,
    ) -> None:
        self.plan = plan.plan if isinstance(plan, FrozenStageBPlan) else _mapping(plan, "plan")
        self.config = config
        self.config.validate()
        forward = tuple(
            _vector(point, 6, "forward waypoint")
            for point in self.plan.get("forward_waypoints_rad", ())
        )
        backward = tuple(
            _vector(point, 6, "return waypoint")
            for point in self.plan.get("return_waypoints_rad", ())
        )
        q0 = _vector(self.plan.get("start_q_rad"), 6, "start_q_rad")
        mirrored = tuple(reversed(forward[:-1])) + (q0,)
        if (
            not forward
            or backward != mirrored
            or self.plan.get("command_count") != len(forward) + len(backward)
            or self.plan.get("forward_once_only") is not True
            or self.plan.get("continuous_exact_advance_required") is not True
            or self.plan.get("lease_purpose") != LEASE_PURPOSE
            or self.plan.get("operation") != "CONTACT_APPROACH_OUT_BACK"
            or self.plan.get("no_contact") is not False
            or self.plan.get("contact_authorized") is not True
            or self.plan.get("force_control") is not False
            or self.plan.get("zero_dwell") is not True
            or self.plan.get("button_activation_claimed") is not False
            or any(self.plan.get(key) is not False for key in ("press", "retract", "automatic_retry", "automatic_compensation"))
        ):
            raise StageBBlocked("PLAN_INVALIDATED", "contact approach contract")
        now = _finite(now_monotonic, "now_monotonic")
        self.forward_waypoints = forward
        self.return_waypoints = backward
        self.start_q_rad = q0
        first_segment_buffer = _finite(
            self.plan.get("runtime_first_segment_cap_buffer_rad"),
            "runtime_first_segment_cap_buffer_rad",
        )
        subsequent_segment_buffer = _finite(
            self.plan.get("runtime_subsequent_segment_cap_buffer_rad"),
            "runtime_subsequent_segment_cap_buffer_rad",
        )
        self.q0_drift_limit_rad = first_segment_buffer - subsequent_segment_buffer
        if self.q0_drift_limit_rad < 0.0:
            raise StageBBlocked("PLAN_INVALIDATED", "q0 drift buffer")
        self.commands = tuple(("FORWARD", point) for point in forward) + tuple(
            ("RETURN", point) for point in backward
        )
        self.command_index = 0
        self.state = "PRECHECK"
        self.terminal_code = ""
        self.terminal_reason = ""
        self.endpoint_error_rad = 0.0
        self.terminal_feedback_age_s = math.inf
        self.publish_log: list[tuple[str, tuple[float, ...]]] = []
        self.forward_endpoint_converged_monotonic = math.inf
        self.first_return_published_monotonic = math.inf
        self._last_now = now
        self._published_at = -math.inf
        self._deadline = math.inf
        self._prior_sequence = -1
        self._accepted_sequence = -1
        self._publish_boundary_positions: tuple[float, ...] = ()
        self._accepted_target: tuple[float, ...] = ()

    @property
    def terminal(self) -> bool:
        return self.state in {"SUCCEEDED", "FAILED", "CANCELLED"}

    @property
    def target(self) -> tuple[float, ...]:
        if self.state not in {"PUBLISH_WAYPOINT", "WAIT_COMMAND_ACK", "WAIT_WAYPOINT_CONVERGENCE"}:
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", f"no target in {self.state}")
        return self.commands[self.command_index][1]

    @property
    def phase(self) -> str:
        return self.commands[self.command_index][0]

    def precheck(self, sample: StageBSafetySample) -> None:
        if self.state != "PRECHECK":
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", self.state)
        sample.validate_owner(
            self.config, expected_publishers=0, expected_lease=False, require_disarmed=True
        )
        self.state = "ACQUIRE_LEASE"

    def lease_acquired(self, sample: StageBSafetySample) -> None:
        if self.state != "ACQUIRE_LEASE":
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", self.state)
        sample.validate_owner(
            self.config, expected_publishers=0, expected_lease=True, require_disarmed=True
        )
        self.state = "CREATE_COMMAND_RESOURCES"

    def publisher_created(self, sample: StageBSafetySample) -> None:
        if self.state != "CREATE_COMMAND_RESOURCES":
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", self.state)
        sample.validate_owner(
            self.config, expected_publishers=1, expected_lease=True, require_disarmed=True
        )
        self.state = "ARM_APPROACH"

    def armed(self, success: bool, reason: str = "") -> None:
        if self.state != "ARM_APPROACH":
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", self.state)
        if not success:
            self.fail("MOVE_ARM_REJECTED", reason)
            return
        self.state = "PUBLISH_WAYPOINT"

    def validate_publish_boundary(
        self,
        sample: StageBSafetySample,
        *,
        first_command: bool,
    ) -> None:
        """Prove the fresh physical/controller boundary for one unique publish."""

        if first_command:
            if self.state != "PUBLISH_WAYPOINT" or self.command_index != 0 or self.publish_log:
                raise StageBBlocked("MOVE_PROTOCOL_ERROR", "first publish boundary")
            sample.validate_owner(
                self.config,
                expected_publishers=1,
                expected_lease=True,
                require_disarmed=False,
            )
            if sample.state != "ARMED" or not sample.armed:
                raise StageBBlocked("MOVE_SAFETY_GATE_FAILED", "fresh post-arm ARMED status required")
            start_error = max(
                abs(value - expected)
                for value, expected in zip(sample.positions_rad, self.start_q_rad)
            )
            if start_error > self.q0_drift_limit_rad:
                raise StageBBlocked("START_Q_DRIFT", f"publish boundary {start_error:.12g}")
            return

        if self.state != "PUBLISH_WAYPOINT" or self.command_index <= 0:
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", "advance publish boundary")
        sample.validate_owner(
            self.config,
            expected_publishers=1,
            expected_lease=True,
            require_disarmed=False,
        )
        if sample.state != "COMMAND_ACTIVE" or not sample.armed or not sample.dynamic_exact_advance_ready:
            raise StageBBlocked("MOVE_OWNER_DRIFT", "continuous exact advance boundary")
        if (
            self._accepted_sequence < 0
            or not self._accepted_target
            or sample.accepted_command_sequence != self._accepted_sequence
        ):
            raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", "accepted sequence changed")
        try:
            accepted = _vector(sample.accepted_target_positions, 6, "accepted target")
        except StageBBlocked as exc:
            raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", "accepted target malformed") from exc
        if accepted != self._accepted_target:
            raise StageBBlocked("MOVE_COMMAND_ACK_INVALID", "accepted target changed")
        endpoint_error = max(
            abs(value - expected)
            for value, expected in zip(sample.positions_rad, accepted)
        )
        if endpoint_error > self.config.endpoint_tolerance_rad:
            raise StageBBlocked("MOVE_ENDPOINT_ERROR", f"publish boundary {endpoint_error:.12g}")

    def mark_published(
        self,
        phase: str,
        target: Sequence[float],
        *,
        prior_sequence: int,
        publish_boundary_positions: Sequence[float],
        now_monotonic: float,
    ) -> None:
        if self.state != "PUBLISH_WAYPOINT" or phase != self.phase:
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", f"{self.state}/{phase}")
        exact = _vector(target, 6, "published target")
        if exact != self.target:
            raise StageBBlocked("PLAN_INVALIDATED", "published target")
        if len(self.publish_log) != self.command_index:
            raise StageBBlocked("MOVE_REISSUE_BLOCKED", str(self.command_index))
        if phase == "RETURN" and self.command_index < len(self.forward_waypoints):
            raise StageBBlocked("MOVE_BLIND_RETURN_BLOCKED")
        now = self._advance_clock(now_monotonic)
        self.publish_log.append((phase, exact))
        if phase == "RETURN" and not math.isfinite(self.first_return_published_monotonic):
            self.first_return_published_monotonic = now
        self._published_at = now
        self._prior_sequence = int(prior_sequence)
        self._accepted_sequence = -1
        self._publish_boundary_positions = _vector(
            publish_boundary_positions, 6, "publish boundary positions"
        )
        self._accepted_target = ()
        self._deadline = now + self.config.command_ack_timeout_s
        self.state = "WAIT_COMMAND_ACK"

    def accept_command(self, sample: StageBSafetySample, *, now_monotonic: float) -> None:
        if self.state != "WAIT_COMMAND_ACK":
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", self.state)
        now = self._advance_clock(now_monotonic)
        if now > self._deadline:
            self.fail("MOVE_COMMAND_ACK_MISSING", "bounded acceptance timeout", sample=sample)
            return
        self._validate_motion_sample(sample)
        if (
            sample.status_received_monotonic <= self._published_at
            or sample.state != "COMMAND_ACTIVE"
            or not sample.armed
        ):
            raise StageBBlocked("MOVE_COMMAND_ACK_MISSING", "fresh COMMAND_ACTIVE status required")
        expected_accepted = validate_command_acceptance(
            sample,
            prior_sequence=self._prior_sequence,
            requested_target=self.target,
            publish_boundary_positions=self._publish_boundary_positions,
            encoder_step_rad=self.config.encoder_step_rad,
        )
        self._accepted_sequence = sample.accepted_command_sequence
        self._accepted_target = expected_accepted
        self._deadline = now + self.config.waypoint_timeout_s
        self.state = "WAIT_WAYPOINT_CONVERGENCE"

    def observe(self, sample: StageBSafetySample, *, now_monotonic: float) -> None:
        if self.state != "WAIT_WAYPOINT_CONVERGENCE":
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", self.state)
        now = self._advance_clock(now_monotonic)
        if now > self._deadline:
            self.fail("MOVE_WAYPOINT_TIMEOUT", "no retry or blind return permitted", sample=sample)
            return
        self._validate_motion_sample(sample)
        if sample.accepted_command_sequence != self._accepted_sequence:
            self.fail("MOVE_COMMAND_ACK_INVALID", "accepted sequence changed", sample=sample)
            return
        try:
            accepted = _vector(sample.accepted_target_positions, 6, "accepted target")
        except StageBBlocked as exc:
            self.fail(exc.code, exc.detail, sample=sample)
            return
        if accepted != self._accepted_target:
            self.fail("MOVE_COMMAND_ACK_INVALID", "accepted target changed", sample=sample)
            return
        error = max(abs(a - b) for a, b in zip(sample.positions_rad, self._accepted_target))
        final_command = self.command_index + 1 == len(self.commands)
        if not final_command:
            if sample.state != "COMMAND_ACTIVE" or not sample.armed:
                self.fail("MOVE_WAYPOINT_TERMINATED_BEFORE_ADVANCE", "no blind return", sample=sample)
                return
            if not sample.dynamic_exact_advance_ready:
                return
            if error > self.config.endpoint_tolerance_rad:
                self.fail("MOVE_ENDPOINT_ERROR", f"{error:.12g}", sample=sample)
                return
            self.endpoint_error_rad = max(self.endpoint_error_rad, error)
            if self.command_index + 1 == len(self.forward_waypoints):
                self.forward_endpoint_converged_monotonic = now
            self.command_index += 1
            self.state = "PUBLISH_WAYPOINT"
            return
        if sample.state == "COMMAND_ACTIVE" and sample.armed:
            return
        if sample.state != "CONVERGED_DISARMED" or sample.armed:
            self.fail("MOVE_RETURN_TERMINAL_INVALID", sample=sample)
            return
        if error > self.config.endpoint_tolerance_rad:
            self.fail("MOVE_RETURN_ENDPOINT_ERROR", f"{error:.12g}", sample=sample)
            return
        self.endpoint_error_rad = max(self.endpoint_error_rad, error)
        self.terminal_feedback_age_s = sample.feedback_age_s
        self.state = "TEARDOWN_SUCCESS"

    def cancel(self, reason: str = "operator cancellation requested") -> None:
        if not self.terminal:
            self.state = "TEARDOWN_CANCEL"
            self.terminal_code = "MOVE_CANCELLED"
            self.terminal_reason = reason

    def fail(
        self,
        code: str,
        reason: str = "",
        *,
        sample: StageBSafetySample | None = None,
    ) -> None:
        if self.terminal:
            return
        self.state = "TEARDOWN_FAILURE"
        self.terminal_code = str(code)
        self.terminal_reason = str(reason)
        if sample is not None:
            self.terminal_feedback_age_s = sample.feedback_age_s

    def teardown_complete(self, *, success: bool) -> None:
        if self.state not in {"TEARDOWN_SUCCESS", "TEARDOWN_FAILURE", "TEARDOWN_CANCEL"}:
            raise StageBBlocked("MOVE_PROTOCOL_ERROR", "teardown boundary")
        if success and self.state == "TEARDOWN_SUCCESS":
            self.state = "SUCCEEDED"
            self.terminal_code = "CONTACT_APPROACH_OUT_BACK_COMPLETE_DISARMED"
            self.terminal_reason = "measured current-panel approach and mirrored exact start return converged; resources released"
        elif self.state == "TEARDOWN_CANCEL":
            self.state = "CANCELLED"
        else:
            self.state = "FAILED"
            if not self.terminal_code:
                self.terminal_code = "MOVE_TEARDOWN_FAILED"

    def _advance_clock(self, value: float) -> float:
        now = _finite(value, "now_monotonic")
        if now < self._last_now:
            raise StageBBlocked("MOVE_CLOCK_INVALID")
        self._last_now = now
        return now

    def _validate_motion_sample(self, sample: StageBSafetySample) -> None:
        sample.validate_owner(
            self.config, expected_publishers=1, expected_lease=True, require_disarmed=False
        )
        if sample.state not in {"COMMAND_ACTIVE", "CONVERGED_DISARMED"}:
            raise StageBBlocked("MOVE_SAFETY_GATE_FAILED", "motion state")
        if (sample.state == "COMMAND_ACTIVE") is not sample.armed:
            raise StageBBlocked("MOVE_SAFETY_GATE_FAILED", "armed/state mismatch")


__all__ = [
    "APPROACH_DISTANCE_M", "APPROACH_GEOMETRY_TOLERANCE_M", "FORWARD_DISTANCE_M", "PRESS_OVERTRAVEL_M", "MIN_CURRENT_PANEL_GAP_M", "MAX_CURRENT_PANEL_GAP_M", "LEASE_PURPOSE", "LOCKED_JOINT_INDICES", "POSITIONAL_CONTACT_CAPSULES", "SAFE_IDLE_EXECUTION_STATES", "RECEIPT_SCHEMA", "WRIST_CORRECTED_RECEIPT_SCHEMA", "MAX_WRIST_TANGENT_CORRECTION_M", "REQUIRED_CONFIRMATION_TOKEN", "SCHEMA",
    "StageBBlocked", "StageBState", "CurrentPanelPressReceipt", "PreTerminalReceipt", "StageAPreTerminalReceipt", "FrozenStageBPlan",
    "StageBGoal", "StageBExecutionConfig", "StageBSafetySample", "StageBExactExecution",
    "canonical_json", "canonical_hash", "prepare_stage_b_plan", "verify_stage_b_plan",
    "validate_command_acceptance", "validate_model_path",
]
