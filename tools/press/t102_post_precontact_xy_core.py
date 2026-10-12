"""Pure, no-contact post-PRE wrist tangent-alignment contract.

This module owns no ROS resources.  It converts one immutable wrist PRE-LOCK
measurement into one bounded tangent target, freezes the exact plan, and
models the one-publish/one-terminal fake lifecycle used by QC/executor tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "t102_post_precontact_xy_plan_v1"
PREPARE_CONFIRMATION_TOKEN = "PREPARE_FROZEN_POST_PRECONTACT_XY_NO_MOTION"
CONFIRMATION_TOKEN = "APPLY_ONE_FROZEN_POST_PRECONTACT_XY_NO_CONTACT"
LEASE_PURPOSE = "T102_POST_PRECONTACT_XY_EXACT_PLAN_NO_REISSUE"
MAX_PER_MOVE_TANGENT_CORRECTION_M = 0.015
MAX_CUMULATIVE_TANGENT_CORRECTION_M = 0.015
MAX_TANGENT_CORRECTION_M = MAX_PER_MOVE_TANGENT_CORRECTION_M
ENDPOINT_TOLERANCE_RAD = math.radians(0.5)
LOCKED_JOINT_INDICES = (4, 5)


class AlignmentBlocked(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(self.code if not detail else f"{self.code}: {detail}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _digest(value: object, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise AlignmentBlocked("ALIGNMENT_IDENTITY_INVALID", name)
    return digest


def _finite_vec(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise AlignmentBlocked("ALIGNMENT_INPUT_INVALID", name)
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise AlignmentBlocked("ALIGNMENT_INPUT_INVALID", name)
    return result


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(value: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = _finite_vec(value, 3, name)
    magnitude = _norm(vector)
    if magnitude <= 1e-12:
        raise AlignmentBlocked("ALIGNMENT_INPUT_INVALID", name)
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise AlignmentBlocked("ALIGNMENT_INPUT_INVALID", "R_base_from_wrist_camera")
    rows = tuple(_finite_vec(row, 3, "R_base_from_wrist_camera") for row in matrix)
    return tuple(_dot(row, vector) for row in rows)  # type: ignore[return-value]


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _scale(value: Sequence[float], scale: float) -> tuple[float, ...]:
    return tuple(float(item) * float(scale) for item in value)


def _point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    segment = _sub(end, start)
    denominator = _dot(segment, segment)
    ratio = 0.0 if denominator <= 1e-15 else max(
        0.0, min(1.0, _dot(_sub(point, start), segment) / denominator)
    )
    return _norm(_sub(point, _add(start, _scale(segment, ratio))))


def _segment_distance(a0, a1, b0, b1) -> float:
    # Small deterministic closest-segment calculation used by Stage-A.
    u, v, w = _sub(a1, a0), _sub(b1, b0), _sub(a0, b0)
    a, b, c = _dot(u, u), _dot(u, v), _dot(v, v)
    d, e = _dot(u, w), _dot(v, w)
    epsilon = 1e-15
    if a <= epsilon and c <= epsilon:
        return _norm(w)
    if a <= epsilon:
        return _point_segment_distance(a0, b0, b1)
    if c <= epsilon:
        return _point_segment_distance(b0, a0, a1)
    denominator = a * c - b * b
    s = 0.0 if denominator <= epsilon else max(
        0.0, min(1.0, (b * e - c * d) / denominator)
    )
    t = (b * s + e) / c
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -d / a))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b - d) / a))
    return _norm(_sub(_add(a0, _scale(u, s)), _add(b0, _scale(v, t))))


def model_path_validator(
    model: Any, *, wall_surface_point_base_m: Sequence[float],
    panel_normal_base: Sequence[float],
) -> Callable[[Sequence[float], Sequence[float]], None]:
    """Return the Stage-A-equivalent sampled collision/half-space validator."""

    wall = _finite_vec(wall_surface_point_base_m, 3, "wall surface")
    normal = _unit(panel_normal_base, "panel normal")
    policy = model.policy
    sample_count = int(policy["segment_sample_count"])
    if sample_count < 2:
        raise AlignmentBlocked("ALIGNMENT_POLICY_INVALID", "segment_sample_count")

    def validate(start_q: Sequence[float], target_q: Sequence[float]) -> None:
        start = _finite_vec(start_q, 6, "path start")
        target = _finite_vec(target_q, 6, "path target")
        for sample_index in range(sample_count):
            ratio = sample_index / (sample_count - 1)
            state = model.fk(_add(start, _scale(_sub(target, start), ratio)))
            capsules = {}
            for item in policy["capsules"]:
                a = state.points[int(item["joint_a"])]
                b = state.points[int(item["joint_b"])]
                radius = float(item["radius_m"])
                capsules[str(item["name"])] = (a, b, radius)
                if min(
                    _dot(normal, _sub(a, wall)),
                    _dot(normal, _sub(b, wall)),
                ) - radius < float(policy["wall_margin_m"]):
                    raise AlignmentBlocked("WALL_HALFSPACE_REJECTED", str(item["name"]))
                for proxy_key, configured_key in (
                    ("base_proxy", "base_check_capsules"),
                    ("camera_bracket_proxy", "bracket_check_capsules"),
                ):
                    if item["name"] not in policy[configured_key]:
                        continue
                    proxy = policy[proxy_key]
                    primitive = str(proxy.get("primitive") or "sphere")
                    if primitive == "sphere":
                        distance = _point_segment_distance(proxy["center_m"], a, b)
                    elif primitive == "capsule":
                        distance = _segment_distance(
                            a, b, proxy["endpoint_a_m"], proxy["endpoint_b_m"]
                        )
                    else:
                        raise AlignmentBlocked("ALIGNMENT_POLICY_INVALID", proxy_key)
                    if distance < radius + float(proxy["radius_m"]):
                        raise AlignmentBlocked("BASE_BRACKET_PROXY_REJECTED", proxy_key)
            for left, right in policy["self_pairs"]:
                a0, a1, ar = capsules[left]
                b0, b1, br = capsules[right]
                if _segment_distance(a0, a1, b0, b1) < ar + br:
                    raise AlignmentBlocked("SELF_PROXY_REJECTED", f"{left}/{right}")

    return validate


@dataclass(frozen=True)
class ArrivalReceipt:
    receipt_id: str
    receipt_sha256: str
    cycle_id: str
    terminal_code: str
    session_id: str
    cart_pose_epoch: str
    target_snapshot_sha256: str
    stage_a_plan_hash: str
    p_button_base_m: tuple[float, float, float]
    p_pre_base_m: tuple[float, float, float]
    panel_normal_base: tuple[float, float, float]
    wall_surface_point_base_m: tuple[float, float, float]
    kinematic_model_hash: str
    collision_proxy_hash: str
    tool_model_hash: str
    calibration_hash: str
    tool_identity: str
    executor_instance_id: str
    wrist_camera_calibration_identity: str
    wrist_camera_calibration_hash: str
    terminal_q_rad: tuple[float, ...]
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

    def unsigned_mapping(self) -> Mapping[str, Any]:
        value = asdict(self)
        value.pop("receipt_id", None)
        value.pop("receipt_sha256", None)
        return value

    def validate(self) -> None:
        for name in (
            "receipt_id", "receipt_sha256", "cycle_id", "terminal_code",
            "session_id", "cart_pose_epoch", "target_snapshot_sha256",
            "stage_a_plan_hash", "kinematic_model_hash",
            "collision_proxy_hash", "tool_model_hash", "calibration_hash",
            "tool_identity", "executor_instance_id",
            "wrist_camera_calibration_identity", "wrist_camera_calibration_hash",
        ):
            if not str(getattr(self, name)).strip():
                raise AlignmentBlocked("ARRIVAL_RECEIPT_INVALID", name)
        for name in (
            "receipt_sha256", "target_snapshot_sha256", "stage_a_plan_hash",
            "kinematic_model_hash", "collision_proxy_hash", "tool_model_hash",
            "calibration_hash", "wrist_camera_calibration_hash",
        ):
            _digest(getattr(self, name), name)
        if self.terminal_code != "DYNAMIC_PRECONTACT_STOPPED_DISARMED":
            raise AlignmentBlocked("STAGE_A_TERMINAL_NOT_ACCEPTED")
        _finite_vec(self.terminal_q_rad, 6, "terminal_q_rad")
        _finite_vec(self.p_button_base_m, 3, "p_button_base_m")
        _finite_vec(self.p_pre_base_m, 3, "p_pre_base_m")
        _finite_vec(self.wall_surface_point_base_m, 3, "wall_surface_point_base_m")
        _unit(self.panel_normal_base, "panel_normal_base")
        if self.feedback_stamp_ns <= 0:
            raise AlignmentBlocked("ARRIVAL_RECEIPT_INVALID", "feedback_stamp_ns")
        if (
            type(self.feedback_fresh) is not bool or not self.feedback_fresh
            or self.stationary_sample_count < 3
            or not math.isfinite(self.stationary_span_rad)
            or self.stationary_span_rad < 0.0
        ):
            raise AlignmentBlocked("ARRIVAL_RECEIPT_POSTURE_UNPROVEN")
        expected = canonical_hash(self.unsigned_mapping())
        if self.receipt_sha256 != expected or self.receipt_id != expected:
            raise AlignmentBlocked("ARRIVAL_RECEIPT_INVALID", "canonical digest")
        if not isinstance(self.terminal_resources, Mapping) or any(
            type(value) is not int or value != 0
            for value in self.terminal_resources.values()
        ):
            raise AlignmentBlocked("ARRIVAL_RECEIPT_UNSAFE", "terminal resources")
        if (
            not self.cleanup_proven or self.joint_target_publishers != 0
            or self.lease_active or self.armed or not self.teardown_complete
        ):
            raise AlignmentBlocked("ARRIVAL_RECEIPT_UNSAFE", "Stage-A cleanup")


@dataclass(frozen=True)
class RayPlaneMeasurement:
    measurement_id: str
    measurement_sha256: str
    pre_lock_id: str
    target_pixel: tuple[float, float]
    touch_pixel: tuple[float, float]
    frame_stamp_ns: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, float, float, float, float]
    camera_origin_base_m: tuple[float, float, float]
    rotation_base_from_camera: tuple[tuple[float, float, float], ...]
    camera_calibration_identity: str
    camera_calibration_hash: str
    camera_calibration_execution_ready: bool

    def unsigned_mapping(self) -> Mapping[str, Any]:
        value = asdict(self)
        value.pop("measurement_id", None)
        value.pop("measurement_sha256", None)
        return value

    def validate(self, receipt: ArrivalReceipt) -> None:
        if not self.measurement_id or not self.pre_lock_id:
            raise AlignmentBlocked("PRE_LOCK_INVALID", "identity")
        expected = canonical_hash(self.unsigned_mapping())
        if self.measurement_sha256 != expected or self.measurement_id != expected:
            raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", "canonical digest")
        _finite_vec(self.target_pixel, 2, "target_pixel")
        _finite_vec(self.touch_pixel, 2, "touch_pixel")
        if self.frame_stamp_ns <= receipt.feedback_stamp_ns:
            raise AlignmentBlocked("PRE_LOCK_STALE", "frame predates arrival receipt")
        if any(
            not math.isfinite(value)
            for value in (self.fx, self.fy, self.cx, self.cy)
        ) or self.fx <= 0 or self.fy <= 0:
            raise AlignmentBlocked("PRE_LOCK_INVALID", "intrinsics")
        if (
            self.distortion_model != "plumb_bob"
            or len(self.distortion_coefficients) != 5
        ):
            raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", "distortion")
        _finite_vec(self.distortion_coefficients, 5, "distortion")
        _finite_vec(self.camera_origin_base_m, 3, "camera origin")
        rows = tuple(
            _finite_vec(row, 3, "R_base_from_wrist_camera")
            for row in self.rotation_base_from_camera
        )
        if len(rows) != 3:
            raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", "SO3 shape")
        for index, row in enumerate(rows):
            if abs(_norm(row) - 1.0) > 1e-6:
                raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", f"SO3 row {index}")
        if any(abs(_dot(rows[i], rows[j])) > 1e-6 for i in range(3) for j in range(i + 1, 3)):
            raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", "SO3 orthogonality")
        determinant = _dot(rows[0], (
            rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1],
            rows[1][2] * rows[2][0] - rows[1][0] * rows[2][2],
            rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0],
        ))
        if abs(determinant - 1.0) > 1e-6:
            raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", "SO3 determinant")
        if (
            self.camera_calibration_execution_ready is not True
            or self.camera_calibration_identity
            != receipt.wrist_camera_calibration_identity
            or self.camera_calibration_hash != receipt.wrist_camera_calibration_hash
        ):
            raise AlignmentBlocked("WRIST_CALIBRATION_NOT_EXECUTION_READY")

    def _ray_camera(self, pixel: Sequence[float]) -> tuple[float, float, float]:
        u, v = _finite_vec(pixel, 2, "pixel")
        xd, yd = (u - self.cx) / self.fx, (v - self.cy) / self.fy
        k1, k2, p1, p2, k3 = self.distortion_coefficients
        x, y = xd, yd
        for _ in range(8):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            if abs(radial) < 1e-9:
                raise AlignmentBlocked("RAY_PLANE_MEASUREMENT_INVALID", "distortion singular")
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x, y = (xd - dx) / radial, (yd - dy) / radial
        return _unit((x, y, 1.0), "undistorted ray")

    def panel_points(
        self, receipt: ArrivalReceipt
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        self.validate(receipt)
        origin = _finite_vec(self.camera_origin_base_m, 3, "camera origin")
        normal = _unit(receipt.panel_normal_base, "panel normal")
        wall = _finite_vec(receipt.wall_surface_point_base_m, 3, "wall")
        points = []
        for pixel in (self.target_pixel, self.touch_pixel):
            ray_base = _unit(
                _mat_vec(self.rotation_base_from_camera, self._ray_camera(pixel)),
                "base ray",
            )
            denominator = _dot(normal, ray_base)
            if abs(denominator) < 1e-6:
                raise AlignmentBlocked("RAY_PLANE_PARALLEL")
            distance = _dot(normal, _sub(wall, origin)) / denominator
            if not math.isfinite(distance) or distance <= 0.0:
                raise AlignmentBlocked("RAY_PLANE_BEHIND_CAMERA")
            points.append(_add(origin, _scale(ray_base, distance)))
        return points[0], points[1]


@dataclass(frozen=True)
class FrozenAlignmentPlan:
    plan: Mapping[str, Any]
    plan_sha256: str


def build_alignment_plan(
    *, alignment_id: str, receipt: ArrivalReceipt,
    measurement: RayPlaneMeasurement, model: Any,
    per_move_joint_caps_rad: Sequence[float],
    joint_hard_limits_rad: Sequence[Sequence[float]],
    path_validator: Callable[[Sequence[float], Sequence[float]], None],
    cumulative_tangent_before_m: float = 0.0,
) -> FrozenAlignmentPlan:
    """Build one capped tangent endpoint from validated ray-plane geometry."""

    receipt.validate()
    if not str(alignment_id).strip():
        raise AlignmentBlocked("ALIGNMENT_INPUT_INVALID", "alignment_id")
    measurement.validate(receipt)
    q0 = _finite_vec(receipt.terminal_q_rad, 6, "q0")
    required_hashes = {
        "kinematic_model_hash": receipt.kinematic_model_hash,
        "collision_proxy_hash": receipt.collision_proxy_hash,
        "tool_model_hash": receipt.tool_model_hash,
        "calibration_hash": receipt.calibration_hash,
    }
    model_hashes = dict(getattr(model, "hashes", {}))
    if any(model_hashes.get(key) != value for key, value in required_hashes.items()):
        raise AlignmentBlocked("ALIGNMENT_MODEL_IDENTITY_MISMATCH")
    start_state = model.fk(q0)
    tcp0 = _finite_vec(start_state.tcp, 3, "current_tcp_base_m")
    axis0 = _unit(start_state.axis, "current_tool_axis_base")
    normal = _unit(receipt.panel_normal_base, "panel_normal_base")
    target_panel, touch_panel = measurement.panel_points(receipt)
    raw_base = _sub(target_panel, touch_panel)
    tangent_requested = tuple(
        raw_base[i] - _dot(raw_base, normal) * normal[i] for i in range(3)
    )
    requested_magnitude = _norm(tangent_requested)
    cumulative_before = float(cumulative_tangent_before_m)
    if not math.isfinite(cumulative_before) or not 0.0 <= cumulative_before <= MAX_CUMULATIVE_TANGENT_CORRECTION_M:
        raise AlignmentBlocked("CUMULATIVE_TANGENT_LIMIT_REJECTED")
    remaining = MAX_CUMULATIVE_TANGENT_CORRECTION_M - cumulative_before
    move_cap = min(MAX_PER_MOVE_TANGENT_CORRECTION_M, remaining)
    if move_cap <= 0.0:
        raise AlignmentBlocked("CUMULATIVE_TANGENT_LIMIT_REJECTED")
    scale = 1.0 if requested_magnitude <= move_cap else move_cap / requested_magnitude
    tangent = tuple(value * scale for value in tangent_requested)
    if abs(_dot(tangent, normal)) > 1e-10:
        raise AlignmentBlocked("NORMAL_DRIFT_REJECTED")
    target_tcp = tuple(tcp0[i] + tangent[i] for i in range(3))
    try:
        audit = model.solve(q0, target_tcp, axis0)
    except Exception as exc:
        raise AlignmentBlocked(str(getattr(exc, "code", "IK_FAILED")), str(exc)) from exc
    target_q = _finite_vec(audit.q, 6, "candidate_target_rad")
    caps = _finite_vec(per_move_joint_caps_rad, 6, "per_move_joint_caps_rad")
    if len(joint_hard_limits_rad) != 6:
        raise AlignmentBlocked("ALIGNMENT_POLICY_INVALID", "joint limits")
    for index, (start, target, cap, limits) in enumerate(zip(q0, target_q, caps, joint_hard_limits_rad)):
        lo, hi = _finite_vec(limits, 2, f"joint_limits[{index}]")
        if not lo <= target <= hi:
            raise AlignmentBlocked("JOINT_LIMIT_REJECTED", str(index))
        if abs(target - start) > cap + 1e-12:
            raise AlignmentBlocked("WAYPOINT_CAP_REJECTED", str(index))
    if any(abs(target_q[index] - q0[index]) > 1e-12 for index in LOCKED_JOINT_INDICES):
        raise AlignmentBlocked("CAMERA_JOINT_LOCK_FAILED")
    final_state = model.fk(target_q)
    if _norm(_sub(final_state.tcp, target_tcp)) > 0.0025:
        raise AlignmentBlocked("IK_RESIDUAL_EXCEEDED")
    if math.degrees(math.acos(max(-1.0, min(1.0, _dot(_unit(final_state.axis, "final axis"), axis0))))) > 8.0:
        raise AlignmentBlocked("ORIENTATION_WINDOW_MISS")
    path_validator(q0, target_q)
    payload = {
        "schema": SCHEMA,
        "alignment_id": str(alignment_id),
        "arrival_receipt_id": receipt.receipt_id,
        "arrival_receipt_sha256": receipt.receipt_sha256,
        "cycle_id": receipt.cycle_id,
        "session_id": receipt.session_id,
        "cart_pose_epoch": receipt.cart_pose_epoch,
        "target_snapshot_sha256": receipt.target_snapshot_sha256,
        "stage_a_plan_hash": receipt.stage_a_plan_hash,
        "calibration_hash": receipt.calibration_hash,
        "tool_identity": receipt.tool_identity,
        "executor_instance_id": receipt.executor_instance_id,
        "pre_lock_id": measurement.pre_lock_id,
        "pre_lock_frame_stamp_ns": measurement.frame_stamp_ns,
        "wrist_calibration_identity": measurement.camera_calibration_identity,
        "wrist_calibration_hash": measurement.camera_calibration_hash,
        "ray_plane_measurement_id": measurement.measurement_id,
        "target_panel_point_base_m": list(target_panel),
        "touch_panel_point_base_m": list(touch_panel),
        "raw_requested_base_m": list(raw_base),
        "requested_tangent_m": list(tangent_requested),
        "requested_tangent_norm_m": requested_magnitude,
        "capped_tangent_m": list(tangent),
        "capped_tangent_norm_m": _norm(tangent),
        "maximum_per_move_tangent_correction_m": MAX_PER_MOVE_TANGENT_CORRECTION_M,
        "maximum_cumulative_tangent_correction_m": MAX_CUMULATIVE_TANGENT_CORRECTION_M,
        "cumulative_tangent_before_m": cumulative_before,
        "cumulative_tangent_after_m": cumulative_before + _norm(tangent),
        "predicted_normal_drift_m": _dot(tangent, normal),
        "panel_normal_base": list(normal),
        "start_tcp_base_m": list(tcp0),
        "target_tcp_base_m": list(target_tcp),
        "start_tool_axis_base": list(axis0),
        "start_q_rad": list(q0),
        "candidate_target_rad": list(target_q),
        "locked_joint_indices": list(LOCKED_JOINT_INDICES),
        "orientation_contract": "START_TOOL_AXIS_PLUS_FIXED_WRIST_ROLL",
        "model_hashes": model_hashes,
        "per_move_joint_caps_rad": list(caps),
        "joint_hard_limits_rad": [list(_finite_vec(item, 2, "joint limits")) for item in joint_hard_limits_rad],
        "no_contact": True,
        "press": False,
        "retract": False,
        "automatic_retry": False,
        "automatic_compensation": False,
        "lease_purpose": LEASE_PURPOSE,
        "confirmation_token": CONFIRMATION_TOKEN,
    }
    return FrozenAlignmentPlan(payload, canonical_hash(payload))


def verify_alignment_plan(
    plan_json: str, plan_sha256: str, *, expected_receipt: ArrivalReceipt,
    expected_alignment_id: str,
    current_q_rad: Sequence[float], path_validator: Callable[[Sequence[float], Sequence[float]], None],
    expected_model_hashes: Mapping[str, str] | None = None,
    q_start_tolerance_rad: float = math.radians(0.25),
) -> Mapping[str, Any]:
    try:
        value = json.loads(plan_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AlignmentBlocked("PLAN_INVALIDATED", "JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise AlignmentBlocked("PLAN_INVALIDATED", "schema")
    if canonical_json(value) != plan_json:
        raise AlignmentBlocked("PLAN_INVALIDATED", "non-canonical JSON")
    if canonical_hash(value) != str(plan_sha256):
        raise AlignmentBlocked("PLAN_INVALIDATED", "hash")
    expected_receipt.validate()
    identities = {
        "alignment_id": str(expected_alignment_id),
        "arrival_receipt_id": expected_receipt.receipt_id,
        "session_id": expected_receipt.session_id,
        "cart_pose_epoch": expected_receipt.cart_pose_epoch,
        "target_snapshot_sha256": expected_receipt.target_snapshot_sha256,
        "stage_a_plan_hash": expected_receipt.stage_a_plan_hash,
        "calibration_hash": expected_receipt.calibration_hash,
        "tool_identity": expected_receipt.tool_identity,
        "executor_instance_id": expected_receipt.executor_instance_id,
        "arrival_receipt_sha256": expected_receipt.receipt_sha256,
        "cycle_id": expected_receipt.cycle_id,
        "wrist_calibration_identity": expected_receipt.wrist_camera_calibration_identity,
        "wrist_calibration_hash": expected_receipt.wrist_camera_calibration_hash,
    }
    if any(value.get(key) != expected for key, expected in identities.items()):
        raise AlignmentBlocked("ALIGNMENT_IDENTITY_MISMATCH")
    if value.get("confirmation_token") != CONFIRMATION_TOKEN or value.get("lease_purpose") != LEASE_PURPOSE:
        raise AlignmentBlocked("PLAN_INVALIDATED", "authority")
    if value.get("orientation_contract") != "START_TOOL_AXIS_PLUS_FIXED_WRIST_ROLL":
        raise AlignmentBlocked("PLAN_INVALIDATED", "orientation contract")
    if expected_model_hashes is not None and value.get("model_hashes") != dict(expected_model_hashes):
        raise AlignmentBlocked("ALIGNMENT_MODEL_IDENTITY_MISMATCH")
    if (
        value.get("no_contact") is not True
        or value.get("press") is not False
        or value.get("retract") is not False
        or value.get("automatic_retry") is not False
        or value.get("automatic_compensation") is not False
    ):
        raise AlignmentBlocked("CONTACT_CONTRACT_REJECTED")
    start_q = _finite_vec(value.get("start_q_rad", ()), 6, "start_q_rad")
    current_q = _finite_vec(current_q_rad, 6, "current_q_rad")
    if max(abs(a - b) for a, b in zip(start_q, current_q)) > q_start_tolerance_rad:
        raise AlignmentBlocked("START_POSTURE_DRIFT")
    target_q = _finite_vec(value.get("candidate_target_rad", ()), 6, "candidate_target_rad")
    caps = _finite_vec(value.get("per_move_joint_caps_rad", ()), 6, "caps")
    limits = value.get("joint_hard_limits_rad")
    if not isinstance(limits, list) or len(limits) != 6:
        raise AlignmentBlocked("PLAN_INVALIDATED", "limits")
    for index, (start, target, cap, bounds) in enumerate(zip(start_q, target_q, caps, limits)):
        lo, hi = _finite_vec(bounds, 2, f"limits[{index}]")
        if abs(target - start) > cap + 1e-12 or not lo <= target <= hi:
            raise AlignmentBlocked("PLAN_INVALIDATED", f"joint {index}")
    if any(abs(target_q[index] - start_q[index]) > 1e-12 for index in LOCKED_JOINT_INDICES):
        raise AlignmentBlocked("CAMERA_JOINT_LOCK_FAILED")
    tangent = _finite_vec(value.get("capped_tangent_m", ()), 3, "capped_tangent_m")
    normal = _unit(value.get("panel_normal_base", ()), "panel_normal_base")
    cumulative_before = float(value.get("cumulative_tangent_before_m", math.inf))
    cumulative_after = float(value.get("cumulative_tangent_after_m", math.inf))
    if (
        _norm(tangent) > MAX_PER_MOVE_TANGENT_CORRECTION_M + 1e-12
        or cumulative_before < 0.0
        or cumulative_after > MAX_CUMULATIVE_TANGENT_CORRECTION_M + 1e-12
        or abs(cumulative_after - cumulative_before - _norm(tangent)) > 1e-10
        or abs(_dot(tangent, normal)) > 1e-10
    ):
        raise AlignmentBlocked("NORMAL_DRIFT_REJECTED")
    path_validator(start_q, target_q)
    return value


class FakeAlignmentLifecycle:
    """Deterministic resource/accounting model for one APPLY_ONE_XY."""

    def __init__(
        self, target_q: Sequence[float], *, start_q: Sequence[float] | None = None,
        encoder_step_rad: float = 0.0015339807878856412,
    ) -> None:
        self.target_q = _finite_vec(target_q, 6, "target_q")
        self.start_q = _finite_vec(
            (0.0,) * 6 if start_q is None else start_q, 6, "start_q"
        )
        self.encoder_step_rad = float(encoder_step_rad)
        if not math.isfinite(self.encoder_step_rad) or self.encoder_step_rad <= 0.0:
            raise AlignmentBlocked("ALIGNMENT_POLICY_INVALID", "encoder_step_rad")
        self.state = "READY"
        self.publish_count = 0
        self.publisher_count = 0
        self.lease_active = False
        self.armed = False
        self.code = ""
        self.endpoint_error_rad = (math.inf,) * 6

    def begin(self) -> None:
        if self.state != "READY":
            raise AlignmentBlocked("ALIGNMENT_PROTOCOL_ERROR")
        self.lease_active = True
        self.publisher_count = 1
        self.armed = True
        self.publish_count = 1
        self.state = "IN_FLIGHT"

    def terminal(self, measured_q: Sequence[float], *, progress_code: str | None = None) -> None:
        if self.state != "IN_FLIGHT":
            raise AlignmentBlocked("ALIGNMENT_PROTOCOL_ERROR")
        measured = _finite_vec(measured_q, 6, "measured_q")
        self.endpoint_error_rad = tuple(abs(a - b) for a, b in zip(measured, self.target_q))
        measured_delta = tuple(a - b for a, b in zip(measured, self.start_q))
        requested_delta = tuple(a - b for a, b in zip(self.target_q, self.start_q))
        moving = tuple(
            index for index, delta in enumerate(requested_delta)
            if abs(delta) > self.encoder_step_rad / 2.0
        )
        if not moving:
            raise AlignmentBlocked("ALIGNMENT_PLAN_INVALID", "no moving joint")
        signed_progress = tuple(
            measured_delta[index] * (1.0 if requested_delta[index] > 0.0 else -1.0)
            for index in moving
        )
        locked = tuple(index for index in range(6) if index not in moving)
        measured_failure = None
        if any(
            progress < -self.encoder_step_rad / 2.0 for progress in signed_progress
        ):
            measured_failure = "WRONG_DIRECTION"
        elif any(
            abs(measured_delta[index]) > self.encoder_step_rad / 2.0
            for index in locked
        ):
            measured_failure = "LOCKED_JOINT_DRIFT"
        elif any(
            progress <= self.encoder_step_rad / 2.0 for progress in signed_progress
        ):
            measured_failure = "NO_PROGRESS"
        elif any(
            self.endpoint_error_rad[index] > ENDPOINT_TOLERANCE_RAD
            for index in moving
        ):
            measured_failure = "UNDERTRAVEL"
        external_failure = (
            progress_code
            if progress_code in {
                "WRONG_DIRECTION", "LOCKED_JOINT_DRIFT", "NO_PROGRESS", "UNDERTRAVEL"
            }
            else None
        )
        classified = external_failure or measured_failure
        if classified is not None:
            self.code = classified
            self.state = "FAILED"
        else:
            self.code = "NEED_PRESS_LOCK"
            self.state = "NEED_PRESS_LOCK"
        self.cleanup()

    def cleanup(self) -> None:
        self.armed = False
        self.publisher_count = 0
        self.lease_active = False

    @property
    def teardown_complete(self) -> bool:
        return not self.armed and self.publisher_count == 0 and not self.lease_active


class FormalArrivalReceiptLatch:
    """Future-cycle-only Stage-A terminal receipt; never a motion ticket."""

    def __init__(self, boot_cycle_floor: str) -> None:
        self.boot_cycle_floor = str(boot_cycle_floor)
        self.receipt: ArrivalReceipt | None = None

    def record(self, receipt: ArrivalReceipt) -> ArrivalReceipt:
        if self.receipt is not None:
            raise AlignmentBlocked("ARRIVAL_RECEIPT_ALREADY_RECORDED")
        if not receipt.cycle_id or receipt.cycle_id == self.boot_cycle_floor:
            raise AlignmentBlocked("NEW_GO_PRE_REQUIRED")
        receipt.validate()
        self.receipt = receipt
        return receipt

    def clear(self) -> None:
        self.receipt = None


class AlignmentIssuanceRegistry:
    """One receipt issues at most one exact alignment, even after active clear."""

    def __init__(self) -> None:
        self._used_receipts: set[str] = set()
        self._used_alignments: set[str] = set()
        self.active: tuple[str, str, str] | None = None

    def issue(
        self, *, receipt: ArrivalReceipt, alignment_id: str, plan_sha256: str
    ) -> tuple[str, str, str]:
        receipt.validate()
        alignment = str(alignment_id).strip()
        plan = _digest(plan_sha256, "plan_sha256")
        if not alignment:
            raise AlignmentBlocked("ALIGNMENT_INPUT_INVALID", "alignment_id")
        if self.active is not None:
            raise AlignmentBlocked("ALIGNMENT_ALREADY_ACTIVE")
        if receipt.receipt_id in self._used_receipts:
            raise AlignmentBlocked("ARRIVAL_RECEIPT_REPLAY")
        if alignment in self._used_alignments:
            raise AlignmentBlocked("ALIGNMENT_ID_REPLAY")
        self.active = (receipt.receipt_id, alignment, plan)
        self._used_receipts.add(receipt.receipt_id)
        self._used_alignments.add(alignment)
        return self.active

    def verify(self, *, receipt_id: str, alignment_id: str, plan_sha256: str) -> None:
        expected = (str(receipt_id), str(alignment_id), str(plan_sha256))
        if self.active != expected:
            raise AlignmentBlocked("ALIGNMENT_ISSUANCE_MISMATCH")

    def clear_active(self) -> None:
        self.active = None


class FakePressAlignCoordinator:
    """End-to-end fake QC/executor/Live handshake with no ROS resources."""

    def __init__(self, *, boot_cycle_id: str) -> None:
        self.arrivals = FormalArrivalReceiptLatch(boot_cycle_id)
        self.issuances = AlignmentIssuanceRegistry()
        self.state = "NEW_GO_PRE_REQUIRED"
        self.alignment_id = ""
        self.plan: FrozenAlignmentPlan | None = None
        self.machine: FakeAlignmentLifecycle | None = None
        self.pre_lock_present = False
        self.press_lock_id = ""

    def record_future_precontact(self, receipt: ArrivalReceipt) -> None:
        self.arrivals.record(receipt)
        self.state = "NEED_PRE_LOCK"

    def prepare(
        self, *, alignment_id: str, measurement: RayPlaneMeasurement,
        model: Any, per_move_joint_caps_rad: Sequence[float],
        joint_hard_limits_rad: Sequence[Sequence[float]],
        path_validator: Callable[[Sequence[float], Sequence[float]], None],
        teleop_active: bool = False, pose_publisher_active: bool = False,
    ) -> FrozenAlignmentPlan:
        if self.state != "NEED_PRE_LOCK" or self.arrivals.receipt is None:
            raise AlignmentBlocked("PRESS_ALIGN_NOT_READY")
        if teleop_active or pose_publisher_active:
            raise AlignmentBlocked("QC_CONTROL_MODE_CONFLICT")
        self.alignment_id = str(alignment_id)
        self.plan = build_alignment_plan(
            alignment_id=self.alignment_id, receipt=self.arrivals.receipt,
            measurement=measurement, model=model,
            per_move_joint_caps_rad=per_move_joint_caps_rad,
            joint_hard_limits_rad=joint_hard_limits_rad,
            path_validator=path_validator,
        )
        self.issuances.issue(
            receipt=self.arrivals.receipt, alignment_id=self.alignment_id,
            plan_sha256=self.plan.plan_sha256,
        )
        self.pre_lock_present = True
        self.state = "ALIGN_READY"
        return self.plan

    def dispatch(self) -> FakeAlignmentLifecycle:
        if self.state != "ALIGN_READY" or self.plan is None:
            raise AlignmentBlocked("PRESS_ALIGN_NOT_READY")
        self.machine = FakeAlignmentLifecycle(
            self.plan.plan["candidate_target_rad"],
            start_q=self.plan.plan["start_q_rad"],
        )
        self.machine.begin()
        self.state = "ALIGN_IN_FLIGHT"
        return self.machine

    def terminal(self, measured_q: Sequence[float], *, progress_code: str | None = None) -> None:
        if self.state != "ALIGN_IN_FLIGHT" or self.machine is None:
            raise AlignmentBlocked("ALIGNMENT_PROTOCOL_ERROR")
        self.machine.terminal(measured_q, progress_code=progress_code)
        if self.machine.state == "NEED_PRESS_LOCK":
            self.pre_lock_present = False
            self.state = "REACQUIRE_PRESS_LOCK"
        else:
            self.state = f"FAILED:{self.machine.code}"

    def submit_press_lock(self, *, lock_id: str, frame_stamp_ns: int) -> None:
        if self.state != "REACQUIRE_PRESS_LOCK" or self.arrivals.receipt is None:
            raise AlignmentBlocked("PRESS_REACQUIRE_NOT_STARTED")
        if not lock_id or int(frame_stamp_ns) <= self.arrivals.receipt.feedback_stamp_ns:
            raise AlignmentBlocked("PRESS_LOCK_INVALID")
        self.press_lock_id = str(lock_id)
        self.state = "ALIGN_COMPLETE_PRESS_LOCKED"

    def stop(self) -> None:
        if self.machine is not None:
            self.machine.cleanup()
        self.state = "CANCELLED"

    def clear_identity(self) -> None:
        if self.machine is not None:
            self.machine.cleanup()
        self.arrivals.clear()
        self.issuances.clear_active()
        self.pre_lock_present = False
        self.press_lock_id = ""
        self.state = "NEW_GO_PRE_REQUIRED"


__all__ = [
    "AlignmentBlocked", "AlignmentIssuanceRegistry", "ArrivalReceipt",
    "CONFIRMATION_TOKEN", "PREPARE_CONFIRMATION_TOKEN",
    "ENDPOINT_TOLERANCE_RAD", "FakeAlignmentLifecycle", "FrozenAlignmentPlan",
    "LEASE_PURPOSE", "LOCKED_JOINT_INDICES",
    "MAX_PER_MOVE_TANGENT_CORRECTION_M",
    "MAX_CUMULATIVE_TANGENT_CORRECTION_M", "MAX_TANGENT_CORRECTION_M",
    "FakePressAlignCoordinator", "FormalArrivalReceiptLatch",
    "RayPlaneMeasurement", "SCHEMA",
    "build_alignment_plan", "canonical_hash",
    "canonical_json", "model_path_validator", "verify_alignment_plan",
]
