"""Pure target/session binding for the fixed T102 Button-1 template.

No ROS, kinematics, URDF, mesh, checker, receipt, subprocess, or motion API is
imported here.  A matching session only admits the fixed route identity; it is
not a claim that a target-derived route was planned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "config/press/prototype/t102_button1_template_xy_policy_20260816_v1.json"


class TemplateBlocked(ValueError):
    def __init__(self, code: str, reason: str, *, offset_m: Sequence[float] = ()) -> None:
        super().__init__(reason)
        self.code = code
        self.offset_m = tuple(float(v) for v in offset_m)


def _finite(values: Sequence[float], count: int, name: str) -> tuple[float, ...]:
    if len(values) != count:
        raise TemplateBlocked("SESSION_INVALID", f"{name} must have {count} values")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise TemplateBlocked("SESSION_INVALID", f"{name} must be finite")
    return result


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_policy(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "t102_button1_template_xy_policy_v1":
        raise TemplateBlocked("SESSION_INVALID", "template policy schema mismatch")
    return value


def _rotate_xyzw(vector: Sequence[float], quaternion: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _finite(vector, 3, "vector")
    qx, qy, qz, qw = _finite(quaternion, 4, "quaternion")
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-12:
        raise TemplateBlocked("SESSION_INVALID", "zero quaternion")
    qx, qy, qz, qw = (v / norm for v in (qx, qy, qz, qw))
    # q * v * q^-1, expanded without a numerical dependency.
    tx, ty, tz = 2.0*(qy*z-qz*y), 2.0*(qz*x-qx*z), 2.0*(qx*y-qy*x)
    return (
        x + qw*tx + qy*tz - qz*ty,
        y + qw*ty + qz*tx - qx*tz,
        z + qw*tz + qx*ty - qy*tx,
    )


def camera_point_to_base(point_camera_m: Sequence[float], policy: Mapping[str, Any]) -> tuple[float, float, float]:
    transform = policy["base_transform"]
    rotated = _rotate_xyzw(point_camera_m, transform["rotation_xyzw"])
    translation = _finite(transform["translation_m"], 3, "translation")
    return tuple(a+b for a, b in zip(rotated, translation))


def camera_direction_to_base(direction_camera: Sequence[float], policy: Mapping[str, Any]) -> tuple[float, float, float]:
    rotated = _rotate_xyzw(direction_camera, policy["base_transform"]["rotation_xyzw"])
    norm = math.sqrt(sum(v*v for v in rotated))
    if norm < 1e-12:
        raise TemplateBlocked("SESSION_INVALID", "zero panel normal")
    return tuple(v/norm for v in rotated)


@dataclass(frozen=True)
class FrozenSession:
    session_id: str
    target_snapshot_sha256: str
    source_stamp_ns: int
    detection_id: str
    confidence: float
    target_status: str
    target_base_m: tuple[float, float, float]
    base_offset_m: tuple[float, float, float]
    calibration_id: str
    calibration_bundle_sha256: str
    start_positions_rad: tuple[float, ...]
    route_sha256: str
    template_code: str = "TEMPLATE_MATCH"


def bind_session(
    snapshot: Mapping[str, Any],
    snapshot_sha256: str,
    start_positions_rad: Sequence[float],
    *,
    policy: Mapping[str, Any] | None = None,
    now_unix_ns: int | None = None,
) -> FrozenSession:
    policy = dict(policy or load_policy())
    frozen = snapshot.get("frozen_target")
    if not isinstance(frozen, Mapping):
        raise TemplateBlocked("SESSION_INVALID", "snapshot missing frozen_target")
    surface = frozen.get("surface_geometry")
    if not isinstance(surface, Mapping):
        raise TemplateBlocked("SESSION_INVALID", "snapshot missing surface geometry")
    point = frozen.get("camera_point_m")
    normal = surface.get("panel_outward_normal")
    if not isinstance(point, Mapping) or not isinstance(normal, Mapping):
        raise TemplateBlocked("SESSION_INVALID", "snapshot missing target point/normal")
    point_base = camera_point_to_base([point.get(k) for k in "xyz"], policy)
    normal_base = camera_direction_to_base([normal.get(k) for k in "xyz"], policy)
    template = policy["template"]
    source_stamp_ns = int(frozen.get("source_stamp_ns") or 0)
    capture_nonce = str(frozen.get("capture_nonce") or "")
    boot_epoch = str(frozen.get("boot_epoch") or "")
    cart_pose_epoch = str(frozen.get("cart_pose_epoch") or "")
    confidence = float(frozen.get("confidence") or 0.0)
    target_status = str(frozen.get("status") or "").upper()
    if not capture_nonce or frozen.get("snapshot_id") != capture_nonce:
        raise TemplateBlocked("SESSION_INVALID", "fresh snapshot_id/capture_nonce is required")
    if not boot_epoch or not cart_pose_epoch:
        raise TemplateBlocked("SESSION_INVALID", "boot_epoch/cart_pose_epoch is required")
    if target_status not in set(template["allowed_target_status"]):
        raise TemplateBlocked("TARGET_NOT_READY", "frozen target status is not READY")
    if not math.isfinite(confidence) or not (float(template["minimum_confidence_exclusive"]) < confidence <= 1.0):
        raise TemplateBlocked("TARGET_CONFIDENCE_INVALID", "frozen target confidence is invalid")
    if now_unix_ns is None:
        raise TemplateBlocked("SESSION_INVALID", "current time is required for target freshness")
    age_s = (int(now_unix_ns) - source_stamp_ns) / 1e9
    if age_s < 0.0 or age_s > float(template["maximum_target_age_s"]):
        raise TemplateBlocked("TARGET_STALE", "frozen target timestamp is stale or in the future")
    nominal = _finite(template["nominal_target_base_m"], 3, "nominal target")
    limits = _finite(template["maximum_absolute_offset_m"], 3, "template window")
    offset = tuple(a-b for a, b in zip(point_base, nominal))
    if any(abs(value) > limit + 1e-12 for value, limit in zip(offset, limits)):
        raise TemplateBlocked("TEMPLATE_MISS", "target is outside the fixed-template validity window", offset_m=offset)
    nominal_normal = _finite(template["nominal_normal_base"], 3, "nominal normal")
    dot = max(-1.0, min(1.0, sum(a*b for a, b in zip(normal_base, nominal_normal))))
    if dot < float(template["minimum_normal_dot"]) - 1e-12:
        raise TemplateBlocked("TEMPLATE_MISS", "panel normal is outside the fixed-template window", offset_m=offset)
    start = _finite(start_positions_rad, 6, "Standard Standby")
    exact_start = _finite(template["start_positions_rad"], 6, "template start")
    if max(abs(a-b) for a, b in zip(start, exact_start)) > float(template["start_tolerance_rad"]) + 1e-12:
        raise TemplateBlocked("START_DRIFT", "measured start differs from exact Standard Standby")
    calibration_id = str(surface.get("section_b_calibration_id") or "")
    expected_calibration_id = str(policy["base_transform"]["calibration_id"])
    if calibration_id != expected_calibration_id:
        raise TemplateBlocked("SESSION_INVALID", "base transform calibration identity mismatch")
    payload = {
        "target_snapshot_sha256": snapshot_sha256,
        "source_stamp_ns": source_stamp_ns,
        "capture_nonce": capture_nonce,
        "boot_epoch": boot_epoch,
        "cart_pose_epoch": cart_pose_epoch,
        "detection_id": str(frozen.get("detection_id") or ""),
        "confidence": confidence,
        "target_status": target_status,
        "target_base_m": point_base,
        "calibration_id": expected_calibration_id,
        "calibration_bundle_sha256": str(policy["base_transform"]["bundle_sha256"]),
        "start_positions_rad": start,
        "route_sha256": str(template["route_package_sha256"]),
    }
    if payload["source_stamp_ns"] <= 0 or not payload["detection_id"]:
        raise TemplateBlocked("SESSION_INVALID", "target identity/stamp is incomplete")
    session_id = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return FrozenSession(
        session_id=session_id,
        target_snapshot_sha256=snapshot_sha256,
        source_stamp_ns=payload["source_stamp_ns"],
        detection_id=payload["detection_id"],
        confidence=payload["confidence"],
        target_status=payload["target_status"],
        target_base_m=point_base,
        base_offset_m=offset,
        calibration_id=expected_calibration_id,
        calibration_bundle_sha256=payload["calibration_bundle_sha256"],
        start_positions_rad=start,
        route_sha256=payload["route_sha256"],
    )
