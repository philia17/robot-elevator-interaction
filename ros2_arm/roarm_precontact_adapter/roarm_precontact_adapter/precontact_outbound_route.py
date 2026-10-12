#!/usr/bin/env python3
"""Installed pure offline builder for one outbound pre-contact route.

This module is deliberately separate from the historical M1/M2/M3 routes.  It
only proposes ``standby -> A + 0.050 * N_out`` and never opens ROS, serial, or
an action client.  A formal artifact is usable *only* by a future dedicated
pre-contact action server; it is not a motion authorization.

The input is a sealed release manifest.  Every embedded release is hashed over
its payload, and the manifest is hashed over its complete content.  The helper
``seal_*`` functions make synthetic/offline fixtures deterministic; they do not
turn a diagnostic result into a release.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence


MANIFEST_SCHEMA = "precontact_outbound_release_manifest_v1"
ROUTE_SCHEMA = "precontact_outbound_route_v1"
DIAGNOSTIC_SCHEMA = "precontact_outbound_diagnostic_v1"
PHASE = "OUTBOUND_TO_PRECONTACT"
EXPECTED_CAMERA_FRAME = "camera_optical_frame"
EXPECTED_HAND_FRAME = "hand_tcp"
EXPECTED_CONTACT_FRAME = "press_contact"
RELEASE_STATUS = "released"


class RouteBlocked(ValueError):
    """A release, geometry, IK, or collision gate failed closed."""


@dataclass(frozen=True)
class IKRequest:
    """Full hand-TCP goal derived from the released contact-frame target."""

    hand_target_position_m: tuple[float, float, float]
    hand_target_rotation_xyzw: tuple[float, float, float, float]
    contact_target_position_m: tuple[float, float, float]
    contact_target_rotation_xyzw: tuple[float, float, float, float]
    required_contact_axis_base: tuple[float, float, float]
    route_phase: str = PHASE


@dataclass(frozen=True)
class CollisionRequest:
    """The whole quantized path, including standby, passed to an offline checker."""

    quantized_start_setpoint_rad: tuple[float, ...]
    emitted_setpoints_rad: tuple[tuple[float, ...], ...]
    path_setpoints_rad: tuple[tuple[float, ...], ...]
    atomic_steps: tuple[Mapping[str, Any], ...]
    joint_order: tuple[int, ...]
    collision_model_sha256: str
    required_minimum_clearance_m: float
    route_phase: str = PHASE


@dataclass(frozen=True)
class FKRequest:
    """One pose in a candidate quantized T101 path to evaluate offline."""

    joint_positions_rad: tuple[float, ...]
    path_index: int
    path_length: int
    joint_order: tuple[int, ...]
    is_final: bool
    route_phase: str = PHASE


IKSolver = Callable[[IKRequest], Mapping[str, Any]]
KinematicsChecker = Callable[[FKRequest], Mapping[str, Any]]
CollisionChecker = Callable[[CollisionRequest], Mapping[str, Any]]


def canonical_json(value: Any) -> bytes:
    """Canonical JSON bytes used for all IDs and cryptographic bindings."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def seal_release(release_id: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a hash-bound release envelope for offline fixtures/tools.

    This only binds bytes.  ``build_outbound_route`` independently requires
    ``status == 'released'`` and all safety semantics below.
    """

    payload_dict = dict(payload)
    return {
        "release_id": str(release_id),
        "status": str(status),
        "artifact_sha256": sha256_canonical(payload_dict),
        "payload": payload_dict,
    }


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a manifest with its canonical hash, without changing releases."""

    result = dict(manifest)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = sha256_canonical(result)
    return result


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteBlocked(f"{where} must be an object")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise RouteBlocked(f"{where} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RouteBlocked(f"{where} must be finite") from exc
    if not math.isfinite(result):
        raise RouteBlocked(f"{where} must be finite")
    return result


def _vec(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RouteBlocked(f"{where} must have {length} values")
    if len(value) != length:
        raise RouteBlocked(f"{where} must have {length} values")
    return tuple(_finite(item, f"{where}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, where: str) -> tuple[float, float, float]:
    vector = _vec(value, 3, where)
    magnitude = math.sqrt(sum(item * item for item in vector))
    if magnitude < 1e-12:
        raise RouteBlocked(f"{where} must be non-zero")
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))  # type: ignore[return-value]


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(vector: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return tuple(float(item) * scalar for item in vector)  # type: ignore[return-value]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[row][column]) for row in range(3)] for column in range(3)]


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [sum(float(left[row][index]) * float(right[index][column]) for index in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _rotate(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _quat_to_matrix(value: Any, where: str) -> list[list[float]]:
    x, y, z, w = _vec(value, 4, where)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise RouteBlocked(f"{where} has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _matrix_to_quat(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to an XYZW quaternion."""

    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / norm, y / norm, z / norm, w / norm)


def _rotation_angle_rad(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    delta = _matmul(_transpose(left), right)
    cosine = max(-1.0, min(1.0, (delta[0][0] + delta[1][1] + delta[2][2] - 1.0) / 2.0))
    return math.acos(cosine)


def _require_hash(value: Any, where: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RouteBlocked(f"{where} must be a lowercase SHA-256")
    return digest


def canonical_route_artifact_hash(artifact: Mapping[str, Any]) -> str:
    """Hash a complete route artifact while excluding only its self-hash field.

    This is the canonical route-artifact hash contract future adapters can
    validate before considering the artifact.  It intentionally covers every
    other field, including route geometry, release/target bindings, emitted
    setpoints, and checker evidence.
    """

    payload = dict(_mapping(artifact, "route artifact"))
    payload.pop("canonical_payload_sha256", None)
    return sha256_canonical(payload)


def _route_content_from_artifact(route: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the subset covered by ``route_hash`` from a formal artifact."""

    bindings = _mapping(route.get("release_bindings"), "route artifact release_bindings")
    geometry = _mapping(route.get("geometry"), "route artifact geometry")
    quantization = _mapping(route.get("quantization"), "route artifact quantization")
    return {
        "manifest_sha256": bindings.get("manifest_sha256"),
        "phase": route.get("phase"),
        "target_binding": _mapping(route.get("target_binding"), "route artifact target_binding"),
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


def validate_outbound_route_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed unless a complete outbound route artifact self-validates.

    This validates artifact identity and the stable target binding only.  It
    is deliberately not motion authority: a future execution package must
    still validate its own fresh feedback, ownership, and authorization gates.
    """

    route = _mapping(artifact, "route artifact")
    if route.get("schema") != ROUTE_SCHEMA:
        raise RouteBlocked("unexpected outbound route artifact schema")
    if route.get("phase") != PHASE or route.get("phases") != [PHASE]:
        raise RouteBlocked("route artifact permits a forbidden phase")
    if route.get("no_contact") is not True or route.get("automatic_retry") is not False:
        raise RouteBlocked("route artifact is not no-contact and no-retry")
    if route.get("executable") is not True or route.get("adapter_consumable") is not True:
        raise RouteBlocked("formal route artifact is not executable/adapter-consumable")
    if route.get("contact_authorized") is not False or route.get("press_authorized") is not False:
        raise RouteBlocked("route artifact improperly authorizes contact")
    _require_hash(route.get("route_hash"), "route artifact route_hash")
    expected_hash = _require_hash(
        route.get("canonical_payload_sha256"), "route artifact canonical_payload_sha256"
    )
    if canonical_route_artifact_hash(route) != expected_hash:
        raise RouteBlocked("route artifact canonical payload hash mismatch")

    target_binding = _mapping(route.get("target_binding"), "route artifact target_binding")
    target_id = str(target_binding.get("target_id") or "").strip()
    detection_id = str(target_binding.get("detection_id") or "").strip()
    if not target_id or not detection_id:
        raise RouteBlocked("route artifact target/detection binding is incomplete")
    if int(target_binding.get("source_stamp_ns") or 0) <= 0:
        raise RouteBlocked("route artifact target stamp is invalid")
    if target_binding.get("source_frame_id") != EXPECTED_CAMERA_FRAME:
        raise RouteBlocked("route artifact target frame is invalid")
    m0_hash = _require_hash(
        target_binding.get("m0_frozen_artifact_sha256"),
        "route artifact m0 frozen hash",
    )
    r6_hash = _require_hash(
        target_binding.get("r6_bundle_artifact_sha256"),
        "route artifact r6 bundle hash",
    )
    if not str(target_binding.get("calibration_release_id") or "").strip():
        raise RouteBlocked("route artifact calibration release binding is incomplete")
    bindings = _mapping(route.get("release_bindings"), "route artifact release_bindings")
    if _mapping(bindings.get("m0_frozen"), "route artifact m0 release").get("artifact_sha256") != m0_hash:
        raise RouteBlocked("route artifact target binding does not match M0 release")
    if _mapping(bindings.get("r6_bundle"), "route artifact r6 release").get("artifact_sha256") != r6_hash:
        raise RouteBlocked("route artifact target binding does not match r6 release")
    if _mapping(bindings.get("r6_bundle"), "route artifact r6 release").get("release_id") != target_binding.get(
        "calibration_release_id"
    ):
        raise RouteBlocked("route artifact target binding does not match calibration release")

    geometry = _mapping(route.get("geometry"), "route artifact geometry")
    a_base = _vec(geometry.get("A_base_m"), 3, "route artifact A_base_m")
    n_out = _unit(geometry.get("N_out_base"), "route artifact N_out_base")
    p_pre = _vec(geometry.get("P_pre_base_m"), 3, "route artifact P_pre_base_m")
    if abs(_distance(a_base, p_pre) - 0.050) > 1e-12:
        raise RouteBlocked("route artifact P_pre is not exactly 50 mm from A_base")
    if _distance(_unit(_sub(p_pre, a_base), "route artifact A_to_P_pre"), n_out) > 1e-9:
        raise RouteBlocked("route artifact P_pre does not follow N_out")
    quantization = _mapping(route.get("quantization"), "route artifact quantization")
    clearance = _mapping(route.get("clearance_evidence"), "route artifact clearance_evidence")
    command_contract = _mapping(
        clearance.get("command_contract"), "route artifact T101 command_contract"
    )
    driver_binding = _mapping(
        bindings.get("driver_profile_release"), "route artifact driver profile release"
    )
    if (
        command_contract.get("driver_profile_release_id") != driver_binding.get("release_id")
        or command_contract.get("driver_profile_artifact_sha256")
        != driver_binding.get("artifact_sha256")
    ):
        raise RouteBlocked("T101 command contract does not bind the released driver profile")
    emitted_raw = quantization.get("emitted_setpoints_rad")
    if not isinstance(emitted_raw, Sequence) or isinstance(emitted_raw, (str, bytes)):
        raise RouteBlocked("route artifact emitted setpoints are invalid")
    _validate_t101_command_contract(
        command_contract,
        _vec(quantization.get("start_quantized_rad"), 6, "route artifact quantized start"),
        emitted_raw,
    )
    if route.get("command_mode") != "T101_SINGLE_JOINT_ONLY":
        raise RouteBlocked("route artifact command_mode is not T101")
    if route.get("command_sequence_sha256") != command_contract.get("command_sequence_sha256"):
        raise RouteBlocked("route artifact command sequence hash alias mismatch")
    if sha256_canonical(_route_content_from_artifact(route)) != route.get("route_hash"):
        raise RouteBlocked("route artifact route hash does not bind its route content")
    expected_route_id = f"precontact-outbound-{str(route.get('route_hash'))[:16]}"
    if route.get("route_id") != expected_route_id:
        raise RouteBlocked("route artifact route_id does not bind route_hash")


def _release(releases: Mapping[str, Any], name: str) -> tuple[Mapping[str, Any], str, str]:
    envelope = _mapping(releases.get(name), f"releases.{name}")
    release_id = str(envelope.get("release_id") or "").strip()
    if not release_id:
        raise RouteBlocked(f"releases.{name}.release_id is required")
    status = str(envelope.get("status") or "")
    if status != RELEASE_STATUS:
        raise RouteBlocked(f"releases.{name} is not released: {status or 'missing'}")
    payload = _mapping(envelope.get("payload"), f"releases.{name}.payload")
    expected_hash = _require_hash(envelope.get("artifact_sha256"), f"releases.{name}.artifact_sha256")
    if sha256_canonical(payload) != expected_hash:
        raise RouteBlocked(f"releases.{name} payload hash mismatch")
    return payload, release_id, expected_hash


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RouteBlocked("unexpected release manifest schema")
    manifest_id = str(manifest.get("manifest_id") or "").strip()
    if not manifest_id:
        raise RouteBlocked("manifest_id is required")
    expected_hash = _require_hash(manifest.get("manifest_sha256"), "manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if sha256_canonical(unsigned) != expected_hash:
        raise RouteBlocked("manifest hash mismatch")
    releases = _mapping(manifest.get("releases"), "releases")
    policy = _mapping(manifest.get("route_policy"), "route_policy")
    if policy.get("no_contact") is not True or policy.get("automatic_retry") is not False:
        raise RouteBlocked("route policy must be no-contact and no-retry")
    if policy.get("allowed_phase") != PHASE:
        raise RouteBlocked("route policy permits a forbidden phase")
    if abs(_finite(policy.get("precontact_distance_m"), "precontact_distance_m") - 0.050) > 1e-12:
        raise RouteBlocked("only exact 50 mm pre-contact is supported")
    return releases, policy, expected_hash


def _desired_contact_rotation(
    normal_out_base: Sequence[float], roll_reference_base: Sequence[float]
) -> list[list[float]]:
    """Return R_base_contact with contact +X facing into the panel (-N_out)."""

    x_axis = _scale(_unit(normal_out_base, "normal_out_base"), -1.0)
    reference = _unit(roll_reference_base, "roll_reference_base")
    y_unscaled = _sub(reference, _scale(x_axis, _dot(reference, x_axis)))
    if math.sqrt(_dot(y_unscaled, y_unscaled)) < 1e-8:
        raise RouteBlocked("roll_reference_base is parallel to the contact axis")
    y_axis = _unit(y_unscaled, "projected roll_reference_base")
    z_axis = _unit(_cross(x_axis, y_axis), "contact z axis")
    # Columns are the child/contact basis expressed in roarm_base_link.
    return [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]


def _quantized_outbound(
    start_positions_rad: Sequence[float],
    end_positions_rad: Sequence[float],
    *,
    encoder_step_rad: float,
    joint_max_delta_rad: Sequence[float],
    joint_limits_rad: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Create grid setpoints with one encoder-step margin to every delta cap."""

    if encoder_step_rad <= 0.0:
        raise RouteBlocked("encoder_step_rad must be positive")
    if len(start_positions_rad) != len(end_positions_rad):
        raise RouteBlocked("IK result has the wrong joint count")
    joint_count = len(start_positions_rad)
    if len(joint_max_delta_rad) != joint_count or len(joint_limits_rad) != joint_count:
        raise RouteBlocked("driver profile joint dimensions disagree")

    start_steps = [int(round(value / encoder_step_rad)) for value in start_positions_rad]
    end_steps = [int(round(value / encoder_step_rad)) for value in end_positions_rad]
    max_steps: list[int] = []
    for index, limit in enumerate(joint_max_delta_rad):
        usable = _finite(limit, f"joint_max_delta_rad[{index}]") - encoder_step_rad
        steps = int(math.floor(usable / encoder_step_rad + 1e-12))
        if steps < 1:
            raise RouteBlocked("joint delta cap leaves no encoder-step safety margin")
        max_steps.append(steps)
    count = max(
        1,
        max(
            int(math.ceil(abs(end - start) / allowed))
            for start, end, allowed in zip(start_steps, end_steps, max_steps)
        ),
    )
    if start_steps == end_steps:
        raise RouteBlocked("quantized pre-contact target would not move any joint")

    setpoints: list[tuple[float, ...]] = []
    previous = tuple(step * encoder_step_rad for step in start_steps)
    for route_index in range(1, count + 1):
        next_steps = [
            int(round(start + route_index * (end - start) / count))
            for start, end in zip(start_steps, end_steps)
        ]
        emitted = tuple(step * encoder_step_rad for step in next_steps)
        if emitted == previous:
            continue
        for joint_index, value in enumerate(emitted):
            lower, upper = _vec(joint_limits_rad[joint_index], 2, f"joint_limits_rad[{joint_index}]")
            if not lower <= value <= upper:
                raise RouteBlocked(f"quantized waypoint joint {joint_index} is outside limits")
            delta = abs(value - previous[joint_index])
            limit = _finite(joint_max_delta_rad[joint_index], f"joint_max_delta_rad[{joint_index}]")
            if delta > limit - encoder_step_rad + 1e-12:
                raise RouteBlocked(
                    f"quantized waypoint joint {joint_index} lacks one encoder-step delta margin"
                )
        setpoints.append(emitted)
        previous = emitted
    if not setpoints or setpoints[-1] != tuple(step * encoder_step_rad for step in end_steps):
        raise RouteBlocked("quantized route does not reach the requested endpoint")
    return tuple(setpoints)


def _released_joint_orders(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise RouteBlocked("driver.joint_order_allowlist must contain released orders")
    if len(value) > 32:
        raise RouteBlocked("driver.joint_order_allowlist exceeds the offline audit bound")
    orders: set[tuple[int, ...]] = set()
    for order_index, raw_order in enumerate(value):
        if not isinstance(raw_order, Sequence) or isinstance(raw_order, (str, bytes)):
            raise RouteBlocked(f"driver.joint_order_allowlist[{order_index}] is invalid")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_order):
            raise RouteBlocked(f"driver.joint_order_allowlist[{order_index}] must use integer indices")
        order = tuple(raw_order)
        if len(order) != 6 or set(order) != set(range(6)):
            raise RouteBlocked(f"driver.joint_order_allowlist[{order_index}] is not a six-joint permutation")
        orders.add(order)
    return tuple(sorted(orders))


def _expand_t101_atomic_steps(
    start_positions_rad: Sequence[float],
    desired_waypoints_rad: Sequence[Sequence[float]],
    *,
    joint_order: Sequence[int],
    joint_names: Sequence[str],
    encoder_step_rad: float,
    joint_max_delta_rad: Sequence[float],
    joint_max_speed_rad_s: Sequence[float],
    joint_max_acceleration_rad_s2: Sequence[float],
    joint_command_timeout_s: Sequence[float],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[tuple[float, ...], ...]]:
    """Expand logical 6-D waypoints into deterministic one-joint T101 steps."""

    current = list(_vec(start_positions_rad, 6, "quantized start"))
    steps: list[Mapping[str, Any]] = []
    emitted: list[tuple[float, ...]] = []
    for waypoint_index, raw_waypoint in enumerate(desired_waypoints_rad):
        desired = _vec(raw_waypoint, 6, f"desired_waypoints_rad[{waypoint_index}]")
        for joint_index in joint_order:
            target = desired[joint_index]
            if abs(target - current[joint_index]) <= 1e-15:
                continue
            previous = tuple(current)
            current[joint_index] = target
            post = tuple(current)
            delta = target - previous[joint_index]
            step: Mapping[str, Any] = {
                "sequence_index": len(steps),
                "desired_waypoint_index": waypoint_index,
                "previous_pose_rad": list(previous),
                "joint_index": joint_index,
                "joint_name": joint_names[joint_index],
                "t101_target_rad": target,
                "post_pose_rad": list(post),
                "delta_rad": delta,
                "encoder_step_rad": encoder_step_rad,
                "max_delta_rad": joint_max_delta_rad[joint_index],
                "max_speed_rad_s": joint_max_speed_rad_s[joint_index],
                "max_acceleration_rad_s2": joint_max_acceleration_rad_s2[joint_index],
                "timeout_s": joint_command_timeout_s[joint_index],
            }
            steps.append(step)
            emitted.append(post)
        if tuple(current) != desired:
            raise RouteBlocked("T101 expansion did not reach a desired quantized waypoint")
    if not steps:
        raise RouteBlocked("T101 expansion produced no atomic command")
    return tuple(steps), tuple(emitted)


def _command_sequence_payload(command_contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "command_mode",
        "driver_profile_release_id",
        "driver_profile_artifact_sha256",
        "joint_order",
        "joint_order_selection_policy",
        "joint_names",
        "encoder_step_rad",
        "joint_max_delta_rad",
        "joint_max_speed_rad_s",
        "joint_max_acceleration_rad_s2",
        "joint_command_timeout_s",
        "joint_limits_rad",
        "desired_waypoints_rad",
        "atomic_steps",
    )
    return {key: command_contract.get(key) for key in keys}


def _validate_t101_command_contract(
    command_contract: Mapping[str, Any],
    start_positions_rad: Sequence[float],
    emitted_setpoints_rad: Sequence[Sequence[float]],
) -> None:
    contract = _mapping(command_contract, "T101 command contract")
    if contract.get("schema") != "t101_atomic_command_contract_v1":
        raise RouteBlocked("T101 command contract schema is invalid")
    if contract.get("command_mode") != "T101_SINGLE_JOINT_ONLY":
        raise RouteBlocked("T101 command contract names another command mode")
    expected_hash = _require_hash(
        contract.get("command_sequence_sha256"), "T101 command_sequence_sha256"
    )
    if sha256_canonical(_command_sequence_payload(contract)) != expected_hash:
        raise RouteBlocked("T101 command sequence hash mismatch")

    joint_names_raw = contract.get("joint_names")
    if not isinstance(joint_names_raw, Sequence) or isinstance(joint_names_raw, (str, bytes)):
        raise RouteBlocked("T101 command joint_names are invalid")
    joint_names = tuple(str(name) for name in joint_names_raw)
    if len(joint_names) != 6 or any(not name for name in joint_names) or len(set(joint_names)) != 6:
        raise RouteBlocked("T101 command joint_names must be six unique names")
    joint_order = _released_joint_orders([contract.get("joint_order")])[0]
    encoder_step_rad = _finite(contract.get("encoder_step_rad"), "T101 encoder_step_rad")
    maximum_delta = _vec(contract.get("joint_max_delta_rad"), 6, "T101 joint_max_delta_rad")
    maximum_speed = _vec(contract.get("joint_max_speed_rad_s"), 6, "T101 joint_max_speed_rad_s")
    maximum_acceleration = _vec(
        contract.get("joint_max_acceleration_rad_s2"), 6, "T101 joint_max_acceleration_rad_s2"
    )
    timeouts = _vec(contract.get("joint_command_timeout_s"), 6, "T101 joint_command_timeout_s")
    joint_limits_raw = contract.get("joint_limits_rad")
    if not isinstance(joint_limits_raw, Sequence) or len(joint_limits_raw) != 6:
        raise RouteBlocked("T101 joint_limits_rad must contain six intervals")
    joint_limits = tuple(
        _vec(interval, 2, f"T101 joint_limits_rad[{index}]")
        for index, interval in enumerate(joint_limits_raw)
    )
    if encoder_step_rad <= 0.0 or any(value <= 0.0 for value in (*maximum_speed, *maximum_acceleration, *timeouts)):
        raise RouteBlocked("T101 encoder, speed, acceleration, and timeout constraints must be positive")

    steps_raw = contract.get("atomic_steps")
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)) or not steps_raw:
        raise RouteBlocked("T101 command contract has no atomic steps")
    if len(steps_raw) != len(emitted_setpoints_rad):
        raise RouteBlocked("T101 atomic step count does not match emitted setpoints")
    desired_raw = contract.get("desired_waypoints_rad")
    if not isinstance(desired_raw, Sequence) or isinstance(desired_raw, (str, bytes)) or not desired_raw:
        raise RouteBlocked("T101 desired waypoints are missing")
    desired_waypoints = tuple(
        _vec(waypoint, 6, f"T101 desired_waypoints_rad[{index}]")
        for index, waypoint in enumerate(desired_raw)
    )
    current = _vec(start_positions_rad, 6, "T101 start_positions_rad")
    previous_waypoint_index = -1
    previous_order_rank = -1
    order_rank = {joint_index: rank for rank, joint_index in enumerate(joint_order)}
    for sequence_index, (raw_step, raw_emitted) in enumerate(zip(steps_raw, emitted_setpoints_rad)):
        step = _mapping(raw_step, f"T101 atomic_steps[{sequence_index}]")
        if int(step.get("sequence_index", -1)) != sequence_index:
            raise RouteBlocked("T101 atomic sequence index is not contiguous")
        waypoint_index = int(step.get("desired_waypoint_index", -1))
        joint_index = int(step.get("joint_index", -1))
        if joint_index not in order_rank or waypoint_index < previous_waypoint_index:
            raise RouteBlocked("T101 atomic step order is invalid")
        rank = order_rank[joint_index]
        if waypoint_index == previous_waypoint_index and rank <= previous_order_rank:
            raise RouteBlocked("T101 atomic steps do not follow the selected fixed joint order")
        if waypoint_index != previous_waypoint_index:
            if previous_waypoint_index >= 0 and _distance(
                current, desired_waypoints[previous_waypoint_index]
            ) > 1e-12:
                raise RouteBlocked("T101 atomic group did not reach its desired waypoint")
            if waypoint_index != previous_waypoint_index + 1 or waypoint_index >= len(desired_waypoints):
                raise RouteBlocked("T101 desired waypoint indices are not contiguous")
            previous_order_rank = -1
        previous = _vec(step.get("previous_pose_rad"), 6, "T101 previous_pose_rad")
        post = _vec(step.get("post_pose_rad"), 6, "T101 post_pose_rad")
        emitted = _vec(raw_emitted, 6, "T101 emitted_setpoint")
        if _distance(previous, current) > 1e-12 or _distance(post, emitted) > 1e-12:
            raise RouteBlocked("T101 atomic pose chain is discontinuous")
        changed = [index for index, (before, after) in enumerate(zip(previous, post)) if abs(after - before) > 1e-12]
        if changed != [joint_index]:
            raise RouteBlocked("T101 atomic step changes other than exactly one joint")
        target = _finite(step.get("t101_target_rad"), "T101 target")
        if abs(target - post[joint_index]) > 1e-12:
            raise RouteBlocked("T101 scalar target does not match its post-pose")
        if step.get("joint_name") != joint_names[joint_index]:
            raise RouteBlocked("T101 joint index/name binding is invalid")
        if abs(target / encoder_step_rad - round(target / encoder_step_rad)) > 1e-9:
            raise RouteBlocked("T101 target is not on the released encoder grid")
        lower, upper = joint_limits[joint_index]
        if not lower <= target <= upper:
            raise RouteBlocked("T101 target is outside the released joint limits")
        delta = target - previous[joint_index]
        if abs(_finite(step.get("delta_rad"), "T101 delta_rad") - delta) > 1e-12:
            raise RouteBlocked("T101 recorded delta does not match its poses")
        if abs(delta) > maximum_delta[joint_index] - encoder_step_rad + 1e-12:
            raise RouteBlocked("T101 atomic delta lacks one encoder-step margin")
        constraints = (
            ("encoder_step_rad", encoder_step_rad),
            ("max_delta_rad", maximum_delta[joint_index]),
            ("max_speed_rad_s", maximum_speed[joint_index]),
            ("max_acceleration_rad_s2", maximum_acceleration[joint_index]),
            ("timeout_s", timeouts[joint_index]),
        )
        for field, expected in constraints:
            if abs(_finite(step.get(field), f"T101 {field}") - expected) > 1e-12:
                raise RouteBlocked(f"T101 step {field} does not match the released driver profile")
        current = post
        previous_waypoint_index = waypoint_index
        previous_order_rank = rank
    if previous_waypoint_index != len(desired_waypoints) - 1 or _distance(
        current, desired_waypoints[-1]
    ) > 1e-12:
        raise RouteBlocked("T101 sequence did not reach every desired waypoint")


def _diagnostic_artifact(manifest: Mapping[str, Any] | None, reason: str) -> dict[str, Any]:
    manifest_id = str((manifest or {}).get("manifest_id") or "unverified")
    result: dict[str, Any] = {
        "schema": DIAGNOSTIC_SCHEMA,
        "manifest_id": manifest_id,
        "status": "blocked_diagnostic_only",
        "blocked_reason": reason,
        "executable": False,
        "adapter_consumable": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "phases": [],
        "no_contact": True,
        "automatic_retry": False,
    }
    result["canonical_payload_sha256"] = sha256_canonical(result)
    return result


def build_outbound_route(
    manifest: Mapping[str, Any],
    *,
    ik_solver: IKSolver | None = None,
    kinematics_checker: KinematicsChecker | None = None,
    collision_checker: CollisionChecker | None = None,
    now_ns: int | None = None,
    diagnostic_on_error: bool = False,
) -> dict[str, Any]:
    """Build one released, no-contact outbound route or fail closed.

    The IK solver, quantized-endpoint kinematics checker, and collision
    checker are mandatory injected dependencies.  This keeps the module
    hardware/ROS-free, and makes the emitted (rather than raw IK) endpoint the
    one that must satisfy the released geometry.
    """

    try:
        releases, policy, manifest_hash = _validate_manifest(manifest)
        now = time.time_ns() if now_ns is None else int(now_ns)

        m0, m0_id, m0_hash = _release(releases, "m0_frozen")
        bundle, bundle_id, bundle_hash = _release(releases, "r6_bundle")
        tool, tool_id, tool_hash = _release(releases, "tool_release")
        standby, standby_id, standby_hash = _release(releases, "standby_release")
        urdf, urdf_id, urdf_hash = _release(releases, "urdf_release")
        driver, driver_id, driver_hash = _release(releases, "driver_profile_release")
        normal, normal_id, normal_hash = _release(releases, "normal_decision")
        collision, collision_id, collision_hash = _release(releases, "collision_model_release")

        if m0.get("schema") != "m0_frozen_target_release_v1" or m0.get("complete") is not True:
            raise RouteBlocked("M0 frozen evidence is not complete")
        if m0.get("immutable") is not True:
            raise RouteBlocked("M0 frozen evidence is mutable")
        if m0.get("frame_id") != EXPECTED_CAMERA_FRAME:
            raise RouteBlocked("M0 frame is not camera_optical_frame")
        source_stamp_ns = int(m0.get("source_stamp_ns") or 0)
        expiry_ns = int(m0.get("expires_at_ns") or 0)
        ttl_ms = _finite(m0.get("ttl_ms"), "m0.ttl_ms")
        if source_stamp_ns <= 0 or source_stamp_ns > now or ttl_ms <= 0.0 or ttl_ms > 1000.0:
            raise RouteBlocked("M0 timestamp or TTL is invalid")
        if expiry_ns <= now or expiry_ns > source_stamp_ns + int(ttl_ms * 1_000_000):
            raise RouteBlocked("M0 frozen evidence is expired or TTL-inconsistent")
        if _finite(m0.get("confidence"), "m0.confidence") < 0.8:
            raise RouteBlocked("M0 confidence is below release threshold")
        if _finite(m0.get("rgb_depth_skew_ms"), "m0.rgb_depth_skew_ms") > 50.0:
            raise RouteBlocked("M0 RGB/depth skew exceeds release threshold")
        target_id = str(m0.get("target_id") or "").strip()
        if not target_id:
            raise RouteBlocked("M0 target_id is required")
        detection_id = str(m0.get("detection_id") or "").strip()
        if not detection_id:
            raise RouteBlocked("M0 detection_id is required")

        if bundle.get("schema") != "camera_base_release_v1":
            raise RouteBlocked("r6 bundle release schema is invalid")
        if bundle.get("relative") is not False or bundle.get("topology_closed") is not True:
            raise RouteBlocked("relative or topology-unclosed camera bundle is not executable")
        if bundle.get("tf_status") != "released_execution":
            raise RouteBlocked("r6 bundle TF status is not released_execution")
        if m0.get("calibration_id") != bundle_id:
            raise RouteBlocked("M0 calibration id does not bind the r6 release")
        camera_translation = _vec(bundle.get("translation_m"), 3, "r6.translation_m")
        r_base_camera = _quat_to_matrix(bundle.get("rotation_xyzw"), "r6.rotation_xyzw")

        if tool.get("schema") != "accepted_tool_frame_release_v1":
            raise RouteBlocked("tool release schema is invalid")
        if tool.get("parent_frame") != EXPECTED_HAND_FRAME or tool.get("child_frame") != EXPECTED_CONTACT_FRAME:
            raise RouteBlocked("tool release is not hand_tcp -> press_contact")
        if tool.get("transform_complete") is not True or tool.get("contact_surface_accepted") is not True:
            raise RouteBlocked("tool frame is incomplete or unaccepted")
        if tool.get("rotation_status") != "accepted":
            raise RouteBlocked("tool rotation is not accepted")
        if _distance(_unit(tool.get("contact_axis_child"), "tool.contact_axis_child"), (1.0, 0.0, 0.0)) > 1e-9:
            raise RouteBlocked("tool release must define press_contact +X as the press axis")
        hand_contact_translation = _vec(tool.get("translation_m"), 3, "tool.translation_m")
        r_hand_contact = _quat_to_matrix(tool.get("rotation_xyzw"), "tool.rotation_xyzw")

        if standby.get("schema") != "standby_execution_release_v1" or standby.get("route_start_accepted") is not True:
            raise RouteBlocked("standby release is not accepted as a route start")
        if standby.get("all_joints_inside_limits") is not True or standby.get("corridor_clearance_accepted") is not True:
            raise RouteBlocked("standby limits or corridor clearance are unaccepted")

        if urdf.get("schema") != "urdf_release_v1" or not str(urdf.get("urdf_id") or ""):
            raise RouteBlocked("URDF release is invalid")
        if driver.get("schema") != "driver_profile_release_v1":
            raise RouteBlocked("driver profile release is invalid")
        encoder_step_rad = _finite(driver.get("encoder_step_rad"), "driver.encoder_step_rad")
        joint_max_delta_rad = _vec(driver.get("joint_max_delta_rad"), 6, "driver.joint_max_delta_rad")
        joint_max_speed_rad_s = _vec(
            driver.get("joint_max_speed_rad_s"), 6, "driver.joint_max_speed_rad_s"
        )
        joint_max_acceleration_rad_s2 = _vec(
            driver.get("joint_max_acceleration_rad_s2"),
            6,
            "driver.joint_max_acceleration_rad_s2",
        )
        joint_command_timeout_s = _vec(
            driver.get("joint_command_timeout_s"), 6, "driver.joint_command_timeout_s"
        )
        if any(
            value <= 0.0
            for value in (
                *joint_max_speed_rad_s,
                *joint_max_acceleration_rad_s2,
                *joint_command_timeout_s,
            )
        ):
            raise RouteBlocked("driver speed, acceleration, and timeout constraints must be positive")
        joint_names_raw = driver.get("joint_names")
        if not isinstance(joint_names_raw, Sequence) or isinstance(joint_names_raw, (str, bytes)):
            raise RouteBlocked("driver.joint_names must contain six unique names")
        joint_names = tuple(str(name).strip() for name in joint_names_raw)
        if len(joint_names) != 6 or any(not name for name in joint_names) or len(set(joint_names)) != 6:
            raise RouteBlocked("driver.joint_names must contain six unique names")
        joint_order_allowlist = _released_joint_orders(driver.get("joint_order_allowlist"))
        joint_limits_rad = driver.get("joint_limits_rad")
        if not isinstance(joint_limits_rad, Sequence) or len(joint_limits_rad) != 6:
            raise RouteBlocked("driver.joint_limits_rad must contain six intervals")
        start_positions = _vec(standby.get("joint_positions_rad"), 6, "standby.joint_positions_rad")

        if normal.get("schema") != "normal_decision_release_v1":
            raise RouteBlocked("normal decision release schema is invalid")
        if normal.get("target_id") != target_id or int(normal.get("source_stamp_ns") or 0) != source_stamp_ns:
            raise RouteBlocked("normal decision does not bind the frozen M0 target")
        if normal.get("status") != "outward_verified" or normal.get("ambiguous") is not False:
            raise RouteBlocked("normal is ambiguous, inward, or unreleased")
        if normal.get("input_normal_was_flipped") is not False:
            raise RouteBlocked("route builder never silently flips an inward normal")
        if _finite(normal.get("dot_n_out_p_minus_a"), "normal.dot_n_out_p_minus_a") <= 0.0:
            raise RouteBlocked("normal does not point from panel toward the verified tool")
        if _finite(normal.get("camera_side_dot"), "normal.camera_side_dot") <= 0.0:
            raise RouteBlocked("normal is not outward toward camera/cart")

        point_camera = _vec(m0.get("point_camera_m"), 3, "m0.point_camera_m")
        normal_camera = _unit(m0.get("normal_camera"), "m0.normal_camera")
        point_base = _add(_rotate(r_base_camera, point_camera), camera_translation)
        transformed_normal = _unit(_rotate(r_base_camera, normal_camera), "transformed normal")
        normal_out_base = _unit(normal.get("normal_out_base"), "normal.normal_out_base")
        if _dot(transformed_normal, normal_out_base) < math.cos(math.radians(0.5)):
            raise RouteBlocked("normal decision disagrees with the released camera transform")
        precontact_point = _add(point_base, _scale(normal_out_base, 0.050))

        roll_reference = _unit(policy.get("roll_reference_base"), "route_policy.roll_reference_base")
        r_base_contact = _desired_contact_rotation(normal_out_base, roll_reference)
        r_base_hand = _matmul(r_base_contact, _transpose(r_hand_contact))
        hand_target = _sub(precontact_point, _rotate(r_base_hand, hand_contact_translation))
        required_axis = _scale(normal_out_base, -1.0)
        request = IKRequest(
            hand_target_position_m=hand_target,
            hand_target_rotation_xyzw=_matrix_to_quat(r_base_hand),
            contact_target_position_m=precontact_point,
            contact_target_rotation_xyzw=_matrix_to_quat(r_base_contact),
            required_contact_axis_base=required_axis,
        )
        if ik_solver is None:
            raise RouteBlocked("no offline IK solver was injected")
        solution = _mapping(ik_solver(request), "IK solver result")
        if solution.get("status") != "converged":
            raise RouteBlocked("offline IK did not converge")
        endpoint_positions = _vec(solution.get("positions_rad"), 6, "IK.positions_rad")
        achieved_hand_position = _vec(solution.get("achieved_hand_position_m"), 3, "IK.achieved_hand_position_m")
        r_achieved_hand = _quat_to_matrix(solution.get("achieved_hand_rotation_xyzw"), "IK.achieved_hand_rotation_xyzw")
        position_tolerance_m = _finite(policy.get("position_tolerance_m"), "route_policy.position_tolerance_m")
        orientation_tolerance_deg = _finite(policy.get("orientation_tolerance_deg"), "route_policy.orientation_tolerance_deg")
        if _distance(achieved_hand_position, hand_target) > position_tolerance_m:
            raise RouteBlocked("IK achieved hand position exceeds tolerance")
        r_achieved_contact = _matmul(r_achieved_hand, r_hand_contact)
        achieved_contact = _add(achieved_hand_position, _rotate(r_achieved_hand, hand_contact_translation))
        if _distance(achieved_contact, precontact_point) > position_tolerance_m:
            raise RouteBlocked("IK achieved contact point exceeds tolerance")
        achieved_axis = _unit(_rotate(r_achieved_contact, (1.0, 0.0, 0.0)), "achieved contact axis")
        if _dot(achieved_axis, required_axis) < math.cos(math.radians(orientation_tolerance_deg)):
            raise RouteBlocked("IK contact axis does not oppose outward normal")
        if _rotation_angle_rad(r_achieved_contact, r_base_contact) > math.radians(orientation_tolerance_deg):
            raise RouteBlocked("IK contact orientation violates the released roll policy")

        desired_waypoints = _quantized_outbound(
            start_positions,
            endpoint_positions,
            encoder_step_rad=encoder_step_rad,
            joint_max_delta_rad=joint_max_delta_rad,
            joint_limits_rad=joint_limits_rad,
        )
        if kinematics_checker is None:
            raise RouteBlocked("no offline quantized-path kinematics checker was injected")
        if collision.get("schema") != "collision_model_release_v1" or collision.get("model_status") != "released":
            raise RouteBlocked("collision model is pending or unreleased")
        required_clearance_m = _finite(collision.get("required_minimum_clearance_m"), "collision.required_minimum_clearance_m")
        if collision_checker is None:
            raise RouteBlocked("no offline collision checker was injected")
        quantized_start = tuple(
            round(value / encoder_step_rad) * encoder_step_rad for value in start_positions
        )
        candidate_records: list[dict[str, Any]] = []
        eligible: list[
            tuple[
                float,
                tuple[int, ...],
                tuple[Mapping[str, Any], ...],
                tuple[tuple[float, ...], ...],
                tuple[tuple[float, ...], ...],
                tuple[Mapping[str, Any], ...],
                Mapping[str, Any],
            ]
        ] = []
        for joint_order in joint_order_allowlist:
            atomic_steps, atomic_emitted = _expand_t101_atomic_steps(
                quantized_start,
                desired_waypoints,
                joint_order=joint_order,
                joint_names=joint_names,
                encoder_step_rad=encoder_step_rad,
                joint_max_delta_rad=joint_max_delta_rad,
                joint_max_speed_rad_s=joint_max_speed_rad_s,
                joint_max_acceleration_rad_s2=joint_max_acceleration_rad_s2,
                joint_command_timeout_s=joint_command_timeout_s,
            )
            candidate_path = (quantized_start, *atomic_emitted)
            fk_evidence: list[Mapping[str, Any]] = []
            for path_index, pose in enumerate(candidate_path):
                fk_result = _mapping(
                    kinematics_checker(
                        FKRequest(
                            joint_positions_rad=pose,
                            path_index=path_index,
                            path_length=len(candidate_path),
                            joint_order=joint_order,
                            is_final=path_index == len(candidate_path) - 1,
                        )
                    ),
                    f"quantized-path kinematics result[{path_index}]",
                )
                if fk_result.get("status") != "validated":
                    raise RouteBlocked("offline FK did not validate every atomic path pose")
                if _distance(
                    _vec(fk_result.get("checked_positions_rad"), 6, "FK.checked_positions_rad"),
                    pose,
                ) > 1e-12:
                    raise RouteBlocked("offline FK evidence does not bind an atomic path pose")
                _vec(fk_result.get("achieved_hand_position_m"), 3, "FK.achieved_hand_position_m")
                _quat_to_matrix(
                    fk_result.get("achieved_hand_rotation_xyzw"),
                    "FK.achieved_hand_rotation_xyzw",
                )
                fk_evidence.append(dict(fk_result))

            final_fk = fk_evidence[-1]
            quantized_hand_position_candidate = _vec(
                final_fk.get("achieved_hand_position_m"), 3, "quantized endpoint hand position"
            )
            r_quantized_hand_candidate = _quat_to_matrix(
                final_fk.get("achieved_hand_rotation_xyzw"), "quantized endpoint hand rotation"
            )
            if _distance(quantized_hand_position_candidate, hand_target) > position_tolerance_m:
                raise RouteBlocked("quantized endpoint hand position exceeds tolerance")
            r_quantized_contact_candidate = _matmul(r_quantized_hand_candidate, r_hand_contact)
            quantized_contact_candidate = _add(
                quantized_hand_position_candidate,
                _rotate(r_quantized_hand_candidate, hand_contact_translation),
            )
            if _distance(quantized_contact_candidate, precontact_point) > position_tolerance_m:
                raise RouteBlocked("quantized endpoint contact point exceeds tolerance")
            quantized_axis_candidate = _unit(
                _rotate(r_quantized_contact_candidate, (1.0, 0.0, 0.0)),
                "quantized endpoint contact axis",
            )
            if _dot(quantized_axis_candidate, required_axis) < math.cos(
                math.radians(orientation_tolerance_deg)
            ):
                raise RouteBlocked("quantized endpoint contact axis does not oppose outward normal")
            if _rotation_angle_rad(r_quantized_contact_candidate, r_base_contact) > math.radians(
                orientation_tolerance_deg
            ):
                raise RouteBlocked("quantized endpoint orientation violates the released roll policy")

            collision_result = _mapping(
                collision_checker(
                    CollisionRequest(
                        quantized_start_setpoint_rad=quantized_start,
                        emitted_setpoints_rad=atomic_emitted,
                        path_setpoints_rad=candidate_path,
                        atomic_steps=atomic_steps,
                        joint_order=joint_order,
                        collision_model_sha256=collision_hash,
                        required_minimum_clearance_m=required_clearance_m,
                    )
                ),
                "collision checker result",
            )
            if collision_result.get("collision_model_sha256") != collision_hash:
                raise RouteBlocked("collision checker did not bind the released collision model")
            if int(collision_result.get("checked_waypoint_count") or 0) != len(candidate_path):
                raise RouteBlocked("collision evidence did not cover standby plus every atomic pose")
            if int(collision_result.get("segments_checked") or -1) != len(candidate_path) - 1:
                raise RouteBlocked("collision evidence did not cover every atomic swept segment")
            clearance = _finite(
                collision_result.get("minimum_clearance_m"), "collision.minimum_clearance_m"
            )
            is_clear = (
                collision_result.get("status") == "clear"
                and collision_result.get("continuous_corridor_status") == "clear"
                and collision_result.get("swept_segments_status") == "clear"
                and clearance >= required_clearance_m
            )
            candidate_record = {
                "joint_order": list(joint_order),
                "eligible": is_clear,
                "minimum_clearance_m": clearance,
                "path_sha256": sha256_canonical([list(point) for point in candidate_path]),
                "fk_path_evidence": [dict(item) for item in fk_evidence],
                "collision_evidence": dict(collision_result),
            }
            candidate_records.append(candidate_record)
            if is_clear:
                eligible.append(
                    (
                        clearance,
                        joint_order,
                        atomic_steps,
                        atomic_emitted,
                        candidate_path,
                        tuple(fk_evidence),
                        collision_result,
                    )
                )
        if not eligible:
            raise RouteBlocked("no allowlisted T101 joint order cleared FK/collision/swept checks")
        (
            achieved_clearance_m,
            selected_joint_order,
            atomic_steps,
            emitted,
            collision_path,
            selected_fk_evidence,
            collision_result,
        ) = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
        kinematics_result = selected_fk_evidence[-1]
        quantized_hand_position = _vec(
            kinematics_result.get("achieved_hand_position_m"), 3, "selected quantized hand position"
        )
        r_quantized_hand = _quat_to_matrix(
            kinematics_result.get("achieved_hand_rotation_xyzw"), "selected quantized hand rotation"
        )
        r_quantized_contact = _matmul(r_quantized_hand, r_hand_contact)
        quantized_contact = _add(
            quantized_hand_position, _rotate(r_quantized_hand, hand_contact_translation)
        )
        quantized_axis = _unit(
            _rotate(r_quantized_contact, (1.0, 0.0, 0.0)), "selected quantized contact axis"
        )

        command_contract: dict[str, Any] = {
            "schema": "t101_atomic_command_contract_v1",
            "command_mode": "T101_SINGLE_JOINT_ONLY",
            "driver_profile_release_id": driver_id,
            "driver_profile_artifact_sha256": driver_hash,
            "joint_order": list(selected_joint_order),
            "joint_order_selection_policy": "MAXIMUM_MINIMUM_CLEARANCE_THEN_LEXICOGRAPHIC",
            "joint_names": list(joint_names),
            "encoder_step_rad": encoder_step_rad,
            "joint_max_delta_rad": list(joint_max_delta_rad),
            "joint_max_speed_rad_s": list(joint_max_speed_rad_s),
            "joint_max_acceleration_rad_s2": list(joint_max_acceleration_rad_s2),
            "joint_command_timeout_s": list(joint_command_timeout_s),
            "joint_limits_rad": [list(_vec(interval, 2, "driver joint limit")) for interval in joint_limits_rad],
            "desired_waypoints_rad": [list(point) for point in desired_waypoints],
            "atomic_steps": [dict(step) for step in atomic_steps],
        }
        command_contract["command_sequence_sha256"] = sha256_canonical(
            _command_sequence_payload(command_contract)
        )
        command_contract["order_selection_evidence"] = candidate_records
        command_contract["selected_fk_path_evidence"] = [
            dict(item) for item in selected_fk_evidence
        ]
        _validate_t101_command_contract(command_contract, quantized_start, emitted)

        target_binding = {
            "target_id": target_id,
            "detection_id": detection_id,
            "source_stamp_ns": source_stamp_ns,
            "source_frame_id": m0["frame_id"],
            "m0_frozen_artifact_sha256": m0_hash,
            "calibration_release_id": bundle_id,
            "r6_bundle_artifact_sha256": bundle_hash,
        }
        collision_evidence = dict(collision_result)
        collision_evidence["quantized_path_setpoints_rad"] = [list(point) for point in collision_path]
        collision_evidence["command_contract"] = command_contract

        route_content = {
            "manifest_sha256": manifest_hash,
            "phase": PHASE,
            "target_binding": target_binding,
            "geometry": {
                "A_base_m": list(point_base),
                "N_out_base": list(normal_out_base),
                "P_pre_base_m": list(precontact_point),
            },
            "start_quantized_rad": list(quantized_start),
            "emitted_setpoints_rad": [list(point) for point in emitted],
            "precontact_contact_point_base_m": list(precontact_point),
            "precontact_distance_m": 0.050,
            "contact_rotation_xyzw": list(_matrix_to_quat(r_base_contact)),
            "quantized_endpoint_evidence": dict(kinematics_result),
            "collision_evidence": collision_evidence,
        }
        result: dict[str, Any] = {
            "schema": ROUTE_SCHEMA,
            "route_id": f"precontact-outbound-{sha256_canonical(route_content)[:16]}",
            "status": "offline_released_route",
            "phase": PHASE,
            "phases": [PHASE],
            "executable": True,
            "adapter_consumable": True,
            "required_execution_interface": "SAFE_ADAPTER_ACTION_SERVER",
            "motion_authorized": False,
            "contact_authorized": False,
            "press_authorized": False,
            "no_contact": True,
            "automatic_retry": False,
            "command_mode": "T101_SINGLE_JOINT_ONLY",
            "command_sequence_sha256": command_contract["command_sequence_sha256"],
            "route_hash": sha256_canonical(route_content),
            "target_binding": target_binding,
            "release_bindings": {
                "manifest_sha256": manifest_hash,
                "m0_frozen": {"release_id": m0_id, "artifact_sha256": m0_hash},
                "r6_bundle": {"release_id": bundle_id, "artifact_sha256": bundle_hash},
                "tool_release": {"release_id": tool_id, "artifact_sha256": tool_hash},
                "standby_release": {"release_id": standby_id, "artifact_sha256": standby_hash},
                "urdf_release": {"release_id": urdf_id, "artifact_sha256": urdf_hash},
                "driver_profile_release": {"release_id": driver_id, "artifact_sha256": driver_hash},
                "normal_decision": {"release_id": normal_id, "artifact_sha256": normal_hash},
                "collision_model_release": {"release_id": collision_id, "artifact_sha256": collision_hash},
            },
            "geometry": {
                "A_base_m": list(point_base),
                "N_out_base": list(normal_out_base),
                "P_pre_base_m": list(precontact_point),
                "target_surface_base_m": list(point_base),
                "normal_out_base": list(normal_out_base),
                "precontact_contact_point_base_m": list(precontact_point),
                "exact_precontact_distance_m": 0.050,
                "contact_rotation_xyzw": list(_matrix_to_quat(r_base_contact)),
                "contact_axis_base": list(required_axis),
                "hand_target_position_m": list(hand_target),
                "hand_target_rotation_xyzw": list(_matrix_to_quat(r_base_hand)),
                "achieved_contact_point_base_m": list(achieved_contact),
                "achieved_contact_axis_base": list(achieved_axis),
                "quantized_achieved_contact_point_base_m": list(quantized_contact),
                "quantized_achieved_contact_axis_base": list(quantized_axis),
            },
            "quantization": {
                "encoder_step_rad": encoder_step_rad,
                "delta_margin_encoder_steps": 1,
                "start_quantized_rad": route_content["start_quantized_rad"],
                "desired_waypoints_rad": [list(point) for point in desired_waypoints],
                "emitted_setpoints_rad": [list(point) for point in emitted],
            },
            "quantized_endpoint_evidence": dict(kinematics_result),
            "clearance_evidence": collision_evidence,
        }
        result["canonical_payload_sha256"] = canonical_route_artifact_hash(result)
        validate_outbound_route_artifact(result)
        return result
    except RouteBlocked as exc:
        if diagnostic_on_error:
            return _diagnostic_artifact(manifest, str(exc))
        raise


__all__ = [
    "CollisionRequest",
    "DIAGNOSTIC_SCHEMA",
    "FKRequest",
    "IKRequest",
    "MANIFEST_SCHEMA",
    "PHASE",
    "ROUTE_SCHEMA",
    "RouteBlocked",
    "build_outbound_route",
    "canonical_json",
    "canonical_route_artifact_hash",
    "seal_manifest",
    "seal_release",
    "sha256_canonical",
    "validate_outbound_route_artifact",
]
