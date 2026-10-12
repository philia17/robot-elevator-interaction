#!/usr/bin/env python3
"""Offline, fail-closed record validator for the installed press tool envelope.

This module defines *evidence*, not a collision checker or a robot interface.
Its only output is a hash-bound static receipt that a separate continuous
keep-out checker may bind into its ``tool_geometry_sha256`` model input.  It
does not import ROS, solve kinematics, publish commands, or grant motion
authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


RECORD_SCHEMA = "installed_tool_envelope_record_v1"
RECEIPT_SCHEMA = "installed_tool_envelope_static_receipt_v1"
MODE = "OFFLINE_STATIC_GEOMETRY"
BASE_FRAME = "roarm_base_link"
HAND_TCP_FRAME = "hand_tcp"
REQUIRED_PARTS = frozenset({"shaft", "soft_tip", "cable"})


class ToolEnvelopeBlocked(ValueError):
    """The installed-tool or rear-obstacle evidence is incomplete."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolEnvelopeBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def seal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic fixture with its unsigned payload hash."""
    result = dict(value)
    result.pop("record_sha256", None)
    result["record_sha256"] = canonical_sha256(result)
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolEnvelopeBlocked(f"{field} must be an object")
    return value


def _sha(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ToolEnvelopeBlocked(f"{field} must be a lowercase SHA-256")
    return text


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ToolEnvelopeBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolEnvelopeBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ToolEnvelopeBlocked(f"{field} must be {'positive' if positive else 'finite'}")
    return result


def _vec3(value: Any, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ToolEnvelopeBlocked(f"{field} must contain three finite values")
    return [_finite(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _pose6d(value: Any, field: str) -> None:
    pose = _mapping(value, field)
    if set(pose) != {"translation_m", "quaternion_xyzw"}:
        raise ToolEnvelopeBlocked(f"{field} must contain only translation_m and quaternion_xyzw")
    _vec3(pose.get("translation_m"), f"{field}.translation_m")
    quat = pose.get("quaternion_xyzw")
    if not isinstance(quat, Sequence) or isinstance(quat, (str, bytes)) or len(quat) != 4:
        raise ToolEnvelopeBlocked(f"{field}.quaternion_xyzw must contain four finite values")
    values = [_finite(item, f"{field}.quaternion_xyzw[{index}]") for index, item in enumerate(quat)]
    if not 0.999 <= math.sqrt(sum(item * item for item in values)) <= 1.001:
        raise ToolEnvelopeBlocked(f"{field}.quaternion_xyzw must be unit length")


def _part(value: Any, index: int) -> str:
    part = _mapping(value, f"tool_parts[{index}]")
    expected = {
        "body_id", "part_kind", "primitive", "frame_id", "attachment_pose_hand_tcp",
        "conservative_radius_m", "clearance_margin_m", "source_sha256", "geometry",
    }
    if set(part) != expected:
        raise ToolEnvelopeBlocked("tool part fields must be exact")
    kind = str(part.get("part_kind") or "")
    if kind not in REQUIRED_PARTS:
        raise ToolEnvelopeBlocked("tool parts must be shaft, soft_tip, or cable")
    if not str(part.get("body_id") or "").strip():
        raise ToolEnvelopeBlocked("tool part body_id is required")
    primitive = part.get("primitive")
    if primitive not in {"capsule", "mesh_reference"}:
        raise ToolEnvelopeBlocked("tool part primitive must be capsule or mesh_reference")
    if part.get("frame_id") != HAND_TCP_FRAME:
        raise ToolEnvelopeBlocked("tool parts must be expressed in hand_tcp")
    _pose6d(part.get("attachment_pose_hand_tcp"), "tool part attachment_pose_hand_tcp")
    _finite(part.get("conservative_radius_m"), "tool part conservative_radius_m", positive=True)
    _finite(part.get("clearance_margin_m"), "tool part clearance_margin_m")
    _sha(part.get("source_sha256"), "tool part source_sha256")
    geometry = _mapping(part.get("geometry"), "tool part geometry")
    if primitive == "capsule":
        if set(geometry) != {"endpoint_a_hand_tcp_m", "endpoint_b_hand_tcp_m"}:
            raise ToolEnvelopeBlocked("capsule geometry must contain two hand_tcp endpoints")
        _vec3(geometry.get("endpoint_a_hand_tcp_m"), "capsule endpoint_a_hand_tcp_m")
        _vec3(geometry.get("endpoint_b_hand_tcp_m"), "capsule endpoint_b_hand_tcp_m")
    else:
        if set(geometry) != {"mesh_uri", "mesh_sha256"}:
            raise ToolEnvelopeBlocked("mesh reference must contain mesh_uri and mesh_sha256")
        if not str(geometry.get("mesh_uri") or "").strip():
            raise ToolEnvelopeBlocked("mesh_uri is required")
        _sha(geometry.get("mesh_sha256"), "mesh_sha256")
    return kind


def _rear_obstacle(value: Any, index: int) -> None:
    obstacle = _mapping(value, f"rear_obstacles_base[{index}]")
    expected = {"obstacle_id", "primitive", "frame_id", "pose_base", "conservative_margin_m", "source_sha256", "geometry"}
    if set(obstacle) != expected:
        raise ToolEnvelopeBlocked("rear obstacle fields must be exact")
    if not str(obstacle.get("obstacle_id") or "").strip():
        raise ToolEnvelopeBlocked("rear obstacle_id is required")
    if obstacle.get("primitive") not in {"box", "capsule", "mesh_reference"}:
        raise ToolEnvelopeBlocked("rear obstacle primitive is unsupported")
    if obstacle.get("frame_id") != BASE_FRAME:
        raise ToolEnvelopeBlocked("rear obstacles must be expressed in roarm_base_link")
    _pose6d(obstacle.get("pose_base"), "rear obstacle pose_base")
    _finite(obstacle.get("conservative_margin_m"), "rear obstacle conservative_margin_m")
    _sha(obstacle.get("source_sha256"), "rear obstacle source_sha256")
    geometry = _mapping(obstacle.get("geometry"), "rear obstacle geometry")
    primitive = obstacle["primitive"]
    if primitive == "box":
        if set(geometry) != {"size_m"}:
            raise ToolEnvelopeBlocked("box obstacle geometry must contain size_m")
        values = _vec3(geometry.get("size_m"), "box size_m")
        if any(item <= 0.0 for item in values):
            raise ToolEnvelopeBlocked("box size_m must be positive")
    elif primitive == "capsule":
        if set(geometry) != {"radius_m", "length_m"}:
            raise ToolEnvelopeBlocked("capsule obstacle geometry must contain radius_m and length_m")
        _finite(geometry.get("radius_m"), "capsule obstacle radius_m", positive=True)
        _finite(geometry.get("length_m"), "capsule obstacle length_m", positive=True)
    else:
        if set(geometry) != {"mesh_uri", "mesh_sha256"}:
            raise ToolEnvelopeBlocked("mesh obstacle geometry must contain mesh_uri and mesh_sha256")
        if not str(geometry.get("mesh_uri") or "").strip():
            raise ToolEnvelopeBlocked("rear obstacle mesh_uri is required")
        _sha(geometry.get("mesh_sha256"), "rear obstacle mesh_sha256")


def validate_installed_tool_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an offline record and return a static-only receipt.

    A malformed, unknown, unreviewed, or deliberately blocked input always
    returns ``blocked_static``.  A static receipt is intentionally not motion
    authority and must still be checked by the continuous geometry validator.
    """
    unsigned_hash = "0" * 64
    try:
        value = dict(_mapping(record, "record"))
        expected = {
            "schema", "mode", "installation", "tool_parts", "rear_obstacles_base",
            "review", "record_sha256",
        }
        if set(value) != expected:
            raise ToolEnvelopeBlocked("record fields must be exact")
        if value.get("schema") != RECORD_SCHEMA or value.get("mode") != MODE:
            raise ToolEnvelopeBlocked("unexpected record schema or mode")
        supplied = _sha(value.get("record_sha256"), "record_sha256")
        unsigned = dict(value)
        unsigned.pop("record_sha256")
        if canonical_sha256(unsigned) != supplied:
            raise ToolEnvelopeBlocked("record hash mismatch")
        unsigned_hash = supplied
        installation = _mapping(value.get("installation"), "installation")
        expected_installation = {"installation_id", "tool_id", "parent_frame_id", "hand_tcp_attachment", "installation_source_sha256"}
        if set(installation) != expected_installation:
            raise ToolEnvelopeBlocked("installation fields must be exact")
        if not str(installation.get("installation_id") or "").strip() or not str(installation.get("tool_id") or "").strip():
            raise ToolEnvelopeBlocked("installation_id and tool_id are required")
        if installation.get("parent_frame_id") != HAND_TCP_FRAME:
            raise ToolEnvelopeBlocked("installed tool must attach to hand_tcp")
        _pose6d(installation.get("hand_tcp_attachment"), "installation hand_tcp_attachment")
        _sha(installation.get("installation_source_sha256"), "installation_source_sha256")
        parts = value.get("tool_parts")
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
            raise ToolEnvelopeBlocked("tool_parts must be a list")
        kinds = [_part(item, index) for index, item in enumerate(parts)]
        if set(kinds) != REQUIRED_PARTS or len(kinds) != len(REQUIRED_PARTS):
            raise ToolEnvelopeBlocked("shaft, soft_tip, and cable must each appear exactly once")
        obstacles = value.get("rear_obstacles_base")
        if not isinstance(obstacles, Sequence) or isinstance(obstacles, (str, bytes)) or not obstacles:
            raise ToolEnvelopeBlocked("at least one rear obstacle in base frame is required")
        for index, obstacle in enumerate(obstacles):
            _rear_obstacle(obstacle, index)
        review = _mapping(value.get("review"), "review")
        if set(review) != {"release_status", "review_record_sha256", "conservative_geometry_confirmed", "installation_identity_confirmed", "rear_obstacles_confirmed"}:
            raise ToolEnvelopeBlocked("review fields must be exact")
        if review.get("release_status") != "released_static":
            raise ToolEnvelopeBlocked("tool envelope remains unreleased_static")
        _sha(review.get("review_record_sha256"), "review_record_sha256")
        if not all(review.get(name) is True for name in ("conservative_geometry_confirmed", "installation_identity_confirmed", "rear_obstacles_confirmed")):
            raise ToolEnvelopeBlocked("static review confirmations are incomplete")
        return {
            "schema": RECEIPT_SCHEMA,
            "mode": MODE,
            "release_status": "released_static",
            "result": "pass",
            "artifact_sha256": unsigned_hash,
            "tool_geometry_sha256": unsigned_hash,
            "installation_id": str(installation["installation_id"]),
            "tool_id": str(installation["tool_id"]),
            "rear_obstacle_count": len(obstacles),
            "execution_ready": False,
            "motion_authorized": False,
            "contact_authorized": False,
            "limitations": ["STATIC_RECORD_ONLY", "REQUIRES_CONTINUOUS_GEOMETRY_CERTIFICATE", "NO_ROS_OR_DRIVER_INTERFACE", "NO_MOTION_AUTHORITY"],
        }
    except ToolEnvelopeBlocked as exc:
        return {
            "schema": RECEIPT_SCHEMA,
            "mode": MODE,
            "release_status": "blocked_static",
            "result": "blocked",
            "artifact_sha256": unsigned_hash,
            "tool_geometry_sha256": None,
            "missing_release_reasons": [str(exc)],
            "execution_ready": False,
            "motion_authorized": False,
            "contact_authorized": False,
            "limitations": ["STATIC_RECORD_ONLY", "NO_ROS_OR_DRIVER_INTERFACE", "NO_MOTION_AUTHORITY"],
        }


__all__ = [
    "BASE_FRAME", "HAND_TCP_FRAME", "MODE", "RECEIPT_SCHEMA", "RECORD_SCHEMA", "ToolEnvelopeBlocked",
    "canonical_sha256", "seal_record", "validate_installed_tool_envelope",
]
