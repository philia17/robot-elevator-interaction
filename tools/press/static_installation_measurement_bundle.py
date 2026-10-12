#!/usr/bin/env python3
"""Offline, fail-closed validator/sealer for the Layer S static installation bundle.

This module seals and validates *measurement evidence only*.  It never grants
motion, contact, adapter, deployment, or execution authority, and it keeps
measurement completeness distinct from static release.  The current field
bundle remains ``blocked_static`` until every required static fact is measured
or conservatively bounded with uncertainty, provenance, and invalidation rules.

It does not import ROS, solve kinematics, publish commands, or interface with
any runtime.  Layer D (session-fresh) inputs are declared by contract only and
must not be mixed into Layer S.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA = "static_installation_measurement_bundle_v1"
RECEIPT_SCHEMA = "static_installation_measurement_bundle_receipt_v1"
MODE = "OFFLINE_STATIC_MEASUREMENT_EVIDENCE_ONLY"
HAND_TCP_FRAME = "hand_tcp"
BASE_FRAME = "roarm_base_link"
ACCEPTED_CLASSES = frozenset({"CURRENT_VERIFIED", "CURRENT_BOUNDED_CONSERVATIVE"})


class StaticBundleBlocked(ValueError):
    """The static installation bundle is incomplete, inconsistent, or unsafe."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StaticBundleBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def seal_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic bundle with its unsigned payload hash."""
    result = dict(value)
    result.pop("canonical_sha256", None)
    result["canonical_sha256"] = canonical_sha256(result)
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StaticBundleBlocked(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], field: str, expected: set[str]) -> None:
    if set(value) != expected:
        raise StaticBundleBlocked(f"{field} fields must be exact")


def _sha(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StaticBundleBlocked(f"{field} must be a lowercase SHA-256")
    return text


def _finite(value: Any, field: str, *, positive: bool = False, min_value: float | None = None) -> float:
    if isinstance(value, bool):
        raise StaticBundleBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StaticBundleBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise StaticBundleBlocked(f"{field} must be finite")
    if positive and result <= 0.0:
        raise StaticBundleBlocked(f"{field} must be positive")
    if min_value is not None and result < min_value:
        raise StaticBundleBlocked(f"{field} must be >= {min_value}")
    return result


def _vec3(value: Any, field: str, *, nullable: bool = False) -> list[float] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise StaticBundleBlocked(f"{field} must contain three finite values")
    return [_finite(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _quat(value: Any, field: str, *, nullable: bool = False) -> list[float] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise StaticBundleBlocked(f"{field} must contain four finite values")
    values = [_finite(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if not 0.999 <= math.sqrt(sum(item * item for item in values)) <= 1.001:
        raise StaticBundleBlocked(f"{field} must be unit length")
    return values


def _source(value: Any, field: str) -> None:
    source = _mapping(value, field)
    expected = {"kind", "synthetic", "artifact_sha256", "measurement_method"}
    _exact_keys(source, field, expected)
    kinds = {
        "operator_measurement", "repository_calibration", "vendor_datasheet",
        "field_observation", "synthetic_fixture", "unprovided",
    }
    if source.get("kind") not in kinds:
        raise StaticBundleBlocked(f"{field}.kind is unsupported")
    if not isinstance(source.get("synthetic"), bool):
        raise StaticBundleBlocked(f"{field}.synthetic must be boolean")
    _sha(source.get("artifact_sha256"), f"{field}.artifact_sha256", nullable=True)
    if not str(source.get("measurement_method") or "").strip():
        raise StaticBundleBlocked(f"{field}.measurement_method is required")


def _non_synthetic(source: Mapping[str, Any], reasons: list[str]) -> None:
    if source.get("synthetic") is True:
        reasons.append("SYNTHETIC_EVIDENCE_PROMOTED")


def _classification(value: Any, field: str) -> str:
    if value not in {"CURRENT_VERIFIED", "CURRENT_BOUNDED_CONSERVATIVE", "HISTORICAL_OR_INSTALLATION_UNPROVEN", "FIELD_OBSERVED_ONLY", "MISSING"}:
        raise StaticBundleBlocked(f"{field} has unsupported classification")
    return str(value)


def _installation(value: Any, reasons: list[str]) -> None:
    item = _mapping(value, "installation")
    _exact_keys(
        item, "installation",
        {"installation_id", "tool_id", "identity_proven", "identity_source_sha256", "invalidation_conditions"},
    )
    if not str(item.get("installation_id") or "").strip() or not str(item.get("tool_id") or "").strip():
        raise StaticBundleBlocked("installation_id and tool_id are required")
    if not isinstance(item.get("identity_proven"), bool):
        raise StaticBundleBlocked("installation.identity_proven must be boolean")
    if item.get("identity_proven") is not True:
        reasons.append("INSTALLATION_IDENTITY_NOT_PROVEN")
    _sha(item.get("identity_source_sha256"), "installation.identity_source_sha256", nullable=True)
    if not isinstance(item.get("invalidation_conditions"), Sequence) or isinstance(item.get("invalidation_conditions"), (str, bytes)):
        raise StaticBundleBlocked("installation.invalidation_conditions must be a list")


def _evidence_inventory(value: Any) -> None:
    items = value.get("evidence_inventory")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise StaticBundleBlocked("evidence_inventory must be a list")
    for index, raw in enumerate(items):
        item = _mapping(raw, f"evidence_inventory[{index}]")
        _exact_keys(item, f"evidence_inventory[{index}]", {"id", "path", "sha256", "classification", "role"})
        if not str(item.get("id") or "").strip():
            raise StaticBundleBlocked("evidence inventory id is required")
        _sha(item.get("sha256"), f"evidence_inventory[{index}].sha256", nullable=True)
        _classification(item.get("classification"), f"evidence_inventory[{index}].classification")


def _raw_operator_evidence(value: Any, reasons: list[str]) -> None:
    raw = _mapping(value.get("raw_operator_evidence"), "raw_operator_evidence")
    _exact_keys(
        raw, "raw_operator_evidence",
        {"received", "source_manifest_path", "source_manifest_sha256", "statements", "photos_with_scale"},
    )
    if not isinstance(raw.get("received"), bool):
        raise StaticBundleBlocked("raw_operator_evidence.received must be boolean")
    _sha(raw.get("source_manifest_sha256"), "raw_operator_evidence.source_manifest_sha256", nullable=True)
    statements = raw.get("statements")
    if not isinstance(statements, Sequence) or isinstance(statements, (str, bytes)):
        raise StaticBundleBlocked("raw_operator_evidence.statements must be a list")
    for index, statement in enumerate(statements):
        item = _mapping(statement, f"raw_operator_evidence.statements[{index}]")
        _exact_keys(item, f"raw_operator_evidence.statements[{index}]", {"fact_id", "statement", "status"})
        if not str(item.get("fact_id") or "").strip():
            raise StaticBundleBlocked("raw evidence fact_id is required")
    photos = raw.get("photos_with_scale")
    if not isinstance(photos, Sequence) or isinstance(photos, (str, bytes)):
        raise StaticBundleBlocked("raw_operator_evidence.photos_with_scale must be a list")
    for index, photo in enumerate(photos):
        item = _mapping(photo, f"raw_operator_evidence.photos_with_scale[{index}]")
        _exact_keys(item, f"raw_operator_evidence.photos_with_scale[{index}]", {"path", "sha256"})
        _sha(item.get("sha256"), f"raw_operator_evidence.photos_with_scale[{index}].sha256")
    if raw.get("received") is True and not statements:
        reasons.append("RAW_OPERATOR_EVIDENCE_DECLARED_BUT_EMPTY")


def _hand_tcp_to_tool(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("hand_tcp_to_tool"), "hand_tcp_to_tool")
    expected = {
        "present", "evidence_class", "source", "length_units", "angle_units",
        "translation_m", "quaternion_xyzw", "translation_uncertainty_m",
        "orientation_uncertainty_rad", "coaxial_with_hand_tcp_axis",
        "invalidation_conditions", "notes",
    }
    _exact_keys(item, "hand_tcp_to_tool", expected)
    if not isinstance(item.get("present"), bool):
        raise StaticBundleBlocked("hand_tcp_to_tool.present must be boolean")
    if item.get("length_units") != "m" or item.get("angle_units") != "rad":
        reasons.append("WRONG_UNITS")
    evidence_class = _classification(item.get("evidence_class"), "hand_tcp_to_tool.evidence_class")
    _source(item.get("source"), "hand_tcp_to_tool.source")
    translation = _vec3(item.get("translation_m"), "hand_tcp_to_tool.translation_m", nullable=True)
    quat = _quat(item.get("quaternion_xyzw"), "hand_tcp_to_tool.quaternion_xyzw", nullable=True)
    uncertainty = _vec3(item.get("translation_uncertainty_m"), "hand_tcp_to_tool.translation_uncertainty_m", nullable=True)
    orientation_unc = item.get("orientation_uncertainty_rad")
    if orientation_unc is not None:
        _finite(orientation_unc, "hand_tcp_to_tool.orientation_uncertainty_rad", min_value=0.0)
    if not isinstance(item.get("coaxial_with_hand_tcp_axis"), (bool, type(None))):
        raise StaticBundleBlocked("hand_tcp_to_tool.coaxial_with_hand_tcp_axis must be boolean or null")
    if not isinstance(item.get("invalidation_conditions"), Sequence) or isinstance(item.get("invalidation_conditions"), (str, bytes)):
        raise StaticBundleBlocked("hand_tcp_to_tool.invalidation_conditions must be a list")
    if item.get("present") is not True:
        reasons.append("HAND_TCP_TO_TOOL_TRANSFORM_MISSING")
        return
    if translation is None:
        reasons.append("HAND_TCP_TO_TOOL_TRANSLATION_MISSING")
    if quat is None:
        reasons.append("HAND_TCP_TO_TOOL_ORIENTATION_MISSING")
    if uncertainty is None:
        reasons.append("HAND_TCP_TO_TOOL_TRANSLATION_UNCERTAINTY_MISSING")
    if orientation_unc is None:
        reasons.append("HAND_TCP_TO_TOOL_ORIENTATION_UNCERTAINTY_MISSING")
    if evidence_class not in ACCEPTED_CLASSES:
        reasons.append("HAND_TCP_TO_TOOL_EVIDENCE_UNACCEPTED")
    _non_synthetic(item.get("source", {}), reasons)


def _capsule_geometry(value: Any, field: str) -> None:
    if value is None:
        return
    geometry = _mapping(value, field)
    _exact_keys(geometry, field, {"primitive", "endpoint_a_hand_tcp_m", "endpoint_b_hand_tcp_m"})
    if geometry.get("primitive") != "capsule":
        raise StaticBundleBlocked(f"{field}.primitive must be capsule")
    _vec3(geometry.get("endpoint_a_hand_tcp_m"), f"{field}.endpoint_a_hand_tcp_m")
    _vec3(geometry.get("endpoint_b_hand_tcp_m"), f"{field}.endpoint_b_hand_tcp_m")


def _part(value: Any, field: str, expected_kind: str, reasons: list[str]) -> None:
    item = _mapping(value, field)
    expected = {
        "present", "part_kind", "body_id", "evidence_class", "source",
        "length_units", "geometry", "conservative_radius_m", "uncertainty_m",
        "conservative_inflation_m", "invalidation_conditions", "notes",
    }
    _exact_keys(item, field, expected)
    if not isinstance(item.get("present"), bool):
        raise StaticBundleBlocked(f"{field}.present must be boolean")
    if item.get("part_kind") != expected_kind:
        raise StaticBundleBlocked(f"{field}.part_kind must be {expected_kind}")
    if item.get("length_units") != "m":
        reasons.append("WRONG_UNITS")
    evidence_class = _classification(item.get("evidence_class"), f"{field}.evidence_class")
    _source(item.get("source"), f"{field}.source")
    _capsule_geometry(item.get("geometry"), f"{field}.geometry")
    if item.get("conservative_radius_m") is not None:
        _finite(item.get("conservative_radius_m"), f"{field}.conservative_radius_m", positive=True)
    if item.get("uncertainty_m") is not None:
        _finite(item.get("uncertainty_m"), f"{field}.uncertainty_m", min_value=0.0)
    if item.get("conservative_inflation_m") is not None:
        _finite(item.get("conservative_inflation_m"), f"{field}.conservative_inflation_m", min_value=0.0)
    if not isinstance(item.get("invalidation_conditions"), Sequence) or isinstance(item.get("invalidation_conditions"), (str, bytes)):
        raise StaticBundleBlocked(f"{field}.invalidation_conditions must be a list")
    if item.get("present") is not True:
        reasons.append(f"{expected_kind.upper()}_GEOMETRY_MISSING")
        return
    if item.get("geometry") is None:
        reasons.append(f"{expected_kind.upper()}_GEOMETRY_MISSING")
    if item.get("conservative_radius_m") is None:
        reasons.append(f"{expected_kind.upper()}_RADIUS_MISSING")
    if item.get("uncertainty_m") is None:
        reasons.append(f"{expected_kind.upper()}_UNCERTAINTY_MISSING")
    if evidence_class not in ACCEPTED_CLASSES:
        reasons.append(f"{expected_kind.upper()}_EVIDENCE_UNACCEPTED")
    _non_synthetic(item.get("source", {}), reasons)


def _adapters(value: Any, reasons: list[str]) -> None:
    item = _mapping(value, "tool_parts.adapters")
    expected = {
        "present", "count", "evidence_class", "source", "length_units",
        "geometry", "uncertainty_m", "conservative_inflation_m",
        "invalidation_conditions", "notes",
    }
    _exact_keys(item, "tool_parts.adapters", expected)
    if not isinstance(item.get("present"), bool):
        raise StaticBundleBlocked("tool_parts.adapters.present must be boolean")
    if item.get("length_units") != "m":
        reasons.append("WRONG_UNITS")
    evidence_class = _classification(item.get("evidence_class"), "tool_parts.adapters.evidence_class")
    _source(item.get("source"), "tool_parts.adapters.source")
    geometry = item.get("geometry")
    if geometry is not None:
        geo = _mapping(geometry, "tool_parts.adapters.geometry")
        _exact_keys(geo, "tool_parts.adapters.geometry", {"primitives"})
        primitives = geo.get("primitives")
        if not isinstance(primitives, Sequence) or isinstance(primitives, (str, bytes)) or not primitives:
            raise StaticBundleBlocked("tool_parts.adapters.geometry.primitives must be a non-empty list")
        for index, raw in enumerate(primitives):
            prim = _mapping(raw, f"tool_parts.adapters.geometry.primitives[{index}]")
            _exact_keys(prim, f"tool_parts.adapters.geometry.primitives[{index}]", {"primitive", "endpoint_a_hand_tcp_m", "endpoint_b_hand_tcp_m", "conservative_radius_m"})
            if prim.get("primitive") != "capsule":
                raise StaticBundleBlocked("adapter primitive must be capsule")
            _vec3(prim.get("endpoint_a_hand_tcp_m"), "adapter endpoint_a_hand_tcp_m")
            _vec3(prim.get("endpoint_b_hand_tcp_m"), "adapter endpoint_b_hand_tcp_m")
            _finite(prim.get("conservative_radius_m"), "adapter conservative_radius_m", positive=True)
    if item.get("uncertainty_m") is not None:
        _finite(item.get("uncertainty_m"), "tool_parts.adapters.uncertainty_m", min_value=0.0)
    if item.get("conservative_inflation_m") is not None:
        _finite(item.get("conservative_inflation_m"), "tool_parts.adapters.conservative_inflation_m", min_value=0.0)
    if item.get("present") is True:
        if geometry is None:
            reasons.append("ADAPTER_GEOMETRY_MISSING")
        if item.get("count") is None or int(item.get("count")) < 1:
            reasons.append("ADAPTER_COUNT_INVALID")
        if item.get("uncertainty_m") is None:
            reasons.append("ADAPTER_UNCERTAINTY_MISSING")
        if evidence_class not in ACCEPTED_CLASSES:
            reasons.append("ADAPTER_EVIDENCE_UNACCEPTED")
        _non_synthetic(item.get("source", {}), reasons)
    elif item.get("count") not in (None, 0) or geometry is not None:
        reasons.append("ADAPTER_DECLARATION_INCONSISTENT")


def _cable(value: Any, reasons: list[str]) -> None:
    item = _mapping(value, "tool_parts.cable")
    expected = {
        "present", "fixed_configuration", "free_length_m", "bow_max_excursion_m",
        "permitted_bow_direction", "evidence_class", "source", "length_units",
        "geometry", "uncertainty_m", "conservative_inflation_m",
        "invalidation_conditions", "notes",
    }
    _exact_keys(item, "tool_parts.cable", expected)
    if not isinstance(item.get("present"), bool):
        raise StaticBundleBlocked("tool_parts.cable.present must be boolean")
    if not isinstance(item.get("fixed_configuration"), bool):
        raise StaticBundleBlocked("tool_parts.cable.fixed_configuration must be boolean")
    if item.get("length_units") != "m":
        reasons.append("WRONG_UNITS")
    evidence_class = _classification(item.get("evidence_class"), "tool_parts.cable.evidence_class")
    _source(item.get("source"), "tool_parts.cable.source")
    geometry = item.get("geometry")
    if geometry is not None:
        geo = _mapping(geometry, "tool_parts.cable.geometry")
        _exact_keys(geo, "tool_parts.cable.geometry", {"primitive", "endpoint_a_hand_tcp_m", "endpoint_b_hand_tcp_m", "route_description"})
        if geo.get("primitive") != "capsule":
            raise StaticBundleBlocked("cable primitive must be capsule")
        _vec3(geo.get("endpoint_a_hand_tcp_m"), "cable endpoint_a_hand_tcp_m")
        _vec3(geo.get("endpoint_b_hand_tcp_m"), "cable endpoint_b_hand_tcp_m")
        if not str(geo.get("route_description") or "").strip():
            raise StaticBundleBlocked("cable route_description is required")
    if item.get("free_length_m") is not None:
        _finite(item.get("free_length_m"), "tool_parts.cable.free_length_m", min_value=0.0)
    if item.get("bow_max_excursion_m") is not None:
        _finite(item.get("bow_max_excursion_m"), "tool_parts.cable.bow_max_excursion_m", min_value=0.0)
    if item.get("uncertainty_m") is not None:
        _finite(item.get("uncertainty_m"), "tool_parts.cable.uncertainty_m", min_value=0.0)
    if item.get("conservative_inflation_m") is not None:
        _finite(item.get("conservative_inflation_m"), "tool_parts.cable.conservative_inflation_m", min_value=0.0)
    if item.get("present") is not True:
        reasons.append("CABLE_GEOMETRY_MISSING")
        return
    if item.get("geometry") is None:
        reasons.append("CABLE_GEOMETRY_MISSING")
    if item.get("fixed_configuration") is not True:
        reasons.append("CABLE_ROUTING_UNBOUNDED")
    if item.get("bow_max_excursion_m") is None:
        reasons.append("CABLE_ROUTING_UNBOUNDED")
    if item.get("free_length_m") is None:
        reasons.append("CABLE_FREE_LENGTH_MISSING")
    if item.get("uncertainty_m") is None:
        reasons.append("CABLE_UNCERTAINTY_MISSING")
    if evidence_class not in ACCEPTED_CLASSES:
        reasons.append("CABLE_EVIDENCE_UNACCEPTED")
    _non_synthetic(item.get("source", {}), reasons)


def _pose(value: Any, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    pose = _mapping(value, field)
    _exact_keys(pose, field, {"translation_m", "quaternion_xyzw"})
    _vec3(pose.get("translation_m"), f"{field}.translation_m")
    _quat(pose.get("quaternion_xyzw"), f"{field}.quaternion_xyzw")


def _camera_bracket(value: Any, reasons: list[str]) -> None:
    item = _mapping(value, "rear_obstacles.camera_bracket")
    expected = {
        "present", "obstacle_id", "primitive", "frame_id", "evidence_class",
        "source", "length_units", "angle_units", "pose_base", "size_m",
        "translation_uncertainty_m", "orientation_uncertainty_rad",
        "invalidation_conditions", "notes",
    }
    _exact_keys(item, "rear_obstacles.camera_bracket", expected)
    if not isinstance(item.get("present"), bool):
        raise StaticBundleBlocked("rear_obstacles.camera_bracket.present must be boolean")
    if item.get("primitive") != "box":
        raise StaticBundleBlocked("camera bracket primitive must be box")
    if item.get("frame_id") != BASE_FRAME:
        reasons.append("REAR_CAMERA_BRACKET_NOT_BASE_FRAME")
    if item.get("length_units") != "m" or item.get("angle_units") != "rad":
        reasons.append("WRONG_UNITS")
    evidence_class = _classification(item.get("evidence_class"), "rear_obstacles.camera_bracket.evidence_class")
    _source(item.get("source"), "rear_obstacles.camera_bracket.source")
    _pose(item.get("pose_base"), "rear_obstacles.camera_bracket.pose_base", nullable=True)
    size = _vec3(item.get("size_m"), "rear_obstacles.camera_bracket.size_m", nullable=True)
    uncertainty = _vec3(item.get("translation_uncertainty_m"), "rear_obstacles.camera_bracket.translation_uncertainty_m", nullable=True)
    orientation_unc = item.get("orientation_uncertainty_rad")
    if orientation_unc is not None:
        _finite(orientation_unc, "rear_obstacles.camera_bracket.orientation_uncertainty_rad", min_value=0.0)
    if not isinstance(item.get("invalidation_conditions"), Sequence) or isinstance(item.get("invalidation_conditions"), (str, bytes)):
        raise StaticBundleBlocked("camera_bracket.invalidation_conditions must be a list")
    if item.get("present") is not True:
        reasons.append("REAR_CAMERA_BRACKET_MISSING")
        return
    if item.get("pose_base") is None:
        reasons.append("REAR_CAMERA_BRACKET_POSE_MISSING")
    if size is None:
        reasons.append("REAR_CAMERA_BRACKET_SIZE_MISSING")
    elif any(value <= 0.0 for value in size):
        raise StaticBundleBlocked("camera bracket size_m must be positive")
    if uncertainty is None:
        reasons.append("REAR_CAMERA_BRACKET_REGISTRATION_UNCERTAINTY_MISSING")
    if orientation_unc is None:
        reasons.append("REAR_CAMERA_BRACKET_REGISTRATION_UNCERTAINTY_MISSING")
    if item.get("frame_id") != BASE_FRAME:
        reasons.append("REAR_CAMERA_BRACKET_NOT_BASE_FRAME")
    if evidence_class not in ACCEPTED_CLASSES:
        reasons.append("REAR_CAMERA_BRACKET_EVIDENCE_UNACCEPTED")
    _non_synthetic(item.get("source", {}), reasons)


def _rear_obstacles(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("rear_obstacles"), "rear_obstacles")
    _exact_keys(item, "rear_obstacles", {"camera_bracket", "other_rear_structure_checked", "other_rear_obstacles"})
    if not isinstance(item.get("other_rear_structure_checked"), bool):
        raise StaticBundleBlocked("rear_obstacles.other_rear_structure_checked must be boolean")
    if item.get("other_rear_structure_checked") is not True:
        reasons.append("REAR_OTHER_STRUCTURE_NOT_CONFIRMED")
    others = item.get("other_rear_obstacles")
    if not isinstance(others, Sequence) or isinstance(others, (str, bytes)):
        raise StaticBundleBlocked("rear_obstacles.other_rear_obstacles must be a list")
    for index, raw in enumerate(others):
        obstacle = _mapping(raw, f"rear_obstacles.other_rear_obstacles[{index}]")
        expected = {
            "obstacle_id", "present", "primitive", "frame_id", "pose_base", "size_m",
            "translation_uncertainty_m", "orientation_uncertainty_rad",
            "source", "invalidation_conditions",
        }
        _exact_keys(obstacle, f"rear_obstacles.other_rear_obstacles[{index}]", expected)
        if obstacle.get("frame_id") != BASE_FRAME:
            reasons.append("REAR_OTHER_OBSTACLE_NOT_BASE_FRAME")
        _pose(obstacle.get("pose_base"), f"rear_obstacles.other_rear_obstacles[{index}].pose_base")
        _vec3(obstacle.get("size_m"), f"rear_obstacles.other_rear_obstacles[{index}].size_m")
        _vec3(obstacle.get("translation_uncertainty_m"), f"rear_obstacles.other_rear_obstacles[{index}].translation_uncertainty_m")
        _source(obstacle.get("source"), f"rear_obstacles.other_rear_obstacles[{index}].source")
    _camera_bracket(item.get("camera_bracket"), reasons)


def _robot_model(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("robot_model"), "robot_model")
    expected = {
        "urdf_or_xacro_path", "urdf_or_xacro_sha256", "collision_mesh_hashes",
        "body_ids", "identity_accepted", "model_status", "self_collision_policy",
    }
    _exact_keys(item, "robot_model", expected)
    if not str(item.get("urdf_or_xacro_path") or "").strip():
        raise StaticBundleBlocked("robot_model.urdf_or_xacro_path is required")
    _sha(item.get("urdf_or_xacro_sha256"), "robot_model.urdf_or_xacro_sha256", nullable=True)
    meshes = item.get("collision_mesh_hashes")
    if not isinstance(meshes, Sequence) or isinstance(meshes, (str, bytes)):
        raise StaticBundleBlocked("robot_model.collision_mesh_hashes must be a list")
    for index, mesh in enumerate(meshes):
        _sha(mesh, f"robot_model.collision_mesh_hashes[{index}]")
    body_ids = item.get("body_ids")
    if not isinstance(body_ids, Sequence) or isinstance(body_ids, (str, bytes)):
        raise StaticBundleBlocked("robot_model.body_ids must be a list")
    if not body_ids or any(not str(entry).strip() for entry in body_ids) or len(set(body_ids)) != len(body_ids):
        reasons.append("ROBOT_MODEL_BODY_SET_MISSING")
    if not isinstance(item.get("identity_accepted"), bool):
        raise StaticBundleBlocked("robot_model.identity_accepted must be boolean")
    if item.get("identity_accepted") is not True:
        reasons.append("ROBOT_MODEL_IDENTITY_NOT_ACCEPTED")
    if item.get("model_status") not in {"unreleased", "diagnostic_only", "released_static"}:
        raise StaticBundleBlocked("robot_model.model_status is unsupported")
    policy = item.get("self_collision_policy")
    if policy is not None:
        pol = _mapping(policy, "robot_model.self_collision_policy")
        _exact_keys(pol, "robot_model.self_collision_policy", {"source_path", "source_sha256", "reviewed"})
        _sha(pol.get("source_sha256"), "robot_model.self_collision_policy.source_sha256", nullable=True)
        if not isinstance(pol.get("reviewed"), bool):
            raise StaticBundleBlocked("self_collision_policy.reviewed must be boolean")


def _margin_policy(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("margin_policy"), "margin_policy")
    expected = {"status", "source_path", "source_sha256", "value_m", "non_authoritative_candidate_m", "notes"}
    _exact_keys(item, "margin_policy", expected)
    if item.get("status") not in {"MARGIN_POLICY_UNSET", "REVIEWED"}:
        raise StaticBundleBlocked("margin_policy.status is unsupported")
    _sha(item.get("source_sha256"), "margin_policy.source_sha256", nullable=True)
    if item.get("value_m") is not None:
        _finite(item.get("value_m"), "margin_policy.value_m", min_value=0.0)
    if item.get("non_authoritative_candidate_m") is not None:
        _finite(item.get("non_authoritative_candidate_m"), "margin_policy.non_authoritative_candidate_m", min_value=0.0)
    if item.get("status") != "REVIEWED":
        reasons.append("MARGIN_POLICY_UNSET")
    elif item.get("source_sha256") is None or item.get("value_m") is None:
        reasons.append("MARGIN_POLICY_SOURCE_INCOMPLETE")


def _layer_d_boundary(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("layer_d_boundary"), "layer_d_boundary")
    _exact_keys(item, "layer_d_boundary", {"statement", "session_fresh_inputs", "sealed_values_present"})
    if not str(item.get("statement") or "").strip():
        raise StaticBundleBlocked("layer_d_boundary.statement is required")
    inputs = item.get("session_fresh_inputs")
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)) or not inputs:
        raise StaticBundleBlocked("layer_d_boundary.session_fresh_inputs must be a non-empty list")
    if item.get("sealed_values_present") is not False:
        reasons.append("LAYER_D_STATIC_MIX")


def _authority(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("authority"), "authority")
    expected = {"execution_ready", "motion_authorized", "contact_authorized", "adapter_consumable", "deployment_authorized"}
    _exact_keys(item, "authority", expected)
    for name in expected:
        if item.get(name) is not False:
            reasons.append("AUTHORITY_FLAG_RAISED")


def _status(value: Any, reasons: list[str]) -> None:
    item = _mapping(value.get("status"), "status")
    expected = {"measurement_bundle_complete", "release_status", "review_status", "raw_operator_evidence_received", "blockers"}
    _exact_keys(item, "status", expected)
    if not isinstance(item.get("measurement_bundle_complete"), bool):
        raise StaticBundleBlocked("status.measurement_bundle_complete must be boolean")
    if item.get("release_status") != "blocked_static":
        reasons.append("RELEASE_STATUS_NOT_BLOCKED")
    if item.get("review_status") != "pending_independent_review":
        reasons.append("REVIEW_STATUS_NOT_PENDING")
    if not isinstance(item.get("raw_operator_evidence_received"), bool):
        raise StaticBundleBlocked("status.raw_operator_evidence_received must be boolean")
    blockers = item.get("blockers")
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        raise StaticBundleBlocked("status.blockers must be a list")


def validate_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a Layer S bundle and return a fail-closed static receipt.

    ``measurement_bundle_complete`` becomes true only when every required
    static input is present, bounded, hashed, and internally consistent.
    ``release_status`` remains ``blocked_static`` regardless of completeness,
    and every authority flag remains false.
    """
    reasons: list[str] = []
    canonical = ""
    try:
        bundle = dict(_mapping(value, "bundle"))
        expected_top = {
            "schema", "mode", "bundle_id", "created_at", "installation",
            "evidence_inventory", "raw_operator_evidence", "hand_tcp_to_tool",
            "tool_parts", "rear_obstacles", "robot_model", "margin_policy",
            "layer_d_boundary", "authority", "status", "canonical_sha256",
        }
        _exact_keys(bundle, "bundle", expected_top)
        if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("mode") != MODE:
            raise StaticBundleBlocked("unexpected bundle schema or mode")
        if not str(bundle.get("bundle_id") or "").strip() or not str(bundle.get("created_at") or "").strip():
            raise StaticBundleBlocked("bundle_id and created_at are required")
        supplied = _sha(bundle.get("canonical_sha256"), "canonical_sha256")
        unsigned = dict(bundle)
        unsigned.pop("canonical_sha256")
        if canonical_sha256(unsigned) != supplied:
            raise StaticBundleBlocked("bundle hash mismatch")
        canonical = supplied
        _installation(bundle.get("installation"), reasons)
        _evidence_inventory(bundle)
        _raw_operator_evidence(bundle, reasons)
        _hand_tcp_to_tool(bundle, reasons)
        parts = _mapping(bundle.get("tool_parts"), "tool_parts")
        _exact_keys(parts, "tool_parts", {"shaft", "soft_tip", "adapters", "cable"})
        _part(parts.get("shaft"), "tool_parts.shaft", "shaft", reasons)
        _part(parts.get("soft_tip"), "tool_parts.soft_tip", "soft_tip", reasons)
        _adapters(parts.get("adapters"), reasons)
        _cable(parts.get("cable"), reasons)
        _rear_obstacles(bundle, reasons)
        _robot_model(bundle, reasons)
        _margin_policy(bundle, reasons)
        _layer_d_boundary(bundle, reasons)
        _authority(bundle, reasons)
        _status(bundle, reasons)
        if bundle.get("status", {}).get("measurement_bundle_complete") is True and reasons:
            reasons.append("COMPLETENESS_FLAG_INCONSISTENT")
        if bundle.get("status", {}).get("measurement_bundle_complete") is False and not reasons:
            reasons.append("COMPLETENESS_FLAG_INCONSISTENT")
    except StaticBundleBlocked as exc:
        reasons.append(str(exc))
    complete = bool(reasons) is False
    result: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "mode": MODE,
        "result": "measurement_complete" if complete else "blocked",
        "measurement_bundle_complete": complete,
        "release_status": "blocked_static",
        "canonical_bundle_sha256": canonical if canonical else "0" * 64,
        "missing_reasons": sorted(set(reasons)),
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "adapter_consumable": False,
        "deployment_authorized": False,
        "limitations": [
            "STATIC_MEASUREMENT_EVIDENCE_ONLY",
            "NOT_A_CONTINUOUS_CLEARANCE_CERTIFICATE",
            "NOT_A_STATIC_RELEASE",
            "NO_ROS_OR_DRIVER_INTERFACE",
            "NO_MOTION_AUTHORITY",
        ],
    }
    result["canonical_payload_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "BASE_FRAME", "BUNDLE_SCHEMA", "HAND_TCP_FRAME", "MODE", "RECEIPT_SCHEMA",
    "StaticBundleBlocked", "canonical_sha256", "seal_bundle", "validate_bundle",
]
