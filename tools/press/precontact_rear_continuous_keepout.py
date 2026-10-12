#!/usr/bin/env python3
"""Fail-closed offline audit for the panel's rear half-space.

This is deliberately a *validator*, not a collision engine.  A separate
offline geometry checker must provide interval certificates for every robot
link and tool primitive on every joint-space segment.  Sampling FK poses alone
is not accepted as proof of continuous clearance.  The result is a static
receipt only and never provides motion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "rear_continuous_keepout_request_v1"
RELEASE_SCHEMA = "rear_continuous_keepout_static_release_v1"
MODE = "OFFLINE_STATIC_GEOMETRY"
METHOD = "interval_fk_link_capsules_v1"


class RearKeepoutBlocked(ValueError):
    """Evidence is incomplete, inconsistent, or cannot prove clearance."""


def canonical_sha256(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise RearKeepoutBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RearKeepoutBlocked(f"{field} must be an object")
    return value


def _sha(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise RearKeepoutBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise RearKeepoutBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RearKeepoutBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise RearKeepoutBlocked(f"{field} must be finite")
    return result


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise RearKeepoutBlocked(f"{field} must contain three values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _unit(value: Any, field: str) -> tuple[float, float, float]:
    vector = _vec3(value, field)
    magnitude = math.sqrt(sum(item * item for item in vector))
    if not 0.999 <= magnitude <= 1.001:
        raise RearKeepoutBlocked(f"{field} must be unit length")
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def seal_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add a deterministic binding hash to an offline request fixture."""
    result = dict(value)
    result.pop("request_sha256", None)
    result["request_sha256"] = canonical_sha256(result)
    return result


def _blocked(preview_hash: str, reasons: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "mode": MODE,
        "release_status": "blocked_static",
        "result": "blocked",
        "preview_hash": preview_hash,
        "method": METHOD,
        "checked_segment_count": 0,
        "checked_body_ids": [],
        "minimum_certified_clearance_m": None,
        "missing_release_reasons": sorted(set(str(reason) for reason in reasons)),
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "automatic_retry": False,
        "limitations": [
            "STATIC_CERTIFICATE_ONLY",
            "NO_ROS_OR_DRIVER_INTERFACE",
            "NO_MOTION_AUTHORITY",
        ],
    }
    result["canonical_payload_sha256"] = canonical_sha256(result)
    return result


def build_rear_continuous_keepout_release(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete interval-FK certificate or return a blocked receipt.

    No attempt is made to infer link dimensions, interpolate joint paths, or
    fill a gap from visual observations.  Any absent/invalid proof returns a
    hash-bound blocked receipt, so callers cannot accidentally promote a
    waypoint-only or synthetic check into a continuous collision claim.
    """
    try:
        value = dict(_mapping(request, "request"))
        required = {
            "schema", "mode", "preview_hash", "route_hash", "panel_plane",
            "minimum_front_clearance_m", "model", "body_ids", "segments", "request_sha256",
        }
        if set(value) != required:
            raise RearKeepoutBlocked("request fields must be exact")
        if value.get("schema") != REQUEST_SCHEMA or value.get("mode") != MODE:
            raise RearKeepoutBlocked("unexpected request schema or mode")
        preview_hash = _sha(value.get("preview_hash"), "preview_hash")
        _sha(value.get("route_hash"), "route_hash")
        supplied = _sha(value.get("request_sha256"), "request_sha256")
        unsigned = dict(value)
        unsigned.pop("request_sha256")
        if canonical_sha256(unsigned) != supplied:
            raise RearKeepoutBlocked("request hash mismatch")
        plane = _mapping(value.get("panel_plane"), "panel_plane")
        if set(plane) != {"frame_id", "point_base_m", "outward_normal_base"}:
            raise RearKeepoutBlocked("panel_plane fields must be exact")
        if plane.get("frame_id") != "roarm_base_link":
            raise RearKeepoutBlocked("panel plane must be in roarm_base_link")
        _vec3(plane.get("point_base_m"), "panel_plane.point_base_m")
        _unit(plane.get("outward_normal_base"), "panel_plane.outward_normal_base")
        required_clearance = _finite(value.get("minimum_front_clearance_m"), "minimum_front_clearance_m")
        if required_clearance < 0.0:
            raise RearKeepoutBlocked("minimum_front_clearance_m cannot be negative")
        model = _mapping(value.get("model"), "model")
        if set(model) != {"urdf_sha256", "collision_geometry_sha256", "tool_geometry_sha256", "model_status"}:
            raise RearKeepoutBlocked("model fields must be exact")
        for field in ("urdf_sha256", "collision_geometry_sha256", "tool_geometry_sha256"):
            _sha(model.get(field), f"model.{field}")
        if model.get("model_status") != "released_static":
            raise RearKeepoutBlocked("collision geometry model is not released_static")
        body_ids_raw = value.get("body_ids")
        if not isinstance(body_ids_raw, Sequence) or isinstance(body_ids_raw, (str, bytes)):
            raise RearKeepoutBlocked("body_ids must be a list")
        body_ids = tuple(str(item).strip() for item in body_ids_raw)
        if not body_ids or any(not item for item in body_ids) or len(set(body_ids)) != len(body_ids):
            raise RearKeepoutBlocked("body_ids must be non-empty and unique")
        segments_raw = value.get("segments")
        if not isinstance(segments_raw, Sequence) or isinstance(segments_raw, (str, bytes)) or not segments_raw:
            raise RearKeepoutBlocked("continuous segment certificates are required")
        min_clearance = math.inf
        checked: list[dict[str, Any]] = []
        for index, raw in enumerate(segments_raw):
            segment = _mapping(raw, f"segments[{index}]")
            expected = {"start_waypoint_index", "end_waypoint_index", "method", "covered_body_ids", "certified_minimum_signed_distance_m", "certificate_sha256"}
            if set(segment) != expected:
                raise RearKeepoutBlocked("segment fields must be exact")
            if int(segment.get("start_waypoint_index", -1)) != index or int(segment.get("end_waypoint_index", -1)) != index + 1:
                raise RearKeepoutBlocked("segments must certify each consecutive route interval")
            if segment.get("method") != METHOD:
                raise RearKeepoutBlocked("segment lacks interval FK/capsule certification")
            coverage = segment.get("covered_body_ids")
            if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes)) or tuple(coverage) != body_ids:
                raise RearKeepoutBlocked("every segment must cover the exact full robot/tool body set")
            clearance = _finite(segment.get("certified_minimum_signed_distance_m"), "certified_minimum_signed_distance_m")
            _sha(segment.get("certificate_sha256"), "segment certificate_sha256")
            if clearance < required_clearance:
                raise RearKeepoutBlocked("rear half-space clearance certificate is below required margin")
            min_clearance = min(min_clearance, clearance)
            checked.append({
                "start_waypoint_index": index,
                "end_waypoint_index": index + 1,
                "certificate_sha256": str(segment["certificate_sha256"]),
                "certified_minimum_signed_distance_m": clearance,
            })
        result: dict[str, Any] = {
            "schema": RELEASE_SCHEMA,
            "mode": MODE,
            "release_status": "released_static",
            "result": "pass",
            "release_id": f"rear-continuous-keepout-{supplied[:16]}",
            "artifact_sha256": supplied,
            "preview_hash": preview_hash,
            "route_hash": str(value["route_hash"]),
            "method": METHOD,
            "model": dict(model),
            "minimum_front_clearance_m": required_clearance,
            "checked_segment_count": len(checked),
            "checked_body_ids": list(body_ids),
            "minimum_certified_clearance_m": min_clearance,
            "segment_certificates": checked,
            "execution_ready": False,
            "motion_authorized": False,
            "contact_authorized": False,
            "automatic_retry": False,
            "limitations": ["STATIC_CERTIFICATE_ONLY", "NO_ROS_OR_DRIVER_INTERFACE", "NO_MOTION_AUTHORITY"],
        }
        result["canonical_payload_sha256"] = canonical_sha256(result)
        return result
    except RearKeepoutBlocked as exc:
        preview = request.get("preview_hash") if isinstance(request, Mapping) else ""
        return _blocked(str(preview) if len(str(preview)) == 64 else "0" * 64, [str(exc)])


__all__ = [
    "METHOD", "MODE", "RELEASE_SCHEMA", "REQUEST_SCHEMA", "RearKeepoutBlocked",
    "build_rear_continuous_keepout_release", "canonical_sha256", "seal_request",
]
