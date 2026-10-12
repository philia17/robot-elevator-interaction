#!/usr/bin/env python3
"""Offline, fail-closed readiness report for a direct Fn pre-contact proposal.

This module composes the existing display-only endpoint preview with *static*
release receipts.  It deliberately does not solve IK, sample a trajectory,
contact ROS, or create anything a driver could consume.  A passing receipt is
only evidence that a separately reviewed static check was recorded; this
report remains non-executable by construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCHEMA = "precontact_route_readiness_report_v1"
MODE = "OFFLINE_STATIC_REVIEW"
REQUIRED_RELEASES = {
    "ik_fk": "ik_fk_static_release_v1",
    "joint_limits": "joint_limits_static_release_v1",
    "rear_continuous_keepout": "rear_continuous_keepout_static_release_v1",
}


class RouteReadinessBlocked(ValueError):
    """A report input was malformed or tried to promote diagnostic evidence."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RouteReadinessBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise RouteReadinessBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteReadinessBlocked(f"{field} must be an object")
    return value


def _preview_hash(preview: Mapping[str, Any]) -> str:
    """Verify the existing preview receipt before binding a readiness report."""
    value = _mapping(preview, "preview")
    if value.get("schema") != "precontact_preview_v1":
        raise RouteReadinessBlocked("preview schema must be precontact_preview_v1")
    for field in ("execution_ready", "motion_authorized", "trajectory_generated", "ik_run"):
        if value.get(field) is not False:
            raise RouteReadinessBlocked(f"preview must keep {field}=false")
    supplied = _sha256(value.get("canonical_payload_sha256"), "preview canonical_payload_sha256")
    if value.get("preview_hash") != supplied:
        raise RouteReadinessBlocked("preview_hash must equal canonical_payload_sha256")
    unsigned = dict(value)
    for field in ("canonical_payload_sha256", "preview_hash", "preview_id"):
        unsigned.pop(field, None)
    if canonical_sha256(unsigned) != supplied:
        raise RouteReadinessBlocked("preview canonical payload hash mismatch")
    return supplied


def _release_status(
    name: str, release: Any, *, preview_hash: str
) -> tuple[dict[str, Any], str | None]:
    """Return a display-safe receipt summary and a fail-closed missing reason."""
    expected_schema = REQUIRED_RELEASES[name]
    if release is None:
        return {"present": False, "accepted": False}, f"{name.upper()}_STATIC_RELEASE_MISSING"
    try:
        item = _mapping(release, f"static_releases.{name}")
        required = {"schema", "release_id", "artifact_sha256", "preview_hash", "release_status", "result"}
        if set(item) != required:
            raise RouteReadinessBlocked("receipt fields must be exact")
        if item.get("schema") != expected_schema:
            raise RouteReadinessBlocked(f"schema must be {expected_schema}")
        if not str(item.get("release_id") or "").strip():
            raise RouteReadinessBlocked("release_id is required")
        artifact_hash = _sha256(item.get("artifact_sha256"), "artifact_sha256")
        if _sha256(item.get("preview_hash"), "release preview_hash") != preview_hash:
            raise RouteReadinessBlocked("release is not bound to this preview")
        if item.get("release_status") != "released_static" or item.get("result") != "pass":
            raise RouteReadinessBlocked("receipt is not a released static pass")
    except RouteReadinessBlocked as exc:
        return {
            "present": True,
            "accepted": False,
            "rejection": str(exc),
        }, f"{name.upper()}_STATIC_RELEASE_UNACCEPTED"
    return {
        "present": True,
        "accepted": True,
        "release_id": str(item["release_id"]),
        "artifact_sha256": artifact_hash,
        "preview_hash": preview_hash,
        "release_status": "released_static",
        "result": "pass",
    }, None


def build_route_readiness_report(
    preview: Mapping[str, Any], *, static_releases: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a canonical, non-executable route-readiness report.

    The static release receipts are intentionally narrow: their schema, hash,
    pass result, and preview binding are checked.  They cannot change the
    report's motion flags or stand in for runtime feedback/operator authority.
    """
    preview_hash = _preview_hash(preview)
    releases = {} if static_releases is None else dict(_mapping(static_releases, "static_releases"))
    unknown = set(releases) - set(REQUIRED_RELEASES)
    if unknown:
        raise RouteReadinessBlocked(f"unknown static release names: {sorted(unknown)}")

    evidence: dict[str, dict[str, Any]] = {}
    missing: list[str] = [
        "FN_REFERENCE_DIAGNOSTIC_ONLY",
        "R6_RELATIVE_UNPUBLISHED",
        "RUNTIME_FEEDBACK_AND_OPERATOR_RELEASE_REQUIRED",
    ]
    for name in REQUIRED_RELEASES:
        summary, reason = _release_status(name, releases.get(name), preview_hash=preview_hash)
        evidence[name] = summary
        if reason:
            missing.append(reason)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": MODE,
        "preview_hash": preview_hash,
        "preview_id": str(preview.get("preview_id") or ""),
        "target_identity": str(preview.get("target_identity") or ""),
        "precontact_fingertip_base_m": list(preview.get("precontact_fingertip_base_m") or []),
        "precontact_distance_m": preview.get("precontact_distance_m"),
        "static_release_evidence": evidence,
        "route_readiness": "BLOCKED_DIAGNOSTIC_OR_UNRELEASED",
        "missing_release_reasons": sorted(missing),
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "command_generated": False,
        "adapter_consumable": False,
        "contact_authorized": False,
        "automatic_retry": False,
        "operator_authorization_required": True,
        "limitations": [
            "STATIC_REVIEW_ONLY",
            "NO_IK_SOLVE_OR_FK_COMPUTATION_IN_THIS_MODULE",
            "NO_JOINT_SPACE_OR_CARTESIAN_TRAJECTORY",
            "NO_RUNTIME_COLLISION_OR_FEEDBACK_CHECK",
            "NO_ROS_OR_DRIVER_INTERFACE",
            "NO_MOTION_AUTHORITY",
        ],
    }
    report["canonical_report_sha256"] = canonical_sha256(report)
    validate_route_readiness_report(report, preview=preview)
    return report


def validate_route_readiness_report(
    report: Mapping[str, Any], *, preview: Mapping[str, Any] | None = None
) -> None:
    value = _mapping(report, "report")
    required = {
        "schema", "mode", "preview_hash", "preview_id", "target_identity",
        "precontact_fingertip_base_m", "precontact_distance_m", "static_release_evidence",
        "route_readiness", "missing_release_reasons", "execution_ready", "motion_authorized",
        "trajectory_generated", "command_generated", "adapter_consumable", "contact_authorized",
        "automatic_retry", "operator_authorization_required",
        "limitations", "canonical_report_sha256",
    }
    if set(value) != required:
        raise RouteReadinessBlocked("report fields must be exact")
    if value.get("schema") != SCHEMA or value.get("mode") != MODE:
        raise RouteReadinessBlocked("unexpected report schema or mode")
    for field in (
        "execution_ready", "motion_authorized", "trajectory_generated", "command_generated",
        "adapter_consumable", "contact_authorized", "automatic_retry",
    ):
        if value.get(field) is not False:
            raise RouteReadinessBlocked(f"report must keep {field}=false")
    if value.get("operator_authorization_required") is not True:
        raise RouteReadinessBlocked("operator_authorization_required must be true")
    if value.get("route_readiness") != "BLOCKED_DIAGNOSTIC_OR_UNRELEASED":
        raise RouteReadinessBlocked("report must remain blocked")
    if not isinstance(value.get("missing_release_reasons"), list) or not value["missing_release_reasons"]:
        raise RouteReadinessBlocked("missing release reasons are required")
    if "FN_REFERENCE_DIAGNOSTIC_ONLY" not in value["missing_release_reasons"]:
        raise RouteReadinessBlocked("diagnostic Fn blocker is required")
    supplied = _sha256(value.get("canonical_report_sha256"), "canonical_report_sha256")
    unsigned = dict(value)
    unsigned.pop("canonical_report_sha256", None)
    if canonical_sha256(unsigned) != supplied:
        raise RouteReadinessBlocked("report canonical hash mismatch")
    if preview is not None and value.get("preview_hash") != _preview_hash(preview):
        raise RouteReadinessBlocked("report does not bind supplied preview")


__all__ = [
    "MODE", "REQUIRED_RELEASES", "SCHEMA", "RouteReadinessBlocked", "build_route_readiness_report",
    "canonical_sha256", "validate_route_readiness_report",
]
