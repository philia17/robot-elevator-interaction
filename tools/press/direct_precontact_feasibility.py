#!/usr/bin/env python3
"""Offline diagnostic feasibility report for direct Standby -> Fn pre-contact.

This module consumes a frozen M0 target (through :mod:`precontact_preview`),
the operator-selected active Fn pointer plus a *current* Fn-base observation,
and a read-only current-feedback/Standby record.  It computes the only
requested Cartesian endpoint, ``Fn_pre = A + 0.050 * N_out``.  It never runs
FK/IK, creates a trajectory, imports ROS, or exposes a control interface.

An accepted report therefore means only that the inputs are internally
consistent enough for offline review.  It is deliberately not an execution
ticket, an adapter input, or permission to move the robot.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from press.precontact_preview import (
    DEFAULT_BUNDLE,
    DEFAULT_REGISTRY,
    TransformError,
    build_precontact_preview,
)


SCHEMA = "direct_precontact_feasibility_report_v1"
MODE = "OFFLINE_NO_MOTION_DIAGNOSTIC"
ACTIVE_FN_POINTER = (
    Path(__file__).resolve().parents[2]
    / "config/calibration/candidates/section_c_fingertip_c0_active_overlay_v1.json"
)
EXPECTED_JOINT_COUNT = 6
STANDBY_MATCH_TOLERANCE_RAD = 0.05
MAX_EVIDENCE_SKEW_S = 3.0


class FeasibilityBlocked(ValueError):
    """An offline input is malformed, stale in identity, or internally inconsistent."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FeasibilityBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeasibilityBlocked(f"{field} must be an object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise FeasibilityBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FeasibilityBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise FeasibilityBlocked(f"{field} must be finite")
    return result


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise FeasibilityBlocked(f"{field} must contain exactly three finite values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _vector(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != EXPECTED_JOINT_COUNT:
        raise FeasibilityBlocked(f"{field} must contain exactly {EXPECTED_JOINT_COUNT} finite values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise FeasibilityBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _active_fn_observation(
    value: Mapping[str, Any], *, active_fn_pointer_path: str | Path
) -> dict[str, Any]:
    """Bind an observed base-frame Fn to the existing selected-pointer file.

    The pointer remains an operator-selected, translation-only diagnostic
    artifact.  Reading it here honours that selection without relabelling it
    as a released contact transform.
    """
    observation = _mapping(value, "active_fn_observation")
    required = {
        "schema", "source_stamp_ns", "active_artifact_id", "pointer_sha256",
        "translation_hand_tcp_m", "fn_base_m",
    }
    if set(observation) != required:
        raise FeasibilityBlocked("active_fn_observation fields must be exact")
    if observation.get("schema") != "active_fn_base_observation_v1":
        raise FeasibilityBlocked("active_fn_observation schema is invalid")
    source_stamp_ns = int(_finite(observation.get("source_stamp_ns"), "active_fn_observation.source_stamp_ns"))
    if source_stamp_ns <= 0:
        raise FeasibilityBlocked("active_fn_observation.source_stamp_ns must be positive")
    pointer_path = Path(active_fn_pointer_path).expanduser().resolve()
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeasibilityBlocked(f"active Fn pointer unreadable: {exc}") from exc
    pointer = _mapping(pointer, "active Fn pointer")
    if pointer.get("status") != "diagnostic_overlay_default":
        raise FeasibilityBlocked("active Fn pointer must remain the selected diagnostic overlay")
    supplied_hash = _sha256(observation.get("pointer_sha256"), "active_fn_observation.pointer_sha256")
    actual_hash = _file_sha256(pointer_path)
    if supplied_hash != actual_hash:
        raise FeasibilityBlocked("active Fn pointer hash does not bind current pointer file")
    if observation.get("active_artifact_id") != pointer.get("active_artifact_id"):
        raise FeasibilityBlocked("active Fn observation does not bind selected artifact id")
    translation = _vec3(observation.get("translation_hand_tcp_m"), "active_fn_observation.translation_hand_tcp_m")
    pointer_translation = _vec3(pointer.get("translation_m"), "active Fn pointer.translation_m")
    if _distance(translation, pointer_translation) > 1e-12:
        raise FeasibilityBlocked("active Fn observation translation does not bind selected pointer")
    return {
        "pointer_path": str(pointer_path),
        "pointer_sha256": actual_hash,
        "active_artifact_id": str(pointer["active_artifact_id"]),
        "translation_hand_tcp_m": list(translation),
        "fn_base_m": list(_vec3(observation.get("fn_base_m"), "active_fn_observation.fn_base_m")),
        "source_stamp_ns": source_stamp_ns,
        "pointer_status": "diagnostic_overlay_default",
    }


def _standby_consistency(
    current_feedback: Mapping[str, Any], standby_record: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare read-only feedback to a recorded Standby pose without authorizing it."""
    feedback = _mapping(current_feedback, "current_feedback")
    expected_feedback = {"schema", "source_stamp_ns", "state", "armed", "fault", "joint_names", "joint_positions_rad"}
    if set(feedback) != expected_feedback:
        raise FeasibilityBlocked("current_feedback fields must be exact")
    if feedback.get("schema") != "roarm_safe_feedback_snapshot_v1":
        raise FeasibilityBlocked("current_feedback schema is invalid")
    if int(_finite(feedback.get("source_stamp_ns"), "current_feedback.source_stamp_ns")) <= 0:
        raise FeasibilityBlocked("current_feedback.source_stamp_ns must be positive")
    if feedback.get("state") != "CONVERGED_DISARMED" or feedback.get("armed") is not False or feedback.get("fault") != "":
        raise FeasibilityBlocked("current feedback is not a clean CONVERGED_DISARMED standby state")
    joint_names = feedback.get("joint_names")
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)) or len(joint_names) != EXPECTED_JOINT_COUNT:
        raise FeasibilityBlocked("current_feedback.joint_names must name six joints")
    feedback_positions = _vector(feedback.get("joint_positions_rad"), "current_feedback.joint_positions_rad")

    standby = _mapping(standby_record, "standby_record")
    if standby.get("role") != "press_standby":
        raise FeasibilityBlocked("standby_record role must be press_standby")
    standby_names = standby.get("joint_names")
    if list(standby_names or []) != list(joint_names):
        raise FeasibilityBlocked("current feedback joints do not match standby record")
    standby_positions = _vector(standby.get("joint_positions_rad"), "standby_record.joint_positions_rad")
    residuals = [abs(actual - expected) for actual, expected in zip(feedback_positions, standby_positions)]
    maximum = max(residuals)
    return {
        "feedback_state": "CONVERGED_DISARMED",
        "feedback_armed": False,
        "feedback_source_stamp_ns": int(feedback["source_stamp_ns"]),
        "standby_pose_id": str(standby.get("pose_id") or ""),
        "joint_residual_rad": residuals,
        "maximum_joint_residual_rad": maximum,
        "matches_recorded_standby": maximum <= STANDBY_MATCH_TOLERANCE_RAD,
        "tolerance_rad": STANDBY_MATCH_TOLERANCE_RAD,
    }


def build_direct_precontact_feasibility_report(
    frozen_target_package: Mapping[str, Any],
    *,
    active_fn_observation: Mapping[str, Any],
    current_feedback: Mapping[str, Any],
    standby_record: Mapping[str, Any],
    bundle_path: str | Path = DEFAULT_BUNDLE,
    registry_path: str | Path = DEFAULT_REGISTRY,
    active_fn_pointer_path: str | Path = ACTIVE_FN_POINTER,
    now_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Build a direct-only diagnostic report from frozen no-motion evidence.

    ``build_precontact_preview`` provides the M0 + Section-B binding and the
    required exact endpoint.  This wrapper adds the selected current Fn and
    Standby feedback solely to make the remaining evidence gaps explicit.
    """
    try:
        preview = build_precontact_preview(
            frozen_target_package,
            bundle_path=bundle_path,
            registry_path=registry_path,
            now_unix_ns=now_unix_ns,
        )
    except TransformError as exc:
        raise FeasibilityBlocked(str(exc)) from exc
    fn = _active_fn_observation(active_fn_observation, active_fn_pointer_path=active_fn_pointer_path)
    standby = _standby_consistency(current_feedback, standby_record)
    endpoint = tuple(float(item) for item in preview["precontact_fingertip_base_m"])
    target = tuple(float(item) for item in preview["target_point_base_m"])
    normal = tuple(float(item) for item in preview["panel_outward_normal_base"])
    fn_current = tuple(float(item) for item in fn["fn_base_m"])
    missing = [
        "FN_CONTACT_ORIENTATION_UNMEASURED",
        "FN_CONTACT_SURFACE_AND_REPEATABILITY_UNRELEASED",
        "SECTION_B_PHYSICAL_XYZ_AND_OPTICAL_AXIS_UNRELEASED",
        "STATIC_IK_FK_RELEASE_MISSING",
        "JOINT_LIMITS_RELEASE_MISSING",
        "CONTINUOUS_REAR_KEEPOUT_RELEASE_MISSING",
        "RUNTIME_FEEDBACK_RECHECK_AND_OPERATOR_MOVE_TICKET_REQUIRED",
    ]
    if not standby["matches_recorded_standby"]:
        missing.append("CURRENT_FEEDBACK_DOES_NOT_MATCH_RECORDED_STANDBY")
    evidence_stamps_ns = {
        "m0_target": int(preview["source_header"]["stamp_ns"]),
        "active_fn": int(fn["source_stamp_ns"]),
        "current_feedback": int(standby["feedback_source_stamp_ns"]),
    }
    maximum_skew_s = (
        max(evidence_stamps_ns.values()) - min(evidence_stamps_ns.values())
    ) / 1_000_000_000
    if maximum_skew_s > MAX_EVIDENCE_SKEW_S:
        missing.append("M0_FN_AND_STANDBY_EVIDENCE_NOT_TIME_COHERENT")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": MODE,
        "decision": "DIAGNOSTIC_FEASIBILITY_ONLY_BLOCKED",
        "target_identity": preview["target_identity"],
        "m0_section_b_preview_hash": preview["preview_hash"],
        "A_base_m": list(target),
        "N_out_base": list(normal),
        "Fn_current_base_m": list(fn_current),
        "Fn_pre_base_m": list(endpoint),
        "precontact_equation": "Fn_pre = A + 0.050 * N_out",
        "precontact_distance_m": 0.05,
        "direct_fn_to_precontact_distance_m": _distance(fn_current, endpoint),
        "active_fn_evidence": fn,
        "standby_feedback_evidence": standby,
        "evidence_stamps_ns": evidence_stamps_ns,
        "maximum_evidence_skew_s": maximum_skew_s,
        "required_missing_fields": sorted(missing),
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "ik_run": False,
        "command_generated": False,
        "adapter_consumable": False,
        "execution_interface": "NOT_IMPLEMENTED",
        "limitations": [
            "NO_ROS_OR_DRIVER_INTERFACE",
            "NO_FK_OR_IK_SOLVE",
            "NO_JOINT_OR_CARTESIAN_TRAJECTORY",
            "NO_COLLISION_CLEARANCE_CLAIM",
            "ACTIVE_FN_POINTER_REMAINS_DIAGNOSTIC_ONLY",
            "NO_MOTION_AUTHORITY",
        ],
    }
    report["canonical_report_sha256"] = canonical_sha256(report)
    validate_direct_precontact_feasibility_report(report)
    return report


def validate_direct_precontact_feasibility_report(report: Mapping[str, Any]) -> None:
    value = _mapping(report, "report")
    expected = {
        "schema", "mode", "decision", "target_identity", "m0_section_b_preview_hash",
        "A_base_m", "N_out_base", "Fn_current_base_m", "Fn_pre_base_m", "precontact_equation",
        "precontact_distance_m", "direct_fn_to_precontact_distance_m", "active_fn_evidence",
        "standby_feedback_evidence", "evidence_stamps_ns", "maximum_evidence_skew_s",
        "required_missing_fields", "execution_ready", "motion_authorized",
        "trajectory_generated", "ik_run", "command_generated", "adapter_consumable", "execution_interface",
        "limitations", "canonical_report_sha256",
    }
    if set(value) != expected:
        raise FeasibilityBlocked("report fields must be exact")
    if value.get("schema") != SCHEMA or value.get("mode") != MODE:
        raise FeasibilityBlocked("report schema or mode is invalid")
    if value.get("decision") != "DIAGNOSTIC_FEASIBILITY_ONLY_BLOCKED":
        raise FeasibilityBlocked("report must remain diagnostic-only and blocked")
    for field in ("execution_ready", "motion_authorized", "trajectory_generated", "ik_run", "command_generated", "adapter_consumable"):
        if value.get(field) is not False:
            raise FeasibilityBlocked(f"report must keep {field}=false")
    if value.get("execution_interface") != "NOT_IMPLEMENTED":
        raise FeasibilityBlocked("report must not expose an execution interface")
    if value.get("precontact_equation") != "Fn_pre = A + 0.050 * N_out":
        raise FeasibilityBlocked("report precontact equation is invalid")
    a, n_out, fn_pre = (_vec3(value.get(key), key) for key in ("A_base_m", "N_out_base", "Fn_pre_base_m"))
    if abs(math.sqrt(sum(component * component for component in n_out)) - 1.0) > 1e-6:
        raise FeasibilityBlocked("N_out_base must be unit length")
    expected_fn_pre = tuple(a[index] + 0.05 * n_out[index] for index in range(3))
    if _distance(fn_pre, expected_fn_pre) > 1e-12:
        raise FeasibilityBlocked("Fn_pre_base_m must exactly equal A + 0.050*N_out")
    if abs(_finite(value.get("precontact_distance_m"), "precontact_distance_m") - 0.05) > 1e-12:
        raise FeasibilityBlocked("precontact_distance_m must be exactly 0.050")
    if not isinstance(value.get("required_missing_fields"), list) or not value["required_missing_fields"]:
        raise FeasibilityBlocked("required_missing_fields must be non-empty")
    if "FN_CONTACT_ORIENTATION_UNMEASURED" not in value["required_missing_fields"]:
        raise FeasibilityBlocked("Fn contact-orientation limitation is required")
    if _finite(value.get("maximum_evidence_skew_s"), "maximum_evidence_skew_s") < 0.0:
        raise FeasibilityBlocked("maximum_evidence_skew_s cannot be negative")
    supplied = _sha256(value.get("canonical_report_sha256"), "canonical_report_sha256")
    unsigned = dict(value)
    unsigned.pop("canonical_report_sha256", None)
    if canonical_sha256(unsigned) != supplied:
        raise FeasibilityBlocked("report canonical hash mismatch")


__all__ = [
    "ACTIVE_FN_POINTER", "FeasibilityBlocked", "MODE", "SCHEMA",
    "build_direct_precontact_feasibility_report", "canonical_sha256",
    "validate_direct_precontact_feasibility_report",
]
