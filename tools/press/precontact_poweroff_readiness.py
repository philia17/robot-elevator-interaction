#!/usr/bin/env python3
"""Offline, power-off preparation validator for the Button 1 supervised trial.

This module validates the *already-projected* Button 1 evidence that will be
handed off for tomorrow's supervised no-motion trial: the canonical frozen M0
target, the sealed outbound-only T102 route, and (when present) the sealed
live projection record carrying A/N/P_pre.  It deliberately accepts the
projected A/N/Fn exactly as frozen; it does not investigate or correct wrist
visual drift.

The module is intentionally self-contained and stdlib-only.  It contains no
ROS or driver imports, no publisher/service/action interface, no process
launching, and no shell execution.  It never rewrites a hash and never
fabricates evidence: release gaps (route, keepout, calibration) are reported
as explicit blocked reasons, never as passed gates.  A built report is always
``BLOCKED`` and explicitly grants no motion authority; actual motion remains
gated on the tomorrow power-on checklist and an operator move ticket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


SCHEMA = "precontact_poweroff_readiness_report_v1"
MODE = "OFFLINE_POWEROFF_PREPARATION_ONLY"
DECISION = "PREPARED_POWEROFF_BLOCKED_NO_MOTION"
TRIAL_SCOPE = "SUPERVISED_NO_CONTACT_ONLY"
HANDOFF = "NO_MOTION_POWEROFF_HANDOFF_TOMORROW"

FROZEN_PACKAGE_SCHEMA = "m0_demo_freeze_v1"
ROUTE_SCHEMA = "t102_outbound_only_route_v1"
ROUTE_MODE = "OFFLINE_DIAGNOSTIC_ONLY"
ROUTE_PHASE = "OUTBOUND_TO_PRECONTACT"
EXPECTED_PRECONTACT_DISTANCE_M = 0.050
PRECONTACT_EQUATION = "P_pre = A + 0.050 * N_out"
DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S = 3600.0
EXPECTED_JOINT_COUNT = 6

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_FN_POINTER = (
    REPO_ROOT / "config/calibration/candidates/section_c_fingertip_c0_active_overlay_v1.json"
)
DEFAULT_R6_BUNDLE = (
    REPO_ROOT / "config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
)

FROZEN_BINDING_KEYS = frozenset(
    {"m0_target_sha256", "fn_sha256", "r6_sha256", "provenance_sha256"}
)
REQUIRED_RUNTIME_GATES = frozenset(
    {
        "INDEPENDENT_IK_FK_RELEASE",
        "JOINT_LIMIT_AND_CONTINUOUS_KEEPOUT_RELEASE",
        "FRESH_SAFE_DRIVER_FEEDBACK",
        "EXCLUSIVE_MOTION_OWNER",
        "EXPLICIT_OPERATOR_MOVE_TICKET",
    }
)

# Release gaps are explicit blockers.  There are no release receipts bound to
# the sealed artifacts, so every one of these must remain blocked.
RELEASE_BLOCKERS = (
    ("route_release", "ROUTE_RELEASE_MISSING"),
    ("ik_fk_release", "INDEPENDENT_IK_FK_RELEASE_MISSING"),
    ("joint_limits_release", "JOINT_LIMITS_STATIC_RELEASE_MISSING"),
    ("rear_keepout_release", "CONTINUOUS_REAR_KEEPOUT_RELEASE_MISSING"),
    ("field_collision_release", "FIELD_COLLISION_CERTIFICATE_NOT_RELEASED"),
)
CALIBRATION_BLOCKERS = (
    ("calibration_release", "SECTION_B_R6_RELATIVE_UNPUBLISHED"),
    ("fn_contact_calibration", "FN_CONTACT_CALIBRATION_UNRELEASED"),
)

POWER_ON_CHECKLIST = (
    {
        "id": "FRESH_FEEDBACK",
        "item": "Fresh feedback",
        "criterion": "current safe-driver feedback snapshot is fresh within the runtime age limit and parses cleanly",
    },
    {
        "id": "FAULT_EMPTY",
        "item": "Fault empty",
        "criterion": "feedback.fault is exactly empty",
    },
    {
        "id": "DISARMED",
        "item": "Disarmed",
        "criterion": "feedback.state == CONVERGED_DISARMED and armed == false",
    },
    {
        "id": "TORQUE_KNOWN",
        "item": "Torque known",
        "criterion": "torque_enabled is measured and known for all six joints",
    },
    {
        "id": "EXACT_STANDBY_MEASURED",
        "item": "Exact Standby confirmed from measured joints",
        "criterion": "measured joint_positions_rad match the recorded standby pose within tolerance",
    },
    {
        "id": "EXCLUSIVE_JOINT_TARGET_OWNERSHIP",
        "item": "Exclusive joint_target ownership",
        "criterion": "exactly one joint_target publisher; no foreign or competing joint_target owner",
    },
    {
        "id": "TARGET_SOURCE_FRESHNESS_IDENTITY",
        "item": "Target/source freshness and identity",
        "criterion": "fresh freeze or re-confirmed source stamp/frame/detection_id exactly match this sealed binding",
    },
    {
        "id": "TICKET_UNUSED_AND_TTL",
        "item": "Ticket unused and TTL",
        "criterion": "single-use operator move ticket is un-consumed and its TTL is still valid",
    },
    {
        "id": "ROUTE_OUTBOUND_ONLY",
        "item": "Route outbound-only",
        "criterion": "re-verified route_sha256 equals this sealed hash; phase OUTBOUND_TO_PRECONTACT; press/retract/return/retry all false",
    },
    {
        "id": "EXPLICIT_USER_CONFIRMATION",
        "item": "Explicit user confirmation",
        "criterion": "operator explicitly confirms SUPERVISED_NO_CONTACT_ONLY scope and the no-motion power-off handoff",
    },
)
POWER_ON_CHECKLIST_IDS = tuple(item["id"] for item in POWER_ON_CHECKLIST)

LIMITATIONS = (
    "OFFLINE_POWEROFF_PREPARATION_ONLY",
    "NO_ROS_OR_DRIVER_INTERFACE",
    "NO_PUBLISHER_SERVICE_OR_ACTION_INTERFACE",
    "NO_PROCESS_LAUNCH_OR_SHELL_EXECUTION",
    "TODAYS_REPORT_GRANTS_NO_MOTION_AUTHORITY",
    "ROUTE_KEEPOUT_AND_CALIBRATION_RELEASE_GAPS_REMAIN_BLOCKED",
    "WRIST_VISUAL_DRIFT_NOT_INVESTIGATED_ACCEPTED_AS_PROJECTED",
    "PROVENANCE_BINDING_SEALED_UNVERIFIABLE_OFFLINE",
)

NO_MOTION_AUTHORITY_STATEMENT = (
    "This power-off report validates the sealed Button 1 target, the "
    "outbound-only route, and the already-projected A/N/P_pre evidence only. "
    "It grants no motion authority: execution_ready and motion_authorized "
    "remain false, and motion stays gated on the tomorrow power-on checklist "
    "plus an explicit operator move ticket at runtime."
)


class PoweroffReadinessBlocked(ValueError):
    """A sealed input was malformed, tampered, or internally inconsistent."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PoweroffReadinessBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise PoweroffReadinessBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PoweroffReadinessBlocked(f"{field} must be an object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise PoweroffReadinessBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PoweroffReadinessBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise PoweroffReadinessBlocked(f"{field} must be finite")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise PoweroffReadinessBlocked(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PoweroffReadinessBlocked(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise PoweroffReadinessBlocked(f"{field} must be a positive integer")
    return result


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    """Accept either [x, y, z] or the frozen ``{x, y, z}`` mapping form."""
    if isinstance(value, Mapping):
        if set(value) != {"x", "y", "z"}:
            raise PoweroffReadinessBlocked(f"{field} must contain exactly x/y/z")
        components = [value["x"], value["y"], value["z"]]
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        components = list(value)
    else:
        raise PoweroffReadinessBlocked(f"{field} must contain exactly three finite values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(components))


def _vector6(value: Any, field: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != EXPECTED_JOINT_COUNT
    ):
        raise PoweroffReadinessBlocked(
            f"{field} must contain exactly {EXPECTED_JOINT_COUNT} finite values"
        )
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoweroffReadinessBlocked(f"{field} is not valid JSON: {exc}") from exc
    return _mapping(value, field)


def _required_button1_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("_", " ")
    if label not in {"1", "button 1"}:
        raise PoweroffReadinessBlocked("target identity must be Button 1")
    return "Button 1"


def _frozen_target_evidence(
    package: Mapping[str, Any], *, now_unix_ns: int, max_source_age_s: float
) -> dict[str, Any]:
    """Validate the frozen M0 package and bind its identity and receipts."""
    value = _mapping(package, "frozen_target_package")
    if value.get("schema") != FROZEN_PACKAGE_SCHEMA:
        raise PoweroffReadinessBlocked(
            f"frozen target package schema must be {FROZEN_PACKAGE_SCHEMA}"
        )
    if value.get("motion_authorized") is not False:
        raise PoweroffReadinessBlocked(
            "frozen target package motion_authorized must be false"
        )
    target_identity = _required_button1_label(value.get("target_label"))
    receipt = _sha256(
        value.get("canonical_payload_sha256"), "canonical_payload_sha256"
    )
    unsigned = dict(value)
    unsigned.pop("canonical_payload_sha256", None)
    if canonical_sha256(unsigned) != receipt:
        raise PoweroffReadinessBlocked(
            "frozen target package canonical payload hash mismatch"
        )
    full_package_sha256 = canonical_sha256(value)

    target = _mapping(value.get("frozen_target"), "frozen_target")
    if _required_button1_label(target.get("target_label")) != target_identity:
        raise PoweroffReadinessBlocked("frozen target Button 1 identity mismatch")
    frame_id = str(target.get("frame_id") or "").strip()
    stamp_ns = _positive_int(
        target.get("source_stamp_ns"), "frozen_target.source_stamp_ns"
    )
    detection_id = str(target.get("detection_id") or "").strip()
    if not frame_id:
        raise PoweroffReadinessBlocked("frozen target frame_id is missing")
    if not detection_id:
        raise PoweroffReadinessBlocked("frozen target detection_id is missing")
    header = target.get("header")
    if header is not None:
        header_value = _mapping(header, "frozen_target.header")
        if str(header_value.get("frame_id") or "").strip() != frame_id:
            raise PoweroffReadinessBlocked(
                "frozen target header frame_id mismatch"
            )
        if _positive_int(
            header_value.get("stamp_ns"), "frozen_target.header.stamp_ns"
        ) != stamp_ns:
            raise PoweroffReadinessBlocked(
                "frozen target header stamp_ns mismatch"
            )
    age_s = (now_unix_ns - stamp_ns) / 1_000_000_000
    if age_s < 0.0:
        raise PoweroffReadinessBlocked(
            "frozen target source stamp is in the future"
        )
    stale = age_s > max_source_age_s

    _vec3(target.get("camera_point_m"), "frozen_target.camera_point_m")
    surface = _mapping(
        target.get("surface_geometry"), "frozen_target.surface_geometry"
    )
    _vec3(
        surface.get("panel_outward_normal"),
        "frozen_target.surface_geometry.panel_outward_normal",
    )
    estimate_valid = surface.get("estimate_valid") is True
    provenance_complete = surface.get("provenance_complete") is True
    return {
        "schema": FROZEN_PACKAGE_SCHEMA,
        "target_identity": target_identity,
        "target_label": "1",
        "detection_id": detection_id,
        "source_stamp_ns": stamp_ns,
        "source_frame_id": frame_id,
        "source_age_s": age_s,
        "source_age_limit_s": max_source_age_s,
        "source_stale": stale,
        "canonical_payload_sha256": receipt,
        "full_package_sha256": full_package_sha256,
        "surface_normal_estimate_valid": estimate_valid,
        "provenance_complete": provenance_complete,
        "motion_authorized": False,
    }


def _route_evidence(route: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the sealed outbound-only route without touching its numbers."""
    value = _mapping(route, "sealed_route")
    if value.get("schema") != ROUTE_SCHEMA or value.get("route_mode") != ROUTE_MODE:
        raise PoweroffReadinessBlocked(
            "sealed route must be an outbound-only offline diagnostic route"
        )
    if value.get("phase") != ROUTE_PHASE or value.get("phases") != [ROUTE_PHASE]:
        raise PoweroffReadinessBlocked(
            "sealed route must contain exactly the outbound pre-contact phase"
        )
    if value.get("joint_path") != "SIMULTANEOUS_T102_JOINT_SPACE_CHORD":
        raise PoweroffReadinessBlocked(
            "sealed route is not a simultaneous joint-space chord"
        )
    if value.get("command_interface") != "NOT_IMPLEMENTED":
        raise PoweroffReadinessBlocked(
            "sealed route must not expose a command interface"
        )
    if value.get("diagnostic_only") is not True or value.get("no_contact") is not True:
        raise PoweroffReadinessBlocked(
            "sealed route must remain diagnostic-only and no-contact"
        )
    for field in (
        "press",
        "retract",
        "return",
        "automatic_retry",
        "execution_ready",
        "motion_authorized",
        "trajectory_generated",
        "command_generated",
        "adapter_consumable",
    ):
        if value.get(field) is not False:
            raise PoweroffReadinessBlocked(
                f"sealed route must keep {field}=false"
            )
    supplied = _sha256(value.get("route_sha256"), "route_sha256")
    unsigned = dict(value)
    unsigned.pop("route_sha256", None)
    if canonical_sha256(unsigned) != supplied:
        raise PoweroffReadinessBlocked("route SHA-256 mismatch")

    bindings = _mapping(value.get("frozen_bindings"), "frozen_bindings")
    if set(bindings) != FROZEN_BINDING_KEYS:
        raise PoweroffReadinessBlocked(
            "frozen_bindings must contain exactly M0/Fn/r6/provenance hashes"
        )
    for key in sorted(FROZEN_BINDING_KEYS):
        _sha256(bindings[key], f"frozen_bindings.{key}")
    _vector6(value.get("start_positions_rad"), "start_positions_rad")
    candidate = _vector6(
        value.get("static_ik_candidate_positions_rad"),
        "static_ik_candidate_positions_rad",
    )
    setpoints = value.get("setpoints_rad")
    if (
        not isinstance(setpoints, Sequence)
        or isinstance(setpoints, (str, bytes))
        or not setpoints
    ):
        raise PoweroffReadinessBlocked("sealed route has no bounded setpoints")
    for index, setpoint in enumerate(setpoints):
        _vector6(setpoint, f"setpoints_rad[{index}]")
    gates = value.get("required_runtime_gates")
    if not isinstance(gates, Sequence) or set(gates) != REQUIRED_RUNTIME_GATES:
        raise PoweroffReadinessBlocked(
            "sealed route runtime gates must be the reviewed gate set"
        )
    return {
        "schema": ROUTE_SCHEMA,
        "route_mode": ROUTE_MODE,
        "phase": ROUTE_PHASE,
        "phases": [ROUTE_PHASE],
        "joint_path": "SIMULTANEOUS_T102_JOINT_SPACE_CHORD",
        "command_interface": "NOT_IMPLEMENTED",
        "route_sha256": supplied,
        "route_hash_consistent": True,
        "diagnostic_only": True,
        "no_contact": True,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "command_generated": False,
        "adapter_consumable": False,
        "outbound_only": True,
        "required_runtime_gates": sorted(gates),
        "setpoint_count": len(setpoints),
        "proposed_precontact_joints_rad": list(candidate),
    }


def _geometry_evidence(
    geometry: Mapping[str, Any] | None,
    *,
    m0_full_sha256: str,
    route_sha256: str,
) -> dict[str, Any]:
    """Validate the already-projected A/N/P_pre record when it is present."""
    if geometry is None:
        return {"present": False, "reason": "LIVE_PROJECTION_GEOMETRY_EVIDENCE_MISSING"}
    value = _mapping(geometry, "geometry_evidence")
    supplied_target = _sha256(
        value.get("target_snapshot_sha256"),
        "geometry_evidence.target_snapshot_sha256",
    )
    if supplied_target != m0_full_sha256:
        raise PoweroffReadinessBlocked(
            "geometry evidence does not bind the frozen target"
        )
    supplied_route = _sha256(
        value.get("route_sha256"), "geometry_evidence.route_sha256"
    )
    if supplied_route != route_sha256:
        raise PoweroffReadinessBlocked(
            "geometry evidence does not bind the sealed route"
        )
    a = _vec3(value.get("A_base_m"), "geometry_evidence.A_base_m")
    n = _vec3(value.get("N_out_base"), "geometry_evidence.N_out_base")
    p_pre = _vec3(value.get("Fn_pre_base_m"), "geometry_evidence.Fn_pre_base_m")
    unit_length = math.sqrt(sum(component * component for component in n))
    if not math.isfinite(unit_length) or abs(unit_length - 1.0) > 1e-6:
        raise PoweroffReadinessBlocked(
            "N_out_base must be finite and unit length"
        )
    expected = tuple(
        a[index] + EXPECTED_PRECONTACT_DISTANCE_M * n[index] for index in range(3)
    )
    if _distance(p_pre, expected) > 1e-9:
        raise PoweroffReadinessBlocked(
            "P_pre must exactly equal A + 0.050*N_out"
        )
    return {
        "present": True,
        "precontact_equation": PRECONTACT_EQUATION,
        "precontact_distance_m": EXPECTED_PRECONTACT_DISTANCE_M,
        "A_base_m": list(a),
        "N_out_base": list(n),
        "P_pre_base_m": list(p_pre),
        "source_field": "Fn_pre_base_m",
        "normal_finite": True,
        "normal_normalized": True,
        "normal_unit_length": unit_length,
        "p_pre_matches_equation": True,
        "route_hash_consistent": True,
        "target_hash_consistent": True,
        "source": "sealed field-trial live projection",
    }


def _artifact_binding(
    artifact_path: str | Path, *, role: str, supplied_sha: str
) -> tuple[dict[str, Any], str | None]:
    """Bind a sealed hash to the current local artifact or report it missing."""
    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        return (
            {"role": role, "path": str(path), "verified": False},
            f"{role.upper()}_ARTIFACT_MISSING",
        )
    actual = _file_sha256(path)
    if actual != supplied_sha:
        raise PoweroffReadinessBlocked(
            f"{role} hash does not bind current artifact bytes"
        )
    return (
        {"role": role, "path": str(path), "verified": True, "sha256": actual},
        None,
    )


def build_poweroff_readiness_report(
    frozen_target_package: Mapping[str, Any],
    sealed_route: Mapping[str, Any],
    *,
    geometry_evidence: Mapping[str, Any] | None = None,
    fn_pointer_path: str | Path = ACTIVE_FN_POINTER,
    r6_bundle_path: str | Path = DEFAULT_R6_BUNDLE,
    now_unix_ns: int | None = None,
    max_source_age_s: float = DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S,
) -> dict[str, Any]:
    """Build the bounded power-off preparation report.

    The frozen M0 package, the sealed outbound-only route, and the optional
    already-projected A/N/P_pre record are validated exactly as sealed.  No
    hash is rewritten and no evidence is fabricated: every release gap is an
    explicit blocked reason and every authority flag stays false.
    """
    clock_ns = time.time_ns() if now_unix_ns is None else _positive_int(
        now_unix_ns, "now_unix_ns"
    )
    age_limit = _finite(max_source_age_s, "max_source_age_s")
    if age_limit <= 0.0:
        raise PoweroffReadinessBlocked(
            "max_source_age_s must be a positive finite number"
        )
    target = _frozen_target_evidence(
        frozen_target_package,
        now_unix_ns=clock_ns,
        max_source_age_s=age_limit,
    )
    route = _route_evidence(sealed_route)
    geometry = _geometry_evidence(
        geometry_evidence,
        m0_full_sha256=target["full_package_sha256"],
        route_sha256=route["route_sha256"],
    )

    bindings = _mapping(sealed_route.get("frozen_bindings"), "frozen_bindings")
    m0_binding = _sha256(
        bindings.get("m0_target_sha256"), "frozen_bindings.m0_target_sha256"
    )
    if m0_binding != target["full_package_sha256"]:
        raise PoweroffReadinessBlocked(
            "sealed route m0 binding does not match the frozen target"
        )
    fn_sha = _sha256(bindings.get("fn_sha256"), "frozen_bindings.fn_sha256")
    r6_sha = _sha256(bindings.get("r6_sha256"), "frozen_bindings.r6_sha256")
    provenance_sha = _sha256(
        bindings.get("provenance_sha256"), "frozen_bindings.provenance_sha256"
    )

    fn_binding, fn_missing = _artifact_binding(
        fn_pointer_path, role="fn_pointer", supplied_sha=fn_sha
    )
    r6_binding, r6_missing = _artifact_binding(
        r6_bundle_path, role="r6_bundle", supplied_sha=r6_sha
    )
    fn_status_changed = False
    r6_status_changed = False
    if fn_binding["verified"]:
        pointer = _load_json(fn_binding["path"], "active Fn pointer")
        if (
            pointer.get("execution_ready") is not False
            or pointer.get("publish_authorized") is not False
        ):
            raise PoweroffReadinessBlocked(
                "active Fn pointer must not grant execution or publication"
            )
        if pointer.get("status") != "diagnostic_overlay_default":
            fn_status_changed = True
        fn_binding["status"] = str(pointer.get("status") or "")
    if r6_binding["verified"]:
        bundle = _load_json(r6_binding["path"], "r6 bundle")
        if (
            bundle.get("execution_ready") is not False
            or bundle.get("publish_authorized") is not False
        ):
            raise PoweroffReadinessBlocked(
                "r6 bundle must remain relative and unpublished"
            )
        if bundle.get("status") != "b_section_accepted_relative_unpublished":
            r6_status_changed = True
        r6_binding["status"] = str(bundle.get("status") or "")
        r6_binding["execution_ready"] = False
        r6_binding["publish_authorized"] = False

    blocked: list[str] = [
        blocker for _role, blocker in RELEASE_BLOCKERS
    ] + [
        blocker for _role, blocker in CALIBRATION_BLOCKERS
    ] + [
        "PROVENANCE_BINDING_UNVERIFIABLE_OFFLINE",
        "RUNTIME_POWERON_CHECKLIST_PENDING",
        "OPERATOR_MOVE_TICKET_NOT_ISSUED",
        "EXPLICIT_USER_CONFIRMATION_PENDING",
    ]
    if target["source_stale"]:
        blocked.append("FROZEN_TARGET_SOURCE_STALE")
    if not target["surface_normal_estimate_valid"]:
        blocked.append("FROZEN_SURFACE_NORMAL_ESTIMATE_INVALID")
    if not target["provenance_complete"]:
        blocked.append("TARGET_PROVENANCE_INCOMPLETE")
    if not geometry["present"]:
        blocked.append("LIVE_PROJECTION_GEOMETRY_EVIDENCE_MISSING")
    if fn_missing:
        blocked.append(fn_missing)
    if fn_status_changed:
        blocked.append("FN_POINTER_STATUS_CHANGED")
    if r6_missing:
        blocked.append(r6_missing)
    if r6_status_changed:
        blocked.append("R6_STATUS_CHANGED")
    blocked = sorted(set(blocked))

    release_evidence = {
        role: {
            "present": False,
            "required": True,
            "blocker": blocker,
        }
        for role, blocker in RELEASE_BLOCKERS + CALIBRATION_BLOCKERS
    }
    checklist = [
        {"id": item["id"], "item": item["item"], "criterion": item["criterion"], "required": True}
        for item in POWER_ON_CHECKLIST
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": MODE,
        "decision": DECISION,
        "trial_scope": TRIAL_SCOPE,
        "handoff": HANDOFF,
        "no_motion_authority_statement": NO_MOTION_AUTHORITY_STATEMENT,
        "target_evidence": target,
        "route_evidence": route,
        "geometry_evidence": geometry,
        "frozen_bindings": {
            "m0_target_sha256": m0_binding,
            "fn_sha256": fn_sha,
            "r6_sha256": r6_sha,
            "provenance_sha256": provenance_sha,
            "fn_pointer": fn_binding,
            "r6_bundle": r6_binding,
            "provenance_sha256_verification": "SEALED_BINDING_UNVERIFIABLE_OFFLINE",
        },
        "release_evidence": release_evidence,
        "power_on_checklist": checklist,
        "blocked_reasons": blocked,
        "limitations": list(LIMITATIONS),
        "execution_ready": False,
        "motion_authorized": False,
        "arm_authorized": False,
        "command_generated": False,
        "adapter_consumable": False,
        "operator_ticket_issued": False,
        "contact_authorized": False,
        "automatic_retry": False,
        "report_grants_motion_authority": False,
    }
    report["canonical_report_sha256"] = canonical_sha256(report)
    validate_poweroff_readiness_report(report)
    return report


def validate_poweroff_readiness_report(report: Mapping[str, Any]) -> None:
    """Fail-closed structural validation of a built report."""
    value = _mapping(report, "report")
    expected = {
        "schema", "mode", "decision", "trial_scope", "handoff",
        "no_motion_authority_statement", "target_evidence", "route_evidence",
        "geometry_evidence", "frozen_bindings", "release_evidence",
        "power_on_checklist", "blocked_reasons", "limitations",
        "execution_ready", "motion_authorized", "arm_authorized",
        "command_generated", "adapter_consumable", "operator_ticket_issued",
        "contact_authorized", "automatic_retry",
        "report_grants_motion_authority", "canonical_report_sha256",
    }
    if set(value) != expected:
        raise PoweroffReadinessBlocked("report fields must be exact")
    if value.get("schema") != SCHEMA or value.get("mode") != MODE:
        raise PoweroffReadinessBlocked("unexpected report schema or mode")
    if value.get("decision") != DECISION:
        raise PoweroffReadinessBlocked(
            "report must remain prepared and blocked, never motion-ready"
        )
    if value.get("trial_scope") != TRIAL_SCOPE or value.get("handoff") != HANDOFF:
        raise PoweroffReadinessBlocked("report scope must remain supervised no-contact")
    for field in (
        "execution_ready", "motion_authorized", "arm_authorized",
        "command_generated", "adapter_consumable", "operator_ticket_issued",
        "contact_authorized", "automatic_retry",
    ):
        if value.get(field) is not False:
            raise PoweroffReadinessBlocked(f"report must keep {field}=false")
    if value.get("report_grants_motion_authority") is not False:
        raise PoweroffReadinessBlocked(
            "report must explicitly grant no motion authority"
        )
    required_blockers = {
        "ROUTE_RELEASE_MISSING",
        "CONTINUOUS_REAR_KEEPOUT_RELEASE_MISSING",
        "SECTION_B_R6_RELATIVE_UNPUBLISHED",
        "RUNTIME_POWERON_CHECKLIST_PENDING",
        "EXPLICIT_USER_CONFIRMATION_PENDING",
    }
    if not isinstance(value.get("blocked_reasons"), list):
        raise PoweroffReadinessBlocked("blocked_reasons must be a list")
    if not required_blockers.issubset(set(value["blocked_reasons"])):
        raise PoweroffReadinessBlocked(
            "route/keepout/calibration and runtime blockers are required"
        )
    checklist = value.get("power_on_checklist")
    if not isinstance(checklist, list):
        raise PoweroffReadinessBlocked("power_on_checklist must be a list")
    ids = [str(item.get("id") or "") for item in checklist]
    if ids != list(POWER_ON_CHECKLIST_IDS):
        raise PoweroffReadinessBlocked(
            "power_on_checklist must contain exactly the required items"
        )
    releases = _mapping(value.get("release_evidence"), "release_evidence")
    if set(releases) != {role for role, _blocker in RELEASE_BLOCKERS + CALIBRATION_BLOCKERS}:
        raise PoweroffReadinessBlocked("release evidence roles must be exact")
    for role, item in releases.items():
        record = _mapping(item, f"release_evidence.{role}")
        if record.get("present") is not False or record.get("required") is not True:
            raise PoweroffReadinessBlocked(
                f"release evidence {role} must remain an explicit blocker"
            )
    target = _mapping(value.get("target_evidence"), "target_evidence")
    if target.get("target_identity") != "Button 1":
        raise PoweroffReadinessBlocked("report must bind Button 1 identity")
    supplied = _sha256(
        value.get("canonical_report_sha256"), "canonical_report_sha256"
    )
    unsigned = dict(value)
    unsigned.pop("canonical_report_sha256", None)
    if canonical_sha256(unsigned) != supplied:
        raise PoweroffReadinessBlocked("report canonical hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-target", required=True, type=Path)
    parser.add_argument("--route", required=True, type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--fn-pointer", type=Path, default=ACTIVE_FN_POINTER)
    parser.add_argument("--r6-bundle", type=Path, default=DEFAULT_R6_BUNDLE)
    parser.add_argument("--now-ns", type=int)
    parser.add_argument(
        "--max-source-age-s",
        type=float,
        default=DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        package = _load_json(args.frozen_target, "frozen target")
        route = _load_json(args.route, "sealed route")
        geometry = (
            None
            if args.geometry is None
            else _load_json(args.geometry, "geometry evidence")
        )
        report = build_poweroff_readiness_report(
            package,
            route,
            geometry_evidence=geometry,
            fn_pointer_path=args.fn_pointer,
            r6_bundle_path=args.r6_bundle,
            now_unix_ns=args.now_ns,
            max_source_age_s=args.max_source_age_s,
        )
    except (OSError, json.JSONDecodeError, PoweroffReadinessBlocked) as exc:
        blocked = {
            "schema": SCHEMA,
            "mode": MODE,
            "decision": "REJECTED_POWEROFF_PREPARATION",
            "blocked": str(exc),
            "execution_ready": False,
            "motion_authorized": False,
            "report_grants_motion_authority": False,
        }
        encoded = json.dumps(blocked, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIVE_FN_POINTER",
    "DECISION",
    "DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S",
    "DEFAULT_R6_BUNDLE",
    "HANDOFF",
    "MODE",
    "NO_MOTION_AUTHORITY_STATEMENT",
    "POWER_ON_CHECKLIST",
    "POWER_ON_CHECKLIST_IDS",
    "PRECONTACT_EQUATION",
    "RELEASE_BLOCKERS",
    "ROUTE_PHASE",
    "ROUTE_SCHEMA",
    "SCHEMA",
    "TRIAL_SCOPE",
    "PoweroffReadinessBlocked",
    "build_poweroff_readiness_report",
    "canonical_sha256",
    "validate_poweroff_readiness_report",
]
