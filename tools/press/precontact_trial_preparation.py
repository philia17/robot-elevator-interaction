#!/usr/bin/env python3
"""Prepare one sealed, no-motion Button 1 direct-precontact trial route.

This is the narrow bridge between fresh *read-only* observations and the
existing T102 diagnostic route builder.  It deliberately contains no ROS,
driver, service, action, publisher, process-launch, or deployment code.  In
particular, producing a preparation package neither arms the robot nor makes
the route executable: every authority flag remains false.

The caller supplies the raw M0 freeze package, the selected-Fn observation,
the current safe-driver feedback snapshot, and the recorded standby pose.  A
fresh M0/feedback/Fn package is checked by the existing feasibility contract;
the actual feedback joints (not a stale hard-coded standby vector) seed a
position-only static IK solve.  The resulting candidate is segmented by the
reviewed T102 delta limits and hash-bound to M0, Fn, r6 and the combined
diagnostic provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from press.direct_precontact_feasibility import (
    ACTIVE_FN_POINTER,
    FeasibilityBlocked,
    build_direct_precontact_feasibility_report,
    canonical_sha256,
)
from press.precontact_static_kinematics import (
    IK_POSITION_TOLERANCE_M,
    StaticKinematicsBlocked,
    _active_fn_translation,
    _distance,
    _load_urdf_joints,
    _sha256_file,
    fingertip_from_joints,
    solve_position_only_ik,
    URDF_PATH,
)
from press.t102_outbound_route_builder import (
    OBSERVED_FIELD_FAILURE_JOINT,
    OBSERVED_FIELD_FAILURE_RESIDUAL_RAD,
    T102RouteBlocked,
    build_t102_outbound_diagnostic_route,
    classify_t102_terminal_residual,
)


SCHEMA = "button1_supervised_trial_preparation_v1"
MODE = "OFFLINE_NO_MOTION_RUNTIME_PREPARATION"
# Mirror of the safe driver's 4096-step encoder grid (read-only reference).
ENCODER_STEP_RAD = 2.0 * math.pi / 4096.0
# The T102 executor binds the measured pose to the sealed route start within
# two encoder steps (the driver's own terminal-settle span).  This preparation
# package reports the same bound so an operator can see, before sealing,
# whether the fresh feedback pose and the recorded standby are interchangeable.
EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD = 2.0 * ENCODER_STEP_RAD
DEFAULT_R6_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
)


class TrialPreparationBlocked(ValueError):
    """A required observation or diagnostic dependency was invalid."""


def _file_sha256(path: str | Path, field: str) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return hashlib.sha256(resolved.read_bytes()).hexdigest()
    except OSError as exc:
        raise TrialPreparationBlocked(f"{field} unreadable: {exc}") from exc


def _canonical(value: Any) -> str:
    try:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TrialPreparationBlocked(f"value is not canonical JSON: {exc}") from exc


def _sha256(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise TrialPreparationBlocked(f"{field} must be a lowercase SHA-256")
    return result


def build_supervised_trial_preparation(
    frozen_target_package: Mapping[str, Any],
    *,
    active_fn_observation: Mapping[str, Any],
    current_feedback: Mapping[str, Any],
    standby_record: Mapping[str, Any],
    bundle_path: str | Path = DEFAULT_R6_BUNDLE,
    active_fn_pointer_path: str | Path = ACTIVE_FN_POINTER,
    now_unix_ns: int | None = None,
) -> dict[str, Any]:
    """Return a sealed three-segment outbound proposal with no motion authority.

    Exact inputs are the schemas already enforced by
    :func:`build_direct_precontact_feasibility_report` plus the r6 bundle path.
    ``current_feedback.joint_positions_rad`` is the only route start vector;
    this function refuses a feedback snapshot that is not clean, disarmed, and
    within the recorded standby tolerance.  M0/Fn/r6/provenance hashes are
    embedded in the resulting route for a future runtime to revalidate.
    The frozen M0 package must also carry a non-empty ``detection_id`` so the
    sealed target binding includes the exact detection identity, not only the
    camera point and stamp/frame.
    """
    frozen_target = frozen_target_package.get("frozen_target")
    if not isinstance(frozen_target, Mapping) or not str(frozen_target.get("detection_id") or "").strip():
        raise TrialPreparationBlocked(
            "frozen target package must bind a non-empty detection_id"
        )
    try:
        feasibility = build_direct_precontact_feasibility_report(
            frozen_target_package,
            active_fn_observation=active_fn_observation,
            current_feedback=current_feedback,
            standby_record=standby_record,
            bundle_path=bundle_path,
            active_fn_pointer_path=active_fn_pointer_path,
            now_unix_ns=now_unix_ns,
        )
    except FeasibilityBlocked as exc:
        raise TrialPreparationBlocked(str(exc)) from exc
    if not feasibility["standby_feedback_evidence"]["matches_recorded_standby"]:
        raise TrialPreparationBlocked("current feedback does not match recorded standby")

    # The feasibility contract has strictly validated the vector and the Fn
    # pointer binding.  Still read the local pointer again, so the static FK
    # uses exactly the bytes whose SHA is placed in the sealed route.
    try:
        fn_translation, fn_pointer = _active_fn_translation(active_fn_pointer_path)
        joints = _load_urdf_joints(URDF_PATH)
        start = tuple(float(value) for value in current_feedback["joint_positions_rad"])
        standby_positions = tuple(float(value) for value in standby_record["joint_positions_rad"])
        if len(start) != 6 or len(standby_positions) != 6:
            raise StaticKinematicsBlocked("feedback/standby joint vectors must have six values")
        target = tuple(float(value) for value in feasibility["Fn_pre_base_m"])
        candidate, ik = solve_position_only_ik(start, target, joints, fn_translation)
        start_fn, _ = fingertip_from_joints(start, joints, fn_translation)
        end_fn, _ = fingertip_from_joints(candidate, joints, fn_translation)
    except (KeyError, TypeError, ValueError, StaticKinematicsBlocked) as exc:
        raise TrialPreparationBlocked(f"static IK preparation failed: {exc}") from exc
    if not ik["converged"] or float(ik["final_position_error_mm"]) > 1000.0 * IK_POSITION_TOLERANCE_M:
        raise TrialPreparationBlocked("position-only static IK did not converge")

    r6_path = Path(bundle_path).expanduser().resolve()
    provenance = {
        "feasibility_report_sha256": feasibility["canonical_report_sha256"],
        "static_ik_method": ik["method"],
        "static_ik_final_error_mm": ik["final_position_error_mm"],
        "urdf_sha256": _sha256_file(URDF_PATH),
        "feedback_source_stamp_ns": feasibility["standby_feedback_evidence"]["feedback_source_stamp_ns"],
    }
    frozen_bindings = {
        "m0_target_sha256": _canonical(frozen_target_package),
        "fn_sha256": _sha256(fn_pointer["pointer_sha256"], "selected Fn pointer SHA"),
        "r6_sha256": _file_sha256(r6_path, "r6 bundle"),
        "provenance_sha256": _canonical(provenance),
    }
    feedback_to_standby_residual_rad = max(
        abs(actual - expected) for actual, expected in zip(start, standby_positions)
    )
    standby_binding_evidence = {
        "standby_pose_id": str(feasibility["standby_feedback_evidence"].get("standby_pose_id") or ""),
        "feedback_source_stamp_ns": feasibility["standby_feedback_evidence"]["feedback_source_stamp_ns"],
        "standby_positions_rad": list(standby_positions),
        "max_feedback_to_standby_residual_rad": feedback_to_standby_residual_rad,
        "max_feedback_to_standby_residual_deg": math.degrees(feedback_to_standby_residual_rad),
        "max_feedback_to_standby_residual_encoder_steps": feedback_to_standby_residual_rad / ENCODER_STEP_RAD,
        "executor_route_start_binding_tolerance_rad": EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD,
        "route_start_matches_standby_within_executor_binding": (
            feedback_to_standby_residual_rad <= EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD
        ),
        "note": (
            "the executor will require measured joints to match the sealed route "
            "start within two encoder steps; configure standby from the same fresh "
            "feedback used here, not a stale record"
        ),
    }
    try:
        route = build_t102_outbound_diagnostic_route(
            verified_start_positions_rad=start,
            verified_static_ik_candidate_rad=candidate,
            frozen_bindings=frozen_bindings,
        )
    except T102RouteBlocked as exc:
        raise TrialPreparationBlocked(str(exc)) from exc

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": MODE,
        "decision": "PREPARED_DIAGNOSTIC_ROUTE_ONLY",
        "input_contract": {
            "m0": "m0_demo_freeze_v1 frozen target package",
            "feedback": "roarm_safe_feedback_snapshot_v1 clean CONVERGED_DISARMED six-joint snapshot",
            "active_fn": "active_fn_base_observation_v1 bound to selected Fn pointer",
            "standby": "press_standby record with matching joint names and positions",
            "r6": str(r6_path),
        },
        "target_identity": feasibility["target_identity"],
        "A_base_m": feasibility["A_base_m"],
        "N_out_base": feasibility["N_out_base"],
        "Fn_pre_base_m": feasibility["Fn_pre_base_m"],
        "start_fn_base_m": list(start_fn),
        "end_fn_base_m": list(end_fn),
        "static_ik_evidence": {
            "method": ik["method"],
            "converged": ik["converged"],
            "final_position_error_mm": ik["final_position_error_mm"],
            "orientation_constrained": False,
            "contact_orientation_claimed": False,
        },
        "frozen_bindings": frozen_bindings,
        "provenance": provenance,
        "standby_binding_evidence": standby_binding_evidence,
        "route": route,
        "execution_ready": False,
        "motion_authorized": False,
        "arm_authorized": False,
        "command_generated": False,
        "adapter_consumable": False,
        "operator_ticket_issued": False,
        "physical_convergence_guaranteed": False,
        "physical_blocker": {
            "summary": (
                "The safe driver absorbed at most one stalled T102 reissue "
                "before faulting on the first field waypoint (elbow residual "
                "0.036815538 rad = 2.11 deg, just beyond its 2.0 deg "
                "soft-converge band).  Bounding commanded deltas into five "
                "chord segments does not reduce physical undertravel, so this "
                "preparation remains blocked until a driver-side or hardware "
                "fix is reviewed."
            ),
            "observed_field_failure_joint": OBSERVED_FIELD_FAILURE_JOINT,
            "observed_field_failure_max_residual_rad": OBSERVED_FIELD_FAILURE_RESIDUAL_RAD,
            "observed_field_failure_max_residual_deg": math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD),
            "driver_soft_converge_band_deg": route["driver_soft_converge_band_deg"],
            "observed_field_failure_exceeded_soft_band": True,
            "segmentation_alone_cannot_fix_undertravel": True,
            "execution_blocked": True,
        },
        "limitations": [
            "NO_ROS_OR_DRIVER_INTERFACE",
            "NO_PUBLISHER_SERVICE_OR_ACTION_INTERFACE",
            "NO_DEPLOYMENT_OR_HARDWARE_COMMAND",
            "POSITION_ONLY_IK_AND_SELECTED_FN_REMAIN_DIAGNOSTIC",
            "CONTINUOUS_PHYSICAL_COLLISION_CERTIFICATE_NOT_CLAIMED",
            "OFFLINE_ROUTE_CANNOT_GUARANTEE_PHYSICAL_CONVERGENCE",
            "FUTURE_RUNTIME_MUST_RECHECK_FRESH_FEEDBACK_EXCLUSIVE_OWNER_AND_OPERATOR_TICKET",
        ],
    }
    result["preparation_sha256"] = _canonical(result)
    validate_supervised_trial_preparation(result)
    return result


def validate_supervised_trial_preparation(value: Mapping[str, Any]) -> None:
    if value.get("schema") != SCHEMA or value.get("mode") != MODE:
        raise TrialPreparationBlocked("unexpected preparation schema or mode")
    if value.get("decision") != "PREPARED_DIAGNOSTIC_ROUTE_ONLY":
        raise TrialPreparationBlocked("preparation must remain diagnostic-only")
    for field in ("execution_ready", "motion_authorized", "arm_authorized", "command_generated", "adapter_consumable", "operator_ticket_issued"):
        if value.get(field) is not False:
            raise TrialPreparationBlocked(f"preparation must keep {field}=false")
    route = value.get("route")
    if not isinstance(route, Mapping):
        raise TrialPreparationBlocked("sealed route is missing")
    # The route's own validator is invoked by its builder.  These redundant
    # fields make it impossible to mistake the enclosing package for authority.
    if route.get("execution_ready") is not False or route.get("motion_authorized") is not False:
        raise TrialPreparationBlocked("sealed route must remain non-executable")
    bindings = value.get("frozen_bindings")
    if not isinstance(bindings, Mapping):
        raise TrialPreparationBlocked("frozen bindings are missing")
    if route.get("frozen_bindings") != bindings:
        raise TrialPreparationBlocked("route bindings differ from preparation bindings")
    for key in ("m0_target_sha256", "fn_sha256", "r6_sha256", "provenance_sha256"):
        _sha256(bindings.get(key), f"frozen_bindings.{key}")
    if value.get("physical_convergence_guaranteed") is not False:
        raise TrialPreparationBlocked("preparation must not guarantee physical convergence")
    blocker = value.get("physical_blocker")
    if not isinstance(blocker, Mapping):
        raise TrialPreparationBlocked("physical blocker evidence is missing")
    if blocker.get("execution_blocked") is not True:
        raise TrialPreparationBlocked("preparation must keep execution blocked")
    if blocker.get("segmentation_alone_cannot_fix_undertravel") is not True:
        raise TrialPreparationBlocked("preparation must not claim segmentation fixes undertravel")
    if blocker.get("observed_field_failure_exceeded_soft_band") is not True:
        raise TrialPreparationBlocked("observed field failure must exceed the soft band")
    try:
        blocker_residual = float(blocker.get("observed_field_failure_max_residual_rad"))
        blocker_residual_deg = float(blocker.get("observed_field_failure_max_residual_deg"))
        blocker_band = float(blocker.get("driver_soft_converge_band_deg"))
    except (TypeError, ValueError) as exc:
        raise TrialPreparationBlocked("physical blocker residual evidence is invalid") from exc
    if not math.isfinite(blocker_residual) or abs(blocker_residual - OBSERVED_FIELD_FAILURE_RESIDUAL_RAD) > 1e-12:
        raise TrialPreparationBlocked("physical blocker residual is inconsistent")
    if not math.isfinite(blocker_residual_deg) or abs(blocker_residual_deg - math.degrees(blocker_residual)) > 1e-9:
        raise TrialPreparationBlocked("physical blocker residual degrees are inconsistent")
    if not math.isfinite(blocker_band) or not blocker_residual_deg > blocker_band:
        raise TrialPreparationBlocked("physical blocker residual must exceed the driver soft band")
    if blocker.get("observed_field_failure_joint") != OBSERVED_FIELD_FAILURE_JOINT:
        raise TrialPreparationBlocked("physical blocker joint is inconsistent")
    if "OFFLINE_ROUTE_CANNOT_GUARANTEE_PHYSICAL_CONVERGENCE" not in value.get("limitations", []):
        raise TrialPreparationBlocked("preparation limitations must declare the physical blocker")
    route_band = value.get("route", {}).get("driver_soft_converge_band_deg")
    if not isinstance(route_band, (int, float)) or abs(float(route_band) - blocker_band) > 1e-9:
        raise TrialPreparationBlocked("physical blocker band differs from sealed route band")
    # The classification helper is exercised here so the sealed artifact and
    # its diagnostic tests agree on the driver-policy boundary.
    observed = classify_t102_terminal_residual(blocker_residual, float(route.get("driver_tolerance_deg")))
    if observed["within_soft_converge_band"] is not False or observed["driver_terminal_policy"] != "DRIVER_FAULTS_AFTER_SINGLE_REISSUE":
        raise TrialPreparationBlocked("physical blocker residual misclassified against driver policy")
    evidence = value.get("standby_binding_evidence")
    if not isinstance(evidence, Mapping):
        raise TrialPreparationBlocked("standby binding evidence is missing")
    route_start = route.get("start_positions_rad")
    if not isinstance(route_start, Sequence) or isinstance(route_start, (str, bytes)) or len(route_start) != 6:
        raise TrialPreparationBlocked("sealed route start must have six values")
    standby_positions = evidence.get("standby_positions_rad")
    if not isinstance(standby_positions, Sequence) or isinstance(standby_positions, (str, bytes)) or len(standby_positions) != 6:
        raise TrialPreparationBlocked("standby binding positions are missing")
    try:
        expected_residual = max(
            abs(float(a) - float(b)) for a, b in zip(route_start, standby_positions)
        )
    except (TypeError, ValueError) as exc:
        raise TrialPreparationBlocked("standby binding positions are invalid") from exc
    declared_residual = evidence.get("max_feedback_to_standby_residual_rad")
    if not isinstance(declared_residual, (int, float)) or not math.isfinite(float(declared_residual)):
        raise TrialPreparationBlocked("standby binding residual is invalid")
    if abs(float(declared_residual) - expected_residual) > 1e-12:
        raise TrialPreparationBlocked("standby binding residual does not match route start")
    if evidence.get("route_start_matches_standby_within_executor_binding") is not (
        expected_residual <= EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD
    ):
        raise TrialPreparationBlocked("standby binding flag is inconsistent")
    unsigned = dict(value)
    supplied = _sha256(unsigned.pop("preparation_sha256", None), "preparation_sha256")
    if _canonical(unsigned) != supplied:
        raise TrialPreparationBlocked("preparation SHA-256 mismatch")


__all__ = [
    "DEFAULT_R6_BUNDLE", "ENCODER_STEP_RAD", "EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD", "MODE", "SCHEMA", "TrialPreparationBlocked",
    "build_supervised_trial_preparation", "validate_supervised_trial_preparation",
]
