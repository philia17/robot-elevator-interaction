#!/usr/bin/env python3
"""Pure offline diagnostic IK/FK for a direct Standby -> active-Fn proposal.

This module is intentionally a *diagnostic*, not a route planner or an
execution input.  It uses the repository's standard URDF FK convention and
the operator-selected Fn translation pointer to answer a narrow question:
whether a numerical, position-only joint solution can be found from the
recorded Standby pose and whether samples of the resulting joint-space chord
remain inside the existing diagnostic T101 intervals.

It imports no ROS library, opens no device, creates no command, and does not
turn the selected translation-only Fn overlay into a contact frame.  In
particular it makes no collision-clearance claim; a continuous rear keep-out
model remains a separate required release.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from press.m1_no_contact_route import (
    GRIPPER_INDEX,
    TASK_JOINT_INDICES,
    T101_COMMAND_LIMITS,
    _load_urdf_joints,
    load_recorded_pose,
    standard_urdf_fk,
    URDF_PATH,
)


SCHEMA = "precontact_static_kinematics_report_v1"
MODE = "OFFLINE_STATIC_DIAGNOSTIC"
ACTIVE_FN_POINTER = (
    Path(__file__).resolve().parents[2]
    / "config/calibration/candidates/section_c_fingertip_c0_active_overlay_v1.json"
)
IK_EPSILON_RAD = 0.0005
IK_MAX_ITERS = 100
IK_POSITION_TOLERANCE_M = 0.0002
JOINT_SPACE_SAMPLES = 65


class StaticKinematicsBlocked(ValueError):
    """The diagnostic inputs are malformed or do not bind the selected Fn."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise StaticKinematicsBlocked(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StaticKinematicsBlocked(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise StaticKinematicsBlocked(f"{field} must be finite")
    return number


def _vec3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise StaticKinematicsBlocked(f"{field} must contain exactly three values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _vec6(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise StaticKinematicsBlocked(f"{field} must contain exactly six values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _rotate_xyzw(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    x, y, z = (float(value) for value in vector)
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    return (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )


def _active_fn_translation(path: str | Path) -> tuple[tuple[float, float, float], dict[str, str]]:
    pointer_path = Path(path).expanduser().resolve()
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticKinematicsBlocked(f"active Fn pointer unreadable: {exc}") from exc
    if not isinstance(pointer, Mapping) or pointer.get("status") != "diagnostic_overlay_default":
        raise StaticKinematicsBlocked("active Fn must remain the selected diagnostic overlay")
    artifact_id = str(pointer.get("active_artifact_id") or "")
    if not artifact_id:
        raise StaticKinematicsBlocked("active Fn pointer needs active_artifact_id")
    return _vec3(pointer.get("translation_m"), "active Fn pointer.translation_m"), {
        "pointer_path": str(pointer_path),
        "pointer_sha256": _sha256_file(pointer_path),
        "active_artifact_id": artifact_id,
        "status": "diagnostic_overlay_default",
    }


def fingertip_from_joints(
    positions_rad: Sequence[float], joints: Mapping[str, Any], fn_translation_hand_m: Sequence[float]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return Fn using the exact selected translation pointer and standard URDF FK."""
    tcp, rotation = standard_urdf_fk(_vec6(positions_rad, "positions_rad"), joints)
    offset = _rotate_xyzw(rotation, fn_translation_hand_m)
    return tuple(float(tcp[index]) + offset[index] for index in range(3)), tuple(float(value) for value in rotation)


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Small deterministic Gaussian solver; singular systems fail closed."""
    size = len(rhs)
    augmented = [matrix[row][:] + [rhs[row]] for row in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        if abs(augmented[col][col]) < 1e-12:
            return None
        for row in range(col + 1, size):
            factor = augmented[row][col] / augmented[col][col]
            for index in range(col, size + 1):
                augmented[row][index] -= factor * augmented[col][index]
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        result[row] = (augmented[row][size] - sum(augmented[row][index] * result[index] for index in range(row + 1, size))) / augmented[row][row]
    return result


def solve_position_only_ik(
    standby_positions_rad: Sequence[float], target_fn_base_m: Sequence[float], joints: Mapping[str, Any], fn_translation_hand_m: Sequence[float]
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Damped numerical position IK, with gripper fixed and no orientation claim."""
    current = list(_vec6(standby_positions_rad, "standby_positions_rad"))
    target = _vec3(target_fn_base_m, "target_fn_base_m")
    audit: list[dict[str, float | int]] = []
    for iteration in range(IK_MAX_ITERS):
        fn, _ = fingertip_from_joints(current, joints, fn_translation_hand_m)
        error = [target[index] - fn[index] for index in range(3)]
        error_m = _distance(fn, target)
        audit.append({"iteration": iteration, "position_error_mm": 1000.0 * error_m})
        if error_m <= IK_POSITION_TOLERANCE_M:
            break
        columns: list[list[float]] = []
        for joint_index in TASK_JOINT_INDICES:
            perturbed = list(current)
            perturbed[joint_index] += IK_EPSILON_RAD
            candidate, _ = fingertip_from_joints(perturbed, joints, fn_translation_hand_m)
            columns.append([(candidate[index] - fn[index]) / IK_EPSILON_RAD for index in range(3)])
        # delta = J^T (J J^T + lambda^2 I)^-1 error
        jjt = [[sum(columns[col][row] * columns[col][other] for col in range(len(columns))) for other in range(3)] for row in range(3)]
        for index in range(3):
            jjt[index][index] += 1e-6
        solved = _solve_linear(jjt, error)
        if solved is None:
            break
        for local_index, joint_index in enumerate(TASK_JOINT_INDICES):
            delta = sum(columns[local_index][row] * solved[row] for row in range(3))
            lower, upper = T101_COMMAND_LIMITS[joint_index]
            current[joint_index] = max(float(lower), min(float(upper), current[joint_index] + delta))
        current[GRIPPER_INDEX] = float(standby_positions_rad[GRIPPER_INDEX])
    final_fn, _ = fingertip_from_joints(current, joints, fn_translation_hand_m)
    final_error_m = _distance(final_fn, target)
    return tuple(current), {
        "method": "damped_numerical_position_ik_standard_urdf_fk_selected_fn_translation",
        "converged": final_error_m <= IK_POSITION_TOLERANCE_M,
        "iterations": len(audit),
        "final_position_error_mm": 1000.0 * final_error_m,
        "audit": audit,
        "orientation_constrained": False,
        "gripper_held": True,
    }


def build_static_kinematics_report(
    *, target_fn_base_m: Sequence[float], standby_pose_path: str | Path,
    active_fn_pointer_path: str | Path = ACTIVE_FN_POINTER,
) -> dict[str, Any]:
    """Produce a fail-closed diagnostic receipt for one direct outbound proposal.

    This function cannot return a released route: even a converged result is
    deliberately blocked because it is position-only and lacks the independent
    collision/geometry releases needed for a physical move.
    """
    standby = load_recorded_pose(standby_pose_path, "press_standby")
    target = _vec3(target_fn_base_m, "target_fn_base_m")
    translation, fn_pointer = _active_fn_translation(active_fn_pointer_path)
    joints = _load_urdf_joints(URDF_PATH)
    start = _vec6(standby["positions"], "standby.positions")
    start_fn, _ = fingertip_from_joints(start, joints, translation)
    solution, ik = solve_position_only_ik(start, target, joints, translation)
    final_fn, final_rotation = fingertip_from_joints(solution, joints, translation)
    samples = []
    limit_ok = True
    for index in range(JOINT_SPACE_SAMPLES):
        ratio = index / (JOINT_SPACE_SAMPLES - 1)
        positions = tuple(start[joint] + ratio * (solution[joint] - start[joint]) for joint in range(6))
        inside = all(float(T101_COMMAND_LIMITS[joint][0]) <= positions[joint] <= float(T101_COMMAND_LIMITS[joint][1]) for joint in range(6))
        limit_ok = limit_ok and inside
        fn, _ = fingertip_from_joints(positions, joints, translation)
        samples.append({"index": index, "inside_diagnostic_t101_limits": inside, "fn_base_m": list(fn)})
    deltas = [solution[index] - start[index] for index in range(6)]
    blockers = [
        "ACTIVE_FN_POINTER_DIAGNOSTIC_ONLY_NO_6D_CONTACT_RELEASE",
        "TARGET_NORMAL_AND_TOOL_ORIENTATION_RELEASE_MISSING",
        "SECTION_B_PHYSICAL_XYZ_AND_OPTICAL_AXIS_RELEASE_MISSING",
        "STATIC_IK_FK_RESULT_DIAGNOSTIC_ONLY_NOT_INDEPENDENTLY_RELEASED",
        "JOINT_LIMITS_CHECK_DIAGNOSTIC_ONLY_NOT_RELEASED",
        "CONTINUOUS_REAR_KEEPOUT_MODEL_AND_SWEEP_RELEASE_MISSING",
        "RUNTIME_FEEDBACK_RECHECK_AND_OPERATOR_MOVE_TICKET_REQUIRED",
    ]
    if not ik["converged"]:
        blockers.append("POSITION_ONLY_IK_DID_NOT_CONVERGE")
    if not limit_ok:
        blockers.append("SAMPLED_JOINT_SPACE_CHORD_OUTSIDE_DIAGNOSTIC_T101_LIMITS")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": MODE,
        "decision": "BLOCKED_DIAGNOSTIC_ONLY",
        "standby_pose_id": str(standby.get("pose_id") or ""),
        "standby_pose_path": str(Path(standby_pose_path).expanduser().resolve()),
        "urdf_path": str(URDF_PATH),
        "urdf_sha256": _sha256_file(URDF_PATH),
        "active_fn_evidence": fn_pointer,
        "fn_translation_hand_tcp_m": list(translation),
        "target_fn_pre_base_m": list(target),
        "start_fn_base_m": list(start_fn),
        "end_fn_base_m": list(final_fn),
        "end_hand_tcp_rotation_xyzw": list(final_rotation),
        "start_positions_rad": list(start),
        "candidate_positions_rad": list(solution),
        "joint_delta_rad": deltas,
        "direct_fn_chord_m": _distance(start_fn, final_fn),
        "ik_evidence": ik,
        "joint_limit_evidence": {
            "source": "roarm_safe_driver.core.T101_COMMAND_LIMITS",
            "sample_count": JOINT_SPACE_SAMPLES,
            "all_samples_inside_diagnostic_t101_limits": limit_ok,
            "samples": samples,
        },
        "continuous_collision_checked": False,
        "route_shape": "joint_space_linear_chord_samples_only",
        "required_blockers": sorted(blockers),
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "command_generated": False,
        "contact_authorized": False,
    }
    report["canonical_report_sha256"] = _canonical_sha256(report)
    validate_static_kinematics_report(report)
    return report


def validate_static_kinematics_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA or report.get("mode") != MODE:
        raise StaticKinematicsBlocked("unexpected static kinematics report schema")
    if report.get("decision") != "BLOCKED_DIAGNOSTIC_ONLY":
        raise StaticKinematicsBlocked("diagnostic report must remain blocked")
    for field in ("execution_ready", "motion_authorized", "trajectory_generated", "command_generated", "contact_authorized"):
        if report.get(field) is not False:
            raise StaticKinematicsBlocked(f"report must keep {field}=false")
    blockers = report.get("required_blockers")
    if not isinstance(blockers, list) or "CONTINUOUS_REAR_KEEPOUT_MODEL_AND_SWEEP_RELEASE_MISSING" not in blockers:
        raise StaticKinematicsBlocked("continuous rear keep-out blocker is required")
    unsigned = dict(report)
    supplied = str(unsigned.pop("canonical_report_sha256", ""))
    if supplied != _canonical_sha256(unsigned):
        raise StaticKinematicsBlocked("canonical report hash mismatch")


__all__ = [
    "ACTIVE_FN_POINTER", "SCHEMA", "StaticKinematicsBlocked", "build_static_kinematics_report",
    "fingertip_from_joints", "solve_position_only_ik", "validate_static_kinematics_report",
]
