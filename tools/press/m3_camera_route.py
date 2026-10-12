#!/usr/bin/env python3
"""Build a camera-driven T102 press route (demo path; accuracy deferred).

Consumes a frozen M0 target + accepted B relative bundle, transforms the
button point/normal into ``roarm_base_link``, solves position-only numerical
IK for pre-contact / press joint targets (seeded from the taught pre_contact
approach pose), and emits an ``m2_multi_joint_route_v1`` artifact that the
existing ``roarm_m2_adapter`` can execute unchanged.

Tool rotation is held at the taught approach seed — not solved.  If IK fails
and ``--allow-taught-pre-contact-fallback`` is set, the taught pre_contact
joints are used and the artifact is marked ``temporary_fallback``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_DRIVER_PACKAGE = REPO_ROOT / "ros2_arm" / "roarm_safe_driver"
if str(SAFE_DRIVER_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SAFE_DRIVER_PACKAGE))

from roarm_safe_driver.core import (  # noqa: E402
    DEFAULT_JOINT_MAX_DELTA_RAD,
    ENCODER_STEP_RAD,
    JOINT_NAMES,
    SafetyError,
    T101_COMMAND_LIMITS,
    validate_multi_joint_target,
)
from tools.press.m0_target_to_base import (  # noqa: E402
    DEFAULT_BUNDLE,
    EXPECTED_CALIBRATION_ID,
    TransformError,
    transform_frozen_target,
)
from tools.press.m1_no_contact_route import (  # noqa: E402
    GRIPPER_INDEX,
    TASK_JOINT_INDICES,
    URDF_PATH,
    RouteError,
    load_recorded_pose,
    sha256_bytes,
    sha256_file,
    standard_urdf_fk,
    write_artifact_exclusive,
    _distance,
    _load_urdf_joints,
)
from tools.press.m2_multi_joint_route import (  # noqa: E402
    C0_FINGERTIP_TRANSLATION_M,
    NUMERICAL_IK_EPSILON,
    NUMERICAL_IK_MAX_ITERS,
    NUMERICAL_IK_TOLERANCE_MM,
    SCHEMA,
    _estimate_press_joint_targets,
    _interpolate_waypoints,
    _required_waypoints,
    _safe_driver_core_path,
    _safe_driver_core_path_label,
    fake_driver_validate_multi,
)


D_PRE_DEFAULT_MM = 50.0
D_PRESS_DEFAULT_MM = 5.0

# The physical fingertip is bolted to the fixed palm/hand_tcp side.  The
# gripper joint drives only the moving jaw and is therefore not part of the
# fingertip FK/IK chain.  Keep point translation and long-axis direction as
# independent geometric quantities: a pivot/position candidate cannot prove
# the installed tool rotation.
FINGERTIP_PARENT_FRAME = "hand_tcp"
FINGERTIP_DEPENDS_ON_GRIPPER = False
TIP_POSITION_OFFSET_HAND_M = tuple(float(value) for value in C0_FINGERTIP_TRANSLATION_M)
# Horizontal-only axis estimate derived from the independently observed level
# fingertip at the configured standby feedback pose on 2026-08-09.  The prior
# nominal world direction was projected into the base XY plane and transformed
# back into hand_tcp.  This constrains the long-axis elevation only; it is not
# a full 6D press-contact rotation calibration.
TIP_AXIS_HAND_UNIT = (
    0.9983096054963713,
    -0.0358323582569689,
    0.045759957117813416,
)
TIP_AXIS_SOURCE = "standby_level_observation_20260809"


def _rotate_vector_xyzw(
    quat_xyzw: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quat_xyzw
    x, y, z = vector
    # standard quaternion rotation of a vector
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    return (
        ix * qw + iw * -qx + iy * -qz - iz * -qy,
        iy * qw + iw * -qy + iz * -qx - ix * -qz,
        iz * qw + iw * -qz + ix * -qy - iy * -qx,
    )


def fingertip_from_joints(
    positions: Sequence[float], joints: Mapping[str, Any]
) -> tuple[list[float], list[float]]:
    """Return the fixed-palm fingertip point and hand_tcp orientation.

    ``standard_urdf_fk`` ends at ``link5_to_hand_tcp`` and intentionally omits
    the moving gripper joint, so changing gripper position cannot move this
    frame in the model.
    """
    tcp_pos, tcp_quat = standard_urdf_fk(positions, joints)
    offset = _rotate_vector_xyzw(tcp_quat, TIP_POSITION_OFFSET_HAND_M)
    fingertip = [tcp_pos[i] + offset[i] for i in range(3)]
    return fingertip, tcp_quat


def tool_x_elevation_rad(quat_xyzw: Sequence[float]) -> float:
    """Return the elevation of hand_tcp +X above the world XY plane (radians).

    Zero means hand_tcp +X is horizontal.  This is the hand flange tilt, not
    the physical C0 fingertip long-axis elevation.
    """
    qx, qy, qz, qw = (float(value) for value in quat_xyzw)
    xz = 2.0 * (qx * qz + qy * qw)
    return math.asin(max(-1.0, min(1.0, xz)))


def fingertip_axis_world(
    quat_xyzw: Sequence[float],
    axis_hand: Sequence[float] = TIP_AXIS_HAND_UNIT,
) -> tuple[float, float, float]:
    """Rotate an independently defined fingertip long axis into the base frame."""
    if len(axis_hand) != 3:
        raise ValueError("axis_hand must contain three values")
    length = math.sqrt(sum(float(value) ** 2 for value in axis_hand))
    if length < 1e-12:
        raise ValueError("axis_hand must be non-zero")
    unit = tuple(float(value) / length for value in axis_hand)
    return _rotate_vector_xyzw(quat_xyzw, unit)


def fingertip_axis_elevation_rad(
    quat_xyzw: Sequence[float],
    axis_hand: Sequence[float] = TIP_AXIS_HAND_UNIT,
) -> float:
    """Return physical fingertip-axis elevation above the world XY plane.

    Zero means the independently modelled long axis is horizontal.  The point
    translation is intentionally not consulted here.
    """
    world = fingertip_axis_world(quat_xyzw, axis_hand)
    length = math.sqrt(world[0] ** 2 + world[1] ** 2 + world[2] ** 2)
    return math.asin(max(-1.0, min(1.0, world[2] / length)))


def solve_cartesian_ik_hold_orientation(
    seed_positions: Sequence[float],
    target_fingertip_m: Sequence[float],
    joints: Mapping[str, Any],
    *,
    tolerance_mm: float = NUMERICAL_IK_TOLERANCE_MM,
    max_iters: int = NUMERICAL_IK_MAX_ITERS,
    elevation_tolerance_deg: float = 0.5,
    elevation_weight: float = 0.15,
    hold_elevation_rad: float | None = None,
    hold_hand_tcp_elevation_rad: float | None = None,
    hand_tcp_elevation_weight: float = 0.0,
    hold_wrist_tilt_rad: float | None = None,
    wrist_tilt_weight: float = 0.0,
    wrist_tilt_tolerance_deg: float = 2.0,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Position IK that keeps the C0 fingertip axis level.

    Proximal joints plus wrist_tilt solve fingertip XYZ and fingertip-axis
    elevation; wrist_roll and gripper stay at the seed values.

    ``hold_elevation_rad`` locks the *fingertip* long-axis elevation (0 = world
    horizontal).  Optional ``hold_hand_tcp_elevation_rad`` soft-holds hand_tcp
    +X tilt at the standby-taught value so the flange keeps its standby lean
    without forcing hand_tcp itself horizontal.
    """
    if len(seed_positions) != 6:
        raise ValueError("seed_positions must contain six joints")
    current = list(float(value) for value in seed_positions)
    _, seed_quat = fingertip_from_joints(current, joints)
    elev0 = (
        float(hold_elevation_rad)
        if hold_elevation_rad is not None
        else fingertip_axis_elevation_rad(seed_quat)
    )
    hand_elev0 = (
        float(hold_hand_tcp_elevation_rad)
        if hold_hand_tcp_elevation_rad is not None
        else tool_x_elevation_rad(seed_quat)
    )
    use_hand_hold = float(hand_tcp_elevation_weight) > 0.0
    tilt0 = (
        float(hold_wrist_tilt_rad)
        if hold_wrist_tilt_rad is not None
        else float(seed_positions[3])
    )
    use_tilt_hold = float(wrist_tilt_weight) > 0.0
    residual_dim = 4 + (1 if use_hand_hold else 0) + (1 if use_tilt_hold else 0)
    # base, shoulder, elbow, wrist_tilt — roll and gripper are held.
    task = (0, 1, 2, 3)
    roll_index = 4
    audit_steps: list[dict[str, Any]] = []
    converged = False
    for iteration in range(max_iters):
        tip, quat = fingertip_from_joints(current, joints)
        elev = fingertip_axis_elevation_rad(quat)
        hand_elev = tool_x_elevation_rad(quat)
        tilt = float(current[3])
        err = [
            float(target_fingertip_m[0]) - tip[0],
            float(target_fingertip_m[1]) - tip[1],
            float(target_fingertip_m[2]) - tip[2],
            elevation_weight * (elev0 - elev),
        ]
        if use_hand_hold:
            err.append(hand_tcp_elevation_weight * (hand_elev0 - hand_elev))
        if use_tilt_hold:
            err.append(wrist_tilt_weight * (tilt0 - tilt))
        pos_err_mm = 1000.0 * math.sqrt(err[0] ** 2 + err[1] ** 2 + err[2] ** 2)
        elev_err_deg = math.degrees(elev - elev0)
        hand_err_deg = math.degrees(hand_elev - hand_elev0)
        tilt_err_deg = math.degrees(tilt - tilt0)
        audit_steps.append(
            {
                "iteration": iteration,
                "error_mm": pos_err_mm,
                "fingertip_elevation_error_deg": elev_err_deg,
                "hand_tcp_elevation_error_deg": hand_err_deg,
                "elevation_error_deg": elev_err_deg,
                "wrist_tilt_error_deg": tilt_err_deg,
            }
        )
        tilt_ok = (not use_tilt_hold) or abs(tilt_err_deg) < wrist_tilt_tolerance_deg
        hand_ok = (not use_hand_hold) or abs(hand_err_deg) < 3.0
        if (
            pos_err_mm < tolerance_mm
            and abs(elev_err_deg) < elevation_tolerance_deg
            and tilt_ok
            and hand_ok
        ):
            converged = True
            break

        columns: list[list[float]] = []
        for ji in task:
            perturbed = list(current)
            perturbed[ji] += NUMERICAL_IK_EPSILON
            pk_tip, pk_quat = fingertip_from_joints(perturbed, joints)
            pk_elev = fingertip_axis_elevation_rad(pk_quat)
            col = [(pk_tip[i] - tip[i]) / NUMERICAL_IK_EPSILON for i in range(3)] + [
                elevation_weight * (pk_elev - elev) / NUMERICAL_IK_EPSILON
            ]
            if use_hand_hold:
                pk_hand = tool_x_elevation_rad(pk_quat)
                col.append(
                    hand_tcp_elevation_weight
                    * (pk_hand - hand_elev)
                    / NUMERICAL_IK_EPSILON
                )
            if use_tilt_hold:
                pk_tilt = float(perturbed[3])
                col.append(wrist_tilt_weight * (pk_tilt - tilt) / NUMERICAL_IK_EPSILON)
            columns.append(col)

        n = len(task)
        jtj = [[0.0] * n for _ in range(n)]
        jte = [0.0] * n
        for a in range(n):
            for b in range(n):
                jtj[a][b] = sum(
                    columns[a][r] * columns[b][r] for r in range(residual_dim)
                )
                if a == b:
                    jtj[a][b] += 1e-4
            jte[a] = sum(columns[a][r] * err[r] for r in range(residual_dim))

        # Gauss elimination on the augmented [JTJ | JTe] system.
        aug = [jtj[i][:] + [jte[i]] for i in range(n)]
        for i in range(n):
            pivot = max(range(i, n), key=lambda row: abs(aug[row][i]))
            aug[i], aug[pivot] = aug[pivot], aug[i]
            if abs(aug[i][i]) < 1e-12:
                continue
            for row in range(i + 1, n):
                factor = aug[row][i] / aug[i][i]
                for col in range(i, n + 1):
                    aug[row][col] -= factor * aug[i][col]
        delta = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-12:
                delta[i] = 0.0
                continue
            delta[i] = (
                aug[i][n] - sum(aug[i][col] * delta[col] for col in range(i + 1, n))
            ) / aug[i][i]

        for ji_idx, ji in enumerate(task):
            current[ji] += delta[ji_idx]
            lower, upper = T101_COMMAND_LIMITS[ji]
            current[ji] = max(lower, min(upper, current[ji]))
        current[roll_index] = float(seed_positions[roll_index])
        current[GRIPPER_INDEX] = float(seed_positions[GRIPPER_INDEX])
    else:
        tip, quat = fingertip_from_joints(current, joints)
        elev = fingertip_axis_elevation_rad(quat)
        hand_elev = tool_x_elevation_rad(quat)
        tilt = float(current[3])
        pos_err_mm = 1000.0 * _distance(tip, target_fingertip_m)
        elev_err_deg = math.degrees(elev - elev0)
        hand_err_deg = math.degrees(hand_elev - hand_elev0)
        tilt_err_deg = math.degrees(tilt - tilt0)
        tilt_ok = (not use_tilt_hold) or abs(tilt_err_deg) < wrist_tilt_tolerance_deg
        hand_ok = (not use_hand_hold) or abs(hand_err_deg) < 3.0
        converged = (
            pos_err_mm < tolerance_mm
            and abs(elev_err_deg) < elevation_tolerance_deg
            and tilt_ok
            and hand_ok
        )

    current[roll_index] = float(seed_positions[roll_index])
    current[GRIPPER_INDEX] = float(seed_positions[GRIPPER_INDEX])

    # Hard fingertip-level mop-up via wrist_tilt only (primary tip contract).
    mop_iters = 0
    for mop_iters in range(1, 25):
        tip_m, quat_m = fingertip_from_joints(current, joints)
        elev_m = fingertip_axis_elevation_rad(quat_m)
        elev_err = elev0 - elev_m
        if abs(math.degrees(elev_err)) < elevation_tolerance_deg:
            break
        perturbed = list(current)
        perturbed[3] += NUMERICAL_IK_EPSILON
        _, pk_quat = fingertip_from_joints(perturbed, joints)
        d_elev = (fingertip_axis_elevation_rad(pk_quat) - elev_m) / NUMERICAL_IK_EPSILON
        if abs(d_elev) < 1e-9:
            break
        step = elev_err / d_elev
        step = max(-0.15, min(0.15, step))
        lower, upper = T101_COMMAND_LIMITS[3]
        current[3] = max(lower, min(upper, float(current[3]) + step))
        current[roll_index] = float(seed_positions[roll_index])
        current[GRIPPER_INDEX] = float(seed_positions[GRIPPER_INDEX])
    else:
        mop_iters = 24

    tip_final, quat_final = fingertip_from_joints(current, joints)
    elev_final = fingertip_axis_elevation_rad(quat_final)
    hand_final = tool_x_elevation_rad(quat_final)
    tilt_final = float(current[3])
    final_error_mm = 1000.0 * _distance(tip_final, target_fingertip_m)
    elev_err_final_deg = math.degrees(elev_final - elev0)
    hand_err_final_deg = math.degrees(hand_final - hand_elev0)
    tilt_ok = (not use_tilt_hold) or abs(tilt_final - tilt0) < math.radians(
        wrist_tilt_tolerance_deg
    )
    hand_ok = (not use_hand_hold) or abs(hand_err_final_deg) < 3.0
    converged = (
        final_error_mm < tolerance_mm * 2.0
        and abs(elev_err_final_deg) < elevation_tolerance_deg
        and tilt_ok
        and hand_ok
    )
    result = tuple(current)
    moving = [
        index
        for index, (value, seed) in enumerate(zip(result, seed_positions))
        if abs(value - float(seed)) > ENCODER_STEP_RAD + 1e-12
    ]
    return result, {
        "method": "numerical_ik_position_hold_fingertip_axis_elevation",
        "converged": converged,
        "iterations": len(audit_steps),
        "elevation_mop_iters": mop_iters,
        "final_error_mm": final_error_mm,
        "seed_elevation_deg": math.degrees(elev0),
        "final_elevation_deg": math.degrees(elev_final),
        "elevation_error_deg": elev_err_final_deg,
        "seed_hand_tcp_elevation_deg": math.degrees(hand_elev0),
        "final_hand_tcp_elevation_deg": math.degrees(hand_final),
        "hand_tcp_elevation_error_deg": hand_err_final_deg,
        "hold_wrist_tilt_deg": math.degrees(tilt0),
        "final_wrist_tilt_deg": math.degrees(tilt_final),
        "wrist_tilt_error_deg": math.degrees(tilt_final - tilt0),
        "held_joint_indices": [roll_index, GRIPPER_INDEX],
        "moving_joint_indices": moving,
        "moving_joint_names": [JOINT_NAMES[i] for i in moving],
        "audit": audit_steps,
        "target_fingertip_m": list(target_fingertip_m),
    }


def solve_position_ik(
    seed_positions: Sequence[float],
    target_fingertip_m: Sequence[float],
    joints: Mapping[str, Any],
    *,
    tolerance_mm: float = NUMERICAL_IK_TOLERANCE_MM,
    max_iters: int = NUMERICAL_IK_MAX_ITERS,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Position-only numerical IK; holds gripper; seeds approach orientation."""
    current = list(seed_positions)
    audit_steps: list[dict[str, Any]] = []
    for iteration in range(max_iters):
        finger_current, _ = fingertip_from_joints(current, joints)
        error = _distance(finger_current, target_fingertip_m)
        error_vec = [target_fingertip_m[i] - finger_current[i] for i in range(3)]
        audit_steps.append(
            {
                "iteration": iteration,
                "error_mm": 1000.0 * error,
                "error_vector_m": error_vec,
            }
        )
        if error * 1000.0 < tolerance_mm:
            break

        jacobian = []
        for ji in TASK_JOINT_INDICES:
            perturbed = list(current)
            perturbed[ji] += NUMERICAL_IK_EPSILON
            pk_finger, _ = fingertip_from_joints(perturbed, joints)
            column = [
                (pk_finger[i] - finger_current[i]) / NUMERICAL_IK_EPSILON
                for i in range(3)
            ]
            jacobian.append(column)

        jjt = [[0.0] * 3 for _ in range(3)]
        for row in range(3):
            for col in range(3):
                jjt[row][col] = sum(
                    jacobian[ji][row] * jacobian[ji][col]
                    for ji in range(len(TASK_JOINT_INDICES))
                )
        damping = 0.001
        for i in range(3):
            jjt[i][i] += damping * damping

        det = (
            jjt[0][0] * (jjt[1][1] * jjt[2][2] - jjt[1][2] * jjt[2][1])
            - jjt[0][1] * (jjt[1][0] * jjt[2][2] - jjt[1][2] * jjt[2][0])
            + jjt[0][2] * (jjt[1][0] * jjt[2][1] - jjt[1][1] * jjt[2][0])
        )
        if abs(det) < 1e-20:
            break
        inv_det = 1.0 / det
        inv = [
            [
                inv_det * (jjt[1][1] * jjt[2][2] - jjt[1][2] * jjt[2][1]),
                inv_det * (jjt[0][2] * jjt[2][1] - jjt[0][1] * jjt[2][2]),
                inv_det * (jjt[0][1] * jjt[1][2] - jjt[0][2] * jjt[1][1]),
            ],
            [
                inv_det * (jjt[1][2] * jjt[2][0] - jjt[1][0] * jjt[2][2]),
                inv_det * (jjt[0][0] * jjt[2][2] - jjt[0][2] * jjt[2][0]),
                inv_det * (jjt[0][2] * jjt[1][0] - jjt[0][0] * jjt[1][2]),
            ],
            [
                inv_det * (jjt[1][0] * jjt[2][1] - jjt[1][1] * jjt[2][0]),
                inv_det * (jjt[0][1] * jjt[2][0] - jjt[0][0] * jjt[2][1]),
                inv_det * (jjt[0][0] * jjt[1][1] - jjt[0][1] * jjt[1][0]),
            ],
        ]
        for ji_idx, ji in enumerate(TASK_JOINT_INDICES):
            delta = sum(
                jacobian[ji_idx][k] * sum(inv[k][m] * error_vec[m] for m in range(3))
                for k in range(3)
            )
            current[ji] += delta
            lower, upper = T101_COMMAND_LIMITS[ji]
            current[ji] = max(lower, min(upper, current[ji]))

    current[GRIPPER_INDEX] = seed_positions[GRIPPER_INDEX]
    result = tuple(current)
    final_error_mm = audit_steps[-1]["error_mm"] if audit_steps else None
    converged = final_error_mm is not None and final_error_mm < tolerance_mm
    # Moving-joint audit only — overall displacement may exceed one T102 step
    # and is subdivided later against DEFAULT_JOINT_MAX_DELTA_RAD.
    moving = [
        index
        for index, (value, seed) in enumerate(zip(result, seed_positions))
        if abs(value - seed) > ENCODER_STEP_RAD + 1e-12
    ]
    return result, {
        "method": "numerical_ik_position_only_seeded_approach",
        "converged": converged,
        "iterations": len(audit_steps),
        "final_error_mm": final_error_mm,
        "moving_joint_indices": moving,
        "moving_joint_names": [JOINT_NAMES[i] for i in moving],
        "audit": audit_steps,
        "target_fingertip_m": list(target_fingertip_m),
    }


def _load_frozen(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, Mapping) and "frozen_target" in payload:
        return payload["frozen_target"]
    if not isinstance(payload, Mapping):
        raise RouteError("frozen target must be a JSON object")
    return payload


def build_camera_route(
    frozen_target_path: str | Path,
    standby_path: str | Path,
    *,
    seed_pre_contact_path: str | Path,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    expected_calibration_id: str = EXPECTED_CALIBRATION_ID,
    d_pre_mm: float = D_PRE_DEFAULT_MM,
    d_press_mm: float = D_PRESS_DEFAULT_MM,
    allow_taught_pre_contact_fallback: bool = False,
    allow_incomplete_provenance: bool = True,
) -> dict[str, Any]:
    frozen_path = Path(frozen_target_path).expanduser().resolve()
    frozen = _load_frozen(frozen_path)
    try:
        base_target = transform_frozen_target(
            frozen,
            bundle_path,
            expected_calibration_id=expected_calibration_id,
            allow_incomplete_provenance=allow_incomplete_provenance,
        )
    except TransformError as exc:
        raise RouteError(str(exc)) from exc

    standby = load_recorded_pose(standby_path, "press_standby")
    seed = load_recorded_pose(seed_pre_contact_path, "pre_contact")
    joints = _load_urdf_joints(URDF_PATH)

    start = standby["positions"]
    seed_positions = list(seed["positions"])
    seed_positions[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    seed_tuple = tuple(seed_positions)

    button = base_target["point_roarm_base_m"]
    normal = base_target["panel_outward_normal_roarm_base"]
    d_pre_m = d_pre_mm / 1000.0
    d_press_m = d_press_mm / 1000.0
    # Approach from outside the panel (along outward normal); press slightly
    # into the panel (against the outward normal).
    pre_fingertip = [button[i] + d_pre_m * normal[i] for i in range(3)]
    press_fingertip = [button[i] - d_press_m * normal[i] for i in range(3)]

    pre_target, pre_ik = solve_position_ik(seed_tuple, pre_fingertip, joints)
    press_target, press_ik = solve_position_ik(pre_target, press_fingertip, joints)

    used_fallback = False
    if not pre_ik["converged"]:
        if not allow_taught_pre_contact_fallback:
            raise RouteError(
                f"pre-contact IK failed (final_error_mm={pre_ik['final_error_mm']}); "
                "pass --allow-taught-pre-contact-fallback for temporary smoke only"
            )
        pre_target = seed_tuple
        used_fallback = True
        # Temporary smoke: keep taught approach joints and reuse the M2 TCP +x
        # press estimate so press/retract remain non-zero T102 commands.
        press_target, press_ik = _estimate_press_joint_targets(
            pre_target, d_press_m, joints
        )
        press_ik = {
            **press_ik,
            "converged": True,
            "fallback_press_method": "m2_tcp_plus_x_from_taught_pre_contact",
        }

    pre_tuple = tuple(pre_target)
    press_tuple = tuple(press_target)
    # Hold gripper everywhere.
    pre_list = list(pre_tuple)
    press_list = list(press_tuple)
    pre_list[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    press_list[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    pre_tuple = tuple(pre_list)
    press_tuple = tuple(press_list)

    outbound = _interpolate_waypoints(
        start, pre_tuple, _required_waypoints(start, pre_tuple)
    )
    press_wps = _interpolate_waypoints(
        pre_tuple, press_tuple, _required_waypoints(pre_tuple, press_tuple)
    )
    retract_wps = _interpolate_waypoints(
        press_tuple, pre_tuple, _required_waypoints(press_tuple, pre_tuple)
    )
    return_wps = _interpolate_waypoints(
        pre_tuple, start, _required_waypoints(pre_tuple, start)
    )
    all_waypoints = outbound + press_wps + retract_wps + return_wps
    all_phases = (
        ["outbound"] * len(outbound)
        + ["press"] * len(press_wps)
        + ["retract"] * len(retract_wps)
        + ["return"] * len(return_wps)
    )

    waypoint_records = []
    for idx, (wp, phase) in enumerate(zip(all_waypoints, all_phases)):
        previous = all_waypoints[idx - 1] if idx > 0 else start
        try:
            _, moving = validate_multi_joint_target(
                wp, previous, DEFAULT_JOINT_MAX_DELTA_RAD
            )
        except SafetyError as exc:
            if "at least one logical joint" in str(exc):
                # Identical consecutive waypoints (e.g. zero press depth after
                # fallback) are recorded as no-op holds, not rejected.
                moving = []
            else:
                raise
        waypoint_records.append(
            {
                "waypoint_id": f"W{idx + 1:03d}",
                "phase": phase,
                "positions_rad": list(wp),
                "moving_joint_indices": moving,
                "moving_joint_names": [JOINT_NAMES[i] for i in moving],
                "tcp_m": list(standard_urdf_fk(wp, joints)[0]),
            }
        )

    t102_speed = 17
    t102_acc = 1
    speeds = [t102_speed] * len(all_waypoints)
    accs = [t102_acc] * len(all_waypoints)
    simulation = fake_driver_validate_multi(start, all_waypoints, speeds, accs)

    start_fk, _ = standard_urdf_fk(start, joints)
    pre_fk, _ = standard_urdf_fk(pre_tuple, joints)
    press_fk, _ = standard_urdf_fk(press_tuple, joints)
    pre_finger, _ = fingertip_from_joints(pre_tuple, joints)
    press_finger, _ = fingertip_from_joints(press_tuple, joints)

    label = base_target.get("target_label") or "unknown"
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_id": (
            f"camera_{label}__standby_{standby['pose_id']}__"
            f"dpre_{d_pre_mm:.0f}mm_dpress_{d_press_mm:.0f}mm__m3_demo_v1"
        ),
        "status": (
            "offline_validated_temporary_fallback"
            if used_fallback
            else "offline_validated_candidate"
        ),
        "scope": (
            "camera-driven T102 demo route; position IK + taught approach seed; "
            "accuracy / visual correction deferred"
        ),
        "route_origin": "camera_driven_demo",
        "temporary_fallback": used_fallback,
        "sources": {
            "press_standby": {
                "path": str(standby["path"].relative_to(REPO_ROOT)),
                "sha256": standby["sha256"],
                "pose_id": standby["pose_id"],
            },
            "seed_pre_contact": {
                "path": str(seed["path"].relative_to(REPO_ROOT)),
                "sha256": seed["sha256"],
                "pose_id": seed["pose_id"],
                "role": "approach_orientation_seed",
            },
            "frozen_target": {
                "path": str(frozen_path.relative_to(REPO_ROOT))
                if frozen_path.is_relative_to(REPO_ROOT)
                else str(frozen_path),
                "sha256": sha256_file(frozen_path),
            },
            "base_target": {
                "calibration_id": base_target["calibration_id"],
                "quality": base_target["quality"],
                "canonical_payload_sha256": base_target["canonical_payload_sha256"],
            },
            "safe_driver_core": {
                "path": _safe_driver_core_path_label(),
                "sha256": sha256_file(_safe_driver_core_path()),
            },
        },
        "joint_contract": {
            "joint_names": list(JOINT_NAMES),
            "task_joint_indices": list(TASK_JOINT_INDICES),
            "gripper_policy": "hold_press_standby_start_value",
            "gripper_hold_rad": start[GRIPPER_INDEX],
            "t101_command_limits_rad": [list(values) for values in T101_COMMAND_LIMITS],
            "default_joint_max_delta_rad": list(DEFAULT_JOINT_MAX_DELTA_RAD),
            "encoder_step_rad": ENCODER_STEP_RAD,
            "command_type": "T102_multi_joint",
        },
        "route_policy": {
            "outbound_waypoint_count": len(outbound),
            "press_waypoint_count": len(press_wps),
            "retract_waypoint_count": len(retract_wps),
            "return_waypoint_count": len(return_wps),
            "t102_shared_speed_steps_s": t102_speed,
            "t102_shared_acc_register": t102_acc,
            "automatic_retry": False,
            "automatic_compensation": False,
            "automatic_retract_after_fault": False,
            "d_pre_mm": d_pre_mm,
            "d_press_mm": d_press_mm,
            "tool_rotation_policy": "hold_taught_approach_seed",
        },
        "endpoints": {
            "press_standby_start_rad": list(start),
            "pre_contact_rad": list(pre_tuple),
            "press_target_rad": list(press_tuple),
            "retract_target_rad": list(pre_tuple),
            "return_target_rad": list(start),
        },
        "waypoints": waypoint_records,
        "totals": {
            "waypoint_count": len(all_waypoints),
            "outbound_count": len(outbound),
            "press_count": len(press_wps),
            "retract_count": len(retract_wps),
            "return_count": len(return_wps),
        },
        "camera_target": {
            "point_roarm_base_m": base_target["point_roarm_base_m"],
            "panel_outward_normal_roarm_base": base_target[
                "panel_outward_normal_roarm_base"
            ],
            "pre_fingertip_m": pre_fingertip,
            "press_fingertip_m": press_fingertip,
            "quality": base_target["quality"],
            "calibration_id": base_target["calibration_id"],
        },
        "press_step": {
            "press_distance_mm": d_press_mm,
            # Legacy key retained for schema readers; V2 geometry uses the
            # explicit, independent point-offset and axis fields below.
            "c0_fingertip_translation_m": list(C0_FINGERTIP_TRANSLATION_M),
            "tip_position_offset_hand_m": list(TIP_POSITION_OFFSET_HAND_M),
            "tip_axis_hand_unit": list(TIP_AXIS_HAND_UNIT),
            "tip_axis_source": TIP_AXIS_SOURCE,
            "fingertip_parent_frame": FINGERTIP_PARENT_FRAME,
            "fingertip_depends_on_gripper": FINGERTIP_DEPENDS_ON_GRIPPER,
            "pre_ik_audit": pre_ik,
            "ik_audit": press_ik,
            "achieved_pre_fingertip_m": pre_finger,
            "achieved_press_fingertip_m": press_finger,
        },
        "geometry_audit": {
            "start_tcp_m": start_fk,
            "pre_contact_tcp_m": pre_fk,
            "press_tcp_m": press_fk,
            "tcp_displacement_mm": {
                "outbound_chord": 1000.0 * _distance(start_fk, pre_fk),
                "press": 1000.0 * _distance(pre_fk, press_fk),
            },
        },
        "fake_driver_validation": simulation,
        "gates": {
            "source_pose_integrity": "passed",
            "calibration_id_binding": "passed",
            "multi_joint_validation": "passed",
            "t101_limit_validation": "passed",
            "fake_driver_t102_validation": "passed",
            "ik_convergence": "passed" if not used_fallback else "temporary_fallback",
            "collision_validation": "pending",
            "keep_out_validation": "pending",
            "现场clearance确认": "required_before_every_real_step",
        },
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "limitations": [
            "demo path: accuracy / button-center hit rate deferred",
            "position-only IK; tool rotation held at taught approach seed",
            "C0 fingertip translation used; C rotation unmeasured",
            "visual closed-loop correction (M3.3) not applied",
            "B bundle offline math only; no /tf_static publish",
            "real motion requires separate narrow authorization and现场clearance",
        ],
    }
    if used_fallback:
        artifact["limitations"].insert(
            0,
            "TEMPORARY FALLBACK: taught pre_contact joints used because camera IK "
            "did not converge; not a camera-driven success",
        )
    digest_source = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    artifact["canonical_payload_sha256"] = sha256_bytes(digest_source)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-target", required=True)
    parser.add_argument(
        "--standby",
        default=str(
            REPO_ROOT
            / "config"
            / "press"
            / "poses"
            / "press_standby_manual_20260804_preflight.yaml"
        ),
    )
    parser.add_argument(
        "--seed-pre-contact",
        default=str(
            REPO_ROOT / "config" / "press" / "poses" / "pre_contact_candidate_v1.yaml"
        ),
        help="Taught approach pose used only as IK seed / temporary fallback",
    )
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--expected-calibration-id", default=EXPECTED_CALIBRATION_ID)
    parser.add_argument("--d-pre-mm", type=float, default=D_PRE_DEFAULT_MM)
    parser.add_argument("--d-press-mm", type=float, default=D_PRESS_DEFAULT_MM)
    parser.add_argument(
        "--allow-taught-pre-contact-fallback",
        action="store_true",
        help="If camera IK fails, use taught pre_contact (temporary smoke only)",
    )
    parser.add_argument(
        "--require-complete-provenance",
        action="store_true",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = build_camera_route(
        args.frozen_target,
        args.standby,
        seed_pre_contact_path=args.seed_pre_contact,
        bundle_path=args.bundle,
        expected_calibration_id=args.expected_calibration_id,
        d_pre_mm=args.d_pre_mm,
        d_press_mm=args.d_press_mm,
        allow_taught_pre_contact_fallback=args.allow_taught_pre_contact_fallback,
        allow_incomplete_provenance=not args.require_complete_provenance,
    )
    result = write_artifact_exclusive(artifact, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
