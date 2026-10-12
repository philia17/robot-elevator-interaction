#!/usr/bin/env python3
"""Build and fake-driver validate a T102 multi-joint press route.

Produces a route artifact where each phase (outbound, press, retract, return)
is one T102 multi-joint command — all task joints move simultaneously.
The press step computes a small Cartesian displacement from pre_contact
using FK-based numerical Jacobian, then solves for joint targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_DRIVER_PACKAGE = REPO_ROOT / "ros2_arm" / "roarm_safe_driver"
if str(SAFE_DRIVER_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SAFE_DRIVER_PACKAGE))

from roarm_safe_driver.core import (  # noqa: E402
    DEFAULT_JOINT_MAX_DELTA_RAD,
    ENCODER_STEP_RAD,
    Feedback,
    JOINT_NAMES,
    MAX_JOINT_DELTA_RAD,
    SafeController,
    T101_COMMAND_LIMITS,
    build_t102,
    validate_multi_joint_target,
)
from tools.press.m1_no_contact_route import (  # noqa: E402
    GRIPPER_INDEX,
    REPO_ROOT,
    TASK_JOINT_INDICES,
    URDF_PATH,
    RouteError,
    load_recorded_pose,
    sha256_bytes,
    sha256_file,
    standard_urdf_fk,
    write_artifact_exclusive,
    _distance,
    _finite_sequence,
    _finite_vector,
    _load_urdf_joints,
    _mapping,
)


def _safe_driver_core_path() -> Path:
    """Resolve core.py from the imported module so both the Mac layout and
    the AGX layout (installed/sourced package) bind the running code."""
    import roarm_safe_driver.core as _core_module

    return Path(_core_module.__file__).resolve()


def _safe_driver_core_path_label() -> str:
    path = _safe_driver_core_path()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


SCHEMA = "m2_multi_joint_route_v1"
PRESS_DEFAULT_MM = 5.0
NUMERICAL_IK_EPSILON = 0.0005  # 0.0005 rad perturbation for Jacobian
NUMERICAL_IK_MAX_ITERS = 20
NUMERICAL_IK_TOLERANCE_MM = 0.1  # converge when TCP error < 0.1 mm
C0_FINGERTIP_TRANSLATION_M = (0.12697, 0.01054, 0.00496)


def _estimate_press_joint_targets(
    pre_contact_positions: tuple[float, ...],
    press_distance_m: float,
    joints: Mapping[str, Any],
) -> tuple[tuple[float, ...], dict[str, Any]]:
    """Compute joint targets for a small press displacement from pre_contact.

    Uses iterative numerical IK: FK finite-difference Jacobian + damped
    least-squares.  The press direction is approximately along the hand_tcp
    x-axis at the pre_contact pose (the fingertip extends ~127 mm forward).
    """
    # Compute current TCP pose
    tcp_pos, tcp_quat = standard_urdf_fk(pre_contact_positions, joints)

    # Fingertip position (C0 translation rotated into base frame)
    # Simple approximation: add fingertip offset in hand_tcp frame.
    # For small rotations near pre_contact, offset ≈ translation.
    fingertip_pos = [
        tcp_pos[i] + C0_FINGERTIP_TRANSLATION_M[i] for i in range(3)
    ]

    # Press target: move fingertip forward by press_distance.
    # Direction: approximate as the hand_tcp +x direction at this pose.
    # At pre_contact, the quaternion is roughly [0.014, -0.108, 0.112, 0.988],
    # which is near identity.  We use the quaternion to rotate [1,0,0].
    qw, qx, qy, qz = tcp_quat[3], tcp_quat[0], tcp_quat[1], tcp_quat[2]
    # Rotate [1, 0, 0] by quaternion to get x-axis in base frame
    x_dir = [
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy + qz * qw),
        2 * (qx * qz - qy * qw),
    ]
    norm = math.sqrt(sum(v * v for v in x_dir))
    x_dir = [v / norm for v in x_dir]

    target_fingertip = [
        fingertip_pos[i] + press_distance_m * x_dir[i] for i in range(3)
    ]

    # Numerical IK: iterate to find joint angles that place TCP close to target.
    current = list(pre_contact_positions)
    audit_steps = []
    for iteration in range(NUMERICAL_IK_MAX_ITERS):
        fk_pos, _ = standard_urdf_fk(current, joints)
        finger_current = [
            fk_pos[i] + C0_FINGERTIP_TRANSLATION_M[i] for i in range(3)
        ]
        error = _distance(finger_current, target_fingertip)
        error_vec = [
            target_fingertip[i] - finger_current[i] for i in range(3)
        ]
        audit_steps.append({
            "iteration": iteration,
            "error_mm": 1000.0 * error,
            "error_vector_m": error_vec,
        })
        if error * 1000.0 < NUMERICAL_IK_TOLERANCE_MM:
            break

        # Compute Jacobian via finite differences on task joints only
        jacobian = []
        for ji in TASK_JOINT_INDICES:
            perturbed = list(current)
            perturbed[ji] += NUMERICAL_IK_EPSILON
            pk_pos, _ = standard_urdf_fk(perturbed, joints)
            pk_finger = [
                pk_pos[i] + C0_FINGERTIP_TRANSLATION_M[i] for i in range(3)
            ]
            column = [
                (pk_finger[i] - finger_current[i]) / NUMERICAL_IK_EPSILON
                for i in range(3)
            ]
            jacobian.append(column)

        # Damped least-squares: delta_q = J^T (J J^T + lambda^2 I)^{-1} error
        # For 5x3 Jacobian, compute (J J^T) as 3x3
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

        # Solve 3x3 system via Cramer's rule or simple Gaussian elimination
        # (3x3 is small enough for direct inversion)
        det = (
            jjt[0][0] * (jjt[1][1] * jjt[2][2] - jjt[1][2] * jjt[2][1])
            - jjt[0][1] * (jjt[1][0] * jjt[2][2] - jjt[1][2] * jjt[2][0])
            + jjt[0][2] * (jjt[1][0] * jjt[2][1] - jjt[1][1] * jjt[2][0])
        )
        if abs(det) < 1e-20:
            break  # singular — stop iterating

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

        # J^T * inv(JJT) * error
        jt_inv_error = [0.0] * len(TASK_JOINT_INDICES)
        for ji_idx, ji in enumerate(TASK_JOINT_INDICES):
            jt_inv_error[ji_idx] = sum(
                jacobian[ji_idx][k] * sum(inv[k][m] * error_vec[m] for m in range(3))
                for k in range(3)
            )

        # Apply delta to task joints
        for ji_idx, ji in enumerate(TASK_JOINT_INDICES):
            current[ji] += jt_inv_error[ji_idx]
            # Clamp to T101 limits
            lower, upper = T101_COMMAND_LIMITS[ji]
            current[ji] = max(lower, min(upper, current[ji]))

    result = tuple(current)
    # Hold gripper at pre_contact value
    result_list = list(result)
    result_list[GRIPPER_INDEX] = pre_contact_positions[GRIPPER_INDEX]
    result = tuple(result_list)

    # Validate the full press displacement against the hard ceilings only;
    # the route subdivides it into profile-compliant waypoints afterwards.
    _, moving = validate_multi_joint_target(
        result, pre_contact_positions, MAX_JOINT_DELTA_RAD,
    )
    return result, {
        "method": "numerical_ik_fk_jacobian_damped_least_squares",
        "iterations": len(audit_steps),
        "final_error_mm": audit_steps[-1]["error_mm"] if audit_steps else None,
        "moving_joint_indices": moving,
        "moving_joint_names": [JOINT_NAMES[i] for i in moving],
        "audit": audit_steps,
    }


def _healthy_feedback(positions: Sequence[float], received_at: float) -> Feedback:
    return Feedback(
        positions=tuple(float(v) for v in positions),
        loads=(0, 0, 0, 0, 0, 0),
        torque_enabled=(True, True, True, True, True, True),
        voltage=12.1,
        received_at=received_at,
        raw={"source": "synthetic_in_memory_only"},
    )


def fake_driver_validate_multi(
    start_positions: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    speeds: Sequence[int],
    accelerations: Sequence[int],
) -> dict[str, Any]:
    """Exercise each T102 waypoint through SafeController without ROS/serial."""
    writes: list[bytes] = []
    # Validate against the deployed AGX profile limits so every waypoint the
    # artifact records is actually acceptable to the real driver.
    controller = SafeController(
        writes.append,
        dry_run=False,
        joint_max_delta_rad=DEFAULT_JOINT_MAX_DELTA_RAD,
    )
    measured = tuple(float(v) for v in start_positions)
    now = 100.0
    automatic_disarms = 0
    for idx, (waypoint, speed, acc) in enumerate(
        zip(waypoints, speeds, accelerations), start=1
    ):
        controller.accept_feedback(_healthy_feedback(measured, now))
        controller.arm(now)
        payload = controller.command_multi(waypoint, now)
        decoded = json.loads(payload)
        if decoded.get("T") != 102:
            raise RouteError(f"fake driver emitted a non-T102 payload at step {idx}")
        target = tuple(float(v) for v in waypoint)
        settle_start = now + 0.10
        for sample_idx in range(25):
            controller.accept_feedback(
                _healthy_feedback(target, settle_start + sample_idx * 0.05)
            )
        if controller.last_result != "CONVERGED_DISARMED" or controller.armed:
            raise RouteError(
                f"fake driver did not auto-disarm at step {idx}: "
                f"{controller.last_result}"
            )
        automatic_disarms += 1
        measured = target
        now += 2.0
    if len(writes) != len(waypoints):
        raise RouteError("fake-driver write count != waypoint count")
    return {
        "backend": "SafeController_in_memory_write_and_synthetic_feedback",
        "ros_used": False,
        "serial_used": False,
        "waypoint_count": len(waypoints),
        "t102_payload_count": len(writes),
        "automatic_disarm_count": automatic_disarms,
        "t102_only": all(json.loads(p).get("T") == 102 for p in writes),
        "t101_absent": all(b'"T":101' not in p for p in writes),
        "final_positions_rad": list(measured),
        "passed": True,
    }


def _required_waypoints(
    start: tuple[float, ...],
    end: tuple[float, ...],
    max_delta_rad: Sequence[float] = DEFAULT_JOINT_MAX_DELTA_RAD,
) -> int:
    """Compute minimum waypoints needed so no joint exceeds its max delta.

    Defaults to the normal AGX driver profile limits (not the hard ceilings):
    the deployed driver rejects any per-command delta above its configured
    profile, so route artifacts must be generated against the same values.

    The per-step budget reserves one encoder step of headroom: the safe driver
    quantizes T102 movers onto the encoder grid (round-to-nearest), which can
    push a planned step up to half a step over its raw delta.  A step planned
    exactly at the profile limit can therefore be rejected after quantization
    (e.g. 1.9944 deg -> 2.021 deg vs a 2.0 deg limit), so routes must plan
    against the limit minus one encoder step.
    """
    max_count = 1
    for i in range(len(start)):
        delta = abs(end[i] - start[i])
        if delta <= ENCODER_STEP_RAD + 1e-12:
            continue
        budget = max(ENCODER_STEP_RAD, float(max_delta_rad[i]) - ENCODER_STEP_RAD)
        needed = int(math.ceil(delta / budget - 1e-12))
        if needed > max_count:
            max_count = needed
    return max(max_count, 1)


def build_route(
    standby_path: str | Path,
    pre_contact_path: str | Path,
    press_distance_mm: float = PRESS_DEFAULT_MM,
    num_outbound_waypoints: int | None = None,
    num_return_waypoints: int | None = None,
) -> dict[str, Any]:
    standby = load_recorded_pose(standby_path, "press_standby")
    pre_contact = load_recorded_pose(pre_contact_path, "pre_contact")
    joints = _load_urdf_joints(URDF_PATH)

    start = standby["positions"]
    pre_vec = list(pre_contact["positions"])
    pre_vec[GRIPPER_INDEX] = start[GRIPPER_INDEX]  # hold gripper
    pre_tuple = tuple(pre_vec)

    # --- Outbound: standby → pre_contact ---
    press_distance_m = press_distance_mm / 1000.0

    if num_outbound_waypoints is None:
        num_outbound_waypoints = _required_waypoints(start, pre_tuple)
    if num_outbound_waypoints < 1:
        raise RouteError("num_outbound_waypoints must be >= 1")
    outbound_waypoints = _interpolate_waypoints(
        start, pre_tuple, num_outbound_waypoints
    )

    # --- Press: pre_contact → press position ---
    press_target, press_ik_audit = _estimate_press_joint_targets(
        pre_tuple, press_distance_m, joints,
    )
    press_waypoints = _interpolate_waypoints(
        pre_tuple, press_target, _required_waypoints(pre_tuple, press_target)
    )

    # --- Retract: press position → pre_contact ---
    retract_waypoints = _interpolate_waypoints(
        press_target, pre_tuple, _required_waypoints(press_target, pre_tuple)
    )

    # --- Return: pre_contact → standby ---
    if num_return_waypoints is None:
        num_return_waypoints = _required_waypoints(pre_tuple, start)
    if num_return_waypoints < 1:
        raise RouteError("num_return_waypoints must be >= 1")
    return_waypoints = _interpolate_waypoints(
        pre_tuple, start, num_return_waypoints
    )

    # --- Consolidate all waypoints ---
    all_waypoints = (
        outbound_waypoints + press_waypoints + retract_waypoints + return_waypoints
    )
    all_phases = (
        ["outbound"] * len(outbound_waypoints)
        + ["press"] * len(press_waypoints)
        + ["retract"] * len(retract_waypoints)
        + ["return"] * len(return_waypoints)
    )

    waypoint_records = []
    for idx, (wp, phase) in enumerate(zip(all_waypoints, all_phases)):
        _, moving = validate_multi_joint_target(
            wp,
            all_waypoints[idx - 1] if idx > 0 else start,
            DEFAULT_JOINT_MAX_DELTA_RAD,
        )
        waypoint_records.append({
            "waypoint_id": f"W{idx + 1:03d}",
            "phase": phase,
            "positions_rad": list(wp),
            "moving_joint_indices": moving,
            "moving_joint_names": [JOINT_NAMES[i] for i in moving],
            "tcp_m": list(standard_urdf_fk(wp, joints)[0]),
        })

    # Speed: use minimum among task joints (conservative, same for all T102)
    min_speed = min(DEFAULT_JOINT_MAX_DELTA_RAD)  # placeholder; T102 speed is separate
    t102_speed = 17  # safe default: matches slowest joint
    t102_acc = 1
    speeds = [t102_speed] * len(all_waypoints)
    accs = [t102_acc] * len(all_waypoints)

    simulation = fake_driver_validate_multi(start, all_waypoints, speeds, accs)

    # FK audit
    start_fk, _ = standard_urdf_fk(start, joints)
    pre_fk, _ = standard_urdf_fk(pre_tuple, joints)
    press_fk, _ = standard_urdf_fk(press_target, joints)

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_id": (
            f"{standby['pose_id']}__to__{pre_contact['pose_id']}__"
            f"press_{press_distance_mm:.0f}mm__m2_multi_joint_v1"
        ),
        "status": "offline_validated_candidate",
        "scope": "multi-joint T102 press route; joint-space interpolation + numerical IK press",
        "sources": {
            "press_standby": {
                "path": str(standby["path"].relative_to(REPO_ROOT)),
                "sha256": standby["sha256"],
                "pose_id": standby["pose_id"],
            },
            "pre_contact": {
                "path": str(pre_contact["path"].relative_to(REPO_ROOT)),
                "sha256": pre_contact["sha256"],
                "pose_id": pre_contact["pose_id"],
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
            "t101_command_limits_rad": [
                list(values) for values in T101_COMMAND_LIMITS
            ],
            "default_joint_max_delta_rad": list(DEFAULT_JOINT_MAX_DELTA_RAD),
            "encoder_step_rad": ENCODER_STEP_RAD,
            "command_type": "T102_multi_joint",
        },
        "route_policy": {
            "outbound_waypoint_count": len(outbound_waypoints),
            "press_waypoint_count": len(press_waypoints),
            "retract_waypoint_count": len(retract_waypoints),
            "return_waypoint_count": len(return_waypoints),
            "t102_shared_speed_steps_s": t102_speed,
            "t102_shared_acc_register": t102_acc,
            "automatic_retry": False,
            "automatic_compensation": False,
            "automatic_retract_after_fault": False,
        },
        "endpoints": {
            "press_standby_start_rad": list(start),
            "pre_contact_rad": list(pre_tuple),
            "press_target_rad": list(press_target),
            "retract_target_rad": list(pre_tuple),
            "return_target_rad": list(start),
        },
        "waypoints": waypoint_records,
        "totals": {
            "waypoint_count": len(all_waypoints),
            "outbound_count": len(outbound_waypoints),
            "press_count": len(press_waypoints),
            "retract_count": len(retract_waypoints),
            "return_count": len(return_waypoints),
        },
        "press_step": {
            "press_distance_mm": press_distance_mm,
            "c0_fingertip_translation_m": list(C0_FINGERTIP_TRANSLATION_M),
            "ik_audit": press_ik_audit,
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
            "multi_joint_validation": "passed",
            "t101_limit_validation": "passed",
            "fake_driver_t102_validation": "passed",
            "collision_validation": "pending",
            "keep_out_validation": "pending",
            "现场clearance确认": "required_before_every_real_step",
        },
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "limitations": [
            "multi-joint T102 uses shared speed/acc; cannot use per-joint profiles",
            "numerical IK is approximate; no analytical guarantee of convergence",
            "C0 fingertip translation used without rotation measurement",
            "press direction estimated from hand_tcp quaternion at pre_contact",
            "real motion requires separate narrow authorization and现场clearance",
        ],
    }
    digest_source = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    artifact["canonical_payload_sha256"] = sha256_bytes(digest_source)
    return artifact


def _interpolate_waypoints(
    start: tuple[float, ...],
    end: tuple[float, ...],
    count: int,
) -> list[tuple[float, ...]]:
    """Generate evenly-spaced joint-space waypoints between start and end."""
    if count < 1:
        raise RouteError("waypoint count must be >= 1")
    if count == 1:
        return [end]
    waypoints = []
    for i in range(1, count + 1):
        t = i / count
        wp = tuple(
            start[j] + t * (end[j] - start[j]) for j in range(len(start))
        )
        waypoints.append(wp)
    return waypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--pre-contact",
        default=str(
            REPO_ROOT / "config" / "press" / "poses" / "pre_contact_candidate_v1.yaml"
        ),
    )
    parser.add_argument(
        "--press-distance-mm",
        type=float,
        default=PRESS_DEFAULT_MM,
    )
    parser.add_argument(
        "--outbound-waypoints",
        type=int,
        default=None,
        help="Number of interpolated waypoints for outbound (auto-computed if omitted)",
    )
    parser.add_argument(
        "--return-waypoints",
        type=int,
        default=None,
        help="Number of interpolated waypoints for return (auto-computed if omitted)",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = build_route(
        args.standby,
        args.pre_contact,
        press_distance_mm=args.press_distance_mm,
        num_outbound_waypoints=args.outbound_waypoints,
        num_return_waypoints=args.return_waypoints,
    )
    result = write_artifact_exclusive(artifact, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
