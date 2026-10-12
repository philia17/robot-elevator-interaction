"""Tests for the camera-driven demo route generator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import tools.press.m3_camera_route as camera_route

from tools.press.m0_target_to_base import (
    DEFAULT_BUNDLE,
    load_relative_bundle,
    quat_xyzw_to_rotation,
    rotate,
)
from tools.press.m1_no_contact_route import (
    URDF_PATH,
    _load_urdf_joints,
    load_recorded_pose,
)
from tools.press.m2_press_task import MultiJointPressTask, PressTaskState
from tools.press.m1_continuous_return_task import TaskAuthorization, measured_digest
from tools.press.m3_camera_route import build_camera_route, fingertip_from_joints
from tools.press.m1_no_contact_route import RouteError

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDBY = (
    REPO_ROOT
    / "config"
    / "press"
    / "poses"
    / "press_standby_manual_20260804_preflight.yaml"
)
SEED = REPO_ROOT / "config" / "press" / "poses" / "pre_contact_candidate_v1.yaml"


def _synthetic_frozen_near_taught(tmp_path: Path) -> Path:
    """Build a camera-frame target whose base-frame button lies near taught pre."""
    bundle = load_relative_bundle(DEFAULT_BUNDLE)
    seed = load_recorded_pose(SEED, "pre_contact")
    joints = _load_urdf_joints(URDF_PATH)
    finger, quat = fingertip_from_joints(seed["positions"], joints)
    qx, qy, qz, qw = quat
    x_dir = [
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy + qz * qw),
        2 * (qx * qz - qy * qw),
    ]
    norm = math.sqrt(sum(v * v for v in x_dir))
    x_dir = [v / norm for v in x_dir]
    outward = [-v for v in x_dir]
    button = [finger[i] - 0.05 * outward[i] for i in range(3)]
    button[0] += 0.005
    button[1] -= 0.005
    matrix = quat_xyzw_to_rotation(bundle["rotation_xyzw"])
    rt = [[matrix[j][i] for j in range(3)] for i in range(3)]
    t = bundle["translation_m"]
    delta = [button[i] - t[i] for i in range(3)]
    p_cam = rotate(rt, delta)
    n_cam = rotate(rt, outward)
    nn = math.sqrt(sum(v * v for v in n_cam))
    n_cam = [v / nn for v in n_cam]
    frozen = {
        "target_label": "1",
        "frame_id": "camera_optical_frame",
        "camera_point_m": {"x": p_cam[0], "y": p_cam[1], "z": p_cam[2]},
        "surface_geometry": {
            "estimate_valid": True,
            "provenance_complete": False,
            "panel_outward_normal": {
                "x": n_cam[0],
                "y": n_cam[1],
                "z": n_cam[2],
            },
        },
    }
    path = tmp_path / "frozen_button_1.json"
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    return path


def test_camera_route_fake_driver_and_schema(tmp_path):
    frozen = _synthetic_frozen_near_taught(tmp_path)
    artifact = build_camera_route(
        frozen,
        STANDBY,
        seed_pre_contact_path=SEED,
    )
    assert artifact["schema"] == "m2_multi_joint_route_v1"
    assert artifact["route_origin"] == "camera_driven_demo"
    assert artifact["temporary_fallback"] is False
    assert artifact["execution_ready"] is False
    assert artifact["fake_driver_validation"]["passed"] is True
    assert artifact["fake_driver_validation"]["t102_only"] is True
    assert artifact["totals"]["waypoint_count"] == len(artifact["waypoints"])
    assert artifact["totals"]["outbound_count"] >= 1
    assert artifact["totals"]["press_count"] >= 1
    assert artifact["totals"]["retract_count"] >= 1
    assert artifact["totals"]["return_count"] >= 1
    assert artifact["press_step"]["pre_ik_audit"]["converged"] is True
    assert artifact["press_step"]["ik_audit"]["converged"] is True
    assert artifact["camera_target"]["quality"] == "diagnostic"
    assert artifact["gates"]["calibration_id_binding"] == "passed"


def test_camera_route_rejected_on_bad_calibration(tmp_path):
    frozen = _synthetic_frozen_near_taught(tmp_path)
    bad = json.loads(DEFAULT_BUNDLE.read_text())
    bad["calibration_id"] = "nope"
    bundle_path = tmp_path / "bad_bundle.json"
    bundle_path.write_text(json.dumps(bad))
    with pytest.raises(RouteError, match="calibration_id mismatch"):
        build_camera_route(
            frozen,
            STANDBY,
            seed_pre_contact_path=SEED,
            bundle_path=bundle_path,
        )


def test_unreachable_target_requires_explicit_fallback(tmp_path):
    frozen = {
        "target_label": "1",
        "camera_point_m": {"x": 0.0, "y": 0.0, "z": 5.0},
        "surface_geometry": {
            "estimate_valid": True,
            "provenance_complete": False,
            "panel_outward_normal": {"x": 0.0, "y": 0.0, "z": -1.0},
        },
    }
    path = tmp_path / "far.json"
    path.write_text(json.dumps(frozen))
    with pytest.raises(RouteError, match="pre-contact IK failed"):
        build_camera_route(path, STANDBY, seed_pre_contact_path=SEED)
    artifact = build_camera_route(
        path,
        STANDBY,
        seed_pre_contact_path=SEED,
        allow_taught_pre_contact_fallback=True,
    )
    assert artifact["temporary_fallback"] is True
    assert artifact["status"] == "offline_validated_temporary_fallback"
    assert artifact["fake_driver_validation"]["passed"] is True


def test_m2_press_task_consumes_camera_route(tmp_path):
    frozen = _synthetic_frozen_near_taught(tmp_path)
    artifact = build_camera_route(frozen, STANDBY, seed_pre_contact_path=SEED)
    out = tmp_path / "camera_route.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    task = MultiJointPressTask(out)
    start = tuple(artifact["endpoints"]["press_standby_start_rad"])
    task.authorize(
        TaskAuthorization(
            token_id="tok-demo-1",
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(start),
            expires_monotonic=9999.0,
            confirmation_text="offline camera-route dry run",
        ),
        start,
        100.0,
    )
    assert task.state is PressTaskState.READY_NEXT_STEP

    now = 101.0
    healthy = {
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "fault": "",
        "torque_enabled": [True] * 6,
        "voltage": 12.12,
        "joint_target_publisher_count": 0,
    }
    while task.state in (
        PressTaskState.READY_NEXT_STEP,
        PressTaskState.AWAIT_PHASE_CONFIRMATION,
    ):
        if task.state is PressTaskState.AWAIT_PHASE_CONFIRMATION:
            task.confirm_phase(task.next_phase, f"operator confirms {task.next_phase}")
            continue
        candidate = task.issue_next(now)
        if candidate is None:
            continue
        task.observe(candidate.target_positions_rad, healthy, now + 0.5)
        now += 1.0

    assert task.state is PressTaskState.AWAIT_POST_VERIFY
    task.record_post_verify(
        {
            "outcome": "FLOW_PASS_NO_LIGHT_EXPECTED",
            "trial_id": "m0b_trial_camera_demo_0001",
            "target_label": "1",
        }
    )
    assert task.state is PressTaskState.COMPLETE_VERIFIED


def test_cartesian_ik_holds_tip_elevation_and_roll():
    joints = _load_urdf_joints(URDF_PATH)
    # Near-horizontal fingertip pose observed on the live AGX Quick Control arm.
    seed = (
        0.042951462,
        -1.182699187,
        2.330116817,
        0.46786414,
        -1.515573018,
        0.0030679615897932777,
    )
    from tools.press.m3_camera_route import (
        fingertip_axis_elevation_rad,
        solve_cartesian_ik_hold_orientation,
        tool_x_elevation_rad,
    )

    tip, quat = fingertip_from_joints(seed, joints)
    elev0 = fingertip_axis_elevation_rad(quat)
    hand0 = tool_x_elevation_rad(quat)
    goal = list(tip)
    goal[2] += 0.05
    target, audit = solve_cartesian_ik_hold_orientation(
        seed,
        goal,
        joints,
        tolerance_mm=1.5,
        max_iters=40,
        elevation_weight=2.0,
        hold_elevation_rad=elev0,
        hold_hand_tcp_elevation_rad=hand0,
        hand_tcp_elevation_weight=0.45,
    )
    assert audit["converged"] is True
    assert abs(target[4] - seed[4]) < 1e-12
    assert abs(target[5] - seed[5]) < 1e-12
    assert abs(audit["elevation_error_deg"]) < 0.8
    tip2, quat2 = fingertip_from_joints(target, joints)
    assert abs(fingertip_axis_elevation_rad(quat2) - elev0) < math.radians(0.8)
    assert math.dist(tip2, goal) * 1000.0 < 3.0


def test_cartesian_ik_locks_standby_elevation_across_steps():
    """Teleop steps keep C0 fingertip horizontal and hand_tcp standby tilt."""
    joints = _load_urdf_joints(URDF_PATH)
    seed = (
        0.042951462,
        -1.182699187,
        2.330116817,
        0.46786414,
        -1.515573018,
        0.0030679615897932777,
    )
    from tools.press.m3_camera_route import (
        fingertip_axis_elevation_rad,
        solve_cartesian_ik_hold_orientation,
        tool_x_elevation_rad,
    )

    tip, quat = fingertip_from_joints(seed, joints)
    # Teleop contract: fingertip axis world-horizontal; hand_tcp keeps standby lean.
    elev0 = 0.0
    hand0 = tool_x_elevation_rad(quat)
    current = seed
    for dx in (0.016, 0.016, 0.020, 0.020, -0.016):
        tip_now, _ = fingertip_from_joints(current, joints)
        goal = list(tip_now)
        goal[0] += dx
        target, audit = solve_cartesian_ik_hold_orientation(
            current,
            goal,
            joints,
            tolerance_mm=2.0,
            max_iters=50,
            elevation_weight=2.0,
            elevation_tolerance_deg=0.8,
            hold_elevation_rad=elev0,
            hold_hand_tcp_elevation_rad=hand0,
            hand_tcp_elevation_weight=0.45,
        )
        assert audit["converged"] is True, audit
        tip2, quat2 = fingertip_from_joints(target, joints)
        assert abs(fingertip_axis_elevation_rad(quat2) - elev0) < math.radians(0.8)
        assert abs(tool_x_elevation_rad(quat2) - hand0) < math.radians(3.0)
        assert abs(target[4] - seed[4]) < 1e-12
        assert math.dist(tip2, goal) * 1000.0 < 5.0
        current = target


def test_fingertip_position_offset_and_long_axis_are_independent(monkeypatch):
    """Changing the position calibration must not redefine the physical axis."""
    joints = _load_urdf_joints(URDF_PATH)
    seed = load_recorded_pose(SEED, "pre_contact")["positions"]

    tip_before, quat_before = fingertip_from_joints(seed, joints)
    elevation_before = camera_route.fingertip_axis_elevation_rad(quat_before)

    monkeypatch.setattr(
        camera_route,
        "TIP_POSITION_OFFSET_HAND_M",
        (0.0, 0.0, 0.250),
    )
    tip_after, quat_after = fingertip_from_joints(seed, joints)
    elevation_after = camera_route.fingertip_axis_elevation_rad(quat_after)

    assert math.dist(tip_before, tip_after) > 0.05
    assert quat_after == pytest.approx(quat_before, abs=1e-12)
    assert elevation_after == pytest.approx(elevation_before, abs=1e-12)


@pytest.mark.parametrize(
    ("quat_xyzw", "expected_elevation"),
    (
        pytest.param((0.0, 0.0, 0.0, 1.0), 0.0, id="identity_is_horizontal"),
        pytest.param(
            (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
            0.0,
            id="yaw_90_remains_horizontal",
        ),
        pytest.param(
            (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
            -math.pi / 2.0,
            id="pitch_plus_90_points_down",
        ),
        pytest.param(
            (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5)),
            math.pi / 2.0,
            id="pitch_minus_90_points_up",
        ),
    ),
)
def test_fingertip_long_axis_known_rotations(quat_xyzw, expected_elevation):
    """The independent hand-TCP +X long axis has known world elevations."""
    actual = camera_route.fingertip_axis_elevation_rad(
        quat_xyzw, axis_hand=(1.0, 0.0, 0.0)
    )
    assert actual == pytest.approx(expected_elevation, abs=1e-9)


def test_gripper_is_excluded_from_fingertip_fk_axis_and_cartesian_ik():
    joints = _load_urdf_joints(URDF_PATH)
    seed = list(load_recorded_pose(SEED, "pre_contact")["positions"])
    seed_closed = tuple(seed[:5] + [0.0])
    seed_open = tuple(seed[:5] + [1.2])

    tip_closed, quat_closed = fingertip_from_joints(seed_closed, joints)
    tip_open, quat_open = fingertip_from_joints(seed_open, joints)
    assert tip_open == pytest.approx(tip_closed, abs=1e-12)
    assert quat_open == pytest.approx(quat_closed, abs=1e-12)
    assert camera_route.fingertip_axis_elevation_rad(quat_open) == pytest.approx(
        camera_route.fingertip_axis_elevation_rad(quat_closed), abs=1e-12
    )

    assert camera_route.GRIPPER_INDEX not in camera_route.TASK_JOINT_INDICES
    for gripper_seed in (seed_closed, seed_open):
        target, audit = camera_route.solve_cartesian_ik_hold_orientation(
            gripper_seed,
            tip_closed,
            joints,
            max_iters=2,
        )
        assert target[camera_route.GRIPPER_INDEX] == pytest.approx(
            gripper_seed[camera_route.GRIPPER_INDEX], abs=1e-12
        )
        assert camera_route.GRIPPER_INDEX in audit["held_joint_indices"]
        assert camera_route.GRIPPER_INDEX not in audit["moving_joint_indices"]
