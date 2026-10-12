#!/usr/bin/env python3
"""Unit tests for button-1 panel-tangent align helpers (no ROS)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from press.button1_xy_align import (
    TARGET_LABEL,
    camera_delta_to_base,
    lock_button1_from_m0_summary,
    lock_button1_from_semantics_and_observations,
    qc_style_tangent_axis_step,
    tangent_delta_camera_m,
    tangent_residual_camera,
)


def test_lock_requires_button_1_ready():
    with pytest.raises(ValueError, match="label"):
        lock_button1_from_m0_summary(
            {
                "label": "2",
                "ready": True,
                "normal_valid": True,
                "camera_xyz": (0.0, 0.0, 0.5),
                "normal": (0.0, 0.0, -1.0),
            }
        )
    with pytest.raises(ValueError, match="READY"):
        lock_button1_from_m0_summary(
            {
                "label": "1",
                "ready": False,
                "normal_valid": True,
                "camera_xyz": (0.0, 0.0, 0.5),
                "normal": (0.0, 0.0, -1.0),
            }
        )


def test_lock_button1_ok():
    locked = lock_button1_from_m0_summary(
        {
            "label": "1",
            "ready": True,
            "normal_valid": True,
            "depth_valid": True,
            "camera_xyz": (0.1, -0.2, 0.8),
            "normal": (0.0, 0.0, -1.0),
            "confirmations": 5,
            "status": "READY",
        }
    )
    assert locked.label == TARGET_LABEL
    assert locked.camera_xyz_m[2] == pytest.approx(0.8)


def test_tangent_delta_has_zero_normal_and_moves_toward_a():
    # Panel facing camera (−Z). Tip is +10 mm in camera +X from A.
    locked = lock_button1_from_m0_summary(
        {
            "label": "1",
            "ready": True,
            "normal_valid": True,
            "camera_xyz": (0.0, 0.0, 0.5),
            "normal": (0.0, 0.0, -1.0),
        }
    )
    tip = (0.010, 0.0, 0.5)
    residual = tangent_residual_camera(tip, locked.camera_xyz_m, locked.normal_camera)
    assert residual["e_u_m"] == pytest.approx(0.010, abs=1e-6)
    assert residual["tangent_norm_m"] == pytest.approx(0.010, abs=1e-6)

    step = tangent_delta_camera_m(tip, locked, deadband_m=0.003, step_m=0.003)
    assert step["aligned"] is False
    dx, dy, dz = step["delta_camera_m"]
    assert dz == pytest.approx(0.0, abs=1e-9)
    assert dx == pytest.approx(-0.003, abs=1e-6)  # toward A, capped
    assert abs(dy) < 1e-9
    # After one ideal step, error shrinks.
    new_tip = (tip[0] + dx, tip[1] + dy, tip[2] + dz)
    new_res = tangent_residual_camera(
        new_tip, locked.camera_xyz_m, locked.normal_camera
    )
    assert new_res["tangent_norm_m"] < residual["tangent_norm_m"]


def test_aligned_inside_deadband():
    locked = lock_button1_from_m0_summary(
        {
            "label": "1",
            "ready": True,
            "normal_valid": True,
            "camera_xyz": (0.0, 0.0, 0.5),
            "normal": (0.0, 0.0, -1.0),
        }
    )
    tip = (0.002, -0.001, 0.5)
    step = tangent_delta_camera_m(tip, locked, deadband_m=0.005, step_m=0.003)
    assert step["aligned"] is True
    assert step["delta_camera_m"] == (0.0, 0.0, 0.0)


def test_lock_from_semantics_obs_join():
    locked = lock_button1_from_semantics_and_observations(
        [{"detection_id": "button_004", "label": "1", "pressable": True}],
        [
            {
                "detection_id": "button_004",
                "depth_valid": True,
                "x_min": 1.0,
                "y_min": 2.0,
                "x_max": 3.0,
                "y_max": 4.0,
                "x_m": 0.1,
                "y_m": -0.2,
                "z_m": 0.5,
                "surface_geometry": {
                    "estimate_valid": True,
                    "panel_outward_normal_x": 0.0,
                    "panel_outward_normal_y": 0.0,
                    "panel_outward_normal_z": -1.0,
                },
            }
        ],
    )
    assert locked.label == TARGET_LABEL
    assert locked.camera_xyz_m[2] == pytest.approx(0.5)
    assert locked.source_status == "SEMANTICS_OBS_JOIN"


def test_lock_rejects_empty_m0_placeholder():
    with pytest.raises(ValueError, match="empty"):
        lock_button1_from_m0_summary(
            {
                "label": "1",
                "ready": True,
                "normal_valid": True,
                "depth_valid": True,
                "camera_xyz": (0.0, 0.0, 0.0),
                "normal": (0.0, 0.0, -1.0),
            }
        )


def test_camera_delta_to_base_rotation():
    # 90° about Z: camera X -> base Y
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    base = camera_delta_to_base((0.01, 0.0, 0.0), rot)
    assert base[0] == pytest.approx(0.0, abs=1e-9)
    assert base[1] == pytest.approx(0.01, abs=1e-9)
    assert base[2] == pytest.approx(0.0, abs=1e-9)


def test_qc_style_picks_y_first_and_locks_x():
    # Tip is left (+Y) and above (+Z) of A; priority Y then Z; X held.
    tip = (0.40, 0.012, 0.020)
    target = (0.40, 0.0, 0.0)
    normal = (-1.0, 0.0, 0.0)
    tip_lock = (0.45, 0.05, 0.12)  # X lock differs to prove hold
    step = qc_style_tangent_axis_step(
        tip, target, normal, tip_lock, deadband_m=0.005, step_m=0.003
    )
    assert step["aligned"] is False
    assert step["axis"] == 1
    assert step["axis_name"] == "Y"
    assert step["direction"] == -1  # tip +Y of A → move −Y
    goal = step["goal_base_m"]
    assert goal[0] == pytest.approx(tip_lock[0])  # X held
    assert goal[2] == pytest.approx(tip_lock[2])  # Z held while Y active
    assert goal[1] == pytest.approx(tip[1] - 0.003)


def test_qc_style_y_before_z_even_if_z_larger():
    tip = (0.40, 0.012, 0.040)
    target = (0.40, 0.0, 0.0)
    normal = (-1.0, 0.0, 0.0)
    tip_lock = tip
    step = qc_style_tangent_axis_step(
        tip, target, normal, tip_lock, deadband_m=0.005, step_m=0.003
    )
    # |e_z|=40mm > |e_y|=12mm, but Task-1 priority is Y then Z.
    assert step["axis"] == 1
    assert step["axis_name"] == "Y"


def test_qc_style_moves_z_after_y_in_deadband():
    tip = (0.40, 0.002, 0.020)
    target = (0.40, 0.0, 0.0)
    normal = (-1.0, 0.0, 0.0)
    tip_lock = tip
    step = qc_style_tangent_axis_step(
        tip, target, normal, tip_lock, deadband_m=0.005, step_m=0.003
    )
    assert step["axis"] == 2
    assert step["axis_name"] == "Z"
    assert step["direction"] == -1


def test_qc_style_aligned_inside_deadband():
    tip = (0.50, 0.002, 0.003)  # large X ignored for YZ-done
    target = (0.40, 0.0, 0.0)
    normal = (-1.0, 0.0, 0.0)
    step = qc_style_tangent_axis_step(
        tip, target, normal, tip, deadband_m=0.005, step_m=0.003
    )
    assert step["aligned"] is True
    assert step["axis"] is None
