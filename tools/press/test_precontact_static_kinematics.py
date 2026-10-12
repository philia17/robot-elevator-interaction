"""Tests for the isolated diagnostic selected-Fn static IK/FK receipt."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from press.precontact_static_kinematics import (
    ACTIVE_FN_POINTER,
    StaticKinematicsBlocked,
    build_static_kinematics_report,
    validate_static_kinematics_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STANDBY = REPO_ROOT / "config/press/poses/press_standby_manual_20260804_preflight.yaml"


def test_selected_fn_direct_static_ik_converges_but_stays_blocked():
    # Representative Button 1 pre-contact endpoint from the current M0
    # diagnostic convention; this test is offline and cannot command a robot.
    report = build_static_kinematics_report(
        target_fn_base_m=[0.27620, 0.05089, 0.30580], standby_pose_path=STANDBY
    )
    assert report["active_fn_evidence"]["pointer_path"] == str(ACTIVE_FN_POINTER.resolve())
    assert report["active_fn_evidence"]["status"] == "diagnostic_overlay_default"
    assert report["ik_evidence"]["converged"] is True
    assert report["ik_evidence"]["orientation_constrained"] is False
    assert report["joint_limit_evidence"]["all_samples_inside_diagnostic_t101_limits"] is True
    assert report["continuous_collision_checked"] is False
    assert report["execution_ready"] is False
    assert report["motion_authorized"] is False
    assert "ACTIVE_FN_POINTER_DIAGNOSTIC_ONLY_NO_6D_CONTACT_RELEASE" in report["required_blockers"]
    assert "CONTINUOUS_REAR_KEEPOUT_MODEL_AND_SWEEP_RELEASE_MISSING" in report["required_blockers"]
    validate_static_kinematics_report(report)


def test_hash_tampering_or_invalid_target_fails_closed():
    report = build_static_kinematics_report(
        target_fn_base_m=[0.27620, 0.05089, 0.30580], standby_pose_path=STANDBY
    )
    tampered = copy.deepcopy(report)
    tampered["motion_authorized"] = True
    with pytest.raises(StaticKinematicsBlocked, match="motion_authorized"):
        validate_static_kinematics_report(tampered)
    with pytest.raises(StaticKinematicsBlocked, match="target_fn_base_m"):
        build_static_kinematics_report(target_fn_base_m=[0.0, 0.0], standby_pose_path=STANDBY)


def test_module_has_no_ros_serial_or_command_interface():
    source = Path(__file__).with_name("precontact_static_kinematics.py").read_text(encoding="utf-8")
    for forbidden in ("import rclpy", "create_publisher", "create_client", "ActionClient", "serial.Serial", "subprocess", "build_t102"):
        assert forbidden not in source
