"""Focused tests for the direct, no-motion Fn pre-contact feasibility report."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from press.direct_precontact_feasibility import (
    ACTIVE_FN_POINTER,
    FeasibilityBlocked,
    build_direct_precontact_feasibility_report,
    validate_direct_precontact_feasibility_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
NOW_NS = 1_750_000_000_000_000_000
JOINT_NAMES = [
    "base_link_to_link1", "link1_to_link2", "link2_to_link3",
    "link3_to_link4", "link4_to_link5", "link5_to_gripper_link",
]


def _target():
    stamp_ns = NOW_NS - 100_000_000
    frozen = {
        "target_label": "1", "frame_id": "camera_color_optical_frame", "source_stamp_ns": stamp_ns,
        "header": {"frame_id": "camera_color_optical_frame", "stamp_ns": stamp_ns},
        "camera_point_m": [0.0, 0.0, 0.55],
        "surface_geometry": {
            "panel_outward_normal": [0.0, 0.0, -1.0], "estimate_valid": True, "provenance_complete": True,
        },
    }
    package = {
        "schema": "m0_demo_freeze_v1", "trial_id": "test-freeze", "mode": "headless_one_shot",
        "motion_authorized": False, "target_label": "1", "frozen_target": frozen,
    }
    package["canonical_payload_sha256"] = hashlib.sha256(
        json.dumps(package, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return package


def _active_fn_observation():
    pointer = json.loads(ACTIVE_FN_POINTER.read_text(encoding="utf-8"))
    digest = hashlib.sha256(ACTIVE_FN_POINTER.read_bytes()).hexdigest()
    return {
        "schema": "active_fn_base_observation_v1", "source_stamp_ns": NOW_NS,
        "active_artifact_id": pointer["active_artifact_id"], "pointer_sha256": digest,
        "translation_hand_tcp_m": pointer["translation_m"], "fn_base_m": [0.11, -0.06, 0.52],
    }


def _feedback():
    return {
        "schema": "roarm_safe_feedback_snapshot_v1", "source_stamp_ns": NOW_NS,
        "state": "CONVERGED_DISARMED", "armed": False, "fault": "",
        "joint_names": JOINT_NAMES, "joint_positions_rad": [0.0] * 6,
    }


def _standby():
    return {"role": "press_standby", "pose_id": "standby-test", "joint_names": JOINT_NAMES, "joint_positions_rad": [0.0] * 6}


def test_report_uses_selected_active_fn_and_exact_direct_endpoint():
    report = build_direct_precontact_feasibility_report(
        _target(), active_fn_observation=_active_fn_observation(), current_feedback=_feedback(), standby_record=_standby(),
        bundle_path=BUNDLE, now_unix_ns=NOW_NS,
    )
    assert report["decision"] == "DIAGNOSTIC_FEASIBILITY_ONLY_BLOCKED"
    assert report["active_fn_evidence"]["active_artifact_id"] == "section_c_fingertip_c0_operator_click_best_20260806_v1"
    assert report["Fn_current_base_m"] == [0.11, -0.06, 0.52]
    assert report["Fn_pre_base_m"] == pytest.approx([
        report["A_base_m"][index] + 0.05 * report["N_out_base"][index] for index in range(3)
    ])
    assert report["precontact_distance_m"] == 0.05
    assert report["standby_feedback_evidence"]["matches_recorded_standby"] is True
    for field in ("execution_ready", "motion_authorized", "trajectory_generated", "ik_run", "command_generated", "adapter_consumable"):
        assert report[field] is False
    validate_direct_precontact_feasibility_report(report)


def test_rejects_wrong_active_fn_pointer_or_non_standby_feedback():
    wrong_pointer = _active_fn_observation()
    wrong_pointer["pointer_sha256"] = "0" * 64
    with pytest.raises(FeasibilityBlocked, match="pointer hash"):
        build_direct_precontact_feasibility_report(
            _target(), active_fn_observation=wrong_pointer, current_feedback=_feedback(), standby_record=_standby(),
            bundle_path=BUNDLE, now_unix_ns=NOW_NS,
        )
    active = _active_fn_observation()
    feedback = _feedback()
    feedback["state"] = "COMMAND_ACTIVE"
    with pytest.raises(FeasibilityBlocked, match="CONVERGED_DISARMED"):
        build_direct_precontact_feasibility_report(
            _target(), active_fn_observation=active, current_feedback=feedback, standby_record=_standby(),
            bundle_path=BUNDLE, now_unix_ns=NOW_NS,
        )


def test_standby_mismatch_stays_explicitly_blocked_and_tampering_fails_closed():
    feedback = _feedback()
    feedback["joint_positions_rad"][1] = 0.1
    report = build_direct_precontact_feasibility_report(
        _target(), active_fn_observation=_active_fn_observation(), current_feedback=feedback, standby_record=_standby(),
        bundle_path=BUNDLE, now_unix_ns=NOW_NS,
    )
    assert report["standby_feedback_evidence"]["matches_recorded_standby"] is False
    assert "CURRENT_FEEDBACK_DOES_NOT_MATCH_RECORDED_STANDBY" in report["required_missing_fields"]
    tampered = copy.deepcopy(report)
    tampered["motion_authorized"] = True
    with pytest.raises(FeasibilityBlocked, match="motion_authorized"):
        validate_direct_precontact_feasibility_report(tampered)


def test_module_has_no_ros_or_execution_interface_imports():
    source = Path(__file__).with_name("direct_precontact_feasibility.py").read_text(encoding="utf-8")
    for forbidden in ("import rclpy", "create_publisher", "create_client", "ActionClient", "serial.Serial", "subprocess"):
        assert forbidden not in source
