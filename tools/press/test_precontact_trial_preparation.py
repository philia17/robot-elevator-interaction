"""Tests for the sealed no-motion direct-precontact runtime preparation helper."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from press.direct_precontact_feasibility import ACTIVE_FN_POINTER
from press.t102_outbound_route_builder import OBSERVED_FIELD_FAILURE_RESIDUAL_RAD
from press.precontact_trial_preparation import (
    ENCODER_STEP_RAD,
    EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD,
    TrialPreparationBlocked,
    build_supervised_trial_preparation,
    validate_supervised_trial_preparation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
NOW_NS = 1_750_000_000_000_000_000
JOINT_NAMES = [
    "base_link_to_link1", "link1_to_link2", "link2_to_link3",
    "link3_to_link4", "link4_to_link5", "link5_to_gripper_link",
]
STANDBY = [0.046019424, -1.188835111, 2.328582836, 0.46786414, -1.515573018, 0.0]


def _m0():
    stamp_ns = NOW_NS - 100_000_000
    frozen = {
        "target_label": "1", "frame_id": "camera_color_optical_frame", "source_stamp_ns": stamp_ns,
        "detection_id": "button_004",
        "header": {"frame_id": "camera_color_optical_frame", "stamp_ns": stamp_ns},
        # Current Button-1 diagnostic freeze geometry.  Its static candidate
        # needs exactly three cap-bounded simultaneous T102 segments.
        "camera_point_m": [0.1216795, -0.0613243, 0.43],
        "surface_geometry": {
            "panel_outward_normal": [0.012942, -0.050689, -0.998631], "estimate_valid": True, "provenance_complete": True,
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


def _fn():
    pointer = json.loads(ACTIVE_FN_POINTER.read_text(encoding="utf-8"))
    return {
        "schema": "active_fn_base_observation_v1", "source_stamp_ns": NOW_NS,
        "active_artifact_id": pointer["active_artifact_id"],
        "pointer_sha256": hashlib.sha256(ACTIVE_FN_POINTER.read_bytes()).hexdigest(),
        "translation_hand_tcp_m": pointer["translation_m"], "fn_base_m": [0.20, 0.01, 0.18],
    }


def _feedback():
    return {
        "schema": "roarm_safe_feedback_snapshot_v1", "source_stamp_ns": NOW_NS,
        "state": "CONVERGED_DISARMED", "armed": False, "fault": "",
        "joint_names": JOINT_NAMES, "joint_positions_rad": list(STANDBY),
    }


def _standby():
    return {"role": "press_standby", "pose_id": "standby-test", "joint_names": JOINT_NAMES, "joint_positions_rad": list(STANDBY)}


def test_seals_current_feedback_ik_and_five_segment_conservative_route_without_authority():
    prepared = build_supervised_trial_preparation(
        _m0(), active_fn_observation=_fn(), current_feedback=_feedback(), standby_record=_standby(),
        bundle_path=BUNDLE, now_unix_ns=NOW_NS,
    )
    assert prepared["route"]["start_positions_rad"] == STANDBY
    assert len(prepared["route"]["setpoints_rad"]) == 5
    assert prepared["static_ik_evidence"]["converged"] is True
    for field in ("execution_ready", "motion_authorized", "arm_authorized", "command_generated", "adapter_consumable", "operator_ticket_issued"):
        assert prepared[field] is False
    assert prepared["route"]["execution_ready"] is False
    evidence = prepared["standby_binding_evidence"]
    assert evidence["standby_pose_id"] == "standby-test"
    assert evidence["max_feedback_to_standby_residual_rad"] == 0.0
    assert evidence["max_feedback_to_standby_residual_encoder_steps"] == 0.0
    assert evidence["route_start_matches_standby_within_executor_binding"] is True
    assert evidence["executor_route_start_binding_tolerance_rad"] == EXECUTOR_ROUTE_START_BINDING_TOLERANCE_RAD
    # The preparation never claims physical success and explicitly documents
    # that the offline route cannot guarantee convergence without driver-side
    # or hardware changes.
    assert prepared["physical_convergence_guaranteed"] is False
    blocker = prepared["physical_blocker"]
    assert blocker["execution_blocked"] is True
    assert blocker["segmentation_alone_cannot_fix_undertravel"] is True
    assert blocker["observed_field_failure_exceeded_soft_band"] is True
    assert blocker["observed_field_failure_max_residual_rad"] == OBSERVED_FIELD_FAILURE_RESIDUAL_RAD
    assert blocker["driver_soft_converge_band_deg"] == prepared["route"]["driver_soft_converge_band_deg"]
    assert "OFFLINE_ROUTE_CANNOT_GUARANTEE_PHYSICAL_CONVERGENCE" in prepared["limitations"]
    validate_supervised_trial_preparation(prepared)


def test_frozen_package_without_detection_identity_is_rejected():
    package = _m0()
    del package["frozen_target"]["detection_id"]
    with pytest.raises(TrialPreparationBlocked, match="detection_id"):
        build_supervised_trial_preparation(
            package, active_fn_observation=_fn(), current_feedback=_feedback(), standby_record=_standby(),
            bundle_path=BUNDLE, now_unix_ns=NOW_NS,
        )


def test_current_feedback_is_route_start_not_recorded_standby_substitute():
    feedback = _feedback()
    feedback["joint_positions_rad"][0] += 0.01
    prepared = build_supervised_trial_preparation(
        _m0(), active_fn_observation=_fn(), current_feedback=feedback, standby_record=_standby(),
        bundle_path=BUNDLE, now_unix_ns=NOW_NS,
    )
    assert prepared["route"]["start_positions_rad"] == feedback["joint_positions_rad"]
    assert prepared["route"]["start_positions_rad"] != STANDBY
    residual = prepared["standby_binding_evidence"]["max_feedback_to_standby_residual_rad"]
    assert abs(residual - 0.01) < 1e-12
    assert abs(
        prepared["standby_binding_evidence"]["max_feedback_to_standby_residual_encoder_steps"]
        - 0.01 / ENCODER_STEP_RAD
    ) < 1e-9
    # A stale standby record is reported as out of binding, never silently
    # substituted for the fresh measured start.
    assert prepared["standby_binding_evidence"]["route_start_matches_standby_within_executor_binding"] is False


def test_rejects_bad_feedback_and_tampering_fails_closed():
    feedback = _feedback()
    feedback["armed"] = True
    with pytest.raises(TrialPreparationBlocked, match="CONVERGED_DISARMED"):
        build_supervised_trial_preparation(
            _m0(), active_fn_observation=_fn(), current_feedback=feedback, standby_record=_standby(),
            bundle_path=BUNDLE, now_unix_ns=NOW_NS,
        )
    prepared = build_supervised_trial_preparation(
        _m0(), active_fn_observation=_fn(), current_feedback=_feedback(), standby_record=_standby(),
        bundle_path=BUNDLE, now_unix_ns=NOW_NS,
    )
    tampered = copy.deepcopy(prepared)
    tampered["execution_ready"] = True
    with pytest.raises(TrialPreparationBlocked, match="execution_ready"):
        validate_supervised_trial_preparation(tampered)
    tampered = copy.deepcopy(prepared)
    tampered["physical_convergence_guaranteed"] = True
    with pytest.raises(TrialPreparationBlocked, match="physical convergence"):
        validate_supervised_trial_preparation(tampered)
    tampered = copy.deepcopy(prepared)
    tampered["physical_blocker"]["execution_blocked"] = False
    with pytest.raises(TrialPreparationBlocked, match="execution blocked"):
        validate_supervised_trial_preparation(tampered)


def test_module_has_no_ros_or_robot_control_interface():
    source = Path(__file__).with_name("precontact_trial_preparation.py").read_text(encoding="utf-8")
    for forbidden in ("import rclpy", "create_publisher", "create_client", "ActionClient", "serial.Serial", "subprocess", "ros2 "):
        assert forbidden not in source
