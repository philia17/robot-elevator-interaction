"""Focused tests for the offline power-off Button 1 precontact validator."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from press.precontact_poweroff_readiness import (
    ACTIVE_FN_POINTER,
    DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S,
    DEFAULT_R6_BUNDLE,
    POWER_ON_CHECKLIST_IDS,
    PoweroffReadinessBlocked,
    build_poweroff_readiness_report,
    canonical_sha256,
    validate_poweroff_readiness_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIAL_DIR = REPO_ROOT / "temp/precontact_trial_20260811"
MODULE = Path(__file__).with_name("precontact_poweroff_readiness.py")

# Deterministic fixture clock: one second after the frozen source stamp.
STAMP_NS = 1_786_452_000_000_000_000
NOW_NS = STAMP_NS + 1_000_000_000


def _frozen_target_package():
    frozen = {
        "target_label": "1",
        "frame_id": "camera_optical_frame",
        "source_stamp_ns": STAMP_NS,
        "header": {"frame_id": "camera_optical_frame", "stamp_ns": STAMP_NS},
        "detection_id": "button_004",
        "camera_point_m": {
            "x": 0.12146051228046417,
            "y": -0.06129693239927292,
            "z": 0.43,
        },
        "surface_geometry": {
            "panel_outward_normal": {
                "x": 0.007872199421483645,
                "y": -0.05501762401684593,
                "z": -0.9984543502453227,
            },
            "estimate_valid": True,
            "provenance_complete": True,
        },
    }
    package = {
        "schema": "m0_demo_freeze_v1",
        "trial_id": "test-poweroff-freeze",
        "mode": "headless_one_shot",
        "motion_authorized": False,
        "target_label": "1",
        "frozen_target": frozen,
    }
    package["canonical_payload_sha256"] = canonical_sha256(package)
    return package


def _route(frozen):
    route = {
        "schema": "t102_outbound_only_route_v1",
        "route_mode": "OFFLINE_DIAGNOSTIC_ONLY",
        "phase": "OUTBOUND_TO_PRECONTACT",
        "phases": ["OUTBOUND_TO_PRECONTACT"],
        "joint_path": "SIMULTANEOUS_T102_JOINT_SPACE_CHORD",
        "command_interface": "NOT_IMPLEMENTED",
        "frozen_bindings": {
            "m0_target_sha256": canonical_sha256(frozen),
            "fn_sha256": hashlib.sha256(ACTIVE_FN_POINTER.read_bytes()).hexdigest(),
            "r6_sha256": hashlib.sha256(DEFAULT_R6_BUNDLE.read_bytes()).hexdigest(),
            "provenance_sha256": "a" * 64,
        },
        "start_positions_rad": [0.03834952, -1.191903072, 2.331650798, 0.464796179, -1.509437095, 0.007669903589793137],
        "static_ik_candidate_positions_rad": [0.17307129001199317, -0.7591680615929813, 2.0792333442816586, 0.030722599145597756, -1.4897231674298133, 0.007669903589793137],
        "setpoints_rad": [
            [0.1057, -0.9755, 2.2054, 0.2478, -1.4996, 0.007669903589793137],
            [0.17307129001199317, -0.7591680615929813, 2.0792333442816586, 0.030722599145597756, -1.4897231674298133, 0.007669903589793137],
        ],
        "driver_delta_caps_deg": [5.0, 10.0, 10.0, 15.0, 30.0, 15.0],
        "max_profile_delta_fraction": 0.5,
        "segment_delta_caps_deg": [2.5, 5.0, 5.0, 7.5, 15.0, 7.5],
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
        "diagnostic_only": True,
        "required_runtime_gates": [
            "INDEPENDENT_IK_FK_RELEASE",
            "JOINT_LIMIT_AND_CONTINUOUS_KEEPOUT_RELEASE",
            "FRESH_SAFE_DRIVER_FEEDBACK",
            "EXCLUSIVE_MOTION_OWNER",
            "EXPLICIT_OPERATOR_MOVE_TICKET",
        ],
    }
    route["route_sha256"] = canonical_sha256(route)
    return route


def _geometry(frozen, route):
    a = [0.32584954126921023, 0.05410619205558746, 0.3108212132435427]
    n = [-0.9938271544605604, -0.054902331626688006, -0.09640187258958627]
    return {
        "A_base_m": a,
        "N_out_base": n,
        "Fn_pre_base_m": [a[i] + 0.05 * n[i] for i in range(3)],
        "target_snapshot_sha256": canonical_sha256(frozen),
        "route_sha256": route["route_sha256"],
    }


def _valid_inputs():
    frozen = _frozen_target_package()
    route = _route(frozen)
    return frozen, route, _geometry(frozen, route)


def _rebuild_receipt(package):
    """Recompute the frozen-package receipt for the current content."""
    value = copy.deepcopy(package)
    value.pop("canonical_payload_sha256", None)
    value["canonical_payload_sha256"] = canonical_sha256(value)
    return value


def test_valid_sealed_package_route_and_projection_builds_blocked_no_motion_report():
    frozen, route, geometry = _valid_inputs()
    report = build_poweroff_readiness_report(
        frozen, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
    )
    assert report["decision"] == "PREPARED_POWEROFF_BLOCKED_NO_MOTION"
    assert report["target_evidence"]["target_identity"] == "Button 1"
    assert report["target_evidence"]["detection_id"] == "button_004"
    assert report["target_evidence"]["source_stale"] is False
    assert report["route_evidence"]["route_hash_consistent"] is True
    assert report["route_evidence"]["outbound_only"] is True
    assert report["geometry_evidence"]["p_pre_matches_equation"] is True
    assert report["geometry_evidence"]["normal_normalized"] is True
    assert report["frozen_bindings"]["m0_target_sha256"] == canonical_sha256(frozen)
    for blocker in (
        "ROUTE_RELEASE_MISSING",
        "CONTINUOUS_REAR_KEEPOUT_RELEASE_MISSING",
        "SECTION_B_R6_RELATIVE_UNPUBLISHED",
        "FN_CONTACT_CALIBRATION_UNRELEASED",
        "RUNTIME_POWERON_CHECKLIST_PENDING",
        "EXPLICIT_USER_CONFIRMATION_PENDING",
    ):
        assert blocker in report["blocked_reasons"]
    assert all(item["present"] is False for item in report["release_evidence"].values())
    assert {item["id"] for item in report["power_on_checklist"]} == set(
        POWER_ON_CHECKLIST_IDS
    )
    for field in (
        "execution_ready",
        "motion_authorized",
        "arm_authorized",
        "command_generated",
        "adapter_consumable",
        "operator_ticket_issued",
        "contact_authorized",
        "automatic_retry",
    ):
        assert report[field] is False
    assert report["report_grants_motion_authority"] is False
    validate_poweroff_readiness_report(report)


def test_tampered_hashes_and_bindings_fail_closed():
    frozen, route, geometry = _valid_inputs()

    tampered_target = copy.deepcopy(frozen)
    tampered_target["canonical_payload_sha256"] = "f" * 64
    with pytest.raises(PoweroffReadinessBlocked, match="canonical payload hash mismatch"):
        build_poweroff_readiness_report(
            tampered_target, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    tampered_route = copy.deepcopy(route)
    tampered_route["route_sha256"] = "b" * 64
    with pytest.raises(PoweroffReadinessBlocked, match="route SHA-256 mismatch"):
        build_poweroff_readiness_report(
            frozen, tampered_route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    rebound = copy.deepcopy(route)
    rebound["frozen_bindings"]["m0_target_sha256"] = "c" * 64
    rebound.pop("route_sha256", None)
    rebound["route_sha256"] = canonical_sha256(rebound)
    with pytest.raises(PoweroffReadinessBlocked, match="does not match the frozen target"):
        build_poweroff_readiness_report(
            frozen, rebound, geometry_evidence=None, now_unix_ns=NOW_NS
        )

    rehashed_geometry = copy.deepcopy(geometry)
    rehashed_geometry["target_snapshot_sha256"] = "d" * 64
    with pytest.raises(PoweroffReadinessBlocked, match="does not bind the frozen target"):
        build_poweroff_readiness_report(
            frozen, route, geometry_evidence=rehashed_geometry, now_unix_ns=NOW_NS
        )

    report = build_poweroff_readiness_report(
        frozen, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
    )
    tampered_report = copy.deepcopy(report)
    tampered_report["execution_ready"] = True
    with pytest.raises(PoweroffReadinessBlocked, match="execution_ready"):
        validate_poweroff_readiness_report(tampered_report)


def test_non_50mm_geometry_or_nonunit_normal_fails_closed():
    frozen, route, geometry = _valid_inputs()

    wrong_distance = copy.deepcopy(geometry)
    wrong_distance["Fn_pre_base_m"] = [
        wrong_distance["A_base_m"][i] + 0.06 * wrong_distance["N_out_base"][i]
        for i in range(3)
    ]
    with pytest.raises(PoweroffReadinessBlocked, match="0.050"):
        build_poweroff_readiness_report(
            frozen, route, geometry_evidence=wrong_distance, now_unix_ns=NOW_NS
        )

    nonunit = copy.deepcopy(geometry)
    nonunit["N_out_base"] = [0.0, 0.0, -2.0]
    with pytest.raises(PoweroffReadinessBlocked, match="unit length"):
        build_poweroff_readiness_report(
            frozen, route, geometry_evidence=nonunit, now_unix_ns=NOW_NS
        )

    nonfinite = copy.deepcopy(geometry)
    nonfinite["N_out_base"] = [float("nan"), 0.0, -1.0]
    with pytest.raises(PoweroffReadinessBlocked, match="finite"):
        build_poweroff_readiness_report(
            frozen, route, geometry_evidence=nonfinite, now_unix_ns=NOW_NS
        )


def test_stale_source_or_mismatched_identity_fails_closed():
    frozen, route, geometry = _valid_inputs()
    stale_now = STAMP_NS + int(2 * DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S * 1e9)
    report = build_poweroff_readiness_report(
        frozen, route, geometry_evidence=geometry, now_unix_ns=stale_now
    )
    assert "FROZEN_TARGET_SOURCE_STALE" in report["blocked_reasons"]
    assert report["target_evidence"]["source_stale"] is True

    wrong_label = copy.deepcopy(frozen)
    wrong_label["target_label"] = "2"
    wrong_label = _rebuild_receipt(wrong_label)
    with pytest.raises(PoweroffReadinessBlocked, match="Button 1"):
        build_poweroff_readiness_report(
            wrong_label, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    no_detection = copy.deepcopy(frozen)
    no_detection["frozen_target"]["detection_id"] = ""
    no_detection = _rebuild_receipt(no_detection)
    with pytest.raises(PoweroffReadinessBlocked, match="detection_id"):
        build_poweroff_readiness_report(
            no_detection, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    mismatched_frame = copy.deepcopy(frozen)
    mismatched_frame["frozen_target"]["header"]["frame_id"] = "other_frame"
    mismatched_frame = _rebuild_receipt(mismatched_frame)
    with pytest.raises(PoweroffReadinessBlocked, match="frame_id mismatch"):
        build_poweroff_readiness_report(
            mismatched_frame, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    future_stamp = copy.deepcopy(frozen)
    future_stamp["frozen_target"]["source_stamp_ns"] = NOW_NS + 5_000_000_000
    future_stamp["frozen_target"]["header"]["stamp_ns"] = NOW_NS + 5_000_000_000
    future_stamp = _rebuild_receipt(future_stamp)
    with pytest.raises(PoweroffReadinessBlocked, match="in the future"):
        build_poweroff_readiness_report(
            future_stamp, route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )


def test_forbidden_phase_or_retry_press_or_authority_flag_fails_closed():
    frozen, route, geometry = _valid_inputs()

    press_phase = copy.deepcopy(route)
    press_phase["phase"] = "PRESS"
    press_phase["route_sha256"] = canonical_sha256(press_phase)
    with pytest.raises(PoweroffReadinessBlocked, match="outbound pre-contact phase"):
        build_poweroff_readiness_report(
            frozen, press_phase, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    retry_route = copy.deepcopy(route)
    retry_route["automatic_retry"] = True
    retry_route["route_sha256"] = canonical_sha256(retry_route)
    with pytest.raises(PoweroffReadinessBlocked, match="automatic_retry"):
        build_poweroff_readiness_report(
            frozen, retry_route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    press_route = copy.deepcopy(route)
    press_route["press"] = True
    press_route["route_sha256"] = canonical_sha256(press_route)
    with pytest.raises(PoweroffReadinessBlocked, match="press"):
        build_poweroff_readiness_report(
            frozen, press_route, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )

    authorized = copy.deepcopy(route)
    authorized["execution_ready"] = True
    authorized["route_sha256"] = canonical_sha256(authorized)
    with pytest.raises(PoweroffReadinessBlocked, match="execution_ready"):
        build_poweroff_readiness_report(
            frozen, authorized, geometry_evidence=geometry, now_unix_ns=NOW_NS
        )


def test_missing_geometry_or_release_evidence_is_an_explicit_blocker():
    frozen, route, _geometry_value = _valid_inputs()
    report = build_poweroff_readiness_report(
        frozen, route, geometry_evidence=None, now_unix_ns=NOW_NS
    )
    assert report["geometry_evidence"]["present"] is False
    assert "LIVE_PROJECTION_GEOMETRY_EVIDENCE_MISSING" in report["blocked_reasons"]
    for role in (
        "route_release",
        "ik_fk_release",
        "joint_limits_release",
        "rear_keepout_release",
        "field_collision_release",
        "calibration_release",
        "fn_contact_calibration",
    ):
        item = report["release_evidence"][role]
        assert item["present"] is False
        assert item["required"] is True
        assert item["blocker"]
    validate_poweroff_readiness_report(report)


def test_module_imports_only_stdlib_and_never_imports_ros_control_or_processes():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    allowlist = {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "time",
        "typing",
    }
    assert imported <= allowlist
    assert not (imported & {"rclpy", "ros", "control", "serial", "subprocess"})


def test_module_never_builds_ros_control_or_process_interfaces():
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_call_names = {
        "create_publisher",
        "create_subscription",
        "create_service",
        "create_client",
        "create_action_client",
        "ActionClient",
        "system",
        "popen",
        "spawn",
        "start_new_session",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        assert name not in forbidden_call_names, name
    for token in (
        "rclpy",
        "create_publisher",
        "ActionClient",
        "serial.Serial",
        "os.system",
        "subprocess",
    ):
        assert token not in source


def test_real_frozen_trial_artifacts_build_report_when_present():
    paths = (
        TRIAL_DIR / "frozen_target_canonical.json",
        TRIAL_DIR / "sealed_field_trial_route.json",
        TRIAL_DIR / "field_trial_geometry.json",
    )
    if not all(path.is_file() for path in paths):
        pytest.skip("2026-08-11 sealed trial artifacts are not present")
    frozen = json.loads(paths[0].read_text(encoding="utf-8"))
    route = json.loads(paths[1].read_text(encoding="utf-8"))
    geometry = json.loads(paths[2].read_text(encoding="utf-8"))
    # The real freeze is from 2026-08-11; a power-off run the next day is
    # stale by construction, so use a clock far beyond the freeze stamp.
    poweroff_now = STAMP_NS + int(2 * DEFAULT_POWEROFF_SOURCE_AGE_LIMIT_S * 1e9)
    report = build_poweroff_readiness_report(
        frozen, route, geometry_evidence=geometry, now_unix_ns=poweroff_now
    )
    assert report["target_evidence"]["detection_id"] == "button_004"
    assert report["target_evidence"]["source_frame_id"] == "camera_optical_frame"
    assert report["route_evidence"]["route_sha256"] == route["route_sha256"]
    assert report["geometry_evidence"]["p_pre_matches_equation"] is True
    assert report["geometry_evidence"]["normal_unit_length"] == pytest.approx(1.0)
    assert report["frozen_bindings"]["fn_sha256"] == route["frozen_bindings"]["fn_sha256"]
    assert report["frozen_bindings"]["r6_sha256"] == route["frozen_bindings"]["r6_sha256"]
    assert "FROZEN_TARGET_SOURCE_STALE" in report["blocked_reasons"]
    validate_poweroff_readiness_report(report)
