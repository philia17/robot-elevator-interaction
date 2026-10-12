from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from tools.press.m1_no_contact_route import (
    ENCODER_STEP_RAD,
    GRIPPER_INDEX,
    HOLD_TOLERANCE_RAD,
    JOINT_NAMES,
    OUTBOUND_ORDER,
    REPO_ROOT,
    RETURN_ORDER,
    RouteError,
    T101_COMMAND_LIMITS,
    build_route,
    load_recorded_pose,
    write_artifact_exclusive,
)
from tools.press.verify_m1_route_kdl import verify as verify_kdl


STANDBY = REPO_ROOT / "config/press/poses/press_standby_candidate_v2.yaml"
PRE_CONTACT = REPO_ROOT / "config/press/poses/pre_contact_candidate_v1.yaml"


def route():
    return build_route(STANDBY, PRE_CONTACT)


def all_steps(artifact):
    return artifact["outbound"]["steps"] + artifact["return"]["steps"]


def test_current_route_counts_order_holds_and_false_flags():
    artifact = route()
    assert artifact["totals"] == {
        "step_count": 102,
        "outbound_step_count": 51,
        "return_step_count": 51,
    }
    assert artifact["route_policy"]["outbound_joint_order"] == [
        JOINT_NAMES[index] for index in OUTBOUND_ORDER
    ]
    assert artifact["route_policy"]["return_joint_order"] == [
        JOINT_NAMES[index] for index in RETURN_ORDER
    ]
    assert [item["joint"] for item in artifact["outbound"]["tolerance_holds"]] == [
        JOINT_NAMES[4]
    ]
    for key in (
        "execution_ready",
        "motion_authorized",
        "contact_authorized",
        "press_authorized",
    ):
        assert artifact[key] is False


def test_every_step_changes_one_task_joint_and_respects_limits_and_delta():
    artifact = route()
    previous = artifact["endpoints"]["press_standby_start_rad"]
    for step in all_steps(artifact):
        target = step["target_positions_rad"]
        changed = [
            index
            for index, (before, after) in enumerate(zip(previous, target))
            if abs(before - after) > ENCODER_STEP_RAD + 1e-12
        ]
        assert changed == [step["joint_index"]]
        assert step["joint_index"] != GRIPPER_INDEX
        assert abs(step["delta_rad"]) <= step["max_delta_rad"] + 1e-12
        lower, upper = T101_COMMAND_LIMITS[step["joint_index"]]
        assert lower <= step["target_rad"] <= upper
        previous = target


def test_task_endpoint_with_tolerance_hold_and_exact_return():
    artifact = route()
    endpoints = artifact["endpoints"]
    recorded = endpoints["pre_contact_recorded_rad"]
    reached = endpoints["planned_pre_contact_reached_rad"]
    for index in range(5):
        assert abs(reached[index] - recorded[index]) <= HOLD_TOLERANCE_RAD + 1e-12
    assert reached[GRIPPER_INDEX] == endpoints["press_standby_start_rad"][GRIPPER_INDEX]
    assert endpoints["planned_return_reached_rad"] == pytest.approx(
        endpoints["press_standby_start_rad"], abs=1e-12
    )


def test_fake_driver_exercises_every_step_and_auto_disarms():
    validation = route()["fake_driver_validation"]
    assert validation["passed"] is True
    assert validation["step_count"] == 102
    assert validation["t101_payload_count"] == 102
    assert validation["automatic_disarm_count"] == 102
    assert validation["t101_only"] is True
    assert validation["t102_absent"] is True
    assert validation["ros_used"] is False
    assert validation["serial_used"] is False


def test_geometry_metrics_expose_large_excursion_and_orientation_blocker():
    geometry = route()["geometry_audit"]
    assert geometry["translation_metrics_status"] == "offline_diagnostic_pass"
    assert geometry["orientation_reproducibility_status"] == (
        "standard_parent_child_xyzw_matches_stored_tf"
    )
    assert geometry["endpoint_chord_mm"] == pytest.approx(139.520, abs=0.01)
    assert geometry["outbound_polyline_length_mm"] == pytest.approx(485.580, abs=0.02)
    assert geometry["path_to_chord_ratio"] == pytest.approx(3.480, abs=0.002)
    assert geometry["maximum_cross_track_mm"] == pytest.approx(175.825, abs=0.02)
    assert geometry["tcp_aabb_span_mm"] == pytest.approx(
        [96.389, 51.639, 234.379], abs=0.02
    )
    assert geometry["ideal_reverse_tcp_max_error_m"] < 1e-12
    assert max(geometry["source_endpoint_translation_reproduction_m"].values()) < 2e-15
    assert max(geometry["source_endpoint_orientation_error_deg"].values()) < 1e-5
    wrist_stage = geometry["stage_metrics"][0]
    assert wrist_stage["joint"] == JOINT_NAMES[3]
    assert wrist_stage["polyline_arc_mm"] == pytest.approx(259.076, abs=0.02)
    assert wrist_stage["end_tcp_m"][2] == pytest.approx(0.402256, abs=1e-6)


def test_canonical_digest_and_exclusive_checksum(tmp_path):
    artifact = route()
    canonical_digest = artifact.pop("canonical_payload_sha256")
    canonical = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert canonical_digest == hashlib.sha256(canonical).hexdigest()
    artifact["canonical_payload_sha256"] = canonical_digest
    output = tmp_path / "route.json"
    result = write_artifact_exclusive(artifact, output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == result["artifact_sha256"]
    assert Path(result["checksum_path"]).read_text() == (
        f"{result['artifact_sha256']}  route.json\n"
    )
    with pytest.raises(FileExistsError):
        write_artifact_exclusive(artifact, output)


def test_independent_pykdl_endpoint_verification(tmp_path):
    artifact = route()
    output = tmp_path / "route.json"
    write_artifact_exclusive(artifact, output)
    report = verify_kdl(output)
    assert report["status"] == "passed"
    assert report["ros_used"] is False
    assert report["serial_used"] is False
    assert max(item["translation_error_m"] for item in report["endpoints"]) < 2e-15
    assert max(item["orientation_error_deg"] for item in report["endpoints"]) < 1e-5


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(status="accepted"), "recorded_candidate"),
        (lambda value: value.update(execution_ready=True), "execution_ready"),
        (
            lambda value: value["safe_driver"].update(
                joint_target_publisher_count_max=1
            ),
            "safe-driver evidence",
        ),
        (
            lambda value: value.update(joint_names=list(reversed(JOINT_NAMES))),
            "exact six",
        ),
    ],
)
def test_pose_loader_fails_closed(tmp_path, mutate, match):
    value = yaml.safe_load(STANDBY.read_text())
    mutate(value)
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(RouteError, match=match):
        load_recorded_pose(candidate, "press_standby")


def test_source_has_no_ros_serial_or_network_control_imports():
    source_paths = [
        REPO_ROOT / "tools/press/m1_no_contact_route.py",
        REPO_ROOT / "tools/press/verify_m1_route_kdl.py",
    ]
    imported = set()
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"rclpy", "serial", "socket", "requests"})
    text = "\n".join(path.read_text().lower() for path in source_paths)
    assert "/roarm_safe/joint_target" not in text
    assert "set_armed" not in text
