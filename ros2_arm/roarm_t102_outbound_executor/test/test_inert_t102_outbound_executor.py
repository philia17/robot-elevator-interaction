"""Pure/source tests; never import rclpy or touch a ROS graph."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parents[1]
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from roarm_t102_outbound_executor.admission import (
    AdmissionError,
    canonical_snapshot,
    make_ticket,
)
from press.t102_fast_runtime_envelope import ENCODER_STEP_RAD
from press.t102_v2_execution_core import (
    DEFAULT_V2_PACKAGE,
    V2ExecutionBlocked,
    V2ExecutionMachine,
    load_v2_package,
)


NODE = PKG_ROOT / "roarm_t102_outbound_executor" / "node.py"


def _snapshot(label="BUTTON 1", provenance=True):
    unsigned = {
        "schema": "m0_demo_freeze_v1",
        "trial_id": "test",
        "mode": "live_freeze_and_prepare",
        "motion_authorized": False,
        "target_label": label,
        "frozen_target": {
            "target_label": label,
            "detection_id": "button_004",
            "source_stamp_ns": 123,
            "frame_id": "camera_optical_frame",
            "surface_geometry": {
                "estimate_valid": True,
                "provenance_complete": provenance,
            },
            "motion_authorized": False,
        },
    }
    text = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    value = dict(unsigned)
    value["canonical_payload_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    final = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return final, hashlib.sha256(final.encode()).hexdigest()


def test_prepare_snapshot_requires_exact_button1_schema_hash_and_provenance():
    text, digest = _snapshot()
    assert canonical_snapshot(text, digest)["schema"] == "m0_demo_freeze_v1"
    with pytest.raises(AdmissionError, match="Button 1"):
        bad, bad_hash = _snapshot(label="BUTTON 2")
        canonical_snapshot(bad, bad_hash)
    with pytest.raises(AdmissionError, match="provenance"):
        bad, bad_hash = _snapshot(provenance=False)
        canonical_snapshot(bad, bad_hash)
    with pytest.raises(AdmissionError, match="SHA-256"):
        canonical_snapshot(text, "0" * 64)


def test_ticket_is_ten_second_one_use_identity_binding():
    ticket = make_ticket(
        executor_instance_nonce="nonce",
        package_sha256="a" * 64,
        target_snapshot_sha256="b" * 64,
        ui_request_id="ui-1",
        start_positions_rad=[0.0] * 6,
        feedback_stamp_ns=123,
        now_monotonic=5.0,
        ttl_s=10.0,
    )
    ticket.validate_for_move(
        ticket.authorization_id,
        "b" * 64,
        14.999,
        executor_instance_nonce="nonce",
        package_sha256="a" * 64,
    )
    with pytest.raises(AdmissionError, match="expired"):
        ticket.validate_for_move(
            ticket.authorization_id,
            "b" * 64,
            15.0,
            executor_instance_nonce="nonce",
            package_sha256="a" * 64,
        )


def test_shared_machine_admits_three_async_handoffs_and_final_only():
    package = load_v2_package(DEFAULT_V2_PACKAGE)
    start = package["start"]["positions_rad"]
    machine = V2ExecutionMachine(package, start, first_published_monotonic=0.0)
    assert machine.first_target == tuple(package["outbound_targets_rad"][0])
    now = 1.0
    releases = []
    segment = list(start)
    for transition in range(3):
        target = package["outbound_targets_rad"][transition]
        # A broad asynchronous state, not the old 0.75-0.85 band.
        actual = [a + 0.9 * (b - a) for a, b in zip(segment, target)]
        release = machine.consider_feedback(
            actual, feedback_age_s=0.01, now_monotonic=now
        )
        assert release is not None
        releases.append(release)
        segment = actual
        now += 1.0
    assert [item.target_index for item in releases] == [1, 2, 3]
    assert machine.complete_handoffs
    terminal = machine.validate_terminal(
        "COARSE_PROGRESS_DISARMED", package["endpoint"]["joint_positions_rad"]
    )
    assert terminal["maximum_final_joint_error_rad"] == 0.0
    with pytest.raises(V2ExecutionBlocked):
        machine.validate_terminal("DISARMED:operator", package["endpoint"]["joint_positions_rad"])


def test_dynamic_node_prepare_has_no_motion_resources_and_move_is_triple_gated():
    source = NODE.read_text()
    tree = ast.parse(source)
    prepare = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_prepare")
    prepare_source = ast.get_source_segment(source, prepare) or ""
    for forbidden in (
        "create_publisher", "create_client", "call_async", "send_goal_async",
        "_acquire_speed_lease", "set_armed", "subprocess", "read_text", "read_bytes",
    ):
        assert forbidden not in prepare_source
    assert "prepare_dynamic_plan" in prepare_source
    assert 'count_publishers("/roarm_safe/joint_target")' in prepare_source
    execute = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute")
    execute_source = ast.get_source_segment(source, execute) or ""
    assert "_create_execution_resources" in execute_source
    assert "verify_frozen_plan" in execute_source
    assert "_publish_exact_waypoint_once" in execute_source
    assert "goal_handle.abort()" in execute_source
    assert "goal_handle.succeed()" in execute_source
    for forbidden in (
        "return_to_configured_standby",
        "clear_fault", "set_torque", "subprocess", "checker", "URDF",
    ):
        assert forbidden not in execute_source

    validation = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_execute_validation"
    )
    validation_source = ast.get_source_segment(source, validation) or ""
    assert "PRECONTACT_HANDOFF_VALIDATED_NO_MOTION" in validation_source
    for forbidden in ("create_publisher", "create_client", "_call_arm", "_call_lease"):
        assert forbidden not in validation_source

    goal = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_goal"
    )
    goal_source = ast.get_source_segment(source, goal) or ""
    assert "VALIDATE_CONFIRMATION_TOKEN" in goal_source
    assert "GoalResponse.ACCEPT" in goal_source
    assert "MOVE_CONFIRMATION_TOKEN" in goal_source
    assert "_move_gate()" in goal_source
    assert "MOVE_RUNTIME_DISABLED" in source


def test_typed_interfaces_bind_session_and_terminal_resource_facts():
    prepare = (REPO_ROOT / "ros2_perception/ros2_ws/src/elevator_interfaces/srv/PreparePrecontact.srv").read_text()
    move = (REPO_ROOT / "ros2_perception/ros2_ws/src/elevator_interfaces/action/PrecontactMove.action").read_text()
    xy = (REPO_ROOT / "ros2_perception/ros2_ws/src/elevator_interfaces/action/PrecontactXY.action").read_text()
    assert "string session_id" in prepare
    assert move.count("string session_id") >= 3
    for token in ("endpoint_max_joint_error_rad", "terminal_joint_target_publishers", "terminal_precontact_lease_active", "terminal_armed"):
        assert token in move
    for token in ("MEASURE", "APPLY_ONE_XY", "correction_count", "wrist_frame_stamp_ns", "candidate_target_rad"):
        assert token in xy


def test_live_has_no_motion_api_and_old_xy_stream_path_is_not_reused():
    live = (REPO_ROOT / "tools/button1_precontact_live_view.py").read_text()
    tree = ast.parse(live)
    imported_modules = {str(node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "button1_xy_align_live_view" not in imported_modules
    assert "press.button1_xy_align" not in imported_modules
    assert "ActionClient" not in live
    assert "create_client(" not in live
    assert 'create_publisher(JointState, "/roarm_safe/joint_target"' not in live


def test_normal_runtime_source_has_zero_heavy_clearance_dependency():
    paths = [
        NODE,
        REPO_ROOT / "tools/press/t102_v2_execution_core.py",
        REPO_ROOT / "tools/press/t102_session_template_core.py",
        REPO_ROOT / "tools/press/t102_discrete_xy_core.py",
    ]
    combined = "\n".join(path.read_text().lower() for path in paths)
    for forbidden in (
        "t102_prototype_continuous_clearance",
        "import subprocess",
        "from subprocess",
        "import trimesh",
        "import urdf",
        "xacro",
        ".stl",
        "clearance_result_",
    ):
        assert forbidden not in combined


def test_dynamic_config_points_only_to_boot_frozen_policy():
    config = (PKG_ROOT / "config/t102_outbound_executor_v2_live.yaml").read_text()
    assert "planning_enabled: true" in config
    assert "t102_dynamic_precontact_policy_20260816_v1.json" in config
    assert "move_runtime_enabled: true" not in config
    assert "fixed_v2" not in config
