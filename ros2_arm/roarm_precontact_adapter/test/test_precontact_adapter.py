"""Source-level contracts for the inert pre-contact ROS shell.

No test imports rclpy, creates a ROS graph, opens serial, or triggers motion.
The injected fake resource lifecycle is covered in the pure core test suite.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ADAPTER_ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ADAPTER_ROOT / "roarm_precontact_adapter" / "node.py"
PACKAGE_XML = ADAPTER_ROOT / "package.xml"
CONFIG = ADAPTER_ROOT / "config" / "precontact_adapter.yaml"
CORE_SOURCE = ADAPTER_ROOT / "roarm_precontact_adapter" / "precontact_execution_core.py"
PLANNING_SOURCE = ADAPTER_ROOT / "roarm_precontact_adapter" / "precontact_planning_core.py"
WRAPPER_SOURCE = ADAPTER_ROOT.parents[1] / "tools" / "press" / "precontact_execution_core.py"
LAUNCH_SOURCE = ADAPTER_ROOT / "launch" / "precontact_adapter.launch.py"
PREPARE_INTERFACE = (
    ADAPTER_ROOT.parents[1]
    / "ros2_perception/ros2_ws/src/elevator_interfaces/srv/PreparePrecontact.srv"
)


def _function_source(name: str) -> str:
    lines = NODE_SOURCE.read_text().splitlines()
    active = False
    body: list[str] = []
    for line in lines:
        if line.startswith(f"    def {name}("):
            active = True
        elif active and line.startswith("    def "):
            break
        if active:
            body.append(line)
    assert body, f"missing function {name}"
    return "\n".join(body)


def test_startup_has_no_motion_publisher_or_arm_client():
    text = NODE_SOURCE.read_text()
    assert '"/roarm_safe/joint_target"' not in text or "count_publishers" in text
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    create_clients = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "create_client"]
    assert not create_clients
    publishers = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "create_publisher"]
    assert len(publishers) == 1
    source = ast.get_source_segment(text, publishers[0]) or ""
    assert '"/roarm_precontact/status"' in source


def test_m0_subscription_only_populates_private_one_use_planning_cache():
    text = NODE_SOURCE.read_text()
    callback = _function_source("_on_m0_target")
    assert "from elevator_interfaces.msg import M0Target3D" in text
    assert '"/elevator/m0/perception_target_3d"' in text
    assert "canonicalize_button1_m0_freeze_request" in callback
    assert "self._dynamic_prepare.clear()" in callback
    assert "self._dynamic_prepare.observe" in callback
    assert '"normal_direction_status": "UNVERIFIED"' in callback
    assert '"dot_n_out_p_minus_a": 0.0' in callback
    status = _function_source("_publish_status")
    assert "cache_token" not in status
    assert '"token"' not in status


def test_prepare_uses_atomic_dynamic_transaction_and_never_forwards_client_bytes_to_planner():
    prepare = _function_source("_on_prepare")
    planning = PLANNING_SOURCE.read_text()
    assert "with self._core_lock:" in prepare
    assert "self._dynamic_prepare.prepare" in prepare
    assert "self._core = outcome.core" in prepare
    assert prepare.index("self._dynamic_prepare.prepare") < prepare.index("self._core = outcome.core")
    assert "self._planner = NotImplementedPlanner()" in NODE_SOURCE.read_text()
    assert 'event = "PREPARED_NO_MOTION" if outcome.accepted else "REJECTED_NO_MOTION"' in prepare
    planner_call = planning[planning.index("planned = self._planner.plan_cached_freeze(") :]
    planner_call = planner_call[: planner_call.index(")\n") + 2]
    assert "canonical_client_freeze_json" not in planner_call
    assert "client_bytes" not in planner_call
    assert "slot.receipt.token" in planner_call


def test_prepare_interface_separates_raw_m0_request_hash_from_planned_v2_target_hash():
    interface = PREPARE_INTERFACE.read_text()
    assert "client bytes" in interface
    assert "private cache token" in interface
    assert "string planned_target_snapshot_sha256" in interface
    prepare = _function_source("_on_prepare")
    assert "response.planned_target_snapshot_sha256 = outcome.target_snapshot_sha256" in prepare


def test_default_configuration_is_explicitly_inert_and_not_execution_ready():
    text = CONFIG.read_text()
    assert "execution_enabled: false" in text
    assert '"execution_ready":false' in text
    assert "physical_authorized: false" in text
    assert '"outbound_route_artifact":null' in text
    assert "precontact_execution_release_v2" in text


def test_action_path_is_explicitly_not_implemented_and_cannot_claim_success():
    execute = _function_source("_execute")
    assert "NOT_IMPLEMENTED" in execute
    assert "goal_handle.abort()" in execute
    assert "executor returned without convergence evidence" in execute


def test_goal_callback_rejects_invalid_goals_and_reserves_before_accepting():
    goal = _function_source("_goal")
    assert "reserve_goal" in goal
    assert "GoalResponse.REJECT" in goal
    assert "if not accepted" in goal
    assert goal.index("GoalResponse.REJECT") < goal.index("GoalResponse.ACCEPT")


def test_status_reports_observed_joint_target_count_or_unknown_not_hardcoded_zero():
    status = _function_source("_publish_status")
    assert "_joint_target_publisher_count" in status
    assert 'else "unknown"' in status
    assert '"joint_target_publishers": 0' not in status


def test_status_is_periodic_timestamped_and_explicitly_inert():
    text = NODE_SOURCE.read_text()
    status = _function_source("_publish_status")
    assert "self.create_timer(0.5, self._publish_periodic_status)" in text
    assert "secrets.token_hex(16)" in text
    assert '"instance_nonce": self._status_instance_nonce' in status
    assert '"status_sequence": self._status_sequence' in status
    assert "self._status_sequence += 1" in status
    assert '"published_at_unix_ns": time.time_ns()' in status
    assert '"published_at_monotonic_ns": time.monotonic_ns()' in status
    assert '"accepting_precontact": False' in status
    assert '"executor": "NOT_IMPLEMENTED"' in status
    periodic = _function_source("_publish_periodic_status")
    assert 'self._publish_status("INERT", "real motion executor is NOT_IMPLEMENTED")' in periodic


def test_adapter_does_not_import_or_reference_m2_m3_press_route_components():
    text = NODE_SOURCE.read_text().lower()
    forbidden = (
        "roarm_m2_adapter",
        "m2_press_task",
        "m2_multi_joint_route",
        "m3_camera_route",
        "confirm_press",
        "confirm_retract",
        "post_verify",
    )
    assert all(token not in text for token in forbidden)


def test_adapter_never_uses_torque_clear_fault_standby_or_initialize_services():
    text = NODE_SOURCE.read_text()
    forbidden = (
        "/roarm_safe/set_torque",
        "/roarm_safe/clear_fault",
        "/roarm_safe/return_to_configured_standby",
        "/roarm_safe/return_to_configured_initialize",
        "T102",
    )
    assert all(token not in text for token in forbidden)


def test_package_declares_required_generated_interface_dependency():
    text = PACKAGE_XML.read_text()
    assert "<buildtool_depend>ament_python</buildtool_depend>" in text
    assert "<exec_depend>elevator_interfaces</exec_depend>" in text
    assert "<exec_depend>ament_index_python</exec_depend>" in text
    assert "<exec_depend>launch</exec_depend>" in text
    assert "<exec_depend>launch_ros</exec_depend>" in text


def test_installed_layout_owns_core_and_node_uses_only_relative_import(tmp_path):
    node = NODE_SOURCE.read_text()
    assert "from .precontact_execution_core import" in node
    assert "_find_repo_root" not in node
    assert "sys.path" not in node
    assert CORE_SOURCE.is_file()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ADAPTER_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import roarm_precontact_adapter.precontact_execution_core as c; "
            "assert c.RELEASE_SCHEMA == 'precontact_execution_release_v2'",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


def test_tools_execution_core_is_a_logic_free_compatibility_reexport():
    tree = ast.parse(WRAPPER_SOURCE.read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in tree.body)
    assert "roarm_precontact_adapter.precontact_execution_core" in WRAPPER_SOURCE.read_text()


def test_launch_loads_installed_inert_yaml():
    text = LAUNCH_SOURCE.read_text()
    assert "get_package_share_directory" in text
    assert '"precontact_adapter.yaml"' in text
    assert "parameters=[config_path]" in text


def test_production_core_has_no_reset_escape_hatch():
    assert "reset_after_operator_review" not in CORE_SOURCE.read_text()
