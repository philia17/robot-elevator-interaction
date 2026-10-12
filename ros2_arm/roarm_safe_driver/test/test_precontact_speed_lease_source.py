"""Source-level contract for the precontact speed lease in the safe driver.

No test imports rclpy or starts ROS: rosidl generation is unavailable in this
checkout, so the generated service cannot be imported.  These tests pin the
.srv contract, the fail-closed wiring, and the pure manager gates instead.
"""

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
NODE = PACKAGE / "roarm_safe_driver" / "node.py"
CORE = PACKAGE / "roarm_safe_driver" / "core.py"
SRV = (
    REPO
    / "ros2_perception"
    / "ros2_ws"
    / "src"
    / "elevator_interfaces"
    / "srv"
    / "PrecontactSpeedLease.srv"
)
CMAKE = (
    REPO
    / "ros2_perception"
    / "ros2_ws"
    / "src"
    / "elevator_interfaces"
    / "CMakeLists.txt"
)
DEFAULT_CONFIG = PACKAGE / "config" / "agx_t102_smooth_precontact.yaml"


def _method(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_service_contract_has_fixed_fields_and_no_arbitrary_speed():
    text = SRV.read_text(encoding="utf-8")
    request, _, response = text.partition("---")
    assert "bool enable" in request
    assert "string purpose" in request
    assert "string lease_id" in request
    assert "bool success" in response
    assert "string message" in response
    assert "bool active" in response
    # The request must not carry a speed field: the lease applies exactly the
    # fixed 34-steps/s profile and nothing else.  Only the declared fields
    # count; prose comments are ignored.
    request_fields = [
        line
        for line in request.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "speed" not in " ".join(request_fields).lower()


def test_interface_is_wired_into_elevator_interfaces_build():
    assert 'srv/PrecontactSpeedLease.srv' in CMAKE.read_text(encoding="utf-8")


def test_lease_constants_and_pure_gates_live_in_core():
    text = CORE.read_text(encoding="utf-8")
    assert 'PRECONTACT_LEASE_SPEED_STEPS_S = (34, 34, 34, 34, 34, 34)' in text
    assert 'PRECONTACT_LEASE_PURPOSE = "T102_OUTBOUND_PRECONTACT_ONLY"' in text
    assert '"T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE"' in text
    assert "dynamic exact precontact forbids target replace" in text
    assert "dynamic_exact_no_reissue" in text
    assert "class PrecontactSpeedLease" in text
    for gate in (
        "precontact speed lease already active",
        "lease purpose is not accepted",
        "lease_id replay rejected",
        "foreign /roarm_safe/joint_target publisher present",
        "precontact speed lease requires non-dry-run driver",
        "precontact speed lease requires healthy fresh feedback",
        "precontact speed lease requires disarmed idle driver",
        "all six torque flags must be enabled",
        "lease identity does not match",
        "lease release requires disarmed idle driver",
        "lease lost its original speed profile",
    ):
        assert gate in text


def test_node_registers_lease_service_only_with_generated_interface():
    source = NODE.read_text(encoding="utf-8")
    assert '"/roarm_safe/precontact_speed_lease"' in source
    assert "from elevator_interfaces.srv import" in source
    assert "PrecontactSpeedLeaseSrv = None" in source
    assert "if PrecontactSpeedLeaseSrv is not None:" in source
    init = source[source.index("def __init__") : source.index("def _inject_fake_feedback")]
    assert '"/roarm_safe/precontact_speed_lease"' in init
    assert "self._precontact_lease = PrecontactSpeedLease()" in init


def test_lease_callback_passes_controller_gates_and_maps_response_fields():
    source = NODE.read_text(encoding="utf-8")
    callback = source[
        source.index("def _precontact_lease_callback") :
        source.index("def _observe_auto_torque_recovery")
    ]
    assert "self._precontact_lease.enable(" in callback
    assert "self._precontact_lease.release(" in callback
    assert "purpose=str(request.purpose)" in callback
    assert "lease_id=str(request.lease_id)" in callback
    assert 'self.count_publishers("/roarm_safe/joint_target") != 0' in callback
    assert "response.success = ok" in callback
    assert "response.message = message" in callback
    assert "response.active = self._precontact_lease.active" in callback
    assert "PRECONTACT_LEASE_REJECTED" in callback
    assert "lease service failure" in callback
    assert '"dynamic_exact_pid_boost_enabled"' in callback
    assert "build_joint_pid(" in callback
    assert "DYNAMIC_PRECONTACT_PID_P" in callback
    assert "DYNAMIC_PRECONTACT_PID_I" in callback
    assert "build_reset_joint_pid()" in callback
    assert "Dynamic exact PID correction is disabled" in callback


def test_release_identity_precheck_precedes_pid_reset_transport_write():
    source = NODE.read_text(encoding="utf-8")
    callback = _method(source, "_precontact_lease_callback")
    request_branch = next(
        node
        for node in ast.walk(callback)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "request.enable"
    )
    release_branch = "\n".join(
        ast.get_source_segment(source, node) or "" for node in request_branch.orelse
    )
    precheck = release_branch.index("release_precheck(")
    pid_reset = release_branch.index("self.transport.write(build_reset_joint_pid())")
    release = release_branch.index("self._precontact_lease.release(", pid_reset)
    assert precheck < pid_reset < release
    assert "if ok:" in release_branch


def test_dynamic_exact_mode_forces_t102_and_cannot_fall_through_to_t101():
    source = NODE.read_text(encoding="utf-8")
    callback = ast.get_source_segment(source, _method(source, "_command_callback")) or ""
    active_target = callback.index("if self.controller.target is not None:")
    exact_active = callback.index(
        "if self.controller.exact_plan_no_reissue_active:", active_target
    )
    exact_advance = callback.index(
        "self.controller.command_dynamic_exact_advance(", exact_active
    )
    exact_initial = callback.index(
        "elif self.controller.exact_plan_no_reissue_active:", exact_advance
    )
    t102 = callback.index("self.controller.command_multi(target", exact_initial)
    ordinary = callback.index("elif moving > 1", t102)
    t101 = callback.index("self.controller.command(target", ordinary)
    assert active_target < exact_active < exact_advance < exact_initial
    assert exact_initial < t102 < ordinary < t101


def test_manual_standby_initialize_and_startup_restore_refuse_while_lease_active():
    source = NODE.read_text(encoding="utf-8")
    standby = source[
        source.index("def _return_to_configured_standby_callback") :
        source.index("def _return_to_configured_initialize_callback")
    ]
    initialize = source[
        source.index("def _return_to_configured_initialize_callback") :
        source.index("def _precontact_lease_callback")
    ]
    restore = source[
        source.index("def _restore_startup_standby_callback") :
        source.index("def _return_to_configured_standby_callback")
    ]
    for block in (standby, initialize, restore):
        assert "self._precontact_lease.active" in block


def test_status_snapshot_exposes_lease_state():
    source = NODE.read_text(encoding="utf-8")
    publish = source[source.index("def _publish_status") : source.index("def destroy_node")]
    assert 'payload["precontact_speed_lease"] = self._precontact_lease.snapshot()' in publish
    assert 'payload["dynamic_exact_pid_boost"]' in publish


def test_dynamic_pid_release_and_shutdown_restore_firmware_defaults():
    source = NODE.read_text(encoding="utf-8")
    callback = source[
        source.index("def _precontact_lease_callback") :
        source.index("def _observe_auto_torque_recovery")
    ]
    shutdown = source[source.index("def destroy_node") : source.index("def main")]
    assert "self.transport.write(build_reset_joint_pid())" in callback
    assert "self.transport.write(build_reset_joint_pid())" in shutdown
    assert "self.controller.disarm(\"shutdown\")" in shutdown


def test_repository_default_keeps_dynamic_pid_correction_disabled():
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert "dynamic_exact_pid_boost_enabled: false" in text


def test_status_heartbeat_keeps_lease_ownership_fresh_for_executor_gate():
    source = NODE.read_text(encoding="utf-8")
    timer = source[source.index("def _timer_callback") : source.index("def _publish_status")]
    assert "now - self._last_status_publish_monotonic >= 0.2" in timer
    assert "self._publish_status(after)" in timer
