import ast
import asyncio
from pathlib import Path
import json
import math
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / "roarm_post_precontact_rehearsal_executor" / "roarm_post_precontact_rehearsal_executor" / "node.py"
INTERFACES = Path(__file__).resolve().parents[3] / "ros2_perception/ros2_ws/src/elevator_interfaces"
CONFIG = Path(__file__).resolve().parents[3] / "config/press/releases"


def test_node_defaults_closed_and_creates_command_resources_only_in_action_path():
    text = NODE.read_text(encoding="utf-8")
    assert 'declare_parameter("runtime_enabled", False)' in text
    assert 'declare_parameter("build_enabled", False)' in text
    assert 'declare_parameter("contact_authorized", False)' in text
    assert 'declare_parameter("press_authorized", False)' in text
    assert 'declare_parameter("force_control", False)' in text
    init = text[text.index("    def __init__"):text.index("    @staticmethod\n    def _load_stage_b_policy")]
    assert 'create_publisher(JointState, self.JOINT_TARGET_TOPIC' not in init
    assert 'create_client(PrecontactSpeedLease' not in init
    prepare = text[text.index("    def _prepare("):text.index("    def _goal(")]
    for forbidden in ("create_client", "create_publisher(JointState", "_call_arm", "_call_lease"):
        assert forbidden not in prepare
    resources = text[text.index("    async def _create_resources"):text.index("    def _publish_once")]
    assert resources.count("create_publisher(JointState, self.JOINT_TARGET_TOPIC, 1)") == 1
    assert "LEASE_PURPOSE" in resources


def test_exact_safe_driver_subscription_match_precedes_arm_and_first_publish():
    text = NODE.read_text(encoding="utf-8")
    resources = text[text.index("    async def _create_resources"):text.index("    def _publish_once")]
    publish_once = text[text.index("    def _publish_once"):text.index("    def _publish_machine_feedback")]
    execute = text[text.index("    async def _execute("):text.index("    def _publish_status")]
    publish = text[text.index("    async def _publish_current_target"):text.index("    async def _reconcile_lease")]

    assert resources.count("self._target_pub.get_subscription_count() >= 1") == 1
    assert resources.index("create_publisher(JointState") < resources.index("get_subscription_count() >= 1")
    assert resources.index("get_subscription_count() >= 1") < resources.index("_safe_driver_joint_target_subscribed()")
    assert resources.index("_safe_driver_joint_target_subscribed()") < resources.index("machine.publisher_created(sample)")
    assert "value.joint_target_publishers == 1" in resources
    assert 'if exc.code == "MOVE_FEEDBACK_TIMEOUT"' in resources
    assert '"MOVE_RESOURCE_UNAVAILABLE"' in resources
    assert publish_once.index("get_subscription_count() < 1") < publish_once.index("_safe_driver_joint_target_subscribed()")
    assert publish_once.index("_safe_driver_joint_target_subscribed()") < publish_once.index("self._target_pub.publish(message)")
    assert execute.index("await self._create_resources") < execute.index("await self._publish_current_target")
    for token in (
        "boundary = self._sample()",
        "boundary.validate_owner(",
        "require_disarmed=True",
        "machine.q0_drift_limit_rad",
        "phase, target = machine.commands[machine.command_index]",
        "prior = boundary.accepted_command_sequence",
        "publish_boundary_positions = boundary.positions_rad",
    ):
        assert token in publish
    pre_arm_sample = publish.index("boundary = self._sample()")
    owner_gate = publish.index("boundary.validate_owner(")
    drift_gate = publish.index("if start_error > machine.q0_drift_limit_rad")
    frozen_target = publish.index("phase, target = machine.commands[machine.command_index]")
    frozen_sequence = publish.index("prior = boundary.accepted_command_sequence")
    frozen_positions = publish.index("publish_boundary_positions = boundary.positions_rad")
    arm_call = publish.index("await self._call_arm(True")
    machine_armed = publish.index("machine.armed(ok, reason)")
    arm_failure_gate = publish.index("if not ok:")
    unique_publish = publish.index("self._publish_once(target)")
    assert (
        pre_arm_sample < owner_gate < drift_gate < frozen_target
        < frozen_sequence < frozen_positions < arm_call < machine_armed
        < arm_failure_gate < unique_publish
    )
    assert "await " not in publish[machine_armed:unique_publish]
    assert 'value.state == "ARMED"' not in publish[:unique_publish]
    assert "machine.validate_publish_boundary(boundary, first_command=False)" in publish
    assert "publish_boundary_positions=publish_boundary_positions" in publish
    ack_wait = publish.index("accepted = await self._wait_sample(")
    assert unique_publish < ack_wait
    for token in (
        'value.state == "COMMAND_ACTIVE"',
        "value.armed",
        "value.accepted_command_sequence != prior",
        "min_status_at=published_at",
    ):
        assert token in publish[ack_wait:]


def test_frozen_waypoint_loop_zero_dwell_boundary_and_teardown_order_are_source_locked():
    text = NODE.read_text(encoding="utf-8")
    core = (Path(__file__).resolve().parents[3] / "tools/press/stage_b_rehearsal_core.py").read_text(encoding="utf-8")
    execute = text[text.index("    async def _execute("):text.index("    def _publish_status")]
    assert 'while self._machine.state in {"ARM_APPROACH", "PUBLISH_WAYPOINT", "WAIT_WAYPOINT_CONVERGENCE"}' in execute
    assert execute.count("_publish_current_target(") == 1
    assert 'if self._machine.state == "WAIT_WAYPOINT_CONVERGENCE":\n                    await self._sleep(0.02)' in execute
    publish = text[text.index("    async def _publish_current_target"):text.index("    async def _reconcile_lease")]
    assert publish.count("self._publish_once(target)") == 1
    assert "publish_boundary_positions=publish_boundary_positions" in publish
    assert "_sleep(" not in publish
    boundary_gate = core[core.index("    def validate_publish_boundary"):core.index("    def mark_published")]
    assert "sample.dynamic_exact_advance_ready" in boundary_gate
    teardown = text[text.index("    async def _teardown("):text.index("    def _fill_result")]
    assert teardown.index("_call_arm(False)") < teardown.index("destroy_publisher")
    assert teardown.index("destroy_publisher") < teardown.index("_reconcile_lease")
    assert "terminal-proof" in teardown


def test_stage_b_ack_uses_lightweight_feedback_and_full_status_only_for_lease():
    node = NODE.read_text(encoding="utf-8")
    feedback = node[node.index("    def _strict_feedback"):node.index("    @staticmethod\n    def _strict_status")]
    status = node[node.index("    def _strict_status"):node.index("    def _sample")]
    sample = node[node.index("    def _sample"):node.index("    def _prepare_graph_gate")]
    wait = node[node.index("    async def _wait_sample"):node.index("    async def _create_resources")]
    publish = node[node.index("    async def _publish_current_target"):node.index("    async def _reconcile_lease")]
    core = (Path(__file__).resolve().parents[3] / "tools/press/stage_b_rehearsal_core.py").read_text(encoding="utf-8")
    accept = core[core.index("    def accept_command"):core.index("    def observe")]

    for token in (
        "positions_rad", "torque_enabled", "voltage", "feedback_stamp_ns",
        '"state"', '"armed"', '"dry_run"', '"fault"',
        '"dynamic_exact_accepted_target_positions"',
        '"dynamic_exact_accepted_command_sequence"',
        '"dynamic_exact_advance_ready"',
    ):
        assert token in feedback
    assert 'value.get("precontact_speed_lease")' in status
    for token in (
        'value.get("state")', 'value.get("armed")', 'value.get("dry_run")',
        'value.get("fault")', 'value.get("dynamic_exact_accepted_target_positions")',
        'value.get("dynamic_exact_accepted_command_sequence")',
        'value.get("dynamic_exact_advance_ready")',
    ):
        assert token not in status
    assert "_strict_feedback(feedback)" in sample
    assert "_strict_status(status)" in sample
    assert sample.count(
        "float(feedback_at if feedback_at is not None else -math.inf)"
    ) == 2
    assert "math.inf if status_at is None else now - status_at" in sample
    assert "min_status_at" in wait
    assert "sample.status_received_monotonic > min_status_at" in wait
    assert "min_status_at=published_at" in publish
    assert "sample.status_received_monotonic <= self._published_at" in accept
    assert "sample.feedback_received_monotonic <= self._published_at" not in accept


def test_lightweight_feedback_parser_is_strict_and_binds_dynamic_ack_fields():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    nodes = []
    for name in ("_strict_feedback", "_strict_status"):
        node = next(
            item for item in ast.walk(tree)
            if isinstance(item, ast.FunctionDef) and item.name == name
        )
        node.decorator_list = []
        nodes.append(node)

    class Blocked(RuntimeError):
        pass

    namespace = {"math": math, "StageBBlocked": Blocked}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(NODE), "exec"),
        namespace,
    )
    feedback = {
        "positions_rad": [0.0] * 6,
        "torque_enabled": [True] * 6,
        "voltage": 12.1,
        "feedback_stamp_ns": 1,
        "state": "ARMED",
        "armed": True,
        "dry_run": False,
        "fault": "",
        "dynamic_exact_accepted_target_positions": [0.1] * 6,
        "dynamic_exact_accepted_command_sequence": 7,
        "dynamic_exact_advance_ready": True,
    }
    parsed = namespace["_strict_feedback"](feedback)
    assert parsed[4:] == (
        "ARMED", True, False, "", [0.1] * 6, 7, True,
    )
    for field in (
        "state", "armed", "dry_run", "fault",
        "dynamic_exact_accepted_target_positions",
        "dynamic_exact_accepted_command_sequence",
        "dynamic_exact_advance_ready",
    ):
        malformed = dict(feedback)
        malformed.pop(field)
        with pytest.raises(Blocked, match="MOVE_FEEDBACK_MALFORMED"):
            namespace["_strict_feedback"](malformed)

    lease = {
        "precontact_speed_lease": {
            "active": True,
            "purpose": "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE",
            "dynamic_exact_no_reissue": True,
        }
    }
    assert namespace["_strict_status"](lease) == (
        True, "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE", True,
    )
    with pytest.raises(Blocked, match="MOVE_LEASE_RECONCILIATION_UNKNOWN"):
        namespace["_strict_status"]({})


def test_safe_driver_subscription_gate_allows_observers_but_not_substitutes():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    gate_node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef)
        and item.name == "_safe_driver_joint_target_subscribed"
    )

    class Blocked(RuntimeError):
        def __init__(self, code, detail=""):
            self.code = code
            super().__init__(f"{code}:{detail}")

    namespace = {"StageBBlocked": Blocked}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[gate_node], type_ignores=[])),
            str(NODE),
            "exec",
        ),
        namespace,
    )

    def endpoint(name, *, namespace="/", topic_type="sensor_msgs/msg/JointState"):
        return SimpleNamespace(
            node_name=name,
            node_namespace=namespace,
            topic_type=topic_type,
        )

    class Holder:
        JOINT_TARGET_TOPIC = "/roarm_safe/joint_target"

        def __init__(self, endpoints):
            self.endpoints = endpoints

        def get_subscriptions_info_by_topic(self, topic):
            assert topic == self.JOINT_TARGET_TOPIC
            return self.endpoints

    gate = namespace["_safe_driver_joint_target_subscribed"]
    assert gate(Holder([
        endpoint("codex_live_test_session_monitor"),
        endpoint("roarm_safe_driver"),
    ])) is True
    assert gate(Holder([endpoint("codex_live_test_session_monitor")])) is False
    assert gate(Holder([endpoint("observer_named_safe_driver")])) is False
    assert gate(Holder([endpoint("roarm_safe_driver", namespace="/other")])) is False
    assert gate(Holder([
        endpoint("roarm_safe_driver", topic_type="std_msgs/msg/String")
    ])) is False

    class BrokenHolder(Holder):
        def get_subscriptions_info_by_topic(self, topic):
            raise RuntimeError("graph unavailable")

    with pytest.raises(Blocked, match="MOVE_RESOURCE_UNAVAILABLE"):
        gate(BrokenHolder([]))


def test_first_publish_uses_prearm_boundary_and_arm_failure_never_publishes():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    publish_node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "_publish_current_target"
    )

    class Blocked(RuntimeError):
        def __init__(self, code, detail=""):
            self.code = code
            self.detail = detail
            super().__init__(f"{code}:{detail}")

    class FakeTime:
        value = 100.0

        @classmethod
        def monotonic(cls):
            cls.value += 1.0
            return cls.value

    namespace = {
        "time": FakeTime,
        "StageBBlocked": Blocked,
        "StageBExactExecution": object,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[publish_node], type_ignores=[])), str(NODE), "exec"),
        namespace,
    )

    class Boundary:
        positions_rad = (0.0,) * 6
        accepted_command_sequence = 17

        def __init__(self, events):
            self.events = events

        def validate_owner(self, config, **kwargs):
            assert kwargs == {
                "expected_publishers": 1,
                "expected_lease": True,
                "require_disarmed": True,
            }
            self.events.append("prearm_boundary")

    class Machine:
        state = "ARM_APPROACH"
        command_index = 0
        publish_log = []
        start_q_rad = (0.0,) * 6
        q0_drift_limit_rad = 0.01
        commands = (("FORWARD", (0.1,) * 6),)
        phase = "FORWARD"
        target = (0.1,) * 6

        def __init__(self, events):
            self.events = events

        def armed(self, ok, reason):
            self.events.append(("armed", ok, reason))

        def mark_published(self, phase, target, **kwargs):
            assert phase == "FORWARD" and target == (0.1,) * 6
            assert kwargs["prior_sequence"] == 17
            assert kwargs["publish_boundary_positions"] == (0.0,) * 6
            self.events.append("marked")

        def accept_command(self, sample, **kwargs):
            self.events.append("accepted")

    class Holder:
        _execution_config = SimpleNamespace(command_ack_timeout_s=0.75)

        def __init__(self, arm_ok):
            self.events = []
            self.boundary = Boundary(self.events)
            self.arm_ok = arm_ok

        def _sample(self):
            return self.boundary

        async def _call_arm(self, enabled, goal_handle):
            assert enabled is True
            self.events.append("arm_call")
            return self.arm_ok, "" if self.arm_ok else "denied"

        def _publish_once(self, target):
            assert target == (0.1,) * 6
            self.events.append("published")

        def _publish_machine_feedback(self, goal_handle, machine):
            self.events.append("feedback")

        async def _wait_sample(self, predicate, timeout, **kwargs):
            accepted = SimpleNamespace(
                state="COMMAND_ACTIVE",
                armed=True,
                accepted_command_sequence=18,
            )
            assert predicate(accepted)
            assert kwargs["min_status_at"] > 100.0
            self.events.append("ack_wait")
            return accepted

    failed = Holder(False)
    with pytest.raises(Blocked, match="MOVE_ARM_REJECTED:denied"):
        asyncio.run(namespace["_publish_current_target"](
            failed, None, Machine(failed.events), first_command=True,
        ))
    assert failed.events == [
        "prearm_boundary", "arm_call", ("armed", False, "denied"),
    ]

    succeeded = Holder(True)
    asyncio.run(namespace["_publish_current_target"](
        succeeded, None, Machine(succeeded.events), first_command=True,
    ))
    assert succeeded.events == [
        "prearm_boundary", "arm_call", ("armed", True, ""), "published",
        "marked", "feedback", "ack_wait", "accepted",
    ]


def test_lease_wait_requires_core_safe_idle_and_disarmed_sample_before_machine():
    tree = ast.parse(NODE.read_text(encoding="utf-8"))
    create_resources = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_create_resources"
    )
    predicates = [
        node
        for node in ast.walk(create_resources)
        if isinstance(node, ast.Lambda)
        and any(
            isinstance(attr, ast.Attribute) and attr.attr == "lease_active"
            for attr in ast.walk(node)
        )
    ]
    assert len(predicates) == 1
    predicate = predicates[0]
    names = {
        node.id for node in ast.walk(predicate) if isinstance(node, ast.Name)
    }
    attrs = {
        node.attr for node in ast.walk(predicate) if isinstance(node, ast.Attribute)
    }
    assert "SAFE_IDLE_EXECUTION_STATES" in names
    assert {"state", "armed", "lease_active"}.issubset(attrs)
    assert any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Attribute)
        and node.operand.attr == "armed"
        for node in ast.walk(predicate)
    )
    wait_call = next(
        node for node in ast.walk(create_resources)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_wait_sample"
    )
    lease_call = next(
        node for node in ast.walk(create_resources)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lease_acquired"
    )
    assert wait_call.lineno < lease_call.lineno


def test_typed_interfaces_goal_has_identity_only():
    action = (INTERFACES / "action/StageBRehearsal.action").read_text(encoding="utf-8")
    goal = "\n".join(line for line in action.split("---", 1)[0].splitlines() if not line.lstrip().startswith("#"))
    assert "plan_hash" in goal
    assert "stage_a_receipt_sha256" in goal
    assert "authorization_id" in goal
    assert "confirmation_token" in goal
    assert "q" not in goal.lower()
    assert "float64" not in goal


def test_prepare_interface_is_receipt_only():
    service = (INTERFACES / "srv/PrepareStageBRehearsal.srv").read_text(encoding="utf-8")
    request = service.split("---", 1)[0]
    assert "canonical_stage_a_receipt_json" in request
    assert "stage_a_receipt_sha256" in request
    assert "q" not in request.lower().replace("request", "")


def test_prepare_rechecks_current_resources_panel_receipt_qc_owner_and_unique_graph():
    text = NODE.read_text(encoding="utf-8")
    gate = text[text.index("    def _prepare_graph_gate"):text.index("    @staticmethod\n    def _clear_prepare_response")]
    for token in (
            "CURRENT_RESOURCE_CLEANUP_UNPROVEN", "QC_CURRENT_PANEL_BINDING_UNPROVEN",
            "OWNER_GRAPH_NOT_UNIQUE", 'expected_publishers=0', 'expected_lease=False',
            'expected_stage_b_phase: str = "PREPARE_IN_FLIGHT"',
            'qc.get("stage_b_panel_press_receipt_sha256")',
            'qc.get("stage_b_panel_geometry_sha256")',
    ):
        assert token in gate


def test_action_gate_binds_exact_qc_confirm_and_stop_state_cannot_create_resources():
    text = NODE.read_text(encoding="utf-8")
    gate = text[text.index("    def _action_graph_gate"):text.index("    @staticmethod\n    def _clear_prepare_response")]
    for token in (
        '"ACTION_IN_FLIGHT"', '"stage_b_controls_locked"',
        '"stage_b_stop_requested"', '"stage_b_plan_hash"',
        '"stage_b_panel_press_receipt_sha256"', '"stage_b_authorization_id"',
        '"stage_b_instance_nonce"', '"stage_b_status_sequence"',
    ):
        assert token in gate
    resources = text[text.index("    async def _create_resources"):text.index("    def _publish_once")]
    assert resources.index("_action_graph_gate") < resources.index("create_client")


def test_every_frozen_command_publishes_typed_feedback_without_dwell():
    text = NODE.read_text(encoding="utf-8")
    feedback = text[text.index("    def _publish_machine_feedback"):text.index("    async def _publish_current_target")]
    for token in (
        "StageBRehearsal.Feedback()", "machine.phase", "machine.command_index + 1",
        "len(machine.commands)", "goal_handle.publish_feedback(feedback)",
    ):
        assert token in feedback
    publish = text[text.index("    async def _publish_current_target"):text.index("    async def _reconcile_lease")]
    assert publish.count("_publish_machine_feedback(") == 1
    assert "_sleep(" not in publish


def test_terminal_cleanup_reopens_prepare_only_for_a_fresh_receipt_and_plan():
    text = NODE.read_text(encoding="utf-8")
    prepare = text[text.index("    def _prepare("):text.index("    def _goal(")]
    execute = text[text.index("    async def _execute("):text.index("    def _publish_status")]

    assert 'self._state not in {"IDLE", "FAILED_DISARMED"}' in prepare
    assert "receipt.receipt_sha256 in self._seen_receipt_hashes" in prepare
    assert "prepare_stage_b_plan(" in prepare
    result_filled = execute.index("self._fill_result(result")
    cleanup_tokens = (
        "self._plan = None",
        "self._receipt = None",
        'self._authorization_id = ""',
        "self._prepared_qc_binding = {}",
        "self._machine = None",
        "self._goal_reserved = False",
        'self._state = "IDLE"',
    )
    cleanup_positions = [execute.index(token) for token in cleanup_tokens]
    assert result_filled < min(cleanup_positions)
    assert cleanup_positions == sorted(cleanup_positions)
    assert max(cleanup_positions) < execute.index("self._publish_status()")
    assert "self._seen_receipt_hashes.clear" not in execute
    assert "self._consumed_plan_hashes.clear" not in execute


def test_executor_runtime_contract_is_quantized_four_first_subsequence_and_exact_mirror():
    core = (Path(__file__).resolve().parents[3] / "tools/press/stage_b_rehearsal_core.py").read_text(encoding="utf-8")
    assert "_quantize_runtime_waypoint" in core
    assert "for count in range(maximum_count, RUNTIME_MIN_FORWARD_WAYPOINTS - 1, -1)" in core
    assert "combinations(range(len(dense) - 1), count - 1)" in core
    assert "runtime_source_waypoint_indices" in core
    assert "minimum_command_delta_rad=encoder_step" in core
    assert "back != expected_back" in core
    assert 'value.get("command_count") != len(forward) + len(back)' in core
    assert "offline_forward_validation_waypoints_rad" in core
    assert "publish_boundary_positions" in core
    assert "round(value / step) * step" in core


def test_default_release_is_closed_and_review_candidate_opens_positional_contact_only():
    default = json.loads((CONFIG / "stage_b_contact_approach_out_back_release_v1.json").read_text())
    candidate = json.loads((CONFIG / "stage_b_contact_approach_out_back_candidate_20260822_v1.json").read_text())
    assert default["runtime_enabled"] is False and default["build_enabled"] is False
    assert candidate["runtime_enabled"] is True and candidate["build_enabled"] is True
    assert default["contact_authorized"] is False and default["press_authorized"] is False
    assert candidate["contact_authorized"] is True and candidate["press_authorized"] is True
    assert default["approach_distance_m"] == 0.07
    assert candidate["approach_distance_mode"] == "MEASURED_CURRENT_PANEL_GAP"
    assert candidate["maximum_approach_distance_m"] == 0.1
    for value in (default, candidate):
        assert value["approach_geometry_tolerance_m"] == 0.000001
        assert value["no_contact"] is False and value["press"] is False
        assert value["force_control"] is False and value["zero_dwell"] is True
        assert value["button_activation_claimed"] is False
        assert value["automatic_retry"] is False
    assert default["lease_purpose"] == "T102_POST_PRECONTACT_XY_EXACT_PLAN_NO_REISSUE"
    assert candidate["lease_purpose"] == "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE"
