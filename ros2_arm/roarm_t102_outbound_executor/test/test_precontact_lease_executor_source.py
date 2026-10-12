"""Source-only exact-plan resource ordering contracts; no ROS import."""

import ast
import asyncio
import math
from pathlib import Path
import sys
import textwrap
import threading
import types

import pytest


NODE = Path(__file__).resolve().parents[1] / "roarm_t102_outbound_executor/node.py"
PKG = NODE.parents[1]
REPO = NODE.parents[3]
sys.path.insert(0, str(PKG))

from roarm_t102_outbound_executor.execution import (
    DYNAMIC_LEASE_PURPOSE, MoveBlocked, runtime_safety_failure_reason,
)


def _func(name: str) -> str:
    source = NODE.read_text()
    tree = ast.parse(source)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    return ast.get_source_segment(source, fn) or ""


def _compiled_func(name: str, namespace: dict):
    exec(textwrap.dedent(_func(name)), namespace)
    return namespace[name]


def test_lease_then_arm_publisher_episode_and_teardown_order():
    resources = _func("_create_execution_resources")
    lease_call = resources.index("await self._call_lease(")
    lease_proof = resources.index("machine.lease_acquired(self._safety_sample())")
    assert lease_call < resources.index(
        "create_client(\n            SetBool, self.ARM_SERVICE,"
    )
    assert lease_call < lease_proof
    assert lease_proof < resources.index(
        'create_publisher(JointState, "/roarm_safe/joint_target"'
    )
    assert lease_call < resources.index(
        'create_publisher(JointState, "/roarm_safe/joint_target"'
    )
    assert resources.index(
        'create_publisher(JointState, "/roarm_safe/joint_target"'
    ) < resources.index("machine.publisher_created(self._safety_sample())")
    teardown = _func("_teardown_execution_resources")
    assert teardown.index("_call_arm(False)") < teardown.index("destroy_publisher")
    assert teardown.index("destroy_publisher") < teardown.index("_reconcile_uncertain_lease")


def test_success_teardown_preserves_fresh_converged_terminal_for_next_prepare():
    teardown = _func("_teardown_execution_resources")
    assert 'sample.state == "CONVERGED_DISARMED"' in teardown
    assert "not sample.armed" in teardown
    assert "not sample.dry_run" in teardown
    assert "not sample.fault" in teardown
    assert "sample.feedback_age_s" in teardown
    assert "if not already_converged_disarmed:" in teardown
    # Sampling uncertainty must retain the explicit fail-closed disarm.
    assert teardown.count("await self._call_arm(False)") == 2
    reconcile = _func("_reconcile_uncertain_lease")
    assert "_call_lease(False)" in reconcile


def test_failure_cancel_timeout_have_one_terminal_teardown_and_no_reissue():
    execute = _func("_execute")
    assert execute.count("_teardown_execution_resources()") == 1
    assert execute.count("_publish_exact_waypoint_once(target)") == 1
    assert "goal_handle.is_cancel_requested" in execute
    assert "retry" not in execute.lower()
    pure = (NODE.parent / "execution.py").read_text()
    assert "no retry permitted" in pure
    assert "MOVE_REISSUE_BLOCKED" in pure


def test_constructor_has_no_motion_client_publisher_or_lease_client():
    init = _func("__init__")
    assert 'create_publisher(JointState, "/roarm_safe/joint_target"' not in init
    assert 'create_client(SetBool, "/roarm_safe/set_armed")' not in init
    assert 'create_client(\n            PrecontactSpeedLease' not in init


def test_prepare_and_validation_have_zero_motion_resource_calls():
    for name in ("_prepare", "_execute_validation"):
        source = _func(name)
        for forbidden in (
            "_create_execution_resources", "_call_arm", "_call_lease",
            "create_client", "create_publisher", "_publish_exact_waypoint_once",
        ):
            assert forbidden not in source


def test_discard_prepared_is_exact_zero_motion_and_only_returns_idle_after_all_gates():
    discard = _func("_discard_prepared")
    for required in (
        "State.AWAIT_MOVE_CONFIRM", "not self._plan.consumed",
        "not self._goal_slot_reserved", "self._active_machine is None",
        "request.executor_instance_nonce == self._instance_nonce",
        "request.plan_hash == plan_hash", "DISCARD_CONFIRMATION_TOKEN",
        "not any(self._resource_counts.values())",
        'int(status["joint_target_publishers"]) == 0',
        'status["precontact_lease_active"] is False',
        'status["terminal_armed"] is False',
    ):
        assert required in discard
    for forbidden in (
        "_create_execution_resources", "_call_arm", "_call_lease",
        "create_client", "create_publisher", "_publish_exact_waypoint_once",
    ):
        assert forbidden not in discard
    assert discard.index("if safe:") < discard.index("self._state = State.IDLE")


def test_discard_prepared_rejects_identity_or_active_resource_without_state_mutation():
    method = _compiled_func("_discard_prepared", {"State": types.SimpleNamespace(AWAIT_MOVE_CONFIRM="AWAIT_MOVE_CONFIRM", IDLE="IDLE")})

    class Plan:
        consumed = False
        plan_hash = "plan"

    def harness(**changes):
        value = types.SimpleNamespace(
            _lock=threading.RLock(), _state="AWAIT_MOVE_CONFIRM", _plan=Plan(),
            _goal_slot_reserved=False, _active_machine=None,
            _instance_nonce="executor", _authorization_id="auth", _failure="",
            DISCARD_CONFIRMATION_TOKEN="token",
            _resource_counts={
                "publisher": 0, "lease_client": 0, "arm_client": 0,
                "safe_driver_client": 0, "service_call": 0, "action_call": 0,
            },
            _status=lambda: {
                "joint_target_publishers": 0,
                "precontact_lease_active": False,
                "terminal_armed": False,
            },
            _publish_status=lambda: None,
        )
        for key, item in changes.items():
            setattr(value, key, item)
        return value

    request = types.SimpleNamespace(
        executor_instance_nonce="wrong", plan_hash="plan",
        confirmation_token="token",
    )
    response = types.SimpleNamespace()
    blocked = harness()
    method(blocked, request, response)
    assert response.accepted is False
    assert blocked._state == "AWAIT_MOVE_CONFIRM"
    assert blocked._plan is not None and blocked._authorization_id == "auth"

    active = harness(_resource_counts={
        "publisher": 1, "lease_client": 0, "arm_client": 0,
        "safe_driver_client": 0, "service_call": 0, "action_call": 0,
    })
    request.executor_instance_nonce = "executor"
    method(active, request, types.SimpleNamespace())
    assert active._state == "AWAIT_MOVE_CONFIRM" and active._plan is not None

    idle = harness()
    accepted = types.SimpleNamespace()
    method(idle, request, accepted)
    assert accepted.accepted is True and idle._state == "IDLE"
    assert idle._plan is None and idle._authorization_id == ""


def test_discard_service_schema_is_exact_and_registered():
    service = (
        REPO
        / "ros2_perception/ros2_ws/src/elevator_interfaces/srv/DiscardPreparedPrecontact.srv"
    ).read_text()
    request, response = service.split("---", 1)
    assert [
        line.strip() for line in request.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ] == [
        "string executor_instance_nonce", "string plan_hash",
        "string confirmation_token",
    ]
    for field in (
        "bool accepted", "string state", "string executor_instance_nonce",
        "string plan_hash", "uint32 publisher_count", "bool lease_active",
        "uint32 arm_client_count", "string reason",
    ):
        assert field in response
    cmake = (
        REPO / "ros2_perception/ros2_ws/src/elevator_interfaces/CMakeLists.txt"
    ).read_text()
    assert '"srv/DiscardPreparedPrecontact.srv"' in cmake


def test_status_schema_and_default_closed_triple_gate():
    source = NODE.read_text()
    assert "MultiThreadedExecutor(num_threads=4)" in source
    assert '"t102_dynamic_executor_status_v1"' in source
    assert 'declare_parameter("move_runtime_enabled", False)' in source
    assert 'model.policy.get("move_execution_approved") is not True' in source
    assert "LIFT_ROBOT_DYNAMIC_MOVE_BUILD_TOKEN" in source
    assert '"move_execution_ready": move_ready' in source
    assert '"dynamic_exact_no_reissue"' in source


def test_move_readiness_checks_required_services_without_command_clients():
    gate = _func("_move_gate")
    assert "get_service_names_and_types" in gate
    assert "LEASE_SERVICE_TYPE" in gate
    assert "ARM_SERVICE_TYPE" in gate
    assert "MOVE_RESOURCE_GRAPH_UNAVAILABLE" in gate
    assert "MOVE_RESOURCE_UNAVAILABLE: speed lease service" in gate
    assert "MOVE_RESOURCE_UNAVAILABLE: arm service" in gate
    assert "create_client" not in gate
    assert "create_publisher" not in gate


def test_goal_slot_is_reserved_in_callback_and_cleared_on_every_execution_exit():
    goal = _func("_goal")
    execute = _func("_execute")
    assert "if self._goal_slot_reserved" in goal
    assert goal.count("self._goal_slot_reserved = True") == 2
    assert execute.count("self._goal_slot_reserved = False") >= 2


def test_fresh_feedback_is_health_authority_and_service_waits_are_bounded():
    sample = _func("_safety_sample")
    assert "_strict_feedback_fields(feedback)" in sample
    assert "stamp_age" in sample
    assert "_current_positions" not in sample
    assert 'status.get("state")' not in sample
    assert 'status.get("armed")' not in sample
    wait = _func("_await_service_response")
    assert "deadline" in wait and "future.cancel()" in wait
    teardown = _func("_teardown_execution_resources")
    assert "publisher-destroy" in teardown
    assert "lease-reconciliation" in teardown


def test_ros_sleep_uses_one_persistent_scheduler_timer():
    init = _func("__init__")
    sleep = _func("_ros_sleep")
    wake = _func("_wake_sleep_waiters")
    assert "self._sleep_timer = self.create_timer(" in init
    assert "self._wake_sleep_waiters" in init
    assert "create_timer" not in sleep
    assert "destroy_timer" not in sleep
    assert "destroy_timer" not in wake


def _sleep_scheduler_harness(*, future_type=None):
    class FakeTime:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

    if future_type is None:
        class FakeFuture:
            def __init__(self, *, executor):
                del executor
                self.inner = asyncio.get_running_loop().create_future()

            def __await__(self):
                return self.inner.__await__()

            def done(self):
                return self.inner.done()

            def set_result(self, value):
                self.inner.set_result(value)

        future_type = FakeFuture

    wake = _compiled_func("_wake_sleep_waiters", {"time": FakeTime})
    sleep = _compiled_func(
        "_ros_sleep", {"time": FakeTime, "Future": future_type},
    )

    class Harness:
        def __init__(self):
            self.executor = object()
            self._lock = threading.RLock()
            self._sleep_waiters = []

    harness = Harness()
    harness.wake = types.MethodType(wake, harness)
    harness.sleep = types.MethodType(sleep, harness)
    return harness, FakeTime


def test_ros_sleep_resolves_multiple_concurrent_deadlines():
    async def scenario():
        harness, clock = _sleep_scheduler_harness()
        tasks = [
            asyncio.create_task(harness.sleep(delay))
            for delay in (0.01, 0.02, 0.03)
        ]
        await asyncio.sleep(0)
        assert len(harness._sleep_waiters) == 3
        for index, now in enumerate((0.011, 0.021, 0.031), start=1):
            clock.now = now
            harness.wake()
            await asyncio.sleep(0)
            assert sum(task.done() for task in tasks) == index
        await asyncio.gather(*tasks)
        assert harness._sleep_waiters == []

    asyncio.run(scenario())


def test_ros_sleep_cancel_and_exception_remove_waiter():
    async def cancelled():
        harness, _clock = _sleep_scheduler_harness()
        task = asyncio.create_task(harness.sleep(10.0))
        await asyncio.sleep(0)
        assert len(harness._sleep_waiters) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert harness._sleep_waiters == []

    asyncio.run(cancelled())

    class ExplodingFuture:
        def __init__(self, *, executor):
            del executor

        def __await__(self):
            async def explode():
                raise RuntimeError("test await failure")
            return explode().__await__()

    async def failed():
        harness, _clock = _sleep_scheduler_harness(
            future_type=ExplodingFuture,
        )
        with pytest.raises(RuntimeError, match="test await failure"):
            await harness.sleep(1.0)
        assert harness._sleep_waiters == []

    asyncio.run(failed())


def test_ros_sleep_repeated_waits_do_not_accumulate_waiters():
    async def scenario():
        harness, clock = _sleep_scheduler_harness()
        for _ in range(2000):
            task = asyncio.create_task(harness.sleep(0.0))
            await asyncio.sleep(0)
            harness.wake()
            await task
            assert harness._sleep_waiters == []
            clock.now += 0.001

    asyncio.run(scenario())


def _valid_feedback() -> dict:
    return {
        "positions_rad": [0.0] * 6,
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "dry_run": False,
        "fault": "",
        "torque_enabled": [True] * 6,
        "voltage": 12.0,
        "feedback_stamp_ns": 123,
        "dynamic_exact_accepted_command_sequence": 0,
        "dynamic_exact_accepted_target_positions": [],
        "dynamic_exact_advance_ready": False,
    }


@pytest.mark.parametrize(
    "field",
    [
        "positions_rad", "state", "armed", "dry_run", "fault",
        "torque_enabled", "voltage", "feedback_stamp_ns",
        "dynamic_exact_accepted_command_sequence",
        "dynamic_exact_accepted_target_positions",
        "dynamic_exact_advance_ready",
    ],
)
@pytest.mark.parametrize("mode", ["missing", "none"])
def test_strict_feedback_requires_every_typed_event_field(field, mode):
    strict = _compiled_func(
        "_strict_feedback_fields",
        {"math": math, "MoveBlocked": MoveBlocked},
    )
    value = _valid_feedback()
    if mode == "missing":
        del value[field]
    else:
        value[field] = None
    with pytest.raises(MoveBlocked, match="MOVE_FEEDBACK_MALFORMED"):
        strict(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("positions_rad", [0.0] * 5),
        ("positions_rad", [0.0] * 5 + [float("nan")]),
        ("armed", 0),
        ("dry_run", 0),
        ("fault", None),
        ("state", ""),
        ("torque_enabled", [True] * 5 + [1]),
        ("voltage", float("nan")),
        ("feedback_stamp_ns", 0),
        ("dynamic_exact_accepted_command_sequence", -1),
        ("dynamic_exact_accepted_command_sequence", True),
        ("dynamic_exact_accepted_target_positions", [0.0] * 5),
        ("dynamic_exact_accepted_target_positions", [0.0] * 5 + [True]),
        ("dynamic_exact_advance_ready", 1),
    ],
)
def test_strict_feedback_rejects_malformed_values(field, value):
    strict = _compiled_func(
        "_strict_feedback_fields",
        {"math": math, "MoveBlocked": MoveBlocked},
    )
    feedback = _valid_feedback()
    feedback[field] = value
    with pytest.raises(MoveBlocked, match="MOVE_FEEDBACK_MALFORMED"):
        strict(feedback)


def test_uncertain_enable_exact_active_status_is_released_then_proven_inactive():
    class FakeTime:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

    class FakeAsyncio:
        @staticmethod
        async def sleep(delay):
            FakeTime.now += float(delay)

    reconcile = _compiled_func(
        "_reconcile_uncertain_lease",
        {"time": FakeTime, "asyncio": FakeAsyncio, "MoveBlocked": MoveBlocked},
    )

    class Harness:
        def __init__(self):
            self._lease_client = object()
            self._lease_active_local = True
            self._lease_enable_uncertain = True
            self._lease_enable_response_definitive = True
            self.decisions = ["RELEASE_EXACT_ACTIVE", "PROVEN_INACTIVE"]
            self.release_calls = 0

        def _lease_reconciliation_snapshot(self):
            return self.decisions.pop(0), {}

        async def _call_lease(self, enable):
            assert enable is False
            self.release_calls += 1
            return True, "released"

        async def _ros_sleep(self, delay):
            FakeTime.now += float(delay)

    harness = Harness()
    asyncio.run(types.MethodType(reconcile, harness)())
    assert harness.release_calls == 1
    assert harness._lease_active_local is False
    assert harness._lease_enable_uncertain is False


def test_enable_ack_timeout_then_exact_active_status_still_releases_once():
    class LeaseType:
        class Request:
            pass

    class Client:
        def call_async(self, request):
            return object()

    call_lease = _compiled_func(
        "_call_lease",
        {
            "PrecontactSpeedLease": LeaseType,
            "DYNAMIC_LEASE_PURPOSE": DYNAMIC_LEASE_PURPOSE,
        },
    )

    class Harness:
        def __init__(self):
            self._lease_client = Client()
            self._lease_id = "exact-id"
            self._lease_enable_uncertain = False
            self._lease_enable_response_definitive = False
            self._lease_active_local = False
            self._resource_counts = {"service_call": 0}
            # The first fresh inactive status is not proof: the queued server
            # call then applies late and must still be observed/released.
            self.decisions = [
                "PROVEN_INACTIVE", "RELEASE_EXACT_ACTIVE", "PROVEN_INACTIVE",
            ]
            self.release_calls = 0

        async def _await_service_response(self, _future, **_kwargs):
            raise MoveBlocked("MOVE_SERVICE_TIMEOUT")

    harness = Harness()
    with pytest.raises(MoveBlocked, match="MOVE_SERVICE_TIMEOUT"):
        asyncio.run(types.MethodType(call_lease, harness)(True))
    assert harness._lease_enable_uncertain is True
    assert harness._lease_enable_response_definitive is False
    assert harness._resource_counts["service_call"] == 0

    class FakeTime:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

    class FakeAsyncio:
        @staticmethod
        async def sleep(delay):
            FakeTime.now += float(delay)

    def snapshot(self):
        decision = self.decisions.pop(0) if self.decisions else "PROVEN_INACTIVE"
        return decision, {}

    async def release(self, enable):
        assert enable is False
        self.release_calls += 1
        return True, "released"

    async def ros_sleep(self, delay):
        FakeTime.now += float(delay)

    harness._lease_reconciliation_snapshot = types.MethodType(snapshot, harness)
    harness._call_lease = types.MethodType(release, harness)
    harness._ros_sleep = types.MethodType(ros_sleep, harness)
    reconcile = _compiled_func(
        "_reconcile_uncertain_lease",
        {"time": FakeTime, "asyncio": FakeAsyncio, "MoveBlocked": MoveBlocked},
    )
    harness._lease_client = object()
    with pytest.raises(
        MoveBlocked,
        match="enable acknowledgement indeterminate.*operator intervention required",
    ):
        asyncio.run(types.MethodType(reconcile, harness)())
    assert harness.release_calls == 1
    assert harness._lease_enable_uncertain is True


def test_enable_uncertainty_is_latched_before_async_dispatch_and_teardown_reconciles():
    lease = _func("_call_lease")
    assert lease.index("self._lease_enable_uncertain = True") < lease.index(
        "self._lease_client.call_async(request)"
    )
    assert lease.index("self._lease_client.call_async(request)") < lease.index(
        "self._lease_enable_response_definitive = True"
    )
    teardown = _func("_teardown_execution_resources")
    assert "self._lease_active_local or self._lease_enable_uncertain" in teardown
    assert "await self._reconcile_uncertain_lease()" in teardown
    terminal = _func("_terminal_resource_evidence")
    assert "_strict_feedback_fields(feedback)" in terminal
    assert "stamp_age" in terminal and "lease_age" in terminal


def test_indeterminate_enable_fresh_inactive_alone_never_proves_clean():
    class FakeTime:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

    class FakeAsyncio:
        @staticmethod
        async def sleep(delay):
            FakeTime.now += float(delay)

    reconcile = _compiled_func(
        "_reconcile_uncertain_lease",
        {"time": FakeTime, "asyncio": FakeAsyncio, "MoveBlocked": MoveBlocked},
    )

    class Harness:
        _lease_client = object()
        _lease_active_local = False
        _lease_enable_uncertain = True
        _lease_enable_response_definitive = False

        def _lease_reconciliation_snapshot(self):
            return "PROVEN_INACTIVE", {}

        async def _call_lease(self, _enable):
            raise AssertionError("inactive evidence must not dispatch release")

        async def _ros_sleep(self, delay):
            FakeTime.now += float(delay)

    harness = Harness()
    with pytest.raises(MoveBlocked, match="queued late apply"):
        asyncio.run(types.MethodType(reconcile, harness)())
    assert harness._lease_enable_uncertain is True


def test_waypoint_wait_requires_one_fresh_atomic_feedback_ack():
    execute = _func("_execute")
    assert execute.index("published_at = time.monotonic()") < execute.index(
        "_publish_exact_waypoint_once(target)"
    )
    assert "prior_sequence = boundary.dynamic_exact_accepted_command_sequence" in execute
    active_start = execute.index("active_sample = await self._wait_sample(")
    active_end = execute.index("validate_command_ack(", active_start)
    active_wait = execute[active_start:active_end]
    assert 'value.state == "COMMAND_ACTIVE"' in active_wait
    assert "value.armed" in active_wait
    assert "value.dynamic_exact_accepted_command_sequence" in active_wait
    assert "min_received_monotonic=published_at" in active_wait
    assert "min_status_received_monotonic=published_at" not in active_wait
    assert execute.count("validate_command_ack(") == 1
    assert "pre_publish_positions=boundary.positions_rad" in execute
    observe = execute.index("machine.observe(active_sample", active_end)
    assert active_start < active_end < observe


def _feedback_ack_harness(samples, feedback_markers):
    class FakeTime:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

    wait_sample = _compiled_func(
        "_wait_sample",
        {
            "time": FakeTime,
            "math": math,
            "MoveBlocked": MoveBlocked,
            "SafetySample": object,
            "runtime_safety_failure_reason": runtime_safety_failure_reason,
        },
    )

    class Harness:
        def __init__(self):
            self.index = 0
            self._move_policy = types.SimpleNamespace(
                feedback_max_age_s=0.5,
                min_voltage_v=11.0,
                max_voltage_v=13.0,
            )
            self._safe_feedback_received_monotonic = feedback_markers[0]

        def _safety_sample(self):
            return samples[self.index]

        async def _ros_sleep(self, delay):
            FakeTime.now += float(delay)
            if self.index + 1 < len(samples):
                self.index += 1
                self._safe_feedback_received_monotonic = feedback_markers[self.index]

    harness = Harness()
    harness.wait_sample = types.MethodType(wait_sample, harness)
    return harness, FakeTime


def _ack_sample(*, state="ARMED", armed=True, sequence=0, status_received=0.0):
    return types.SimpleNamespace(
        state=state,
        armed=armed,
        dynamic_exact_accepted_command_sequence=sequence,
        status_received_monotonic=status_received,
        feedback_age_s=0.01,
        feedback_stamp_age_s=0.01,
        lease_status_age_s=0.01,
        fault="",
        dry_run=False,
        torque_enabled=(True,) * 6,
        voltage_v=12.0,
    )


def test_feedback_ack_succeeds_without_post_publish_status_delivery():
    marker = 1.0
    harness, _ = _feedback_ack_harness(
        [
            _ack_sample(
                state="COMMAND_ACTIVE", sequence=1, status_received=0.5,
            ),
        ],
        [1.2],
    )
    active = asyncio.run(harness.wait_sample(
        lambda value: (
            value.state == "COMMAND_ACTIVE"
            and value.armed
            and value.dynamic_exact_accepted_command_sequence != 0
        ),
        0.75,
        min_received_monotonic=marker,
    ))
    assert active.state == "COMMAND_ACTIVE"
    assert active.dynamic_exact_accepted_command_sequence == 1


def test_feedback_ack_waits_for_atomic_sequence_in_later_feedback():
    marker = 1.0
    harness, _ = _feedback_ack_harness(
        [
            _ack_sample(
                state="COMMAND_ACTIVE", sequence=0, status_received=0.9,
            ),
            _ack_sample(
                state="COMMAND_ACTIVE", sequence=1, status_received=1.2,
            ),
        ],
        [1.1, 1.2],
    )
    active = asyncio.run(harness.wait_sample(
        lambda value: (
            value.state == "COMMAND_ACTIVE"
            and value.armed
            and value.dynamic_exact_accepted_command_sequence != 0
        ),
        0.75,
        min_received_monotonic=marker,
    ))
    assert active.dynamic_exact_accepted_command_sequence == 1


def test_feedback_ack_preserves_fresh_torque_fault_before_stale_source_stamp():
    observed = _ack_sample(
        state="FAULT:one or more joint torque switches are off", armed=False,
    )
    observed.fault = "one or more joint torque switches are off"
    observed.torque_enabled = (False,) * 6
    observed.feedback_stamp_age_s = 99.0
    harness, _ = _feedback_ack_harness([observed], [1.2])

    with pytest.raises(
        MoveBlocked,
        match=(
            "MOVE_SAFETY_GATE_FAILED: FAULT:one or more joint torque switches "
            "are off"
        ),
    ):
        asyncio.run(harness.wait_sample(lambda _value: True, 0.75))


@pytest.mark.parametrize("missing", ["sequence", "command_active"])
def test_feedback_ack_missing_atomic_evidence_fails_closed(missing):
    marker = 1.0
    if missing == "sequence":
        samples = [_ack_sample(
            state="COMMAND_ACTIVE", sequence=0, status_received=0.9,
        )]
        feedback_markers = [1.1]
    else:
        samples = [_ack_sample(sequence=1, status_received=1.1)]
        feedback_markers = [1.1]
    predicate = lambda value: (
        value.state == "COMMAND_ACTIVE"
        and value.armed
        and value.dynamic_exact_accepted_command_sequence != 0
    )
    harness, _ = _feedback_ack_harness(samples, feedback_markers)
    with pytest.raises(MoveBlocked, match="MOVE_FEEDBACK_STALE"):
        asyncio.run(harness.wait_sample(
            predicate, 0.75, min_received_monotonic=marker,
        ))


def test_status_terminal_armed_requires_complete_source_stamp_fresh_feedback():
    status = _func("_status")
    assert "_strict_feedback_fields(safe_feedback)" in status
    assert "feedback_stamp_age" in status
    assert "terminal_armed_known" in status
    assert 'safe_status.get("armed")' not in status


def test_ticket_ttl_is_checked_once_before_resources_not_inside_waypoint_loop():
    execute = _func("_execute")
    assert execute.count("verify_frozen_plan(") == 1
    loop = execute[execute.index('while machine.state in {"ARM_WAYPOINT", "PUBLISH_WAYPOINT"}'):]
    assert "expires_monotonic" not in loop
    assert "prepare_dynamic_plan" not in loop
    assert "canonical_hash(plan.plan)" in loop


def test_terminal_action_releases_consumed_plan_for_a_fresh_prepare_cycle():
    execute = _func("_execute")
    terminal = execute[execute.index("self._fill_result("):]
    assert "self._plan = None" in terminal
    assert 'self._authorization_id = ""' in terminal
    assert "self._state = State.IDLE" in terminal
