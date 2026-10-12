from __future__ import annotations

import ast
import asyncio
import math
from pathlib import Path
import sys

import pytest


PKG = Path(__file__).resolve().parents[1]
NODE = PKG / "roarm_t102_outbound_executor" / "node.py"
sys.path.insert(0, str(PKG))

from roarm_t102_outbound_executor.execution import (
    DYNAMIC_LEASE_PURPOSE, ExactPlanExecution, FAULT_CLEARED_DISARMED,
    MoveBlocked, MovePolicy, OPERATOR_DISARMED, SafetySample,
    lease_reconciliation_decision, validate_command_ack,
)


def function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for item in ast.walk(tree):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return ast.get_source_segment(source, item) or ""
    raise AssertionError(name)


def extracted_function(path: Path, name: str, namespace: dict):
    scope = dict(namespace)
    exec(compile(function_source(path, name), str(path), "exec"), scope)
    return scope[name]


def policy():
    return MovePolicy.from_mapping({
        "move_feedback_max_age_s": 0.5,
        "move_waypoint_tolerance_rad": math.radians(0.5),
        "move_waypoint_timeout_s": 5.0,
        "move_min_voltage_v": 11.0,
        "move_max_voltage_v": 13.0,
        "encoder_step_rad": 2.0 * math.pi / 4096.0,
        "per_command_caps_rad": tuple(math.radians(x) for x in (5, 10, 10, 15, 30, 15)),
    })


def plan(count=6):
    q0 = [0.0, -1.0, 2.0, 0.1, -1.5, 0.0]
    return {
        "session": {"q0_rad": q0},
        "waypoints_rad": [[0.01 * i, -1.0 + 0.004 * i, 2.0, 0.1, -1.5, 0.0] for i in range(1, count + 1)],
        "automatic_retry": False, "fixed_v2_fallback": False,
        "no_contact": True, "press": False, "retract": False, "return": False,
    }


def standby_plan(count=4):
    start = [0.0, -1.0, 2.4, 0.1, -1.5, 0.0]
    end = (0.0, -1.0, 2.0, 0.1, -1.5, 0.0)
    waypoints = [
        [0.0, -1.0, 2.4 - 0.4 * i / count, 0.1, -1.5, 0.0]
        for i in range(1, count + 1)
    ]
    waypoints[-1] = list(end)
    return {
        "route_kind": "STANDBY",
        "start_positions_rad": start,
        "standard_standby_rad": list(end),
        "waypoints_rad": waypoints,
        "automatic_retry": False,
        "automatic_compensation": False,
        "overshoot": False,
        "clear_fault": False,
    }


def cold_boot_standby_plan(admission_state=FAULT_CLEARED_DISARMED):
    value = standby_plan(2)
    value["start_positions_rad"] = [0.0, -1.0, 2.04, 0.1, -1.5, 0.0]
    value["waypoints_rad"] = [list(value["standard_standby_rad"])]
    value["bootstrap_identity"] = {
        "mode": "LOCAL_STANDARD_STANDBY_CORRECTION",
        "admission_state": admission_state,
        "per_joint_envelope_rad": math.radians(3.0),
        "start_max_joint_error_rad": 0.04,
    }
    return value


def sample(positions=(0.0, -1.0, 2.0, 0.1, -1.5, 0.0), *, pubs=0,
           lease=False, state="CONVERGED_DISARMED", armed=False, age=0.01,
           fault="", torque=True, dry=False, voltage=12.0, received=100.0,
           sequence=0, accepted_target=(), advance_ready=False,
           status_received=100.0, stamp_age=0.01):
    return SafetySample(tuple(positions), age, state, armed, dry, fault,
        (torque,) * 6, voltage, pubs, lease,
        DYNAMIC_LEASE_PURPOSE if lease else "", bool(lease), 0.01, received,
        stamp_age,
        sequence, tuple(accepted_target), advance_ready, status_received)


def test_safe_driver_command_ack_requires_exact_next_sequence_and_target():
    target = tuple(plan(3)["waypoints_rad"][0])
    pre_publish = tuple(plan(3)["session"]["q0_rad"])
    exact = sample(
        target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
        sequence=1, accepted_target=target,
    )
    validate_command_ack(
        exact, prior_sequence=0, expected_target=target,
        pre_publish_positions=pre_publish,
        encoder_step_rad=policy().encoder_step_rad,
    )
    for bad in (
        sample(sequence=0, accepted_target=target),
        sample(sequence=2, accepted_target=target),
        sample(sequence=1, accepted_target=(9.0,) * 6),
    ):
        with pytest.raises(MoveBlocked, match="MOVE_COMMAND_ACK_INVALID"):
            validate_command_ack(
                bad, prior_sequence=0, expected_target=target,
                pre_publish_positions=pre_publish,
                encoder_step_rad=policy().encoder_step_rad,
            )


def test_ack_accepts_safe_driver_snap_only_for_one_step_held_joint():
    step = policy().encoder_step_rad
    measured = (0.0, -1.0, 2.0, 0.1, -1.5, 0.25)
    expected = (0.03, -1.0, 2.0, 0.1, -1.5, 0.25 + step)
    accepted = list(expected)
    accepted[5] = measured[5]
    validate_command_ack(
        sample(sequence=8, accepted_target=accepted),
        prior_sequence=7,
        expected_target=expected,
        pre_publish_positions=measured,
        encoder_step_rad=step,
    )

    moving_mismatch = list(accepted)
    moving_mismatch[0] += step
    with pytest.raises(MoveBlocked, match="accepted target mismatch"):
        validate_command_ack(
            sample(sequence=8, accepted_target=moving_mismatch),
            prior_sequence=7,
            expected_target=expected,
            pre_publish_positions=measured,
            encoder_step_rad=step,
        )
    with pytest.raises(MoveBlocked, match="sequence"):
        validate_command_ack(
            sample(sequence=7, accepted_target=accepted),
            prior_sequence=7,
            expected_target=expected,
            pre_publish_positions=measured,
            encoder_step_rad=step,
        )


def test_aug27_handoff_waits_for_next_command_cap_without_adding_waypoints():
    # 15:13 hardware telemetry: the old 0.5-degree arrival gate opened while
    # the next elbow command still needed 10.107 degrees (driver cap: 10).
    value = plan(3)
    value["session"]["q0_rad"] = [
        .064427193, -1.191903072, 2.331650798, .464796179,
        -1.515573018, .001533980589793149,
    ]
    value["waypoints_rad"] = [
        [.13345632854605077, -1.038504993398579, 2.1644468917066395,
         .46479617872934925, -1.5155730184310134, .0015339807878856412],
        [.20401944478879028, -.8835729338221293, 1.9957090050392192,
         .4632621979414636, -1.5155730184310134, .0015339807878856412],
        [.27304858024364415, -.7301748550335652, 1.8285050991596843,
         .4632621979414636, -1.5155730184310134, .0015339807878856412],
    ]
    machine = ExactPlanExecution(value, policy(), now=0.)
    start = value["session"]["q0_rad"]
    machine.precheck(sample(start))
    machine.lease_acquired(sample(start, lease=True))
    machine.publisher_created(sample(start, pubs=1, lease=True))
    machine.armed(True)
    machine.mark_published(machine.target, now=1.)
    # Exercise the actual SafeDriver controller too, with an in-memory serial
    # sink. The old handoff is rejected by the real delta gate, not a mock.
    import time
    sys.path.insert(0, str(PKG.parent / "roarm_safe_driver"))
    from roarm_safe_driver.core import Feedback, PrecontactSpeedLease, SafeController, SafetyError
    writes = []
    driver = SafeController(writes.append, dry_run=False,
                            joint_max_delta_rad=policy().per_command_caps_rad)
    clock = time.monotonic()

    def feed(positions, offset):
        driver.accept_feedback(Feedback(tuple(positions), (0,) * 6, (True,) * 6,
                                        12.12, clock + offset, {}))

    feed(start, 0.)
    lease = PrecontactSpeedLease()
    assert lease.enable(driver, purpose=DYNAMIC_LEASE_PURPOSE,
                        lease_id="aug27-offline", foreign_publishers=False)[0]
    driver.arm(clock)
    driver.command_multi(machine.target, clock)
    measured = [.133456329, -1.035437032, 2.172116796, .464796179,
                -1.515573018, .001533980589793149]
    assert max(abs(a-b) for a,b in zip(measured, machine.target)) < policy().waypoint_tolerance_rad
    assert abs(measured[2] - value["waypoints_rad"][1][2]) > math.radians(10)
    feed(measured, .1)
    with pytest.raises(SafetyError, match="link2_to_link3 exceeds max delta"):
        driver.command_dynamic_exact_advance(value["waypoints_rad"][1], clock + .1)
    assert len(writes) == driver.dynamic_exact_accepted_command_sequence == 1
    machine.observe(sample(measured, pubs=1, lease=True, state="COMMAND_ACTIVE",
                           armed=True, received=2., sequence=1,
                           accepted_target=machine.target, advance_ready=True), now=2.)
    assert machine.state == "WAIT_CONVERGENCE"
    assert machine.waypoint_index == 0 and machine.publish_count == 1
    # Same real feedback stream, 63 ms later: no new command / retry needed.
    measured[2] = 2.169048834
    machine.observe(sample(measured, pubs=1, lease=True, state="COMMAND_ACTIVE",
                           armed=True, received=2.063, sequence=1,
                           accepted_target=machine.target, advance_ready=True), now=2.063)
    assert machine.state == "PUBLISH_WAYPOINT"
    assert machine.waypoint_index == 1 and machine.publish_count == 1
    assert len(machine.waypoints) == 3
    assert machine.target == tuple(value["waypoints_rad"][1])
    feed(measured, .163)
    driver.command_dynamic_exact_advance(machine.target, clock + .163)
    assert len(writes) == driver.dynamic_exact_accepted_command_sequence == 2


@pytest.mark.parametrize("joint", range(6))
def test_handoff_cap_checks_each_joint_before_and_after_quantization(joint):
    machine = ExactPlanExecution(plan(3), policy(), now=0.)
    step = policy().encoder_step_rad
    target = [0.] * 6
    target[joint] = round(.3 / step) * step
    measured = list(target)
    cap = policy().per_command_caps_rad[joint]
    measured[joint] = target[joint] - cap - step
    assert not machine.command_within_caps(target, measured)
    measured[joint] = target[joint] - cap + step
    assert machine.command_within_caps(target, measured)
    # Raw delta fits but rounding would exceed the cap.
    target[joint] += .49 * step
    measured[joint] = target[joint] + cap - .1 * step
    assert not machine.command_within_caps(target, measured)


@pytest.mark.parametrize("terminal", ["cancel", "timeout"])
def test_handoff_wait_does_not_bypass_cancel_or_timeout(terminal):
    machine = ExactPlanExecution(plan(3), policy(), now=0.)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    machine.mark_published(machine.target, now=1.)
    if terminal == "cancel":
        machine.cancel()
        assert machine.terminal_code == "MOVE_CANCELLED"
    else:
        machine.observe(sample(received=7.), now=7.)
        assert machine.terminal_code == "MOVE_WAYPOINT_TIMEOUT"
    assert machine.publish_count == 1


@pytest.mark.parametrize("caps", [None, [], [float("nan")] * 6, [0.] * 6, [1.] * 5])
def test_move_policy_requires_real_per_joint_command_caps(caps):
    import json
    raw = json.loads((PKG.parents[1] / "config/press/releases/t102_dynamic_precontact_policy_release_20260818.json").read_text())
    raw["per_command_caps_rad"] = caps
    with pytest.raises(MoveBlocked, match="MOVE_POLICY_INVALID"):
        MovePolicy.from_mapping(raw)


def test_three_waypoint_smooth_handoff_uses_one_publish_each_and_final_settle():
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    now = 1.0
    pre_publish = tuple(plan(3)["session"]["q0_rad"])
    for index in range(3):
        target = machine.target
        machine.mark_published(target, now=now)
        ack = sample(
            target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
            received=now + 0.05, sequence=index + 1,
            accepted_target=target, advance_ready=True,
        )
        validate_command_ack(
            ack, prior_sequence=index, expected_target=target,
            pre_publish_positions=pre_publish,
            encoder_step_rad=policy().encoder_step_rad,
        )
        machine.observe(ack, now=now + 0.05)
        if index < 2:
            assert machine.state == "PUBLISH_WAYPOINT"
            assert machine.waypoint_index == index + 1
        else:
            assert machine.state == "WAIT_CONVERGENCE"
            machine.observe(
                sample(target, pubs=1, lease=True, received=now + 0.1),
                now=now + 0.1,
            )
            assert machine.state == "TEARDOWN_SUCCESS"
        pre_publish = target
        now += 1.0
    assert machine.publish_count == 3
    assert machine.published_targets == [
        tuple(waypoint) for waypoint in plan(3)["waypoints_rad"]
    ]


class FakeRuntime:
    def __init__(self, machine):
        self.machine = machine
        self.published = []
        self.publisher_count = 0
        self.lease_active = False
        self.armed = False
        self.arm_calls = 0
        self.teardown_calls = 0

    def run_success(self):
        self.machine.precheck(sample())
        self.lease_active = True
        self.machine.lease_acquired(sample(lease=True))
        self.publisher_count = 1
        self.machine.publisher_created(sample(pubs=1, lease=True))
        now = 1.0
        while self.machine.state in {"ARM_WAYPOINT", "PUBLISH_WAYPOINT"}:
            if self.machine.state == "ARM_WAYPOINT":
                self.armed = True
                self.arm_calls += 1
                self.machine.armed(True)
            target = self.machine.target
            self.published.append(target)
            self.machine.mark_published(target, now=now)
            self.machine.observe(
                sample(
                    target, pubs=1, lease=True, state="COMMAND_ACTIVE",
                    armed=True, received=now + 0.05,
                ),
                now=now + 0.05,
            )
            assert self.machine.state == "WAIT_CONVERGENCE"
            self.armed = False
            self.machine.observe(
                sample(target, pubs=1, lease=True, received=now + 0.1),
                now=now + 0.1,
            )
            now += 1.0
        assert self.machine.state == "TEARDOWN_SUCCESS"
        self.teardown(success=True)

    def teardown(self, *, success):
        self.teardown_calls += 1
        self.armed = False
        self.publisher_count = 0
        self.lease_active = False
        self.machine.teardown_complete(success=success)


def test_fake_end_to_end_six_waypoints_exact_once_and_resources_zero():
    machine = ExactPlanExecution(plan(6), policy(), now=0.0)
    runtime = FakeRuntime(machine)
    runtime.run_success()
    assert runtime.published == [tuple(x) for x in plan(6)["waypoints_rad"]]
    assert machine.publish_count == 6
    assert runtime.arm_calls == 6
    assert machine.state == "SUCCEEDED"
    assert machine.terminal_code == "DYNAMIC_PRECONTACT_STOPPED_DISARMED"
    assert runtime.publisher_count == 0 and not runtime.lease_active and not runtime.armed
    assert runtime.teardown_calls == 1


def test_segmented_standby_reuses_ack_protocol_and_ends_exact_disarmed():
    route = standby_plan(4)
    machine = ExactPlanExecution(route, policy(), now=0.0)
    start = tuple(route["start_positions_rad"])
    machine.precheck(sample(start))
    machine.lease_acquired(sample(start, lease=True))
    machine.publisher_created(sample(start, pubs=1, lease=True))
    now = 1.0
    prior_sequence = 10
    pre_publish = start
    for index, expected in enumerate(route["waypoints_rad"]):
        if machine.state == "ARM_WAYPOINT":
            machine.armed(True)
        target = machine.target
        assert target == tuple(expected)
        machine.mark_published(target, now=now)
        ack = sample(
            target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
            received=now + 0.01, sequence=prior_sequence + 1,
            accepted_target=target, advance_ready=True,
        )
        validate_command_ack(
            ack, prior_sequence=prior_sequence, expected_target=target,
            pre_publish_positions=pre_publish,
            encoder_step_rad=policy().encoder_step_rad,
        )
        machine.observe(ack, now=now + 0.01)
        prior_sequence += 1
        if index + 1 == len(route["waypoints_rad"]):
            machine.observe(
                sample(target, pubs=1, lease=True, received=now + 0.02),
                now=now + 0.02,
            )
        now += 1.0
        pre_publish = target
    assert machine.state == "TEARDOWN_SUCCESS"
    assert machine.endpoint_error_rad <= math.radians(0.5)
    machine.teardown_complete(success=True)
    assert machine.state == "SUCCEEDED"
    assert machine.terminal_code == "EXACT_STANDBY_COMPLETE_DISARMED"
    assert machine.published_targets[-1] == tuple(route["standard_standby_rad"])


def test_cold_boot_standby_allows_exact_fault_cleared_state_only_at_start_gates():
    route = cold_boot_standby_plan()
    machine = ExactPlanExecution(route, policy(), now=0.0)
    start = tuple(route["start_positions_rad"])
    machine.precheck(sample(start, state=FAULT_CLEARED_DISARMED))
    machine.lease_acquired(
        sample(start, lease=True, state=FAULT_CLEARED_DISARMED)
    )
    machine.publisher_created(
        sample(start, pubs=1, lease=True, state=FAULT_CLEARED_DISARMED)
    )
    assert machine.state == "ARM_WAYPOINT"

    wrong_bootstrap_state = ExactPlanExecution(route, policy(), now=0.0)
    with pytest.raises(MoveBlocked, match="FAULT_CLEARED_DISARMED required"):
        wrong_bootstrap_state.precheck(sample(start, state="CONVERGED_DISARMED"))

    ordinary = ExactPlanExecution(standby_plan(2), policy(), now=0.0)
    with pytest.raises(MoveBlocked, match="MOVE_SAFETY_GATE_FAILED"):
        ordinary.precheck(
            sample(
                ordinary.plan["start_positions_rad"],
                state=FAULT_CLEARED_DISARMED,
            )
        )


def test_cold_boot_standby_allows_converged_disarmed_with_same_local_plan():
    route = cold_boot_standby_plan("CONVERGED_DISARMED")
    machine = ExactPlanExecution(route, policy(), now=0.0)
    start = tuple(route["start_positions_rad"])
    machine.precheck(sample(start, state="CONVERGED_DISARMED"))
    machine.lease_acquired(
        sample(start, lease=True, state="CONVERGED_DISARMED")
    )
    machine.publisher_created(
        sample(start, pubs=1, lease=True, state="CONVERGED_DISARMED")
    )
    assert machine.state == "ARM_WAYPOINT"
    assert machine.local_standby_bootstrap is True


@pytest.mark.parametrize("initial_state", [
    OPERATOR_DISARMED, FAULT_CLEARED_DISARMED, "TORQUE_ON_VERIFIED",
])
def test_completed_pre_standby_binds_exact_disarmed_start_state(initial_state):
    route = standby_plan(2)
    route["source_precontact_plan_hash"] = "a" * 64
    route["initial_disarmed_state"] = initial_state
    start = tuple(route["start_positions_rad"])
    machine = ExactPlanExecution(
        route, policy(), now=0.0,
        initial_disarmed_state=initial_state,
    )
    machine.precheck(sample(start, state=initial_state))
    machine.lease_acquired(
        sample(start, lease=True, state=initial_state)
    )
    machine.publisher_created(
        sample(start, pubs=1, lease=True, state=initial_state)
    )
    assert machine.state == "ARM_WAYPOINT"

    with pytest.raises(MoveBlocked, match="identity drift"):
        ExactPlanExecution(route, policy(), now=0.0)

    for unsafe in (
        sample(start, state=initial_state, armed=True),
        sample(start, state=initial_state, fault="fault"),
        sample(start, state=initial_state, torque=False),
        sample(start, state=initial_state, age=1.0),
        sample(start, state=initial_state, pubs=1),
        sample(start, state=initial_state, lease=True),
        sample(start, state="CONVERGED_DISARMED"),
        sample(start, state="COMMAND_ACTIVE", armed=True),
    ):
        rejected = ExactPlanExecution(
            route, policy(), now=0.0,
            initial_disarmed_state=initial_state,
        )
        with pytest.raises(MoveBlocked):
            rejected.precheck(unsafe)
    for phase in ("lease_acquired", "publisher_created"):
        drifted = ExactPlanExecution(
            route, policy(), now=0.0, initial_disarmed_state=initial_state,
        )
        drifted.precheck(sample(start, state=initial_state))
        if phase == "publisher_created":
            drifted.lease_acquired(sample(start, state=initial_state, lease=True))
        with pytest.raises(MoveBlocked, match="initial state drift"):
            getattr(drifted, phase)(sample(
                start, state="CONVERGED_DISARMED", lease=True,
                pubs=int(phase == "publisher_created"),
            ))
    missing_source = dict(route, source_precontact_plan_hash="")
    with pytest.raises(MoveBlocked, match="source-route initial"):
        ExactPlanExecution(
            missing_source, policy(), now=0.0, initial_disarmed_state=initial_state,
        )


def test_precontact_torque_start_requires_executor_proof_binding():
    torque = sample(state="TORQUE_ON_VERIFIED")
    unbound = ExactPlanExecution(plan(3), policy(), now=0.0)
    with pytest.raises(MoveBlocked, match="CONVERGED_DISARMED required"):
        unbound.precheck(torque)

    bound = ExactPlanExecution(
        plan(3), policy(), now=0.0,
        initial_disarmed_state="TORQUE_ON_VERIFIED",
    )
    bound.precheck(torque)
    bound.lease_acquired(sample(lease=True, state="TORQUE_ON_VERIFIED"))
    bound.publisher_created(
        sample(pubs=1, lease=True, state="TORQUE_ON_VERIFIED")
    )
    assert bound.state == "ARM_WAYPOINT"


def test_precontact_fault_cleared_start_requires_executor_proof_binding():
    fault_cleared = sample(state=FAULT_CLEARED_DISARMED)
    unbound = ExactPlanExecution(plan(3), policy(), now=0.0)
    with pytest.raises(MoveBlocked, match="CONVERGED_DISARMED required"):
        unbound.precheck(fault_cleared)

    bound = ExactPlanExecution(
        plan(3), policy(), now=0.0,
        initial_disarmed_state=FAULT_CLEARED_DISARMED,
    )
    bound.precheck(fault_cleared)
    bound.lease_acquired(
        sample(lease=True, state=FAULT_CLEARED_DISARMED)
    )
    bound.publisher_created(
        sample(pubs=1, lease=True, state=FAULT_CLEARED_DISARMED)
    )
    assert bound.state == "ARM_WAYPOINT"

    wrong_state = ExactPlanExecution(
        plan(3), policy(), now=0.0,
        initial_disarmed_state=FAULT_CLEARED_DISARMED,
    )
    with pytest.raises(MoveBlocked, match="FAULT_CLEARED_DISARMED required"):
        wrong_state.precheck(sample(state="TORQUE_ON_VERIFIED"))


def test_bound_fault_cleared_execute_reaches_resource_boundary_then_clears_proof():
    class State:
        MOVE_IN_FLIGHT = "MOVE_IN_FLIGHT"
        MOVE_COMPLETE_DISARMED = "MOVE_COMPLETE_DISARMED"
        FAILED_DISARMED = "FAILED_DISARMED"
        IDLE = "IDLE"

    class PrecontactMove:
        class Result:
            pass

        class Feedback:
            pass

    route = plan(3)
    q0 = tuple(route["session"]["q0_rad"])
    ticket = type("Ticket", (), {
        "plan": route, "plan_hash": "fcd-stage-a-plan",
    })()
    stages = []

    async def create_resources(_goal, machine):
        stages.append(machine.state)
        machine.lease_acquired(
            sample(q0, lease=True, state=FAULT_CLEARED_DISARMED)
        )
        machine.publisher_created(
            sample(
                q0, pubs=1, lease=True,
                state=FAULT_CLEARED_DISARMED,
            )
        )
        stages.append(machine.state)
        machine.cancel("test stop after resource boundary")

    async def teardown_resources():
        return True, ""

    execute = extracted_function(
        NODE, "_execute", {
            "PrecontactMove": PrecontactMove,
            "ExactPlanExecution": ExactPlanExecution,
            "MoveBlocked": MoveBlocked,
            "DynamicPrepareBlocked": MoveBlocked,
            "FrozenSession": type(
                "FrozenSession", (), {
                    "from_mapping": staticmethod(lambda value: value),
                },
            ),
            "verify_frozen_plan": lambda value, **_kwargs: value,
            "canonical_hash": lambda _value: "unused",
            "State": State,
            "time": type("Time", (), {"monotonic": staticmethod(lambda: 10.0)}),
            "math": math,
            "FAULT_CLEARED_DISARMED": FAULT_CLEARED_DISARMED,
            "VALIDATE_CONFIRMATION_TOKEN": "VALIDATE_ONLY",
        },
    )
    goal_events = []
    goal = type("Goal", (), {
        "request": type("Request", (), {"confirmation_text": "MOVE"})(),
        "is_cancel_requested": False,
        "succeed": lambda self: goal_events.append("succeed"),
        "canceled": lambda self: goal_events.append("canceled"),
        "abort": lambda self: goal_events.append("abort"),
    })()
    result_fields = []
    node = type("Node", (), {})()
    node._lock = __import__("threading").RLock()
    node._plan = ticket
    node._model = type("Model", (), {"boot_epoch": "boot-fcd"})()
    node._move_policy = policy()
    node._soft_standby_proof = {
        "admission_state": FAULT_CLEARED_DISARMED,
        "executor_boot_epoch": "boot-fcd",
        "measured_q_rad": q0,
    }
    node._soft_precontact_plan_hash = "fcd-stage-a-plan"
    node._state = "AWAIT_MOVE_CONFIRM"
    node._failure = ""
    node._active_machine = None
    node._goal_slot_reserved = True
    node._authorization_id = "auth"
    node._last_completed_plan = None
    node._resource_counts = {"action_call": 0}
    node._safety_sample = lambda: sample(
        q0, state=FAULT_CLEARED_DISARMED,
    )
    node._exact_goal_locked = lambda _request: True
    node._move_gate = lambda: (True, "")
    node._create_execution_resources = create_resources
    node._teardown_execution_resources = teardown_resources
    node._fill_result = lambda result, **kwargs: result_fields.append(kwargs)
    node._publish_status = lambda: None

    asyncio.run(execute(node, goal))

    assert stages == ["ACQUIRE_LEASE", "ARM_WAYPOINT"]
    assert goal_events == ["abort"]
    assert result_fields[0]["code"] == "MOVE_CANCELLED"
    assert node._soft_standby_proof == {}
    assert node._soft_precontact_plan_hash == ""
    assert node._plan is None
    assert node._resource_counts["action_call"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"armed": True},
        {"dry": True},
        {"fault": "fault"},
        {"torque": False},
        {"voltage": 10.0},
        {"age": 0.6},
    ],
)
def test_cold_boot_standby_rejects_unhealthy_fault_cleared_sample(kwargs):
    route = cold_boot_standby_plan()
    machine = ExactPlanExecution(route, policy(), now=0.0)
    with pytest.raises(MoveBlocked, match="MOVE_"):
        machine.precheck(
            sample(
                route["start_positions_rad"],
                state=FAULT_CLEARED_DISARMED,
                **kwargs,
            )
        )


def test_cold_boot_standby_rejects_nonfinite_joint_sample():
    route = cold_boot_standby_plan()
    machine = ExactPlanExecution(route, policy(), now=0.0)
    positions = list(route["start_positions_rad"])
    positions[2] = math.nan
    with pytest.raises(MoveBlocked, match="six finite joints"):
        machine.precheck(sample(positions, state=FAULT_CLEARED_DISARMED))


def test_cold_boot_direct_q0_still_requires_ack_and_exact_converged_terminal():
    route = cold_boot_standby_plan()
    machine = ExactPlanExecution(route, policy(), now=0.0)
    start = tuple(route["start_positions_rad"])
    target = tuple(route["standard_standby_rad"])
    machine.precheck(sample(start, state=FAULT_CLEARED_DISARMED))
    machine.lease_acquired(
        sample(start, lease=True, state=FAULT_CLEARED_DISARMED)
    )
    machine.publisher_created(
        sample(start, pubs=1, lease=True, state=FAULT_CLEARED_DISARMED)
    )
    machine.armed(True)
    machine.mark_published(target, now=1.0)
    acknowledged = sample(
        start, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
        received=1.01, sequence=1, accepted_target=target,
    )
    validate_command_ack(
        acknowledged, prior_sequence=0, expected_target=target,
        pre_publish_positions=start,
        encoder_step_rad=policy().encoder_step_rad,
    )
    machine.observe(acknowledged, now=1.01)
    assert machine.state == "WAIT_CONVERGENCE"
    machine.observe(
        sample(target, pubs=1, lease=True, received=1.02), now=1.02,
    )
    assert machine.state == "TEARDOWN_SUCCESS"
    assert machine.endpoint_error_rad <= math.radians(0.5)
    machine.teardown_complete(success=True)
    assert machine.state == "SUCCEEDED"
    assert machine.publish_count == 1
    assert machine.published_targets == [target]


def test_cold_boot_standby_accepts_one_but_ordinary_standby_rejects_one():
    route = cold_boot_standby_plan()
    assert len(ExactPlanExecution(route, policy(), now=0.0).waypoints) == 1
    ordinary = standby_plan(2)
    ordinary["waypoints_rad"] = [ordinary["waypoints_rad"][-1]]
    with pytest.raises(MoveBlocked, match="exactly 2-64 frozen waypoints"):
        ExactPlanExecution(ordinary, policy(), now=0.0)


def test_standby_contract_rejects_retry_compensation_and_nonexact_endpoint():
    for field in ("automatic_retry", "automatic_compensation", "overshoot", "clear_fault"):
        route = standby_plan(4)
        route[field] = True
        with pytest.raises(MoveBlocked, match="PLAN_INVALIDATED"):
            ExactPlanExecution(route, policy(), now=0.0)
    route = standby_plan(4)
    route["waypoints_rad"][-1][2] += 0.01
    with pytest.raises(MoveBlocked, match="standby endpoint"):
        ExactPlanExecution(route, policy(), now=0.0)


def test_each_intermediate_waypoint_requires_disarmed_terminal_before_rearm():
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    first = machine.target
    machine.mark_published(first, now=1.0)
    machine.observe(sample(first, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True, received=1.1), now=1.1)
    assert machine.state == "WAIT_CONVERGENCE"
    assert machine.waypoint_index == 0
    assert machine.publish_count == 1
    machine.observe(sample(first, pubs=1, lease=True, received=1.2), now=1.2)
    assert machine.state == "ARM_WAYPOINT"
    assert machine.waypoint_index == 1
    machine.armed(True)
    second = machine.target
    machine.mark_published(second, now=1.3)
    machine.observe(sample(second, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True, received=1.4), now=1.4)
    assert machine.state == "WAIT_CONVERGENCE"
    assert machine.waypoint_index == 1
    machine.observe(sample(second, pubs=1, lease=True, received=1.5), now=1.5)
    assert machine.state == "ARM_WAYPOINT"
    assert machine.waypoint_index == 2
    machine.armed(True)
    final = machine.target
    machine.mark_published(final, now=1.6)
    machine.observe(sample(final, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True, received=1.7), now=1.7)
    assert machine.state == "WAIT_CONVERGENCE"
    machine.observe(sample(final, pubs=1, lease=True, received=1.8), now=1.8)
    assert machine.state == "TEARDOWN_SUCCESS"


def test_consumed_execution_may_finish_after_admission_ticket_ttl():
    value = plan(6)
    value["ttl_s"] = 0.01
    value["created_monotonic"] = 0.0
    machine = ExactPlanExecution(value, policy(), now=0.009)
    runtime = FakeRuntime(machine)
    runtime.run_success()
    # Fake time reaches 6.1 s, long after the admission-only 0.01 s ticket.
    assert machine.state == "SUCCEEDED" and machine.publish_count == 6


@pytest.mark.parametrize("field,value", [
    ("feedback_max_age_s", float("inf")), ("waypoint_tolerance_rad", 0.0),
    ("waypoint_timeout_s", 31.0), ("min_voltage_v", 14.0),
])
def test_policy_invalid(field, value):
    raw = {
        "move_feedback_max_age_s": 0.5, "move_waypoint_tolerance_rad": 0.005,
        "move_waypoint_timeout_s": 5.0, "move_min_voltage_v": 11.0,
        "move_max_voltage_v": 13.0,
        "encoder_step_rad": 2.0 * math.pi / 4096.0,
    }
    raw[{"feedback_max_age_s":"move_feedback_max_age_s", "waypoint_tolerance_rad":"move_waypoint_tolerance_rad", "waypoint_timeout_s":"move_waypoint_timeout_s", "min_voltage_v":"move_min_voltage_v"}[field]] = value
    with pytest.raises(MoveBlocked, match="MOVE_POLICY_INVALID"):
        MovePolicy.from_mapping(raw)


@pytest.mark.parametrize("kwargs", [
    {"age": 0.6}, {"state": "FAULT:bad"}, {"armed": True}, {"dry": True},
    {"fault": "bad"}, {"torque": False}, {"voltage": 10.0}, {"pubs": 1},
])
def test_precheck_failures_create_zero_resources(kwargs):
    machine = ExactPlanExecution(plan(), policy(), now=0.0)
    with pytest.raises(MoveBlocked):
        machine.precheck(sample(**kwargs))
    assert machine.publish_count == 0 and machine.state == "PRECHECK"


def test_precheck_preserves_fresh_torque_fault_before_stale_source_stamp():
    machine = ExactPlanExecution(plan(), policy(), now=0.0)
    with pytest.raises(
        MoveBlocked,
        match=(
            "MOVE_SAFETY_GATE_FAILED: FAULT:one or more joint torque switches "
            "are off"
        ),
    ):
        machine.precheck(sample(
            state="FAULT:one or more joint torque switches are off",
            fault="one or more joint torque switches are off", torque=False,
            stamp_age=99.0,
        ))
    assert machine.publish_count == 0 and machine.state == "PRECHECK"


def test_retained_or_stale_lease_status_never_proves_owner_health():
    machine = ExactPlanExecution(plan(), policy(), now=0.0)
    stale = SafetySample(
        (0.0, -1.0, 2.0, 0.1, -1.5, 0.0), 0.01,
        "CONVERGED_DISARMED", False, False, "", (True,) * 6, 12.0,
        0, False, "", False, 99.0, 100.0, 0.01,
    )
    with pytest.raises(MoveBlocked, match="lease status is stale"):
        machine.precheck(stale)


def test_stale_fault_owner_timeout_and_cancel_require_teardown_without_retry():
    cases = [
        (sample(pubs=1, lease=True, age=0.6), 1.1, "MOVE_FEEDBACK_STALE"),
        (sample(pubs=1, lease=True, fault="bad"), 1.1, "MOVE_SAFETY_GATE_FAILED"),
        (sample(pubs=0, lease=True), 1.1, "MOVE_OWNER_DRIFT"),
        (sample(pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True), 6.0, "MOVE_WAYPOINT_TIMEOUT"),
    ]
    for observed, at, code in cases:
        machine = ExactPlanExecution(plan(3), policy(), now=0.0)
        machine.precheck(sample()); machine.lease_acquired(sample(lease=True)); machine.publisher_created(sample(pubs=1, lease=True)); machine.armed(True)
        machine.mark_published(machine.target, now=1.0)
        machine.observe(observed, now=at)
        assert machine.state == "TEARDOWN_FAILURE" and machine.terminal_code == code
        assert machine.publish_count == 1
        machine.teardown_complete(success=False)
        assert machine.state == "FAILED"
    cancelled = ExactPlanExecution(plan(3), policy(), now=0.0)
    cancelled.cancel()
    assert cancelled.state == "TEARDOWN_CANCEL"
    cancelled.teardown_complete(success=False)
    assert cancelled.state == "CANCELLED"


@pytest.mark.parametrize(
    "observed,expected_reason",
    [
        (
            sample(
                pubs=1, lease=True, state="FAULT:one or more joint torque switches are off",
                fault="one or more joint torque switches are off", torque=False,
                stamp_age=99.0,
            ),
            "FAULT:one or more joint torque switches are off",
        ),
        (
            SafetySample(
                (0.0,) * 6, 0.01, "COMMAND_ACTIVE", False, False, "",
                (True, True, False, True, True, True), 12.0, 1, True,
                DYNAMIC_LEASE_PURPOSE, True, 0.01, 100.0, 99.0,
            ),
            "TORQUE_SWITCH_OFF:5/6 enabled",
        ),
    ],
)
def test_fresh_torque_dropout_precedes_stale_source_stamp(observed, expected_reason):
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    machine.mark_published(machine.target, now=1.0)

    machine.observe(observed, now=1.1)

    assert machine.state == "TEARDOWN_FAILURE"
    assert machine.terminal_code == "MOVE_SAFETY_GATE_FAILED"
    assert machine.terminal_reason == expected_reason
    assert machine.publish_count == 1
    machine.teardown_complete(success=False)
    assert machine.state == "FAILED"


@pytest.mark.parametrize("observed,code", [
    (sample(pubs=1, lease=True, age=float("nan")), "MOVE_FEEDBACK_STALE"),
    (sample(pubs=1, lease=True, age=-0.1), "MOVE_FEEDBACK_STALE"),
    (sample((float("nan"), 0, 0, 0, 0, 0), pubs=1, lease=True), "MOVE_SAFETY_GATE_FAILED"),
    (SafetySample((0.0,) * 6, 0.1, "COMMAND_ACTIVE", False, False, "", (True,) * 6, 12.0, 1, True, DYNAMIC_LEASE_PURPOSE, True, 0.01, 100.0, 0.01), "MOVE_SAFETY_GATE_FAILED"),
    (sample(pubs=1, lease=True, state="FAULT:bad"), "MOVE_SAFETY_GATE_FAILED"),
    (SafetySample((0.0,) * 6, 0.1, "COMMAND_ACTIVE", True, False, "", (True,) * 5, 12.0, 1, True, DYNAMIC_LEASE_PURPOSE, True, 0.01, 100.0, 0.01), "MOVE_SAFETY_GATE_FAILED"),
    (sample(pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True, voltage=float("nan")), "MOVE_SAFETY_GATE_FAILED"),
    (SafetySample((0.0,) * 6, 0.1, "COMMAND_ACTIVE", True, False, "", (True,) * 6, 12.0, 1, True, DYNAMIC_LEASE_PURPOSE, True, 0.01, 100.0, 99.0), "MOVE_FEEDBACK_STALE"),
])
def test_active_observation_is_structurally_fail_closed(observed, code):
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample()); machine.lease_acquired(sample(lease=True)); machine.publisher_created(sample(pubs=1, lease=True)); machine.armed(True)
    machine.mark_published(machine.target, now=1.0)
    machine.observe(observed, now=1.1)
    assert machine.state == "TEARDOWN_FAILURE"
    assert machine.terminal_code == code


@pytest.mark.parametrize("now", [float("nan"), 0.5])
def test_nonfinite_or_regressed_monotonic_time_fails_closed(now):
    machine = ExactPlanExecution(plan(3), policy(), now=1.0)
    machine.precheck(sample()); machine.lease_acquired(sample(lease=True)); machine.publisher_created(sample(pubs=1, lease=True)); machine.armed(True)
    if math.isnan(now):
        with pytest.raises(MoveBlocked, match="MOVE_CLOCK_INVALID"):
            machine.mark_published(machine.target, now=now)
    else:
        with pytest.raises(MoveBlocked, match="MOVE_CLOCK_INVALID"):
            machine.mark_published(machine.target, now=now)


def test_duplicate_publish_and_modified_target_fail_closed():
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample()); machine.lease_acquired(sample(lease=True)); machine.publisher_created(sample(pubs=1, lease=True)); machine.armed(True)
    with pytest.raises(MoveBlocked, match="PLAN_INVALIDATED"):
        machine.mark_published([9.0] * 6, now=1.0)
    machine.mark_published(machine.target, now=1.0)
    with pytest.raises(MoveBlocked, match="MOVE_PROTOCOL_ERROR"):
        machine.mark_published(machine.target, now=1.1)


@pytest.mark.parametrize("mutation", [
    {"waypoints_rad": [[0.0] * 6] * 2}, {"automatic_retry": True},
    {"fixed_v2_fallback": True}, {"no_contact": False}, {"press": True},
])
def test_plan_shape_and_forbidden_semantics(mutation):
    value = plan(3); value.update(mutation)
    with pytest.raises(MoveBlocked, match="PLAN_INVALIDATED"):
        ExactPlanExecution(value, policy(), now=0.0)


def test_zero_delta_waypoint_is_rejected_but_single_joint_still_uses_dynamic_t102_mode():
    value = plan(3)
    value["waypoints_rad"][0] = list(value["session"]["q0_rad"])
    with pytest.raises(MoveBlocked, match="PLAN_COMMAND_SHAPE_REJECTED"):
        ExactPlanExecution(value, policy(), now=0.0)


def test_sub_tolerance_point_and_direct_disarmed_ack_are_rejected():
    value = plan(3)
    q0 = list(value["session"]["q0_rad"])
    value["waypoints_rad"][0] = [q0[0] + 0.004, *q0[1:]]
    with pytest.raises(MoveBlocked, match="PLAN_COMMAND_SHAPE_REJECTED"):
        ExactPlanExecution(value, policy(), now=0.0)

    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    machine.mark_published(machine.target, now=1.0)
    machine.observe(
        sample(machine.target, pubs=1, lease=True, received=1.1), now=1.1
    )
    assert machine.state == "TEARDOWN_FAILURE"
    assert machine.terminal_code == "MOVE_COMMAND_ACK_MISSING"


def test_disarmed_endpoint_outside_existing_tolerance_fails_closed():
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    target = machine.target
    machine.mark_published(target, now=1.0)
    machine.observe(
        sample(
            target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
            received=1.1,
        ),
        now=1.1,
    )
    outside = list(target)
    outside[0] += policy().waypoint_tolerance_rad + 0.001
    machine.observe(
        sample(outside, pubs=1, lease=True, received=1.2), now=1.2
    )
    assert machine.state == "TEARDOWN_FAILURE"
    assert machine.terminal_code == "MOVE_ENDPOINT_ERROR"
    assert machine.publish_count == 1


def test_pre_publish_feedback_identity_cannot_acknowledge_command():
    machine = ExactPlanExecution(plan(3), policy(), now=0.0)
    machine.precheck(sample())
    machine.lease_acquired(sample(lease=True))
    machine.publisher_created(sample(pubs=1, lease=True))
    machine.armed(True)
    machine.mark_published(machine.target, now=1.0)
    machine.observe(
        sample(
            pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
            received=1.0,
        ),
        now=1.1,
    )
    assert machine.terminal_code == "MOVE_COMMAND_ACK_STALE"


def test_lease_reconciliation_requires_fresh_exact_identity():
    expected = "lease-exact"
    assert lease_reconciliation_decision(
        status_age_s=0.1, maximum_age_s=0.5, active=False,
        purpose="", lease_id="", expected_lease_id=expected,
    ) == "PROVEN_INACTIVE"
    assert lease_reconciliation_decision(
        status_age_s=0.1, maximum_age_s=0.5, active=True,
        purpose=DYNAMIC_LEASE_PURPOSE, lease_id=expected,
        expected_lease_id=expected,
    ) == "RELEASE_EXACT_ACTIVE"
    with pytest.raises(MoveBlocked, match="FOREIGN"):
        lease_reconciliation_decision(
            status_age_s=0.1, maximum_age_s=0.5, active=True,
            purpose=DYNAMIC_LEASE_PURPOSE, lease_id="other",
            expected_lease_id=expected,
        )
    with pytest.raises(MoveBlocked, match="UNKNOWN"):
        lease_reconciliation_decision(
            status_age_s=99.0, maximum_age_s=0.5, active=False,
            purpose="", lease_id="", expected_lease_id=expected,
        )
