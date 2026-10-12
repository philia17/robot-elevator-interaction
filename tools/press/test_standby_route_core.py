from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from press.standby_route_core import (
    BOOTSTRAP_ENVELOPE_RAD,
    BOOTSTRAP_STATE,
    COMPLETED_PRE_RETURN_START_LIMIT_RAD,
    SOURCE_ROUTE_INITIAL_STATE,
    SOFT_STANDBY_LIMIT_RAD,
    StandbyRouteBlocked,
    canonical_hash,
    consume_standby_route,
    prepare_standby_route,
)
from press.t102_dynamic_precontact_core import DynamicPrepareBlocked, FKState, PreparedPlan


@dataclass
class FakeModel:
    policy: dict
    boot_epoch: str = "boot-1"
    hashes: dict = None

    def __post_init__(self):
        if self.hashes is None:
            self.hashes = {"model": "a" * 64}

    def _validate_limits(self, q):
        if len(q) != 6 or any(not math.isfinite(float(value)) for value in q):
            raise DynamicPrepareBlocked("NONFINITE_INPUT")
        for index, (value, limits) in enumerate(zip(q, self.policy["joint_hard_limits_rad"])):
            if not limits[0] <= value <= limits[1]:
                raise DynamicPrepareBlocked("JOINT_LIMIT_REJECTED", str(index))

    def fk(self, q):
        points = tuple((0.0, 0.0, 0.0) for _ in range(8))
        return FKState((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), points)


def policy(**overrides):
    value = {
        "standard_standby_rad": [0.0] * 6,
        "encoder_step_rad": 0.01,
        "per_command_caps_rad": [0.1] * 6,
        "move_waypoint_tolerance_rad": math.radians(0.5),
        "ticket_ttl_s": 10.0,
        "q0_drift_limit_rad": 0.02,
        "joint_hard_limits_rad": [[-1.0, 1.0]] * 6,
        "segment_sample_count": 9,
        "capsules": [],
        "self_pairs": [],
        "base_check_capsules": [],
        "bracket_check_capsules": [],
        "base_proxy": {"primitive": "sphere", "center_m": [0, 0, 0], "radius_m": 0.0},
        "camera_bracket_proxy": {"primitive": "sphere", "center_m": [0, 0, 0], "radius_m": 0.0},
        "wall_margin_m": 0.0,
    }
    value.update(overrides)
    return value


def source_plan(model, outbound):
    payload = {
        "schema": "t102_dynamic_precontact_plan_v1",
        "model_hashes": model.hashes,
        "session": {
            "session_id": "session-1",
            "wall_surface_point_base_m": [-1.0, 0.0, 0.0],
        },
        "normal_proof": {"n_out": [1.0, 0.0, 0.0]},
        "waypoints_rad": outbound,
    }
    return PreparedPlan(payload, canonical_hash(payload), 100.0, True)


def test_segmented_reverse_route_respects_caps_quantization_and_exact_q0():
    model = FakeModel(policy())
    outbound = [[0, 0, 0.25, 0, 0, 0], [0, 0, 0.55, 0, 0, 0]]
    ticket = prepare_standby_route(
        model, source_plan(model, outbound), outbound[-1],
        ui_request_id="sta-1", now_monotonic=1.0,
    )
    waypoints = ticket.payload["waypoints_rad"]
    assert len(waypoints) >= 2
    assert tuple(waypoints[-1]) == (0.0,) * 6
    previous = tuple(outbound[-1])
    for waypoint in waypoints:
        assert all(abs(value / 0.01 - round(value / 0.01)) < 1e-9 for value in waypoint)
        assert max(abs(a - b) for a, b in zip(waypoint, previous)) <= 0.1 + 1e-12
        previous = tuple(waypoint)
    assert ticket.payload["automatic_retry"] is False
    assert ticket.payload["automatic_compensation"] is False
    assert ticket.payload["overshoot"] is False
    assert ticket.payload["clear_fault"] is False


def test_completed_pre_terminal_with_observed_endpoint_error_can_return():
    model = FakeModel(policy())
    outbound = [[0, 0, 0.25, 0, 0, 0], [0, 0, 0.55, 0, 0, 0]]
    endpoint_error = 0.01994
    current = [0, 0, outbound[-1][2] + endpoint_error, 0, 0, 0]

    ticket = prepare_standby_route(
        model, source_plan(model, outbound), current,
        ui_request_id="today-completed-pre", now_monotonic=1.0,
    )

    assert math.degrees(endpoint_error) == pytest.approx(1.1425, abs=0.0001)
    assert COMPLETED_PRE_RETURN_START_LIMIT_RAD == pytest.approx(
        math.radians(2.0)
    )
    assert ticket.payload["start_positions_rad"] == tuple(current)
    assert ticket.payload["source_precontact_plan_hash"]
    assert ticket.payload["path_validation"]["mode"] == (
        "REVERSE_OF_FROZEN_OUTBOUND_WITH_ACTUAL_START_RESAMPLED"
    )
    assert len(ticket.payload["waypoints_rad"]) == 7
    assert ticket.payload["waypoints_rad"][-1] == (0.0,) * 6


@pytest.mark.parametrize("initial_state", [
    SOURCE_ROUTE_INITIAL_STATE, "FAULT_CLEARED_DISARMED", "TORQUE_ON_VERIFIED",
])
def test_completed_pre_disarmed_state_is_exactly_hash_bound(initial_state):
    model = FakeModel(policy())
    outbound = [[0, 0, 0.25, 0, 0, 0], [0, 0, 0.55, 0, 0, 0]]
    source = source_plan(model, outbound)
    ticket = prepare_standby_route(
        model, source, outbound[-1],
        ui_request_id="operator-disarmed-return", now_monotonic=1.0,
        bootstrap_state=initial_state,
    )

    assert ticket.payload["source_precontact_plan_hash"] == source.plan_hash
    assert ticket.payload["initial_disarmed_state"] == initial_state
    assert ticket.plan_hash == canonical_hash(ticket.payload)
    changed = dict(ticket.payload)
    changed["initial_disarmed_state"] = "DISARMED:unknown"
    assert canonical_hash(changed) != ticket.plan_hash

    with pytest.raises(StandbyRouteBlocked, match="STANDBY_ROUTE_STATE_REJECTED"):
        prepare_standby_route(
            model, source, outbound[-1],
            ui_request_id="unknown-disarmed-return", now_monotonic=1.0,
            bootstrap_state="DISARMED:unknown",
        )


def test_completed_pre_return_start_still_rejects_more_than_two_degrees():
    model = FakeModel(policy())
    outbound = [[0, 0, 0.25, 0, 0, 0], [0, 0, 0.55, 0, 0, 0]]
    current = [0, 0, outbound[-1][2] + math.radians(2.001), 0, 0, 0]

    with pytest.raises(StandbyRouteBlocked, match="STANDBY_ROUTE_ENDPOINT_DRIFT"):
        prepare_standby_route(
            model, source_plan(model, outbound), current,
            ui_request_id="over-two-degrees", now_monotonic=1.0,
        )


def test_already_at_q0_is_accepted_no_motion_and_single_use():
    model = FakeModel(policy())
    ticket = prepare_standby_route(
        model, None, [0.0] * 6, ui_request_id="sta-q0", now_monotonic=2.0,
    )
    assert ticket.payload["already_at_standby"] is True
    assert ticket.payload["waypoints_rad"] == []
    consumed = consume_standby_route(
        ticket, model=model, current_positions_rad=[0.0] * 6, now_monotonic=2.1,
    )
    assert consumed.consumed is True
    with pytest.raises(StandbyRouteBlocked, match="REPLAYED"):
        consume_standby_route(
            consumed, model=model, current_positions_rad=[0.0] * 6,
            now_monotonic=2.2,
        )


def test_source_less_fault_cleared_bootstrap_is_one_direct_quantized_exact_q0():
    model = FakeModel(policy())
    current = [math.radians(2.0), 0.0, math.radians(-2.5), 0.0, 0.0, 0.0]
    ticket = prepare_standby_route(
        model, None, current, ui_request_id="cold-boot", now_monotonic=2.0,
        bootstrap_state=BOOTSTRAP_STATE,
    )
    assert ticket.payload["bootstrap_identity"] == {
        "mode": "LOCAL_STANDARD_STANDBY_CORRECTION",
        "admission_state": BOOTSTRAP_STATE,
        "per_joint_envelope_rad": BOOTSTRAP_ENVELOPE_RAD,
        "start_max_joint_error_rad": math.radians(2.5),
    }
    waypoints = ticket.payload["waypoints_rad"]
    assert len(waypoints) == 1
    assert tuple(waypoints[-1]) == (0.0,) * 6
    for waypoint in waypoints:
        assert all(abs(value / 0.01 - round(value / 0.01)) < 1e-9 for value in waypoint)
    assert ticket.plan_hash == canonical_hash(ticket.payload)
    changed = dict(ticket.payload)
    changed["bootstrap_identity"] = dict(changed["bootstrap_identity"])
    changed["bootstrap_identity"]["admission_state"] = "CONVERGED_DISARMED"
    assert canonical_hash(changed) != ticket.plan_hash


def test_source_less_converged_bootstrap_uses_same_one_waypoint_identity():
    model = FakeModel(policy())
    current = [math.radians(2.0), 0.0, math.radians(-2.5), 0.0, 0.0, 0.0]
    ticket = prepare_standby_route(
        model, None, current, ui_request_id="cold-converged",
        now_monotonic=2.0, bootstrap_state="CONVERGED_DISARMED",
    )
    assert ticket.payload["bootstrap_identity"]["admission_state"] == (
        "CONVERGED_DISARMED"
    )
    assert ticket.payload["waypoints_rad"] == [tuple([0.0] * 6)]


def test_converged_state_does_not_replace_available_consumed_reverse_route():
    model = FakeModel(policy())
    outbound = [
        [0.0, 0.0, 0.02, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.04, 0.0, 0.0, 0.0],
    ]
    ticket = prepare_standby_route(
        model, source_plan(model, outbound), outbound[-1],
        ui_request_id="consumed-near-q0", now_monotonic=2.0,
        bootstrap_state="CONVERGED_DISARMED",
    )
    assert "bootstrap_identity" not in ticket.payload
    assert len(ticket.payload["waypoints_rad"]) >= 2
    assert ticket.payload["path_validation"]["mode"] == (
        "REVERSE_OF_FROZEN_OUTBOUND_WITH_ACTUAL_START_RESAMPLED"
    )


@pytest.mark.parametrize(
    "state",
    ["CONVERGED_DISARMED", "FAULT_CLEARED_DISARMED", "TORQUE_ON_VERIFIED"],
)
def test_source_less_soft_standby_is_identity_bound_no_motion(state):
    model = FakeModel(policy())
    current = [math.radians(1.58), 0.0, 0.0, 0.0, 0.0, 0.0]
    ticket = prepare_standby_route(
        model, None, current, ui_request_id=f"soft-{state}",
        now_monotonic=2.0, bootstrap_state=state,
    )
    assert ticket.payload["already_at_standby"] is True
    assert ticket.payload["waypoints_rad"] == []
    assert ticket.payload["soft_standby_identity"] == {
        "mode": "SOURCELESS_SOFT_STANDBY_NO_MOTION",
        "admission_state": state,
        "measured_q_rad": tuple(current),
        "max_joint_error_rad": math.radians(1.58),
        "limit_rad": SOFT_STANDBY_LIMIT_RAD,
    }
    assert ticket.plan_hash == canonical_hash(ticket.payload)


def test_torque_on_verified_soft_standby_rejects_more_than_two_degrees():
    model = FakeModel(policy())
    with pytest.raises(StandbyRouteBlocked, match="STATE_REJECTED"):
        prepare_standby_route(
            model, None, [math.radians(2.01)] + [0.0] * 5,
            ui_request_id="soft-too-far", now_monotonic=2.0,
            bootstrap_state="TORQUE_ON_VERIFIED",
        )


@pytest.mark.parametrize("state", [
    "CONVERGED_DISARMED", "FAULT_CLEARED_DISARMED", "TORQUE_ON_VERIFIED",
])
@pytest.mark.parametrize("error_deg", [0.44, 1.58])
def test_returned_standby_ignores_retained_pre_route_without_motion(state, error_deg):
    model = FakeModel(policy())
    old = source_plan(model, [[0, 0, .25, 0, 0, 0], [0, 0, .55, 0, 0, 0]])
    current = [math.radians(error_deg)] + [0.0] * 5
    ticket = prepare_standby_route(
        model, old, current, ui_request_id="returned-standby",
        now_monotonic=2.0, bootstrap_state=state,
    )
    assert ticket.payload["already_at_standby"] is True
    assert ticket.payload["waypoints_rad"] == []
    assert "source_precontact_plan_hash" not in ticket.payload
    assert ticket.payload["clear_fault"] is False
    assert old.consumed is True  # No mutation/discard of the previous plan here.
    assert consume_standby_route(
        ticket, model=model, current_positions_rad=current, now_monotonic=2.1,
    ).consumed is True


def test_retained_pre_outside_standby_envelope_still_rejects_drift():
    model = FakeModel(policy())
    old = source_plan(model, [[0, 0, .25, 0, 0, 0], [0, 0, .55, 0, 0, 0]])
    with pytest.raises(StandbyRouteBlocked, match="STANDBY_ROUTE_ENDPOINT_DRIFT"):
        prepare_standby_route(
            model, old, [math.radians(2.01)] + [0.0] * 5,
            ui_request_id="not-standby", now_monotonic=2.0,
            bootstrap_state="FAULT_CLEARED_DISARMED",
        )


def test_fault_cleared_more_than_two_degrees_keeps_existing_bootstrap_route():
    model = FakeModel(policy())
    ticket = prepare_standby_route(
        model, None, [math.radians(2.01)] + [0.0] * 5,
        ui_request_id="fault-cleared-bootstrap", now_monotonic=2.0,
        bootstrap_state=BOOTSTRAP_STATE,
    )
    assert "soft_standby_identity" not in ticket.payload
    assert ticket.payload["bootstrap_identity"]["admission_state"] == (
        BOOTSTRAP_STATE
    )
    assert ticket.payload["already_at_standby"] is False
    assert len(ticket.payload["waypoints_rad"]) == 1


@pytest.mark.parametrize(
    "state,current,code",
    [
        ("FAULTED_DISARMED", [math.radians(2.0)] * 6, "STATE_REJECTED"),
        (BOOTSTRAP_STATE, [math.radians(3.01)] + [0.0] * 5, "ENVELOPE_REJECTED"),
        ("CONVERGED_DISARMED", [math.radians(3.01)] + [0.0] * 5, "ENVELOPE_REJECTED"),
    ],
)
def test_source_less_bootstrap_rejects_wrong_state_or_more_than_three_degrees(
    state, current, code,
):
    model = FakeModel(policy())
    with pytest.raises(StandbyRouteBlocked, match=code):
        prepare_standby_route(
            model, None, current, ui_request_id="cold-bad", now_monotonic=2.0,
            bootstrap_state=state,
        )


def test_source_less_non_bootstrap_still_requires_consumed_pre_route():
    model = FakeModel(policy())
    with pytest.raises(StandbyRouteBlocked, match="STANDBY_ROUTE_SOURCE_REQUIRED"):
        prepare_standby_route(
            model, None, [math.radians(2.0)] * 6,
            ui_request_id="no-source", now_monotonic=2.0,
        )


@pytest.mark.parametrize(
    "current,code",
    [
        ([0, 0, math.nan, 0, 0, 0], "STANDBY_ROUTE_INVALID"),
        ([0, 0, 1.1, 0, 0, 0], "JOINT_LIMIT_REJECTED"),
    ],
)
def test_bad_nonfinite_and_limits_reject(current, code):
    model = FakeModel(policy())
    with pytest.raises((StandbyRouteBlocked, DynamicPrepareBlocked), match=code):
        prepare_standby_route(
            model, None, current, ui_request_id="bad", now_monotonic=1.0,
        )


def test_proxy_collision_rejects_return_route():
    collision_policy = policy(
        capsules=[{"name": "tool", "joint_a": 0, "joint_b": 1, "radius_m": 0.1}],
        wall_margin_m=0.01,
    )
    model = FakeModel(collision_policy)
    outbound = [[0, 0, 0.2, 0, 0, 0], [0, 0, 0.4, 0, 0, 0]]
    source = source_plan(model, outbound)
    source.plan["session"]["wall_surface_point_base_m"] = [1.0, 0.0, 0.0]
    source = PreparedPlan(source.plan, canonical_hash(source.plan), 100.0, True)
    with pytest.raises(StandbyRouteBlocked, match="WALL_HALFSPACE_REJECTED"):
        prepare_standby_route(
            model, source, outbound[-1], ui_request_id="collision", now_monotonic=1.0,
        )
