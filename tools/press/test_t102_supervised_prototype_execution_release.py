import copy
import json
import math

import pytest

from press.t102_outbound_executor_core import (
    ExecutionTicket,
    ExecutorConfig,
    FakeController,
    Feedback,
    SealedOutboundRoute,
    StatusTargetAck,
    T102ExecutorError,
    T102OutboundExecutorCore,
    canonical_sha256 as core_sha256,
    setpoint_sha256,
)
from press.t102_outbound_route_builder import build_t102_outbound_diagnostic_route
from press.t102_supervised_prototype_execution_release import (
    SupervisedReleaseError,
    build_supervised_executable_route,
    build_supervised_release,
    canonical_sha256,
    validate_supervised_executable_route,
    validate_supervised_release,
)


Q0 = [0.042951462, -1.188835111, 2.330116817, 0.458660256, -1.515573018, 0.001533980589793149]
Q1 = [0.22174167293271996, -0.8355064319221046, 2.0873657212767704, 0.09539618810757883, -1.4938576229532001, 0.001533980589793149]
HASHES = {
    "m0_target_sha256": "1" * 64,
    "fn_sha256": "2" * 64,
    "r6_sha256": "3" * 64,
    "provenance_sha256": "4" * 64,
}


def _bundle():
    diagnostic = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=Q0,
        verified_static_ik_candidate_rad=Q1,
        frozen_bindings=HASHES,
        max_profile_delta_fraction=0.105,
    )
    route = build_supervised_executable_route(
        diagnostic,
        diagnostic_route_sha256=diagnostic["route_sha256"],
        route_identity="test-v5",
    )
    bindings = {
        "target_artifact_file_sha256": "5" * 64,
        "q0_sha256": "6" * 64,
        "fn_sha256": HASHES["fn_sha256"],
        "r6_sha256": HASHES["r6_sha256"],
        "camera_acceptance_sha256": "7" * 64,
        "ik_fk_evidence_file_sha256": "8" * 64,
        "geometry_profile_file_sha256": "9" * 64,
        "continuous_clearance_file_sha256": "a" * 64,
        "continuous_clearance_canonical_sha256": "b" * 64,
        "source_diagnostic_route_sha256": diagnostic["route_sha256"],
        "executor_source_bundle_sha256": "c" * 64,
    }
    release = build_supervised_release(
        route,
        bindings=bindings,
        issued_at_unix_s=100.0,
        expires_at_unix_s=200.0,
    )
    return diagnostic, route, release


def _rehash(value, field):
    updated = copy.deepcopy(value)
    updated.pop(field, None)
    updated[field] = canonical_sha256(updated)
    return updated


def test_positive_route_is_small_effective_and_outbound_only():
    diagnostic, route, release = _bundle()
    assert diagnostic["diagnostic_only"] is True
    assert validate_supervised_executable_route(route) == route["route_sha256"]
    assert 1.0 < route["minimum_effective_increment_deg"]
    assert route["maximum_increment_deg"] <= 1.10
    assert all(item["worst_case_next_gap_deg"] < 2.0 for item in route["interval_convergence_bounds"])
    assert validate_supervised_release(release, route, now_unix_s=150.0) == release["release_sha256"]


@pytest.mark.parametrize("field,value", [
    ("press", True),
    ("automatic_retry", True),
    ("motion_authorized", True),
    ("waypoint_tolerance_rad", 0.02),
    ("endpoint_scope", "BUTTON1_70MM"),
])
def test_route_manual_field_flips_are_rejected(field, value):
    _, route, _ = _bundle()
    route[field] = value
    route = _rehash(route, "route_sha256")
    with pytest.raises(SupervisedReleaseError):
        validate_supervised_executable_route(route)


def test_increment_over_1p10_and_ineffective_interval_are_rejected():
    _, route, _ = _bundle()
    route["setpoints_rad"][0][0] += math.radians(0.2)
    route = _rehash(route, "route_sha256")
    with pytest.raises(SupervisedReleaseError):
        validate_supervised_executable_route(route)
    _, route, _ = _bundle()
    route["setpoints_rad"].insert(0, list(route["start_positions_rad"]))
    route = _rehash(route, "route_sha256")
    with pytest.raises(SupervisedReleaseError):
        validate_supervised_executable_route(route)


def test_release_expiry_hash_route_and_clearance_binding_fail_closed():
    _, route, release = _bundle()
    with pytest.raises(SupervisedReleaseError, match="not current"):
        validate_supervised_release(release, route, now_unix_s=201.0)
    drifted = copy.deepcopy(release)
    drifted["bindings"]["continuous_clearance_file_sha256"] = "d" * 64
    with pytest.raises(SupervisedReleaseError, match="release_sha256 mismatch"):
        validate_supervised_release(drifted, route, now_unix_s=150.0)
    other_route = copy.deepcopy(route)
    other_route["route_identity"] = "other"
    other_route = _rehash(other_route, "route_sha256")
    with pytest.raises(SupervisedReleaseError, match="route hash mismatch"):
        validate_supervised_release(release, other_route, now_unix_s=150.0)


def test_supervised_core_aborts_first_no_progress_and_emits_no_second_point():
    _, route_raw, _ = _bundle()
    route = SealedOutboundRoute.from_mapping(route_raw, route_raw["route_sha256"])
    first = route.setpoints_rad[0]
    controller = FakeController(
        latest_feedback=Feedback(10.05, "CONVERGED_DISARMED", measured_positions_rad=tuple(Q0)),
        latest_status_ack=StatusTargetAck(10.05),
        post_emit_status_acks=[StatusTargetAck(10.10, first)],
        post_emit_feedback=[Feedback(
            10.10,
            "CONVERGED_DISARMED",
            acknowledged_setpoint_sha256=setpoint_sha256(first),
            measured_positions_rad=tuple(Q0),
        )],
    )
    ticket = ExecutionTicket(
        route.route_sha256,
        route.bindings,
        10.0,
        12.0,
        "MOVE_TO_PRECONTACT_ONLY",
    )
    core = T102OutboundExecutorCore(ExecutorConfig(enabled=True, final_joint_tolerance_rad=0.005))
    with pytest.raises(T102ExecutorError, match="outside tolerance"):
        core.execute(route, ticket, controller, now_monotonic=10.2)
    assert len(controller.emitted) == 1
    assert controller.released == ["t102-outbound-only-core"]
