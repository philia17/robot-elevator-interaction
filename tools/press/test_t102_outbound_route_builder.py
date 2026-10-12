"""No-hardware tests for the diagnostic direct T102 chord builder."""

from __future__ import annotations

import copy
import math

import pytest

from press.t102_outbound_route_builder import (
    ENCODER_STEP_RAD,
    EXPECTED_DELTA_CAPS_DEG,
    OBSERVED_FIELD_FAILURE_JOINT,
    OBSERVED_FIELD_FAILURE_RESIDUAL_RAD,
    T102RouteBlocked,
    build_t102_outbound_diagnostic_route,
    canonical_sha256,
    classify_t102_terminal_residual,
    load_declared_t102_driver_tolerance_deg,
    validate_t102_outbound_diagnostic_route,
)


# Current Button 1 position-only static-IK candidate.  It is used only to
# verify deterministic offline segmentation; it is not an execution release.
START = [0.046019424, -1.188835111, 2.328582836, 0.46786414, -1.515573018, 0.0]
CANDIDATE = [0.17195432546434475, -0.7592116272033396, 2.0769242265081016, 0.03485803655203858, -1.497108414621783, 0.0]
BINDINGS = {
    "m0_target_sha256": "a" * 64,
    "fn_sha256": "b" * 64,
    "r6_sha256": "c" * 64,
    "provenance_sha256": "d" * 64,
}


def test_current_static_candidate_has_five_conservative_segments_and_respects_caps():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    # Half-profile caps turn the former three-segment chord into five smaller,
    # still simultaneous T102 chord segments.
    assert route["max_profile_delta_fraction"] == 0.5
    assert len(route["setpoints_rad"]) == 5
    assert len([route["start_positions_rad"], *route["setpoints_rad"]]) == 6
    previous = START
    for point in route["setpoints_rad"]:
        assert all(math.isfinite(value) for value in point)
        assert all(
            abs(value - prior) <= 0.5 * math.radians(cap) + 1e-12
            for value, prior, cap in zip(point, previous, EXPECTED_DELTA_CAPS_DEG)
        )
        previous = point
    assert previous == CANDIDATE


def test_five_waypoint_route_has_exact_endpoint_hash_and_cap_bindings():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    # Exact route bindings: start, endpoint, frozen hashes, and the sealed
    # self-hash over every other field.
    assert route["start_positions_rad"] == list(START)
    assert route["setpoints_rad"][-1] == list(CANDIDATE)
    assert route["setpoints_rad"][-1] == route["static_ik_candidate_positions_rad"]
    assert route["frozen_bindings"] == BINDINGS
    unsigned = {key: value for key, value in route.items() if key != "route_sha256"}
    assert route["route_sha256"] == canonical_sha256(unsigned)
    assert len(route["setpoints_rad"]) == 5
    # Per-segment cap binding: every segment is inside the half-profile caps.
    previous = START
    for point in route["setpoints_rad"]:
        assert all(
            abs(value - prior) <= 0.5 * math.radians(cap) + 1e-12
            for value, prior, cap in zip(point, previous, EXPECTED_DELTA_CAPS_DEG)
        )
        previous = point
    assert route["segment_delta_caps_deg"] == [0.5 * cap for cap in EXPECTED_DELTA_CAPS_DEG]
    assert route["driver_delta_caps_deg"] == list(EXPECTED_DELTA_CAPS_DEG)


def test_full_profile_fraction_is_explicit_legacy_option_but_not_default():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
        max_profile_delta_fraction=1.0,
    )
    assert len(route["setpoints_rad"]) == 3
    assert route["segment_delta_caps_deg"] == list(EXPECTED_DELTA_CAPS_DEG)


def test_route_embeds_encoder_settle_diagnostics_and_flags_first_segment_risk():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    assert abs(route["encoder_step_rad"] - ENCODER_STEP_RAD) < 1e-15
    assert route["driver_tolerance_deg"] == load_declared_t102_driver_tolerance_deg()
    assert route["driver_soft_converge_band_deg"] == 2.0 * route["driver_tolerance_deg"]
    segment_steps = route["segment_deltas_encoder_steps"]
    assert len(segment_steps) == len(route["setpoints_rad"])
    assert all(len(segment) == 6 for segment in segment_steps)
    previous = START
    for point, steps in zip(route["setpoints_rad"], segment_steps):
        for actual, expected, step in zip(point, previous, steps):
            assert abs(abs(actual - expected) / ENCODER_STEP_RAD - step) < 1e-9
        previous = point
    first_max_deg = max(segment_steps[0]) * math.degrees(ENCODER_STEP_RAD)
    assert abs(route["first_segment_max_delta_deg"] - first_max_deg) < 1e-9
    assert abs(route["first_segment_max_delta_rad"] - math.radians(first_max_deg)) < 1e-12
    # The reviewed current candidate's first simultaneous segment moves the
    # shoulder/wrist ~4.9 deg in one T102 command, far beyond the deployed
    # profile's 2.0 deg soft-converge band; the artifact must say so.
    assert route["first_segment_max_delta_deg"] > 4.9
    assert route["first_segment_exceeds_soft_converge_band"] is True
    validate_t102_outbound_diagnostic_route(route)


def test_terminal_residual_policy_classification_is_deterministic():
    observed = classify_t102_terminal_residual(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD, 1.0)
    assert observed["within_soft_converge_band"] is False
    assert observed["driver_terminal_policy"] == "DRIVER_FAULTS_AFTER_SINGLE_REISSUE"
    assert observed["soft_converge_band_deg"] == 2.0
    assert abs(observed["residual_deg"] - math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD)) < 1e-9
    assert observed["residual_deg"] > 2.0
    stable = classify_t102_terminal_residual(math.radians(1.9), 1.0)
    assert stable["within_soft_converge_band"] is True
    assert stable["driver_terminal_policy"] == "SOFT_CONVERGE_ELIGIBLE_AFTER_SINGLE_REISSUE"
    assert stable["residual_deg"] <= 2.0
    for value in (float("nan"), -0.1):
        with pytest.raises(T102RouteBlocked, match="terminal residual"):
            classify_t102_terminal_residual(value, 1.0)
    converged = classify_t102_terminal_residual(0.0, 1.0)
    assert converged["within_soft_converge_band"] is True


def test_route_documents_physical_blocker_without_claiming_segmentation_fixes_undertravel():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    assert route["physical_convergence_guaranteed"] is False
    assert route["segmentation_does_not_fix_physical_undertravel"] is True
    assert isinstance(route["execution_blocked_reason"], str) and route["execution_blocked_reason"].strip()
    failure = route["observed_field_failure_reference"]
    assert failure["joint"] == OBSERVED_FIELD_FAILURE_JOINT
    assert failure["max_residual_rad"] == OBSERVED_FIELD_FAILURE_RESIDUAL_RAD
    assert abs(failure["max_residual_deg"] - math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD)) < 1e-9
    assert failure["soft_converge_band_deg"] == route["driver_soft_converge_band_deg"]
    assert failure["residual_exceeded_soft_converge_band"] is True
    assert failure["driver_outcome"] == "FAULT_LATCHED_AFTER_ONE_REISSUE"
    policy = route["terminal_residual_policy"]
    assert policy["driver_tolerance_deg"] == route["driver_tolerance_deg"]
    assert policy["single_stalled_reissue"] is True
    assert policy["executor_keeps_own_final_joint_gate_rad"] == 0.02
    validate_t102_outbound_diagnostic_route(route)


def test_tampering_physical_blocker_diagnostics_fails_closed():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    tampered = copy.deepcopy(route)
    tampered["physical_convergence_guaranteed"] = True
    with pytest.raises(T102RouteBlocked, match="physical convergence"):
        validate_t102_outbound_diagnostic_route(tampered)
    tampered = copy.deepcopy(route)
    tampered["segmentation_does_not_fix_physical_undertravel"] = False
    with pytest.raises(T102RouteBlocked, match="segmentation-only claim"):
        validate_t102_outbound_diagnostic_route(tampered)
    tampered = copy.deepcopy(route)
    tampered["observed_field_failure_reference"]["residual_exceeded_soft_converge_band"] = False
    with pytest.raises(T102RouteBlocked, match="soft-band flag"):
        validate_t102_outbound_diagnostic_route(tampered)


def test_validate_rejects_inconsistent_settle_diagnostics():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    tampered = copy.deepcopy(route)
    tampered["first_segment_exceeds_soft_converge_band"] = not tampered["first_segment_exceeds_soft_converge_band"]
    with pytest.raises(T102RouteBlocked, match="soft-band flag"):
        validate_t102_outbound_diagnostic_route(tampered)
    tampered = copy.deepcopy(route)
    tampered["segment_deltas_encoder_steps"][0][0] += 0.5
    with pytest.raises(T102RouteBlocked, match="segment deltas"):
        validate_t102_outbound_diagnostic_route(tampered)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_conservative_fraction_fails_closed(fraction):
    with pytest.raises(T102RouteBlocked, match="max_profile_delta_fraction"):
        build_t102_outbound_diagnostic_route(
            verified_start_positions_rad=START,
            verified_static_ik_candidate_rad=CANDIDATE,
            frozen_bindings=BINDINGS,
            max_profile_delta_fraction=fraction,
        )


def test_route_is_outbound_only_and_permanently_diagnostic():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    assert route["phases"] == ["OUTBOUND_TO_PRECONTACT"]
    assert route["joint_path"] == "SIMULTANEOUS_T102_JOINT_SPACE_CHORD"
    for name in ("press", "retract", "return", "automatic_retry", "execution_ready", "motion_authorized", "trajectory_generated", "command_generated", "adapter_consumable"):
        assert route[name] is False
    assert route["command_interface"] == "NOT_IMPLEMENTED"
    validate_t102_outbound_diagnostic_route(route)


def test_tampering_or_bad_binding_fails_closed():
    route = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=START,
        verified_static_ik_candidate_rad=CANDIDATE,
        frozen_bindings=BINDINGS,
    )
    tampered = copy.deepcopy(route)
    tampered["motion_authorized"] = True
    with pytest.raises(T102RouteBlocked, match="motion_authorized"):
        validate_t102_outbound_diagnostic_route(tampered)
    bad = dict(BINDINGS)
    bad["fn_sha256"] = "wrong"
    with pytest.raises(T102RouteBlocked, match="fn_sha256"):
        build_t102_outbound_diagnostic_route(
            verified_start_positions_rad=START,
            verified_static_ik_candidate_rad=CANDIDATE,
            frozen_bindings=bad,
        )
