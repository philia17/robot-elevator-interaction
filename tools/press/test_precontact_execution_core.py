"""Pure tests for the installed outbound-only pre-contact admission core."""

from __future__ import annotations

import copy
import json

import pytest

from press.precontact_execution_core import (
    AdmissionGates,
    CompletionEvidence,
    FakeExecutionResources,
    MAX_TICKET_TTL_S,
    PrecontactError,
    PrecontactExecutionCore,
    PrecontactState,
    REQUIRED_CONFIRMATION_TEXT,
    TerminalCode,
    canonical_json,
    canonical_route_artifact_sha256,
    canonical_setpoint_sha256,
    canonical_sha256,
    parse_outbound_route_artifact,
)


def _route_content(route):
    geometry = route["geometry"]
    return {
        "manifest_sha256": route["release_bindings"]["manifest_sha256"],
        "phase": route["phase"],
        "target_binding": route["target_binding"],
        "geometry": {
            "A_base_m": geometry["A_base_m"],
            "N_out_base": geometry["N_out_base"],
            "P_pre_base_m": geometry["P_pre_base_m"],
        },
        "start_quantized_rad": route["quantization"]["start_quantized_rad"],
        "emitted_setpoints_rad": route["quantization"]["emitted_setpoints_rad"],
        "precontact_contact_point_base_m": geometry["precontact_contact_point_base_m"],
        "precontact_distance_m": geometry["exact_precontact_distance_m"],
        "contact_rotation_xyzw": geometry["contact_rotation_xyzw"],
        "quantized_endpoint_evidence": route["quantized_endpoint_evidence"],
        "collision_evidence": route["clearance_evidence"],
    }


def _reseal_route(route):
    route = copy.deepcopy(route)
    route["route_hash"] = canonical_sha256(_route_content(route))
    route["route_id"] = f"precontact-outbound-{route['route_hash'][:16]}"
    route["canonical_payload_sha256"] = canonical_route_artifact_sha256(route)
    return route


def _route():
    target_binding = {
        "target_id": "button-1",
        "detection_id": "det-1",
        "source_stamp_ns": 123,
        "source_frame_id": "camera_optical_frame",
        "m0_frozen_artifact_sha256": "c" * 64,
        "calibration_release_id": "r6-release",
        "r6_bundle_artifact_sha256": "a" * 64,
    }
    endpoint = {
        "status": "validated",
        "checked_positions_rad": [0.1, -0.1, 0.0, 0.0, 0.0, 0.0],
    }
    path = [
        [0.0] * 6,
        [0.05, -0.05, 0.0, 0.0, 0.0, 0.0],
        endpoint["checked_positions_rad"],
    ]
    clearance = {
        "status": "clear",
        "checked_waypoint_count": len(path),
        "segments_checked": len(path) - 1,
        "continuous_corridor_status": "clear",
        "swept_segments_status": "clear",
        "quantized_path_setpoints_rad": path,
    }
    route = {
        "schema": "precontact_outbound_route_v1",
        "status": "offline_released_route",
        "phase": "OUTBOUND_TO_PRECONTACT",
        "phases": ["OUTBOUND_TO_PRECONTACT"],
        "executable": True,
        "adapter_consumable": True,
        "required_execution_interface": "SAFE_ADAPTER_ACTION_SERVER",
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "no_contact": True,
        "automatic_retry": False,
        "route_hash": "0" * 64,
        "route_id": "pending",
        "target_binding": target_binding,
        "release_bindings": {
            "manifest_sha256": "b" * 64,
            "m0_frozen": {"release_id": "m0-release", "artifact_sha256": "c" * 64},
            "r6_bundle": {"release_id": "r6-release", "artifact_sha256": "a" * 64},
        },
        "geometry": {
            "A_base_m": [0.3, 0.1, 0.2],
            "N_out_base": [0.0, 0.0, 1.0],
            "P_pre_base_m": [0.3, 0.1, 0.25],
            "target_surface_base_m": [0.3, 0.1, 0.2],
            "normal_out_base": [0.0, 0.0, 1.0],
            "precontact_contact_point_base_m": [0.3, 0.1, 0.25],
            "exact_precontact_distance_m": 0.050,
            "contact_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "quantization": {
            "start_quantized_rad": path[0],
            "emitted_setpoints_rad": path[1:],
        },
        "quantized_endpoint_evidence": endpoint,
        "clearance_evidence": clearance,
    }
    return _reseal_route(route)


def _release(*, enabled=True, ready=True, route=None):
    artifact = _route() if route is None else route
    return {
        "schema": "precontact_execution_release_v2",
        "release_id": "test-release",
        "execution_enabled": enabled,
        "execution_ready": ready,
        "command_mode": "T101_SINGLE_JOINT_ONLY",
        "outbound_release_manifest_sha256": artifact["release_bindings"]["manifest_sha256"],
        "outbound_route_artifact_sha256": canonical_sha256(artifact),
        "outbound_route_artifact": artifact,
    }


def _snapshot(release, *, frozen_at=10.0, **changes):
    route = release["outbound_route_artifact"]
    binding = route["target_binding"]
    geometry = route["geometry"]
    payload = {
        "schema": "button1_frozen_target_v2",
        "label": "1",
        "target_id": binding["target_id"],
        "detection_id": binding["detection_id"],
        "source_stamp_ns": binding["source_stamp_ns"],
        "source_frame_id": binding["source_frame_id"],
        "frozen_monotonic": frozen_at,
        "normal_direction_status": "VALID_OUTWARD",
        "m0_frozen_artifact_sha256": binding["m0_frozen_artifact_sha256"],
        "A_base_m": geometry["A_base_m"],
        "N_out_base": geometry["N_out_base"],
        "P_pre_base_m": geometry["P_pre_base_m"],
        "outbound_release_manifest_sha256": route["release_bindings"]["manifest_sha256"],
        "execution_release_manifest_sha256": canonical_sha256(release),
    }
    payload.update(changes)
    return canonical_json(payload), canonical_sha256(payload)


def _gates(**changes):
    values = {field: True for field in AdmissionGates.__dataclass_fields__}
    values.update(changes)
    return AdmissionGates(**values)


def _prepare_and_reserve(core, release, *, now=10.1):
    snapshot, snapshot_hash = _snapshot(release)
    prepared = core.prepare(snapshot, snapshot_hash, _gates(), now_monotonic=now)
    assert prepared.accepted, prepared.reason
    accepted, reason = core.reserve_goal(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        now_monotonic=now + 0.05,
    )
    assert accepted, reason
    return prepared, snapshot_hash


def _completion(core, release, **changes):
    route = release["outbound_route_artifact"]
    values = {
        "feedback_snapshot_sha256": "f" * 64,
        "route_artifact_sha256": canonical_sha256(route),
        "final_setpoint_sha256": canonical_setpoint_sha256(
            route["quantization"]["emitted_setpoints_rad"][-1]
        ),
        "feedback_fresh": True,
        "feedback_aligned": True,
        "fault_clear": True,
        "armed": False,
        "safe_driver_state": "CONVERGED_DISARMED",
    }
    values.update(changes)
    return CompletionEvidence(**values)


def test_canonical_snapshot_and_complete_route_artifact_hash_chain():
    route = _route()
    parsed = parse_outbound_route_artifact(route, canonical_sha256(route))
    assert parsed.p_pre_base_m == pytest.approx((0.3, 0.1, 0.25))
    assert parsed.target_id == "button-1"
    assert parsed.detection_id == "det-1"
    assert parsed.m0_frozen_artifact_sha256 == "c" * 64
    assert parsed.final_setpoint_sha256 == canonical_setpoint_sha256([0.1, -0.1, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("adapter_consumable", False, "adapter_consumable"),
        ("automatic_retry", True, "automatic_retry"),
        ("contact_authorized", True, "contact_authorized"),
        ("phases", ["OUTBOUND_TO_PRECONTACT", "PRESS"], "forbidden phase"),
    ],
)
def test_route_safety_semantics_cannot_be_resealed_into_acceptance(field, value, reason):
    route = _route()
    route[field] = value
    route = _reseal_route(route)
    with pytest.raises(PrecontactError, match=reason):
        parse_outbound_route_artifact(route, canonical_sha256(route))


def test_route_geometry_tamper_rejected_even_after_all_hashes_are_resealed():
    route = _route()
    route["geometry"]["P_pre_base_m"] = [0.3, 0.1, 0.251]
    route["geometry"]["precontact_contact_point_base_m"] = [0.3, 0.1, 0.251]
    route = _reseal_route(route)
    with pytest.raises(PrecontactError, match="P_pre=A"):
        parse_outbound_route_artifact(route, canonical_sha256(route))


def test_tampered_outbound_manifest_resealed_in_route_and_outer_release_rejects_frozen_target():
    original_release = _release()
    snapshot, snapshot_hash = _snapshot(original_release)
    tampered_route = copy.deepcopy(original_release["outbound_route_artifact"])
    tampered_route["release_bindings"]["manifest_sha256"] = "e" * 64
    tampered_route = _reseal_route(tampered_route)
    tampered_release = _release(route=tampered_route)
    result = PrecontactExecutionCore(tampered_release).prepare(
        snapshot,
        snapshot_hash,
        _gates(),
        now_monotonic=10.1,
    )
    assert result.accepted is False
    assert "outbound_release_manifest_hash" in result.reason


def test_prepare_binds_release_route_m0_and_exact_50mm_target_without_resources():
    release = _release()
    executor = FakeExecutionResources()
    core = PrecontactExecutionCore(release, executor=executor)
    snapshot, snapshot_hash = _snapshot(release)
    result = core.prepare(snapshot, snapshot_hash, _gates(), now_monotonic=10.1)
    assert result.accepted is True
    assert result.ttl_ms <= int(MAX_TICKET_TTL_S * 1000)
    assert result.release_manifest_sha256 == canonical_sha256(release)
    assert result.route_artifact_sha256 == canonical_sha256(release["outbound_route_artifact"])
    assert result.m0_frozen_artifact_sha256 == "c" * 64
    assert core.target.precontact_fingertip_base_m == pytest.approx((0.3, 0.1, 0.25))
    assert executor.publisher_count == executor.arm_client_count == 0


def test_every_runtime_gate_is_independently_fail_closed():
    release = _release()
    snapshot, snapshot_hash = _snapshot(release)
    for field in AdmissionGates.__dataclass_fields__:
        core = PrecontactExecutionCore(release)
        result = core.prepare(
            snapshot,
            snapshot_hash,
            _gates(**{field: False}),
            now_monotonic=10.1,
        )
        assert result.accepted is False, field
        assert field in result.reason
        assert core.state is PrecontactState.INERT


def test_default_or_not_ready_release_stays_reject_by_default():
    default = PrecontactExecutionCore()
    assert default.release.execution_enabled is False
    assert default.release.execution_ready is False
    release = _release(enabled=False, ready=False)
    snapshot, snapshot_hash = _snapshot(release)
    result = PrecontactExecutionCore(release).prepare(
        snapshot, snapshot_hash, _gates(), now_monotonic=10.1
    )
    assert result.accepted is False
    assert "execution_enabled" in result.reason and "execution_ready" in result.reason


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"target_id": "button-other"}, "target_id"),
        ({"detection_id": "det-other"}, "detection_id"),
        ({"source_stamp_ns": 124}, "source_stamp_ns"),
        ({"source_frame_id": "other_frame"}, "source_frame_id"),
        ({"m0_frozen_artifact_sha256": "d" * 64}, "m0_frozen_artifact_hash"),
        ({"outbound_release_manifest_sha256": "d" * 64}, "outbound_release_manifest_hash"),
        ({"execution_release_manifest_sha256": "d" * 64}, "execution_release_manifest_hash"),
        ({"A_base_m": [0.301, 0.1, 0.2], "P_pre_base_m": [0.301, 0.1, 0.25]}, "A_base_m"),
        ({"N_out_base": [0.0, 1.0, 0.0], "P_pre_base_m": [0.3, 0.15, 0.2]}, "N_out_base"),
    ],
)
def test_frozen_snapshot_identity_must_equal_pinned_route_binding(change, reason):
    release = _release()
    snapshot, snapshot_hash = _snapshot(release, **change)
    result = PrecontactExecutionCore(release).prepare(
        snapshot, snapshot_hash, _gates(), now_monotonic=10.1
    )
    assert result.accepted is False
    assert reason in result.reason


def test_invalid_goals_are_rejected_without_latching_or_consuming_ticket():
    release = _release()
    core = PrecontactExecutionCore(release)
    snapshot, snapshot_hash = _snapshot(release)
    prepared = core.prepare(snapshot, snapshot_hash, _gates(), now_monotonic=10.1)
    cases = (
        ("bad", snapshot_hash, REQUIRED_CONFIRMATION_TEXT, 10.2),
        (prepared.authorization_id, "d" * 64, REQUIRED_CONFIRMATION_TEXT, 10.2),
        (prepared.authorization_id, snapshot_hash, "wrong", 10.2),
        (prepared.authorization_id, snapshot_hash, REQUIRED_CONFIRMATION_TEXT, 12.2),
    )
    for authorization_id, target_hash, confirmation, now in cases:
        accepted, _ = core.reserve_goal(
            authorization_id, target_hash, confirmation, now_monotonic=now
        )
        assert accepted is False
        assert core.state is PrecontactState.PREPARED_NO_MOTION
        assert core.ticket.consumed is False and core.ticket.reserved is False


def test_ticket_is_one_use_and_second_reservation_is_rejected():
    release = _release()
    core = PrecontactExecutionCore(release, executor=FakeExecutionResources())
    prepared, snapshot_hash = _prepare_and_reserve(core, release)
    accepted, reason = core.reserve_goal(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        now_monotonic=10.16,
    )
    assert accepted is False and "reserved" in reason


def test_ticket_is_expired_at_exact_deadline_for_goal_reservation():
    release = _release()
    core = PrecontactExecutionCore(release)
    snapshot, snapshot_hash = _snapshot(release)
    prepared = core.prepare(snapshot, snapshot_hash, _gates(), now_monotonic=10.1)
    accepted, reason = core.reserve_goal(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        now_monotonic=core.ticket.expires_monotonic,
    )
    assert accepted is False
    assert "expired" in reason


def test_ticket_is_expired_at_exact_deadline_for_admission():
    release = _release()
    core = PrecontactExecutionCore(release, executor=FakeExecutionResources())
    prepared, snapshot_hash = _prepare_and_reserve(core, release)
    result = core.admit(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        _gates(),
        now_monotonic=core.ticket.expires_monotonic,
    )
    assert result.success is False
    assert "expired" in result.reason
    assert core.state is PrecontactState.ABORT_LATCHED


def test_fake_resources_zero_one_zero_require_strong_completion_evidence():
    release = _release()
    executor = FakeExecutionResources()
    core = PrecontactExecutionCore(release, executor=executor)
    prepared, snapshot_hash = _prepare_and_reserve(core, release)
    admitted = core.admit(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        _gates(),
        now_monotonic=10.2,
    )
    assert admitted.code is TerminalCode.EXECUTING
    assert executor.publisher_count == executor.arm_client_count == 1
    complete = core.complete_disarmed(_completion(core, release))
    assert complete.success is True
    assert core.state is PrecontactState.COMPLETE_DISARMED
    assert executor.publisher_count == executor.arm_client_count == 0
    assert executor.events == ["create", "destroy"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"feedback_fresh": False}, "feedback_fresh"),
        ({"feedback_aligned": False}, "feedback_aligned"),
        ({"fault_clear": False}, "fault_clear"),
        ({"armed": True}, "armed_false"),
        ({"safe_driver_state": "DISARMED"}, "driver_state"),
        ({"route_artifact_sha256": "d" * 64}, "route_artifact"),
        ({"final_setpoint_sha256": "d" * 64}, "final_setpoint"),
    ],
)
def test_incomplete_terminal_evidence_aborts_and_destroys(change, reason):
    release = _release()
    executor = FakeExecutionResources()
    core = PrecontactExecutionCore(release, executor=executor)
    prepared, snapshot_hash = _prepare_and_reserve(core, release)
    core.admit(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        _gates(),
        now_monotonic=10.2,
    )
    result = core.complete_disarmed(_completion(core, release, **change))
    assert result.success is False and reason in result.reason
    assert core.state is PrecontactState.ABORT_LATCHED
    assert executor.publisher_count == executor.arm_client_count == 0


def test_resource_destroy_exception_is_caught_and_latched():
    release = _release()
    executor = FakeExecutionResources(fail_destroy=True)
    core = PrecontactExecutionCore(release, executor=executor)
    prepared, snapshot_hash = _prepare_and_reserve(core, release)
    core.admit(
        prepared.authorization_id,
        snapshot_hash,
        REQUIRED_CONFIRMATION_TEXT,
        _gates(),
        now_monotonic=10.2,
    )
    result = core.complete_disarmed(_completion(core, release))
    assert result.success is False
    assert "destroy failed" in result.reason
    assert core.state is PrecontactState.ABORT_LATCHED
    assert executor.publisher_count == executor.arm_client_count == 0


def test_abort_is_one_shot_latched_and_has_no_reset_api():
    core = PrecontactExecutionCore()
    result = core.abort("operator cancel")
    assert result.success is False
    assert core.state is PrecontactState.ABORT_LATCHED
    assert not hasattr(core, "reset_after_operator_review")
