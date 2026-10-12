"""Tests for the pure, inert-safe Live-to-Quick-Control lease protocol."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from press.precontact_qc_core import (
    ActionReceipt,
    DynamicSessionAssemblyError,
    LeasePhase,
    LiveIntent,
    PrecontactLeaseCore,
    PrepareReceipt,
    canonical_json,
    dynamic_plan_summary,
    target_stamp_ns_from_intent,
    validate_q0_after_target,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dynamic_plan(*, residual: float = 0.001, audit_residual: float | None = None) -> dict:
    audit_residual = residual if audit_residual is None else audit_residual
    return {
        "normal_proof": {"n_out": [1.0, 0.0, 0.0]},
        "session": {
            "kinematic_model_hash": "1" * 64,
            "collision_proxy_hash": "2" * 64,
            "tool_model_hash": "3" * 64,
            "template_hash": "4" * 64,
            "calibration_hash": "5" * 64,
        },
        "sampled_constraints": {"counts": {"samples": 3}},
        "post_quantization": {
            "position_residual_m": residual,
            "waypoint_audit": [{
                "position_residual_m": audit_residual,
                "condition": 10.0,
                "manipulability": 0.1,
            }],
        },
        "waypoints_rad": [[0.1, 0.0, 0.0, 0.0, 0.2, 0.3]],
        "p_button_base_m": [0.03, 0.0, 0.0],
        "p_stage_base_m": [0.15, 0.0, 0.0],
        "p_pre_base_m": [0.10, 0.0, 0.0],
    }


def test_dynamic_plan_summary_binds_final_waypoint_and_post_quantization_residual():
    plan = _dynamic_plan(audit_residual=0.0002)
    summary = dynamic_plan_summary(plan, _sha(canonical_json(plan)))
    assert summary["planned_terminal_q_rad"] == pytest.approx(plan["waypoints_rad"][-1])
    assert summary["planned_terminal_position_residual_m"] == pytest.approx(0.001)


def test_terminal_core_adopts_only_its_exact_inert_ticket_for_clear():
    core = _prepared_core()
    core.fail_local("Prepare callback failed: summary parse")
    assert core.phase is LeasePhase.TERMINAL

    wrong_nonce = core.adopt_inert_ticket_for_clear(
        executor_instance_nonce="executor-2", plan_hash="a" * 64,
    )
    assert not wrong_nonce.accepted
    assert core.phase is LeasePhase.TERMINAL

    wrong = core.adopt_inert_ticket_for_clear(
        executor_instance_nonce="executor-1", plan_hash="b" * 64,
    )
    assert not wrong.accepted
    assert core.phase is LeasePhase.TERMINAL

    adopted = core.adopt_inert_ticket_for_clear(
        executor_instance_nonce="executor-1", plan_hash="a" * 64,
    )
    assert adopted.accepted
    assert adopted.command == "DISCARD_PREPARED_PRECONTACT"
    assert core.phase is LeasePhase.AWAIT_MOVE_CONFIRM


def test_target_frozen_core_adopts_exact_ticket_for_discard_only():
    core = _frozen_core()
    target_hash = core.target_snapshot_sha256
    adopted = core.adopt_inert_ticket_for_clear(
        executor_instance_nonce="executor-orphan",
        plan_hash="f" * 64,
    )
    assert adopted.accepted
    assert adopted.command == "DISCARD_PREPARED_PRECONTACT"
    assert core.phase is LeasePhase.AWAIT_MOVE_CONFIRM
    assert core.executor_instance_nonce == "executor-orphan"
    assert core.route_artifact_sha256 == "f" * 64
    assert core.discard_only is True
    assert core.authorization_id == ""
    assert core.session_id == ""

    # Retaining the local display target cannot turn an adopted orphan into a
    # Move/validation/cart capability while the typed discard is pending.
    assert core.target_snapshot_sha256 == target_hash
    blocked_move = core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="blocked-move"),
        now_unix_s=100.0,
        now_monotonic=5.0,
    )
    blocked_validation = core.begin_handoff_validation(
        _intent(
            "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION",
            intent_id="blocked-validation",
        ),
        now_unix_s=100.0,
        now_monotonic=5.0,
    )
    blocked_cart = core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="blocked-cart"),
        now_unix_s=100.0,
    )
    for decision in (blocked_move, blocked_validation, blocked_cart):
        assert not decision.accepted
        assert "discard only" in decision.reason
        assert decision.phase is LeasePhase.AWAIT_MOVE_CONFIRM


@pytest.mark.parametrize(
    "phase", (LeasePhase.PREPARE_IN_FLIGHT, LeasePhase.ACTION_IN_FLIGHT),
)
def test_active_core_never_adopts_ticket_for_discard(phase):
    core = PrecontactLeaseCore()
    core.phase = phase
    decision = core.adopt_inert_ticket_for_clear(
        executor_instance_nonce="executor",
        plan_hash="f" * 64,
    )
    assert not decision.accepted
    assert decision.phase is phase


@pytest.mark.parametrize("residual", [-0.1, 0.0025000001])
def test_dynamic_plan_summary_rejects_invalid_terminal_residual(residual):
    plan = _dynamic_plan(residual=residual)
    with pytest.raises(DynamicSessionAssemblyError):
        dynamic_plan_summary(plan, _sha(canonical_json(plan)))


def test_dynamic_plan_summary_requires_final_waypoint():
    plan = _dynamic_plan()
    plan.pop("waypoints_rad")
    with pytest.raises(DynamicSessionAssemblyError):
        dynamic_plan_summary(plan, _sha(canonical_json(plan)))


def _healthy_q0_feedback(**changes):
    value = {
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "dry_run": False,
        "fault": "",
        "torque_enabled": [True] * 6,
        "voltage": 12.0,
        "positions_rad": [
            0.036815539, -1.191903072, 2.331650798,
            0.486271910, -1.515573018, 0.001533981,
        ],
        "feedback_stamp_ns": 1_050_000_000,
    }
    value.update(changes)
    return value


def test_fresh_stationary_healthy_nonstandby_q0_is_admitted():
    feedback = _healthy_q0_feedback()
    standard_standby = (
        0.04601942363656923, -1.188835110611372, 2.3285828360104035,
        0.4678641403051206, -1.5155730184310134, 0.001533980589793149,
    )
    assert max(abs(a-b) for a,b in zip(feedback["positions_rad"], standard_standby)) > math.radians(1.0)
    q0, stamp = validate_q0_after_target(
        feedback, feedback_age_s=0.01,
        target_stamp_ns=1_000_000_000, now_unix_ns=1_100_000_000,
    )
    assert q0 == pytest.approx(feedback["positions_rad"])
    assert stamp == feedback["feedback_stamp_ns"]


@pytest.mark.parametrize(
    ("changes", "age_s"),
    [
        ({"armed": True}, 0.01),
        ({"fault": "FAULT"}, 0.01),
        ({"dry_run": True}, 0.01),
        ({"torque_enabled": [True, True, False, True, True, True]}, 0.01),
        ({"state": "MOVING"}, 0.01),
        ({"voltage": 10.999}, 0.01),
        ({"voltage": 13.001}, 0.01),
        ({"voltage": float("nan")}, 0.01),
        ({"positions_rad": [0.0, 0.0, 0.0, 0.0, float("nan"), 0.0]}, 0.01),
        ({}, -0.001),
        ({}, 1.001),
    ],
)
def test_q0_health_fields_fail_closed_individually(changes, age_s):
    with pytest.raises(DynamicSessionAssemblyError):
        validate_q0_after_target(
            _healthy_q0_feedback(**changes), feedback_age_s=age_s,
            target_stamp_ns=1_000_000_000, now_unix_ns=1_100_000_000,
        )


@pytest.mark.parametrize("voltage", [11.0, 13.0])
def test_q0_voltage_boundaries_are_inclusive(voltage):
    q0, _ = validate_q0_after_target(
        _healthy_q0_feedback(voltage=voltage), feedback_age_s=1.0,
        target_stamp_ns=1_000_000_000, now_unix_ns=1_100_000_000,
    )
    assert len(q0) == 6


def _snapshot() -> tuple[str, str]:
    payload = {
        "schema": "m0_demo_freeze_v1",
        "trial_id": "button1-live-test",
        "mode": "live_freeze_and_prepare",
        "motion_authorized": False,
        "target_label": "BUTTON 1",
        "cart_pose_epoch": "cart-1",
        "frozen_target": {
            "target_label": "BUTTON 1",
            "detection_id": "det-1",
            "source_stamp_ns": 123,
            "frame_id": "camera_optical_frame",
            "cart_pose_epoch": "cart-1",
            "surface_geometry": {
                "estimate_valid": True,
                "provenance_complete": True,
            },
            "motion_authorized": False,
        },
    }
    canonical = canonical_json(payload)
    return canonical, _sha(canonical)


def _intent(
    action: str, *, intent_id: str = "prepare-1", prepared: str = "",
    cart_epoch: str = "cart-1", issued_at: float = 100.0,
) -> LiveIntent:
    snapshot, digest = _snapshot()
    return LiveIntent(
        intent_id, action, issued_at, snapshot, digest, prepared,
        cart_pose_epoch=cart_epoch, cart_confirmed_stamp_ns=90_000_000_000,
    )


def _selected_intent(label="2", **frozen_changes):
    payload = json.loads(_snapshot()[0])
    payload["target_label"] = label
    frozen = payload["frozen_target"]
    frozen.update({
        "target_label": label,
        "detection_id": "current-frame-id",
        "semantic_track_id": "selected-stable-track",
        "pressable": True,
        "target_selection": {
            "selection_id": "click-2",
            "label": label,
            "detection_id": "initial-frame-id",
            "semantic_track_id": "selected-stable-track",
            "after_stamp_ns": 122,
        },
    })
    frozen.update(frozen_changes)
    canonical = canonical_json(payload)
    return LiveIntent(**{
        **_intent("FREEZE_BUTTON1_TARGET", intent_id="selected-freeze").__dict__,
        "canonical_target_snapshot_json": canonical,
        "target_snapshot_sha256": _sha(canonical),
    })


@pytest.mark.parametrize("label", ["2", "open", "close", "up", "down", "b1", "g"])
def test_selected_target_freeze_preserves_identity_and_lease_gates(label):
    intent = _selected_intent(label)
    assert target_stamp_ns_from_intent(intent) == 123
    core = PrecontactLeaseCore()
    assert core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="selected-cart"), now_unix_s=100.0,
    ).accepted
    decision = core.freeze_button1_target(intent, now_unix_s=100.0)
    assert decision.accepted and decision.command == "AWAIT_PREPARE"
    assert core.target_snapshot_sha256 == intent.target_snapshot_sha256
    assert core.target_snapshot_sha256 != _snapshot()[1]


@pytest.mark.parametrize("changes", [
    {"semantic_track_id": "other-physical-button"},
    {"pressable": False},
    {"pressable": None},
    {"target_selection": None},
    {"source_stamp_ns": 122},
    {"target_label": "open"},
])
def test_selected_target_cannot_be_replaced_by_other_entity_or_old_frame(changes):
    with pytest.raises(ValueError):
        target_stamp_ns_from_intent(_selected_intent(**changes))


@pytest.mark.parametrize("label", ["unknown", "", "?", "unrecognized"])
def test_unknown_selected_target_never_passes_qc_freeze(label):
    with pytest.raises(ValueError):
        target_stamp_ns_from_intent(_selected_intent(label))


def _receipt(*, accepted: bool = True, **changes) -> PrepareReceipt:
    values = {
        "accepted": accepted,
        "authorization_id": "ticket-1",
        "ttl_ms": 10000,
        "execution_release_manifest_sha256": "b" * 64,
        "route_artifact_sha256": "c" * 64,
        "m0_frozen_artifact_sha256": _snapshot()[1],
        "session_id": "e" * 64,
        "executor_instance_nonce": "executor-1",
        "template_code": "SAMPLED_LIGHTWEIGHT_CONSTRAINT_PASS",
        "plan_hash": "a" * 64,
        "cart_pose_epoch": "cart-1",
        "kinematic_model_hash": "1" * 64,
        "collision_proxy_hash": "2" * 64,
        "tool_model_hash": "3" * 64,
        "template_hash": "4" * 64,
        "calibration_hash": "5" * 64,
        "publisher_count": 0,
        "lease_active": False,
        "arm_client_count": 0,
    }
    values.update(changes)
    return PrepareReceipt(**values)


def _frozen_core() -> PrecontactLeaseCore:
    core = PrecontactLeaseCore()
    decision = core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="cart-1"), now_unix_s=100.0,
    )
    assert decision.accepted and decision.command == "AWAIT_TARGET_FREEZE"
    decision = core.freeze_button1_target(
        _intent("FREEZE_BUTTON1_TARGET", intent_id="freeze-1"), now_unix_s=100.0,
    )
    assert decision.accepted and decision.command == "AWAIT_PREPARE"
    return core


def _prepared_core() -> PrecontactLeaseCore:
    core = _frozen_core()
    decision = core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0)
    assert decision.accepted and decision.command == "CALL_PREPARE"
    decision = core.record_prepare(_receipt(), now_monotonic=5.0)
    assert decision.accepted and core.phase is LeasePhase.AWAIT_MOVE_CONFIRM
    return core


def _success_action(**changes) -> ActionReceipt:
    values = {
        "success": True,
        "terminal_code": "DYNAMIC_PRECONTACT_STOPPED_DISARMED",
        "plan_hash": "a" * 64,
        "session_id": "e" * 64,
        "endpoint_max_joint_error_rad": 0.0,
        "terminal_joint_target_publishers": 0,
        "terminal_precontact_lease_active": False,
        "terminal_armed": False,
        "terminal_feedback_age_s": 0.01,
        "teardown_complete": True,
    }
    values.update(changes)
    return ActionReceipt(**values)


def _success_validation(**changes) -> ActionReceipt:
    values = {
        "success": True,
        "terminal_code": "PRECONTACT_HANDOFF_VALIDATED_NO_MOTION",
        "plan_hash": "a" * 64,
        "session_id": "e" * 64,
        "endpoint_max_joint_error_rad": float("inf"),
        "terminal_joint_target_publishers": 0,
        "terminal_precontact_lease_active": False,
        "terminal_armed": False,
        "terminal_feedback_age_s": float("inf"),
        "teardown_complete": True,
    }
    values.update(changes)
    return ActionReceipt(**values)


def test_prepare_requires_fresh_canonical_released_snapshot_and_rejects_replay():
    core = PrecontactLeaseCore()
    stale = core.begin_prepare(
        _intent("PREPARE_DYNAMIC_IK", issued_at=97.0), now_unix_s=100.0,
    )
    assert not stale.accepted and "stale" in stale.reason

    core = _frozen_core()
    invalid = _intent("PREPARE_DYNAMIC_IK")
    invalid = LiveIntent(**{**invalid.__dict__, "target_snapshot_sha256": "d" * 64})
    result = core.begin_prepare(invalid, now_unix_s=100.0)
    assert not result.accepted and "SHA-256 mismatch" in result.reason

    core = _frozen_core()
    first = core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0)
    assert first.accepted
    replay = core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0)
    assert not replay.accepted and core.phase is LeasePhase.PREPARE_IN_FLIGHT


def test_prepare_receipt_must_bind_m0_and_release_or_latch_terminal():
    core = _frozen_core()
    assert core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0).accepted
    result = core.record_prepare(_receipt(m0_frozen_artifact_sha256="d" * 64), now_monotonic=5.0)
    assert not result.accepted
    assert core.phase is LeasePhase.TERMINAL
    assert "M0" in result.reason


def test_confirm_requires_second_fresh_bound_intent_and_short_ticket():
    core = _prepared_core()
    bad = core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="other"),
        now_unix_s=100.1,
        now_monotonic=5.1,
    )
    assert not bad.accepted and core.phase is LeasePhase.TERMINAL

    core = _prepared_core()
    expired = core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1"),
        now_unix_s=100.1,
        now_monotonic=15.01,
    )
    assert not expired.accepted and "expired" in expired.reason

    core = _prepared_core()
    at_deadline = core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1"),
        now_unix_s=100.1,
        now_monotonic=15.0,
    )
    assert not at_deadline.accepted and "expired" in at_deadline.reason


def test_expire_latches_terminal_at_the_exact_ticket_deadline():
    core = _prepared_core()
    decision = core.expire(now_monotonic=15.0)
    assert decision is not None
    assert not decision.accepted and "expired" in decision.reason
    assert core.phase is LeasePhase.TERMINAL


def test_enabled_live_motion_ticket_allows_operator_inspection_for_sixty_seconds():
    core = _frozen_core()
    assert core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0).accepted
    decision = core.record_prepare(_receipt(ttl_ms=60_000), now_monotonic=5.0)
    assert decision.accepted
    assert core.ticket_remaining_s(now_monotonic=24.0) == pytest.approx(41.0)
    assert core.expire(now_monotonic=64.999) is None
    expired = core.expire(now_monotonic=65.0)
    assert expired is not None and not expired.accepted
    assert "expired" in expired.reason


def test_action_result_identity_mismatch_or_failure_is_terminal_without_retry():
    core = _prepared_core()
    assert core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1"),
        now_unix_s=100.1,
        now_monotonic=5.1,
    ).accepted
    mismatch = _success_action(plan_hash="d" * 64)
    result = core.record_action_result(mismatch)
    assert not result.accepted and core.phase is LeasePhase.TERMINAL
    assert core.begin_prepare(_intent("PREPARE_DYNAMIC_IK", intent_id="prepare-2"), now_unix_s=100.2).accepted is False


def test_matching_success_is_terminal_and_does_not_offer_retry_command():
    core = _prepared_core()
    assert core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1"),
        now_unix_s=100.1,
        now_monotonic=5.1,
    ).accepted
    result = core.record_action_result(_success_action())
    assert result.accepted and result.command == "NONE"
    assert core.phase is LeasePhase.TERMINAL


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"terminal_feedback_age_s": 0.500001}, "feedback is stale"),
        ({"endpoint_max_joint_error_rad": math.radians(0.5) + 1e-9}, "endpoint error"),
        ({"terminal_feedback_age_s": float("nan")}, "feedback is stale"),
        ({"endpoint_max_joint_error_rad": float("nan")}, "endpoint error"),
        ({"terminal_feedback_age_s": -0.1}, "feedback is stale"),
        ({"endpoint_max_joint_error_rad": -0.1}, "endpoint error"),
    ],
)
def test_action_terminal_metrics_match_hard_move_policy(changes, reason):
    core = _prepared_core()
    assert core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1"),
        now_unix_s=100.1, now_monotonic=5.1,
    ).accepted
    result = core.record_action_result(_success_action(**changes))
    assert not result.accepted and reason in result.reason


def test_no_motion_handoff_validation_binds_ticket_and_requires_zero_resources():
    core = _prepared_core()
    decision = core.begin_handoff_validation(
        _intent(
            "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION",
            intent_id="validate-1",
            prepared="prepare-1",
        ),
        now_unix_s=100.1,
        now_monotonic=5.1,
    )
    assert decision.accepted
    assert decision.command == "SEND_NO_MOTION_VALIDATION_GOAL"
    assert core.phase is LeasePhase.ACTION_IN_FLIGHT
    result = core.record_handoff_validation(_success_validation())
    assert result.accepted
    assert result.reason == "precontact handoff validated; no motion executed"
    assert core.phase is LeasePhase.TERMINAL

    for changes in (
        {"plan_hash": "d" * 64},
        {"session_id": "f" * 64},
        {"terminal_joint_target_publishers": 1},
        {"terminal_precontact_lease_active": True},
        {"terminal_armed": True},
        {"teardown_complete": False},
    ):
        core = _prepared_core()
        assert core.begin_handoff_validation(
            _intent(
                "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION",
                intent_id="validate-bad",
                prepared="prepare-1",
            ),
            now_unix_s=100.1,
            now_monotonic=5.1,
        ).accepted
        rejected = core.record_handoff_validation(_success_validation(**changes))
        assert not rejected.accepted
        assert core.phase is LeasePhase.TERMINAL


def test_no_motion_handoff_validation_rejects_stale_or_wrong_prepare_binding():
    core = _prepared_core()
    wrong = core.begin_handoff_validation(
        _intent(
            "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION",
            intent_id="validate-wrong",
            prepared="other-prepare",
        ),
        now_unix_s=100.1,
        now_monotonic=5.1,
    )
    assert not wrong.accepted
    assert core.phase is LeasePhase.TERMINAL

    core = _prepared_core()
    expired = core.begin_handoff_validation(
        _intent(
            "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION",
            intent_id="validate-expired",
            prepared="prepare-1",
        ),
        now_unix_s=100.1,
        now_monotonic=15.0,
    )
    assert not expired.accepted
    assert "expired" in expired.reason


def test_prepare_binds_session_executor_nonce_and_terminal_teardown_facts():
    core = _prepared_core()
    assert core.session_id == "e" * 64
    assert core.executor_instance_nonce == "executor-1"
    assert core.begin_confirm(
        _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1"),
        now_unix_s=100.1,
        now_monotonic=5.1,
    ).accepted
    result = core.record_action_result(
        _success_action(terminal_joint_target_publishers=1)
    )
    assert not result.accepted
    assert "teardown" in result.reason


def test_new_freeze_hash_rejects_old_prepared_ticket_confirmation():
    core = _prepared_core()
    intent = _intent("CONFIRM_PRECONTACT", intent_id="move-1", prepared="prepare-1")
    intent = LiveIntent(**{**intent.__dict__, "target_snapshot_sha256": "f" * 64})
    result = core.begin_confirm(intent, now_unix_s=100.1, now_monotonic=5.1)
    assert not result.accepted
    assert "snapshot hash mismatch" in result.reason


def test_protocol_order_requires_cart_then_freeze_then_prepare():
    core = PrecontactLeaseCore()
    assert not core.freeze_button1_target(
        _intent("FREEZE_BUTTON1_TARGET", intent_id="freeze-early"), now_unix_s=100.0,
    ).accepted
    assert not core.begin_prepare(
        _intent("PREPARE_DYNAMIC_IK", intent_id="prepare-early"), now_unix_s=100.0,
    ).accepted
    assert core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="cart-1"), now_unix_s=100.0,
    ).accepted
    blocked = core.begin_prepare(
        _intent("PREPARE_DYNAMIC_IK", intent_id="prepare-before-freeze"), now_unix_s=100.0,
    )
    assert not blocked.accepted and core.phase is LeasePhase.CART_PARKED


def test_new_cart_epoch_invalidates_frozen_target_and_ticket_bindings():
    core = _frozen_core()
    decision = core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="cart-2", cart_epoch="cart-2"),
        now_unix_s=100.0,
    )
    assert decision.accepted and core.phase is LeasePhase.CART_PARKED
    assert core.cart_pose_epoch == "cart-2"
    assert core.target_snapshot_sha256 == ""
    assert core.canonical_target_snapshot_json == ""


def test_new_cart_epoch_clears_prior_terminal_prepare_reason():
    core = _frozen_core()
    assert core.begin_prepare(
        _intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0,
    ).accepted
    rejected = core.record_prepare(
        _receipt(accepted=False, reason="WAYPOINT_CAP_REJECTED"),
        now_monotonic=5.0,
    )
    assert not rejected.accepted
    assert core.phase is LeasePhase.TERMINAL
    assert "WAYPOINT_CAP_REJECTED" in core.terminal_reason

    accepted = core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="cart-2", cart_epoch="cart-2"),
        now_unix_s=100.0,
    )
    assert accepted.accepted
    assert core.phase is LeasePhase.CART_PARKED
    assert core.terminal_reason == ""


def test_freeze_rejects_intent_cart_epoch_mismatch():
    core = PrecontactLeaseCore()
    assert core.confirm_cart_parked(
        _intent("CONFIRM_CART_PARKED", intent_id="cart-1"), now_unix_s=100.0,
    ).accepted
    result = core.freeze_button1_target(
        _intent("FREEZE_BUTTON1_TARGET", intent_id="freeze-wrong", cart_epoch="cart-2"),
        now_unix_s=100.0,
    )
    assert not result.accepted and "cart epoch mismatch" in result.reason
    assert core.phase is LeasePhase.CART_PARKED


def test_prepare_rejects_coordinated_replacement_snapshot_after_freeze():
    core = _frozen_core()
    snapshot, _ = _snapshot()
    value = json.loads(snapshot)
    value["frozen_target"]["detection_id"] = "replacement"
    replacement = canonical_json(value)
    intent = _intent("PREPARE_DYNAMIC_IK", intent_id="prepare-replacement")
    intent = LiveIntent(**{
        **intent.__dict__,
        "canonical_target_snapshot_json": replacement,
        "target_snapshot_sha256": _sha(replacement),
    })
    result = core.begin_prepare(intent, now_unix_s=100.0)
    assert not result.accepted and "snapshot hash mismatch" in result.reason
    assert core.phase is LeasePhase.TARGET_FROZEN


def test_prepare_receipt_cart_epoch_mismatch_latches_terminal():
    core = _frozen_core()
    assert core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0).accepted
    result = core.record_prepare(_receipt(cart_pose_epoch="cart-2"), now_monotonic=5.0)
    assert not result.accepted and "cart epoch mismatch" in result.reason
    assert core.phase is LeasePhase.TERMINAL


def test_prepare_receipt_motion_resource_fact_mismatch_latches_terminal():
    for changes in (
        {"publisher_count": 1}, {"lease_active": True}, {"arm_client_count": 1},
    ):
        core = _frozen_core()
        assert core.begin_prepare(_intent("PREPARE_DYNAMIC_IK"), now_unix_s=100.0).accepted
        result = core.record_prepare(_receipt(**changes), now_monotonic=5.0)
        assert not result.accepted and "motion resources" in result.reason
        assert core.phase is LeasePhase.TERMINAL
