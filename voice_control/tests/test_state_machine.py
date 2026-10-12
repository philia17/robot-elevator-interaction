import json

from voice_control.audit import AuditLogger
from voice_control.config import VoiceControlConfig
from voice_control.gateway import DryRunGateway
from voice_control.state_machine import State, VoiceStateMachine


def machine(tmp_path, *, ttl=5.0, auto_complete=True):
    gateway = DryRunGateway(auto_complete=auto_complete)
    state_machine = VoiceStateMachine(
        VoiceControlConfig(confirmation_ttl_seconds=ttl),
        gateway,
        AuditLogger(tmp_path / "audit.jsonl"),
    )
    return state_machine, gateway


def test_confirmation_binds_command_id_and_floor(tmp_path):
    sm, gateway = machine(tmp_path)
    pending = sm.process_text("去三楼", now=100.0)
    assert sm.state is State.AWAITING_CONFIRMATION

    wrong_id = sm.confirm("not-the-id", 3, now=101.0)
    assert wrong_id.safety_decision == "reject_confirm_command_id_mismatch"
    wrong_floor = sm.confirm(pending.command_id, 4, now=101.0)
    assert wrong_floor.safety_decision == "reject_confirm_floor_mismatch"
    assert gateway.submissions == []

    accepted = sm.confirm(pending.command_id, 3, now=101.0)
    assert accepted.safety_decision == "submitted"
    assert gateway.submissions[0].floor == 3
    assert gateway.submissions[0].command_id == pending.command_id
    assert sm.state is State.IDLE


def test_expired_confirmation_never_submits(tmp_path):
    sm, gateway = machine(tmp_path, ttl=5.0)
    pending = sm.process_text("去三楼", now=10.0)
    result = sm.confirm(pending.command_id, 3, now=15.001)
    assert result.safety_decision == "reject_confirm_expired"
    assert gateway.submissions == []


def test_new_floor_request_replaces_pending_but_wrong_confirmation_does_not(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)
    floor_three = sm.process_text("去三楼", now=100.0)

    mismatch = sm.process_text("确认四楼", now=101.0)
    assert mismatch.safety_decision == "reject_confirm_floor_mismatch"
    assert sm.pending is not None
    assert sm.pending.floor == 3
    assert sm.pending.command_id == floor_three.command_id
    assert gateway.submissions == []

    floor_four = sm.process_text("帮我按四楼", now=102.0)
    assert floor_four.safety_decision == "pending_replaced"
    assert floor_four.command_id != floor_three.command_id
    assert sm.pending is not None
    assert sm.pending.floor == 4
    assert sm.pending.command_id == floor_four.command_id

    accepted = sm.process_text("确认四楼", now=103.0)
    assert accepted.safety_decision == "submitted"
    assert accepted.command_id == floor_four.command_id
    assert len(gateway.submissions) == 1
    assert gateway.submissions[0].floor == 4


def test_cancel_marker_with_new_floor_replaces_pending_and_requests_confirmation(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)
    floor_three = sm.process_text("去三楼", now=100.0)

    floor_four = sm.process_text("算了，我想去四楼", now=101.0)
    assert floor_four.safety_decision == "pending_replaced"
    assert floor_four.command_id != floor_three.command_id
    assert sm.pending is not None
    assert sm.pending.floor == 4
    assert gateway.submissions == []

    accepted = sm.process_text("确认四楼", now=102.0)
    assert accepted.safety_decision == "submitted"
    assert len(gateway.submissions) == 1
    assert gateway.submissions[0].floor == 4


def test_natural_correction_replaces_pending_but_remains_unsubmitted(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)
    floor_three = sm.process_text("我想去三楼", now=100.0)

    floor_four = sm.process_text("等下，不对，现在我想去四楼", now=101.0)
    assert floor_four.safety_decision == "pending_replaced"
    assert floor_four.command_id != floor_three.command_id
    assert sm.pending is not None
    assert sm.pending.floor == 4
    assert gateway.submissions == []


def test_english_flow_uses_identical_confirmation_binding(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)
    floor_three = sm.process_text("Go to floor three", now=100.0)
    assert floor_three.safety_decision == "await_confirmation"

    mismatch = sm.process_text("Confirm floor four", now=101.0)
    assert mismatch.safety_decision == "reject_confirm_floor_mismatch"
    assert sm.pending is not None and sm.pending.floor == 3

    floor_four = sm.process_text("Change it to floor four", now=102.0)
    assert floor_four.safety_decision == "pending_replaced"
    assert floor_four.command_id != floor_three.command_id
    assert gateway.submissions == []

    accepted = sm.process_text("Yes", now=103.0)
    assert accepted.safety_decision == "submitted"
    assert len(gateway.submissions) == 1
    assert gateway.submissions[0].floor == 4


def test_english_yes_without_pending_never_submits(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)
    outcome = sm.process_text("Yes", now=100.0)
    assert outcome.safety_decision == "reject_confirm_no_pending"
    assert gateway.submissions == []


def test_concise_english_no_floor_replaces_only_a_pending_target(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)

    outside_confirmation = sm.process_text("No, first floor", now=99.0)
    assert outside_confirmation.safety_decision == "reject_unknown"
    assert gateway.submissions == []

    floor_three = sm.process_text("Go to floor three", now=100.0)
    corrected = sm.process_text("No, first floor", now=101.0)
    assert corrected.safety_decision == "pending_replaced"
    assert corrected.command_id != floor_three.command_id
    assert sm.pending is not None
    assert sm.pending.floor == 1
    assert gateway.submissions == []

    accepted = sm.process_text("Yes", now=102.0)
    assert accepted.safety_decision == "submitted"
    assert len(gateway.submissions) == 1
    assert gateway.submissions[0].floor == 1


def test_repeating_same_pending_floor_keeps_command_binding_and_ttl(tmp_path):
    sm, gateway = machine(tmp_path, ttl=30.0)
    original = sm.process_text("去三楼", now=100.0)
    assert sm.pending is not None
    original_expiry = sm.pending.expires_monotonic

    repeated = sm.process_text("我想去三楼", now=110.0)
    assert repeated.safety_decision == "repeat_pending_floor"
    assert repeated.command_id == original.command_id
    assert sm.pending is not None
    assert sm.pending.expires_monotonic == original_expiry
    assert gateway.submissions == []


def test_stop_preempts_execution_abort_is_idempotent_and_callback_is_stale(tmp_path):
    sm, gateway = machine(tmp_path, auto_complete=False)
    pending = sm.process_text("去三楼", now=10.0)
    sm.confirm(pending.command_id, 3, now=11.0)
    assert sm.state is State.EXECUTING
    original_generation = sm.generation

    stopped = sm.process_text("停止")
    assert stopped.state is State.STOP_LATCHED
    assert sm.generation == original_generation + 1
    assert gateway.abort_count == 1

    sm.process_text("紧急停止")
    assert gateway.abort_count == 1
    gateway.complete_last("success")
    assert sm.state is State.STOP_LATCHED

    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    stale = [record for record in records if record.get("reason") == "stale_generation"]
    assert stale and stale[-1]["applied"] is False
