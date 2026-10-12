import ast
from pathlib import Path


NODE = Path(__file__).resolve().parents[1] / "roarm_safe_driver" / "node.py"


def _method(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_set_torque_service_is_wired_to_t210():
    source = NODE.read_text(encoding="utf-8")
    assert '"/roarm_safe/set_torque"' in source
    assert '"T": 210' in source or '"T":210' in source
    assert "def _torque_callback" in source
    assert "refuse torque change while joint_target publishers exist" in source
    assert "refuse torque change while a command is active" in source
    assert "self.transport.write(FEEDBACK_REQUEST)" in source
    assert "wait_for_requested_state" in source
    assert "TORQUE_{label}_VERIFIED" in source
    gate = source.index(
        "refuse torque change without recent direct T:1051 feedback"
    )
    torque_write = source.index("self.transport.write(payload)", gate)
    feedback_write = source.index("self.transport.write(FEEDBACK_REQUEST)", gate)
    assert gate < torque_write < feedback_write


def test_auto_torque_recovery_is_single_attempt_and_never_arms_motion():
    source = Path(__file__).resolve().parents[1].joinpath(
        "roarm_safe_driver", "node.py"
    ).read_text()
    assert "auto_torque_recovery_enabled" in source
    assert "AUTO_TORQUE_ON_ATTEMPT_DISARMED" in source
    assert "AUTO_TORQUE_ON_VERIFIED_DISARMED" in source
    assert "_auto_torque_recovery_operator_off_latched" in source
    worker = source[source.index("def _auto_torque_recovery_worker"):source.index("def _set_torque_state")]
    assert "self.controller.arm(" not in worker
    assert "self.controller.clear_fault(" not in worker
    assert "self._set_torque_state(" in worker


def test_feedback_poll_is_read_only_t105():
    source = NODE.read_text(encoding="utf-8")
    assert 'self.declare_parameter("feedback_poll_hz", 0.0)' in source
    poll = source[source.index("def _request_feedback_poll"):source.index("def _reader_loop")]
    assert "self.transport.write(FEEDBACK_REQUEST)" in poll
    assert '"T": 210' not in poll
    assert "command_multi(" not in poll


def test_non_reset_transport_fix_has_no_boot_grace_residue():
    source = NODE.read_text(encoding="utf-8")
    assert "serial_startup_grace_s" not in source


def test_manual_standby_return_is_explicit_and_never_uses_initialize_target():
    source = NODE.read_text(encoding="utf-8")
    assert '"/roarm_safe/return_to_configured_standby"' in source
    callback = (
        ast.get_source_segment(
            source, _method(source, "_return_to_configured_standby_callback")
        )
        or ""
    )
    assert "manual_standby_return_enabled" in callback
    assert "configured standby target" in callback
    assert "startup_restore_initialize_rad" not in callback
    assert "manual_standby_speed_steps_s" in callback
    assert "self.controller.joint_speed_steps_s = (speed,) * len(JOINT_NAMES)" in callback
    assert "self.controller.command_multi(target, now)" in callback


def test_implausible_joint_limits_frames_are_dropped_without_sticky_latch():
    source = NODE.read_text(encoding="utf-8")
    assert "IMPLAUSIBLE_JOINT_LIMITS_ERROR" in source
    assert "if str(exc) == IMPLAUSIBLE_JOINT_LIMITS_ERROR:" in source
    drop = source.index("if str(exc) == IMPLAUSIBLE_JOINT_LIMITS_ERROR:")
    continue_idx = source.index("continue", drop)
    latch = source.index('latch_fault(f"invalid feedback: {exc}")', drop)
    assert drop < continue_idx < latch


def test_feedback_single_frame_drop_allowlist_is_exact_and_sticky_errors_remain():
    source = NODE.read_text(encoding="utf-8")
    handler = source[
        source.index("except SafetyError as exc:") : source.index(
            "except Exception as exc", source.index("except SafetyError as exc:")
        )
    ]
    latch = handler.index('latch_fault(f"invalid feedback: {exc}")')
    drop_endpoints = []
    for condition in (
        "if str(exc) == IMPLAUSIBLE_JOINT_LIMITS_ERROR:",
        'if str(exc) == "invalid torque status":',
    ):
        drop = handler.index(condition)
        continue_idx = handler.index("continue", drop)
        assert drop < continue_idx < latch
        drop_endpoints.append((drop, continue_idx))
    allowlist = handler[min(drop for drop, _ in drop_endpoints) : latch]
    assert "malformed JSON" not in allowlist
    assert "JSON frame is not an object" not in allowlist
    assert "feedback is not UTF-8" not in allowlist


def test_feedback_status_atomically_carries_dynamic_command_acceptance():
    source = NODE.read_text(encoding="utf-8")
    reader = source[source.index("def _reader_loop"):source.index("def _publish_status")]
    lock = reader.index("with self._controller_lock:")
    accept = reader.index("self.controller.accept_feedback(feedback)", lock)
    sequence = reader.index('"dynamic_exact_accepted_command_sequence"', accept)
    target = reader.index('"dynamic_exact_accepted_target_positions"', sequence)
    ready = reader.index('"dynamic_exact_advance_ready"', target)
    recovery = reader.index("self._observe_auto_torque_recovery", ready)
    assert lock < accept < sequence < target < ready < recovery
