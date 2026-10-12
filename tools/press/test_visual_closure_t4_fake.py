#!/usr/bin/env python3
"""Unit tests for T4 fake visual-closure state machine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from press.visual_closure_t4_fake import (
    FakeMotionCommand,
    Phase,
    T4Config,
    VisualClosureT4Fake,
    VisualObservation,
    outward_normal,
    run_synthetic_rehearsal,
    write_rehearsal_report,
)


def _obs(
    *,
    point=(0.0, 0.0, 0.90),
    target=(0.0, 0.0, 1.0),
    normal=(0.0, 0.0, -1.0),
    p_stable=True,
    p_available=True,
    depth_live=True,
    target_depth_valid=True,
    normal_valid=True,
    feedback_fresh=True,
    joints_converged=True,
    tick=0,
    tip_source="p_depth",
    fn_camera_m=None,
    active_candidate_id=None,
    candidate_status=None,
) -> VisualObservation:
    return VisualObservation(
        point_camera_m=point,
        target_camera_xyz_m=target,
        panel_normal_camera=normal,
        p_available=p_available,
        p_stable=p_stable,
        depth_live=depth_live,
        target_depth_valid=target_depth_valid,
        normal_valid=normal_valid,
        feedback_fresh=feedback_fresh,
        joints_converged=joints_converged,
        tick=tick,
        tip_source=tip_source,
        fn_camera_m=fn_camera_m,
        active_candidate_id=active_candidate_id,
        candidate_status=candidate_status,
    )


def test_outward_normal_flips_when_dot_negative():
    n = outward_normal((0.0, 0.0, 1.0), (0.0, 0.0, 0.90), (0.0, 0.0, 1.0))
    # P-A = (0,0,-0.1); measured +Z would point into panel, so flip to -Z.
    assert n[2] == pytest.approx(-1.0)


def test_fake_command_rejects_serial_claims():
    with pytest.raises(ValueError):
        FakeMotionCommand(
            phase="OBSERVE",
            kind="hold",
            delta_camera_m=(0.0, 0.0, 0.0),
            reason="bad",
            serial_write=True,
        )


def test_happy_path_reaches_verify_precontact_without_serial():
    result = run_synthetic_rehearsal()
    assert result.passed is True
    assert result.final_phase == Phase.DONE.value
    assert result.serial_write_count == 0
    assert result.abort_reason is None
    phases = [row["phase_in"] for row in result.history]
    assert Phase.OBSERVE.value in phases
    assert Phase.ALIGN_PANEL.value in phases
    assert Phase.APPROACH_NORMAL.value in phases
    assert Phase.VERIFY_PRECONTACT.value in phases
    assert all(not cmd["serial_write"] for cmd in result.commands)
    assert all(not cmd["t101"] and not cmd["t102"] for cmd in result.commands)
    kinds = {cmd["kind"] for cmd in result.commands}
    assert "tangent" in kinds
    assert "normal" in kinds
    assert result.tip_source == "fn_overlay"
    assert result.active_candidate_id == "section_c_fingertip_c0_operator_click_best_20260806_v1"


def test_abort_lost_target_during_align():
    machine = VisualClosureT4Fake()
    machine.step(_obs(tick=0))
    assert machine.phase == Phase.ALIGN_PANEL
    machine.step(
        VisualObservation(
            point_camera_m=(0.01, 0.0, 0.90),
            target_camera_xyz_m=None,
            panel_normal_camera=(0.0, 0.0, -1.0),
            p_available=True,
            p_stable=True,
            depth_live=True,
            target_depth_valid=False,
            normal_valid=True,
            feedback_fresh=True,
            tick=1,
        )
    )
    assert machine.phase == Phase.ABORTED
    assert machine.abort_reason == "lost target / depth invalid"
    assert machine.result().passed is False


def test_abort_invalid_normal():
    machine = VisualClosureT4Fake()
    machine.step(_obs(tick=0))
    machine.step(
        VisualObservation(
            point_camera_m=(0.01, 0.0, 0.90),
            target_camera_xyz_m=(0.0, 0.0, 1.0),
            panel_normal_camera=None,
            p_available=True,
            p_stable=True,
            depth_live=True,
            target_depth_valid=True,
            normal_valid=False,
            feedback_fresh=True,
            tick=1,
        )
    )
    assert machine.phase == Phase.ABORTED
    assert machine.abort_reason == "invalid panel normal"


def test_abort_stale_feedback():
    machine = VisualClosureT4Fake()
    machine.step(_obs(tick=0))
    machine.step(_obs(feedback_fresh=False, point=(0.01, 0.0, 0.90), tick=1))
    assert machine.phase == Phase.ABORTED
    assert machine.abort_reason == "stale feedback"


def test_abort_non_convergence_align():
    cfg = T4Config(max_align_steps=2, tangent_step_m=0.0005, gain=0.05)
    machine = VisualClosureT4Fake(config=cfg)
    machine.step(_obs(tick=0))
    # Large residual, tiny steps → hit max_align_steps
    for tick in range(1, 10):
        machine.step(_obs(point=(0.05, 0.05, 0.90), tick=tick))
        if machine.phase == Phase.ABORTED:
            break
    assert machine.phase == Phase.ABORTED
    assert machine.abort_reason == "non-convergence in ALIGN_PANEL"


def test_observe_waits_then_enters_align():
    machine = VisualClosureT4Fake()
    cmd = machine.step(_obs(p_stable=False, tick=0))
    assert machine.phase == Phase.OBSERVE
    assert cmd is not None and cmd.kind == "hold"
    machine.step(_obs(p_stable=True, tick=1))
    assert machine.phase == Phase.ALIGN_PANEL


def test_fn_profile_does_not_gate_on_optional_p_depth():
    result = run_synthetic_rehearsal()
    assert result.passed

    # A selected frozen Fn remains the source of this rehearsal even when
    # markerless P-depth diagnostics are absent or stale.
    machine = VisualClosureT4Fake()
    obs = _obs(
        p_available=False,
        p_stable=False,
        depth_live=False,
        tip_source="fn_overlay",
        fn_camera_m=(0.0, 0.0, 0.90),
        active_candidate_id="section_c_fingertip_c0_operator_click_best_20260806_v1",
        candidate_status="diagnostic_overlay_default",
    )
    machine.step(obs)
    assert machine.phase == Phase.ALIGN_PANEL


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"fn_camera_m": None}, "lost fingertip Fn"),
        ({"active_candidate_id": "wrong"}, "Fn candidate ID mismatch"),
        ({"candidate_status": "accepted"}, "Fn candidate is not diagnostic overlay"),
    ],
)
def test_fn_profile_fails_closed_on_missing_or_wrong_provenance(kwargs, reason):
    machine = VisualClosureT4Fake()
    values = {
        "tip_source": "fn_overlay",
        "fn_camera_m": (0.0, 0.0, 0.90),
        "active_candidate_id": "section_c_fingertip_c0_operator_click_best_20260806_v1",
        "candidate_status": "diagnostic_overlay_default",
    }
    values.update(kwargs)
    obs = _obs(**values)
    command = machine.step(obs)
    assert machine.phase == Phase.OBSERVE
    assert command is not None
    assert reason in command.reason


def test_p_depth_profile_keeps_legacy_stale_depth_gate():
    machine = VisualClosureT4Fake()
    machine.step(_obs(tick=0))
    machine.step(_obs(depth_live=False, tick=1))
    assert machine.phase == Phase.ABORTED
    assert machine.abort_reason == "invalid depth / P stale"


def test_write_rehearsal_report(tmp_path: Path):
    result = run_synthetic_rehearsal()
    out = write_rehearsal_report(result, tmp_path / "t4.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["serial_used"] is False
    assert payload["execution_ready"] is False
    assert payload["final_phase"] == "DONE"
