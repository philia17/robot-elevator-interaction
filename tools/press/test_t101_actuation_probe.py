"""Pure safety tests for the one-shot T101 actuation probe."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.press.t101_actuation_probe import (  # noqa: E402
    FeedbackSnapshot,
    ProbeError,
    build_probe_target,
    evaluate_probe_samples,
    parse_feedback_payload,
    validate_preflight,
)


Q0 = (0.042951462, -1.188835111, 2.330116817, 0.458660256, -1.515573018, 0.001533980589793149)


def feedback_payload(**overrides):
    value = {
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "dry_run": False,
        "fault": "",
        "positions_rad": list(Q0),
        "torque_enabled": [True] * 6,
        "voltage": 12.1,
        "feedback_stamp_ns": 123,
    }
    value.update(overrides)
    return json.dumps(value)


def test_parse_and_preflight_accept_fresh_disarmed_feedback():
    snapshot = parse_feedback_payload(feedback_payload(), received_monotonic=10.0)
    validate_preflight(snapshot, now=10.2)
    assert snapshot.positions_rad == Q0


def test_preflight_rejects_stale_fault_or_partial_torque():
    with pytest.raises(ProbeError, match="stale"):
        validate_preflight(parse_feedback_payload(feedback_payload(), received_monotonic=1.0), now=3.0)
    with pytest.raises(ProbeError, match="fault"):
        validate_preflight(parse_feedback_payload(feedback_payload(fault="fault"), received_monotonic=1.0), now=1.1)
    with pytest.raises(ProbeError, match="torque"):
        validate_preflight(parse_feedback_payload(feedback_payload(torque_enabled=[True, True, False, True, True, True]), received_monotonic=1.0), now=1.1)


def test_build_probe_target_changes_only_roll_and_respects_delta():
    target = build_probe_target(Q0)
    assert target[4] == pytest.approx(Q0[4] + 0.040)
    assert [i for i, (a, b) in enumerate(zip(Q0, target)) if a != b] == [4]
    with pytest.raises(ProbeError, match="0.035..0.050"):
        build_probe_target(Q0, delta_rad=0.020)


def test_probe_success_requires_measured_correct_direction_motion():
    samples = [Q0[:4] + (Q0[4] + 0.004,) + Q0[5:]]
    samples.append(Q0[:4] + (Q0[4] + 0.016,) + Q0[5:])
    result = evaluate_probe_samples(Q0, samples)
    assert result["success"] is True
    assert result["reason"] == "measured_correct_direction_motion"


def test_zero_motion_cannot_pass_even_if_target_exists():
    result = evaluate_probe_samples(Q0, [Q0, Q0, Q0])
    assert result == {
        "success": False,
        "reason": "no_measurable_motion",
        "sample_count": 3,
        "best_signed_motion_rad": 0.0,
        "max_non_target_drift_rad": 0.0,
    }


def test_wrong_direction_and_non_target_drift_abort():
    wrong = list(Q0)
    wrong[4] -= 0.006
    result = evaluate_probe_samples(Q0, [wrong])
    assert result["reason"] == "wrong_direction"
    drift = list(Q0)
    drift[0] += 0.021
    result = evaluate_probe_samples(Q0, [drift])
    assert result["reason"] == "unexpected_non_target_drift"
