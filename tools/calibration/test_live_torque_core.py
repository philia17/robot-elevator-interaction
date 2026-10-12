from pathlib import Path

import pytest

from tools.calibration.live_torque_core import (
    LEGACY_ACKNOWLEDGEMENT,
    SERVER_VERIFICATION_CONFLICT,
    SERVER_VERIFIED,
    TORQUE_FEEDBACK_VERIFY_TIMEOUT_S,
    TorqueVerification,
    classify_server_acknowledgement,
    torque_flags_match,
)


LIVE_VIEW_SOURCE = (
    Path(__file__).resolve().parents[2] / "tools/apriltag_calibration_live_view.py"
)


def test_torque_flags_require_exactly_six_direct_boolean_matches():
    assert torque_flags_match([True] * 6, True)
    assert torque_flags_match([False] * 6, False)
    assert not torque_flags_match([True] * 5 + [False], True)
    assert not torque_flags_match([1] * 6, True)
    assert not torque_flags_match((True,) * 6, True)


def test_verification_requires_new_feedback_after_service_acknowledgement():
    verification = TorqueVerification.from_acknowledgement(True, 100.0)
    assert not verification.feedback_verifies(99.999, [True] * 6)
    assert not verification.feedback_verifies(100.0, [True] * 6)
    assert not verification.feedback_verifies(100.001, [True] * 5 + [False])
    assert verification.feedback_verifies(100.001, [True] * 6)
    assert not verification.expired(
        100.0 + TORQUE_FEEDBACK_VERIFY_TIMEOUT_S - 0.001
    )
    assert verification.expired(100.0 + TORQUE_FEEDBACK_VERIFY_TIMEOUT_S)


def test_verification_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        TorqueVerification.from_acknowledgement(True, 1.0, timeout_s=0.0)


@pytest.mark.parametrize(
    ("requested_enabled", "message"),
    [
        (True, "TORQUE_ON_VERIFIED"),
        (False, "TORQUE_OFF_VERIFIED"),
    ],
)
def test_matching_server_verified_response_is_authoritative(
    requested_enabled, message
):
    assert (
        classify_server_acknowledgement(message, requested_enabled)
        == SERVER_VERIFIED
    )


@pytest.mark.parametrize(
    ("requested_enabled", "message"),
    [
        (True, "TORQUE_OFF_VERIFIED"),
        (False, "TORQUE_ON_VERIFIED"),
    ],
)
def test_conflicting_server_verified_response_is_rejected(
    requested_enabled, message
):
    assert (
        classify_server_acknowledgement(message, requested_enabled)
        == SERVER_VERIFICATION_CONFLICT
    )


def test_legacy_success_response_still_requires_fresh_feedback():
    assert (
        classify_server_acknowledgement("TORQUE_ON_SENT", True)
        == LEGACY_ACKNOWLEDGEMENT
    )


def test_live_view_keeps_service_ack_distinct_from_feedback_verification():
    source = LIVE_VIEW_SOURCE.read_text(encoding="utf-8")
    assert "TorqueVerification.from_acknowledgement" in source
    assert "awaiting fresh feedback" in source
    assert "server-confirmed direct feedback" in source
    assert "conflicting verified response" in source
    assert "Torque {label}: VERIFIED" in source
    assert "Torque {label}: UNVERIFIED" in source
    assert "Torque {label}: OK" not in source
