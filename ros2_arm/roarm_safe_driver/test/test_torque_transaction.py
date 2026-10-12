import threading

import pytest

from roarm_safe_driver.torque_transaction import (
    TorqueFeedbackTracker,
    has_recent_direct_feedback,
    torque_flags_match,
)


def test_recent_direct_feedback_requires_a_present_nonstale_timestamp():
    assert has_recent_direct_feedback(9.5, now=10.0, timeout_s=0.5)
    assert not has_recent_direct_feedback(None, now=10.0, timeout_s=0.5)
    assert not has_recent_direct_feedback(9.499, now=10.0, timeout_s=0.5)
    assert not has_recent_direct_feedback(10.001, now=10.0, timeout_s=0.5)


def test_recent_direct_feedback_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        has_recent_direct_feedback(1.0, now=1.0, timeout_s=0.0)


def test_torque_flags_require_six_direct_flags_matching_request():
    assert torque_flags_match((True,) * 6, True)
    assert torque_flags_match((False,) * 6, False)
    assert not torque_flags_match((True,) * 5, True)
    assert not torque_flags_match((True,) * 5 + (False,), True)


def test_wait_verifies_only_feedback_newer_than_snapshot():
    tracker = TorqueFeedbackTracker()
    tracker.record((False,) * 6)
    snapshot = tracker.snapshot()
    ready = threading.Event()

    def publish_new_feedback():
        ready.wait(timeout=0.5)
        tracker.record((True,) * 6)

    worker = threading.Thread(target=publish_new_feedback)
    worker.start()
    ready.set()
    result = tracker.wait_for_requested_state(
        after_sequence=snapshot, requested=True, timeout_s=0.5
    )
    worker.join(timeout=0.5)

    assert result.verified
    assert result.reason == "verified"
    assert result.observed_flags == (True,) * 6


def test_wait_rejects_newer_observed_mismatch():
    tracker = TorqueFeedbackTracker()
    snapshot = tracker.snapshot()
    tracker.record((False,) * 6)

    result = tracker.wait_for_requested_state(
        after_sequence=snapshot, requested=True, timeout_s=0.1
    )

    assert not result.verified
    assert result.reason == "mismatch"
    assert result.observed_flags == (False,) * 6


def test_wait_accepts_a_later_match_after_an_initial_mismatch():
    tracker = TorqueFeedbackTracker()
    snapshot = tracker.snapshot()
    tracker.record((False,) * 6)
    worker = threading.Timer(0.02, tracker.record, args=((True,) * 6,))
    worker.start()
    result = tracker.wait_for_requested_state(
        after_sequence=snapshot, requested=True, timeout_s=0.5
    )
    worker.join(timeout=0.5)

    assert result.verified
    assert result.reason == "verified"
    assert result.observed_flags == (True,) * 6


def test_wait_times_out_without_newer_feedback():
    tracker = TorqueFeedbackTracker()

    result = tracker.wait_for_requested_state(
        after_sequence=tracker.snapshot(), requested=True, timeout_s=0.01
    )

    assert not result.verified
    assert result.reason == "timeout"
    assert result.observed_flags is None


def test_wait_rejects_nonpositive_timeout():
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        TorqueFeedbackTracker().wait_for_requested_state(
            after_sequence=0, requested=True, timeout_s=0.0
        )
