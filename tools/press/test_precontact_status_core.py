"""Unit tests for replay-safe, read-only pre-contact status validation."""

from __future__ import annotations

from press.precontact_status_core import (
    ADAPTER_STATUS_SCHEMA,
    QC_STATUS_SCHEMA,
    MonotonicStatusTracker,
)


def _status(*, schema=ADAPTER_STATUS_SCHEMA, nonce="a" * 32, sequence=1) -> dict:
    return {
        "schema": schema,
        "instance_nonce": nonce,
        "status_sequence": sequence,
    }


def test_tracker_requires_fixed_schema_and_strict_monotonic_sequence():
    tracker = MonotonicStatusTracker(schema=ADAPTER_STATUS_SCHEMA)
    assert not tracker.observe(_status(schema=QC_STATUS_SCHEMA)).accepted
    assert tracker.observe(_status(sequence=1)).accepted
    assert not tracker.observe(_status(sequence=1)).accepted
    assert not tracker.observe(_status(sequence=0)).accepted
    assert tracker.observe(_status(sequence=2)).accepted


def test_tracker_latches_instance_nonce_and_rejects_a_fresh_replay_from_another_instance():
    tracker = MonotonicStatusTracker(schema=QC_STATUS_SCHEMA)
    assert tracker.observe(_status(schema=QC_STATUS_SCHEMA, nonce="a" * 32, sequence=9)).accepted
    substituted = tracker.observe(
        _status(schema=QC_STATUS_SCHEMA, nonce="b" * 32, sequence=10)
    )
    assert not substituted.accepted
    assert "nonce changed" in substituted.reason
    assert tracker.observe(_status(schema=QC_STATUS_SCHEMA, nonce="a" * 32, sequence=10)).accepted


def test_tracker_rejects_missing_or_short_nonce_without_latching_it():
    tracker = MonotonicStatusTracker(schema=ADAPTER_STATUS_SCHEMA)
    assert not tracker.observe({"schema": ADAPTER_STATUS_SCHEMA, "status_sequence": 1}).accepted
    assert not tracker.observe(_status(nonce="short")).accepted
    assert tracker.instance_nonce is None
