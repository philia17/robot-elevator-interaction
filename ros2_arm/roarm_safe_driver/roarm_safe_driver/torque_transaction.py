"""Measured-feedback transaction support for supervised torque changes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Sequence


TORQUE_FEEDBACK_VERIFY_TIMEOUT_S = 2.0
TORQUE_FLAG_COUNT = 6


def has_recent_direct_feedback(
    received_at: float | None,
    *,
    now: float,
    timeout_s: float,
) -> bool:
    """Return true only when a direct feedback timestamp is recent."""

    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    if received_at is None:
        return False
    age_s = now - received_at
    return 0.0 <= age_s <= timeout_s


def torque_flags_match(flags: Sequence[bool] | None, requested: bool) -> bool:
    """Return true only for six direct firmware flags matching ``requested``."""

    return (
        flags is not None
        and len(flags) == TORQUE_FLAG_COUNT
        and all(flag is requested for flag in flags)
    )


@dataclass(frozen=True)
class TorqueFeedbackWait:
    """Outcome of waiting for a feedback frame newer than a serial request."""

    verified: bool
    reason: str
    observed_flags: tuple[bool, ...] | None


class TorqueFeedbackTracker:
    """Sequence-numbered direct firmware feedback with a bounded wait API.

    The driver reader records every accepted ``T:1051`` torque state.  A service
    callback snapshots the sequence before writing ``T:210`` and can therefore
    reject stale pre-command feedback without holding the controller lock while
    it waits for the corresponding measured state.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._flags: tuple[bool, ...] | None = None
        self._records: deque[tuple[int, tuple[bool, ...]]] = deque(maxlen=32)

    def snapshot(self) -> int:
        with self._condition:
            return self._sequence

    def record(self, flags: Sequence[bool]) -> None:
        checked = tuple(flags)
        with self._condition:
            self._sequence += 1
            self._flags = checked
            self._records.append((self._sequence, checked))
            self._condition.notify_all()

    def wait_for_requested_state(
        self,
        *,
        after_sequence: int,
        requested: bool,
        timeout_s: float = TORQUE_FEEDBACK_VERIFY_TIMEOUT_S,
    ) -> TorqueFeedbackWait:
        """Wait through newer frames until one verifies or the deadline expires."""

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        examined_sequence = after_sequence
        last_observed: tuple[bool, ...] | None = None
        with self._condition:
            while True:
                for sequence, flags in self._records:
                    if sequence <= examined_sequence:
                        continue
                    examined_sequence = sequence
                    last_observed = flags
                    if torque_flags_match(flags, requested):
                        return TorqueFeedbackWait(True, "verified", flags)
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    if last_observed is not None:
                        return TorqueFeedbackWait(False, "mismatch", last_observed)
                    return TorqueFeedbackWait(False, "timeout", None)
                self._condition.wait(remaining_s)
