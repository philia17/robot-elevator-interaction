"""Pure replay-safe validation for read-only pre-contact status topics.

The two status streams are advisory UI evidence, never execution permission on
their own.  A dashboard must reject schema drift, sequence replays, and a
publisher instance replacement rather than treating a recently received JSON
message as trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ADAPTER_STATUS_SCHEMA = "roarm_precontact_status_v1"
QC_STATUS_SCHEMA = "roarm_precontact_qc_status_v1"


@dataclass(frozen=True)
class StatusObservation:
    """One status validation result; an invalid result is never ready."""

    accepted: bool
    reason: str
    instance_nonce: str | None
    status_sequence: int | None


class MonotonicStatusTracker:
    """Latch one publisher instance and require strictly increasing sequence.

    A different nonce is not an implicit restart permission.  Restarting the
    Live/QC observer is a deliberate operator-visible lifecycle boundary; a
    stale or substituted publisher cannot reset the sequence in-process.
    """

    def __init__(self, *, schema: str) -> None:
        self._schema = schema
        self._instance_nonce: str | None = None
        self._last_status_sequence: int | None = None

    @property
    def instance_nonce(self) -> str | None:
        return self._instance_nonce

    @property
    def last_status_sequence(self) -> int | None:
        return self._last_status_sequence

    def observe(self, value: Any) -> StatusObservation:
        """Validate a decoded JSON object without mutating state on failure."""

        if not isinstance(value, Mapping):
            return StatusObservation(False, "status is not an object", None, None)
        if value.get("schema") != self._schema:
            return StatusObservation(False, "status schema mismatch", None, None)
        nonce = value.get("instance_nonce")
        if not isinstance(nonce, str) or len(nonce) < 16:
            return StatusObservation(False, "status instance nonce is invalid", None, None)
        sequence = value.get("status_sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            return StatusObservation(False, "status sequence is invalid", nonce, None)
        if self._instance_nonce is not None and nonce != self._instance_nonce:
            return StatusObservation(False, "status instance nonce changed", nonce, sequence)
        if self._last_status_sequence is not None and sequence <= self._last_status_sequence:
            return StatusObservation(False, "status sequence replay/out-of-order", nonce, sequence)
        self._instance_nonce = nonce
        self._last_status_sequence = sequence
        return StatusObservation(True, "OK", nonce, sequence)
