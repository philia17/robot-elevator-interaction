"""ROS-free state for the two-click supervised pre-contact trial panel.

This module intentionally owns no ROS resources.  It exists so the GUI cannot
silently turn a successful Prepare response into a Move: the caller must make
a separate, explicit ``move_request`` call while the returned ticket is still
valid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time


MAX_TICKET_TTL_S = 2.0


class SnapshotError(ValueError):
    """The displayed frozen target must be canonical bytes, not a re-dump."""


@dataclass(frozen=True)
class CanonicalSnapshot:
    text: str
    sha256: str


@dataclass(frozen=True)
class PreparedTicket:
    authorization_id: str
    target_sha256: str
    route_sha256: str
    expires_monotonic: float

    def remaining_ms(self, now: float) -> int:
        return max(0, int((self.expires_monotonic - now) * 1000))

    def valid(self, now: float) -> bool:
        return bool(self.authorization_id and self.target_sha256 and self.route_sha256 and now < self.expires_monotonic)


def read_canonical_snapshot(path: str) -> CanonicalSnapshot:
    """Load exact canonical JSON bytes and bind the hash used by Prepare."""

    text = open(path, "r", encoding="utf-8").read()
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"target snapshot is not JSON: {exc}") from exc
    canonical = json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if text != canonical:
        raise SnapshotError("target snapshot bytes are not canonical JSON; refusing to rewrite evidence")
    return CanonicalSnapshot(text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def accept_prepare(*, accepted: bool, authorization_id: str, ttl_ms: int, planned_target_sha256: str, route_sha256: str, now: float) -> PreparedTicket | None:
    """Convert one accepted response into a bounded single-use GUI ticket."""

    if not accepted or not authorization_id or not planned_target_sha256 or not route_sha256:
        return None
    if ttl_ms <= 0 or ttl_ms > int(MAX_TICKET_TTL_S * 1000):
        return None
    return PreparedTicket(
        authorization_id=authorization_id,
        target_sha256=planned_target_sha256,
        route_sha256=route_sha256,
        expires_monotonic=now + ttl_ms / 1000.0,
    )


def consume_move(ticket: PreparedTicket | None, now: float) -> PreparedTicket | None:
    """Return a valid ticket exactly once; caller must clear it before sending."""

    return ticket if ticket is not None and ticket.valid(now) else None
