"""Pure state helpers for fail-visible Torque ON/OFF verification."""

from __future__ import annotations

from dataclasses import dataclass


TORQUE_FEEDBACK_VERIFY_TIMEOUT_S = 3.0
SERVER_VERIFIED = "server_verified"
SERVER_VERIFICATION_CONFLICT = "server_verification_conflict"
LEGACY_ACKNOWLEDGEMENT = "legacy_acknowledgement"


def classify_server_acknowledgement(
    message: str,
    requested_enabled: bool,
) -> str:
    """Classify a successful service response without trusting the wrong state."""

    requested_label = "ON" if requested_enabled else "OFF"
    expected = f"TORQUE_{requested_label}_VERIFIED"
    if message == expected:
        return SERVER_VERIFIED
    if message in {"TORQUE_ON_VERIFIED", "TORQUE_OFF_VERIFIED"}:
        return SERVER_VERIFICATION_CONFLICT
    return LEGACY_ACKNOWLEDGEMENT


def torque_flags_match(flags: object, requested_enabled: bool) -> bool:
    """Return true only for six direct firmware flags matching the request."""

    return (
        isinstance(flags, list)
        and len(flags) == 6
        and all(flag is requested_enabled for flag in flags)
    )


@dataclass(frozen=True)
class TorqueVerification:
    """A service acknowledgement awaiting newer direct firmware feedback."""

    requested_enabled: bool
    acknowledged_at: float
    deadline: float

    @classmethod
    def from_acknowledgement(
        cls,
        requested_enabled: bool,
        acknowledged_at: float,
        timeout_s: float = TORQUE_FEEDBACK_VERIFY_TIMEOUT_S,
    ) -> "TorqueVerification":
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        return cls(
            requested_enabled=bool(requested_enabled),
            acknowledged_at=acknowledged_at,
            deadline=acknowledged_at + timeout_s,
        )

    def feedback_verifies(self, received_at: float, flags: object) -> bool:
        """Require feedback received after the service acknowledgement."""

        return (
            received_at > self.acknowledged_at
            and torque_flags_match(flags, self.requested_enabled)
        )

    def expired(self, now: float) -> bool:
        return now >= self.deadline
