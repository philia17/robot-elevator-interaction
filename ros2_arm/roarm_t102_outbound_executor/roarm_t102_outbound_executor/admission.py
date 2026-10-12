"""Pure, no-motion admission and one-use ticket checks for T102 fast v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import secrets
from typing import Any, Mapping, Sequence


MAX_TICKET_TTL_S = 10.0


class AdmissionError(RuntimeError):
    pass


def canonical_snapshot(snapshot_json: str, supplied_sha256: str) -> Mapping[str, Any]:
    if not isinstance(snapshot_json, str) or not snapshot_json:
        raise AdmissionError("canonical target snapshot is required")
    try:
        value = json.loads(snapshot_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AdmissionError("target snapshot is not JSON") from exc
    if not isinstance(value, Mapping):
        raise AdmissionError("target snapshot must be an object")
    try:
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise AdmissionError("target snapshot is not canonical finite JSON") from exc
    if canonical != snapshot_json:
        raise AdmissionError("target snapshot is not canonical JSON")
    actual = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
    if not isinstance(supplied_sha256, str) or not secrets.compare_digest(
        actual, supplied_sha256
    ):
        raise AdmissionError("target snapshot SHA-256 mismatch")
    if value.get("schema") != "m0_demo_freeze_v1":
        raise AdmissionError("target snapshot schema is not m0_demo_freeze_v1")
    frozen = value.get("frozen_target")
    if not isinstance(frozen, Mapping):
        raise AdmissionError("target snapshot has no frozen_target object")
    label = str(frozen.get("target_label") or value.get("target_label") or "")
    normalized = label.strip().lower().replace("_", " ").replace("button ", "").strip()
    if normalized not in {"1", "one"}:
        raise AdmissionError("target snapshot is not Button 1")
    if not str(frozen.get("detection_id") or "").strip():
        raise AdmissionError("target snapshot detection_id is missing")
    if int(frozen.get("source_stamp_ns") or 0) <= 0:
        raise AdmissionError("target snapshot source stamp is missing")
    if not str(frozen.get("frame_id") or "").strip():
        raise AdmissionError("target snapshot frame is missing")
    surface = frozen.get("surface_geometry")
    if not isinstance(surface, Mapping):
        raise AdmissionError("target snapshot surface provenance is missing")
    if surface.get("estimate_valid") is not True:
        raise AdmissionError("target snapshot surface estimate is not valid")
    if surface.get("provenance_complete") is not True:
        raise AdmissionError("target snapshot provenance is incomplete")
    if frozen.get("motion_authorized") is not False or value.get("motion_authorized") is not False:
        raise AdmissionError("snapshot authority flags are malformed")
    embedded = str(value.get("canonical_payload_sha256") or "")
    unsigned = dict(value)
    unsigned.pop("canonical_payload_sha256", None)
    unsigned_text = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if not secrets.compare_digest(
        embedded, hashlib.sha256(unsigned_text.encode("utf-8")).hexdigest()
    ):
        raise AdmissionError("target snapshot embedded payload hash mismatch")
    return value


@dataclass(frozen=True)
class MoveTicket:
    authorization_id: str
    executor_instance_nonce: str
    package_sha256: str
    target_snapshot_sha256: str
    ui_request_id: str
    start_positions_rad: tuple[float, ...]
    feedback_stamp_ns: int
    issued_monotonic: float
    expires_monotonic: float

    def validate_for_move(
        self,
        authorization_id: str,
        target_snapshot_sha256: str,
        now_monotonic: float,
        *,
        executor_instance_nonce: str,
        package_sha256: str,
    ) -> None:
        checks = (
            (self.authorization_id, authorization_id, "authorization id"),
            (self.target_snapshot_sha256, target_snapshot_sha256, "target snapshot"),
            (self.executor_instance_nonce, executor_instance_nonce, "executor instance"),
            (self.package_sha256, package_sha256, "v2 package"),
        )
        for expected, actual, label in checks:
            if not secrets.compare_digest(expected, str(actual)):
                raise AdmissionError(f"{label} does not match prepared ticket")
        if not self.issued_monotonic <= float(now_monotonic) < self.expires_monotonic:
            raise AdmissionError("prepare ticket is expired")


def make_ticket(
    *,
    executor_instance_nonce: str,
    package_sha256: str,
    target_snapshot_sha256: str,
    ui_request_id: str,
    start_positions_rad: Sequence[float],
    feedback_stamp_ns: int,
    now_monotonic: float,
    ttl_s: float = MAX_TICKET_TTL_S,
) -> MoveTicket:
    try:
        ttl = float(ttl_s)
        start = tuple(float(value) for value in start_positions_rad)
    except (TypeError, ValueError) as exc:
        raise AdmissionError("ticket observation is invalid") from exc
    if not 0.0 < ttl <= MAX_TICKET_TTL_S:
        raise AdmissionError("ticket TTL must be in (0, 10] seconds")
    if len(start) != 6 or not all(math.isfinite(value) for value in start):
        raise AdmissionError("ticket start observation must contain six finite joints")
    if not str(ui_request_id).strip():
        raise AdmissionError("ui_request_id is required")
    if int(feedback_stamp_ns) <= 0:
        raise AdmissionError("ticket feedback stamp must be positive")
    if not math.isfinite(float(now_monotonic)):
        raise AdmissionError("ticket monotonic time is invalid")
    return MoveTicket(
        authorization_id=secrets.token_urlsafe(32),
        executor_instance_nonce=str(executor_instance_nonce),
        package_sha256=str(package_sha256),
        target_snapshot_sha256=str(target_snapshot_sha256),
        ui_request_id=str(ui_request_id),
        start_positions_rad=start,
        feedback_stamp_ns=int(feedback_stamp_ns),
        issued_monotonic=float(now_monotonic),
        expires_monotonic=float(now_monotonic) + ttl,
    )
