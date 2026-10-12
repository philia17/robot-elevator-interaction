"""Pure state and evidence contract for a standby-to-pre-contact Live panel.

This module deliberately has no ROS or motion imports.  It lets the UI freeze
one button-1 observation and render why a pre-contact request is disabled or
ready.  A future action server, not this UI, must own any physical execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

from press.button1_xy_align import (
    LockedButton1,
    TARGET_LABEL,
    lock_button1_from_m0_summary,
    lock_button1_from_semantics_and_observations,
)
from press.display_projection import (
    target_frame_center_state,
)
from press.button_target import (
    get_semantic_track_id, normalize_button_label, target_identity_matches,
)


class LivePhase(str, Enum):
    TARGET_LIVE = "TARGET_LIVE"
    TARGET_FROZEN = "TARGET_FROZEN"
    READY_TO_PRECONTACT = "READY_TO_PRECONTACT"
    DISABLED = "DISABLED"


# These values are display / diagnostic evidence boundaries only.  They do not
# loosen, feed, or replace any M0, freeze, motion, or action-server gate.
RAW_PAIR_DEGRADED_GAP_S = 1.5
RAW_PAIR_DROP_S = 12.0
# Display may retain a last-known pair longer, but a diagnostic capture must
# only promote a genuinely current same-header pair.
DIAGNOSTIC_RAW_PAIR_MAX_AGE_S = 2.0
# Bounded diagnostic fallback window after an operator capture click.  A
# *current* raw pair is attempted immediately at click time; this window only
# bounds how long the Live panel keeps waiting for a fresh READY M0 (or a
# later current raw pair) before sealing a display-only DiagnosticSnapshot or
# failing closed.  Raw pairs are never motion-capable; only a fresh READY M0
# may seal a motion-capable TargetSnapshot.  This is separate from
# ``DIAGNOSTIC_RAW_PAIR_MAX_AGE_S`` (the 2 s receive TTL applied when a raw
# pair is actually sealed).
DIAGNOSTIC_FALLBACK_DELAY_S = 2.5
M0_RECEIVE_MAX_AGE_S = 1.5
RAW_PAIR_RECOVERY_MATCHED_PAIRS = 3
DIAGNOSTIC_TARGET_ARTIFACT_SCHEMA = "button1_diagnostic_target_artifact_v1"
# Single-frame diagnostic capture schema.  This is deliberately distinct from
# ``DIAGNOSTIC_TARGET_ARTIFACT_SCHEMA`` (the M0 freeze record) so a diagnostic
# snapshot can never be confused with a motion-capable frozen target.
BUTTON1_DIAGNOSTIC_SNAPSHOT_SCHEMA = "button1_diagnostic_snapshot_v1"
# Matches the one-click launcher's read-only health gate (feedback stamp age
# <= 5 s).  The Live panel only displays this; it never feeds the value into a
# torque or motion gate.
SAFE_FEEDBACK_MAX_AGE_S = 5.0
SAFE_JOINT_AXES = ("B", "S", "E", "T", "R", "G")


@dataclass(frozen=True)
class EvidenceHeader:
    """The source identity needed to correlate two ROS array messages."""

    stamp_ns: int
    frame_id: str


@dataclass(frozen=True)
class DisplayCadence:
    """Display-only status for a sequence of qualified raw Button-1 pairs."""

    state: str
    consecutive_matched_pairs: int
    last_pair_received_at: float | None


@dataclass(frozen=True)
class ProjectionEvidence:
    """A display point selected without changing M0 or motion authority."""

    camera_xyz_m: tuple[float, float, float] | None
    normal_camera: tuple[float, float, float] | None
    source: str
    a2d: tuple[float, float] | None
    source_frame_id: str | None = None
    source_stamp_ns: int | None = None
    calibration_id: str | None = None
    provenance_complete: bool | None = None
    raw_to_canonical_alias_status: str | None = None


def m0_summary(message: Any | None) -> dict[str, Any] | None:
    """Convert the M0 ROS shape into a pure, immutable-friendly summary.

    This intentionally uses only attribute access so a lightweight message
    double can exercise the same conversion in unit tests.  Geometry vectors
    are tuples: the existing freeze validator deliberately rejects mutable or
    non-sequence array-like placeholders.
    """
    if message is None:
        return None
    observation = message.observation
    surface = observation.surface_geometry
    return {
        "header": EvidenceHeader(
            stamp_ns=(
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            ),
            frame_id=str(message.header.frame_id),
        ),
        "detection_id": str(getattr(message, "detection_id", "") or ""),
        "semantic_track_id": str(getattr(message, "semantic_track_id", "") or ""),
        "label": str(message.effective_label or message.target_label or "?").upper(),
        # M0 READY is emitted only after its semantic pressable/association
        # gates pass. The ROS type intentionally carries no motion authority.
        "pressable": bool(message.perception_ready) and str(message.status) == "READY",
        "ready": bool(message.perception_ready),
        "confirmations": int(message.confirmation_count),
        "bbox": (
            float(observation.x_min),
            float(observation.y_min),
            float(observation.x_max),
            float(observation.y_max),
        ),
        "camera_xyz": (
            float(observation.x_m),
            float(observation.y_m),
            float(observation.z_m),
        ),
        "depth_valid": bool(observation.depth_valid),
        "normal_valid": bool(surface.estimate_valid),
        "normal": (
            float(surface.panel_outward_normal_x),
            float(surface.panel_outward_normal_y),
            float(surface.panel_outward_normal_z),
        ),
        "status": str(message.status),
        "provenance_complete": getattr(surface, "provenance_complete", None),
        "source_frame_id": str(getattr(surface, "source_frame_id", "") or "") or None,
        "canonical_frame_id": str(getattr(surface, "canonical_frame_id", "") or "") or None,
        "raw_to_canonical_alias_id": str(getattr(surface, "raw_to_canonical_alias_id", "") or "") or None,
        "raw_to_canonical_alias_status": str(getattr(surface, "raw_to_canonical_alias_status", "") or "") or None,
        "section_b_calibration_id": str(getattr(surface, "section_b_calibration_id", "") or "") or None,
        "camera_optical_axis_status": str(getattr(surface, "camera_optical_axis_status", "") or "") or None,
        "rgb_depth_registration_status": str(getattr(surface, "rgb_depth_registration_status", "") or "") or None,
        "depth_scale_status": str(getattr(surface, "depth_scale_status", "") or "") or None,
    }


def receive_age_s(received_at: float | None, *, now: float) -> float | None:
    """Return a monotonic receive age without allowing a negative clock skew."""
    if received_at is None:
        return None
    return max(0.0, float(now) - float(received_at))


def diagnostic_fallback_due(
    capture_started_monotonic: float | None,
    now: float,
    *,
    delay_s: float = DIAGNOSTIC_FALLBACK_DELAY_S,
) -> bool:
    """Whether the bounded raw-pair diagnostic fallback window has elapsed.

    ``None`` is treated as already due so a pure one-shot capture helper can
    still fail closed when no monotonic capture start has been recorded.
    """
    if capture_started_monotonic is None:
        return True
    elapsed = max(0.0, float(now) - float(capture_started_monotonic))
    return elapsed >= float(delay_s)


def diagnostic_fallback_remaining_s(
    capture_started_monotonic: float | None,
    now: float,
    *,
    delay_s: float = DIAGNOSTIC_FALLBACK_DELAY_S,
) -> float:
    """Remaining bounded fallback time, clamped to zero (display only)."""
    if capture_started_monotonic is None:
        return 0.0
    elapsed = max(0.0, float(now) - float(capture_started_monotonic))
    return max(0.0, float(delay_s) - elapsed)


def headers_match(
    observations_header: EvidenceHeader | None,
    semantics_header: EvidenceHeader | None,
) -> tuple[bool, str | None]:
    """Require exact stamp and frame equality before an identity join.

    ``detection_id`` is only unique within its source frame.  It is never a
    cross-frame or cross-stamp correlation key.
    """
    if observations_header is None or semantics_header is None:
        return False, "MISSING_HEADER"
    if observations_header.stamp_ns != semantics_header.stamp_ns:
        return False, "HEADER_STAMP_MISMATCH"
    if observations_header.frame_id != semantics_header.frame_id:
        return False, "HEADER_FRAME_MISMATCH"
    if not observations_header.frame_id:
        return False, "HEADER_FRAME_MISSING"
    return True, None


def raw_button1_pair(
    semantics: Sequence[Mapping[str, Any]] | None,
    observations: Sequence[Mapping[str, Any]] | None,
    *,
    semantics_header: EvidenceHeader | None,
    observations_header: EvidenceHeader | None,
    target_label: str = TARGET_LABEL,
    detection_id: str | None = None,
    semantic_track_id: str | None = None,
) -> dict[str, Any]:
    """Build one display-only Button-1 pair from the *same* source frame.

    A successful identity lookup is deliberately insufficient: the two array
    headers must first carry the identical source timestamp and frame.  The
    result is data-only and does not affect M0, freeze, or execution state.
    """
    result: dict[str, Any] = {
        "matched": False,
        "a2d": None,
        "observation_3d": None,
        "observation_bbox": None,
        "observation_normal": None,
        "depth_valid": None,
        "normal_valid": None,
        "detection_id": None,
        "label": normalize_button_label(target_label),
        "semantic_track_id": "",
        "pressable": False,
        "error": None,
        "semantics_header": semantics_header,
        "observations_header": observations_header,
    }
    header_ok, header_error = headers_match(observations_header, semantics_header)
    if not header_ok:
        result["error"] = header_error
        return result
    if not semantics or not observations:
        result["error"] = "NO_RAW_TARGET_STREAM"
        return result
    matches = [
        item
        for item in semantics
        if target_identity_matches(
            item, target_label=target_label, detection_id=detection_id,
            semantic_track_id=semantic_track_id,
        )
    ]
    if len(matches) != 1:
        label = normalize_button_label(target_label).upper() or "UNKNOWN"
        result["error"] = f"NO_LABEL_{label}" if not matches else f"AMBIGUOUS_LABEL_{label}_{len(matches)}"
        return result
    detection_id = str(matches[0].get("detection_id") or "")
    if not detection_id:
        result["error"] = "LABEL_1_NO_DETECTION_ID"
        return result
    result["detection_id"] = detection_id
    result["semantic_track_id"] = get_semantic_track_id(matches[0])
    result["pressable"] = matches[0].get("pressable") is True
    result["semantic_status"] = str(matches[0].get("semantic_status") or "")
    matching_observations = [
        item for item in observations if str(item.get("detection_id") or "") == detection_id
    ]
    if not matching_observations:
        result["error"] = "NO_MATCHING_3D_OBSERVATION"
        return result
    if len(matching_observations) != 1:
        result["error"] = f"AMBIGUOUS_3D_OBSERVATION_ID_{len(matching_observations)}"
        return result
    observation = matching_observations[0]
    result["matched"] = True
    surface = observation.get("surface_geometry") or {}
    result["depth_valid"] = bool(observation.get("depth_valid"))
    result["normal_valid"] = bool(surface.get("estimate_valid"))
    result["provenance_complete"] = surface.get("provenance_complete")
    result["source_frame_id"] = str(surface.get("source_frame_id") or "") or None
    result["canonical_frame_id"] = str(surface.get("canonical_frame_id") or "") or None
    result["raw_to_canonical_alias_id"] = str(surface.get("raw_to_canonical_alias_id") or "") or None
    result["raw_to_canonical_alias_status"] = str(surface.get("raw_to_canonical_alias_status") or "") or None
    result["section_b_calibration_id"] = str(surface.get("section_b_calibration_id") or "") or None
    result["camera_optical_axis_status"] = str(surface.get("camera_optical_axis_status") or "") or None
    result["rgb_depth_registration_status"] = str(surface.get("rgb_depth_registration_status") or "") or None
    result["depth_scale_status"] = str(surface.get("depth_scale_status") or "") or None
    try:
        bbox = tuple(float(observation[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
        if not all(math.isfinite(value) for value in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("INVALID_BBOX")
        result["a2d"] = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    except (KeyError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    try:
        locked = lock_button1_from_semantics_and_observations(
            semantics, observations, target_label=target_label,
            detection_id=detection_id, semantic_track_id=semantic_track_id,
        )
    except (TypeError, ValueError) as exc:
        result["observation_error"] = str(exc)
    else:
        result["observation_3d"] = tuple(float(value) for value in locked.camera_xyz_m)
        result["observation_bbox"] = locked.bbox
        result["observation_normal"] = tuple(float(value) for value in locked.normal_camera)
    return result


def raw_pair_is_projectable(pair: Mapping[str, Any] | None) -> bool:
    """A raw pair can be projected only after a real matched 3-D observation."""
    return bool(pair and pair.get("matched") and pair.get("observation_3d") is not None)


@dataclass
class DisplayCadenceLatch:
    """Sticky UI hysteresis for slow perception cadence.

    This class is intentionally unaware of robot state and is only allowed to
    label display evidence.  It starts degraded, degrades after a receive gap,
    and needs distinct qualified source pairs to recover.
    """

    gap_s: float = RAW_PAIR_DEGRADED_GAP_S
    recovery_matched_pairs: int = RAW_PAIR_RECOVERY_MATCHED_PAIRS
    _last_header: EvidenceHeader | None = None
    _last_pair_received_at: float | None = None
    _consecutive_matched_pairs: int = 0
    _degraded: bool = True

    def observe_matched_pair(self, header: EvidenceHeader, *, received_at: float) -> DisplayCadence:
        """Record one distinct, geometry-qualified pair received by the UI."""
        timestamp = float(received_at)
        if self._last_header == header:
            # Duplicate callbacks for the same evidence must not refresh age or
            # satisfy the recovery count.
            return self.snapshot()
        if (
            self._last_pair_received_at is None
            or timestamp - self._last_pair_received_at > self.gap_s
        ):
            self._degraded = True
            self._consecutive_matched_pairs = 1
        else:
            self._consecutive_matched_pairs += 1
        self._last_header = header
        self._last_pair_received_at = timestamp
        if self._consecutive_matched_pairs >= self.recovery_matched_pairs:
            self._degraded = False
        return self.snapshot()

    def tick(self, *, now: float) -> DisplayCadence:
        """Enter degraded after a receive gap; never manufacture a recovery."""
        age = receive_age_s(self._last_pair_received_at, now=float(now))
        if age is None or age > self.gap_s:
            self._degraded = True
            self._consecutive_matched_pairs = 0
        return self.snapshot()

    def snapshot(self) -> DisplayCadence:
        return DisplayCadence(
            state="DEGRADED" if self._degraded else "VALID",
            consecutive_matched_pairs=self._consecutive_matched_pairs,
            last_pair_received_at=self._last_pair_received_at,
        )


def m0_is_fresh_ready(
    summary: Mapping[str, Any] | None,
    *,
    receive_age_s: float | None,
    max_receive_age_s: float = M0_RECEIVE_MAX_AGE_S,
) -> bool:
    """Strictly identify a fresh geometrically READY M0 display record."""
    return bool(
        summary
        and receive_age_s is not None
        and float(receive_age_s) <= max_receive_age_s
        and bool(summary.get("ready"))
        and bool(summary.get("depth_valid"))
        and bool(summary.get("normal_valid"))
        and str(summary.get("status") or "").strip().upper() == "READY"
    )


def target_projection_evidence(
    *,
    m0: Mapping[str, Any] | None,
    m0_receive_age_s: float | None,
    raw_pair: Mapping[str, Any] | None,
    raw_pair_receive_age_s: float | None,
    cadence: DisplayCadence | None,
    raw_pair_drop_s: float = RAW_PAIR_DROP_S,
) -> ProjectionEvidence:
    """Choose a diagnostic target point without changing M0 authority.

    M0 gets precedence only while it is both fresh and READY.  An expired or
    not-ready M0 cannot appear as ``M0_A3D``; a retained pair is explicitly
    labelled last-known whenever it is aged or cadence-degraded.
    """
    raw_a2d = None if raw_pair is None else raw_pair.get("a2d")
    if m0_is_fresh_ready(m0, receive_age_s=m0_receive_age_s):
        raw_is_fresh_for_m0 = bool(
            raw_pair_is_projectable(raw_pair)
            and raw_pair_receive_age_s is not None
            and float(raw_pair_receive_age_s) <= RAW_PAIR_DEGRADED_GAP_S
            and cadence is not None
            and cadence.state == "VALID"
        )
        m0_header = m0.get("header")
        raw_header = None if raw_pair is None else raw_pair.get("observations_header")
        same_source = not (
            m0_header is not None
            and raw_header is not None
            and m0_header != raw_header
        )
        m0_frame = (
            str(m0.get("source_frame_id") or "").strip()
            or (m0_header.frame_id if m0_header is not None else "")
            or None
        )
        return ProjectionEvidence(
            camera_xyz_m=tuple(float(value) for value in m0["camera_xyz"]),
            normal_camera=tuple(float(value) for value in m0["normal"]),
            source="M0_A3D",
            a2d=raw_a2d if raw_is_fresh_for_m0 and same_source else None,
            source_frame_id=m0_frame,
            source_stamp_ns=(
                m0_header.stamp_ns if m0_header is not None else None
            ),
            calibration_id=str(m0.get("section_b_calibration_id") or "") or None,
            provenance_complete=m0.get("provenance_complete"),
            raw_to_canonical_alias_status=(
                str(m0.get("raw_to_canonical_alias_status") or "") or None
            ),
        )
    raw_fresh_enough = (
        raw_pair_receive_age_s is not None
        and float(raw_pair_receive_age_s) <= raw_pair_drop_s
    )
    if raw_fresh_enough and raw_pair_is_projectable(raw_pair):
        degraded = bool(cadence and cadence.state == "DEGRADED")
        source = "OBS_A3D_LAST_KNOWN" if degraded or raw_pair_receive_age_s > RAW_PAIR_DEGRADED_GAP_S else "OBS_A3D"
        raw_header = None if raw_pair is None else raw_pair.get("observations_header")
        raw_frame = (
            str(raw_pair.get("source_frame_id") or "").strip()
            or (raw_header.frame_id if raw_header is not None else "")
            or None
        )
        return ProjectionEvidence(
            camera_xyz_m=tuple(float(value) for value in raw_pair["observation_3d"]),
            normal_camera=tuple(float(value) for value in raw_pair["observation_normal"]),
            source=source,
            a2d=raw_a2d,
            source_frame_id=raw_frame,
            source_stamp_ns=(
                raw_header.stamp_ns if raw_header is not None else None
            ),
            calibration_id=str(raw_pair.get("section_b_calibration_id") or "") or None,
            provenance_complete=raw_pair.get("provenance_complete"),
            raw_to_canonical_alias_status=(
                str(raw_pair.get("raw_to_canonical_alias_status") or "") or None
            ),
        )
    if raw_fresh_enough and raw_a2d is not None:
        raw_header = None if raw_pair is None else raw_pair.get("observations_header")
        return ProjectionEvidence(
            camera_xyz_m=None,
            normal_camera=None,
            source="A2D_ONLY_LAST_KNOWN" if cadence and cadence.state == "DEGRADED" else "A2D_ONLY",
            a2d=raw_a2d,
            source_frame_id=(
                str(raw_pair.get("source_frame_id") or "").strip()
                or (raw_header.frame_id if raw_header is not None else "")
                or None
            ),
            source_stamp_ns=(
                raw_header.stamp_ns if raw_header is not None else None
            ),
        )
    return ProjectionEvidence(
        camera_xyz_m=None,
        normal_camera=None,
        source="NO_TARGET",
        a2d=None,
    )


def wrist_center_evidence_state(
    evidence: ProjectionEvidence | None,
    *,
    expected_calibration_id: str | None,
    wrist_calibration_contract_ok: bool = False,
    bundle_contract_ok: bool = False,
    require_canonical_frame: bool = True,
) -> tuple[bool, str]:
    """Fail-closed gate for a *centered* Button-1 wrist marker.

    The wrist pane may still draw diagnostic TCP/Fn projections, but the
    Button-1 ``A`` marker is only allowed to look like a target center when the
    whole chain is contract-ready:

    * the eye-in-hand solve is not a diagnostic-only/unpublished artifact;
    * the Section-B bundle is not a relative, unpublished placeholder;
    * the target is fresh, not last-known;
    * the target is in the canonical ``camera_optical_frame`` (or a fully
      verified raw->canonical alias, when required);
    * the observation provenance is complete and the calibration ID matches.
    """
    if evidence is None or evidence.camera_xyz_m is None:
        return False, "NO_TARGET"
    if evidence.source not in {"M0_A3D", "OBS_A3D"}:
        return False, "LAST_KNOWN_ONLY"
    if not wrist_calibration_contract_ok or not bundle_contract_ok:
        return False, "NO_CENTER_CONTRACT"
    frame_ok, frame_reason = target_frame_center_state(
        evidence.source_frame_id,
        require_canonical=require_canonical_frame,
    )
    if not frame_ok:
        return False, frame_reason
    if evidence.provenance_complete is not True:
        return False, "PROVENANCE_INCOMPLETE"
    if (
        expected_calibration_id
        and evidence.calibration_id
        and evidence.calibration_id != expected_calibration_id
    ):
        return False, "CALIBRATION_ID_MISMATCH"
    if (
        require_canonical_frame
        and evidence.raw_to_canonical_alias_status
        and "pending" in str(evidence.raw_to_canonical_alias_status).lower()
    ):
        return False, "PHYSICAL_AXES_PENDING"
    return True, "OK"


@dataclass(frozen=True)
class TargetSnapshot:
    """Immutable diagnostic record created by the first click; never a command."""

    snapshot_id: str
    snapshot_hash: str
    artifact_schema: str
    target_id: str
    detection_id: str
    source_stamp_ns: int
    source_frame_id: str
    provenance_hash: str
    created_monotonic: float
    locked: LockedButton1
    m0_status: str


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Single-frame, display-only capture from a strictly matched raw pair.

    Unlike ``TargetSnapshot`` (an M0 freeze), this record is sealed without
    any M0 fresh-READY requirement.  It is intentionally impossible to confuse
    with a motion-capable freeze: it has its own artifact schema, an explicit
    ``diagnostic_only=True`` marker, and every motion consumer (prepare,
    move, precontact gate) treats it as permanently BLOCKED.  It never becomes
    an ``m0_demo_freeze_v1`` package, never acquires a ticket, and never sets
    ``execution_ready``/``motion_authorized``.
    """

    snapshot_id: str
    snapshot_hash: str
    artifact_schema: str
    diagnostic_only: bool
    target_id: str
    detection_id: str
    source_stamp_ns: int
    source_frame_id: str
    provenance_hash: str
    created_monotonic: float
    locked: LockedButton1
    m0_status: str
    source_frame_raw: str | None = None
    canonical_frame_id: str | None = None
    section_b_calibration_id: str | None = None
    provenance_complete: bool | None = None
    raw_to_canonical_alias_id: str | None = None
    raw_to_canonical_alias_status: str | None = None


@dataclass(frozen=True)
class GateResult:
    phase: LivePhase
    enabled: bool
    reason: str


def freeze_button1(
    summary: Mapping[str, Any],
    *,
    now: float | None = None,
    m0_receive_age_s: float | None,
    max_m0_receive_age_s: float = M0_RECEIVE_MAX_AGE_S,
    target_label: str = TARGET_LABEL,
    detection_id: str | None = None,
    semantic_track_id: str | None = None,
) -> TargetSnapshot:
    """Freeze only a fresh geometrically READY M0 Button-1 record.

    The receive-age check is deliberately fail-closed.  A retained M0 message
    remains useful for display, but cannot create a new frozen evidence record.
    """
    if not m0_is_fresh_ready(
        summary,
        receive_age_s=m0_receive_age_s,
        max_receive_age_s=max_m0_receive_age_s,
    ):
        age_text = "missing" if m0_receive_age_s is None else f"{float(m0_receive_age_s):.2f}s"
        raise ValueError(f"M0 is not fresh READY (receive age {age_text})")
    header = summary.get("header")
    if not isinstance(header, EvidenceHeader) or header.stamp_ns <= 0 or not header.frame_id:
        raise ValueError("M0 target provenance header is missing")
    current_detection_id = str(summary.get("detection_id") or "").strip()
    if not current_detection_id:
        raise ValueError("M0 target provenance detection_id is missing")
    locked = lock_button1_from_m0_summary(
        summary, target_label=target_label, detection_id=detection_id,
        semantic_track_id=semantic_track_id,
    )
    detection_id = current_detection_id
    stamp = time.monotonic() if now is None else float(now)
    payload = {
        "schema": DIAGNOSTIC_TARGET_ARTIFACT_SCHEMA,
        "target_id": f"button{locked.label}:{header.frame_id}:{header.stamp_ns}:{detection_id}",
        "detection_id": detection_id,
        "semantic_track_id": get_semantic_track_id(summary),
        "source_stamp_ns": header.stamp_ns,
        "source_frame_id": header.frame_id,
        "label": locked.label,
        "camera_xyz_m": locked.camera_xyz_m,
        "normal_camera": locked.normal_camera,
        "bbox": locked.bbox,
        "source_status": locked.source_status,
        "confirmations": locked.confirmations,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return TargetSnapshot(
        snapshot_id=f"button{locked.label}-{digest[:12]}",
        snapshot_hash=digest,
        artifact_schema=DIAGNOSTIC_TARGET_ARTIFACT_SCHEMA,
        target_id=str(payload["target_id"]),
        detection_id=detection_id,
        source_stamp_ns=header.stamp_ns,
        source_frame_id=header.frame_id,
        provenance_hash=digest,
        created_monotonic=stamp,
        locked=locked,
        m0_status=str(summary.get("status") or ""),
    )


def diagnostic_capture_from_raw_pair(
    raw_pair: Mapping[str, Any] | None,
    *,
    now: float | None = None,
    raw_pair_receive_age_s: float | None,
    max_raw_pair_age_s: float = DIAGNOSTIC_RAW_PAIR_MAX_AGE_S,
    m0_status: str = "NOT_READY",
    target_label: str = TARGET_LABEL,
    detection_id: str | None = None,
    semantic_track_id: str | None = None,
) -> DiagnosticSnapshot:
    """Freeze one *current* matched raw pair as a display-only diagnostic.

    This is the single-frame diagnostic capture path.  It accepts only an
    evidence pair produced by ``raw_button1_pair`` with an exact
    semantics/observation header match, a unique label-1 detection, a
    non-empty detection_id, valid depth/normal/bbox/xyz, and a receive age
    inside the explicit short TTL (``DIAGNOSTIC_RAW_PAIR_MAX_AGE_S`` by
    default).  It never
    stitches headers and never treats a retained last-known pair as current --
    the caller must only pass the pair while it is the live join.

    The result is a ``DiagnosticSnapshot`` with ``diagnostic_only=True`` and
    its own schema; it is never an ``m0_demo_freeze_v1`` package and never
    creates a ticket.  Failures raise ``ValueError`` with a stable reason so
    the view can preserve the existing M0 failure reporting.
    """
    if raw_pair is None:
        raise ValueError("no raw Button-1 pair available for diagnostic capture")
    error = raw_pair.get("error")
    if error:
        raise ValueError(f"raw Button-1 pair error: {error}")
    observation_error = raw_pair.get("observation_error")
    if observation_error:
        raise ValueError(f"raw Button-1 pair observation error: {observation_error}")
    if raw_pair.get("matched") is not True:
        raise ValueError("raw Button-1 pair is not matched")
    if raw_pair_receive_age_s is None:
        raise ValueError("raw pair receive age is missing; not a current pair")
    age = max(0.0, float(raw_pair_receive_age_s))
    if age > float(max_raw_pair_age_s):
        raise ValueError(
            f"raw pair is stale (receive age {age:.2f}s > TTL {float(max_raw_pair_age_s):.2f}s)"
        )
    current_detection_id = str(raw_pair.get("detection_id") or "").strip()
    if not current_detection_id:
        raise ValueError("raw Button-1 pair detection_id is missing")
    if not target_identity_matches(
        raw_pair, target_label=target_label, detection_id=detection_id,
        semantic_track_id=semantic_track_id,
    ):
        raise ValueError("raw pair selected label/identity mismatch")
    if raw_pair.get("pressable") is not True:
        raise ValueError("raw pair selected button is not confirmed pressable")
    detection_id = current_detection_id
    semantics_header = raw_pair.get("semantics_header")
    observations_header = raw_pair.get("observations_header")
    if (
        not isinstance(semantics_header, EvidenceHeader)
        or not isinstance(observations_header, EvidenceHeader)
        or semantics_header != observations_header
    ):
        raise ValueError("raw Button-1 pair headers do not match exactly")
    if semantics_header.stamp_ns <= 0 or not semantics_header.frame_id:
        raise ValueError("raw Button-1 pair header stamp/frame is missing")
    if raw_pair.get("depth_valid") is not True:
        raise ValueError("raw Button-1 observation depth_valid is false")
    if raw_pair.get("normal_valid") is not True:
        raise ValueError("raw Button-1 surface normal estimate_valid is false")
    xyz = raw_pair.get("observation_3d")
    normal = raw_pair.get("observation_normal")
    bbox = raw_pair.get("observation_bbox")
    try:
        xyz_t = tuple(float(value) for value in xyz)
        normal_t = tuple(float(value) for value in normal)
        bbox_t = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"raw Button-1 geometry is invalid: {exc}") from exc
    if len(xyz_t) != 3 or not all(math.isfinite(value) for value in xyz_t):
        raise ValueError("raw Button-1 camera_xyz must be a finite 3-vector")
    if len(normal_t) != 3 or not all(math.isfinite(value) for value in normal_t):
        raise ValueError("raw Button-1 normal must be a finite 3-vector")
    if len(bbox_t) != 4 or not all(math.isfinite(value) for value in bbox_t):
        raise ValueError("raw Button-1 bbox must be a finite 4-vector")
    if bbox_t[2] <= bbox_t[0] or bbox_t[3] <= bbox_t[1]:
        raise ValueError("raw Button-1 bbox is invalid")
    locked = LockedButton1(
        label=normalize_button_label(target_label),
        camera_xyz_m=xyz_t,
        normal_camera=normal_t,
        bbox=bbox_t,
        source_status="SEMANTICS_OBS_JOIN",
        confirmations=1,
    )
    stamp = time.monotonic() if now is None else float(now)
    frame_id = str(observations_header.frame_id)
    target_id = f"button{locked.label}:{frame_id}:{semantics_header.stamp_ns}:{detection_id}"
    source_frame_raw = str(raw_pair.get("source_frame_id") or "") or None
    canonical_frame_id = str(raw_pair.get("canonical_frame_id") or "") or None
    section_b_calibration_id = str(raw_pair.get("section_b_calibration_id") or "") or None
    raw_to_canonical_alias_id = str(raw_pair.get("raw_to_canonical_alias_id") or "") or None
    raw_to_canonical_alias_status = str(raw_pair.get("raw_to_canonical_alias_status") or "") or None
    payload = {
        "schema": BUTTON1_DIAGNOSTIC_SNAPSHOT_SCHEMA,
        "diagnostic_only": True,
        "target_id": target_id,
        "detection_id": detection_id,
        "source_stamp_ns": semantics_header.stamp_ns,
        "source_frame_id": frame_id,
        "label": locked.label,
        "semantic_track_id": get_semantic_track_id(raw_pair),
        "camera_xyz_m": xyz_t,
        "normal_camera": normal_t,
        "bbox": bbox_t,
        "m0_status": str(m0_status or ""),
        "source_frame_raw": source_frame_raw,
        "canonical_frame_id": canonical_frame_id,
        "section_b_calibration_id": section_b_calibration_id,
        "provenance_complete": raw_pair.get("provenance_complete"),
        "raw_to_canonical_alias_id": raw_to_canonical_alias_id,
        "raw_to_canonical_alias_status": raw_to_canonical_alias_status,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return DiagnosticSnapshot(
        snapshot_id=f"button{locked.label}-diag-{digest[:12]}",
        snapshot_hash=digest,
        artifact_schema=BUTTON1_DIAGNOSTIC_SNAPSHOT_SCHEMA,
        diagnostic_only=True,
        target_id=target_id,
        detection_id=detection_id,
        source_stamp_ns=int(semantics_header.stamp_ns),
        source_frame_id=frame_id,
        provenance_hash=digest,
        created_monotonic=stamp,
        locked=locked,
        m0_status=str(m0_status or ""),
        source_frame_raw=source_frame_raw,
        canonical_frame_id=canonical_frame_id,
        section_b_calibration_id=section_b_calibration_id,
        provenance_complete=raw_pair.get("provenance_complete"),
        raw_to_canonical_alias_id=raw_to_canonical_alias_id,
        raw_to_canonical_alias_status=raw_to_canonical_alias_status,
    )


def evaluate_precontact_gate(
    snapshot: TargetSnapshot | None,
    *,
    execution_ready: bool,
    action_server_ready: bool,
    feedback_fresh: bool,
    robot_at_standby: bool,
    foreign_joint_target_publishers: int,
    max_snapshot_age_s: float = 2.0,
    now: float | None = None,
) -> GateResult:
    """Fail closed; Live never converts a green result into a motor command."""
    if isinstance(snapshot, DiagnosticSnapshot):
        return GateResult(
            LivePhase.DISABLED,
            False,
            "diagnostic snapshot is display-only; NO MOTION TICKET",
        )
    if snapshot is None:
        return GateResult(LivePhase.TARGET_LIVE, False, "LOCK 1 first")
    stamp = time.monotonic() if now is None else float(now)
    if stamp - snapshot.created_monotonic > max_snapshot_age_s:
        return GateResult(LivePhase.TARGET_FROZEN, False, "frozen target expired; LOCK 1 again")
    if not execution_ready:
        return GateResult(LivePhase.DISABLED, False, "execution_ready=false")
    if not action_server_ready:
        return GateResult(LivePhase.DISABLED, False, "pre-contact action server unavailable")
    if not feedback_fresh:
        return GateResult(LivePhase.TARGET_FROZEN, False, "safe feedback stale")
    if not robot_at_standby:
        return GateResult(LivePhase.TARGET_FROZEN, False, "robot is not verified at standby")
    if foreign_joint_target_publishers:
        return GateResult(
            LivePhase.TARGET_FROZEN,
            False,
            f"{foreign_joint_target_publishers} joint_target publisher(s) active",
        )
    return GateResult(
        LivePhase.READY_TO_PRECONTACT,
        True,
        "request may be sent only to the one-shot pre-contact action server",
    )


def _ui_short(text: str, limit: int = 80) -> str:
    value = str(text).replace("—", "-").replace("–", "-").replace("→", ">")
    value = value.encode("ascii", "replace").decode("ascii")
    return value[:limit]


def torque_diagnostic_state(
    feedback: Mapping[str, Any] | None,
    received_at: float | None,
    now: float,
    *,
    max_age_s: float = SAFE_FEEDBACK_MAX_AGE_S,
    joint_names: Sequence[str] = SAFE_JOINT_AXES,
) -> tuple[str, list[str], str]:
    """Classify safe-driver feedback for display, never for a torque gate.

    The result is advisory UI evidence only.  It intentionally has no side
    channel back into the safe driver: an abnormal result never requests
    torque, never clears a fault, and never labels itself as a recognition
    failure.
    """
    if feedback is None or received_at is None:
        return "UNKNOWN", [], "no safe-driver feedback received yet"
    age = max(0.0, float(now) - float(received_at))
    if age > max_age_s:
        return "STALE", [], f"safe-driver feedback is stale ({age:.1f}s)"
    fault = str(feedback.get("fault") or "").strip()
    if fault:
        return "FAULTED", [], f"safe-driver fault: {_ui_short(fault)}"
    flags = feedback.get("torque_enabled")
    if not isinstance(flags, (list, tuple)) or len(flags) != len(joint_names):
        return "TORQUE_FLAGS_UNKNOWN", [], "torque_enabled is missing or malformed"
    off = [
        str(name)
        for name, flag in zip(joint_names, flags)
        if flag is not True
    ]
    if off:
        return (
            "TORQUE_FLAGS_OFF",
            off,
            f"{len(off)} of {len(joint_names)} torque flags off: {','.join(off)}",
        )
    return "OK", [], f"{len(joint_names)}/{len(joint_names)} torque flags on"


class TorqueDiagnosticLatch:
    """Sticky display-only torque diagnostic latch.

    Once any abnormal feedback state (flags off, stale, faulted, or unknown
    flags) is observed, the diagnostic stays visible until the process is
    restarted.  A later healthy sample never silently clears the problem and
    nothing here ever asks the safe driver to change torque.
    """

    _DIAGNOSTIC_STATES = frozenset(
        {"TORQUE_FLAGS_OFF", "TORQUE_FLAGS_UNKNOWN", "STALE", "FAULTED"}
    )

    def __init__(
        self,
        *,
        max_age_s: float = SAFE_FEEDBACK_MAX_AGE_S,
        joint_names: Sequence[str] = SAFE_JOINT_AXES,
    ) -> None:
        self._max_age_s = float(max_age_s)
        self._joint_names = tuple(str(name) for name in joint_names)
        self._feedback: Mapping[str, Any] | None = None
        self._received_at: float | None = None
        self._latched = False
        self._latched_state = "OK"
        self._latched_joints: list[str] = []
        self._latched_detail = ""

    def observe(
        self,
        feedback: Mapping[str, Any] | None,
        received_at: float,
        now: float,
    ) -> dict[str, Any]:
        """Record one feedback sample and return the display snapshot."""
        self._feedback = feedback
        self._received_at = received_at
        state, joints, detail = torque_diagnostic_state(
            self._feedback,
            self._received_at,
            now,
            max_age_s=self._max_age_s,
            joint_names=self._joint_names,
        )
        if state in self._DIAGNOSTIC_STATES:
            self._latched = True
            self._latched_state = state
            self._latched_joints = list(joints)
            self._latched_detail = detail
        return self.snapshot(now)

    def snapshot(self, now: float) -> dict[str, Any]:
        """Return the current display state, preserving any latched problem."""
        state, joints, detail = torque_diagnostic_state(
            self._feedback,
            self._received_at,
            now,
            max_age_s=self._max_age_s,
            joint_names=self._joint_names,
        )
        latched = self._latched
        if latched and state == "OK":
            state = self._latched_state
            joints = list(self._latched_joints)
            detail = self._latched_detail
        diagnostic = state != "OK"
        label = (
            "TORQUE DIAGNOSTIC ONLY - not a recognition failure"
            if diagnostic
            else "OK"
        )
        if latched and diagnostic:
            detail = f"{detail} (LATCHED - restart viewer to clear)"
        return {
            "state": state,
            "off_joints": list(joints),
            "detail": detail,
            "latched": latched,
            "diagnostic": diagnostic,
            "label": label,
        }
