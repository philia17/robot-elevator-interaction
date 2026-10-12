#!/usr/bin/env python3
"""Headless one-shot M0 target freeze for the e2e demo path (NO MOTION).

Subscribes to ``/elevator/m0/perception_target_3d`` and writes a frozen-target
JSON compatible with ``m0_target_to_base`` / ``m3_camera_route``.

Two capture modes:

- Formal (default): waits for ``--stable-frames`` (default 5) consecutive READY
  M0 messages and applies the strict READY gate (status READY,
  perception_ready=true, exact Button-1 label/effective_label,
  motion_authorized=false, valid normal, valid surface unless
  ``--allow-invalid-surface``).
- Diagnostic one-shot (``--single-frame-diagnostic``): latches the first
  *current* exact raw Button-1 observation+semantic pair (same header stamp
  and frame, same detection_id, unique label, valid depth/normal/bbox/XYZ,
  receive age inside the bounded TTL).  It never waits for ``--stable-frames``
  and does not require the M0 aggregate to be READY; STABILIZING /
  SEMANTIC_SET_INCOMPLETE / CACHED aggregate states therefore cannot block a
  diagnostic snapshot.  The artifact is clearly marked diagnostic_only /
  no_motion (execution_ready=false, no route/trajectory/command fields) and is
  intended for stationary-scene snapshot debugging only.  It never enables
  motion.

Does not open serial, arm services, or publishers. Does not require a display.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import rclpy
from elevator_interfaces.msg import (
    ButtonObservation3DArray,
    ButtonSemanticArray,
    M0Target3D,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from press.precontact_live_core import (
    DIAGNOSTIC_RAW_PAIR_MAX_AGE_S,
    EvidenceHeader,
    raw_button1_pair,
)


DEFAULT_TARGET_TOPIC = "/elevator/m0/perception_target_3d"
DEFAULT_OBSERVATIONS_TOPIC = "/elevator/button_observations_3d"
DEFAULT_SEMANTICS_TOPIC = "/elevator/button_semantics"
DEFAULT_EVIDENCE_ROOT = (
    "/data/lift_robot_calibration_evidence/m0_demo_freeze_trials"
)

# Diagnostic one-shot gates (fail-closed; never loosened by CLI flags).
DIAGNOSTIC_MAX_SEMANTIC_AGE_SEC = 2.0
DIAGNOSTIC_MIN_DEPTH_VALID_RATIO = 0.5
MIN_NORMAL_NORM = 1e-6

# Machine-readable failure codes.
FAIL_NO_M0_READY = "NO_M0_READY"
FAIL_STALE_TARGET = "STALE_TARGET"
FAIL_INVALID_NORMAL = "INVALID_NORMAL"
FAIL_TARGET_IDENTITY_MISMATCH = "TARGET_IDENTITY_MISMATCH"
FAIL_OUTPUT_EXISTS = "OUTPUT_EXISTS"
FAIL_HEADER_MISMATCH = "HEADER_MISMATCH"

FORMAL_SCHEMA = "m0_demo_freeze_v1"
DIAGNOSTIC_SCHEMA = "m0_single_frame_diagnostic_v1"
DIAGNOSTIC_MODE = "single_frame_diagnostic"


def _stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _evidence_header(message) -> EvidenceHeader:
    return EvidenceHeader(
        stamp_ns=_stamp_ns(message),
        frame_id=str(message.header.frame_id),
    )


def _semantic_dict(item) -> dict:
    metadata = {}
    raw_response = str(getattr(item, "raw_response", "") or "")
    if raw_response:
        try:
            metadata = json.loads(raw_response)
        except (TypeError, ValueError):
            metadata = {}
    return {
        "detection_id": str(getattr(item, "detection_id", "") or ""),
        "label": str(getattr(item, "label", "") or ""),
        "pressable": bool(getattr(item, "pressable", False)),
        "track_id": str(metadata.get("track_id", "") or ""),
        "semantic_status": str(metadata.get("semantic_status", "") or ""),
        "semantic_age_sec": float(metadata.get("semantic_age_sec", math.nan)),
        "association_valid": bool(metadata.get("association_valid", False)),
    }


def _observation_dict(item) -> dict:
    surface = item.surface_geometry
    return {
        "detection_id": str(getattr(item, "detection_id", "") or ""),
        "x_min": float(item.x_min),
        "y_min": float(item.y_min),
        "x_max": float(item.x_max),
        "y_max": float(item.y_max),
        "detector_confidence": float(item.detector_confidence),
        "depth_valid": bool(item.depth_valid),
        "depth_valid_ratio": float(item.depth_valid_ratio),
        "x_m": float(item.x_m),
        "y_m": float(item.y_m),
        "z_m": float(item.z_m),
        "surface_geometry": {
            "estimate_valid": bool(surface.estimate_valid),
            "provenance_complete": bool(surface.provenance_complete),
            "failure_reason": str(surface.failure_reason),
            "surface_point_x_m": float(surface.surface_point_x_m),
            "surface_point_y_m": float(surface.surface_point_y_m),
            "surface_point_z_m": float(surface.surface_point_z_m),
            "panel_outward_normal_x": float(surface.panel_outward_normal_x),
            "panel_outward_normal_y": float(surface.panel_outward_normal_y),
            "panel_outward_normal_z": float(surface.panel_outward_normal_z),
            "normal_angular_uncertainty_rad": float(
                surface.normal_angular_uncertainty_rad
            ),
            "plane_rms_residual_m": float(surface.plane_rms_residual_m),
            "plane_sample_count": int(surface.plane_sample_count),
            "geometry_method_id": str(surface.geometry_method_id),
            "section_b_calibration_id": str(surface.section_b_calibration_id),
            "source_frame_id": str(surface.source_frame_id),
            "canonical_frame_id": str(surface.canonical_frame_id),
        },
    }


def diagnostic_pair_failure(
    pair: dict | None,
    *,
    received_at: float | None,
    now: float,
    depth_valid_ratio: float = math.nan,
    source_frame_id: str = "",
    canonical_frame_id: str = "",
    semantic_age_sec: float | None = None,
    association_valid: bool | None = None,
    pressable: bool | None = None,
):
    """Return ``(failure_code, detail)`` for a raw pair, else None.

    This is the diagnostic single-frame gate.  It deliberately does not look
    at the M0 aggregate status: a fresh Button-1 observation+semantic pair is
    the unit of capture.  Header equality, unique identity, depth, normal,
    geometry, and a bounded receive age are still required so the snapshot is
    never last-known, degraded, or stitched across frames.
    """
    if pair is None:
        return FAIL_NO_M0_READY, "no_raw_pair"
    error = pair.get("error")
    if error:
        text = str(error)
        if text.startswith("HEADER_"):
            return FAIL_HEADER_MISMATCH, text
        if text.startswith(
            (
                "NO_LABEL_1",
                "AMBIGUOUS_LABEL_1",
                "LABEL_1_NO_DETECTION_ID",
                "NO_MATCHING_3D_OBSERVATION",
                "AMBIGUOUS_3D",
            )
        ):
            return FAIL_TARGET_IDENTITY_MISMATCH, text
        return FAIL_NO_M0_READY, text
    observation_error = pair.get("observation_error")
    if observation_error:
        text = str(observation_error)
        if "normal" in text or "estimate_valid" in text:
            return FAIL_INVALID_NORMAL, text
        return FAIL_NO_M0_READY, text
    if pair.get("matched") is not True:
        return FAIL_NO_M0_READY, "not_matched"
    semantics_header = pair.get("semantics_header")
    observations_header = pair.get("observations_header")
    if (
        not isinstance(semantics_header, EvidenceHeader)
        or not isinstance(observations_header, EvidenceHeader)
        or semantics_header != observations_header
        or semantics_header.stamp_ns <= 0
        or not semantics_header.frame_id
    ):
        return FAIL_STALE_TARGET, "header_stamp_or_frame_missing"
    if received_at is None:
        return FAIL_STALE_TARGET, "raw_pair_receive_age_missing"
    age = max(0.0, float(now) - float(received_at))
    if age > DIAGNOSTIC_RAW_PAIR_MAX_AGE_S:
        return FAIL_STALE_TARGET, f"raw_pair_receive_age={age:.3f}"
    if pair.get("depth_valid") is not True:
        return FAIL_NO_M0_READY, "depth_valid=false"
    try:
        ratio = float(depth_valid_ratio)
    except (TypeError, ValueError):
        ratio = math.nan
    if not math.isfinite(ratio) or ratio < DIAGNOSTIC_MIN_DEPTH_VALID_RATIO:
        return FAIL_NO_M0_READY, f"depth_valid_ratio={depth_valid_ratio!r}"
    if semantic_age_sec is None:
        semantic_age_sec = pair.get("semantic_age_sec", math.nan)
    if association_valid is None:
        association_valid = pair.get("association_valid", False)
    if pressable is None:
        pressable = pair.get("pressable", False)
    if association_valid is not True or pressable is not True:
        return (
            FAIL_TARGET_IDENTITY_MISMATCH,
            f"association_valid={association_valid!r} pressable={pressable!r}",
        )
    semantic_age = _finite_or_none(semantic_age_sec)
    if semantic_age is None or semantic_age > DIAGNOSTIC_MAX_SEMANTIC_AGE_SEC:
        return (
            FAIL_STALE_TARGET,
            f"semantic_age_sec={semantic_age_sec!r}",
        )
    normal = pair.get("observation_normal")
    try:
        nx, ny, nz = (float(value) for value in normal)
    except (TypeError, ValueError):
        return FAIL_INVALID_NORMAL, "normal_missing_or_nonfinite"
    nnorm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not all(math.isfinite(value) for value in (nx, ny, nz)) or nnorm <= MIN_NORMAL_NORM:
        return FAIL_INVALID_NORMAL, f"normal_norm={nnorm}"
    if not str(source_frame_id or "").strip() or not str(
        canonical_frame_id or ""
    ).strip():
        return FAIL_STALE_TARGET, "surface_frame_ids_empty"
    return None


def diagnostic_target_dict(
    pair: dict,
    semantic_item: dict,
    observation_item: dict,
    label: str,
    m0_status: str = "NOT_READY",
) -> dict:
    """Shape a qualified raw pair into the diagnostic frozen-target record."""
    surface = observation_item.get("surface_geometry") or {}
    normal = pair.get("observation_normal")
    xyz = pair.get("observation_3d")
    bbox = pair.get("observation_bbox")
    return {
        "status": str(m0_status or "NOT_READY"),
        "failure_reason": "",
        "perception_ready": False,
        "target_label": str(label),
        "effective_label": str(label),
        "source_stamp_ns": int(pair["semantics_header"].stamp_ns),
        "frame_id": str(pair["observations_header"].frame_id),
        "matching_label_count": 1,
        "detection_id": str(pair.get("detection_id") or ""),
        "semantic_track_id": str(semantic_item.get("track_id", "") or ""),
        "semantic_status": str(semantic_item.get("semantic_status", "") or ""),
        "semantic_age_sec": _finite_or_none(
            semantic_item.get("semantic_age_sec", math.nan)
        ),
        "confirmation_count": 1,
        "required_confirmations": 1,
        "position_spread_m": 0.0,
        "bbox": [float(value) for value in bbox],
        "camera_point_m": {
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        },
        "surface_geometry": {
            "estimate_valid": bool(surface.get("estimate_valid")),
            "provenance_complete": bool(surface.get("provenance_complete")),
            "failure_reason": str(surface.get("failure_reason", "")),
            "surface_point_m": {
                "x": float(surface.get("surface_point_x_m", xyz[0])),
                "y": float(surface.get("surface_point_y_m", xyz[1])),
                "z": float(surface.get("surface_point_z_m", xyz[2])),
            },
            "panel_outward_normal": {
                "x": float(normal[0]),
                "y": float(normal[1]),
                "z": float(normal[2]),
            },
            "normal_angular_uncertainty_rad": _finite_or_none(
                surface.get("normal_angular_uncertainty_rad", math.nan)
            ),
            "plane_rms_residual_m": _finite_or_none(
                surface.get("plane_rms_residual_m", math.nan)
            ),
            "plane_sample_count": int(surface.get("plane_sample_count", 0)),
            "geometry_method_id": str(surface.get("geometry_method_id", "")),
            "section_b_calibration_id": str(
                surface.get("section_b_calibration_id", "")
            ),
            "source_frame_id": str(surface.get("source_frame_id", "")),
            "canonical_frame_id": str(surface.get("canonical_frame_id", "")),
        },
        "detector_confidence": float(
            observation_item.get("detector_confidence", 0.0)
        ),
        "depth_valid_ratio": _finite_or_none(
            observation_item.get("depth_valid_ratio", math.nan)
        ),
        "motion_authorized": False,
    }


def target_dict(message) -> dict:
    observation = message.observation
    surface = observation.surface_geometry
    return {
        "status": str(message.status),
        "failure_reason": str(message.failure_reason),
        "perception_ready": bool(message.perception_ready),
        "target_label": str(message.target_label),
        "effective_label": str(message.effective_label),
        "source_stamp_ns": _stamp_ns(message),
        "frame_id": str(message.header.frame_id),
        "matching_label_count": int(message.matching_label_count),
        "detection_id": str(message.detection_id),
        "semantic_track_id": str(message.semantic_track_id),
        "semantic_status": str(message.semantic_status),
        "semantic_age_sec": float(message.semantic_age_sec),
        "confirmation_count": int(message.confirmation_count),
        "required_confirmations": int(message.required_confirmations),
        "position_spread_m": float(message.position_spread_m),
        "bbox": [
            float(observation.x_min),
            float(observation.y_min),
            float(observation.x_max),
            float(observation.y_max),
        ],
        "camera_point_m": {
            "x": float(observation.x_m),
            "y": float(observation.y_m),
            "z": float(observation.z_m),
        },
        "surface_geometry": {
            "estimate_valid": bool(surface.estimate_valid),
            "provenance_complete": bool(surface.provenance_complete),
            "failure_reason": str(surface.failure_reason),
            "surface_point_m": {
                "x": float(surface.surface_point_x_m),
                "y": float(surface.surface_point_y_m),
                "z": float(surface.surface_point_z_m),
            },
            "panel_outward_normal": {
                "x": float(surface.panel_outward_normal_x),
                "y": float(surface.panel_outward_normal_y),
                "z": float(surface.panel_outward_normal_z),
            },
            "normal_angular_uncertainty_rad": float(
                surface.normal_angular_uncertainty_rad
            ),
            "plane_rms_residual_m": float(surface.plane_rms_residual_m),
            "plane_sample_count": int(surface.plane_sample_count),
            "geometry_method_id": str(surface.geometry_method_id),
            "section_b_calibration_id": str(surface.section_b_calibration_id),
            "source_frame_id": str(surface.source_frame_id),
            "canonical_frame_id": str(surface.canonical_frame_id),
        },
        "detector_confidence": float(observation.detector_confidence),
        "depth_valid_ratio": float(observation.depth_valid_ratio),
        "motion_authorized": bool(message.motion_authorized),
    }


def _ready(target: dict, label: str) -> bool:
    """Formal READY gate: unchanged from the original capture utility."""
    if not target:
        return False
    normal = (target.get("surface_geometry") or {}).get("panel_outward_normal") or {}
    nx = float(normal.get("x", 0.0))
    ny = float(normal.get("y", 0.0))
    nz = float(normal.get("z", 0.0))
    nnorm = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (
        target.get("status") == "READY"
        and bool(target.get("perception_ready"))
        and str(target.get("target_label", "")).strip() == label
        and str(target.get("effective_label", "")).strip() == label
        and target.get("motion_authorized") is False
        and nnorm > MIN_NORMAL_NORM
    )


def reject_reason(
    target: dict,
    label: str,
    *,
    diagnostic: bool,
    allow_invalid_surface: bool = False,
):
    """Return ``(failure_code, detail)`` if target must not freeze, else None.

    Formal mode classifies only the existing formal gates, so timeout/failure
    output can distinguish the named codes without tightening or loosening the
    formal behavior.  The ``diagnostic=True`` branch here is the legacy
    M0-shaped gate; the headless diagnostic node now latches raw synchronized
    observation+semantic pairs through ``diagnostic_pair_failure`` instead.
    No gate is ever loosened by CLI flags.
    """
    surface = target.get("surface_geometry") or {}
    normal = surface.get("panel_outward_normal") or {}

    if str(target.get("status", "")).strip() != "READY":
        return FAIL_NO_M0_READY, f"status={target.get('status')!r}"
    if not bool(target.get("perception_ready")):
        return FAIL_NO_M0_READY, "perception_ready=false"
    if target.get("motion_authorized") is not False:
        return FAIL_NO_M0_READY, "motion_authorized=true"

    if (
        str(target.get("target_label", "")).strip() != label
        or str(target.get("effective_label", "")).strip() != label
    ):
        return (
            FAIL_TARGET_IDENTITY_MISMATCH,
            f"target_label={target.get('target_label')!r} "
            f"effective_label={target.get('effective_label')!r}",
        )

    if diagnostic:
        detection_id = str(target.get("detection_id", "")).strip()
        track_id = str(target.get("semantic_track_id", "")).strip()
        if not detection_id or not track_id:
            return (
                FAIL_TARGET_IDENTITY_MISMATCH,
                f"detection_id={detection_id!r} semantic_track_id={track_id!r}",
            )
        if int(target.get("matching_label_count", 0)) != 1:
            return (
                FAIL_TARGET_IDENTITY_MISMATCH,
                f"matching_label_count={target.get('matching_label_count')!r}",
            )

    if diagnostic:
        semantic_status = str(target.get("semantic_status", "")).strip()
        if semantic_status != "FRESH":
            return (
                FAIL_STALE_TARGET,
                f"semantic_status={semantic_status or 'EMPTY'}",
            )
        try:
            age = float(target.get("semantic_age_sec"))
        except (TypeError, ValueError):
            age = math.nan
        if not math.isfinite(age) or age > DIAGNOSTIC_MAX_SEMANTIC_AGE_SEC:
            return (
                FAIL_STALE_TARGET,
                f"semantic_age_sec={target.get('semantic_age_sec')!r}",
            )
        try:
            stamp_ns = int(target.get("source_stamp_ns") or 0)
        except (TypeError, ValueError):
            stamp_ns = 0
        if stamp_ns <= 0:
            return FAIL_STALE_TARGET, "source_stamp_ns_missing"
        if not str(target.get("frame_id", "")).strip():
            return FAIL_STALE_TARGET, "frame_id_empty"
        if not str(surface.get("source_frame_id", "")).strip() or not str(
            surface.get("canonical_frame_id", "")
        ).strip():
            return FAIL_STALE_TARGET, "surface_frame_ids_empty"

    nx = float(normal.get("x", 0.0))
    ny = float(normal.get("y", 0.0))
    nz = float(normal.get("z", 0.0))
    nnorm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not all(math.isfinite(v) for v in (nx, ny, nz)) or nnorm <= MIN_NORMAL_NORM:
        return FAIL_INVALID_NORMAL, f"normal_norm={nnorm}"

    if diagnostic:
        if not bool(surface.get("estimate_valid")):
            return FAIL_INVALID_NORMAL, "surface_estimate_invalid"
        # The live M0 stream is explicitly a no-motion diagnostic source and
        # may report provenance_complete=false while still carrying a valid
        # observed surface point/normal.  Keep that fact in the frozen record,
        # but do not make it a second diagnostic identity/depth/normal gate.
        # Formal execution/release validators remain responsible for rejecting
        # incomplete provenance; this utility never authorizes motion.
        try:
            ratio = float(target.get("depth_valid_ratio"))
        except (TypeError, ValueError):
            ratio = math.nan
        if not math.isfinite(ratio) or ratio < DIAGNOSTIC_MIN_DEPTH_VALID_RATIO:
            return (
                FAIL_NO_M0_READY,
                f"depth_valid_ratio={target.get('depth_valid_ratio')!r}",
            )
        point = target.get("camera_point_m") or {}
        for axis in ("x", "y", "z"):
            try:
                value = float(point.get(axis))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                return FAIL_NO_M0_READY, f"camera_point_{axis}_not_finite"
        bbox = target.get("bbox") or []
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
            return FAIL_NO_M0_READY, "bbox_invalid"
        if float(bbox[0]) >= float(bbox[2]) or float(bbox[1]) >= float(bbox[3]):
            return FAIL_NO_M0_READY, "bbox_degenerate"
        try:
            confirmations = int(target.get("confirmation_count", 0))
            required = int(target.get("required_confirmations", 0))
        except (TypeError, ValueError):
            return FAIL_NO_M0_READY, "confirmation_counts_invalid"
        if confirmations < required:
            return (
                FAIL_NO_M0_READY,
                f"confirmation_count={confirmations}/{required}",
            )

    if (
        not diagnostic
        and not allow_invalid_surface
        and not bool(surface.get("estimate_valid"))
    ):
        return FAIL_NO_M0_READY, "surface_estimate_invalid"

    return None


class HeadlessFreezeNode(Node):
    def __init__(
        self,
        topic: str,
        label: str,
        *,
        diagnostic: bool = False,
        allow_invalid_surface: bool = False,
        observations_topic: str = DEFAULT_OBSERVATIONS_TOPIC,
        semantics_topic: str = DEFAULT_SEMANTICS_TOPIC,
    ):
        super().__init__("headless_m0_freeze")
        self.label = label
        self.diagnostic = diagnostic
        self.allow_invalid_surface = allow_invalid_surface
        self.latest: dict | None = None
        self._latest_observations = None
        self._latest_semantics = None
        self._latest_observations_at = None
        self._latest_semantics_at = None
        self.ready_count = 0
        self.diagnostic_accepted: dict | None = None
        self.last_reject_code: str | None = None
        self.last_reject_reason: str | None = None
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(M0Target3D, topic, self._on_target, qos)
        if self.diagnostic:
            self.create_subscription(
                ButtonObservation3DArray,
                observations_topic,
                self._on_observations,
                qos,
            )
            self.create_subscription(
                ButtonSemanticArray,
                semantics_topic,
                self._on_semantics,
                qos,
            )

    def _on_target(self, message: M0Target3D) -> None:
        self.latest = target_dict(message)
        if self.diagnostic:
            return
        if _ready(self.latest, self.label):
            self.ready_count += 1
        else:
            self.ready_count = 0
        rejected = reject_reason(
            self.latest,
            self.label,
            diagnostic=False,
            allow_invalid_surface=self.allow_invalid_surface,
        )
        if rejected is not None:
            self.last_reject_code, self.last_reject_reason = rejected
        else:
            self.last_reject_code = None
            self.last_reject_reason = None

    def _on_observations(self, message: ButtonObservation3DArray) -> None:
        self._latest_observations = message
        self._latest_observations_at = time.monotonic()
        self._try_diagnostic_pair()

    def _on_semantics(self, message: ButtonSemanticArray) -> None:
        self._latest_semantics = message
        self._latest_semantics_at = time.monotonic()
        self._try_diagnostic_pair()

    def _try_diagnostic_pair(self) -> None:
        if (
            not self.diagnostic
            or self.diagnostic_accepted is not None
            or self._latest_observations is None
            or self._latest_semantics is None
        ):
            return
        observations_header = _evidence_header(self._latest_observations)
        semantics_header = _evidence_header(self._latest_semantics)
        observations = [
            _observation_dict(item)
            for item in self._latest_observations.observations
        ]
        semantics = [
            _semantic_dict(item) for item in self._latest_semantics.semantics
        ]
        pair = raw_button1_pair(
            semantics,
            observations,
            semantics_header=semantics_header,
            observations_header=observations_header,
        )
        now = time.monotonic()
        received_at = max(
            self._latest_observations_at or 0.0,
            self._latest_semantics_at or 0.0,
        )
        detection_id = str(pair.get("detection_id") or "")
        observation_item = next(
            (
                item
                for item in observations
                if str(item.get("detection_id", "")) == detection_id
            ),
            {},
        )
        semantic_item = next(
            (
                item
                for item in semantics
                if str(item.get("detection_id", "")) == detection_id
                and str(item.get("label", "")).strip().upper() == self.label
            ),
            {},
        )
        surface = observation_item.get("surface_geometry") or {}
        failure = diagnostic_pair_failure(
            pair,
            received_at=received_at,
            now=now,
            depth_valid_ratio=observation_item.get(
                "depth_valid_ratio", math.nan
            ),
            source_frame_id=str(surface.get("source_frame_id", "") or ""),
            canonical_frame_id=str(surface.get("canonical_frame_id", "") or ""),
            semantic_age_sec=semantic_item.get(
                "semantic_age_sec", math.nan
            ),
            association_valid=semantic_item.get(
                "association_valid", False
            ),
            pressable=semantic_item.get("pressable", False),
        )
        if failure is not None:
            self.last_reject_code, self.last_reject_reason = failure
            return
        m0_status = (
            str(self.latest.get("status") or "NOT_READY")
            if self.latest is not None
            else "NO_M0_MESSAGE"
        )
        self.diagnostic_accepted = diagnostic_target_dict(
            pair,
            semantic_item,
            observation_item,
            self.label,
            m0_status,
        )
        self.last_reject_code = None
        self.last_reject_reason = None


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-topic", default=DEFAULT_TARGET_TOPIC)
    parser.add_argument(
        "--observations-topic",
        default=DEFAULT_OBSERVATIONS_TOPIC,
        help=(
            "Raw Button-1 observation stream used by --single-frame-diagnostic"
        ),
    )
    parser.add_argument(
        "--semantics-topic",
        default=DEFAULT_SEMANTICS_TOPIC,
        help=(
            "Raw button semantic stream used by --single-frame-diagnostic"
        ),
    )
    parser.add_argument("--target-label", default="1")
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=5,
        help="Consecutive READY frames required before freeze",
    )
    parser.add_argument(
        "--single-frame-diagnostic",
        action="store_true",
        help=(
            "Opt-in diagnostic one-shot: latch the first current exact raw "
            "Button-1 observation+semantic pair whose header/frame, identity, "
            "depth, normal, and geometry checks pass. Does not require M0 "
            "READY and never waits for --stable-frames; output is marked "
            "diagnostic_only/no_motion."
        ),
    )
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--evidence-root", default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit output path; otherwise evidence_root/trial_dir",
    )
    parser.add_argument(
        "--allow-invalid-surface",
        action="store_true",
        help=(
            "Accept READY targets even when surface estimate_valid is false "
            "(ignored in --single-frame-diagnostic mode)"
        ),
    )
    return parser.parse_args(argv)


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def with_canonical_hash(payload: dict) -> dict:
    stamped = dict(payload)
    stamped["canonical_payload_sha256"] = hashlib.sha256(
        canonical_bytes(stamped)
    ).hexdigest()
    return stamped


def _build_payload(args: argparse.Namespace, frozen: dict, diagnostic: bool):
    saved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if diagnostic:
        trial_id = (
            f"m0_single_frame_diagnostic_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        payload = {
            "schema": DIAGNOSTIC_SCHEMA,
            "trial_id": trial_id,
            "mode": DIAGNOSTIC_MODE,
            "diagnostic_only": True,
            "no_motion": True,
            "execution_ready": False,
            "motion_authorized": False,
            "target_label": args.target_label,
            "source_stamp_ns": frozen["source_stamp_ns"],
            "source_frame_id": frozen["frame_id"],
            "detection_id": frozen["detection_id"],
            "semantic_track_id": frozen["semantic_track_id"],
            "semantic_status": frozen.get("semantic_status", ""),
            "semantic_age_sec": frozen.get("semantic_age_sec"),
            "m0_status": frozen["status"],
            "source_status": frozen.get("source_status", ""),
            "confirmation_count": frozen["confirmation_count"],
            "required_confirmations": frozen["required_confirmations"],
            "frozen_target": frozen,
            "saved_at": saved_at,
        }
        return trial_id, payload
    trial_id = f"m0_demo_freeze_{time.strftime('%Y%m%d_%H%M%S')}"
    payload = {
        "schema": FORMAL_SCHEMA,
        "trial_id": trial_id,
        "mode": "headless_one_shot",
        "motion_authorized": False,
        "target_label": args.target_label,
        "frozen_target": frozen,
        "saved_at": saved_at,
    }
    return trial_id, payload


def _print_failure(code: str, reason: str) -> None:
    print("RESULT=FAILURE", flush=True)
    print(f"freeze_failure={code}", flush=True)
    print(f"last_reject_reason={reason}", flush=True)


def _timeout_failure(node, args: argparse.Namespace, diagnostic: bool):
    if diagnostic:
        if node.last_reject_code is not None:
            return node.last_reject_code, node.last_reject_reason or ""
        if (
            node.latest is None
            and node._latest_observations is None
            and node._latest_semantics is None
        ):
            return FAIL_NO_M0_READY, "no_m0_or_raw_pair"
        return FAIL_NO_M0_READY, "timeout_no_qualified_raw_pair"
    latest = node.latest
    if latest is None:
        return FAIL_NO_M0_READY, "no_m0_message"
    rejected = reject_reason(
        latest,
        args.target_label,
        diagnostic=diagnostic,
        allow_invalid_surface=args.allow_invalid_surface,
    )
    if rejected is not None:
        return rejected
    if not diagnostic and node.ready_count < args.stable_frames:
        return (
            FAIL_NO_M0_READY,
            f"insufficient_stable_frames={node.ready_count}/{args.stable_frames}",
        )
    return FAIL_NO_M0_READY, "timeout"


def main(argv=None) -> int:
    args = parse_args(argv)
    diagnostic = bool(args.single_frame_diagnostic)
    print("=== Headless M0 freeze (NO MOTION) ===", flush=True)
    print(f"target_topic={args.target_topic}", flush=True)
    print(f"target_label={args.target_label}", flush=True)
    print(f"mode={DIAGNOSTIC_MODE if diagnostic else 'formal'}", flush=True)
    if diagnostic:
        print("stable_frames=ignored", flush=True)
        print(
            "diagnostic_source=raw_observation_semantic_pair",
            flush=True,
        )
        if args.allow_invalid_surface:
            print(
                "note=--allow-invalid-surface ignored in "
                "single-frame-diagnostic mode",
                flush=True,
            )
    else:
        print(f"stable_frames={args.stable_frames}", flush=True)
    print("motion_authorized=false", flush=True)

    rclpy.init()
    node = HeadlessFreezeNode(
        args.target_topic,
        args.target_label,
        diagnostic=diagnostic,
        allow_invalid_surface=args.allow_invalid_surface,
        observations_topic=args.observations_topic,
        semantics_topic=args.semantics_topic,
    )
    deadline = time.monotonic() + args.timeout_sec
    frozen = None
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if diagnostic:
                if node.diagnostic_accepted is not None:
                    frozen = node.diagnostic_accepted
                    break
            latest = node.latest
            if latest is None:
                continue
            surface = latest.get("surface_geometry") or {}
            print(
                f"status={latest.get('status')} "
                f"label={latest.get('effective_label')!r} "
                f"ready_frames={node.ready_count}/"
                f"{'one-shot' if diagnostic else args.stable_frames} "
                f"reject={node.last_reject_code or 'none'} "
                f"surface_valid={surface.get('estimate_valid')} "
                f"prov={surface.get('provenance_complete')} "
                f"reason={latest.get('failure_reason')!r}",
                flush=True,
            )
            if diagnostic:
                continue
            if node.ready_count < args.stable_frames:
                continue
            if (
                not args.allow_invalid_surface
                and not bool(surface.get("estimate_valid"))
            ):
                print(
                    "READY but surface estimate_valid=false; "
                    "pass --allow-invalid-surface for diagnostic demo",
                    flush=True,
                )
                continue
            frozen = latest
            break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if frozen is None:
        _print_failure(*_timeout_failure(node, args, diagnostic))
        return 2

    trial_id, payload = _build_payload(args, frozen, diagnostic)
    payload = with_canonical_hash(payload)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        trial_dir = Path(args.evidence_root).expanduser() / trial_id
        trial_dir.mkdir(parents=True, exist_ok=False)
        out = trial_dir / (
            "single_frame_diagnostic.json" if diagnostic else "frozen_target.json"
        )

    if out.exists():
        _print_failure(FAIL_OUTPUT_EXISTS, f"refusing_to_overwrite={out}")
        return 3
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sha_path = Path(str(out) + ".sha256")
    sha_path.write_text(
        f"{payload['canonical_payload_sha256']}  {out.name}\n"
    )
    print("RESULT=SUCCESS", flush=True)
    print(f"mode={DIAGNOSTIC_MODE if diagnostic else 'formal'}", flush=True)
    print(f"diagnostic_only={'true' if diagnostic else 'false'}", flush=True)
    print("execution_ready=false", flush=True)
    print(f"frozen_saved={out}", flush=True)
    print(
        f"point={frozen['camera_point_m']} "
        f"normal={frozen['surface_geometry']['panel_outward_normal']} "
        f"quality_hint="
        f"{'accepted' if frozen['surface_geometry'].get('provenance_complete') else 'diagnostic'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
