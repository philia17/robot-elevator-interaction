"""Pure state for the diagnostic-only wrist button overlay.

The deployed detector has one ``button`` class.  This module therefore never
assigns a floor number.  The immutable fixed-camera result supplies the
Button-1 semantic label and 3-D projection; wrist geometry and a lightweight
glyph check only verify a bounded local candidate before display-only lock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import statistics
from typing import Any


@dataclass(frozen=True)
class WristButtonCandidate:
    xyxy: tuple[float, float, float, float]
    confidence: float | None
    class_id: int = 0
    source: str = "YOLO"
    radius_px: float | None = None
    shape: str = "UNKNOWN"

    @property
    def center_px(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


@dataclass(frozen=True)
class WristDetectorObservation:
    sequence: int
    received_monotonic_s: float
    frame_size_wh: tuple[int, int]
    candidates: tuple[WristButtonCandidate, ...] = ()
    error: str = ""
    yolo_candidates: tuple[WristButtonCandidate, ...] = ()
    circle_candidates: tuple[WristButtonCandidate, ...] = ()
    contour_candidates: tuple[WristButtonCandidate, ...] = ()


@dataclass(frozen=True)
class WristTargetDiagnostic:
    available: bool
    reason: str
    candidate: WristButtonCandidate | None = None
    center_px: tuple[float, float] | None = None
    dx_px: float | None = None
    dy_px: float | None = None
    confidence: float | None = None
    stable_count: int = 0
    stable_required: int = 5
    spread_px: float | None = None
    age_s: float | None = None


POST_PRE_SUCCESS_REASON = "exact frozen plan completed; no automatic retry"


@dataclass(frozen=True)
class CompletedPrecontactEvidence:
    """Display-only evidence retained before terminal cleanup."""

    frozen_package: Any
    qc_status: dict
    completed_monotonic_s: float
    cart_pose_epoch: str
    session_id: str
    plan_hash: str
    target_snapshot_sha256: str
    calibration_hash: str
    terminal_positions_rad: tuple[float, ...]
    executor_instance_nonce: str
    motion_authorized: bool = False
    adoption_source: str = "LIVE_TERMINAL_CALLBACK"
    display_target_base_m: tuple[float, float, float] | None = None
    display_pre_base_m: tuple[float, float, float] | None = None
    display_normal_out_base: tuple[float, float, float] | None = None
    # Immutable Stage-A PRE anchor.  ``terminal_positions_rad`` may be
    # advanced after one exactly-bound Stage-B OUT+BACK cycle, but this anchor
    # keeps repeated presses inside one bounded neighbourhood of the original
    # GO PRE result.  ``None`` preserves compatibility with older/boot-created
    # evidence and safely falls back to the initial terminal positions.
    original_terminal_positions_rad: tuple[float, ...] | None = None


def boot_completed_precontact_evidence(
    qc_status: dict,
    executor_status: dict,
    feedback_status: dict,
    feedback_age_s: float | None,
    boot_positions: tuple[tuple[float, ...], ...],
    *,
    boot_started_monotonic_s: float,
    maximum_feedback_age_s: float = 5.0,
    stationary_window: int = 3,
    stationary_spread_rad: float = 0.004,
) -> tuple[CompletedPrecontactEvidence | None, str]:
    """Adopt immutable QC-summary geometry for display only after restart."""
    identity = (
        str(qc_status.get("cart_pose_epoch") or ""),
        str(qc_status.get("session_id") or ""),
        str(qc_status.get("plan_hash") or ""),
        str(qc_status.get("prepared_snapshot_sha256") or ""),
        str(qc_status.get("calibration_hash") or ""),
    )
    if not all(identity):
        return None, "BOOT_TERMINAL_IDENTITY_MISSING"
    summary = qc_status.get("frozen_plan_summary")
    if not isinstance(summary, dict):
        return None, "BOOT_PLAN_SUMMARY_MISSING"
    if str(summary.get("plan_hash") or "") != identity[2]:
        return None, "BOOT_PLAN_HASH_MISMATCH"
    if str(summary.get("calibration_hash") or "") != identity[4]:
        return None, "BOOT_CALIBRATION_MISMATCH"
    geometry = []
    for field in ("p_button_base_m", "p_pre_base_m", "normal_out_base"):
        value = summary.get(field)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None, f"BOOT_SUMMARY_{field.upper()}_INVALID"
        try:
            vector = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None, f"BOOT_SUMMARY_{field.upper()}_INVALID"
        if any(not math.isfinite(item) for item in vector):
            return None, f"BOOT_SUMMARY_{field.upper()}_NONFINITE"
        geometry.append(vector)
    qc_executor = str(qc_status.get("executor_instance_nonce") or "")
    executor_instance = str(executor_status.get("instance_nonce") or "")
    if qc_executor and qc_executor != executor_instance:
        return None, "BOOT_EXECUTOR_IDENTITY_MISMATCH"
    try:
        current_positions = tuple(float(value) for value in feedback_status["positions_rad"])
    except (KeyError, TypeError, ValueError):
        return None, "SAFE_POSITIONS_MISSING"
    if len(current_positions) != 6:
        return None, "SAFE_POSITIONS_INVALID"
    evidence = CompletedPrecontactEvidence(
        frozen_package=None,
        qc_status=dict(qc_status),
        completed_monotonic_s=float(boot_started_monotonic_s),
        cart_pose_epoch=identity[0],
        session_id=identity[1],
        plan_hash=identity[2],
        target_snapshot_sha256=identity[3],
        calibration_hash=identity[4],
        terminal_positions_rad=current_positions,
        executor_instance_nonce=executor_instance,
        motion_authorized=False,
        adoption_source="QC_SUMMARY_BOOT_ADOPTION",
        display_target_base_m=geometry[0],
        display_pre_base_m=geometry[1],
        display_normal_out_base=geometry[2],
        # Boot adoption remains display-only.  Its safe local anchor is the
        # adopted stationary pose; it is never accepted by Live as PRESS
        # authority and therefore can never be motion-rebased.
        original_terminal_positions_rad=current_positions,
    )
    enabled, reason, _invalidate = completed_precontact_gate(
        evidence,
        qc_status,
        executor_status,
        feedback_status,
        feedback_age_s,
        boot_positions,
        maximum_feedback_age_s=maximum_feedback_age_s,
        stationary_window=stationary_window,
        stationary_spread_rad=stationary_spread_rad,
    )
    if not enabled:
        return None, reason
    return evidence, "BOOT_QC_SUMMARY_DISPLAY_ONLY_ADOPTED"


def completed_precontact_gate(
    evidence: CompletedPrecontactEvidence | None,
    qc_status: dict,
    executor_status: dict,
    feedback_status: dict,
    feedback_age_s: float | None,
    recent_positions: tuple[tuple[float, ...], ...],
    *,
    maximum_feedback_age_s: float = 5.0,
    stationary_window: int = 3,
    stationary_spread_rad: float = 0.004,
    posture_drift_rad: float = 0.012,
    original_posture_drift_rad: float | None = None,
) -> tuple[bool, str, bool]:
    """Return enabled, reason, and whether retained evidence is invalid."""
    if evidence is None:
        return False, "NO_COMPLETED_PRE", False
    if evidence.motion_authorized:
        return False, "INVALID_MOTION_AUTHORITY", True
    if str(qc_status.get("phase") or "").upper() != "TERMINAL":
        return False, "LEFT_PRE_TERMINAL", True
    if str(qc_status.get("terminal_reason") or "") != POST_PRE_SUCCESS_REASON:
        return False, "PRE_TERMINAL_NOT_ACCEPTED", True
    identity = (
        str(qc_status.get("cart_pose_epoch") or ""),
        str(qc_status.get("session_id") or ""),
        str(qc_status.get("plan_hash") or ""),
        str(qc_status.get("prepared_snapshot_sha256") or ""),
        str(qc_status.get("calibration_hash") or ""),
    )
    expected = (
        evidence.cart_pose_epoch,
        evidence.session_id,
        evidence.plan_hash,
        evidence.target_snapshot_sha256,
        evidence.calibration_hash,
    )
    if identity != expected:
        return False, "PRE_IDENTITY_CHANGED", True
    if str(executor_status.get("state") or "").upper() != "IDLE":
        return False, "EXECUTOR_NOT_IDLE", False
    if (
        evidence.executor_instance_nonce
        and str(executor_status.get("instance_nonce") or "")
        != evidence.executor_instance_nonce
    ):
        return False, "EXECUTOR_IDENTITY_CHANGED", True
    if executor_status.get("precontact_lease_active") is not False:
        return False, "EXECUTOR_LEASE_ACTIVE", False
    if executor_status.get("terminal_armed") is not False:
        return False, "EXECUTOR_ARMED", False
    if int(executor_status.get("joint_target_publishers") or 0) != 0:
        return False, "JOINT_TARGET_PUBLISHER", False
    resources = executor_status.get("resource_counts")
    if not isinstance(resources, dict) or any(int(value or 0) != 0 for value in resources.values()):
        return False, "EXECUTOR_RESOURCES_ACTIVE", False
    if feedback_age_s is None or feedback_age_s > maximum_feedback_age_s:
        return False, "SAFE_FEEDBACK_STALE", False
    if str(feedback_status.get("state") or "") != "CONVERGED_DISARMED":
        return False, "SAFE_NOT_CONVERGED_DISARMED", True
    if feedback_status.get("armed") is not False:
        return False, "SAFE_ARMED", True
    if str(feedback_status.get("fault") or ""):
        return False, "SAFE_FAULT", True
    torque = feedback_status.get("torque_enabled")
    if not isinstance(torque, list) or len(torque) != 6 or any(flag is not True for flag in torque):
        return False, "SAFE_TORQUE_NOT_6_OF_6", True
    try:
        current = tuple(float(value) for value in feedback_status["positions_rad"])
    except (KeyError, TypeError, ValueError):
        return False, "SAFE_POSITIONS_MISSING", False
    try:
        rolling = tuple(float(value) for value in evidence.terminal_positions_rad)
    except (TypeError, ValueError):
        return False, "SAFE_POSITIONS_INVALID", True
    if (
        len(current) != 6
        or len(rolling) != 6
        or any(not math.isfinite(value) for value in current)
        or any(not math.isfinite(value) for value in rolling)
    ):
        return False, "SAFE_POSITIONS_INVALID", True
    raw_original = getattr(evidence, "original_terminal_positions_rad", None)
    if raw_original is None:
        raw_original = rolling
    try:
        original = tuple(float(value) for value in raw_original)
    except (TypeError, ValueError):
        return False, "ORIGINAL_PRE_POSITIONS_INVALID", True
    if len(original) != 6 or any(not math.isfinite(value) for value in original):
        return False, "ORIGINAL_PRE_POSITIONS_INVALID", True
    if (
        original_posture_drift_rad is not None
        and max(abs(a - b) for a, b in zip(current, original))
        > float(original_posture_drift_rad)
    ):
        return False, "ORIGINAL_PRE_POSTURE_DRIFT", True
    if max(abs(a - b) for a, b in zip(current, rolling)) > posture_drift_rad:
        return False, "POSTURE_DRIFT", True
    if len(recent_positions) < stationary_window:
        return False, "WAIT_STATIONARY_SAMPLES", False
    for axis in range(6):
        try:
            values = [float(positions[axis]) for positions in recent_positions]
        except (IndexError, TypeError, ValueError):
            return False, "SAFE_POSITIONS_INVALID", True
        if any(not math.isfinite(value) for value in values):
            return False, "SAFE_POSITIONS_INVALID", True
        if max(values) - min(values) > stationary_spread_rad:
            return False, "POSTURE_NOT_STATIONARY", False
    return True, "ACCEPTED_PRE_TERMINAL_STATIONARY", False


def extract_circle_candidates(
    frame: Any,
    *,
    accumulator_threshold: float = 34.0,
) -> tuple[WristButtonCandidate, ...]:
    """Find circular wrist candidates after normalizing to 640x480.

    This is deterministic image logic only: it performs no capture or I/O.
    Coordinates are returned in the caller's original frame geometry.
    """
    import cv2
    import numpy as np

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3:
        return ()
    height, width = (int(frame.shape[0]), int(frame.shape[1]))
    if height <= 0 or width <= 0:
        return ()
    normalized = frame
    if (width, height) != (640, 480):
        normalized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edge_support = cv2.dilate(
        cv2.Canny(gray, 45, 125), np.ones((3, 3), dtype=np.uint8),
    )
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80.0,
        param1=120.0,
        param2=float(accumulator_threshold),
        minRadius=24,
        maxRadius=76,
    )
    if circles is None:
        return ()
    scale_x = width / 640.0
    scale_y = height / 480.0
    radius_scale = (scale_x + scale_y) * 0.5
    candidates = []
    for center_x, center_y, radius in np.asarray(circles[0], dtype=float):
        angles = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
        sample_x = np.clip(
            np.rint(center_x + radius * np.cos(angles)).astype(int), 0, 639,
        )
        sample_y = np.clip(
            np.rint(center_y + radius * np.sin(angles)).astype(int), 0, 479,
        )
        if float(np.mean(edge_support[sample_y, sample_x] > 0)) < 0.65:
            continue
        x = float(center_x * scale_x)
        y = float(center_y * scale_y)
        r = float(radius * radius_scale)
        candidates.append(WristButtonCandidate(
            (x - r, y - r, x + r, y + r),
            None,
            0,
            "CIRCLE",
            r,
            "CIRCLE",
        ))
    return tuple(sorted(candidates, key=lambda item: (
        item.center_px[1], item.center_px[0], float(item.radius_px or 0.0),
    )))


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def deduplicate_candidates(
    candidates: tuple[WristButtonCandidate, ...],
    *,
    minimum_iou: float = 0.45,
) -> tuple[WristButtonCandidate, ...]:
    """Keep one stable geometry proposal for each physical button."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item.source == "YOLO" else 1,
            -(item.confidence if item.confidence is not None else 0.0),
            -((item.xyxy[2] - item.xyxy[0]) * (item.xyxy[3] - item.xyxy[1])),
        ),
    )
    kept: list[WristButtonCandidate] = []
    for candidate in ranked:
        width = max(1.0, candidate.xyxy[2] - candidate.xyxy[0])
        height = max(1.0, candidate.xyxy[3] - candidate.xyxy[1])
        duplicate = False
        for previous in kept:
            center_limit = 0.22 * max(
                width,
                height,
                previous.xyxy[2] - previous.xyxy[0],
                previous.xyxy[3] - previous.xyxy[1],
            )
            if (
                _bbox_iou(candidate.xyxy, previous.xyxy) >= minimum_iou
                or math.dist(candidate.center_px, previous.center_px) <= center_limit
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda item: (item.center_px[1], item.center_px[0])))


def extract_shape_candidates(frame: Any) -> tuple[WristButtonCandidate, ...]:
    """Extract circle, square, and rounded-rectangle proposals with contours.

    The proposal gate deliberately does not require circularity or an exact
    vertex count.  It performs no capture or I/O and returns caller-frame
    coordinates.
    """
    import cv2
    import numpy as np

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3:
        return ()
    height, width = (int(frame.shape[0]), int(frame.shape[1]))
    if height <= 0 or width <= 0:
        return ()
    normalized = frame
    if (width, height) != (640, 480):
        normalized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 125)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
        iterations=2,
    )
    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    scale_x = width / 640.0
    scale_y = height / 480.0
    proposals: list[WristButtonCandidate] = []
    for contour in contours:
        area = float(abs(cv2.contourArea(contour)))
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if area < 900.0:
            continue
        if not (42 <= box_width <= 165 and 42 <= box_height <= 165):
            continue
        aspect = float(box_width) / float(box_height)
        if not 0.60 <= aspect <= 1.65:
            continue
        extent = area / float(box_width * box_height)
        if extent < 0.25:
            continue
        hull_area = float(abs(cv2.contourArea(cv2.convexHull(contour))))
        if hull_area <= 1e-6 or area / hull_area < 0.60:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        vertices = len(cv2.approxPolyDP(contour, 0.035 * perimeter, True))
        shape = (
            "RECT" if vertices <= 6
            else "ROUND" if circularity >= 0.72
            else "ROUNDED_RECT"
        )
        proposals.append(WristButtonCandidate(
            (
                float(x * scale_x),
                float(y * scale_y),
                float((x + box_width) * scale_x),
                float((y + box_height) * scale_y),
            ),
            None,
            0,
            "CONTOUR",
            None,
            shape,
        ))
    return deduplicate_candidates(tuple(proposals))


ACQUISITION_ROI_HALF_WIDTH_PX = 110.0
ACQUISITION_ROI_ABOVE_PX = 180.0
ACQUISITION_ROI_BELOW_PX = 45.0
ACQUISITION_ROI_TOUCH_CLEARANCE_PX = 18.0
ACQUISITION_TARGET_RESIDUAL_PX = 90.0
STRICT_PROJECTION_RESIDUAL_PX = 55.0


def projected_acquisition_roi(
    frame_size_wh: tuple[int, int],
    projected_target_px: tuple[float, float],
    touch_px: tuple[float, float],
) -> tuple[int, int, int, int] | None:
    """Return the bounded raw-pixel search region around immutable A-PROJ."""
    width, height = (int(value) for value in frame_size_wh)
    if width <= 0 or height <= 0:
        return None
    values = (*projected_target_px, *touch_px)
    if any(not math.isfinite(float(value)) for value in values):
        return None
    projected_x, projected_y = projected_target_px
    _touch_x, touch_y = touch_px
    x0 = max(0, int(math.floor(projected_x - ACQUISITION_ROI_HALF_WIDTH_PX)))
    x1 = min(width, int(math.ceil(projected_x + ACQUISITION_ROI_HALF_WIDTH_PX)))
    y0 = max(0, int(math.floor(projected_y - ACQUISITION_ROI_ABOVE_PX)))
    y1 = min(
        height,
        int(math.ceil(projected_y + ACQUISITION_ROI_BELOW_PX)),
        int(math.floor(touch_y - ACQUISITION_ROI_TOUCH_CLEARANCE_PX)),
    )
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def admit_candidates_in_acquisition_roi(
    candidates: tuple[WristButtonCandidate, ...],
    roi_xyxy: tuple[int, int, int, int] | None,
    touch_px: tuple[float, float],
    projected_target_px: tuple[float, float],
    *,
    maximum_residual_px: float = ACQUISITION_TARGET_RESIDUAL_PX,
    roi_border_clearance_px: float = 2.0,
) -> tuple[WristButtonCandidate, ...]:
    """Admit only complete button-sized outer boundaries inside the ROI."""
    if roi_xyxy is None:
        return ()
    roi_x0, roi_y0, roi_x1, roi_y1 = roi_xyxy
    admitted = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate.xyxy
        box_width = x1 - x0
        box_height = y1 - y0
        aspect = box_width / box_height if box_height > 0.0 else 0.0
        if not (
            x0 >= roi_x0 + roi_border_clearance_px
            and y0 >= roi_y0 + roi_border_clearance_px
            and x1 <= roi_x1 - roi_border_clearance_px
            and y1 <= roi_y1 - roi_border_clearance_px
        ):
            continue
        if not (42.0 <= box_width <= 165.0 and 42.0 <= box_height <= 165.0):
            continue
        if not 0.68 <= aspect <= 1.47:
            continue
        if candidate.center_px[1] > touch_px[1] - ACQUISITION_ROI_TOUCH_CLEARANCE_PX:
            continue
        if math.dist(candidate.center_px, projected_target_px) > maximum_residual_px:
            continue
        admitted.append(candidate)
    return deduplicate_candidates(tuple(admitted))


@dataclass(frozen=True)
class GlyphOneVerification:
    passed: bool
    reason: str
    vertical_run_fraction: float = 0.0
    foreground_fraction: float = 0.0


def verify_glyph_one(
    frame: Any,
    candidate: WristButtonCandidate,
    *,
    inner_fraction: float = 0.60,
) -> GlyphOneVerification:
    """Verify a bright central ``1`` without cross-camera appearance data."""
    import cv2
    import numpy as np

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3:
        return GlyphOneVerification(False, "GLYPH_FRAME_INVALID")
    x0, y0, x1, y1 = candidate.xyxy
    center_x, center_y = candidate.center_px
    half_width = max(1, int(round((x1 - x0) * inner_fraction * 0.5)))
    half_height = max(1, int(round((y1 - y0) * inner_fraction * 0.5)))
    left = max(0, int(round(center_x)) - half_width)
    right = min(int(frame.shape[1]), int(round(center_x)) + half_width)
    top = max(0, int(round(center_y)) - half_height)
    bottom = min(int(frame.shape[0]), int(round(center_y)) + half_height)
    if right - left < 18 or bottom - top < 18:
        return GlyphOneVerification(False, "GLYPH_CROP_TOO_SMALL")
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] < 90)
        & (hsv[:, :, 2] > 145)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 1), dtype=np.uint8),
    )
    crop_height, crop_width = mask.shape
    foreground_fraction = float(mask.mean())
    if not 0.055 <= foreground_fraction <= 0.48:
        return GlyphOneVerification(
            False, "GLYPH_FOREGROUND_INVALID", 0.0, foreground_fraction,
        )
    central_left = int(math.floor(crop_width * 0.25))
    central_right = max(central_left + 1, int(math.ceil(crop_width * 0.75)))
    longest_run = 0
    for column in range(central_left, central_right):
        changes = np.diff(np.concatenate((
            np.zeros(1, dtype=int),
            mask[:, column].astype(int),
            np.zeros(1, dtype=int),
        )))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size:
            longest_run = max(longest_run, int(np.max(ends - starts)))
    vertical_run_fraction = longest_run / float(crop_height)
    if vertical_run_fraction < 0.72:
        return GlyphOneVerification(
            False, "GLYPH_NOT_1", vertical_run_fraction, foreground_fraction,
        )
    return GlyphOneVerification(
        True, "GLYPH_1_VERIFIED", vertical_run_fraction, foreground_fraction,
    )


def merge_shape_candidates(
    yolo_candidates: tuple[WristButtonCandidate, ...],
    contour_candidates: tuple[WristButtonCandidate, ...],
) -> tuple[WristButtonCandidate, ...]:
    """Merge coarse YOLO geometry with shape-agnostic proposals."""
    return deduplicate_candidates(tuple(yolo_candidates) + tuple(contour_candidates))


def exclude_touch_overlapping_candidates(
    candidates: tuple[WristButtonCandidate, ...],
    touch_px: tuple[float, float],
    *,
    margin_px: float = 4.0,
) -> tuple[tuple[WristButtonCandidate, ...], tuple[WristButtonCandidate, ...]]:
    """Remove any proposal whose physical extent contains the TOUCH point."""
    usable = []
    excluded = []
    touch_x, touch_y = touch_px
    for candidate in candidates:
        x0, y0, x1, y1 = candidate.xyxy
        overlaps = (
            x0 - margin_px <= touch_x <= x1 + margin_px
            and y0 - margin_px <= touch_y <= y1 + margin_px
        )
        if candidate.radius_px is not None:
            overlaps = overlaps or (
                math.dist(candidate.center_px, touch_px)
                <= candidate.radius_px + margin_px
            )
        (excluded if overlaps else usable).append(candidate)
    return tuple(usable), tuple(excluded)


def exclude_touch_overlapping_circles(
    candidates: tuple[WristButtonCandidate, ...],
    touch_px: tuple[float, float],
    *,
    margin_px: float = 4.0,
) -> tuple[tuple[WristButtonCandidate, ...], tuple[WristButtonCandidate, ...]]:
    """Remove the soft tool-head circle that contains the calibrated TOUCH."""
    usable = []
    excluded = []
    for candidate in candidates:
        radius = candidate.radius_px
        if (
            candidate.source == "CIRCLE"
            and radius is not None
            and math.dist(candidate.center_px, touch_px) <= radius + margin_px
        ):
            excluded.append(candidate)
        else:
            usable.append(candidate)
    return tuple(usable), tuple(excluded)


def admit_button_sized_candidates(
    candidates: tuple[WristButtonCandidate, ...],
) -> tuple[WristButtonCandidate, ...]:
    """Keep complete button-sized proposals for the bounded TOUCH fallback.

    This helper deliberately does not use the projected target or assign a
    button number.  It is only the proposal-size gate used when the primary
    A-PROJ ROI admits no candidate during one fresh WRIST_ACQUIRE sample.
    """
    admitted = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate.xyxy
        box_width = x1 - x0
        box_height = y1 - y0
        aspect = box_width / box_height if box_height > 0.0 else 0.0
        if (
            42.0 <= box_width <= 165.0
            and 42.0 <= box_height <= 165.0
            and 0.68 <= aspect <= 1.47
        ):
            admitted.append(candidate)
    return deduplicate_candidates(tuple(admitted))


def provisional_nearest_candidate(
    candidates: tuple[WristButtonCandidate, ...],
    touch_px: tuple[float, float],
    *,
    maximum_distance_px: float,
    runner_up_margin_px: float,
) -> tuple[WristButtonCandidate | None, str]:
    """Select by geometry only; the one-class model does not know Button 1."""
    if not candidates:
        return None, "NO_DETECTION"
    ranked = sorted(
        candidates,
        key=lambda item: math.dist(item.center_px, touch_px),
    )
    nearest_distance = math.dist(ranked[0].center_px, touch_px)
    if nearest_distance > maximum_distance_px:
        return None, "NO_CANDIDATE_NEAR_TOUCH"
    if len(ranked) > 1:
        runner_up_distance = math.dist(ranked[1].center_px, touch_px)
        if runner_up_distance - nearest_distance < runner_up_margin_px:
            return None, "AMBIGUOUS_NEAREST_TO_TOUCH"
    return ranked[0], "PROVISIONAL_NEAREST_TO_TOUCH"


@dataclass(frozen=True)
class ProjectionVerification:
    candidate: WristButtonCandidate | None
    reason: str
    projection_residual_px: float | None = None


def verify_candidate_against_projection(
    candidates: tuple[WristButtonCandidate, ...],
    touch_px: tuple[float, float],
    projected_target_px: tuple[float, float],
    *,
    maximum_projection_residual_px: float = ACQUISITION_TARGET_RESIDUAL_PX,
    projection_runner_up_margin_px: float = 18.0,
    maximum_touch_distance_px: float = 220.0,
    touch_runner_up_margin_px: float = 30.0,
    minimum_above_touch_px: float = 18.0,
) -> ProjectionVerification:
    """Cross-check geometry against the immutable fixed-camera target.

    Projection is the primary identity cue.  TOUCH proximity, a clear
    runner-up margin, and the expected above-TOUCH relation are secondary.
    No semantic button number is invented here.
    """
    if not candidates:
        return ProjectionVerification(None, "NO_CANDIDATE")
    ranked_projection = sorted(
        candidates,
        key=lambda item: math.dist(item.center_px, projected_target_px),
    )
    best = ranked_projection[0]
    residual = math.dist(best.center_px, projected_target_px)
    if residual > maximum_projection_residual_px:
        return ProjectionVerification(None, "A_PROJ_RESIDUAL_TOO_LARGE", residual)
    if len(ranked_projection) > 1:
        runner_residual = math.dist(
            ranked_projection[1].center_px, projected_target_px,
        )
        if runner_residual - residual < projection_runner_up_margin_px:
            return ProjectionVerification(None, "A_PROJ_AMBIGUOUS", residual)
    if best.center_px[1] > touch_px[1] - minimum_above_touch_px:
        return ProjectionVerification(None, "WRONG_SIDE_OF_TOUCH", residual)
    touch_distance = math.dist(best.center_px, touch_px)
    if touch_distance > maximum_touch_distance_px:
        return ProjectionVerification(None, "NO_CANDIDATE_NEAR_TOUCH", residual)
    ranked_touch = sorted(
        candidates,
        key=lambda item: math.dist(item.center_px, touch_px),
    )
    if ranked_touch[0] is not best:
        return ProjectionVerification(None, "A_PROJ_TOUCH_DISAGREE", residual)
    if len(ranked_touch) > 1:
        first_distance = math.dist(ranked_touch[0].center_px, touch_px)
        second_distance = math.dist(ranked_touch[1].center_px, touch_px)
        if second_distance - first_distance < touch_runner_up_margin_px:
            return ProjectionVerification(None, "TOUCH_AMBIGUOUS", residual)
    reason = (
        "A_PROJ_VERIFIED"
        if residual <= STRICT_PROJECTION_RESIDUAL_PX
        else "A_PROJ_ACQUISITION_VERIFIED"
    )
    return ProjectionVerification(best, reason, residual)


def verify_candidate_against_touch_geometry(
    candidates: tuple[WristButtonCandidate, ...],
    touch_px: tuple[float, float],
    *,
    maximum_distance_px: float = 220.0,
    runner_up_margin_px: float = 30.0,
    minimum_above_touch_px: float = 18.0,
) -> ProjectionVerification:
    """Select one fresh candidate from TOUCH geometry when A-PROJ is unusable.

    This is a bounded acquisition fallback only.  TOUCH itself is removed
    before the candidate is required to be above TOUCH and the existing
    nearest/runner-up rule is applied.  No button identity is inferred.
    ``projection_residual_px=0`` is an explicit sentinel: the returned
    ``TOUCH_GEOMETRY_PROVISIONAL`` reason tells the display that no projection
    residual was measured, while allowing the existing 3-of-5 lock machinery
    to bind its pixel pair to the observed candidate center.
    """
    usable, _excluded = exclude_touch_overlapping_candidates(candidates, touch_px)
    selected, reason = provisional_nearest_candidate(
        usable,
        touch_px,
        maximum_distance_px=float(maximum_distance_px),
        runner_up_margin_px=float(runner_up_margin_px),
    )
    if selected is None:
        return ProjectionVerification(None, reason, None)
    if float(selected.center_px[1]) > float(touch_px[1]) - float(minimum_above_touch_px):
        return ProjectionVerification(None, "WRONG_SIDE_OF_TOUCH", None)
    return ProjectionVerification(selected, "TOUCH_GEOMETRY_PROVISIONAL", 0.0)


@dataclass(frozen=True)
class WristVisualLock:
    stage: str
    center_px: tuple[float, float]
    xyxy: tuple[float, float, float, float]
    source: str
    shape: str
    acquired_sequence: int
    acquired_frame_stamp_ns: int
    acquired_projection_px: tuple[float, float]
    projection_residual_px: float
    spread_px: float
    identity: tuple[str, ...]
    target_base_m: tuple[float, float, float]
    feedback_positions_rad: tuple[float, ...]
    contact_reference_px: tuple[float, float] | None = None
    motion_authorized: bool = False
    future_motion_active: bool = False
    projection_context_json: str = ""


@dataclass(frozen=True)
class _VerifiedSample:
    candidate: WristButtonCandidate
    sequence: int
    frame_stamp_ns: int
    projected_target_px: tuple[float, float]
    contact_reference_px: tuple[float, float] | None
    projection_residual_px: float
    identity: tuple[str, ...]
    target_base_m: tuple[float, float, float]
    feedback_positions_rad: tuple[float, ...]
    projection_context_json: str = ""


class WristTargetLock:
    """Acquire 3-of-5 verified geometry once, then remain immutable."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        required_count: int = 3,
        cluster_radius_px: float = 7.0,
        press_anchor_radius_px: float = 24.0,
    ) -> None:
        self.window_size = int(window_size)
        self.required_count = int(required_count)
        self.cluster_radius_px = float(cluster_radius_px)
        self.press_anchor_radius_px = float(press_anchor_radius_px)
        self._samples: deque[_VerifiedSample | None] = deque(maxlen=self.window_size)
        self._lock: WristVisualLock | None = None
        self._base_identity: tuple[str, ...] = ()
        self._last_reason = "NO_LOCK"
        self._last_sequence = -1
        self._press_samples: deque[_VerifiedSample | None] = deque(
            maxlen=self.window_size,
        )
        self._press_reacquire_active = False
        self._press_last_sequence = -1
        self._press_acquisition_id = ""
        self._press_lock_acquired = False

    @property
    def locked(self) -> WristVisualLock | None:
        return self._lock

    @property
    def stable_count(self) -> int:
        if self._press_reacquire_active:
            return len(self._best_cluster(self._press_samples))
        return len(self._best_cluster())

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def clear(self, reason: str) -> None:
        self._samples.clear()
        self._lock = None
        self._base_identity = ()
        self._last_reason = str(reason)
        self._last_sequence = -1
        self._press_samples.clear()
        self._press_reacquire_active = False
        self._press_last_sequence = -1
        self._press_acquisition_id = ""
        self._press_lock_acquired = False

    def _best_cluster(
        self,
        samples: deque[_VerifiedSample | None] | None = None,
    ) -> tuple[_VerifiedSample, ...]:
        sample_window = self._samples if samples is None else samples
        if not sample_window:
            return ()
        clusters = []
        for seed in sample_window:
            if seed is None:
                continue
            members = tuple(
                item for item in sample_window
                if item is not None
                and item.identity == seed.identity
                and math.dist(item.candidate.center_px, seed.candidate.center_px)
                <= self.cluster_radius_px
            )
            clusters.append(members)
        return max(clusters, key=len, default=())

    def observe_verified(
        self,
        verification: ProjectionVerification,
        *,
        sequence: int,
        frame_stamp_ns: int,
        projected_target_px: tuple[float, float],
        identity: tuple[str, ...],
        target_base_m: tuple[float, float, float],
        feedback_positions_rad: tuple[float, ...],
        contact_reference_px: tuple[float, float] | None = None,
        projection_context_json: str = "",
    ) -> WristVisualLock | None:
        if self._lock is not None:
            return self._lock
        if int(sequence) == self._last_sequence:
            return None
        self._last_sequence = int(sequence)
        if verification.candidate is None or verification.projection_residual_px is None:
            self._samples.append(None)
            self._last_reason = verification.reason
            return None
        target_base = tuple(float(value) for value in target_base_m)
        if len(target_base) != 3 or any(not math.isfinite(value) for value in target_base):
            self._samples.append(None)
            self._last_reason = "TARGET_BASE_INVALID"
            return None
        contact_reference = None
        if contact_reference_px is not None:
            try:
                candidate_reference = tuple(
                    float(value) for value in contact_reference_px
                )
            except (TypeError, ValueError):
                candidate_reference = ()
            if len(candidate_reference) == 2 and all(
                math.isfinite(value) for value in candidate_reference
            ):
                contact_reference = candidate_reference
        self._samples.append(_VerifiedSample(
            verification.candidate,
            int(sequence),
            int(frame_stamp_ns),
            tuple(float(value) for value in projected_target_px),
            contact_reference,
            float(verification.projection_residual_px),
            tuple(str(value) for value in identity),
            target_base,
            tuple(float(value) for value in feedback_positions_rad),
            str(projection_context_json),
        ))
        cluster = self._best_cluster()
        if len(cluster) < self.required_count:
            self._last_reason = "PENDING_3_OF_5"
            return None
        center = (
            float(statistics.median(item.candidate.center_px[0] for item in cluster)),
            float(statistics.median(item.candidate.center_px[1] for item in cluster)),
        )
        spread = max(math.dist(item.candidate.center_px, center) for item in cluster)
        xyxy = tuple(float(statistics.median(
            item.candidate.xyxy[index] for item in cluster
        )) for index in range(4))
        newest = max(cluster, key=lambda item: item.sequence)
        sources = sorted({item.candidate.source for item in cluster})
        shapes = sorted({item.candidate.shape for item in cluster})
        self._lock = WristVisualLock(
            "PRE-LOCK",
            center,
            xyxy,
            "+".join(sources),
            "+".join(shapes),
            newest.sequence,
            newest.frame_stamp_ns,
            newest.projected_target_px,
            float(statistics.median(
                item.projection_residual_px for item in cluster
            )),
            spread,
            newest.identity,
            newest.target_base_m,
            newest.feedback_positions_rad,
            newest.contact_reference_px,
            False,
            False,
            newest.projection_context_json,
        )
        self._base_identity = newest.identity
        self._last_reason = "PRE_LOCK_ACQUIRED"
        return self._lock

    def begin_press_reacquire(self, acquisition_id: str) -> bool:
        """Bind one fresh 3-of-5 window to the displayed departure lock."""
        if self._lock is None or self._lock.stage != "PRE-LOCK":
            self._last_reason = "PRE_LOCK_REQUIRED"
            return False
        if not str(acquisition_id).strip():
            self._last_reason = "PRESS_ACQUISITION_ID_REQUIRED"
            return False
        self._press_samples.clear()
        self._press_reacquire_active = True
        self._press_last_sequence = -1
        self._press_acquisition_id = str(acquisition_id)
        self._press_lock_acquired = False
        self._last_reason = "PRESS_REACQUIRE_STARTED"
        return True

    def acquire_press_lock(
        self,
        verification: ProjectionVerification,
        *,
        sequence: int,
        frame_stamp_ns: int,
        projected_target_px: tuple[float, float],
        identity: tuple[str, ...],
        target_base_m: tuple[float, float, float],
        feedback_positions_rad: tuple[float, ...],
        contact_reference_px: tuple[float, float] | None = None,
        projection_context_json: str = "",
    ) -> WristVisualLock | None:
        """Consume one explicit post-alignment sample; lock only at 3-of-5."""
        if self._lock is not None and self._press_lock_acquired:
            return self._lock
        if (
            self._lock is None
            or self._lock.stage != "PRE-LOCK"
            or not self._press_reacquire_active
        ):
            self._last_reason = "PRESS_REACQUIRE_NOT_STARTED"
            return None
        if int(sequence) == self._press_last_sequence:
            return None
        self._press_last_sequence = int(sequence)
        if verification.candidate is None or verification.projection_residual_px is None:
            self._press_samples.append(None)
            self._last_reason = verification.reason
            return None
        if (
            math.dist(verification.candidate.center_px, self._lock.center_px)
            > self.press_anchor_radius_px
        ):
            self._press_samples.append(None)
            self._last_reason = "PRESS_CANDIDATE_OUTSIDE_DEPARTURE_ANCHOR"
            return None
        provided_target = tuple(float(value) for value in target_base_m)
        if (
            tuple(str(value) for value in identity) != self._base_identity
            or len(provided_target) != 3
            or any(not math.isfinite(value) for value in provided_target)
            or any(abs(a - b) > 1e-9 for a, b in zip(provided_target, self._lock.target_base_m))
        ):
            self._press_samples.append(None)
            self._last_reason = "PRESS_TARGET_IDENTITY_MISMATCH"
            return None
        self._press_samples.append(_VerifiedSample(
            verification.candidate,
            int(sequence),
            int(frame_stamp_ns),
            tuple(float(value) for value in projected_target_px),
            contact_reference_px,
            float(verification.projection_residual_px),
            self._base_identity,
            self._lock.target_base_m,
            tuple(float(value) for value in feedback_positions_rad),
            str(projection_context_json),
        ))
        cluster = self._best_cluster(self._press_samples)
        if len(cluster) < self.required_count:
            self._last_reason = "PRESS_PENDING_3_OF_5"
            return None
        center = (
            float(statistics.median(item.candidate.center_px[0] for item in cluster)),
            float(statistics.median(item.candidate.center_px[1] for item in cluster)),
        )
        xyxy = tuple(float(statistics.median(
            item.candidate.xyxy[index] for item in cluster
        )) for index in range(4))
        spread = max(math.dist(item.candidate.center_px, center) for item in cluster)
        newest = max(cluster, key=lambda item: item.sequence)
        sources = sorted({item.candidate.source for item in cluster})
        shapes = sorted({item.candidate.shape for item in cluster})
        self._lock = WristVisualLock(
            "PRE-LOCK",
            newest.candidate.center_px,
            xyxy,
            "+".join(sources),
            "+".join(shapes),
            newest.sequence,
            newest.frame_stamp_ns,
            newest.projected_target_px,
            float(statistics.median(
                item.projection_residual_px for item in cluster
            )),
            spread,
            (self._press_acquisition_id, *self._base_identity),
            self._lock.target_base_m,
            newest.feedback_positions_rad,
            newest.contact_reference_px,
            False,
            False,
            newest.projection_context_json,
        )
        self._press_samples.clear()
        self._press_reacquire_active = False
        self._press_lock_acquired = True
        self._last_reason = "PRESS_LOCK_ACQUIRED"
        return self._lock

    def mark_future_motion_active(self) -> WristVisualLock | None:
        """Pure future hook; it grants no motion authority."""
        if self._lock is None or not self._press_lock_acquired:
            self._last_reason = "PRESS_LOCK_REQUIRED"
            return None
        self._lock = replace(self._lock, future_motion_active=True)
        self._last_reason = "FUTURE_MOTION_ACTIVE_DEBUG_ONLY"
        return self._lock


class WristTargetTracker:
    """Require a fresh, stable provisional association before exposing W-TGT."""

    def __init__(
        self,
        *,
        stable_required: int = 5,
        maximum_spread_px: float = 3.0,
        maximum_age_s: float = 0.75,
        maximum_distance_px: float = 220.0,
        runner_up_margin_px: float = 40.0,
    ) -> None:
        self.stable_required = int(stable_required)
        self.maximum_spread_px = float(maximum_spread_px)
        self.maximum_age_s = float(maximum_age_s)
        self.maximum_distance_px = float(maximum_distance_px)
        self.runner_up_margin_px = float(runner_up_margin_px)
        self._centers: deque[tuple[float, float]] = deque(maxlen=self.stable_required)
        self._last_sequence = -1
        self._last_diagnostic = WristTargetDiagnostic(
            False, "NO_INFERENCE", stable_required=self.stable_required,
        )

    def clear(self, reason: str, *, age_s: float | None = None) -> WristTargetDiagnostic:
        self._centers.clear()
        self._last_diagnostic = WristTargetDiagnostic(
            False,
            reason,
            stable_required=self.stable_required,
            age_s=age_s,
        )
        return self._last_diagnostic

    def update(
        self,
        observation: WristDetectorObservation | None,
        touch_px: tuple[float, float] | None,
        *,
        now_monotonic_s: float,
    ) -> WristTargetDiagnostic:
        if observation is None:
            return self.clear("NO_INFERENCE")
        age_s = max(0.0, now_monotonic_s - observation.received_monotonic_s)
        if age_s > self.maximum_age_s:
            return self.clear("INFERENCE_STALE", age_s=age_s)
        if observation.sequence == self._last_sequence:
            return WristTargetDiagnostic(
                **{
                    **self._last_diagnostic.__dict__,
                    "age_s": age_s,
                }
            )
        self._last_sequence = observation.sequence
        if observation.error:
            return self.clear(f"DETECTOR_ERROR:{observation.error}", age_s=age_s)
        if touch_px is None:
            return self.clear("TOUCH_UNAVAILABLE", age_s=age_s)
        candidate, reason = provisional_nearest_candidate(
            observation.candidates,
            touch_px,
            maximum_distance_px=self.maximum_distance_px,
            runner_up_margin_px=self.runner_up_margin_px,
        )
        if candidate is None:
            return self.clear(reason, age_s=age_s)
        center = candidate.center_px
        self._centers.append(center)
        median_center = (
            statistics.median(point[0] for point in self._centers),
            statistics.median(point[1] for point in self._centers),
        )
        spread = max(math.dist(point, median_center) for point in self._centers)
        stable_count = len(self._centers)
        stable = stable_count >= self.stable_required and spread <= self.maximum_spread_px
        diagnostic = WristTargetDiagnostic(
            stable,
            reason if stable else "PENDING_STABILITY",
            candidate=candidate,
            center_px=median_center if stable else center,
            dx_px=(median_center[0] if stable else center[0]) - touch_px[0],
            dy_px=(median_center[1] if stable else center[1]) - touch_px[1],
            confidence=candidate.confidence,
            stable_count=stable_count,
            stable_required=self.stable_required,
            spread_px=spread,
            age_s=age_s,
        )
        self._last_diagnostic = diagnostic
        return diagnostic
