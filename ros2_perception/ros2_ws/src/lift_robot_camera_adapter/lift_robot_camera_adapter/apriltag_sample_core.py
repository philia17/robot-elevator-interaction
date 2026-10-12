"""Pure helpers for summarising static AprilTag calibration observations."""

from __future__ import annotations

import math
import statistics


CAMERA_FRAME = "camera_optical_frame"
TAG_FRAME = "calibration_tag"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


class SamplingDeadline:
    """Separate discovery, stalled-progress, and total capture deadlines."""

    def __init__(self, started_at, *, startup_timeout_s, stall_timeout_s, timeout_s):
        for name, value in (
            ("startup_timeout_s", startup_timeout_s),
            ("stall_timeout_s", stall_timeout_s),
            ("timeout_s", timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.started_at = started_at
        self.startup_timeout_s = startup_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self.timeout_s = timeout_s
        self.last_progress_at = started_at

    def record_progress(self, now):
        self.last_progress_at = now

    def failure(self, now, sample_count, target_count):
        elapsed = now - self.started_at
        if elapsed >= self.timeout_s:
            return (
                f"overall timeout after {elapsed:.1f}s with "
                f"{sample_count}/{target_count} paired samples"
            )
        if sample_count == 0 and elapsed >= self.startup_timeout_s:
            return f"no valid pair within startup timeout {self.startup_timeout_s:.1f}s"
        stalled = now - self.last_progress_at
        if sample_count > 0 and stalled >= self.stall_timeout_s:
            return (
                f"capture stalled for {stalled:.1f}s with "
                f"{sample_count}/{target_count} paired samples"
            )
        return None


def tag_display_state(detection_age_s):
    """Return an honest UI state with hysteresis for isolated missed frames."""
    if detection_age_s is None or not math.isfinite(detection_age_s) or detection_age_s < 0.0:
        return "no_tag"
    if detection_age_s < 0.5:
        return "tag_ok"
    if detection_age_s < 1.2:
        return "tag_intermittent"
    return "no_tag"


def validate_sample(sample, *, min_decision_margin=50.0):
    """Fail closed on frame, timestamp, geometry, pose, and detector quality."""
    if sample.get("camera_frame") != CAMERA_FRAME:
        raise ValueError("sample camera frame is not canonical")
    if sample.get("tag_frame") != TAG_FRAME:
        raise ValueError("sample tag frame is not calibration_tag")
    stamp_ns = sample.get("stamp_ns")
    if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns <= 0:
        raise ValueError("sample timestamp must be a positive integer")
    if sample.get("hamming") != 0:
        raise ValueError("sample hamming must be zero")
    margin = sample.get("decision_margin")
    if not isinstance(margin, (int, float)) or not math.isfinite(margin):
        raise ValueError("sample decision margin must be finite")
    if margin < min_decision_margin:
        raise ValueError("sample decision margin is below threshold")

    translation = sample.get("translation_m")
    rotation = sample.get("rotation_xyzw")
    centre = sample.get("centre_px")
    corners = sample.get("corners_px")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("sample translation must have three values")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise ValueError("sample quaternion must have four values")
    if not isinstance(centre, list) or len(centre) != 2:
        raise ValueError("sample centre must have two values")
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError("sample must contain four corners")
    numeric_groups = [translation, rotation, centre] + corners
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for group in numeric_groups
        for value in group
    ):
        raise ValueError("sample geometry contains a non-finite value")
    if translation[2] <= 0.0:
        raise ValueError("sample tag depth must be positive")
    quaternion_norm = math.sqrt(sum(value * value for value in rotation))
    if abs(quaternion_norm - 1.0) > 1e-3:
        raise ValueError("sample quaternion is not unit length")
    points = [centre] + corners
    if any(not (0.0 <= point[0] < IMAGE_WIDTH and 0.0 <= point[1] < IMAGE_HEIGHT) for point in points):
        raise ValueError("sample target is not fully inside the image")
    return sample


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _mean_quaternion_xyzw(quaternions):
    reference = quaternions[0]
    aligned = []
    for quaternion in quaternions:
        dot = sum(a * b for a, b in zip(reference, quaternion))
        aligned.append(quaternion if dot >= 0.0 else [-value for value in quaternion])
    mean = [statistics.fmean(values) for values in zip(*aligned)]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm <= 0.0:
        raise ValueError("invalid zero-norm mean quaternion")
    return [value / norm for value in mean]


def summarize_samples(samples):
    """Return translation, rotation, image-centre and detector quality metrics."""
    if not samples:
        raise ValueError("at least one sample is required")

    samples = sorted(samples, key=lambda sample: sample["stamp_ns"])
    if len({sample["stamp_ns"] for sample in samples}) != len(samples):
        raise ValueError("sample timestamps must be unique")
    translations = [sample["translation_m"] for sample in samples]
    rotations = [sample["rotation_xyzw"] for sample in samples]
    centres = [sample["centre_px"] for sample in samples]
    mean_translation = [statistics.fmean(values) for values in zip(*translations)]
    mean_rotation = _mean_quaternion_xyzw(rotations)
    mean_centre = [statistics.fmean(values) for values in zip(*centres)]

    translation_errors_mm = [
        1000.0 * math.sqrt(sum((value - mean) ** 2 for value, mean in zip(point, mean_translation)))
        for point in translations
    ]
    angular_errors_deg = []
    for quaternion in rotations:
        dot = abs(sum(a * b for a, b in zip(quaternion, mean_rotation)))
        angular_errors_deg.append(math.degrees(2.0 * math.acos(min(1.0, dot))))

    return {
        "sample_count": len(samples),
        "duration_s": (samples[-1]["stamp_ns"] - samples[0]["stamp_ns"]) / 1e9,
        "translation_mean_m": mean_translation,
        "translation_axis_std_mm": [
            1000.0 * statistics.pstdev(values) for values in zip(*translations)
        ],
        "translation_error_p95_mm": _percentile(translation_errors_mm, 0.95),
        "translation_error_max_mm": max(translation_errors_mm),
        "rotation_mean_xyzw": mean_rotation,
        "rotation_error_p95_deg": _percentile(angular_errors_deg, 0.95),
        "rotation_error_max_deg": max(angular_errors_deg),
        "centre_mean_px": mean_centre,
        "centre_axis_std_px": [statistics.pstdev(values) for values in zip(*centres)],
        "decision_margin_min": min(sample["decision_margin"] for sample in samples),
        "decision_margin_mean": statistics.fmean(
            sample["decision_margin"] for sample in samples
        ),
        "hamming_values": sorted({sample["hamming"] for sample in samples}),
    }
