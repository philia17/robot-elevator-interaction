"""Fail-closed validation for the AprilTag optical-axis physical gate."""

from __future__ import annotations

from .apriltag_sample_core import summarize_samples, validate_sample


CAPTURE_FILES = {
    "center": "center.json",
    "right": "right.json",
    "down_baseline": "down_baseline.json",
    "down": "down.json",
    "forward_baseline": "forward_baseline.json",
    "forward": "forward.json",
}


def validate_capture_document(document, expected_label, *, min_decision_margin=50.0):
    if document.get("label") != expected_label:
        raise ValueError(f"expected label {expected_label}")
    target = document.get("target")
    if target != {"family": "tag36h11", "id": 0, "edge_m": 0.064}:
        raise ValueError(f"{expected_label}: target contract mismatch")
    samples = document.get("samples")
    if not isinstance(samples, list) or len(samples) < 30:
        raise ValueError(f"{expected_label}: at least 30 samples are required")
    stamps = []
    for sample in samples:
        validate_sample(sample, min_decision_margin=min_decision_margin)
        stamps.append(sample["stamp_ns"])
    if stamps != sorted(stamps) or len(set(stamps)) != len(stamps):
        raise ValueError(f"{expected_label}: timestamps must be strictly increasing")
    summary = summarize_samples(samples)
    if summary["translation_error_p95_mm"] > 2.0:
        raise ValueError(f"{expected_label}: static translation noise exceeds 2 mm P95")
    if summary["rotation_error_p95_deg"] > 1.0:
        raise ValueError(f"{expected_label}: static rotation noise exceeds 1 degree P95")
    if max(summary["centre_axis_std_px"]) > 2.0:
        raise ValueError(f"{expected_label}: image-centre noise exceeds 2 px")
    return summary


def evaluate_axis_captures(documents):
    missing = sorted(set(CAPTURE_FILES) - set(documents))
    if missing:
        raise ValueError(f"missing capture documents: {missing}")
    summaries = {
        label: validate_capture_document(documents[label], label)
        for label in CAPTURE_FILES
    }

    center_uv = summaries["center"]["centre_mean_px"]
    right_uv = summaries["right"]["centre_mean_px"]
    down_base_uv = summaries["down_baseline"]["centre_mean_px"]
    down_uv = summaries["down"]["centre_mean_px"]
    forward_base = summaries["forward_baseline"]
    forward = summaries["forward"]
    forward_base_uv = forward_base["centre_mean_px"]
    forward_uv = forward["centre_mean_px"]
    forward_base_z = forward_base["translation_mean_m"][2]
    forward_z = forward["translation_mean_m"][2]

    right_delta_u_px = right_uv[0] - center_uv[0]
    down_delta_v_px = down_uv[1] - down_base_uv[1]
    forward_delta_u_px = forward_uv[0] - forward_base_uv[0]
    forward_delta_v_px = forward_uv[1] - forward_base_uv[1]
    forward_delta_z_m = forward_z - forward_base_z
    forward_relative_z_change = forward_delta_z_m / forward_base_z

    checks = {
        "image_right_u_increases": {
            "value_px": right_delta_u_px,
            "minimum_px": 100.0,
            "pass": right_delta_u_px >= 100.0,
        },
        "image_down_v_increases": {
            "value_px": down_delta_v_px,
            "minimum_px": 100.0,
            "pass": down_delta_v_px >= 100.0,
        },
        "toward_camera_z_decreases": {
            "delta_z_m": forward_delta_z_m,
            "relative_change": forward_relative_z_change,
            "maximum_relative_change": -0.05,
            "pass": forward_relative_z_change <= -0.05,
        },
        "forward_lateral_drift_bounded": {
            "delta_u_px": forward_delta_u_px,
            "delta_v_px": forward_delta_v_px,
            "maximum_absolute_px": 120.0,
            "pass": max(abs(forward_delta_u_px), abs(forward_delta_v_px)) <= 120.0,
        },
    }
    passed = all(check["pass"] for check in checks.values())
    return {
        "status": "pass" if passed else "fail",
        "checks": checks,
        "coordinate_observation": {
            "camera_optical_frame_x": "image_right",
            "camera_optical_frame_y": "image_down",
            "camera_optical_frame_z": "forward_from_camera",
        },
        "summaries": summaries,
    }
