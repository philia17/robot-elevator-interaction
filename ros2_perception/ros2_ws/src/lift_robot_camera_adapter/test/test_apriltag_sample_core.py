import math

import pytest

from lift_robot_camera_adapter.apriltag_sample_core import (
    SamplingDeadline,
    summarize_samples,
    tag_display_state,
    validate_sample,
)


def _sample(stamp_ns, x, y, z, centre_x, centre_y, margin=100.0):
    return {
        "stamp_ns": stamp_ns,
        "translation_m": [x, y, z],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "centre_px": [centre_x, centre_y],
        "decision_margin": margin,
        "hamming": 0,
    }


def test_summarize_samples_reports_static_metrics():
    summary = summarize_samples(
        [
            _sample(1_000_000_000, 0.0, 0.0, 0.400, 960.0, 540.0, 90.0),
            _sample(1_100_000_000, 0.002, 0.0, 0.400, 962.0, 540.0, 110.0),
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["duration_s"] == pytest.approx(0.1)
    assert summary["translation_mean_m"] == pytest.approx([0.001, 0.0, 0.4])
    assert summary["translation_axis_std_mm"] == pytest.approx([1.0, 0.0, 0.0])
    assert summary["centre_mean_px"] == pytest.approx([961.0, 540.0])
    assert summary["decision_margin_min"] == 90.0
    assert summary["decision_margin_mean"] == 100.0
    assert summary["hamming_values"] == [0]


def test_summarize_samples_handles_quaternion_sign_equivalence():
    first = _sample(1, 0.0, 0.0, 0.4, 0.0, 0.0)
    second = _sample(2, 0.0, 0.0, 0.4, 0.0, 0.0)
    second["rotation_xyzw"] = [-0.0, -0.0, -0.0, -1.0]

    summary = summarize_samples([first, second])

    assert summary["rotation_error_max_deg"] == pytest.approx(0.0)


def test_summarize_samples_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        summarize_samples([])


def _valid_full_sample():
    sample = _sample(1_000_000_000, 0.0, 0.0, 0.4, 960.0, 540.0)
    sample.update(
        {
            "camera_frame": "camera_optical_frame",
            "tag_frame": "calibration_tag",
            "corners_px": [
                [900.0, 480.0],
                [1020.0, 480.0],
                [1020.0, 600.0],
                [900.0, 600.0],
            ],
        }
    )
    return sample


def test_validate_sample_enforces_canonical_frame_and_margin():
    sample = _valid_full_sample()
    assert validate_sample(sample) is sample
    sample["camera_frame"] = "vendor_frame"
    with pytest.raises(ValueError, match="canonical"):
        validate_sample(sample)


def test_validate_sample_rejects_target_outside_image():
    sample = _valid_full_sample()
    sample["corners_px"][0][0] = -1.0
    with pytest.raises(ValueError, match="inside"):
        validate_sample(sample)


def test_sampling_deadline_distinguishes_startup_stall_and_overall():
    deadline = SamplingDeadline(
        0.0, startup_timeout_s=5.0, stall_timeout_s=3.0, timeout_s=20.0
    )
    assert deadline.failure(4.9, 0, 30) is None
    assert "startup" in deadline.failure(5.0, 0, 30)
    deadline.record_progress(6.0)
    assert deadline.failure(8.9, 1, 30) is None
    assert "stalled" in deadline.failure(9.0, 1, 30)
    assert "overall" in deadline.failure(20.0, 1, 30)


@pytest.mark.parametrize(
    ("age", "state"),
    [
        (None, "no_tag"),
        (0.49, "tag_ok"),
        (0.5, "tag_intermittent"),
        (1.19, "tag_intermittent"),
        (1.2, "no_tag"),
    ],
)
def test_tag_display_state_hysteresis(age, state):
    assert tag_display_state(age) == state
