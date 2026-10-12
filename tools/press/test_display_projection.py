"""Focused tests for the pure display-projection core (no ROS/cv2)."""

from __future__ import annotations

import numpy as np
import pytest

from press.display_projection import (
    CANONICAL_TARGET_FRAME,
    RAW_GEMINI_TARGET_FRAME,
    project_camera_point,
    target_frame_center_state,
    transform_point,
)


def _k() -> np.ndarray:
    return np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 510.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_pinhole_projection_and_center_math():
    pixel, reason = project_camera_point((0.0, 0.0, 1.0), _k())
    assert reason == "OK"
    assert pixel == (320, 240)
    pixel, reason = project_camera_point((-0.2, 0.1, 1.0), _k())
    assert reason == "OK"
    assert pixel == pytest.approx((220.0, 291.0))


def test_distortion_changes_projection_consistent_with_radial_model():
    point = (0.2, 0.1, 1.0)
    no_dist, _ = project_camera_point(point, _k())
    with_dist, _ = project_camera_point(
        point, _k(), dist_coeffs=[0.2, 0.0, 0.0, 0.0, 0.0]
    )
    assert no_dist is not None and with_dist is not None
    assert with_dist != no_dist
    # For positive k1, the marker moves away from the principal point.
    assert with_dist[0] > no_dist[0]
    assert with_dist[1] > no_dist[1]


def test_rational_distortion_model_is_supported():
    pixel, reason = project_camera_point(
        (0.2, 0.1, 1.0),
        _k(),
        dist_coeffs=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    )
    assert reason == "OK"
    assert pixel is not None


def test_projection_rejects_behind_camera_and_out_of_image():
    behind, reason = project_camera_point((0.1, 0.0, -0.5), _k())
    assert behind is None
    assert reason == "BEHIND_CAMERA"
    outside, reason = project_camera_point(
        (2.0, 0.0, 1.0), _k(), frame_shape=(480, 640)
    )
    assert outside is None
    assert reason == "OUT_OF_IMAGE"


def test_projection_rejects_non_finite_and_invalid_camera_matrix():
    bad, reason = project_camera_point((np.nan, 0.0, 1.0), _k())
    assert bad is None
    assert reason == "INVALID_POINT"
    bad, reason = project_camera_point((0.0, 0.0, 1.0), np.eye(2))
    assert bad is None
    assert reason == "INVALID_CAMERA_MATRIX"


def test_transform_direction_is_explicit_parent_child():
    t_parent_child = np.eye(4)
    t_parent_child[:3, 3] = [1.0, 2.0, 3.0]
    child = np.array([0.1, 0.2, 0.3])
    parent = transform_point(t_parent_child, child)
    assert parent == pytest.approx([1.1, 2.2, 3.3])
    inverse = np.linalg.inv(t_parent_child)
    restored = transform_point(inverse, parent)
    assert restored == pytest.approx(child)


def test_target_frame_contract_distinguishes_canonical_raw_and_unknown():
    ok, reason = target_frame_center_state(CANONICAL_TARGET_FRAME)
    assert ok is True
    assert reason == "OK"
    raw_ok, raw_reason = target_frame_center_state(
        RAW_GEMINI_TARGET_FRAME, require_canonical=False
    )
    assert raw_ok is True
    assert raw_reason == "RAW_GEMINI_ALIAS"
    centered, centered_reason = target_frame_center_state(
        RAW_GEMINI_TARGET_FRAME, require_canonical=True
    )
    assert centered is False
    assert centered_reason == "RAW_FRAME_NO_CENTER_CONTRACT"
    unknown, unknown_reason = target_frame_center_state("some_other_frame")
    assert unknown is False
    assert unknown_reason == "UNKNOWN_FRAME"
