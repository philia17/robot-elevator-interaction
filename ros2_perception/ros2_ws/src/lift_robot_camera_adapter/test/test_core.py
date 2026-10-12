import math

import pytest

from lift_robot_camera_adapter.core import (
    BODY_TO_OPTICAL_ROTATION_XYZW,
    CANONICAL_OPTICAL_FRAME,
    AdapterInvariantError,
    body_vectors_for_optical_basis,
    canonical_frame_for_raw,
    quaternion_to_matrix_xyzw,
)


def test_audited_color_frame_rewrites_to_canonical_optical_frame():
    assert canonical_frame_for_raw(
        'gemini335_color_optical_frame', ['gemini335_color_optical_frame']
    ) == CANONICAL_OPTICAL_FRAME


@pytest.mark.parametrize('raw_frame', ['', 'gemini335_link', 'camera_link', 'ascamera_hp60c_color_0'])
def test_unexpected_or_non_optical_raw_frames_fail_closed(raw_frame):
    with pytest.raises(AdapterInvariantError):
        canonical_frame_for_raw(raw_frame, ['gemini335_color_optical_frame'])


def test_frozen_body_to_optical_quaternion_is_unit_and_proper():
    matrix = quaternion_to_matrix_xyzw(BODY_TO_OPTICAL_ROTATION_XYZW)
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    assert math.isclose(determinant, 1.0, abs_tol=1e-9)


def test_optical_basis_maps_to_frozen_body_axes():
    optical_x, optical_y, optical_z = body_vectors_for_optical_basis()
    assert optical_x == (0.0, -1.0, 0.0)
    assert optical_y == (0.0, 0.0, -1.0)
    assert optical_z == (1.0, 0.0, 0.0)
