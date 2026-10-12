"""Pure display projection helpers for the Button-1 dual-camera Live panel.

This module intentionally has no ROS, cv2, or motion imports.  It implements
the pinhole + OpenCV-style distortion math used by the diagnostic overlays so
the projection and frame contracts can be unit-tested offline.

The wrist USB camera is a wide-angle VGA module with strong barrel distortion;
projecting with ``K`` alone visibly displaces markers.  Callers therefore pass
the solver's ``dist_coeffs`` when projecting wrist points.  The fixed Gemini
projection can also pass its CameraInfo distortion vector when one is
available.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


CANONICAL_TARGET_FRAME = "camera_optical_frame"
RAW_GEMINI_TARGET_FRAME = "gemini335_color_optical_frame"
RAW_GEMINI_TO_CANONICAL_ALIAS_ID = (
    "gemini335_color_optical_frame_to_camera_optical_frame_identity_v1"
)


def _finite3(point: Sequence[float]) -> tuple[float, float, float] | None:
    if len(point) != 3:
        return None
    values = tuple(float(value) for value in point)
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _distortion_array(dist_coeffs: Sequence[float] | None) -> np.ndarray:
    if dist_coeffs is None:
        return np.zeros(8, dtype=float)
    array = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    if array.size == 0:
        return np.zeros(8, dtype=float)
    result = np.zeros(8, dtype=float)
    result[: min(array.size, 8)] = array[:8]
    return result


def _distort_normalized(x: float, y: float, dist: np.ndarray) -> tuple[float, float] | None:
    """Apply OpenCV's radial-tangential (and rational) distortion model."""
    k1, k2, p1, p2, k3, k4, k5, k6 = (float(v) for v in dist[:8])
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    if abs(k4) > 1e-12 or abs(k5) > 1e-12 or abs(k6) > 1e-12:
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        if not math.isfinite(denominator) or abs(denominator) < 1e-12:
            return None
        radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / denominator
    else:
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    if not (math.isfinite(xd) and math.isfinite(yd)):
        return None
    return xd, yd


def project_camera_point(
    point: Sequence[float],
    camera_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    dist_coeffs: Sequence[float] | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, str]:
    """Project one camera-optical point, returning (pixel, reason).

    ``frame_shape`` is ``(height, width)``; when supplied the result rejects
    pixels outside the image so the UI never draws a marker that is not in the
    current frame.
    """
    values = _finite3(point)
    if values is None:
        return None, "INVALID_POINT"
    x, y, z = values
    if z <= 1e-6:
        return None, "BEHIND_CAMERA"
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (3, 3):
        return None, "INVALID_CAMERA_MATRIX"
    dist = _distortion_array(dist_coeffs)
    xd, yd = _distort_normalized(x / z, y / z, dist)
    if xd is None or yd is None:
        return None, "INVALID_DISTORTION"
    u = float(matrix[0, 0]) * xd + float(matrix[0, 2])
    v = float(matrix[1, 1]) * yd + float(matrix[1, 2])
    if not (math.isfinite(u) and math.isfinite(v)):
        return None, "INVALID_PROJECTION"
    pixel = int(round(u)), int(round(v))
    if frame_shape is not None:
        height, width = frame_shape
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            return None, "OUT_OF_IMAGE"
    return pixel, "OK"


def transform_point(
    t_parent_child: np.ndarray | Sequence[Sequence[float]],
    point_child: Sequence[float],
) -> np.ndarray:
    """Apply ``T_parent_child`` to a point expressed in the child frame."""
    matrix = np.asarray(t_parent_child, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("t_parent_child must be a 4x4 homogeneous transform")
    values = _finite3(point_child)
    if values is None:
        raise ValueError("point_child must be a finite length-3 vector")
    point = np.asarray(values, dtype=float)
    return matrix[:3, :3] @ point + matrix[:3, 3]


def target_frame_center_state(
    frame_id: str | None,
    *,
    require_canonical: bool = True,
) -> tuple[bool, str]:
    """Return whether a target frame can support a *centered* wrist marker.

    The coordinate contract only allows the canonical ``camera_optical_frame``
    for target poses.  The live localizer still emits raw
    ``gemini335_color_optical_frame`` headers with a software-identity alias
    whose physical axes are pending, so those frames are explicitly
    diagnostic-only here.
    """
    if frame_id is None or not str(frame_id).strip():
        return False, "SOURCE_FRAME_MISSING"
    frame = str(frame_id)
    if frame == CANONICAL_TARGET_FRAME:
        return True, "OK"
    if frame == RAW_GEMINI_TARGET_FRAME:
        if require_canonical:
            return False, "RAW_FRAME_NO_CENTER_CONTRACT"
        return True, "RAW_GEMINI_ALIAS"
    return False, "UNKNOWN_FRAME"
