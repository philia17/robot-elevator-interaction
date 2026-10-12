"""ROS-independent invariants for the Gemini 335 canonical adapter."""

from __future__ import annotations

import math
from typing import Iterable


CANONICAL_BODY_FRAME = 'camera_link'
CANONICAL_OPTICAL_FRAME = 'camera_optical_frame'
BODY_TO_OPTICAL_TRANSLATION_M = (0.0, 0.0, 0.0)
BODY_TO_OPTICAL_ROTATION_XYZW = (-0.5, 0.5, -0.5, 0.5)
EXPECTED_RAW_OPTICAL_FRAMES = frozenset(
    {'gemini335_color_optical_frame', 'gemini335_depth_optical_frame'}
)


class AdapterInvariantError(ValueError):
    """Raised when raw camera data cannot be normalized safely."""


def validate_raw_optical_frame(frame_id: str, allowed_frames: Iterable[str]) -> str:
    """Return a validated vendor optical frame or fail closed."""
    normalized = str(frame_id).strip()
    allowed = frozenset(str(value).strip() for value in allowed_frames)
    if not normalized:
        raise AdapterInvariantError('raw header.frame_id is empty')
    if normalized not in allowed:
        raise AdapterInvariantError(
            f'unexpected raw optical frame {normalized!r}; allowed={sorted(allowed)!r}'
        )
    return normalized


def canonical_frame_for_raw(frame_id: str, allowed_frames: Iterable[str]) -> str:
    """Validate a vendor optical frame and map it to the frozen canonical name."""
    validate_raw_optical_frame(frame_id, allowed_frames)
    return CANONICAL_OPTICAL_FRAME


def quaternion_to_matrix_xyzw(quaternion: Iterable[float]) -> tuple[tuple[float, ...], ...]:
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise AdapterInvariantError('quaternion must contain four finite XYZW values')
    x, y, z, w = values
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise AdapterInvariantError(f'quaternion must be unit length; norm={norm}')
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def body_vectors_for_optical_basis() -> tuple[tuple[float, ...], ...]:
    """Map optical +X/+Y/+Z unit vectors into canonical body coordinates."""
    matrix = quaternion_to_matrix_xyzw(BODY_TO_OPTICAL_ROTATION_XYZW)
    return tuple(tuple(matrix[row][column] for row in range(3)) for column in range(3))
