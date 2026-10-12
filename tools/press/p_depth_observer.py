#!/usr/bin/env python3
"""Markerless RGB-D fingertip observation helpers (display-only).

The model-projected fingertip is used only to bound the search.  The returned
point is computed from depth pixels that lie in front of the measured panel
plane.  This module never commands the robot.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PDepthEstimate:
    valid: bool
    point_camera_m: tuple[float, float, float]
    component_pixels: int
    anchor_distance_px: float
    signed_gap_m: float
    failure_reason: str


def invalid_estimate(reason: str) -> PDepthEstimate:
    return PDepthEstimate(
        valid=False,
        point_camera_m=(math.nan, math.nan, math.nan),
        component_pixels=0,
        anchor_distance_px=math.nan,
        signed_gap_m=math.nan,
        failure_reason=str(reason),
    )


def depth_to_meters(depth_image: np.ndarray, encoding: str) -> np.ndarray:
    """Convert canonical 16UC1 millimetres or 32FC1 metres to metres."""
    array = np.asarray(depth_image)
    if array.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    normalized = str(encoding or "").strip().upper()
    if normalized == "16UC1":
        converted = array.astype(np.float64) * 0.001
    elif normalized == "32FC1":
        converted = array.astype(np.float64, copy=True)
    else:
        raise ValueError(f"unsupported depth encoding: {encoding}")
    converted[~np.isfinite(converted) | (converted <= 0.0)] = np.nan
    return converted


def project_camera_to_pixel(point, fx, fy, cx, cy):
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    return (
        float(fx) * float(point[0]) / float(point[2]) + float(cx),
        float(fy) * float(point[1]) / float(point[2]) + float(cy),
    )


def _components_8(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components as local [row, column] arrays."""
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for start_row, start_col in np.argwhere(mask):
        if visited[start_row, start_col]:
            continue
        queue = deque([(int(start_row), int(start_col))])
        visited[start_row, start_col] = True
        pixels = []
        while queue:
            row, col = queue.pop()
            pixels.append((row, col))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        components.append(np.asarray(pixels, dtype=np.int32))
    return components


def observe_fingertip_from_depth(
    depth_image: np.ndarray,
    encoding: str,
    *,
    panel_point_camera_m,
    panel_outward_normal_camera,
    predicted_tip_camera_m,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    roi_radius_px: int = 90,
    minimum_foreground_gap_m: float = 0.006,
    maximum_foreground_gap_m: float = 0.18,
    minimum_component_pixels: int = 20,
    maximum_anchor_distance_px: float = 45.0,
    nearest_plane_quantile: float = 0.20,
) -> PDepthEstimate:
    """Observe the physical tip from a unique panel-front depth component.

    The nearest-plane quantile of the selected component represents its end
    closest to the panel, which is the relevant press point.  The prediction
    only selects a local connected component and is never returned directly.
    """
    if not (float(fx) > 0.0 and float(fy) > 0.0):
        return invalid_estimate("invalid camera intrinsics")
    if int(roi_radius_px) < 8 or int(minimum_component_pixels) < 1:
        return invalid_estimate("invalid observer thresholds")
    if not 0.0 < float(minimum_foreground_gap_m) < float(maximum_foreground_gap_m):
        return invalid_estimate("invalid foreground gap range")
    if not 0.0 < float(nearest_plane_quantile) <= 0.5:
        return invalid_estimate("invalid nearest-plane quantile")

    panel_point = np.asarray(panel_point_camera_m, dtype=np.float64)
    normal = np.asarray(panel_outward_normal_camera, dtype=np.float64)
    predicted_tip = np.asarray(predicted_tip_camera_m, dtype=np.float64)
    if any(value.shape != (3,) for value in (panel_point, normal, predicted_tip)):
        return invalid_estimate("camera points must be 3-vectors")
    if not all(np.all(np.isfinite(value)) for value in (panel_point, normal, predicted_tip)):
        return invalid_estimate("camera geometry is non-finite")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return invalid_estimate("panel normal is degenerate")
    normal = normal / normal_norm
    # Existing panel geometry defines outward as camera-facing.  Re-orient a
    # numerically flipped input rather than silently searching behind the panel.
    if float(np.dot(normal, panel_point)) > 0.0:
        normal = -normal

    anchor = project_camera_to_pixel(predicted_tip, fx, fy, cx, cy)
    if anchor is None:
        return invalid_estimate("predicted search anchor is not projectable")
    try:
        depth_m = depth_to_meters(depth_image, encoding)
    except (TypeError, ValueError) as exc:
        return invalid_estimate(str(exc))
    height, width = depth_m.shape
    anchor_u, anchor_v = anchor
    if not (0.0 <= anchor_u < width and 0.0 <= anchor_v < height):
        return invalid_estimate("predicted search anchor is outside depth image")

    radius = int(roi_radius_px)
    left = max(0, int(math.floor(anchor_u)) - radius)
    right = min(width, int(math.floor(anchor_u)) + radius + 1)
    top = max(0, int(math.floor(anchor_v)) - radius)
    bottom = min(height, int(math.floor(anchor_v)) + radius + 1)
    depths = depth_m[top:bottom, left:right]
    rows, columns = np.mgrid[top:bottom, left:right]
    finite = np.isfinite(depths)
    points = np.empty(depths.shape + (3,), dtype=np.float64)
    points[..., 0] = (columns - float(cx)) * depths / float(fx)
    points[..., 1] = (rows - float(cy)) * depths / float(fy)
    points[..., 2] = depths
    signed_gap = np.einsum("...i,i->...", points - panel_point, normal)
    foreground = (
        finite
        & np.isfinite(signed_gap)
        & (signed_gap >= float(minimum_foreground_gap_m))
        & (signed_gap <= float(maximum_foreground_gap_m))
    )

    components = [
        component
        for component in _components_8(foreground)
        if component.shape[0] >= int(minimum_component_pixels)
    ]
    if not components:
        return invalid_estimate("no panel-front depth component near F")

    ranked = []
    for component in components:
        global_rows = component[:, 0] + top
        global_cols = component[:, 1] + left
        distances = np.hypot(global_cols - anchor_u, global_rows - anchor_v)
        ranked.append((float(np.min(distances)), -int(component.shape[0]), component))
    ranked.sort(key=lambda item: (item[0], item[1]))
    best_distance, _negative_area, best = ranked[0]
    if best_distance > float(maximum_anchor_distance_px):
        return invalid_estimate("nearest depth component is too far from F")
    if len(ranked) > 1:
        second_distance, second_negative_area, _second = ranked[1]
        best_area = int(best.shape[0])
        second_area = -int(second_negative_area)
        if second_distance <= best_distance + 8.0 and second_area >= 0.5 * best_area:
            return invalid_estimate("multiple plausible depth components near F")

    component_points = points[best[:, 0], best[:, 1]]
    component_gaps = signed_gap[best[:, 0], best[:, 1]]
    cutoff = float(np.quantile(component_gaps, float(nearest_plane_quantile)))
    tip_band = component_points[component_gaps <= cutoff]
    if tip_band.shape[0] < 3:
        return invalid_estimate("tip nearest-plane band has too few pixels")
    tip = np.median(tip_band, axis=0)
    tip_gap = float(np.dot(tip - panel_point, normal))
    if not (np.all(np.isfinite(tip)) and math.isfinite(tip_gap)):
        return invalid_estimate("observed tip is non-finite")
    return PDepthEstimate(
        valid=True,
        point_camera_m=tuple(float(value) for value in tip),
        component_pixels=int(best.shape[0]),
        anchor_distance_px=float(best_distance),
        signed_gap_m=tip_gap,
        failure_reason="",
    )


def temporal_stability(points, required_frames: int = 5, maximum_spread_m: float = 0.003):
    """Return stable flag, robust point and maximum distance from its median."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or array.shape[0] == 0:
        return False, np.full(3, np.nan), math.nan
    center = np.median(array, axis=0)
    spread = float(np.max(np.linalg.norm(array - center, axis=1)))
    stable = bool(array.shape[0] >= int(required_frames) and spread <= float(maximum_spread_m))
    return stable, center, spread
