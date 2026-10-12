#!/usr/bin/env python3
"""Button-1 panel-tangent (XY) align helpers — no ROS, no motion.

LOCK freezes a semantic button-1 target (A + N). Align motion mirrors Quick
Control Cartesian teleop: one ``roarm_base_link`` tip axis per tick, with the
other two axes frozen at hold-start (panel normal / depth held when it is a
non-active axis). Used by the LIVE LOCK / START XY ALIGN panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from press.button_target import normalize_button_label, target_identity_matches

TARGET_LABEL = "1"
DEFAULT_DEADBAND_M = 0.005
DEFAULT_STEP_M = 0.003
DEFAULT_MAX_STEP_M = 0.003
AXIS_NAMES = ("X", "Y", "Z")


@dataclass(frozen=True)
class LockedButton1:
    """Frozen desired button in camera optical frame (+ optional base)."""

    label: str
    camera_xyz_m: tuple[float, float, float]
    normal_camera: tuple[float, float, float]
    bbox: tuple[float, float, float, float] | None = None
    source_status: str = ""
    confirmations: int = 0
    base_xyz_m: tuple[float, float, float] | None = None
    normal_base: tuple[float, float, float] | None = None

    def as_target_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "camera_xyz": self.camera_xyz_m,
            "normal": self.normal_camera,
            "depth_valid": True,
            "normal_valid": True,
        }

    def with_base_frame(
        self,
        *,
        base_xyz_m: Sequence[float],
        normal_base: Sequence[float],
    ) -> "LockedButton1":
        return LockedButton1(
            label=self.label,
            camera_xyz_m=self.camera_xyz_m,
            normal_camera=self.normal_camera,
            bbox=self.bbox,
            source_status=self.source_status,
            confirmations=self.confirmations,
            base_xyz_m=_finite3(base_xyz_m, "base_xyz"),
            normal_base=_finite3(normal_base, "normal_base"),
        )


def _finite3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 elements")
    out = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out


def panel_tangent_basis(
    normal_camera: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (u, v, n_out) in camera frame."""
    normal = np.asarray(_finite3(normal_camera, "normal"), dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ValueError("panel normal must be non-zero")
    n_out = normal / norm
    camera_x = np.array([1.0, 0.0, 0.0], dtype=float)
    horizontal = camera_x - float(np.dot(camera_x, n_out)) * n_out
    if float(np.linalg.norm(horizontal)) < 1e-6:
        horizontal = np.array([0.0, 1.0, 0.0], dtype=float)
        horizontal -= float(np.dot(horizontal, n_out)) * n_out
    u = horizontal / np.linalg.norm(horizontal)
    camera_y = np.array([0.0, 1.0, 0.0], dtype=float)
    vertical = camera_y - float(np.dot(camera_y, n_out)) * n_out
    vertical -= float(np.dot(vertical, u)) * u
    if float(np.linalg.norm(vertical)) < 1e-6:
        vertical = np.cross(n_out, u)
    v = vertical / np.linalg.norm(vertical)
    return u, v, n_out


def tangent_residual_camera(
    tip_camera_m: Sequence[float],
    target_camera_xyz_m: Sequence[float],
    normal_camera: Sequence[float],
) -> dict[str, Any]:
    """Decompose tip−A into panel u/v/normal (camera frame)."""
    tip = np.asarray(_finite3(tip_camera_m, "tip"), dtype=float)
    target = np.asarray(_finite3(target_camera_xyz_m, "target"), dtype=float)
    u, v, n_out = panel_tangent_basis(normal_camera)
    residual = tip - target
    e_u = float(np.dot(residual, u))
    e_v = float(np.dot(residual, v))
    e_n = float(np.dot(residual, n_out))
    return {
        "e_u_m": e_u,
        "e_v_m": e_v,
        "e_n_m": e_n,
        "tangent_norm_m": math.hypot(e_u, e_v),
        "u_camera": (float(u[0]), float(u[1]), float(u[2])),
        "v_camera": (float(v[0]), float(v[1]), float(v[2])),
        "n_camera": (float(n_out[0]), float(n_out[1]), float(n_out[2])),
    }


def lock_button1_from_m0_summary(
    summary: Mapping[str, Any] | None,
    *,
    allow_unready: bool = False,
    target_label: str = TARGET_LABEL,
    detection_id: str | None = None,
    semantic_track_id: str | None = None,
) -> LockedButton1:
    """Freeze the selected M0 target; legacy callers still request button 1."""
    if summary is None:
        raise ValueError("no M0 target to lock")
    label = normalize_button_label(target_label)
    if not target_identity_matches(
        summary, target_label=target_label, detection_id=detection_id,
        semantic_track_id=semantic_track_id,
    ):
        raise ValueError(f"expected button label/identity {target_label!r}, got {summary.get('label')!r}")
    if summary.get("pressable") is False or (
        (label != TARGET_LABEL or semantic_track_id) and summary.get("pressable") is not True
    ):
        raise ValueError("selected button is not confirmed pressable")
    if not bool(summary.get("ready")) and not allow_unready:
        raise ValueError("button 1 is not READY")
    if not bool(summary.get("depth_valid", True)):
        raise ValueError("button 1 depth_valid is false")
    if not bool(summary.get("normal_valid")):
        raise ValueError("button 1 normal is invalid")
    xyz = summary.get("camera_xyz")
    normal = summary.get("normal")
    if not isinstance(xyz, (list, tuple)) or not isinstance(normal, (list, tuple)):
        raise ValueError("camera_xyz and normal are required")
    # Reject empty M0 placeholders published while SEMANTIC_SET_INCOMPLETE.
    if all(abs(float(v)) < 1e-9 for v in xyz):
        raise ValueError("button 1 camera_xyz looks empty (M0 not geometrically ready)")
    bbox = summary.get("bbox")
    bbox_t = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox_t = tuple(float(v) for v in bbox)  # type: ignore[assignment]
    return LockedButton1(
        label=label,
        camera_xyz_m=_finite3(xyz, "camera_xyz"),
        normal_camera=_finite3(normal, "normal"),
        bbox=bbox_t,
        source_status=str(summary.get("status") or ""),
        confirmations=int(summary.get("confirmations") or 0),
    )


def lock_button1_from_semantics_and_observations(
    semantics: Sequence[Mapping[str, Any]] | None,
    observations: Sequence[Mapping[str, Any]] | None,
    *,
    target_label: str = TARGET_LABEL,
    detection_id: str | None = None,
    semantic_track_id: str | None = None,
) -> LockedButton1:
    """Freeze button 1 by joining VLM label「1」to its 3D observation.

    Used when M0 stays SEMANTIC_SET_INCOMPLETE because other visible labels are
    stale, while button 1 itself already has depth + plane.
    """
    if not semantics:
        raise ValueError("no button semantics")
    if not observations:
        raise ValueError("no 3D observations")
    matches = [
        item
        for item in semantics
        if target_identity_matches(
            item, target_label=target_label, detection_id=detection_id,
            semantic_track_id=semantic_track_id,
        )
    ]
    if not matches:
        raise ValueError(f"VLM/semantics has no selected label {target_label}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous label {target_label} count={len(matches)}")
    semantic = matches[0]
    if semantic.get("pressable") is not True:
        raise ValueError("selected button is not confirmed pressable")
    det_id = str(semantic.get("detection_id") or "")
    if not det_id:
        raise ValueError("label 1 missing detection_id")
    matching_observations = [
        item for item in observations if str(item.get("detection_id") or "") == det_id
    ]
    if not matching_observations:
        raise ValueError(f"no 3D observation for detection_id={det_id}")
    if len(matching_observations) != 1:
        raise ValueError(f"ambiguous 3D observation for detection_id={det_id}")
    obs = matching_observations[0]
    if not bool(obs.get("depth_valid")):
        raise ValueError("button 1 observation depth_valid is false")
    surface = obs.get("surface_geometry") or {}
    if not bool(surface.get("estimate_valid") or surface.get("normal_valid")):
        # Support both raw ROS dicts and flattened summaries.
        if "panel_outward_normal_x" in surface:
            if not bool(surface.get("estimate_valid")):
                raise ValueError("button 1 plane/normal estimate_valid is false")
        else:
            raise ValueError("button 1 surface geometry missing")
    xyz = (
        float(obs.get("x_m", surface.get("surface_point_x_m", math.nan))),
        float(obs.get("y_m", surface.get("surface_point_y_m", math.nan))),
        float(obs.get("z_m", surface.get("surface_point_z_m", math.nan))),
    )
    if "panel_outward_normal_x" in surface:
        normal = (
            float(surface["panel_outward_normal_x"]),
            float(surface["panel_outward_normal_y"]),
            float(surface["panel_outward_normal_z"]),
        )
    else:
        normal = tuple(float(v) for v in surface.get("normal", ()))  # type: ignore[misc]
        if len(normal) != 3:
            raise ValueError("button 1 normal missing")
    bbox = (
        float(obs["x_min"]),
        float(obs["y_min"]),
        float(obs["x_max"]),
        float(obs["y_max"]),
    )
    return LockedButton1(
        label=normalize_button_label(target_label),
        camera_xyz_m=_finite3(xyz, "camera_xyz"),
        normal_camera=_finite3(normal, "normal"),
        bbox=bbox,
        source_status="SEMANTICS_OBS_JOIN",
        confirmations=1,
    )


def tangent_delta_camera_m(
    tip_camera_m: Sequence[float],
    locked: LockedButton1,
    *,
    deadband_m: float = DEFAULT_DEADBAND_M,
    step_m: float = DEFAULT_STEP_M,
    max_step_m: float = DEFAULT_MAX_STEP_M,
) -> dict[str, Any]:
    """Compute a single tangent-only tip delta in camera frame (toward A).

    Normal component is forced to zero. Step length is min(error, step_m)
    and never exceeds max_step_m.
    """
    if deadband_m < 0.0 or step_m <= 0.0 or max_step_m <= 0.0:
        raise ValueError("deadband/step/max_step must be non-negative / positive")
    step_cap = min(float(step_m), float(max_step_m))
    residual = tangent_residual_camera(
        tip_camera_m, locked.camera_xyz_m, locked.normal_camera
    )
    e_u = residual["e_u_m"]
    e_v = residual["e_v_m"]
    tangent_norm = residual["tangent_norm_m"]
    aligned = tangent_norm <= float(deadband_m)
    if aligned or tangent_norm < 1e-9:
        return {
            "aligned": aligned,
            "delta_camera_m": (0.0, 0.0, 0.0),
            "step_m": 0.0,
            "residual": residual,
            "reason": "ALIGNED" if aligned else "zero residual",
        }
    # Move tip toward A in tangent plane: delta = -(e_u u + e_v v)
    u = np.asarray(residual["u_camera"], dtype=float)
    v = np.asarray(residual["v_camera"], dtype=float)
    full = -(e_u * u + e_v * v)
    # Numerical guard: kill any leaked normal component.
    n = np.asarray(residual["n_camera"], dtype=float)
    full = full - float(np.dot(full, n)) * n
    full_norm = float(np.linalg.norm(full))
    if full_norm < 1e-12:
        return {
            "aligned": False,
            "delta_camera_m": (0.0, 0.0, 0.0),
            "step_m": 0.0,
            "residual": residual,
            "reason": "tangent delta vanished after normal purge",
        }
    scale = min(1.0, step_cap / full_norm)
    delta = full * scale
    # Re-purge after scale.
    delta = delta - float(np.dot(delta, n)) * n
    if abs(float(np.dot(delta, n))) > 1e-9:
        raise RuntimeError("normal leak in tangent delta")
    return {
        "aligned": False,
        "delta_camera_m": (float(delta[0]), float(delta[1]), float(delta[2])),
        "step_m": float(np.linalg.norm(delta)),
        "residual": residual,
        "reason": "STEP",
    }


def camera_delta_to_base(
    delta_camera_m: Sequence[float],
    rotation_base_from_camera: np.ndarray,
) -> tuple[float, float, float]:
    """Rotate a camera-frame translation into roarm_base_link."""
    r = np.asarray(rotation_base_from_camera, dtype=float)
    if r.shape != (3, 3):
        raise ValueError("rotation_base_from_camera must be 3x3")
    d = r @ np.asarray(_finite3(delta_camera_m, "delta_camera"), dtype=float)
    return (float(d[0]), float(d[1]), float(d[2]))


def camera_point_to_base(
    point_camera_m: Sequence[float],
    t_base_cam: np.ndarray,
) -> tuple[float, float, float]:
    """Transform a camera-optical point into roarm_base_link."""
    t = np.asarray(t_base_cam, dtype=float)
    if t.shape != (4, 4):
        raise ValueError("t_base_cam must be 4x4")
    p = np.asarray(_finite3(point_camera_m, "point_camera"), dtype=float)
    out = t[:3, :3] @ p + t[:3, 3]
    return (float(out[0]), float(out[1]), float(out[2]))


def camera_dir_to_base(
    direction_camera: Sequence[float],
    t_base_cam: np.ndarray,
) -> tuple[float, float, float]:
    """Rotate a camera-optical direction into roarm_base_link (unit)."""
    t = np.asarray(t_base_cam, dtype=float)
    if t.shape != (4, 4):
        raise ValueError("t_base_cam must be 4x4")
    d = t[:3, :3] @ np.asarray(_finite3(direction_camera, "direction_camera"), dtype=float)
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise ValueError("direction maps to zero in base")
    d = d / norm
    return (float(d[0]), float(d[1]), float(d[2]))


def tangent_error_base(
    tip_base_m: Sequence[float],
    target_base_m: Sequence[float],
    normal_base: Sequence[float],
) -> dict[str, Any]:
    """Panel-tangent residual tip−A in roarm_base_link."""
    tip = np.asarray(_finite3(tip_base_m, "tip_base"), dtype=float)
    target = np.asarray(_finite3(target_base_m, "target_base"), dtype=float)
    normal = np.asarray(_finite3(normal_base, "normal_base"), dtype=float)
    n_norm = float(np.linalg.norm(normal))
    if n_norm < 1e-9:
        raise ValueError("normal_base must be non-zero")
    n_out = normal / n_norm
    residual = tip - target
    e_n = float(np.dot(residual, n_out))
    e_t = residual - e_n * n_out
    return {
        "e_base_m": (float(e_t[0]), float(e_t[1]), float(e_t[2])),
        "e_n_m": e_n,
        "tangent_norm_m": float(np.linalg.norm(e_t)),
        "n_base": (float(n_out[0]), float(n_out[1]), float(n_out[2])),
    }


def qc_style_tangent_axis_step(
    tip_base_m: Sequence[float],
    target_base_m: Sequence[float],
    normal_base: Sequence[float],
    tip_lock_base_m: Sequence[float],
    *,
    deadband_m: float = DEFAULT_DEADBAND_M,
    step_m: float = DEFAULT_STEP_M,
    axis_order: Sequence[int] = (1, 2),
) -> dict[str, Any]:
    """One Quick-Control-style Cartesian tip step (default: Y then Z).

    Matches Quick Control base axes (+X forward, +Y left, +Z up) / WASD+JK map.
    Default Task-1 order is Y (A/D left-right) then Z (J/K up-down); X stays at
    tip_lock. ``axis_order`` lists allowed base-axis indices in priority order.
    """
    if deadband_m < 0.0 or step_m <= 0.0:
        raise ValueError("deadband/step must be non-negative / positive")
    if not axis_order:
        raise ValueError("axis_order must be non-empty")
    for axis in axis_order:
        if int(axis) not in (0, 1, 2):
            raise ValueError("axis_order entries must be 0/1/2")
    tip = _finite3(tip_base_m, "tip_base")
    tip_lock = _finite3(tip_lock_base_m, "tip_lock")
    err = tangent_error_base(tip, target_base_m, normal_base)
    e_t = err["e_base_m"]
    # Done when every allowed axis is inside the deadband (X ignored by default).
    ordered = [int(a) for a in axis_order]
    xy_norm = math.sqrt(sum(float(e_t[i]) ** 2 for i in ordered))
    if xy_norm <= float(deadband_m):
        return {
            "aligned": True,
            "axis": None,
            "direction": 0,
            "axis_name": "",
            "goal_base_m": tip_lock,
            "step_m": 0.0,
            "residual": err,
            "plane_norm_m": xy_norm,
            "reason": "ALIGNED",
        }
    # Priority order: first axis still outside deadband (Y before Z).
    axis = None
    for candidate in ordered:
        if abs(e_t[candidate]) > float(deadband_m):
            axis = candidate
            break
    if axis is None:
        # All allowed axes inside deadband but hypot > deadband (numerical).
        axis = max(ordered, key=lambda i: abs(e_t[i]))
    if abs(e_t[axis]) < 1e-9:
        return {
            "aligned": False,
            "axis": None,
            "direction": 0,
            "axis_name": "",
            "goal_base_m": tip_lock,
            "step_m": 0.0,
            "residual": err,
            "plane_norm_m": xy_norm,
            "reason": "no motion on allowed QC axes",
        }
    direction = -1 if e_t[axis] > 0.0 else 1
    step = min(float(step_m), abs(e_t[axis]))
    goal = [float(tip_lock[0]), float(tip_lock[1]), float(tip_lock[2])]
    goal[axis] = float(tip[axis]) + direction * step
    return {
        "aligned": False,
        "axis": axis,
        "direction": direction,
        "axis_name": AXIS_NAMES[axis],
        "goal_base_m": (goal[0], goal[1], goal[2]),
        "step_m": step,
        "residual": err,
        "plane_norm_m": xy_norm,
        "reason": "STEP",
    }
