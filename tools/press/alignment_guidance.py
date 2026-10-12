#!/usr/bin/env python3
"""Pure, display-only guidance for the fixed-pose elevator press trial.

The helper deliberately does not know about ROS, inverse kinematics, or motor
commands.  It turns the physical depth-tip (P) minus selected button (A)
residual into two *operator suggestions* only:

* vertical height (raise/lower), and
* panel-normal pre-contact standoff (advance/retract).

Panel-tangent horizontal residual is reported for diagnosis, but is never
turned into a suggested adjustment in this simplified, face-on workflow.
"""

from __future__ import annotations

import math

import numpy as np


def panel_tangent_residual(point_camera, target: dict) -> tuple[float, float, float]:
    """Return P-A residual in camera-horizontal, camera-vertical and normal axes.

    ``normal`` is the measured panel *outward* normal.  Thus a positive normal
    residual means P is further out from the panel than A (too much standoff).
    Positive camera-vertical means P projects lower than A in the live image.
    """
    point = np.asarray(point_camera, dtype=float)
    target_point = np.asarray(target["camera_xyz"], dtype=float)
    normal = np.asarray(target["normal"], dtype=float)
    if point.shape != (3,) or target_point.shape != (3,) or normal.shape != (3,):
        raise ValueError("point, target camera_xyz, and normal must be 3-vectors")
    if not (np.all(np.isfinite(point)) and np.all(np.isfinite(target_point)) and np.all(np.isfinite(normal))):
        raise ValueError("point, target camera_xyz, and normal must be finite")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        raise ValueError("panel normal must be non-zero")
    normal /= normal_norm
    camera_x = np.array([1.0, 0.0, 0.0])
    horizontal = camera_x - float(np.dot(camera_x, normal)) * normal
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.array([0.0, 1.0, 0.0])
        horizontal -= float(np.dot(horizontal, normal)) * normal
    horizontal /= np.linalg.norm(horizontal)
    camera_y = np.array([0.0, 1.0, 0.0])
    vertical = camera_y - float(np.dot(camera_y, normal)) * normal
    vertical -= float(np.dot(vertical, horizontal)) * horizontal
    if np.linalg.norm(vertical) < 1e-6:
        vertical = np.cross(normal, horizontal)
    vertical /= np.linalg.norm(vertical)
    residual = point - target_point
    return (
        float(np.dot(residual, horizontal)),
        float(np.dot(residual, vertical)),
        float(np.dot(residual, normal)),
    )


def fixed_pose_alignment_guidance(
    point_camera,
    target: dict | None,
    *,
    p_available: bool,
    p_stable: bool,
    depth_live: bool,
    vertical_deadband_m: float = 0.003,
    normal_deadband_m: float = 0.005,
) -> dict:
    """Return bounded, display-only fixed-pose alignment guidance.

    No result represents an executable robot command.  A non-ready result is
    intentionally explicit so callers do not infer an adjustment from missing,
    stale, or still-settling depth evidence.
    """
    if not p_available or point_camera is None:
        return {"ready": False, "reason": "P unavailable", "height": "WAIT", "standoff": "WAIT"}
    if not depth_live:
        return {"ready": False, "reason": "P/depth stale", "height": "WAIT", "standoff": "WAIT"}
    if not p_stable:
        return {"ready": False, "reason": "P not stable", "height": "WAIT", "standoff": "WAIT"}
    if target is None or not bool(target.get("depth_valid")) or not bool(target.get("normal_valid")):
        return {"ready": False, "reason": "A target plane/normal unavailable", "height": "WAIT", "standoff": "WAIT"}
    if vertical_deadband_m < 0.0 or normal_deadband_m < 0.0:
        raise ValueError("guidance deadbands must be non-negative")

    horizontal_m, vertical_m, normal_m = panel_tangent_residual(point_camera, target)
    if vertical_m > vertical_deadband_m:
        height = "RAISE (P projects below A)"
    elif vertical_m < -vertical_deadband_m:
        height = "LOWER (P projects above A)"
    else:
        height = "HOLD"
    if normal_m > normal_deadband_m:
        standoff = "ADVANCE (reduce pre-contact standoff)"
    elif normal_m < -normal_deadband_m:
        standoff = "RETRACT (increase pre-contact standoff)"
    else:
        standoff = "HOLD"
    return {
        "ready": True,
        "reason": "P stable; diagnostic guidance only",
        "horizontal_m": horizontal_m,
        "vertical_m": vertical_m,
        "normal_m": normal_m,
        "height": height,
        "standoff": standoff,
        "horizontal_policy": "OBSERVE ONLY (no lateral correction)",
    }
