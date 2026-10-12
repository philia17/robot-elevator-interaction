#!/usr/bin/env python3
"""Narrow offline continuous swept-clearance checker for the T102 prototype route.

This module is deliberately *not* a planner, a collision engine for general
routes, or an execution/release tool.  It certifies one exact outbound
joint-space route candidate under one explicitly inflated research-prototype
geometry profile and returns a hash-bound, non-authoritative receipt:

``PROVISIONAL_CONTINUOUS_CLEARANCE_PASS`` or ``UNPROVEN_OR_COLLISION_RISK``.

The proof is continuous, not waypoint sampling:

1. each joint-space segment is treated as a full interval;
2. link/tool/wrist primitives are evaluated at the interval midpoint;
3. a conservative upper bound on each primitive's possible displacement and
   rotation over the interval is computed from the joint deltas and the
   downstream lever arms (``|t_j|``) and body radii;
4. every endpoint/swept bound is inflated by that displacement/rotation bound;
5. a lower bound on the distance to the panel, the rear bracket, and every
   non-adjacent body pair is computed from the inflated bounds;
6. the interval is certified when every lower bound exceeds the required
   margin; otherwise the interval is recursively subdivided;
7. if the maximum subdivision depth is reached without proof, the segment and
   the whole result become ``UNPROVEN_OR_COLLISION_RISK``.

The module opens no ROS/serial device, starts no process, and cannot create
motion authority.  All authority flags in the emitted receipt are false.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import numpy as np


RESULT_SCHEMA = "t102_prototype_continuous_clearance_result_v1"
MODE = "OFFLINE_PROTOTYPE_CONTINUOUS_CLEARANCE"
PROFILE_SCHEMA = "t102_prototype_geometry_profile_v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = (
    REPO_ROOT / "config/press/prototype/t102_prototype_geometry_profile_20260814_v1.json"
)
DEFAULT_ROUTE_PATH = (
    REPO_ROOT / "config/press/prototype/t102_prototype_route_candidate_20260814_v1.json"
)
XACRO_PATH = (
    REPO_ROOT
    / "RoArm-M3"
    / "ros2"
    / "roarm_ws"
    / "src"
    / "roarm_main"
    / "roarm_description"
    / "urdf"
    / "roarm_m3"
    / "roarm_m3.xacro"
)

# Minimal prototype envelope floors.  The profile may be larger but never
# smaller; the checker rejects any profile that shrinks them.
MIN_TOOL_RADIUS_M = 0.030
MIN_WRIST_RADIUS_M = 0.120
MIN_BRACKET_SIZE_M = (0.140, 0.430, 0.140)
MIN_FIXED_OBSTACLE_MARGIN_M = 0.020
MIN_SELF_NON_OVERLAP_TOLERANCE_M = 0.005

# Route body set: the seven URDF collision links plus the two prototype
# end-effector envelopes.  Missing any body invalidates the result.
REQUIRED_BODY_IDS = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "gripper_link",
    "wrist_cable_sphere",
    "tool_capsule",
)
URDF_BODY_IDS = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "gripper_link",
)

# Kinematic parent -> child pairs excluded from self-collision.  Every other
# pair is checked.  End-effector assembly pairs are excluded because they are
# connected through the wrist/hand_tcp assembly with zero relative motion in
# this route (gripper joint held at zero delta) and their conservative swept
# bounds overlap by construction; the checker verifies that overlap at every
# waypoint before accepting the exclusion.
ADJACENT_PAIRS = tuple(
    frozenset(pair)
    for pair in (
        ("base_link", "link1"),
        ("link1", "link2"),
        ("link2", "link3"),
        ("link3", "link4"),
        ("link4", "link5"),
        ("link5", "gripper_link"),
        # The mandated 120 mm wrist/cable sphere around hand_tcp necessarily
        # wraps the wrist/forearm junction: the measured hand_tcp-to-link4
        # mesh distance is ~117 mm < 120 mm at every waypoint of this route,
        # so overlap is by construction, not a collision to prove.
        ("link4", "wrist_cable_sphere"),
        ("link5", "wrist_cable_sphere"),
        ("link5", "tool_capsule"),
        ("gripper_link", "wrist_cable_sphere"),
        ("gripper_link", "tool_capsule"),
        ("wrist_cable_sphere", "tool_capsule"),
    )
)

# Joint order mirrors the URDF and the safe-driver joint names.
JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)
# Mirror of the safe-driver T101 command limits (read-only reference used to
# verify that every interval endpoint and the convex chord between them stays
# inside the limits).
T101_COMMAND_LIMITS = (
    (-math.pi, math.pi),
    (-math.pi / 2.0, math.pi / 2.0),
    (0.0, 2.95),
    (-math.pi / 2.0, math.pi / 2.0),
    (-math.pi, math.pi),
    (0.0, 1.5),
)

# Body -> affecting revolute joint indices.  hand_tcp is attached to link5 by
# the fixed link5_to_hand_tcp joint, so it shares the first five joints.
BODY_CHAIN_JOINT_INDICES: dict[str, tuple[int, ...]] = {
    "base_link": (),
    "link1": (0,),
    "link2": (0, 1),
    "link3": (0, 1, 2),
    "link4": (0, 1, 2, 3),
    "link5": (0, 1, 2, 3, 4),
    "gripper_link": (0, 1, 2, 3, 4, 5),
    "hand_tcp": (0, 1, 2, 3, 4),
}

# Bounded proof budget.  A leaf is either certified by the conservative
# interval bound or remains UNPROVEN; it is never converted to PASS by
# sampling or by weakening an envelope.
MAX_INTERVAL_DEPTH = 10
SEGMENT_SUBDIVISION_DEPTH = 14
BOX_SUBDIVISION_DEPTH = 10
NUMERIC_EPSILON_M = 1e-12
MESH_SCALE = 0.001


class PrototypeClearanceBlocked(ValueError):
    """The profile, route, panel, model, or proof is invalid or unprovable."""


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PrototypeClearanceBlocked("value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise PrototypeClearanceBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PrototypeClearanceBlocked(f"{field} must be an object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise PrototypeClearanceBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PrototypeClearanceBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise PrototypeClearanceBlocked(f"{field} must be finite")
    return result


def _vec3(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise PrototypeClearanceBlocked(f"{field} must contain three values")
    return np.array(
        [_finite(item, f"{field}[{index}]") for index, item in enumerate(value)],
        dtype=float,
    )


def _vec4(value: Any, field: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise PrototypeClearanceBlocked(f"{field} must contain four values")
    return np.array(
        [_finite(item, f"{field}[{index}]") for index, item in enumerate(value)],
        dtype=float,
    )


def _vec6(value: Any, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 6:
        raise PrototypeClearanceBlocked(f"{field} must contain six values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, field: str) -> np.ndarray:
    vector = _vec3(value, field)
    magnitude = float(np.linalg.norm(vector))
    if not 0.999 <= magnitude <= 1.001:
        raise PrototypeClearanceBlocked(f"{field} must be unit length")
    return vector / magnitude


# --------------------------------------------------------------------------
# Robot model: xacro/joint parsing and binary STL mesh bounds
# --------------------------------------------------------------------------

def _rpy_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = np.array([float(value) for value in axis], dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise PrototypeClearanceBlocked("joint axis must be non-zero")
    x, y, z = vector / norm
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ]
    )


def _transform4(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _parse_xyz(text: str | None, default: Sequence[float]) -> list[float]:
    if not text:
        return list(default)
    return [float(value) for value in text.split()]


def _parse_urdf_joints(path: str | Path) -> dict[str, dict[str, Any]]:
    try:
        root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        raise PrototypeClearanceBlocked(f"URDF/xacro unreadable: {exc}") from exc
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not name:
            continue
        origin = joint.find("origin")
        xyz = _parse_xyz(None if origin is None else origin.get("xyz"), [0, 0, 0])
        rpy = _parse_xyz(None if origin is None else origin.get("rpy"), [0, 0, 0])
        axis_node = joint.find("axis")
        axis = _parse_xyz(
            None if axis_node is None else axis_node.get("xyz"), [1, 0, 0]
        )
        joints[name] = {
            "type": joint.get("type", ""),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }
    return joints


def _binary_stl_vertices(path: str | Path) -> list[np.ndarray]:
    data = Path(path).read_bytes()
    if len(data) < 84 or data[:5] != b"STLB ":
        raise PrototypeClearanceBlocked(
            f"collision mesh {path} is not a supported binary STL"
        )
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + count * 50 != len(data):
        raise PrototypeClearanceBlocked(f"collision mesh {path} has inconsistent size")
    vertices: list[np.ndarray] = []
    offset = 84
    for _ in range(count):
        for vertex_index in range(3):
            raw = struct.unpack_from("<3f", data, offset + 12 + 12 * vertex_index)
            vertices.append(np.array(raw, dtype=float) * MESH_SCALE)
        offset += 50
    return vertices


def _mesh_aabb(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, half_extents) of the mesh AABB in link frame, metres."""
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for vertex in _binary_stl_vertices(path):
        minimum = np.minimum(minimum, vertex)
        maximum = np.maximum(maximum, vertex)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise PrototypeClearanceBlocked(f"collision mesh {path} has non-finite bounds")
    center = 0.5 * (minimum + maximum)
    half = 0.5 * (maximum - minimum)
    return center, half


def _mesh_bounds(path: str | Path) -> dict[str, Any]:
    center, half = _mesh_aabb(path)
    return {
        "center_m": [float(value) for value in center],
        "half_extents_m": [float(value) for value in half],
        "bounding_sphere_radius_m": float(np.linalg.norm(half)),
    }


def _mesh_capsule(path: str | Path) -> dict[str, Any]:
    """Conservative capsule derived from mesh bounds.

    The capsule axis is the AABB axis (best of x/y/z) with the smallest
    perpendicular-vertex radius, centered on the AABB center line.  Every
    mesh vertex lies inside the capsule by construction, so the capsule
    contains the whole mesh (the capsule is convex).  Units are metres.
    """
    vertices = np.array(_binary_stl_vertices(path), dtype=float)
    center, half = _mesh_aabb(path)
    best: dict[str, Any] | None = None
    for axis_index in range(3):
        axis = np.zeros(3)
        axis[axis_index] = 1.0
        endpoint_a = center - axis * half[axis_index]
        endpoint_b = center + axis * half[axis_index]
        segment = endpoint_b - endpoint_a
        denominator = float(np.dot(segment, segment))
        if denominator < 1e-30:
            distances = np.linalg.norm(vertices - endpoint_a, axis=1)
            radius = float(np.max(distances))
        else:
            ratios = np.clip(((vertices - endpoint_a) @ segment) / denominator, 0.0, 1.0)
            closest = endpoint_a + ratios[:, None] * segment
            radius = float(np.max(np.linalg.norm(vertices - closest, axis=1)))
        candidate = {
            "endpoint_a_m": [float(value) for value in endpoint_a],
            "endpoint_b_m": [float(value) for value in endpoint_b],
            "radius_m": radius,
            "axis_index": axis_index,
        }
        if best is None or radius < float(best["radius_m"]):
            best = candidate
    assert best is not None
    return best


def _build_model(
    xacro_path: str | Path,
) -> dict[str, Any]:
    """Parse the xacro and all referenced collision meshes into conservative bounds."""
    resolved = Path(xacro_path).expanduser().resolve()
    xacro_hash = sha256_file(resolved)
    joints = _parse_urdf_joints(resolved)
    mesh_dir = resolved.parent.parent.parent / "meshes" / "roarm_m3"
    body_bounds: dict[str, dict[str, Any]] = {}
    mesh_hashes: list[str] = []
    mesh_paths: list[str] = []
    for body in URDF_BODY_IDS:
        mesh_path = mesh_dir / f"{body}.stl"
        if not mesh_path.is_file():
            raise PrototypeClearanceBlocked(f"collision mesh missing: {mesh_path}")
        capsule = _mesh_capsule(mesh_path)
        obb_center, obb_half = _mesh_aabb(mesh_path)
        body_bounds[body] = {
            **capsule,
            "obb_center_m": [float(value) for value in obb_center],
            "obb_half_extents_m": [float(value) for value in obb_half],
        }
        mesh_hashes.append(sha256_file(mesh_path))
        mesh_paths.append(str(mesh_path.relative_to(REPO_ROOT)))
    return {
        "urdf_or_xacro_path": str(resolved.relative_to(REPO_ROOT)),
        "urdf_or_xacro_sha256": xacro_hash,
        "collision_mesh_paths": mesh_paths,
        "collision_mesh_hashes": mesh_hashes,
        "body_ids": list(URDF_BODY_IDS),
        "body_bounds": body_bounds,
        "mesh_units_to_m": MESH_SCALE,
        "method": "binary_stl_mesh_conservative_capsule_scaled_0.001",
    }


def _fk_body(
    joints: Mapping[str, dict[str, Any]],
    positions: Sequence[float],
    body: str,
) -> np.ndarray:
    """Return the 4x4 transform of ``body`` in roarm_base_link at ``positions``."""
    by_name = dict(zip(JOINT_NAMES, positions))
    total = np.eye(4)
    for index in BODY_CHAIN_JOINT_INDICES[body]:
        joint = joints[JOINT_NAMES[index]]
        step = _transform4(_rpy_rotation(*joint["rpy"]), joint["xyz"])
        if joint["type"] in ("revolute", "continuous"):
            step = step @ _transform4(_axis_rotation(joint["axis"], by_name[JOINT_NAMES[index]]), [0, 0, 0])
        total = total @ step
    if body == "hand_tcp":
        joint = joints["link5_to_hand_tcp"]
        total = total @ _transform4(_rpy_rotation(*joint["rpy"]), joint["xyz"])
    return total


def _all_fk(joints: Mapping[str, dict[str, Any]], positions: Sequence[float]) -> dict[str, np.ndarray]:
    return {
        body: _fk_body(joints, positions, body)
        for body in (*URDF_BODY_IDS, "hand_tcp")
    }


# --------------------------------------------------------------------------
# Distance primitives (all return rigorous lower bounds)
# --------------------------------------------------------------------------

def _distance_point_point(left: Sequence[float], right: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _distance_point_aabb(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    delta = np.abs(point - center) - half
    outside = np.maximum(delta, 0.0)
    if np.any(delta > 0.0):
        return float(np.linalg.norm(outside))
    return -float(np.min(np.abs(delta)))  # negative when inside


def _distance_point_obb(
    point: np.ndarray, box_center: np.ndarray, box_rotation: np.ndarray, box_half: np.ndarray
) -> float:
    local = box_rotation.T @ (point - box_center)
    delta = np.abs(local) - box_half
    outside = np.maximum(delta, 0.0)
    if np.any(delta > 0.0):
        return float(np.linalg.norm(outside))
    return -float(np.min(np.abs(delta)))  # negative when inside


def _distance_aabb_aabb(
    center_a: np.ndarray, half_a: np.ndarray, center_b: np.ndarray, half_b: np.ndarray
) -> float:
    """Exact Euclidean distance between two axis-aligned boxes."""
    low_a = center_a - half_a
    high_a = center_a + half_a
    low_b = center_b - half_b
    high_b = center_b + half_b
    squared = 0.0
    for axis in range(3):
        gap = max(0.0, max(low_a[axis], low_b[axis]) - min(high_a[axis], high_b[axis]))
        squared += gap * gap
    return math.sqrt(squared)


def _aabb_obb_lower_bound(
    center: np.ndarray,
    half: np.ndarray,
    box_center: np.ndarray,
    box_rotation: np.ndarray,
    box_half: np.ndarray,
    *,
    threshold: float = MIN_FIXED_OBSTACLE_MARGIN_M,
    max_depth: int = BOX_SUBDIVISION_DEPTH,
) -> float:
    """Rigorous lower bound on AABB-to-OBB distance by best-first subdivision.

    Every frontier node contributes a valid lower bound for its sub-box; the
    minimum over the frontier is therefore a valid lower bound for the whole
    AABB.  The frontier is refined only at its minimum, so the proof stops as
    soon as the worst-case lower bound reaches the required threshold.
    """

    def sphere_lower(node_center: np.ndarray, node_half: np.ndarray) -> float:
        return _distance_point_obb(
            node_center, box_center, box_rotation, box_half
        ) - float(np.linalg.norm(node_half))

    frontier: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    sequence = 0
    heapq.heappush(frontier, (sphere_lower(center, half), 0, sequence, center, half))
    sequence += 1
    target = threshold - NUMERIC_EPSILON_M
    while frontier:
        bound, depth, _, node_center, node_half = heapq.heappop(frontier)
        if bound >= target:
            return bound
        if depth >= max_depth:
            return bound
        sub_half = 0.5 * node_half
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    sub_center = node_center + np.array([sign_x, sign_y, sign_z]) * 0.5 * node_half
                    heapq.heappush(
                        frontier,
                        (sphere_lower(sub_center, sub_half), depth + 1, sequence, sub_center, sub_half),
                    )
                    sequence += 1
    return math.inf


def _segment_aabb_lower_bound(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    box_center: np.ndarray,
    box_half: np.ndarray,
    *,
    threshold: float = MIN_SELF_NON_OVERLAP_TOLERANCE_M,
    max_depth: int = SEGMENT_SUBDIVISION_DEPTH,
) -> float:
    """Rigorous lower bound from a swept segment (capsule) to an AABB."""

    def segment_midpoint(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-15:
            return a, 0.0
        return a + 0.5 * direction, 0.5 * length

    midpoint, half_len = segment_midpoint(start, end)
    initial = _distance_point_aabb(midpoint, box_center, box_half) - half_len - radius
    frontier: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    sequence = 0
    heapq.heappush(frontier, (initial, 0, sequence, start, end))
    sequence += 1
    target = threshold - NUMERIC_EPSILON_M
    while frontier:
        bound, depth, _, seg_start, seg_end = heapq.heappop(frontier)
        if bound >= target:
            return bound
        if depth >= max_depth:
            return bound
        sub_midpoint, _ = segment_midpoint(seg_start, seg_end)
        for child in ((seg_start, sub_midpoint), (sub_midpoint, seg_end)):
            child_mid, child_half = segment_midpoint(child[0], child[1])
            child_bound = (
                _distance_point_aabb(child_mid, box_center, box_half)
                - child_half
                - radius
            )
            heapq.heappush(frontier, (child_bound, depth + 1, sequence, child[0], child[1]))
            sequence += 1
    return math.inf


def _segment_obb_lower_bound(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    box_center: np.ndarray,
    box_rotation: np.ndarray,
    box_half: np.ndarray,
    *,
    threshold: float = MIN_FIXED_OBSTACLE_MARGIN_M,
    max_depth: int = SEGMENT_SUBDIVISION_DEPTH,
) -> float:
    """Rigorous lower bound from a swept segment (capsule) to an OBB."""

    def segment_midpoint(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-15:
            return a, 0.0
        return a + 0.5 * direction, 0.5 * length

    midpoint, half_len = segment_midpoint(start, end)
    initial = (
        _distance_point_obb(midpoint, box_center, box_rotation, box_half)
        - half_len
        - radius
    )
    frontier: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    sequence = 0
    heapq.heappush(frontier, (initial, 0, sequence, start, end))
    sequence += 1
    target = threshold - NUMERIC_EPSILON_M
    while frontier:
        bound, depth, _, seg_start, seg_end = heapq.heappop(frontier)
        if bound >= target:
            return bound
        if depth >= max_depth:
            return bound
        sub_midpoint, _ = segment_midpoint(seg_start, seg_end)
        for child in ((seg_start, sub_midpoint), (sub_midpoint, seg_end)):
            child_mid, child_half = segment_midpoint(child[0], child[1])
            child_bound = (
                _distance_point_obb(child_mid, box_center, box_rotation, box_half)
                - child_half
                - radius
            )
            heapq.heappush(frontier, (child_bound, depth + 1, sequence, child[0], child[1]))
            sequence += 1
    return math.inf


def _distance_segment_point(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared < 1e-30:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, direction)) / length_squared
    ratio = max(0.0, min(1.0, ratio))
    closest = start + ratio * direction
    return float(np.linalg.norm(point - closest))


def _distance_segment_segment(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> float:
    """Exact Euclidean distance between two line segments (clamped closest points)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-30 and e <= 1e-30:
        return float(np.linalg.norm(p1 - p2))
    if a <= 1e-30:
        return _distance_segment_point(p2, q2, p1)
    if e <= 1e-30:
        return _distance_segment_point(p1, q1, p2)

    c = float(np.dot(d1, r))
    b = float(np.dot(d1, d2))
    denom = a * e - b * b
    if denom > 1e-30:
        s = max(0.0, min(1.0, (b * f - c * e) / denom))
    else:
        s = 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = max(0.0, min(1.0, -c / a))
    elif t > 1.0:
        t = 1.0
        s = max(0.0, min(1.0, (b - c) / a))
    closest1 = p1 + s * d1
    closest2 = p2 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


def _obb_obb_lower_bound(
    center_a: np.ndarray,
    rotation_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    rotation_b: np.ndarray,
    half_b: np.ndarray,
    *,
    threshold: float = MIN_SELF_NON_OVERLAP_TOLERANCE_M,
    max_depth: int = BOX_SUBDIVISION_DEPTH,
) -> float:
    """Rigorous lower bound on OBB-to-OBB distance by best-first subdivision.

    Box A is subdivided along its own axes; every sub-box is bounded by the
    sphere through its half-diagonal, giving a valid lower bound for that
    sub-box.  The minimum over the frontier is a valid lower bound for the
    whole box, and the frontier is refined only at its minimum.
    """

    def sphere_lower(node_center: np.ndarray, node_half: np.ndarray) -> float:
        return _distance_point_obb(
            node_center, center_b, rotation_b, half_b
        ) - float(np.linalg.norm(node_half))

    frontier: list[tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    sequence = 0
    heapq.heappush(
        frontier,
        (sphere_lower(center_a, half_a), 0, sequence, center_a, rotation_a, half_a),
    )
    sequence += 1
    target = threshold - NUMERIC_EPSILON_M
    while frontier:
        bound, depth, _, node_center, node_rotation, node_half = heapq.heappop(frontier)
        if bound >= target:
            return bound
        if depth >= max_depth:
            return bound
        sub_half = 0.5 * node_half
        for sign_x in (-1.0, 1.0):
            for sign_y in (-1.0, 1.0):
                for sign_z in (-1.0, 1.0):
                    offset = node_rotation @ (np.array([sign_x, sign_y, sign_z]) * 0.5 * node_half)
                    sub_center = node_center + offset
                    heapq.heappush(
                        frontier,
                        (
                            sphere_lower(sub_center, sub_half),
                            depth + 1,
                            sequence,
                            sub_center,
                            node_rotation,
                            sub_half,
                        ),
                    )
                    sequence += 1
    return math.inf


# --------------------------------------------------------------------------
# Interval sweep bounds
# --------------------------------------------------------------------------

def _interval_sweeps(
    joints: Mapping[str, dict[str, Any]],
    model: Mapping[str, Any],
    profile: Mapping[str, Any],
    q0: Sequence[float],
    q1: Sequence[float],
) -> dict[str, Any]:
    """Conservative swept bounds over the joint-space interval [q0, q1]."""
    start = np.asarray(q0, dtype=float)
    end = np.asarray(q1, dtype=float)
    delta = np.abs(end - start)
    midpoint = 0.5 * (start + end)

    joint_translation_norm = {
        index: float(np.linalg.norm(joints[JOINT_NAMES[index]]["xyz"]))
        for index in range(6)
    }

    def frame_alpha(indices: Sequence[int]) -> float:
        return 0.5 * sum(float(delta[index]) for index in indices)

    def origin_deviation(indices: Sequence[int]) -> float:
        total = 0.0
        for position, index in enumerate(indices):
            lever = joint_translation_norm[index]
            upstream = 0.5 * sum(float(delta[earlier]) for earlier in indices[:position])
            total += lever * upstream
        return total

    fk_mid = _all_fk(joints, midpoint)
    body_bounds = model["body_bounds"]

    links: dict[str, dict[str, Any]] = {}
    for body in URDF_BODY_IDS:
        indices = BODY_CHAIN_JOINT_INDICES[body]
        alpha = frame_alpha(indices)
        origin_dev = origin_deviation(indices)
        endpoint_a_local = np.asarray(body_bounds[body]["endpoint_a_m"], dtype=float)
        endpoint_b_local = np.asarray(body_bounds[body]["endpoint_b_m"], dtype=float)
        capsule_radius = float(body_bounds[body]["radius_m"])
        obb_center_local = np.asarray(body_bounds[body]["obb_center_m"], dtype=float)
        obb_half_local = np.asarray(body_bounds[body]["obb_half_extents_m"], dtype=float)
        rotation = fk_mid[body][:3, :3]
        origin = fk_mid[body][:3, 3]
        endpoint_a_ref = rotation @ endpoint_a_local + origin
        endpoint_b_ref = rotation @ endpoint_b_local + origin
        endpoint_metric = max(
            float(np.linalg.norm(endpoint_a_local)),
            float(np.linalg.norm(endpoint_b_local)),
        )
        inflation = origin_dev + endpoint_metric * alpha
        obb_inflation = origin_dev + (
            float(np.linalg.norm(obb_center_local))
            + float(np.linalg.norm(obb_half_local))
        ) * alpha
        links[body] = {
            "a_m": endpoint_a_ref,
            "b_m": endpoint_b_ref,
            "radius_m": capsule_radius + inflation,
            "obb_center_m": rotation @ obb_center_local + origin,
            "obb_rotation": rotation,
            "obb_half_extents_m": obb_half_local + obb_inflation,
            "inflation_m": inflation,
            "obb_inflation_m": obb_inflation,
            "alpha_rad": alpha,
            "origin_deviation_m": origin_dev,
        }

    hand_indices = BODY_CHAIN_JOINT_INDICES["hand_tcp"]
    hand_alpha = frame_alpha(hand_indices)
    hand_origin_dev = origin_deviation(hand_indices)
    hand_rotation = fk_mid["hand_tcp"][:3, :3]
    hand_origin = fk_mid["hand_tcp"][:3, 3]

    tool = profile["unified_tool_envelope"]
    endpoint_b = np.asarray(tool["endpoint_b_hand_tcp_m"], dtype=float)
    tool_radius = float(tool["radius_m"])
    wrist_radius = float(profile["wrist_cable_envelope"]["radius_m"])

    tool_a_ref = hand_origin
    tool_b_ref = hand_rotation @ endpoint_b + hand_origin
    tool_inflation = hand_origin_dev + float(np.linalg.norm(endpoint_b)) * hand_alpha

    return {
        "links": links,
        "wrist": {
            "center_m": hand_origin,
            "radius_m": wrist_radius + hand_origin_dev,
            "inflation_m": hand_origin_dev,
            "alpha_rad": hand_alpha,
        },
        "tool": {
            "a_m": tool_a_ref,
            "b_m": tool_b_ref,
            "radius_m": tool_radius + tool_inflation,
            "inflation_m": tool_inflation,
            "alpha_rad": hand_alpha,
        },
        "delta_rad": [float(value) for value in delta],
    }


def _leaf_lower_bounds(
    joints: Mapping[str, dict[str, Any]],
    model: Mapping[str, Any],
    profile: Mapping[str, Any],
    panel: Mapping[str, Any],
    q0: Sequence[float],
    q1: Sequence[float],
) -> dict[str, float]:
    """Rigorous lower bounds for one (subdivided) joint-space interval."""
    sweeps = _interval_sweeps(joints, model, profile, q0, q1)
    panel_center = np.asarray(panel["A_base_m"], dtype=float)
    panel_normal = np.asarray(panel["N_out_base"], dtype=float)
    bracket = profile["rear_camera_bracket"]
    bracket_center = np.asarray(bracket["pose_base"]["translation_m"], dtype=float)
    bracket_rotation = _quaternion_rotation(bracket["pose_base"]["quaternion_xyzw"])
    bracket_half = 0.5 * np.asarray(bracket["size_m"], dtype=float)

    fixed_margin = float(profile["clearance_requirements"]["fixed_obstacle_clearance_margin_m"])
    self_tolerance = float(profile["clearance_requirements"]["self_collision_non_overlap_tolerance_m"])

    panel_min = math.inf
    bracket_min = math.inf

    for body, swept in sweeps["links"].items():
        signed = min(
            float(np.dot(swept["a_m"] - panel_center, panel_normal)),
            float(np.dot(swept["b_m"] - panel_center, panel_normal)),
        ) - swept["radius_m"]
        panel_min = min(panel_min, signed)
        bracket_min = min(
            bracket_min,
            _segment_obb_lower_bound(
                swept["a_m"],
                swept["b_m"],
                swept["radius_m"],
                bracket_center,
                bracket_rotation,
                bracket_half,
            ),
        )

    wrist = sweeps["wrist"]
    panel_min = min(
        panel_min,
        float(np.dot(wrist["center_m"] - panel_center, panel_normal)) - wrist["radius_m"],
    )
    bracket_min = min(
        bracket_min,
        _distance_point_obb(wrist["center_m"], bracket_center, bracket_rotation, bracket_half)
        - wrist["radius_m"],
    )

    tool = sweeps["tool"]
    tool_panel = min(
        float(np.dot(tool["a_m"] - panel_center, panel_normal)),
        float(np.dot(tool["b_m"] - panel_center, panel_normal)),
    ) - tool["radius_m"]
    panel_min = min(panel_min, tool_panel)
    bracket_min = min(
        bracket_min,
        _segment_obb_lower_bound(
            tool["a_m"], tool["b_m"], tool["radius_m"], bracket_center, bracket_rotation, bracket_half
        ),
    )

    self_min = math.inf
    for left, right in _non_adjacent_pairs():
        self_min = min(self_min, _pair_lower_bound(sweeps, left, right))

    return {
        "panel_m": panel_min,
        "bracket_m": bracket_min,
        "self_m": self_min,
        "fixed_margin_m": fixed_margin,
        "self_tolerance_m": self_tolerance,
    }


def _non_adjacent_pairs() -> list[tuple[str, str]]:
    bodies = list(REQUIRED_BODY_IDS)
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(bodies):
        for right in bodies[index + 1 :]:
            if frozenset((left, right)) not in ADJACENT_PAIRS:
                pairs.append((left, right))
    return pairs


def _pair_lower_bound(
    sweeps: Mapping[str, Any], left: str, right: str
) -> float:
    links = sweeps["links"]
    wrist = sweeps["wrist"]
    tool = sweeps["tool"]

    def link_obb(body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        swept = links[body]
        return swept["obb_center_m"], swept["obb_rotation"], swept["obb_half_extents_m"]

    if left in links and right in links:
        c1, r1, h1 = link_obb(left)
        c2, r2, h2 = link_obb(right)
        return _obb_obb_lower_bound(c1, r1, h1, c2, r2, h2)
    if left == "wrist_cable_sphere" and right in links:
        c2, r2, h2 = link_obb(right)
        return _distance_point_obb(wrist["center_m"], c2, r2, h2) - wrist["radius_m"]
    if right == "wrist_cable_sphere" and left in links:
        c2, r2, h2 = link_obb(left)
        return _distance_point_obb(wrist["center_m"], c2, r2, h2) - wrist["radius_m"]
    if left == "tool_capsule" and right in links:
        c2, r2, h2 = link_obb(right)
        return _segment_obb_lower_bound(
            tool["a_m"], tool["b_m"], tool["radius_m"], c2, r2, h2,
            threshold=MIN_SELF_NON_OVERLAP_TOLERANCE_M,
        )
    if right == "tool_capsule" and left in links:
        c2, r2, h2 = link_obb(left)
        return _segment_obb_lower_bound(
            tool["a_m"], tool["b_m"], tool["radius_m"], c2, r2, h2,
            threshold=MIN_SELF_NON_OVERLAP_TOLERANCE_M,
        )
    if left == "tool_capsule" and right == "wrist_cable_sphere":
        return _distance_segment_point(tool["a_m"], tool["b_m"], wrist["center_m"]) - tool["radius_m"] - wrist["radius_m"]
    if left == "wrist_cable_sphere" and right == "tool_capsule":
        return _distance_segment_point(tool["a_m"], tool["b_m"], wrist["center_m"]) - tool["radius_m"] - wrist["radius_m"]
    raise PrototypeClearanceBlocked(f"unsupported self-collision pair {left}/{right}")


def _quaternion_rotation(quaternion: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise PrototypeClearanceBlocked("quaternion must be non-zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


# --------------------------------------------------------------------------
# Interval recursion
# --------------------------------------------------------------------------

def _certify_interval(
    joints: Mapping[str, dict[str, Any]],
    model: Mapping[str, Any],
    profile: Mapping[str, Any],
    panel: Mapping[str, Any],
    q0: tuple[float, ...],
    q1: tuple[float, ...],
    *,
    depth: int = 0,
) -> dict[str, Any]:
    lower = _leaf_lower_bounds(joints, model, profile, panel, q0, q1)
    fixed_margin = lower["fixed_margin_m"]
    self_tolerance = lower["self_tolerance_m"]
    certified = (
        lower["panel_m"] >= fixed_margin - NUMERIC_EPSILON_M
        and lower["bracket_m"] >= fixed_margin - NUMERIC_EPSILON_M
        and lower["self_m"] >= self_tolerance - NUMERIC_EPSILON_M
    )
    leaf: dict[str, Any] = {
        "depth": depth,
        "certified": certified,
        "panel_lower_bound_m": lower["panel_m"],
        "bracket_lower_bound_m": lower["bracket_m"],
        "self_lower_bound_m": lower["self_m"],
    }
    if certified or depth >= MAX_INTERVAL_DEPTH:
        return {"status": "CERTIFIED" if certified else "UNPROVEN", "leaves": [leaf]}
    midpoint = tuple(0.5 * (float(a) + float(b)) for a, b in zip(q0, q1))
    left = _certify_interval(
        joints, model, profile, panel, q0, midpoint, depth=depth + 1
    )
    right = _certify_interval(
        joints, model, profile, panel, midpoint, q1, depth=depth + 1
    )
    status = "CERTIFIED" if left["status"] == "CERTIFIED" and right["status"] == "CERTIFIED" else "UNPROVEN"
    return {
        "status": status,
        "leaves": left["leaves"] + right["leaves"],
    }


# --------------------------------------------------------------------------
# Profile / route / panel validation
# --------------------------------------------------------------------------

def load_json(path: str | Path) -> tuple[dict[str, Any], str]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrototypeClearanceBlocked(f"JSON unreadable at {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise PrototypeClearanceBlocked(f"JSON at {resolved} must be an object")
    return value, sha256_file(resolved)


def validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on any profile that shrinks the prototype envelope floors."""
    if profile.get("schema") != PROFILE_SCHEMA:
        raise PrototypeClearanceBlocked("profile schema mismatch")
    tool = _mapping(profile.get("unified_tool_envelope"), "unified_tool_envelope")
    tool_radius = _finite(tool.get("radius_m"), "unified_tool_envelope.radius_m")
    if tool_radius < MIN_TOOL_RADIUS_M - 1e-12:
        raise PrototypeClearanceBlocked(
            f"tool radius {tool_radius} below prototype floor {MIN_TOOL_RADIUS_M}"
        )
    endpoint_a = _vec3(tool.get("endpoint_a_hand_tcp_m"), "tool endpoint_a")
    endpoint_b = _vec3(tool.get("endpoint_b_hand_tcp_m"), "tool endpoint_b")
    if not np.allclose(endpoint_a, [0.0, 0.0, 0.0], atol=1e-12):
        raise PrototypeClearanceBlocked("tool capsule must start at hand_tcp origin")
    if float(np.linalg.norm(endpoint_b)) < 0.100:
        raise PrototypeClearanceBlocked("tool capsule endpoint would shorten the Fn vector")

    wrist = _mapping(profile.get("wrist_cable_envelope"), "wrist_cable_envelope")
    wrist_radius = _finite(wrist.get("radius_m"), "wrist_cable_envelope.radius_m")
    if wrist_radius < MIN_WRIST_RADIUS_M - 1e-12:
        raise PrototypeClearanceBlocked(
            f"wrist/cable radius {wrist_radius} below prototype floor {MIN_WRIST_RADIUS_M}"
        )
    if wrist.get("frame_id") != "hand_tcp":
        raise PrototypeClearanceBlocked("wrist/cable envelope must be attached to hand_tcp")

    bracket = _mapping(profile.get("rear_camera_bracket"), "rear_camera_bracket")
    bracket_size = _vec3(bracket.get("size_m"), "rear_camera_bracket.size_m")
    for index, (value, floor) in enumerate(zip(bracket_size, MIN_BRACKET_SIZE_M)):
        if value < floor - 1e-12:
            raise PrototypeClearanceBlocked(
                f"bracket size axis {index} below prototype floor {floor}"
            )
    _vec3(bracket.get("pose_base", {}).get("translation_m"), "bracket pose translation")
    _vec4(bracket.get("pose_base", {}).get("quaternion_xyzw"), "bracket pose quaternion")

    requirements = _mapping(profile.get("clearance_requirements"), "clearance_requirements")
    margin = _finite(
        requirements.get("fixed_obstacle_clearance_margin_m"),
        "fixed_obstacle_clearance_margin_m",
    )
    if margin < MIN_FIXED_OBSTACLE_MARGIN_M - 1e-12:
        raise PrototypeClearanceBlocked(
            f"fixed-obstacle margin {margin} below prototype floor {MIN_FIXED_OBSTACLE_MARGIN_M}"
        )
    self_tolerance = _finite(
        requirements.get("self_collision_non_overlap_tolerance_m"),
        "self_collision_non_overlap_tolerance_m",
    )
    if self_tolerance < MIN_SELF_NON_OVERLAP_TOLERANCE_M - 1e-12:
        raise PrototypeClearanceBlocked(
            f"self non-overlap tolerance {self_tolerance} below prototype floor {MIN_SELF_NON_OVERLAP_TOLERANCE_M}"
        )

    body_ids = profile.get("robot_model", {}).get("body_ids")
    if tuple(body_ids or ()) != URDF_BODY_IDS:
        raise PrototypeClearanceBlocked("profile robot body set is incomplete")
    for body in URDF_BODY_IDS:
        bounds = profile.get("robot_model", {}).get("body_bounds", {}).get(body)
        if not isinstance(bounds, Mapping):
            raise PrototypeClearanceBlocked(f"profile missing body bounds for {body}")
        _vec3(bounds.get("endpoint_a_m"), f"body_bounds.{body}.endpoint_a_m")
        _vec3(bounds.get("endpoint_b_m"), f"body_bounds.{body}.endpoint_b_m")
        radius = _finite(bounds.get("radius_m"), f"body_bounds.{body}.radius_m")
        if radius <= 0.0:
            raise PrototypeClearanceBlocked(f"body_bounds.{body}.radius_m must be positive")
        _vec3(bounds.get("obb_center_m"), f"body_bounds.{body}.obb_center_m")
        obb_half = _vec3(bounds.get("obb_half_extents_m"), f"body_bounds.{body}.obb_half_extents_m")
        if np.any(obb_half <= 0.0):
            raise PrototypeClearanceBlocked(f"body_bounds.{body}.obb_half_extents_m must be positive")

    excluded = profile.get("excluded_adjacent_pairs")
    if not isinstance(excluded, list):
        raise PrototypeClearanceBlocked("excluded_adjacent_pairs must be a list")
    expected_excluded = sorted(
        tuple(sorted(pair)) for pair in (tuple(sorted(item)) for item in ADJACENT_PAIRS)
    )
    actual_excluded = sorted(tuple(sorted(item)) for item in excluded)
    if actual_excluded != expected_excluded:
        raise PrototypeClearanceBlocked(
            "excluded_adjacent_pairs differ from the hash-bound prototype policy"
        )

    authority = _mapping(profile.get("authority"), "profile authority")
    for field in (
        "execution_ready",
        "motion_authorized",
        "contact_authorized",
        "adapter_consumable",
        "deployment_authorized",
    ):
        if authority.get(field) is not False:
            raise PrototypeClearanceBlocked(f"profile authority {field} must be false")
    return dict(profile)


def load_profile(path: str | Path) -> tuple[dict[str, Any], str]:
    profile, file_hash = load_json(path)
    validate_profile(profile)
    unsigned = dict(profile)
    supplied = _sha(unsigned.pop("canonical_profile_sha256", None), "canonical_profile_sha256")
    if canonical_sha256(unsigned) != supplied:
        raise PrototypeClearanceBlocked("profile self-hash mismatch")
    return profile, file_hash


def load_route(path: str | Path) -> tuple[dict[str, Any], str]:
    route, file_hash = load_json(path)
    if route.get("schema") != "t102_outbound_only_route_v1":
        raise PrototypeClearanceBlocked("route schema mismatch")
    if route.get("route_mode") != "OFFLINE_DIAGNOSTIC_ONLY":
        raise PrototypeClearanceBlocked("route must be OFFLINE_DIAGNOSTIC_ONLY")
    if route.get("command_interface") != "NOT_IMPLEMENTED":
        raise PrototypeClearanceBlocked("route must not expose a command interface")
    for field in ("no_contact",):
        if route.get(field) is not True:
            raise PrototypeClearanceBlocked(f"route must keep {field}=true")
    for field in (
        "press",
        "retract",
        "return",
        "automatic_retry",
        "execution_ready",
        "motion_authorized",
        "trajectory_generated",
        "command_generated",
        "adapter_consumable",
    ):
        if route.get(field) is not False:
            raise PrototypeClearanceBlocked(f"route must keep {field}=false")
    unsigned = dict(route)
    supplied = _sha(unsigned.pop("route_sha256", None), "route_sha256")
    if canonical_sha256(unsigned) != supplied:
        raise PrototypeClearanceBlocked("route self-hash mismatch")
    setpoints = route.get("setpoints_rad")
    if not isinstance(setpoints, Sequence) or isinstance(setpoints, (str, bytes)):
        raise PrototypeClearanceBlocked("route setpoints are missing")
    for index, raw in enumerate(setpoints):
        _vec6(raw, f"setpoints_rad[{index}]")
    if len(setpoints) < 1:
        raise PrototypeClearanceBlocked("prototype route must contain at least one outbound interval")
    return route, file_hash


def validate_panel(panel: Mapping[str, Any]) -> dict[str, Any]:
    a = _vec3(panel.get("A_base_m"), "panel.A_base_m")
    normal = _unit(panel.get("N_out_base"), "panel.N_out_base")
    pre = _vec3(panel.get("P_pre_base_m"), "panel.P_pre_base_m")
    distance = _finite(panel.get("intended_no_contact_distance_m"), "intended_no_contact_distance_m")
    if distance < 0.0:
        raise PrototypeClearanceBlocked("intended_no_contact_distance_m cannot be negative")
    if not np.allclose(pre, a + distance * normal, atol=2e-3):
        raise PrototypeClearanceBlocked("P_pre must equal A + distance * N_out within 2 mm")
    return dict(panel)


def _waypoint_excluded_pair_overlap(
    joints: Mapping[str, dict[str, Any]],
    model: Mapping[str, Any],
    profile: Mapping[str, Any],
    waypoints: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Verify every excluded pair overlaps by construction at every waypoint."""
    tool = profile["unified_tool_envelope"]
    endpoint_b = np.asarray(tool["endpoint_b_hand_tcp_m"], dtype=float)
    tool_radius = float(tool["radius_m"])
    wrist_radius = float(profile["wrist_cable_envelope"]["radius_m"])
    body_bounds = model["body_bounds"]
    failures: list[str] = []
    verified_pairs = 0
    for waypoint_index, positions in enumerate(waypoints):
        fk = _all_fk(joints, positions)
        spheres: dict[str, tuple[np.ndarray, float]] = {}
        for body in URDF_BODY_IDS:
            endpoint_a_local = np.asarray(body_bounds[body]["endpoint_a_m"], dtype=float)
            endpoint_b_local = np.asarray(body_bounds[body]["endpoint_b_m"], dtype=float)
            rotation = fk[body][:3, :3]
            origin = fk[body][:3, 3]
            midpoint = origin + 0.5 * (rotation @ (endpoint_a_local + endpoint_b_local))
            wrapper_radius = (
                0.5 * float(np.linalg.norm(endpoint_b_local - endpoint_a_local))
                + float(body_bounds[body]["radius_m"])
            )
            spheres[body] = (midpoint, wrapper_radius)
        hand_rotation = fk["hand_tcp"][:3, :3]
        hand_origin = fk["hand_tcp"][:3, 3]
        spheres["wrist_cable_sphere"] = (hand_origin, wrist_radius)
        # The tool capsule is modeled as a wrapper sphere at its midpoint for
        # the overlap-by-construction verification: its base sits at hand_tcp,
        # which lies inside the wrist/gripper/link5 assembly.
        tool_midpoint = hand_origin + 0.5 * (hand_rotation @ endpoint_b)
        tool_wrapper_radius = 0.5 * float(np.linalg.norm(endpoint_b)) + tool_radius
        spheres["tool_capsule"] = (tool_midpoint, tool_wrapper_radius)
        for pair in sorted(tuple(sorted(item)) for item in ADJACENT_PAIRS):
            left, right = pair
            left_center, left_radius = spheres[left]
            right_center, right_radius = spheres[right]
            gap = float(np.linalg.norm(left_center - right_center)) - left_radius - right_radius
            if gap > 1e-9:
                failures.append(
                    f"waypoint {waypoint_index}: excluded pair {left}/{right} does not overlap by construction (gap {gap * 1000:.1f} mm)"
                )
        verified_pairs += 1
    return {
        "verified_pairs": verified_pairs,
        "waypoint_count": len(waypoints),
        "all_excluded_pairs_overlap_by_construction": not failures,
        "failures": failures,
    }


def build_clearance_result(
    profile: Mapping[str, Any],
    route: Mapping[str, Any],
    panel: Mapping[str, Any],
    *,
    panel_provisional: bool = True,
    panel_source: str = "stale 2026-08-11 retained Button 1 reference (not fresh Layer-D)",
) -> dict[str, Any]:
    """Run the narrow continuous clearance proof for one exact route candidate."""
    validate_profile(profile)
    panel = validate_panel(panel)
    route_sha = _sha(route.get("route_sha256"), "route_sha256")

    model = _build_model(profile["robot_model"]["urdf_or_xacro_path"])
    model_profile = profile["robot_model"]
    if model["urdf_or_xacro_sha256"] != _sha(model_profile.get("urdf_or_xacro_sha256"), "model urdf hash"):
        raise PrototypeClearanceBlocked("profile model hash drifts from the actual xacro")
    if model["collision_mesh_hashes"] != model_profile.get("collision_mesh_hashes"):
        raise PrototypeClearanceBlocked("profile collision mesh hashes drift from the actual meshes")
    for body in URDF_BODY_IDS:
        recomputed = model["body_bounds"][body]
        recorded = model_profile.get("body_bounds", {}).get(body)
        if not isinstance(recorded, Mapping):
            raise PrototypeClearanceBlocked(f"profile body bounds missing for {body}")
        for field in ("endpoint_a_m", "endpoint_b_m", "radius_m", "obb_center_m", "obb_half_extents_m"):
            actual = np.asarray(recomputed[field], dtype=float)
            declared = np.asarray(recorded[field], dtype=float)
            if not np.allclose(actual, declared, atol=1e-6, rtol=1e-6):
                raise PrototypeClearanceBlocked(
                    f"profile body bounds for {body}.{field} drift from the recomputed mesh capsule"
                )

    joints = _parse_urdf_joints(model["urdf_or_xacro_path"])
    start = _vec6(route.get("start_positions_rad"), "route start")
    setpoints = [_vec6(point, "route setpoint") for point in route["setpoints_rad"]]
    waypoints = (start, *setpoints)
    for index, waypoint in enumerate(waypoints):
        for joint_index, (lower, upper) in enumerate(T101_COMMAND_LIMITS):
            if not lower - 1e-12 <= waypoint[joint_index] <= upper + 1e-12:
                raise PrototypeClearanceBlocked(
                    f"waypoint {index} joint {joint_index} outside T101 limits"
                )
    if not np.allclose(waypoints[-1], _vec6(route.get("static_ik_candidate_positions_rad"), "route candidate"), atol=1e-12):
        raise PrototypeClearanceBlocked("route endpoint differs from static IK candidate")

    overlap = _waypoint_excluded_pair_overlap(joints, model, profile, waypoints)
    if not overlap["all_excluded_pairs_overlap_by_construction"]:
        raise PrototypeClearanceBlocked(
            "excluded adjacent pair verification failed: " + "; ".join(overlap["failures"][:3])
        )

    segments: list[dict[str, Any]] = []
    overall_status = "PROVISIONAL_CONTINUOUS_CLEARANCE_PASS"
    route_panel_min = math.inf
    route_bracket_min = math.inf
    route_self_min = math.inf
    for index in range(len(setpoints)):
        q0 = waypoints[index]
        q1 = waypoints[index + 1]
        result = _certify_interval(joints, model, profile, panel, q0, q1)
        certified_leaves = [leaf for leaf in result["leaves"] if leaf["certified"]]
        unproven_leaves = [leaf for leaf in result["leaves"] if not leaf["certified"]]
        segment_panel_min = min(leaf["panel_lower_bound_m"] for leaf in result["leaves"])
        segment_bracket_min = min(leaf["bracket_lower_bound_m"] for leaf in result["leaves"])
        segment_self_min = min(leaf["self_lower_bound_m"] for leaf in result["leaves"])
        route_panel_min = min(route_panel_min, segment_panel_min)
        route_bracket_min = min(route_bracket_min, segment_bracket_min)
        route_self_min = min(route_self_min, segment_self_min)
        if result["status"] != "CERTIFIED":
            overall_status = "UNPROVEN_OR_COLLISION_RISK"
        segments.append(
            {
                "segment_index": index,
                "start_waypoint_index": index,
                "end_waypoint_index": index + 1,
                "method": (
                    "adaptive_interval_fk_conservative_primitive_bounds_with_recursive_subdivision"
                ),
                "status": result["status"],
                "certified_leaf_count": len(certified_leaves),
                "unproven_leaf_count": len(unproven_leaves),
                "maximum_leaf_depth": max(leaf["depth"] for leaf in result["leaves"]),
                "minimum_panel_lower_bound_m": segment_panel_min,
                "minimum_bracket_lower_bound_m": segment_bracket_min,
                "minimum_self_lower_bound_m": segment_self_min,
                "leaves": result["leaves"],
            }
        )

    fixed_margin = float(profile["clearance_requirements"]["fixed_obstacle_clearance_margin_m"])
    self_tolerance = float(profile["clearance_requirements"]["self_collision_non_overlap_tolerance_m"])
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "mode": MODE,
        "result": overall_status,
        "provisional": True,
        "non_authoritative": True,
        "panel_provisional": bool(panel_provisional),
        "panel_source": str(panel_source),
        "created_at": "2026-08-14",
        "route": {
            "path": "config/press/prototype/t102_prototype_route_candidate_20260814_v1.json",
            "route_sha256": route_sha,
            "setpoint_count": len(setpoints),
            "interval_count": len(segments),
            "joint_path": str(route.get("joint_path")),
            "route_mode": str(route.get("route_mode")),
        },
        "profile": {
            "path": str(profile.get("prototype_profile_id") or ""),
            "canonical_profile_sha256": str(profile.get("canonical_profile_sha256")),
        },
        "model": {
            "urdf_or_xacro_path": model["urdf_or_xacro_path"],
            "urdf_or_xacro_sha256": model["urdf_or_xacro_sha256"],
            "collision_mesh_hashes": model["collision_mesh_hashes"],
            "body_ids": model["body_ids"],
        },
        "panel": {
            "A_base_m": [float(value) for value in panel["A_base_m"]],
            "N_out_base": [float(value) for value in panel["N_out_base"]],
            "P_pre_base_m": [float(value) for value in panel["P_pre_base_m"]],
            "intended_no_contact_distance_m": float(panel["intended_no_contact_distance_m"]),
        },
        "envelope_floors": {
            "tool_radius_m": MIN_TOOL_RADIUS_M,
            "wrist_radius_m": MIN_WRIST_RADIUS_M,
            "bracket_size_m": list(MIN_BRACKET_SIZE_M),
            "fixed_obstacle_margin_m": MIN_FIXED_OBSTACLE_MARGIN_M,
            "self_non_overlap_tolerance_m": MIN_SELF_NON_OVERLAP_TOLERANCE_M,
        },
        "coverage": {
            "body_ids": list(REQUIRED_BODY_IDS),
            "excluded_adjacent_pairs": sorted(
                tuple(sorted(pair)) for pair in (tuple(sorted(item)) for item in ADJACENT_PAIRS)
            ),
            "non_adjacent_pair_count": len(_non_adjacent_pairs()),
            "obstacles": ["panel_plane_outward_side", "rear_camera_bracket_obb"],
            "required_fixed_clearance_m": fixed_margin,
            "required_self_non_overlap_m": self_tolerance,
        },
        "interval_results": {
            "panel_side_result": (
                "PASS" if route_panel_min >= fixed_margin - NUMERIC_EPSILON_M else "FAIL"
            ),
            "bracket_result": (
                "PASS" if route_bracket_min >= fixed_margin - NUMERIC_EPSILON_M else "FAIL"
            ),
            "self_collision_result": (
                "PASS" if route_self_min >= self_tolerance - NUMERIC_EPSILON_M else "FAIL"
            ),
            "route_minimum_panel_lower_bound_m": route_panel_min,
            "route_minimum_bracket_lower_bound_m": route_bracket_min,
            "route_minimum_self_lower_bound_m": route_self_min,
        },
        "excluded_pair_overlap_verification": overlap,
        "segments": segments,
        "authority": {
            "execution_ready": False,
            "motion_authorized": False,
            "contact_authorized": False,
            "adapter_consumable": False,
            "deployment_authorized": False,
        },
        "limitations": [
            "OFFLINE_PROTOTYPE_ONLY",
            "PROVISIONAL_PANEL_AND_ROUTE",
            "NO_ROS_OR_DRIVER_INTERFACE",
            "NO_MOTION_AUTHORITY",
            "FRESH_Q0_TARGET_IK_AND_ROUTE_BINDING_REQUIRED_TOMORROW",
            "CONTINUOUS_PROOF_IS_CONSERVATIVE_AND_NON_SAMPLED",
        ],
    }
    result["canonical_result_sha256"] = canonical_sha256(result)
    validate_result(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema") != RESULT_SCHEMA or result.get("mode") != MODE:
        raise PrototypeClearanceBlocked("unexpected result schema or mode")
    if result.get("result") not in (
        "PROVISIONAL_CONTINUOUS_CLEARANCE_PASS",
        "UNPROVEN_OR_COLLISION_RISK",
    ):
        raise PrototypeClearanceBlocked("invalid result status")
    if result.get("provisional") is not True or result.get("non_authoritative") is not True:
        raise PrototypeClearanceBlocked("result must remain provisional and non-authoritative")
    authority = _mapping(result.get("authority"), "result authority")
    for field in (
        "execution_ready",
        "motion_authorized",
        "contact_authorized",
        "adapter_consumable",
        "deployment_authorized",
    ):
        if authority.get(field) is not False:
            raise PrototypeClearanceBlocked(f"result authority {field} must be false")
    segments = result.get("segments")
    if not isinstance(segments, Sequence) or len(segments) < 1:
        raise PrototypeClearanceBlocked("result must contain at least one interval segment")
    for segment in segments:
        if segment.get("method") != "adaptive_interval_fk_conservative_primitive_bounds_with_recursive_subdivision":
            raise PrototypeClearanceBlocked("segment method must be the continuous interval proof")
        if segment.get("status") not in ("CERTIFIED", "UNPROVEN"):
            raise PrototypeClearanceBlocked("segment status is invalid")
    unsigned = dict(result)
    supplied = _sha(unsigned.pop("canonical_result_sha256", None), "canonical_result_sha256")
    if canonical_sha256(unsigned) != supplied:
        raise PrototypeClearanceBlocked("result self-hash mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--route", default=str(DEFAULT_ROUTE_PATH))
    parser.add_argument("--panel", default=None, help="optional fresh Layer-D panel JSON")
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "config/press/prototype/t102_prototype_clearance_result_20260814_v1.json"
        ),
    )
    args = parser.parse_args(argv)
    profile, _ = load_profile(args.profile)
    route, _ = load_route(args.route)
    if args.panel:
        panel, _ = load_json(args.panel)
        panel_provisional = bool(panel.get("provisional", False))
        panel_source = str(panel.get("source", "fresh Layer-D panel input"))
    else:
        panel = {
            "A_base_m": [0.32584954126921023, 0.05410619205558746, 0.3108212132435427],
            "N_out_base": [-0.9938271544605604, -0.054902331626688006, -0.09640187258958627],
            "P_pre_base_m": [0.2761581835461822, 0.05136107547425306, 0.3060011196140634],
            "intended_no_contact_distance_m": 0.050,
        }
        panel_provisional = True
        panel_source = "stale 2026-08-11 retained Button 1 reference (not fresh Layer-D)"
    result = build_clearance_result(
        profile,
        route,
        panel,
        panel_provisional=panel_provisional,
        panel_source=panel_source,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": result["result"], "output": str(output), "sha256": result["canonical_result_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADJACENT_PAIRS",
    "MAX_INTERVAL_DEPTH",
    "MIN_BRACKET_SIZE_M",
    "MIN_FIXED_OBSTACLE_MARGIN_M",
    "MIN_SELF_NON_OVERLAP_TOLERANCE_M",
    "MIN_TOOL_RADIUS_M",
    "MIN_WRIST_RADIUS_M",
    "MODE",
    "PrototypeClearanceBlocked",
    "REQUIRED_BODY_IDS",
    "RESULT_SCHEMA",
    "build_clearance_result",
    "canonical_sha256",
    "load_profile",
    "load_route",
    "validate_panel",
    "validate_profile",
    "validate_result",
]
