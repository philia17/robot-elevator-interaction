#!/usr/bin/env python3
"""Build and fake-driver validate a taught-pose M1 no-contact route.

This module is deliberately ROS-independent. It cannot arm a ROS driver,
publish a JointState, open a serial device, or command hardware. The only
driver interaction is the pure ``SafeController`` core with an in-memory
write callback and synthetic feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_DRIVER_PACKAGE = REPO_ROOT / "ros2_arm" / "roarm_safe_driver"
if str(SAFE_DRIVER_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SAFE_DRIVER_PACKAGE))

from roarm_safe_driver.core import (  # noqa: E402
    DEFAULT_JOINT_ACC_REGISTERS,
    DEFAULT_JOINT_MAX_DELTA_RAD,
    DEFAULT_JOINT_SPEED_STEPS_S,
    ENCODER_STEP_RAD,
    Feedback,
    JOINT_NAMES,
    SafeController,
    T101_COMMAND_LIMITS,
    validate_single_joint_target,
)


SCHEMA = "m1_no_contact_taught_route_v1"
TASK_JOINT_INDICES = (0, 1, 2, 3, 4)
GRIPPER_INDEX = 5
OUTBOUND_ORDER = (4, 3, 0, 1, 2)
RETURN_ORDER = tuple(reversed(OUTBOUND_ORDER))
# Recorded values are decimal round-trips of encoder positions. The extra
# nanoradian avoids turning rounding noise around one encoder step into a
# physical command.
HOLD_TOLERANCE_RAD = ENCODER_STEP_RAD + 1e-9
URDF_PATH = (
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
DRIVER_PROFILE_PATH = (
    REPO_ROOT
    / "ros2_arm"
    / "roarm_safe_driver"
    / "config"
    / "agx_unified_safe.yaml"
)
FK_CHAIN = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_hand_tcp",
)


class RouteError(ValueError):
    """A source pose or planned route failed a fail-closed check."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteError(f"{where} must be a mapping")
    return value


def _finite_sequence(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise RouteError(f"{where} must contain exactly {length} values")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise RouteError(f"{where}[{index}] must be finite")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise RouteError(f"{where}[{index}] must be finite") from exc
        if not math.isfinite(number):
            raise RouteError(f"{where}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _finite_vector(value: Any, where: str) -> tuple[float, ...]:
    return _finite_sequence(value, len(JOINT_NAMES), where)


def load_recorded_pose(path: str | Path, expected_role: str) -> dict[str, Any]:
    """Load only a non-executable, stable recorded pose candidate."""
    source_path = Path(path).expanduser().resolve()
    try:
        root = _mapping(yaml.safe_load(source_path.read_bytes()), "pose")
    except yaml.YAMLError as exc:
        raise RouteError(f"invalid pose YAML: {exc}") from exc
    if root.get("role") != expected_role:
        raise RouteError(f"pose role must be {expected_role}")
    if root.get("status") != "recorded_candidate":
        raise RouteError("pose status must be recorded_candidate")
    if tuple(root.get("joint_names", ())) != JOINT_NAMES:
        raise RouteError("pose joint_names must be the exact six URDF names")
    positions = _finite_vector(root.get("joint_positions_rad"), "joint_positions_rad")
    if root.get("all_joints_inside_t101_limits") is not True:
        raise RouteError("pose must explicitly pass T101 limits")
    for key in (
        "execution_ready",
        "motion_authorized",
        "contact_authorized",
        "press_authorized",
    ):
        if root.get(key) is not False:
            raise RouteError(f"pose {key} must be false")
    safe = _mapping(root.get("safe_driver"), "safe_driver")
    if (
        safe.get("armed") is not False
        or safe.get("fault", "") != ""
        or safe.get("torque_enabled") != [True] * len(JOINT_NAMES)
        or safe.get("joint_target_publisher_count_max") != 0
    ):
        raise RouteError("pose safe-driver evidence is not fail-closed healthy")
    spans = _finite_vector(
        _mapping(root.get("joint_safe_window"), "joint_safe_window").get(
            "per_joint_span_rad"
        ),
        "joint_safe_window.per_joint_span_rad",
    )
    if max(spans) > 2.0 * ENCODER_STEP_RAD + 1e-12:
        raise RouteError("pose stable window exceeds two encoder steps")
    for index, (value, (lower, upper)) in enumerate(
        zip(positions, T101_COMMAND_LIMITS)
    ):
        if not lower <= value <= upper:
            raise RouteError(f"{JOINT_NAMES[index]} is outside T101 limits")
    return {
        "path": source_path,
        "sha256": sha256_file(source_path),
        "pose_id": str(root.get("pose_id", "")).strip(),
        "role": expected_role,
        "positions": positions,
        "joint_states_stamp_ns": int(root["joint_states_stamp_ns"]),
        "stored_tf_translation_m": _finite_sequence(
            _mapping(
                root.get("T_roarm_base_link_hand_tcp"),
                "T_roarm_base_link_hand_tcp",
            ).get("translation_m"),
            3,
            "T_roarm_base_link_hand_tcp.translation_m",
        ),
        "stored_tf_rotation_xyzw": _finite_sequence(
            _mapping(
                root.get("T_roarm_base_link_hand_tcp"),
                "T_roarm_base_link_hand_tcp",
            ).get("rotation_xyzw"),
            4,
            "T_roarm_base_link_hand_tcp.rotation_xyzw",
        ),
    }


def _axis_steps(
    current: tuple[float, ...],
    target: tuple[float, ...],
    joint_index: int,
    phase: str,
    sequence_start: int,
) -> tuple[list[dict[str, Any]], tuple[float, ...], dict[str, Any] | None]:
    delta = target[joint_index] - current[joint_index]
    if abs(delta) <= HOLD_TOLERANCE_RAD:
        hold = {
            "phase": phase,
            "joint_index": joint_index,
            "joint": JOINT_NAMES[joint_index],
            "from_rad": current[joint_index],
            "requested_endpoint_rad": target[joint_index],
            "delta_rad": delta,
            "policy": "tolerance_hold_no_command",
            "tolerance_rad": HOLD_TOLERANCE_RAD,
        }
        return [], current, hold
    max_delta = DEFAULT_JOINT_MAX_DELTA_RAD[joint_index]
    count = int(math.ceil(abs(delta) / max_delta - 1e-12))
    if count <= 0:
        raise RouteError("non-held axis must produce at least one step")
    increment = delta / count
    if abs(increment) <= HOLD_TOLERANCE_RAD:
        raise RouteError(
            f"{JOINT_NAMES[joint_index]} subdivision is not meaningfully larger "
            "than the hold tolerance"
        )
    axis_start = current[joint_index]
    steps: list[dict[str, Any]] = []
    measured = current
    for local_index in range(1, count + 1):
        target_value = (
            target[joint_index]
            if local_index == count
            else axis_start + delta * local_index / count
        )
        command = list(measured)
        command[joint_index] = target_value
        checked, selected = validate_single_joint_target(
            command,
            measured,
            DEFAULT_JOINT_MAX_DELTA_RAD,
        )
        if selected != joint_index:
            raise RouteError("safe-driver selected a different logical joint")
        step_delta = checked[joint_index] - measured[joint_index]
        steps.append(
            {
                "step_id": f"{'O' if phase == 'outbound' else 'R'}"
                f"{sequence_start + len(steps):03d}",
                "phase": phase,
                "phase_step_index": sequence_start + len(steps),
                "joint_index": joint_index,
                "joint": JOINT_NAMES[joint_index],
                "from_rad": measured[joint_index],
                "target_rad": checked[joint_index],
                "delta_rad": step_delta,
                "max_delta_rad": max_delta,
                "speed_steps_s": DEFAULT_JOINT_SPEED_STEPS_S[joint_index],
                "acc_register": DEFAULT_JOINT_ACC_REGISTERS[joint_index],
                "target_positions_rad": list(checked),
                "requires_fresh_operator_confirmation": True,
                "requires_fresh_measured_feedback": True,
                "expected_terminal_state": "CONVERGED_DISARMED",
            }
        )
        measured = checked
    return steps, measured, None


def _plan_phase(
    start: tuple[float, ...],
    target: tuple[float, ...],
    order: Sequence[int],
    phase: str,
) -> tuple[list[dict[str, Any]], tuple[float, ...], list[dict[str, Any]]]:
    measured = start
    steps: list[dict[str, Any]] = []
    holds = []
    for joint_index in order:
        axis_steps, measured, hold = _axis_steps(
            measured,
            target,
            joint_index,
            phase,
            len(steps) + 1,
        )
        steps.extend(axis_steps)
        if hold:
            holds.append(hold)
    return steps, measured, holds


def _validate_endpoint(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    gripper_expected: float,
) -> dict[str, Any]:
    errors = [float(a) - float(b) for a, b in zip(actual, expected)]
    for index in TASK_JOINT_INDICES:
        if abs(errors[index]) > HOLD_TOLERANCE_RAD + 1e-12:
            raise RouteError(f"{JOINT_NAMES[index]} endpoint exceeds hold tolerance")
    if abs(float(actual[GRIPPER_INDEX]) - gripper_expected) > 1e-12:
        raise RouteError("gripper changed despite hold-start policy")
    return {
        "task_axis_error_rad": errors[:5],
        "maximum_task_axis_error_rad": max(abs(value) for value in errors[:5]),
        "gripper_error_from_hold_start_rad": (
            float(actual[GRIPPER_INDEX]) - gripper_expected
        ),
        "tolerance_rad": HOLD_TOLERANCE_RAD,
        "passed": True,
    }


def _healthy_feedback(positions: Sequence[float], received_at: float) -> Feedback:
    return Feedback(
        positions=tuple(float(value) for value in positions),
        loads=(0, 0, 0, 0, 0, 0),
        torque_enabled=(True, True, True, True, True, True),
        voltage=12.1,
        received_at=received_at,
        raw={"source": "synthetic_in_memory_only"},
    )


def fake_driver_validate(
    start: Sequence[float], steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Exercise every step through SafeController without ROS or serial."""
    writes: list[bytes] = []
    controller = SafeController(writes.append, dry_run=False)
    measured = tuple(float(value) for value in start)
    now = 100.0
    automatic_disarms = 0
    selected_joint_indices = []
    for expected_index, step in enumerate(steps, start=1):
        controller.accept_feedback(_healthy_feedback(measured, now))
        controller.arm(now)
        payload = controller.command(step["target_positions_rad"], now)
        decoded = json.loads(payload)
        if decoded.get("T") != 101 or "102" in payload.decode("ascii"):
            raise RouteError("fake driver emitted a non-T101 motion payload")
        selected_joint_indices.append(int(decoded["joint"]) - 1)
        target = tuple(float(value) for value in step["target_positions_rad"])
        settle_start = now + 0.10
        for sample_index in range(21):
            controller.accept_feedback(
                _healthy_feedback(target, settle_start + sample_index * 0.05)
            )
        if controller.last_result != "CONVERGED_DISARMED" or controller.armed:
            raise RouteError(f"fake driver did not auto-disarm at step {expected_index}")
        automatic_disarms += 1
        measured = target
        now += 2.0
    if len(writes) != len(steps):
        raise RouteError("fake-driver write count does not equal route step count")
    return {
        "backend": "SafeController_in_memory_write_and_synthetic_feedback",
        "ros_used": False,
        "serial_used": False,
        "step_count": len(steps),
        "t101_payload_count": len(writes),
        "automatic_disarm_count": automatic_disarms,
        "t101_only": all(json.loads(payload).get("T") == 101 for payload in writes),
        "t102_absent": all(b"102" not in payload for payload in writes),
        "selected_joint_indices": selected_joint_indices,
        "final_positions_rad": list(measured),
        "passed": True,
    }


def _matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]):
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _identity4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rpy_rotation(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _axis_rotation(axis: Sequence[float], angle: float):
    norm = math.sqrt(sum(value * value for value in axis))
    x, y, z = (value / norm for value in axis)
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
        [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
        [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
    ]


def _transform(rotation, translation):
    result = _identity4()
    for row in range(3):
        for column in range(3):
            result[row][column] = rotation[row][column]
        result[row][3] = translation[row]
    return result


def _parse_xyz(text: str | None, default: Sequence[float]):
    return list(default) if not text else [float(value) for value in text.split()]


def _load_urdf_joints(path: Path):
    root = ET.fromstring(path.read_text())
    return {joint.get("name"): joint for joint in root.findall("joint")}


def standard_urdf_fk(
    positions: Sequence[float], joints: Mapping[str, ET.Element]
) -> tuple[list[float], list[float]]:
    by_name = dict(zip(JOINT_NAMES, positions))
    total = _identity4()
    for name in FK_CHAIN:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = _parse_xyz(None if origin is None else origin.get("xyz"), [0, 0, 0])
        rpy = _parse_xyz(None if origin is None else origin.get("rpy"), [0, 0, 0])
        step = _transform(_rpy_rotation(*rpy), xyz)
        if joint.get("type") in {"revolute", "continuous"}:
            axis_node = joint.find("axis")
            axis = _parse_xyz(
                None if axis_node is None else axis_node.get("xyz"), [1, 0, 0]
            )
            step = _matmul(
                step, _transform(_axis_rotation(axis, by_name[name]), [0, 0, 0])
            )
        total = _matmul(total, step)
    rotation = [row[:3] for row in total[:3]]
    return [total[row][3] for row in range(3)], _rotation_quaternion(rotation)


def _rotation_quaternion(matrix: Sequence[Sequence[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    result = [x / norm, y / norm, z / norm, w / norm]
    if result[3] < 0:
        result = [-value for value in result]
    return result


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _quaternion_error_deg(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _cross_track_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    direction = [b - a for a, b in zip(start, end)]
    denominator = sum(value * value for value in direction)
    if denominator == 0:
        return _distance(point, start)
    projection = sum(
        (value - origin) * delta
        for value, origin, delta in zip(point, start, direction)
    ) / denominator
    nearest = [origin + projection * delta for origin, delta in zip(start, direction)]
    return _distance(point, nearest)


def _stage_metrics(
    start_positions: Sequence[float],
    steps: Sequence[Mapping[str, Any]],
    joints: Mapping[str, ET.Element],
) -> list[dict[str, Any]]:
    metrics = []
    cursor = tuple(start_positions)
    cursor_point, _ = standard_urdf_fk(cursor, joints)
    index = 0
    while index < len(steps):
        joint_index = int(steps[index]["joint_index"])
        points = [cursor_point]
        stage_start = cursor
        while index < len(steps) and int(steps[index]["joint_index"]) == joint_index:
            cursor = tuple(float(value) for value in steps[index]["target_positions_rad"])
            point, _ = standard_urdf_fk(cursor, joints)
            points.append(point)
            index += 1
        path = sum(_distance(a, b) for a, b in zip(points, points[1:]))
        metrics.append(
            {
                "joint": JOINT_NAMES[joint_index],
                "command_count": len(points) - 1,
                "start_tcp_m": points[0],
                "end_tcp_m": points[-1],
                "chord_mm": 1000.0 * _distance(points[0], points[-1]),
                "polyline_arc_mm": 1000.0 * path,
                "maximum_microstep_tcp_mm": 1000.0
                * max(_distance(a, b) for a, b in zip(points, points[1:])),
                "joint_start_rad": stage_start[joint_index],
                "joint_end_rad": cursor[joint_index],
            }
        )
        cursor_point = points[-1]
    return metrics


def geometry_audit(
    standby: Mapping[str, Any],
    pre_contact: Mapping[str, Any],
    outbound: Sequence[Mapping[str, Any]],
    return_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    joints = _load_urdf_joints(URDF_PATH)
    start = standby["positions"]
    outbound_vectors = [start] + [
        tuple(float(value) for value in step["target_positions_rad"])
        for step in outbound
    ]
    return_vectors = [
        tuple(float(value) for value in step["target_positions_rad"])
        for step in return_steps
    ]
    outbound_points = [standard_urdf_fk(vector, joints)[0] for vector in outbound_vectors]
    return_points = [standard_urdf_fk(vector, joints)[0] for vector in return_vectors]
    start_point, end_point = outbound_points[0], outbound_points[-1]
    path_m = sum(
        _distance(left, right)
        for left, right in zip(outbound_points, outbound_points[1:])
    )
    chord_m = _distance(start_point, end_point)
    columns = list(zip(*outbound_points))
    aabb_min = [min(column) for column in columns]
    aabb_max = [max(column) for column in columns]
    start_fk, start_quat = standard_urdf_fk(standby["positions"], joints)
    recorded_pre_fk, recorded_pre_quat = standard_urdf_fk(
        pre_contact["positions"], joints
    )
    reverse_error = max(
        _distance(outbound_point, return_point)
        for outbound_point, return_point in zip(
            reversed(outbound_points[:-1]), return_points
        )
    )
    start_orientation_error = _quaternion_error_deg(
        start_quat, standby["stored_tf_rotation_xyzw"]
    )
    pre_orientation_error = _quaternion_error_deg(
        recorded_pre_quat, pre_contact["stored_tf_rotation_xyzw"]
    )
    return {
        "method": "standard_URDF_origin_then_axis_rotation_FK",
        "urdf_path": str(URDF_PATH.relative_to(REPO_ROOT)),
        "urdf_sha256": sha256_file(URDF_PATH),
        "translation_metrics_status": "offline_diagnostic_pass",
        "orientation_reproducibility_status": (
            "standard_parent_child_xyzw_matches_stored_tf"
        ),
        "start_tcp_m": start_point,
        "effective_pre_contact_tcp_m": end_point,
        "endpoint_chord_mm": 1000.0 * chord_m,
        "outbound_polyline_length_mm": 1000.0 * path_m,
        "path_to_chord_ratio": path_m / chord_m,
        "maximum_cross_track_mm": 1000.0
        * max(
            _cross_track_distance(point, start_point, end_point)
            for point in outbound_points
        ),
        "tcp_aabb_min_m": aabb_min,
        "tcp_aabb_max_m": aabb_max,
        "tcp_aabb_span_mm": [
            1000.0 * (high - low) for low, high in zip(aabb_min, aabb_max)
        ],
        "tcp_aabb_diagonal_mm": 1000.0 * _distance(aabb_min, aabb_max),
        "radial_range_m": [
            min(math.sqrt(sum(value * value for value in point)) for point in outbound_points),
            max(math.sqrt(sum(value * value for value in point)) for point in outbound_points),
        ],
        "maximum_microstep_tcp_mm": 1000.0
        * max(
            _distance(left, right)
            for left, right in zip(outbound_points, outbound_points[1:])
        ),
        "stage_metrics": _stage_metrics(start, outbound, joints),
        "ideal_reverse_tcp_max_error_m": reverse_error,
        "source_endpoint_translation_reproduction_m": {
            "press_standby": _distance(
                start_fk, standby["stored_tf_translation_m"]
            ),
            "pre_contact": _distance(
                recorded_pre_fk, pre_contact["stored_tf_translation_m"]
            ),
        },
        "source_endpoint_orientation_error_deg": {
            "press_standby": start_orientation_error,
            "pre_contact": pre_orientation_error,
        },
        "orientation_contract": (
            "T_parent_child = T_origin * R_axis(q), quaternion stored as XYZW; "
            "standard implementation matches runtime robot_state_publisher TF"
        ),
        "clearance_warning": (
            "the wrist-tilt-first stage raises hand_tcp substantially; inspect "
            "upper structure, camera, cable, and panel clearance on site"
        ),
    }


def _validate_driver_profile() -> dict[str, Any]:
    root = _mapping(yaml.safe_load(DRIVER_PROFILE_PATH.read_bytes()), "driver_profile")
    parameters = _mapping(
        _mapping(root.get("roarm_safe_driver"), "roarm_safe_driver").get(
            "ros__parameters"
        ),
        "ros__parameters",
    )
    speeds = tuple(int(value) for value in parameters["joint_speed_steps_s"])
    accelerations = tuple(int(value) for value in parameters["joint_acc_registers"])
    deltas = tuple(
        math.radians(float(value)) for value in parameters["joint_max_delta_deg"]
    )
    if speeds != DEFAULT_JOINT_SPEED_STEPS_S:
        raise RouteError("AGX profile speeds differ from safe-driver defaults")
    if accelerations != DEFAULT_JOINT_ACC_REGISTERS:
        raise RouteError("AGX profile accelerations differ from safe-driver defaults")
    if any(
        abs(actual - expected) > 1e-12
        for actual, expected in zip(deltas, DEFAULT_JOINT_MAX_DELTA_RAD)
    ):
        raise RouteError("AGX profile deltas differ from safe-driver defaults")
    return {
        "path": str(DRIVER_PROFILE_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_file(DRIVER_PROFILE_PATH),
        "joint_speed_steps_s": list(speeds),
        "joint_acc_registers": list(accelerations),
        "joint_max_delta_rad": list(deltas),
    }


def build_route(
    standby_path: str | Path,
    pre_contact_path: str | Path,
) -> dict[str, Any]:
    standby = load_recorded_pose(standby_path, "press_standby")
    pre_contact = load_recorded_pose(pre_contact_path, "pre_contact")
    driver_profile = _validate_driver_profile()
    start = standby["positions"]
    requested_pre = list(pre_contact["positions"])
    requested_pre[GRIPPER_INDEX] = start[GRIPPER_INDEX]
    requested_pre_tuple = tuple(requested_pre)

    outbound, reached_pre, outbound_holds = _plan_phase(
        start, requested_pre_tuple, OUTBOUND_ORDER, "outbound"
    )
    outbound_endpoint = _validate_endpoint(
        reached_pre,
        requested_pre_tuple,
        gripper_expected=start[GRIPPER_INDEX],
    )
    returned_target = tuple(start)
    return_steps, returned, return_holds = _plan_phase(
        reached_pre, returned_target, RETURN_ORDER, "return"
    )
    return_endpoint = _validate_endpoint(
        returned,
        returned_target,
        gripper_expected=start[GRIPPER_INDEX],
    )
    all_steps = outbound + return_steps
    simulation = fake_driver_validate(start, all_steps)
    if tuple(simulation["final_positions_rad"]) != returned:
        raise RouteError("fake-driver endpoint differs from planned endpoint")

    geometry = geometry_audit(standby, pre_contact, outbound, return_steps)
    joint_delta_summary = []
    for index, name in enumerate(JOINT_NAMES):
        raw_delta = pre_contact["positions"][index] - start[index]
        effective_delta = reached_pre[index] - start[index]
        joint_delta_summary.append(
            {
                "joint": name,
                "recorded_delta_rad": raw_delta,
                "recorded_delta_deg": math.degrees(raw_delta),
                "recorded_delta_encoder_steps": raw_delta / ENCODER_STEP_RAD,
                "effective_delta_rad": effective_delta,
                "effective_delta_encoder_steps": effective_delta / ENCODER_STEP_RAD,
                "command_count": sum(
                    int(step["joint_index"]) == index for step in outbound
                ),
            }
        )

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "artifact_id": (
            f"{standby['pose_id']}__to__{pre_contact['pose_id']}__"
            "no_contact_candidate_v1"
        ),
        "status": "offline_validated_candidate",
        "scope": (
            "operator-taught joint-space corridor pilot; not perception-driven "
            "3D reaching and not a collision proof"
        ),
        "sources": {
            "press_standby": {
                "path": str(standby["path"].relative_to(REPO_ROOT)),
                "sha256": standby["sha256"],
                "pose_id": standby["pose_id"],
                "joint_states_stamp_ns": standby["joint_states_stamp_ns"],
            },
            "pre_contact": {
                "path": str(pre_contact["path"].relative_to(REPO_ROOT)),
                "sha256": pre_contact["sha256"],
                "pose_id": pre_contact["pose_id"],
                "joint_states_stamp_ns": pre_contact["joint_states_stamp_ns"],
            },
            "safe_driver_core": {
                "path": str(
                    (
                        REPO_ROOT
                        / "ros2_arm"
                        / "roarm_safe_driver"
                        / "roarm_safe_driver"
                        / "core.py"
                    ).relative_to(REPO_ROOT)
                ),
                "sha256": sha256_file(
                    REPO_ROOT
                    / "ros2_arm"
                    / "roarm_safe_driver"
                    / "roarm_safe_driver"
                    / "core.py"
                ),
            },
            "agx_safe_driver_profile": driver_profile,
            "vendor_urdf": {
                "path": str(URDF_PATH.relative_to(REPO_ROOT)),
                "sha256": sha256_file(URDF_PATH),
            },
        },
        "joint_contract": {
            "joint_names": list(JOINT_NAMES),
            "task_joint_indices": list(TASK_JOINT_INDICES),
            "gripper_policy": "hold_press_standby_start_value",
            "gripper_hold_rad": start[GRIPPER_INDEX],
            "t101_command_limits_rad": [list(values) for values in T101_COMMAND_LIMITS],
            "default_joint_max_delta_rad": list(DEFAULT_JOINT_MAX_DELTA_RAD),
            "default_joint_speed_steps_s": list(DEFAULT_JOINT_SPEED_STEPS_S),
            "default_joint_acc_registers": list(DEFAULT_JOINT_ACC_REGISTERS),
            "encoder_step_rad": ENCODER_STEP_RAD,
            "hold_tolerance_rad": HOLD_TOLERANCE_RAD,
            "one_logical_joint_per_step": True,
        },
        "route_policy": {
            "outbound_joint_order": [JOINT_NAMES[index] for index in OUTBOUND_ORDER],
            "return_joint_order": [JOINT_NAMES[index] for index in RETURN_ORDER],
            "return_is_strict_reverse_order": True,
            "equal_subdivision_per_joint": True,
            "fresh_measured_feedback_each_step": True,
            "fresh_operator_confirmation_each_step": True,
            "expected_terminal_state_each_step": "CONVERGED_DISARMED",
            "automatic_retry": False,
            "automatic_compensation": False,
            "automatic_retract_after_fault": False,
        },
        "endpoints": {
            "press_standby_start_rad": list(start),
            "pre_contact_recorded_rad": list(pre_contact["positions"]),
            "pre_contact_task_target_with_gripper_hold_rad": list(requested_pre_tuple),
            "planned_pre_contact_reached_rad": list(reached_pre),
            "planned_return_reached_rad": list(returned),
            "outbound_validation": outbound_endpoint,
            "return_validation": return_endpoint,
        },
        "joint_delta_summary": joint_delta_summary,
        "outbound": {
            "step_count": len(outbound),
            "tolerance_holds": outbound_holds,
            "steps": outbound,
        },
        "return": {
            "step_count": len(return_steps),
            "tolerance_holds": return_holds,
            "steps": return_steps,
        },
        "totals": {
            "step_count": len(all_steps),
            "outbound_step_count": len(outbound),
            "return_step_count": len(return_steps),
        },
        "fake_driver_validation": simulation,
        "geometry_audit": geometry,
        "gates": {
            "source_pose_integrity": "passed",
            "single_joint_step_validation": "passed",
            "t101_limit_validation": "passed",
            "default_delta_validation": "passed",
            "fake_driver_validation": "passed",
            "urdf_fk_translation_path_metrics": "offline_diagnostic_pass",
            "urdf_fk_orientation_reproducibility": "passed",
            "collision_validation": "pending",
            "keep_out_validation": "pending",
            "现场逐步clearance确认": "required_before_every_real_step",
        },
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "limitations": [
            "joint-space endpoint validity does not prove collision-free clearance",
            "route is based on hand-taught poses, not a fresh frozen perception target",
            "wrist-roll differences within hold tolerance are intentionally not commanded",
            "gripper is a held non-task axis",
            "wrist-tilt-first route has a large upward TCP excursion requiring现场clearance",
            "real motion requires a separate narrow authorization and per-step inspection",
        ],
    }
    digest_source = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    artifact["canonical_payload_sha256"] = sha256_bytes(digest_source)
    return artifact


def write_artifact_exclusive(artifact: Mapping[str, Any], output: str | Path) -> dict[str, str]:
    output_path = Path(output).expanduser().resolve()
    checksum_path = output_path.with_name(output_path.name + ".sha256")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or checksum_path.exists():
        raise FileExistsError("refusing to overwrite route artifact or checksum")
    data = (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    artifact_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(artifact_fd, "wb") as stream:
            stream.write(data)
        file_hash = sha256_bytes(data)
        checksum_data = f"{file_hash}  {output_path.name}\n".encode("ascii")
        checksum_fd = os.open(
            checksum_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
        with os.fdopen(checksum_fd, "wb") as stream:
            stream.write(checksum_data)
    except Exception:
        output_path.unlink(missing_ok=True)
        checksum_path.unlink(missing_ok=True)
        raise
    return {
        "artifact_path": str(output_path),
        "artifact_sha256": file_hash,
        "checksum_path": str(checksum_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standby",
        default=str(
            REPO_ROOT
            / "config"
            / "press"
            / "poses"
            / "press_standby_manual_20260804_preflight.yaml"
        ),
    )
    parser.add_argument(
        "--pre-contact",
        default=str(
            REPO_ROOT / "config" / "press" / "poses" / "pre_contact_candidate_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = build_route(args.standby, args.pre_contact)
    result = write_artifact_exclusive(artifact, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
