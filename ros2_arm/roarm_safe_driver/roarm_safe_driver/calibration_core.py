"""Fail-closed, ROS-independent primitives for calibration sample recording.

The recorder is intentionally separate from the serial driver: it operates only
on already measured ROS data and never contains an arm command or serial path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence

from .core import ENCODER_STEP_RAD, JOINT_NAMES, SafetyError, ordered_positions

MAX_RGB_JOINT_SKEW_NS = 50_000_000
SETTLE_WINDOW_NS = 1_000_000_000
MAX_JOINT_SAMPLE_GAP_NS = 100_000_000
MIN_SETTLE_SAMPLES = 20
MAX_SETTLE_DELTA_RAD = 2.0 * ENCODER_STEP_RAD
TAG_WINDOW_NS = 1_000_000_000
MAX_TAG_SAMPLE_GAP_NS = 250_000_000
# The current AGX calibration stack delivers about 12.7 exact-stamp
# detection/TF pairs per second.  Keep the complete one-second window and all
# gap/spread gates, but require a count that the measured stream can satisfy.
MIN_TAG_SAMPLES = 10
MIN_TAG_DECISION_MARGIN = 50.0
MAX_TAG_TRANSLATION_SPREAD_M = 0.003
MAX_TAG_ROTATION_SPREAD_DEG = 1.5
TAG_AGGREGATION_METHOD = (
    "translation_component_median;"
    "quaternion_hemisphere_aligned_normalized_component_mean"
)
MIN_SAFE_VOLTAGE_V = 11.0
MAX_SAFE_VOLTAGE_V = 13.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_FILE_PATTERN = re.compile(r"^S[0-9]{4}\.json$")
ACTIVE_SOURCE_RELATIVE_PATHS = {
    "calibration_core.py": (
        "ros2_arm/roarm_safe_driver/roarm_safe_driver/calibration_core.py"
    ),
    "calibration_recorder.py": (
        "ros2_arm/roarm_safe_driver/roarm_safe_driver/calibration_recorder.py"
    ),
    "safe_driver_core.py": "ros2_arm/roarm_safe_driver/roarm_safe_driver/core.py",
    "safe_driver_node.py": "ros2_arm/roarm_safe_driver/roarm_safe_driver/node.py",
    "safe_driver_transport.py": (
        "ros2_arm/roarm_safe_driver/roarm_safe_driver/transport.py"
    ),
    "torque_transaction.py": (
        "ros2_arm/roarm_safe_driver/roarm_safe_driver/torque_transaction.py"
    ),
    "tf_bridge.py": "ros2_arm/roarm_safe_driver/roarm_safe_driver/tf_bridge.py",
    "phase4_tf_bridge.launch.py": (
        "ros2_arm/roarm_safe_driver/launch/phase4_tf_bridge.launch.py"
    ),
    "roarm_m3.xacro": (
        "RoArm-M3/ros2/roarm_ws/src/roarm_main/roarm_description/"
        "urdf/roarm_m3/roarm_m3.xacro"
    ),
    "apriltag_calibration_live_view.py": "tools/apriltag_calibration_live_view.py",
    "live_capture_core.py": "tools/calibration/live_capture_core.py",
    "live_torque_core.py": "tools/calibration/live_torque_core.py",
    "validate_formal_capture_plan.py": (
        "tools/calibration/validate_formal_capture_plan.py"
    ),
    "apriltag_gemini335.yaml": "config/calibration/apriltag_gemini335.yaml",
    "start_gemini335_apriltag.sh": "tools/start_gemini335_apriltag.sh",
    "gemini335_canonical_adapter_manifest.yaml": (
        "ros2_perception/ros2_ws/src/lift_robot_camera_adapter/config/"
        "gemini335_canonical_adapter_manifest.yaml"
    ),
    "gemini335_canonical_cpp.yaml": (
        "ros2_perception/ros2_ws/src/lift_robot_camera_adapter_cpp/config/"
        "gemini335_canonical_cpp.yaml"
    ),
    "gemini335_canonical_adapter.cpp": (
        "ros2_perception/ros2_ws/src/lift_robot_camera_adapter_cpp/src/"
        "gemini335_canonical_adapter.cpp"
    ),
    "gemini335_canonical_components.launch.py": (
        "ros2_perception/ros2_ws/src/lift_robot_camera_adapter_cpp/launch/"
        "gemini335_canonical_components.launch.py"
    ),
    "coordinate_contract_v1.yaml": "config/coordinate_contract_v1.yaml",
    "agx_unified_safe.yaml": (
        "ros2_arm/roarm_safe_driver/config/agx_unified_safe.yaml"
    ),
    "eye_to_hand_pilot_solver.py": "tools/calibration/eye_to_hand_pilot_solver.py",
    "eye_to_hand_formal_solver.py": "tools/calibration/eye_to_hand_formal_solver.py",
}
AGX_ACTIVE_SOURCE_PATHS = {
    "calibration_core.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/calibration_core.py"
    ),
    "calibration_recorder.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/calibration_recorder.py"
    ),
    "safe_driver_core.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/core.py"
    ),
    "safe_driver_node.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/node.py"
    ),
    "safe_driver_transport.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/transport.py"
    ),
    "torque_transaction.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/torque_transaction.py"
    ),
    "tf_bridge.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/tf_bridge.py"
    ),
    "phase4_tf_bridge.launch.py": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "launch/phase4_tf_bridge.launch.py"
    ),
    "roarm_m3.xacro": (
        "/home/robot/roarm_ws/src/roarm_main/roarm_description/"
        "urdf/roarm_m3/roarm_m3.xacro"
    ),
    "agx_unified_safe.yaml": (
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "config/agx_unified_safe.yaml"
    ),
    "gemini335_canonical_cpp.yaml": (
        "/data/ros_ws/lift_robot_calibration_ws/src/"
        "lift_robot_camera_adapter_cpp/config/gemini335_canonical_cpp.yaml"
    ),
    "gemini335_canonical_adapter.cpp": (
        "/data/ros_ws/lift_robot_calibration_ws/src/"
        "lift_robot_camera_adapter_cpp/src/gemini335_canonical_adapter.cpp"
    ),
    "gemini335_canonical_components.launch.py": (
        "/data/ros_ws/lift_robot_calibration_ws/src/"
        "lift_robot_camera_adapter_cpp/launch/"
        "gemini335_canonical_components.launch.py"
    ),
}
MONITOR_STATUS_SOURCE = "monitor"
SAFE_DRIVER_STATUS_SOURCE = "safe_driver"
SAFE_DRIVER_CAPTURE_STATES = (
    "DISARMED",
    "DISARMED:operator",
    "CONVERGED_DISARMED",
    "FAULT_CLEARED_DISARMED",
)


@dataclass(frozen=True)
class JointSample:
    stamp_ns: int
    positions: tuple[float, ...]


@dataclass(frozen=True)
class TagObservation:
    stamp_ns: int
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    family: str
    tag_id: int
    hamming: int
    decision_margin: float


@dataclass(frozen=True)
class TagWindow:
    observations: tuple[TagObservation, ...]
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    duration_ns: int
    translation_spread_m: float
    rotation_spread_deg: float
    aggregation_method: str


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise SafetyError(f"{field} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise SafetyError(f"{field} is not finite")
    return result


def stamp_ns(stamp) -> int:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise SafetyError("ROS timestamp is not positive")
    return value


def joint_sample(stamp_value_ns: int, names: Sequence[str], positions: Sequence[float]) -> JointSample:
    if isinstance(stamp_value_ns, bool) or not isinstance(stamp_value_ns, int) or stamp_value_ns <= 0:
        raise SafetyError("joint timestamp must be a positive integer")
    return JointSample(stamp_value_ns, ordered_positions(names, positions))


def nearest_joint_sample(
    samples: Iterable[JointSample], rgb_stamp_ns: int, max_skew_ns: int = MAX_RGB_JOINT_SKEW_NS
) -> JointSample:
    if max_skew_ns < 0:
        raise SafetyError("maximum skew cannot be negative")
    candidates = list(samples)
    if not candidates:
        raise SafetyError("no joint feedback available")
    # Earlier stamp is deterministic tie-breaker, never an unqualified latest sample.
    selected = min(candidates, key=lambda item: (abs(item.stamp_ns - rgb_stamp_ns), item.stamp_ns))
    if abs(selected.stamp_ns - rgb_stamp_ns) > max_skew_ns:
        raise SafetyError("RGB and joint feedback skew exceeds 50 ms")
    return selected


def validate_settled(
    samples: Iterable[JointSample], selected_stamp_ns: int,
    *, window_ns: int = SETTLE_WINDOW_NS, min_samples: int = MIN_SETTLE_SAMPLES,
    max_gap_ns: int = MAX_JOINT_SAMPLE_GAP_NS, max_delta_rad: float = MAX_SETTLE_DELTA_RAD,
) -> tuple[JointSample, ...]:
    """Require one full, contiguous, measured window ending at the sample."""
    if min_samples < 1 or window_ns <= 0 or max_gap_ns <= 0 or max_delta_rad < 0:
        raise SafetyError("invalid settle policy")
    lower = selected_stamp_ns - window_ns
    ordered = tuple(sorted((item for item in samples if item.stamp_ns <= selected_stamp_ns), key=lambda item: item.stamp_ns))
    window = tuple(item for item in ordered if lower <= item.stamp_ns <= selected_stamp_ns)
    if len(window) < min_samples:
        raise SafetyError("joint feedback has fewer than 20 settle samples")
    # Sensor messages do not normally land exactly on the one-second boundary.
    # Retain the immediately preceding measured sample to prove continuity across
    # it; the existing maximum-gap policy then limits that boundary uncertainty.
    preceding = [item for item in ordered if item.stamp_ns < lower]
    if preceding:
        window = (preceding[-1],) + window
    if window[0].stamp_ns > lower:
        raise SafetyError("joint feedback does not cover full settle window")
    for previous, current in zip(window, window[1:]):
        gap = current.stamp_ns - previous.stamp_ns
        if gap <= 0 or gap > max_gap_ns:
            raise SafetyError("joint feedback settle window has a gap")
    for index, name in enumerate(JOINT_NAMES):
        span = max(item.positions[index] for item in window) - min(item.positions[index] for item in window)
        if span > max_delta_rad + 1e-12:
            raise SafetyError(f"joint is not settled: {name}")
    return window


def validate_transform(translation: Sequence[object], rotation_xyzw: Sequence[object], label: str) -> tuple[list[float], list[float]]:
    if len(translation) != 3 or len(rotation_xyzw) != 4:
        raise SafetyError(f"{label} transform shape is invalid")
    translation_checked = [_finite(value, f"{label}.translation") for value in translation]
    rotation_checked = [_finite(value, f"{label}.rotation") for value in rotation_xyzw]
    norm = math.sqrt(sum(value * value for value in rotation_checked))
    if abs(norm - 1.0) > 1e-3:
        raise SafetyError(f"{label} quaternion is not unit length")
    return translation_checked, rotation_checked


def tag_observation(
    stamp_value_ns: int,
    translation: Sequence[object],
    rotation_xyzw: Sequence[object],
    *,
    family: str,
    tag_id: int,
    hamming: int,
    decision_margin: object,
) -> TagObservation:
    """Create one fully validated, timestamped Tag pose observation."""
    if (
        isinstance(stamp_value_ns, bool)
        or not isinstance(stamp_value_ns, int)
        or stamp_value_ns <= 0
    ):
        raise SafetyError("Tag timestamp must be a positive integer")
    if family != "tag36h11" or tag_id != 0 or hamming != 0:
        raise SafetyError("Tag observation identity or hamming is invalid")
    margin = _finite(decision_margin, "Tag.decision_margin")
    if margin < MIN_TAG_DECISION_MARGIN:
        raise SafetyError("Tag decision margin is below 50")
    translation_checked, rotation_checked = validate_transform(
        translation, rotation_xyzw, "Tag"
    )
    if translation_checked[2] <= 0.0:
        raise SafetyError("AprilTag depth is not positive")
    return TagObservation(
        stamp_ns=stamp_value_ns,
        translation_m=tuple(translation_checked),
        rotation_xyzw=tuple(rotation_checked),
        family=family,
        tag_id=tag_id,
        hamming=hamming,
        decision_margin=margin,
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _normalized_quaternion(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-15:
        raise SafetyError("Tag quaternion aggregate has zero norm")
    return tuple(value / norm for value in quaternion)


def _quaternion_distance_deg(
    left: Sequence[float], right: Sequence[float]
) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def validate_tag_window(
    observations: Iterable[TagObservation],
    selected_stamp_ns: int,
    *,
    window_ns: int = TAG_WINDOW_NS,
    min_samples: int = MIN_TAG_SAMPLES,
    max_gap_ns: int = MAX_TAG_SAMPLE_GAP_NS,
    max_translation_spread_m: float = MAX_TAG_TRANSLATION_SPREAD_M,
    max_rotation_spread_deg: float = MAX_TAG_ROTATION_SPREAD_DEG,
) -> TagWindow:
    """Freeze, validate, and robustly aggregate a static Tag window.

    Spread is deliberately conservative: translation uses the maximum
    pairwise Euclidean distance and rotation uses the maximum pairwise
    quaternion geodesic distance.
    """
    if (
        isinstance(selected_stamp_ns, bool)
        or not isinstance(selected_stamp_ns, int)
        or selected_stamp_ns <= 0
        or window_ns <= 0
        or min_samples < 1
        or max_gap_ns <= 0
        or max_translation_spread_m < 0.0
        or max_rotation_spread_deg < 0.0
    ):
        raise SafetyError("invalid Tag window policy")
    frozen = tuple(
        sorted(
            (item for item in observations if item.stamp_ns <= selected_stamp_ns),
            key=lambda item: item.stamp_ns,
        )
    )
    if not frozen or frozen[-1].stamp_ns != selected_stamp_ns:
        raise SafetyError("selected RGB stamp has no Tag observation")
    lower = selected_stamp_ns - window_ns
    window = tuple(item for item in frozen if lower <= item.stamp_ns)
    preceding = tuple(item for item in frozen if item.stamp_ns < lower)
    if preceding:
        window = (preceding[-1],) + window
    if not window or window[0].stamp_ns > lower:
        raise SafetyError("Tag observations do not cover full static window")
    if len(window) < min_samples:
        raise SafetyError(
            f"Tag window has fewer than {min_samples} observations"
        )
    for previous, current in zip(window, window[1:]):
        gap = current.stamp_ns - previous.stamp_ns
        if gap <= 0 or gap > max_gap_ns:
            raise SafetyError("Tag static window has a timestamp gap")
    window = tuple(
        tag_observation(
            item.stamp_ns,
            item.translation_m,
            item.rotation_xyzw,
            family=item.family,
            tag_id=item.tag_id,
            hamming=item.hamming,
            decision_margin=item.decision_margin,
        )
        for item in window
    )

    translation = tuple(
        _median(tuple(item.translation_m[index] for item in window))
        for index in range(3)
    )
    reference = _normalized_quaternion(window[0].rotation_xyzw)
    aligned = []
    for item in window:
        quaternion = _normalized_quaternion(item.rotation_xyzw)
        if sum(a * b for a, b in zip(reference, quaternion)) < 0.0:
            quaternion = tuple(-value for value in quaternion)
        aligned.append(quaternion)
    rotation = _normalized_quaternion(
        tuple(
            sum(item[index] for item in aligned) / len(aligned)
            for index in range(4)
        )
    )

    translation_spread = 0.0
    rotation_spread = 0.0
    for index, left in enumerate(window):
        for right in window[index + 1 :]:
            translation_spread = max(
                translation_spread,
                math.sqrt(
                    sum(
                        (a - b) ** 2
                        for a, b in zip(left.translation_m, right.translation_m)
                    )
                ),
            )
            rotation_spread = max(
                rotation_spread,
                _quaternion_distance_deg(
                    _normalized_quaternion(left.rotation_xyzw),
                    _normalized_quaternion(right.rotation_xyzw),
                ),
            )
    if translation_spread > max_translation_spread_m + 1e-12:
        raise SafetyError("Tag translation spread exceeds 0.003 m")
    if rotation_spread > max_rotation_spread_deg + 1e-12:
        raise SafetyError("Tag rotation spread exceeds 1.5 deg")
    return TagWindow(
        observations=window,
        translation_m=translation,
        rotation_xyzw=rotation,
        duration_ns=window[-1].stamp_ns - window[0].stamp_ns,
        translation_spread_m=translation_spread,
        rotation_spread_deg=rotation_spread,
        aggregation_method=TAG_AGGREGATION_METHOD,
    )


def validate_status_payload(payload: object, source: str) -> dict:
    """Validate one recorder status input and describe its torque evidence.

    The read-only monitor exports the six measured torque switches directly.
    Safe-driver capture requires the sample-aligned feedback status added for
    this purpose; the retained control-state topic is deliberately insufficient.
    """
    if not isinstance(payload, dict):
        raise SafetyError("status payload is not an object")
    voltage = _finite(payload.get("voltage"), "status.voltage")
    if not MIN_SAFE_VOLTAGE_V <= voltage <= MAX_SAFE_VOLTAGE_V:
        raise SafetyError("status voltage is outside the safe range")

    if source == MONITOR_STATUS_SOURCE:
        if payload.get("state") != "MONITOR_ONLY_NO_WRITES":
            raise SafetyError("monitor is not healthy")
        torque = payload.get("torque_enabled")
        if (
            not isinstance(torque, list)
            or len(torque) != len(JOINT_NAMES)
            or any(value is not True for value in torque)
        ):
            raise SafetyError("all six direct torque indicators must be true")
        return {
            "voltage_v": voltage,
            "torque_enabled": [True] * len(JOINT_NAMES),
            "torque_evidence": {
                "source_topic": "/roarm_monitor/status",
                "mode": "direct_t1051_joint_flags",
                "direct_joint_torque_flags_available": True,
            },
        }

    if source != SAFE_DRIVER_STATUS_SOURCE:
        raise SafetyError("unknown recorder status source")
    if payload.get("state") not in SAFE_DRIVER_CAPTURE_STATES:
        raise SafetyError("safe driver is not in an allowlisted disarmed state")
    if payload.get("armed") is not False:
        raise SafetyError("safe driver must be explicitly disarmed")
    if payload.get("dry_run") is not False:
        raise SafetyError("safe driver must be the non-dry-run feedback owner")
    if payload.get("fault") != "":
        raise SafetyError("safe driver has a latched fault")
    feedback_stamp_ns = payload.get("feedback_stamp_ns")
    if (
        isinstance(feedback_stamp_ns, bool)
        or not isinstance(feedback_stamp_ns, int)
        or feedback_stamp_ns <= 0
    ):
        raise SafetyError("safe feedback timestamp must be a positive integer")
    positions = payload.get("positions_rad")
    if not isinstance(positions, list) or len(positions) != len(JOINT_NAMES):
        raise SafetyError("safe feedback positions must contain all six joints")
    positions_checked = [
        _finite(value, f"status.positions_rad[{index}]")
        for index, value in enumerate(positions)
    ]
    torque = payload.get("torque_enabled")
    if (
        not isinstance(torque, list)
        or len(torque) != len(JOINT_NAMES)
        or any(value is not True for value in torque)
    ):
        raise SafetyError("all six sample-aligned torque indicators must be true")
    return {
        "voltage_v": voltage,
        "torque_enabled": [True] * len(JOINT_NAMES),
        "feedback_stamp_ns": feedback_stamp_ns,
        "positions_rad": positions_checked,
        "torque_evidence": {
            "source_topic": "/roarm_safe/feedback_status",
            "mode": "sample_aligned_t1051_joint_flags",
            "direct_joint_torque_flags_available": True,
            "safe_driver_state": payload["state"],
            "armed": payload["armed"],
            "dry_run": payload["dry_run"],
            "fault": payload["fault"],
            "feedback_stamp_ns": feedback_stamp_ns,
        },
    }


def validate_status_alignment(status_evidence: dict, joint: JointSample) -> None:
    """Require safe feedback evidence to describe the exact selected joint frame."""
    if status_evidence.get("feedback_stamp_ns") != joint.stamp_ns:
        raise SafetyError("safe status timestamp does not match joint feedback")
    positions = status_evidence.get("positions_rad")
    if not isinstance(positions, list) or len(positions) != len(joint.positions):
        raise SafetyError("safe status positions do not match joint feedback shape")
    if any(
        abs(measured - expected) > 1e-12
        for measured, expected in zip(positions, joint.positions)
    ):
        raise SafetyError("safe status positions do not match joint feedback")


def validate_safe_status_window(
    samples: Iterable[JointSample],
    status_payload_by_stamp: Mapping[int, dict],
) -> tuple[dict, ...]:
    """Require healthy, sample-aligned safe-driver evidence for a settle window.

    Invalid safe-driver status frames are rejected before entering the mapping.
    A missing entry therefore fails closed instead of allowing a Torque Off,
    armed, faulted, dry-run, or otherwise unhealthy interval to contribute to
    the measured settle proof.
    """
    window = tuple(samples)
    if not window:
        raise SafetyError("safe status settle window is empty")
    evidence_window = []
    for joint in window:
        payload = status_payload_by_stamp.get(joint.stamp_ns)
        if payload is None:
            raise SafetyError(
                "safe status is missing for a joint settle-window frame"
            )
        evidence = validate_status_payload(payload, SAFE_DRIVER_STATUS_SOURCE)
        validate_status_alignment(evidence, joint)
        evidence_window.append(evidence)
    return tuple(evidence_window)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_source_path_for_plan(capture_plan_path: Path, source_name: str) -> Path:
    """Resolve the active source exactly as the formal plan validator does."""

    if source_name not in ACTIVE_SOURCE_RELATIVE_PATHS:
        raise SafetyError(f"unknown frozen source name: {source_name}")
    capture_plan_path = Path(capture_plan_path).expanduser().resolve()
    try:
        repository = capture_plan_path.parents[3]
    except IndexError as exc:
        raise SafetyError("capture plan path is not inside the repository") from exc
    agx_path_raw = AGX_ACTIVE_SOURCE_PATHS.get(source_name)
    if agx_path_raw is not None:
        agx_path = Path(agx_path_raw)
        if agx_path.is_file():
            return agx_path
    return repository / ACTIVE_SOURCE_RELATIVE_PATHS[source_name]


def validate_active_source_hashes(
    capture_plan_path: Path,
    declared_hashes: Mapping[str, str],
) -> dict[str, Path]:
    """Require every active source/config byte hash to match the frozen plan."""

    if (
        not isinstance(declared_hashes, Mapping)
        or set(declared_hashes) != set(ACTIVE_SOURCE_RELATIVE_PATHS)
    ):
        raise SafetyError("frozen source set does not match the active source contract")
    resolved: dict[str, Path] = {}
    for source_name in ACTIVE_SOURCE_RELATIVE_PATHS:
        expected_digest = declared_hashes.get(source_name)
        if (
            not isinstance(expected_digest, str)
            or not SHA256_PATTERN.fullmatch(expected_digest)
        ):
            raise SafetyError(f"frozen source digest is invalid: {source_name}")
        source_path = active_source_path_for_plan(capture_plan_path, source_name)
        if not source_path.is_file():
            raise SafetyError(f"active frozen source is missing: {source_name}")
        if sha256_file(source_path) != expected_digest:
            raise SafetyError(f"active frozen source hash mismatch: {source_name}")
        resolved[source_name] = source_path
    return resolved


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SafetyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_strict(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f"JSON evidence is unreadable: {path.name}") from exc


@contextmanager
def raw_directory_lock(raw_dir: Path):
    """Serialize validation, append, and integrity-index commit."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_path = raw_dir / ".capture.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_raw_integrity(raw_dir: Path, *, dataset_id: str) -> dict[str, str]:
    """Verify that every existing raw sample matches both integrity indexes."""

    raw_dir = Path(raw_dir)
    raw_candidates = sorted(path for path in raw_dir.glob("S*.json"))
    if any(
        not SAMPLE_FILE_PATTERN.fullmatch(path.name)
        or not path.is_file()
        or path.is_symlink()
        for path in raw_candidates
    ):
        raise SafetyError("raw sample filename or file type is invalid")
    stale_transactions = sorted(
        (
            *raw_dir.glob(".S*.json.*"),
            *raw_dir.glob(".manifest.json.*"),
            *raw_dir.glob(".SHA256SUMS.*"),
        )
    )
    if stale_transactions:
        raise SafetyError("stale raw integrity transaction artifact exists")
    sample_paths = raw_candidates
    expected_files = [path.name for path in sample_paths]
    manifest_path = raw_dir / "manifest.json"
    sums_path = raw_dir / "SHA256SUMS"
    if not sample_paths and not manifest_path.exists() and not sums_path.exists():
        return {}
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not sums_path.is_file()
        or sums_path.is_symlink()
    ):
        raise SafetyError("raw integrity indexes are missing")

    manifest = _read_json_strict(manifest_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != dataset_id
        or not isinstance(manifest.get("raw_samples"), list)
    ):
        raise SafetyError("raw manifest header is invalid")

    manifest_hashes: dict[str, str] = {}
    for entry in manifest["raw_samples"]:
        if not isinstance(entry, dict):
            raise SafetyError("raw manifest entry is invalid")
        name = entry.get("file")
        digest = entry.get("sha256")
        if (
            not isinstance(name, str)
            or not SAMPLE_FILE_PATTERN.fullmatch(name)
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
            or name in manifest_hashes
        ):
            raise SafetyError("raw manifest entry is invalid")
        manifest_hashes[name] = digest
    if list(manifest_hashes) != expected_files:
        raise SafetyError("raw manifest file set or order does not match raw samples")

    checksum_hashes: dict[str, str] = {}
    try:
        checksum_lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SafetyError("raw SHA256SUMS is unreadable") from exc
    for line in checksum_lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise SafetyError("raw SHA256SUMS entry is invalid")
        digest, name = parts[0], parts[1].lstrip("*")
        if (
            not SHA256_PATTERN.fullmatch(digest)
            or not SAMPLE_FILE_PATTERN.fullmatch(name)
            or name in checksum_hashes
        ):
            raise SafetyError("raw SHA256SUMS entry is invalid")
        checksum_hashes[name] = digest
    if list(checksum_hashes) != expected_files:
        raise SafetyError("raw SHA256SUMS file set or order does not match raw samples")

    computed_hashes = {path.name: sha256_file(path) for path in sample_paths}
    if manifest_hashes != checksum_hashes or manifest_hashes != computed_hashes:
        raise SafetyError("raw sample digest does not match integrity indexes")
    return computed_hashes


def canonical_integrity_index_hashes(
    dataset_id: str,
    sample_hashes: Mapping[str, str],
) -> tuple[str, str]:
    """Return canonical manifest and SHA256SUMS hashes for one raw prefix."""

    entries = [
        {"file": filename, "sha256": digest}
        for filename, digest in sample_hashes.items()
    ]
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "raw_samples": entries,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    sums_bytes = "".join(
        f"{entry['sha256']}  {entry['file']}\n" for entry in entries
    ).encode("utf-8")
    return (
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(sums_bytes).hexdigest(),
    )


def validate_preappend_integrity_chain(
    raw_dir: Path,
    *,
    dataset_id: str,
    existing_hashes: Mapping[str, str],
    formal_start: int = 12,
) -> None:
    """Verify every formal sample against its actual preceding raw prefix."""

    ordered_names = list(existing_hashes)
    for index, filename in enumerate(ordered_names):
        match = SAMPLE_FILE_PATTERN.fullmatch(filename)
        if match is None or int(filename[1:5]) < formal_start:
            continue
        document = _read_json_strict(Path(raw_dir) / filename)
        if not isinstance(document, dict):
            raise SafetyError(f"formal sample is not a JSON object: {filename}")
        metadata = document.get("metadata")
        preappend = (
            None
            if not isinstance(metadata, dict)
            else metadata.get("preappend_raw_integrity")
        )
        if not isinstance(preappend, dict):
            raise SafetyError(
                f"formal sample pre-append integrity proof is missing: {filename}"
            )
        expected_prior = {
            name: existing_hashes[name]
            for name in ordered_names[:index]
        }
        if (
            preappend.get("existing_sample_count") != len(expected_prior)
            or preappend.get("existing_sample_hashes_sha256") != expected_prior
        ):
            raise SafetyError(
                f"formal sample pre-append raw hash chain is invalid: {filename}"
            )
        expected_manifest, expected_sums = canonical_integrity_index_hashes(
            dataset_id,
            expected_prior,
        )
        indexes_were_absent = (
            not expected_prior
            and preappend.get("manifest_sha256") is None
            and preappend.get("sha256sums_sha256") is None
        )
        if not indexes_were_absent:
            if preappend.get("manifest_sha256") != expected_manifest:
                raise SafetyError(
                    f"formal sample pre-append manifest hash is invalid: {filename}"
                )
            if preappend.get("sha256sums_sha256") != expected_sums:
                raise SafetyError(
                    f"formal sample pre-append SHA256SUMS hash is invalid: {filename}"
                )


def write_sample_exclusive(raw_dir: Path, sample_id: str, document: dict) -> Path:
    """Atomically create an immutable sample; existing evidence is never replaced."""
    if not sample_id or any(part in sample_id for part in ("/", "\\", "..")):
        raise SafetyError("sample_id is not a safe filename")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{sample_id}.json"
    try:
        payload = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SafetyError("sample document is not JSON-safe") from exc
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{sample_id}.json.",
        dir=raw_dir,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o640)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise SafetyError(
                f"refusing to overwrite existing sample: {destination}"
            ) from exc
        directory_descriptor = os.open(raw_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.unlink(temporary)
        directory_descriptor = os.open(raw_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def append_integrity_indexes(
    raw_dir: Path,
    *,
    dataset_id: str,
    new_sample_path: Path,
    existing_hashes: Mapping[str, str],
) -> tuple[Path, Path]:
    """Commit one new hash while preserving every prior indexed digest."""

    raw_dir = Path(raw_dir)
    new_sample_path = Path(new_sample_path)
    if new_sample_path.parent != raw_dir or new_sample_path.is_symlink():
        raise SafetyError("new sample path is outside the raw directory")
    expected_files = [*existing_hashes, new_sample_path.name]
    current_paths = sorted(
        path
        for path in raw_dir.glob("S*.json")
        if path.is_file() and not path.is_symlink()
    )
    if [path.name for path in current_paths] != expected_files:
        raise SafetyError("raw sample set changed during append")
    for filename, expected_digest in existing_hashes.items():
        if sha256_file(raw_dir / filename) != expected_digest:
            raise SafetyError("existing raw sample changed during append")

    entries = [
        {"file": filename, "sha256": digest}
        for filename, digest in existing_hashes.items()
    ]
    entries.append(
        {
            "file": new_sample_path.name,
            "sha256": sha256_file(new_sample_path),
        }
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "raw_samples": entries,
    }
    for name, content in (
        (
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        ),
        (
            "SHA256SUMS",
            "".join(
                f"{entry['sha256']}  {entry['file']}\n" for entry in entries
            ),
        ),
    ):
        target = raw_dir / name
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=raw_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_descriptor = os.open(raw_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    validate_raw_integrity(raw_dir, dataset_id=dataset_id)
    return raw_dir / "manifest.json", raw_dir / "SHA256SUMS"


def rebuild_manifest(raw_dir: Path, *, dataset_id: str) -> tuple[Path, Path]:
    """Initialize empty integrity indexes; never re-baseline existing evidence."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    sums_path = raw_dir / "SHA256SUMS"
    if (
        any(raw_dir.glob("S*.json"))
        or manifest_path.exists()
        or sums_path.exists()
    ):
        raise SafetyError(
            "refusing to rebuild integrity indexes for an established raw dataset"
        )
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "raw_samples": []}
    for name, content in (
        (
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        ),
        ("SHA256SUMS", ""),
    ):
        target = raw_dir / name
        descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=raw_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise SafetyError(
                    "refusing to replace an existing raw integrity index"
                ) from exc
            os.unlink(temporary)
            directory_descriptor = os.open(raw_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
    validate_raw_integrity(raw_dir, dataset_id=dataset_id)
    return manifest_path, sums_path
