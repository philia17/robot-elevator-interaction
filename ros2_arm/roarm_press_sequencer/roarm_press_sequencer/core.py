"""ROS-independent safety core for the Phase 3 press sequencer.

This module deliberately knows nothing about serial protocols.  It produces at
most one six-name JointState target whose delta from the latest measured sample
is exactly one joint.  The downstream safe driver remains responsible for
serializing the corresponding T:101 command.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml


JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)
POSE_NAMES = ("park", "ready", "pre_press", "contact", "press", "retract")
STATE_CONFIG_NAMES = (
    "monitor_disarmed", "park", "ready", "pre_press", "visual_5mm_gate",
    "contact", "press", "retract",
)
FORBIDDEN_TEXT = ("vendor_home", "vendor home")


class ConfigError(ValueError):
    pass


class SequencerError(RuntimeError):
    pass


class PressMode(str, Enum):
    NO_CONTACT = "NO_CONTACT"
    VISUAL_5MM = "VISUAL_5MM"
    CONTACT_FIND = "CONTACT_FIND"
    PRESS = "PRESS"
    RETRACT = "RETRACT"


class HighLevelState(str, Enum):
    MONITOR_DISARMED = "MONITOR_DISARMED"
    PARKED = "PARKED"
    READY = "READY"
    PRE_PRESS = "PRE_PRESS"
    VISUAL_5MM_GATE = "VISUAL_5MM_GATE"
    CONTACT = "CONTACT"
    PRESSED = "PRESSED"
    RETRACTED = "RETRACTED"
    ABORT_LATCHED_DISARMED = "ABORT_LATCHED_DISARMED"


class ControlState(str, Enum):
    STABLE_DISARMED = "STABLE_DISARMED"
    AWAIT_HUMAN_CONFIRM = "AWAIT_HUMAN_CONFIRM"
    WAIT_DRIVER = "WAIT_DRIVER"
    HUMAN_INSPECTION = "HUMAN_INSPECTION"
    ABORT_LATCHED_DISARMED = "ABORT_LATCHED_DISARMED"


POSE_TO_STATE = {
    "park": HighLevelState.PARKED,
    "ready": HighLevelState.READY,
    "pre_press": HighLevelState.PRE_PRESS,
    "contact": HighLevelState.CONTACT,
    "press": HighLevelState.PRESSED,
    "retract": HighLevelState.RETRACTED,
}

ALLOWED_EDGES = {
    (HighLevelState.MONITOR_DISARMED, HighLevelState.PARKED),
    (HighLevelState.PARKED, HighLevelState.READY),
    (HighLevelState.READY, HighLevelState.PRE_PRESS),
    (HighLevelState.PRE_PRESS, HighLevelState.VISUAL_5MM_GATE),
    (HighLevelState.VISUAL_5MM_GATE, HighLevelState.CONTACT),
    (HighLevelState.CONTACT, HighLevelState.PRESSED),
    (HighLevelState.PRESSED, HighLevelState.RETRACTED),
    (HighLevelState.RETRACTED, HighLevelState.PRE_PRESS),
    (HighLevelState.PRE_PRESS, HighLevelState.READY),
    (HighLevelState.READY, HighLevelState.PARKED),
}

MODE_EDGES = {
    PressMode.NO_CONTACT: {
        (HighLevelState.MONITOR_DISARMED, HighLevelState.PARKED),
        (HighLevelState.PARKED, HighLevelState.READY),
        (HighLevelState.READY, HighLevelState.PRE_PRESS),
        (HighLevelState.PRE_PRESS, HighLevelState.READY),
        (HighLevelState.READY, HighLevelState.PARKED),
    },
    PressMode.VISUAL_5MM: {
        (HighLevelState.PRE_PRESS, HighLevelState.VISUAL_5MM_GATE),
    },
    PressMode.CONTACT_FIND: {
        (HighLevelState.VISUAL_5MM_GATE, HighLevelState.CONTACT),
    },
    PressMode.PRESS: {
        (HighLevelState.CONTACT, HighLevelState.PRESSED),
    },
    PressMode.RETRACT: {
        (HighLevelState.PRESSED, HighLevelState.RETRACTED),
        (HighLevelState.RETRACTED, HighLevelState.PRE_PRESS),
        (HighLevelState.PRE_PRESS, HighLevelState.READY),
        (HighLevelState.READY, HighLevelState.PARKED),
    },
}


@dataclass(frozen=True)
class Pose:
    name: str
    positions: tuple[float, ...]
    fixture_fingerprint: str
    operator_confirmation_id: str


@dataclass(frozen=True)
class Step:
    joint: str
    target_rad: float
    expected_contact_delta_mm: float
    max_joint_error_deg: float


@dataclass(frozen=True)
class Segment:
    from_pose: str
    to_pose: str
    steps: tuple[Step, ...]
    require_human_confirm: bool


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    artifact_id: str
    fixture_fingerprint: str
    encoder_step_rad: float
    validated_joints: frozenset[str]
    poses: Mapping[str, Pose]
    segments: Mapping[tuple[str, str], Segment]
    press_qualified: bool
    required_successes: int


@dataclass(frozen=True)
class Confirmation:
    token_id: str
    from_state: HighLevelState
    to_state: HighLevelState
    step_index: int
    joint: str
    target_rad: float
    measured_digest: str
    expires_monotonic: float
    text: str


@dataclass(frozen=True)
class JointTarget:
    names: tuple[str, ...]
    positions: tuple[float, ...]
    changed_joint: str
    target_rad: float
    transition: tuple[str, str]
    step_index: int
    confirmation_id: str


@dataclass
class PendingTransition:
    source: HighLevelState
    target: HighLevelState
    segment: Segment
    next_step: int = 0
    command: Optional[JointTarget] = None
    deadline_monotonic: Optional[float] = None


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigError(f"{where}.{key} is required and cannot be null")
    return mapping[key]


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{where} must be a finite number")
    return result


def _finite_vector(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ConfigError(f"{where} must contain exactly {length} values")
    return tuple(_finite(item, f"{where}[{index}]") for index, item in enumerate(value))


def _no_nulls(value: Any, where: str) -> None:
    if value is None:
        raise ConfigError(f"{where} cannot be null")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _no_nulls(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_nulls(child, f"{where}[{index}]")


def load_artifact(path: str | Path) -> Artifact:
    """Load a complete taught artifact. Any placeholder/null fails closed."""
    source_path = Path(path).expanduser().resolve()
    raw_bytes = source_path.read_bytes()
    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    root = _mapping(data, "artifact")
    lowered = raw_bytes.decode("utf-8", errors="replace").lower().replace(" ", "")
    if any(term.replace(" ", "") in lowered for term in FORBIDDEN_TEXT):
        raise ConfigError("artifact contains prohibited vendor-home or T:102 text")
    if _required(root, "schema_version", "artifact") != 1:
        raise ConfigError("artifact.schema_version must be 1")
    artifact_id = str(_required(root, "artifact_id", "artifact")).strip()
    if not artifact_id:
        raise ConfigError("artifact.artifact_id cannot be empty")
    _required(root, "created_at", "artifact")

    robot = _mapping(_required(root, "robot", "artifact"), "robot")
    if robot.get("model") != "roarm_m3" or robot.get("controller") != "t101_single_joint_only":
        raise ConfigError("robot model/controller must be roarm_m3/t101_single_joint_only")
    names = tuple(_required(robot, "joint_names", "robot"))
    if names != JOINT_NAMES:
        raise ConfigError("robot.joint_names must be the exact six URDF names in order")
    encoder_step = _finite(_required(robot, "encoder_step_rad", "robot"), "robot.encoder_step_rad")
    if encoder_step <= 0:
        raise ConfigError("robot.encoder_step_rad must be positive")

    fixture = _mapping(_required(root, "fixture", "artifact"), "fixture")
    fingerprint = str(_required(fixture, "fingerprint", "fixture")).strip()
    for key in ("arm_mount_marks_mm", "panel_stop_marks_mm", "panel_plane_x_mm", "panel_yaw_deg", "max_shift_at_2n_mm"):
        _no_nulls(_required(fixture, key, "fixture"), f"fixture.{key}")

    contact = _mapping(_required(root, "contact", "artifact"), "contact")
    if contact.get("frame") != "stock_contact" or contact.get("source") != "stock_end_effector":
        raise ConfigError("contact frame/source must be stock_contact/stock_end_effector")
    for key in ("patch_size_mm", "offset_from_hand_tcp_mm", "normal_in_hand_tcp", "gripper_locked_ros_rad"):
        _no_nulls(_required(contact, key, "contact"), f"contact.{key}")
    button = _mapping(_required(root, "button", "artifact"), "button")
    for key in ("id", "center_base_mm", "visible_size_mm", "travel_mm", "actuation_force_n", "activation_signal"):
        _no_nulls(_required(button, key, "button"), f"button.{key}")

    profiles = _mapping(_required(root, "driver_profiles", "artifact"), "driver_profiles")
    validated: set[str] = set()
    profile_names = ("base", "shoulder", "elbow", "wrist", "roll", "gripper")
    if set(profiles) != set(profile_names):
        raise ConfigError("driver_profiles must declare exactly base/shoulder/elbow/wrist/roll/gripper")
    for index, profile_name in enumerate(profile_names):
        profile = _mapping(profiles[profile_name], f"driver_profiles.{profile_name}")
        if profile.get("firmware_joint") != index + 1:
            raise ConfigError(f"driver_profiles.{profile_name}.firmware_joint mismatch")
        if profile.get("qualified") is True:
            validated.add(JOINT_NAMES[index])

    pose_data = _mapping(_required(root, "poses", "artifact"), "poses")
    if set(pose_data) != set(POSE_NAMES):
        raise ConfigError("poses must contain exactly park/ready/pre_press/contact/press/retract")
    poses: dict[str, Pose] = {}
    for pose_name in POSE_NAMES:
        item = _mapping(_required(pose_data, pose_name, "poses"), f"poses.{pose_name}")
        if item.get("qualified") is not True:
            raise ConfigError(f"poses.{pose_name}.qualified must be true")
        positions = _finite_vector(_required(item, "positions_rad", f"poses.{pose_name}"), 6, f"poses.{pose_name}.positions_rad")
        measured_at = str(_required(item, "measured_at", f"poses.{pose_name}")).strip()
        confirmation_id = str(_required(item, "operator_confirmation_id", f"poses.{pose_name}")).strip()
        pose_fingerprint = str(_required(item, "fixture_fingerprint", f"poses.{pose_name}")).strip()
        if not measured_at or not confirmation_id or pose_fingerprint != fingerprint:
            raise ConfigError(f"poses.{pose_name} metadata/fingerprint is invalid")
        _finite_vector(_required(item, "sample_range_rad", f"poses.{pose_name}"), 6, f"poses.{pose_name}.sample_range_rad")
        poses[pose_name] = Pose(pose_name, positions, pose_fingerprint, confirmation_id)

    segment_list = _required(root, "segments", "artifact")
    if not isinstance(segment_list, list) or not segment_list:
        raise ConfigError("segments must be a non-empty list")
    segments: dict[tuple[str, str], Segment] = {}
    for si, value in enumerate(segment_list):
        item = _mapping(value, f"segments[{si}]")
        from_pose = str(_required(item, "from", f"segments[{si}]")).strip()
        to_pose = str(_required(item, "to", f"segments[{si}]")).strip()
        if from_pose not in STATE_CONFIG_NAMES or to_pose not in STATE_CONFIG_NAMES:
            raise ConfigError(f"segments[{si}] references an unknown state")
        raw_steps = _required(item, "steps", f"segments[{si}]")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ConfigError(f"segments[{si}].steps must be non-empty")
        steps: list[Step] = []
        for sj, raw_step in enumerate(raw_steps):
            step = _mapping(raw_step, f"segments[{si}].steps[{sj}]")
            joint = str(_required(step, "joint", f"segments[{si}].steps[{sj}]")).strip()
            if joint not in JOINT_NAMES or joint not in validated:
                raise ConfigError(f"segment joint {joint!r} is unsupported or not physically qualified")
            if joint == JOINT_NAMES[5] and from_pose not in {"monitor_disarmed", "park"}:
                raise ConfigError("gripper is locked from ready through retract")
            steps.append(Step(
                joint=joint,
                target_rad=_finite(_required(step, "target_rad", "step"), "step.target_rad"),
                expected_contact_delta_mm=_finite(_required(step, "expected_contact_delta_mm", "step"), "step.expected_contact_delta_mm"),
                max_joint_error_deg=_finite(_required(step, "max_joint_error_deg", "step"), "step.max_joint_error_deg"),
            ))
        key = (from_pose, to_pose)
        if key in segments:
            raise ConfigError(f"duplicate segment {from_pose}->{to_pose}")
        segments[key] = Segment(from_pose, to_pose, tuple(steps), item.get("require_human_confirm") is True)

    qualification = _mapping(_required(root, "qualification", "artifact"), "qualification")
    required_successes = int(_required(qualification, "required_consecutive_press_successes", "qualification"))
    if required_successes != 10:
        raise ConfigError("qualification.required_consecutive_press_successes must be 10")
    press_qualified = all(qualification.get(key) is True for key in (
        "measurements_qualified", "contact_qualified", "activation_qualified", "physical_press_enabled"
    ))
    return Artifact(
        source_path, hashlib.sha256(raw_bytes).hexdigest(), artifact_id, fingerprint,
        encoder_step, frozenset(validated), poses, segments, press_qualified, required_successes,
    )


def measured_digest(positions: Iterable[float]) -> str:
    canonical = json.dumps([round(float(value), 9) for value in positions], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLog:
    """Append-only JSONL when a path is explicitly supplied; otherwise memory-only."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.records: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "wall_time": time.time(),
            "monotonic_time": time.monotonic(),
            "event": event,
            **fields,
        }
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        return record


class TeachCapture:
    """Capture stable measured feedback; never requests torque release or motion."""

    def __init__(self, encoder_step_rad: float) -> None:
        self.encoder_step_rad = encoder_step_rad

    def capture(self, name: str, samples: Iterable[Mapping[str, float]], confirmation_id: str) -> dict[str, Any]:
        if name not in POSE_NAMES or not confirmation_id:
            raise SequencerError("valid pose name and operator confirmation are required")
        rows = list(samples)
        if len(rows) < 10:
            raise SequencerError("teach capture requires at least 10 measured samples")
        columns: list[list[float]] = []
        for joint in JOINT_NAMES:
            values = [_finite(row.get(joint), f"sample.{joint}") for row in rows]
            if max(values) - min(values) > self.encoder_step_rad:
                raise SequencerError(f"measured joint {joint} is not stable")
            columns.append(sorted(values))
        medians = [values[len(values) // 2] for values in columns]
        ranges = [max(values) - min(values) for values in columns]
        return {"name": name, "positions_rad": medians, "sample_range_rad": ranges,
                "operator_confirmation_id": confirmation_id, "source": "measured_joint_states_only"}


class Sequencer:
    def __init__(self, artifact: Artifact, *, dry_run: bool = True,
                 execution_enabled: bool = False, timeout_sec: float = 10.0,
                 audit_path: str | Path | None = None) -> None:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self.artifact = artifact
        self.dry_run = bool(dry_run)
        self.execution_enabled = bool(execution_enabled)
        self.timeout_sec = float(timeout_sec)
        self.audit = AuditLog(audit_path)
        self.state = HighLevelState.MONITOR_DISARMED
        self.control = ControlState.STABLE_DISARMED
        self.mode = PressMode.NO_CONTACT
        self.pending: Optional[PendingTransition] = None
        self.measured: Optional[tuple[float, ...]] = None
        self.consumed_tokens: set[str] = set()
        self.consecutive_press_successes = 0
        self.audit.record("session_started", artifact_id=artifact.artifact_id,
                          artifact_sha256=artifact.sha256, dry_run=self.dry_run,
                          execution_enabled=self.execution_enabled)

    def set_mode(self, mode: PressMode) -> None:
        if self.pending:
            raise SequencerError("cannot change staged mode during a transition")
        if mode is PressMode.PRESS and not self.artifact.press_qualified:
            raise SequencerError("physical PRESS disabled: measurements/contact/activation qualification incomplete")
        self.mode = mode
        self.audit.record("mode_changed", mode=mode.value)

    def update_measured(self, names: Iterable[str], positions: Iterable[float]) -> None:
        if tuple(names) != JOINT_NAMES:
            self.abort("measured feedback does not contain exact six URDF names in order")
            raise SequencerError("invalid measured feedback names")
        vector = tuple(_finite(value, "measured position") for value in positions)
        if len(vector) != 6:
            self.abort("measured feedback length is not six")
            raise SequencerError("invalid measured feedback length")
        self.measured = vector

    def request_transition(self, target: HighLevelState) -> None:
        if self.state is HighLevelState.ABORT_LATCHED_DISARMED:
            raise SequencerError("abort is latched")
        if self.pending or self.control is not ControlState.STABLE_DISARMED:
            raise SequencerError("another transition is active")
        edge = (self.state, target)
        if edge not in ALLOWED_EDGES or edge not in MODE_EDGES[self.mode]:
            raise SequencerError(f"transition {self.state.value}->{target.value} is not allowed in {self.mode.value}")
        if target is HighLevelState.PRESSED and not self.artifact.press_qualified:
            raise SequencerError("physical PRESS disabled")
        source_pose = self._state_pose(self.state)
        target_pose = self._state_pose(target)
        segment = self.artifact.segments.get((source_pose, target_pose))
        if segment is None:
            raise SequencerError(f"no explicit segment for {source_pose}->{target_pose}")
        if not segment.require_human_confirm:
            raise SequencerError("every segment must require human confirmation")
        self.pending = PendingTransition(self.state, target, segment)
        self.control = ControlState.AWAIT_HUMAN_CONFIRM
        self.audit.record("transition_requested", source=self.state.value, target=target.value)

    def expected_confirmation(self, now_monotonic: float, token_id: str, text: str,
                              ttl_sec: float = 30.0) -> Confirmation:
        pending, measured, step = self._pending_context()
        return Confirmation(token_id, pending.source, pending.target, pending.next_step,
                            step.joint, step.target_rad, measured_digest(measured),
                            now_monotonic + ttl_sec, text)

    def approve_next(self, confirmation: Confirmation, now_monotonic: float) -> Optional[JointTarget]:
        pending, measured, step = self._pending_context()
        if not confirmation.text.strip():
            raise SequencerError("human confirmation text is required")
        expected = (pending.source, pending.target, pending.next_step, step.joint, step.target_rad, measured_digest(measured))
        actual = (confirmation.from_state, confirmation.to_state, confirmation.step_index,
                  confirmation.joint, confirmation.target_rad, confirmation.measured_digest)
        if confirmation.token_id in self.consumed_tokens or actual != expected or now_monotonic > confirmation.expires_monotonic:
            raise SequencerError("confirmation is expired, reused, or bound to different state/feedback/target")
        joint_index = JOINT_NAMES.index(step.joint)
        if step.joint not in self.artifact.validated_joints:
            raise SequencerError("joint has not been physically validated")
        if abs(step.target_rad - measured[joint_index]) <= self.artifact.encoder_step_rad:
            raise SequencerError("microstep must exceed one encoder step")
        target_positions = list(measured)
        target_positions[joint_index] = step.target_rad
        changed = [index for index, (a, b) in enumerate(zip(measured, target_positions))
                   if abs(a - b) > self.artifact.encoder_step_rad]
        if changed != [joint_index]:
            raise SequencerError("microstep must change exactly one joint")
        self.consumed_tokens.add(confirmation.token_id)
        command = JointTarget(JOINT_NAMES, tuple(target_positions), step.joint,
                              step.target_rad, (pending.source.value, pending.target.value),
                              pending.next_step, confirmation.token_id)
        pending.command = command
        pending.deadline_monotonic = now_monotonic + self.timeout_sec
        self.control = ControlState.WAIT_DRIVER
        self.audit.record("microstep_approved", confirmation_id=confirmation.token_id,
                          joint=step.joint, target_rad=step.target_rad,
                          published=bool(self.execution_enabled and not self.dry_run))
        return command if self.execution_enabled and not self.dry_run else None

    def driver_status(self, status: str, now_monotonic: float) -> None:
        if self.control is not ControlState.WAIT_DRIVER or not self.pending:
            return
        normalized = status.upper()
        if "FAULT" in normalized:
            self.abort(f"safe-driver terminal fault: {status}")
            return
        if "CONVERGED_DISARMED" not in normalized:
            return
        self.pending.next_step += 1
        self.pending.command = None
        self.pending.deadline_monotonic = None
        if self.pending.next_step == len(self.pending.segment.steps):
            previous, target = self.state, self.pending.target
            self.state = target
            self.pending = None
            self.control = ControlState.HUMAN_INSPECTION
            self.audit.record("transition_converged", source=previous.value, target=target.value)
        else:
            self.control = ControlState.AWAIT_HUMAN_CONFIRM

    def complete_inspection(self, confirmation_text: str) -> None:
        if self.control is not ControlState.HUMAN_INSPECTION or not confirmation_text.strip():
            raise SequencerError("fresh human inspection confirmation is required")
        self.control = ControlState.STABLE_DISARMED
        self.audit.record("human_inspection_complete", state=self.state.value, text=confirmation_text)

    def poll(self, now_monotonic: float) -> None:
        if (self.control is ControlState.WAIT_DRIVER and self.pending and
                self.pending.deadline_monotonic is not None and now_monotonic > self.pending.deadline_monotonic):
            self.abort("safe-driver convergence timeout")

    def abort(self, reason: str) -> None:
        self.pending = None
        self.state = HighLevelState.ABORT_LATCHED_DISARMED
        self.control = ControlState.ABORT_LATCHED_DISARMED
        self.consecutive_press_successes = 0
        self.audit.record("abort_latched", reason=reason, retry_count=0, compensation_count=0)

    def record_cycle(self, success: bool, *, activation_evidence: str = "") -> int:
        if success and self.state is not HighLevelState.PARKED:
            raise SequencerError("a successful cycle must finish at project PARKED")
        if success and not activation_evidence.strip():
            raise SequencerError("objective activation evidence is required")
        self.consecutive_press_successes = self.consecutive_press_successes + 1 if success else 0
        self.audit.record("cycle_result", success=success,
                          consecutive_successes=self.consecutive_press_successes,
                          required=self.artifact.required_successes,
                          activation_evidence=activation_evidence,
                          retry_count=0, compensation_count=0)
        return self.consecutive_press_successes

    @property
    def qualified_for_repeatability(self) -> bool:
        return self.consecutive_press_successes >= self.artifact.required_successes

    def _pending_context(self) -> tuple[PendingTransition, tuple[float, ...], Step]:
        if self.control is not ControlState.AWAIT_HUMAN_CONFIRM or not self.pending:
            raise SequencerError("no microstep is awaiting confirmation")
        if self.measured is None:
            raise SequencerError("fresh measured joint state is required")
        return self.pending, self.measured, self.pending.segment.steps[self.pending.next_step]

    @staticmethod
    def _state_pose(state: HighLevelState) -> str:
        if state is HighLevelState.MONITOR_DISARMED:
            return "monitor_disarmed"
        for pose_name, pose_state in POSE_TO_STATE.items():
            if pose_state is state:
                return pose_name
        if state is HighLevelState.VISUAL_5MM_GATE:
            return "visual_5mm_gate"
        raise SequencerError(f"state {state.value} has no taught pose")
