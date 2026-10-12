"""Pure safety and protocol logic, independent of ROS and pyserial."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
import secrets
import time
from typing import Callable, Optional, Sequence

JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)

# Firmware-defined read-only pose/servo feedback request. This is deliberately
# separate from every motion payload builder and is disabled at node startup.
FEEDBACK_REQUEST = b'{"T":105}\n'
FEEDBACK_KEYS = ("b", "s", "e", "t", "r", "g")
TORQUE_KEYS = (
    "torswitchB",
    "torswitchS",
    "torswitchE",
    "torswitchT",
    "torswitchR",
    "torswitchG",
)
LOAD_KEYS = ("tB", "tS", "tE", "tT", "tR", "tG")
JOINT_LIMITS = (
    (-3.1416, 3.1416),
    (-1.5708, 1.5708),
    (-1.0, 2.95),
    (-1.5708, 1.5708),
    (-3.1416, 3.1416),
    (0.0, 1.5),
)
# Transient out-of-plausibility T:1051 frames must not sticky-latch capture.
# The reader drops this exact parse error; accept_feedback can self-heal the
# matching latched fault once measured feedback is healthy again.
IMPLAUSIBLE_JOINT_LIMITS_ERROR = "feedback violates plausible joint limits"
INVALID_FEEDBACK_JOINT_LIMITS_FAULT = (
    f"invalid feedback: {IMPLAUSIBLE_JOINT_LIMITS_ERROR}"
)
TORQUE_SWITCH_OFF_FAULT = "one or more joint torque switches are off"
RECOVERABLE_STICKY_FAULTS = frozenset(
    (
        INVALID_FEEDBACK_JOINT_LIMITS_FAULT,
        TORQUE_SWITCH_OFF_FAULT,
    )
)
# T101's firmware-side command limits are slightly stricter than the broad
# feedback plausibility limits above.  In particular, negative elbow commands
# are silently clamped by the firmware and therefore are not accepted here.
T101_COMMAND_LIMITS = (
    (-math.pi, math.pi),
    (-math.pi / 2.0, math.pi / 2.0),
    (0.0, 2.95),
    (-math.pi / 2.0, math.pi / 2.0),
    (-math.pi, math.pi),
    (0.0, 1.5),
)
T101_JOINT_IDS = (1, 2, 3, 4, 5, 6)
ENCODER_STEPS_PER_REV = 4096
ENCODER_STEP_RAD = 2.0 * math.pi / ENCODER_STEPS_PER_REV
# Terminal success uses the same measured-settle evidence policy as the
# calibration recorder: one full second, at least 20 samples, no gap over
# 100 ms, and no joint spanning more than two encoder steps.  These are hard
# fail-closed floors/ceilings rather than launch-tunable motion parameters.
TERMINAL_SETTLE_WINDOW_S = 1.0
TERMINAL_MIN_SETTLE_SAMPLES = 20
TERMINAL_MAX_FEEDBACK_GAP_S = 0.1
TERMINAL_MAX_SETTLE_SPAN_RAD = 2.0 * ENCODER_STEP_RAD
# Arming is only a brief precondition for one T101.  A host process that dies
# between `set_armed(true)` and the first command must not leave the hardware
# accepting a later stray command.  This is deliberately a hard deadline,
# rather than a launch-tunable convenience timeout.
ARM_COMMAND_DEADLINE_S = 1.0
# A stalled T102 may be reissued exactly once with the same target.  Keep a
# bounded observation interval after that write so the existing 1.5 s
# soft-settle evidence can actually be collected.  This neither changes the
# goal nor permits another command/retry.
T102_REISSUE_POST_SETTLE_DEADLINE_S = 2.0

# These are hard safety ceilings, not merely launch defaults.  A deployment
# may lower any value but cannot raise it without a code review and new tests.
DEFAULT_JOINT_SPEED_STEPS_S = (34, 17, 17, 23, 34, 17)
MAX_JOINT_SPEED_STEPS_S = DEFAULT_JOINT_SPEED_STEPS_S
# Exact precontact-only speed lease profile.  The supervised outbound-only
# T102 trial shares one firmware speed across all moving joints, and the
# generic profile's per-joint min (17 steps/s) leaves the first waypoint
# short of the 2x-tolerance soft band.  The lease forces the shared T102
# speed to the global maximum (34 steps/s) for exactly one executor action,
# then restores the captured tuple.  The request carries no speed field, so
# this constant is the only value the lease can apply.
PRECONTACT_LEASE_SPEED_STEPS_S = (34, 34, 34, 34, 34, 34)
# Exact purpose string required to open/close the lease.  Any other purpose
# fails closed; no client may request an arbitrary purpose or speed.
PRECONTACT_LEASE_PURPOSE = "T102_OUTBOUND_PRECONTACT_ONLY"
# Narrow research-prototype lease.  It is intentionally a different purpose
# from the precise outbound executor so coarse progress can never be enabled
# by an existing client accidentally.
COARSE_PRECONTACT_LEASE_PURPOSE = "T102_OUTBOUND_COARSE_PRECONTACT_ONLY"
# Exact frozen-plan Dynamic IK execution is intentionally isolated from both
# the ordinary precontact and coarse prototype purposes.  The lease owner may
# issue one absolute T102 per frozen waypoint, but the controller must never
# rewrite or soft-accept that command on the owner's behalf.
DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE = (
    "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE"
)
# Separate post-PRE, tangent-only exact-plan owner.  It intentionally shares
# the reviewed conservative speed/PID/no-reissue mechanics, but not the
# Dynamic PRE purpose identity or any contact authority.
POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE = (
    "T102_POST_PRECONTACT_XY_EXACT_PLAN_NO_REISSUE"
)
EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES = (
    DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
    POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE,
)
# Dynamic exact execution has a stricter measured waypoint gate than the
# ordinary Quick Control profile.  Keep the driver active until the same
# 0.5-degree threshold is satisfied; otherwise a 1-degree driver completion
# can disarm before the exact-plan executor is allowed to advance.
DYNAMIC_PRECONTACT_TOLERANCE_RAD = math.radians(0.5)
# The vendor documents the lowest non-zero integral setting (multiples of 8)
# specifically for compensating load-induced position error.  The default
# P=16/I=0 is retained everywhere except a separately enabled Dynamic exact
# lease, where P=16/I=8 is temporary and reset on lease release.
DYNAMIC_PRECONTACT_PID_P = 16
DYNAMIC_PRECONTACT_PID_I = 8
DYNAMIC_PRECONTACT_PID_JOINT_IDS = (1, 2, 3, 4, 5)
COARSE_MIN_USEFUL_MOTION_RAD = 0.035
COARSE_MAX_WRONG_DIRECTION_RAD = 0.0087
# The vendor's documented T102 example uses acc=10.  Keep the deployed
# default at 1; allow the documented 1..10 range only when an explicitly
# selected diagnostic profile asks for it.
DEFAULT_JOINT_ACC_REGISTERS = (1, 1, 1, 1, 1, 1)
MAX_JOINT_ACC_REGISTERS = (10, 10, 10, 10, 10, 10)
DEFAULT_JOINT_MAX_DELTA_RAD = tuple(
    math.radians(value) for value in (5.0, 2.0, 2.0, 3.0, 5.0, 2.0)
)
# A dedicated diagnostic deployment may expand base travel only. Production
# defaults remain 5 degrees and all other joint hard ceilings remain unchanged.
MAX_JOINT_DELTA_RAD = (
    math.radians(95.0),
    math.radians(10.0),
    math.radians(10.0),
    math.radians(15.0),
    math.radians(30.0),
    math.radians(15.0),
)


class SafetyError(ValueError):
    """A command or feedback frame failed a fail-closed safety check."""


@dataclass(frozen=True)
class Feedback:
    positions: tuple[float, ...]
    loads: tuple[int, ...]
    torque_enabled: tuple[bool, ...]
    voltage: float
    received_at: float
    raw: dict


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise SafetyError(f"{field} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise SafetyError(f"{field} is not finite")
    return result


def parse_feedback_line(
    line: bytes | str, now: Optional[float] = None
) -> Optional[Feedback]:
    """Parse one unsolicited newline frame; unrelated valid JSON is ignored."""
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SafetyError("feedback is not UTF-8") from exc
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SafetyError("malformed JSON") from exc
    if not isinstance(obj, dict):
        raise SafetyError("JSON frame is not an object")
    if obj.get("T") != 1051:
        return None
    firmware = tuple(_finite_float(obj.get(key), key) for key in FEEDBACK_KEYS)
    positions = firmware[:5] + (math.pi - firmware[5],)
    plausible = all(
        lower - 0.15 <= value <= upper + 0.15
        for value, (lower, upper) in zip(positions, JOINT_LIMITS)
    )
    if not plausible:
        raise SafetyError(IMPLAUSIBLE_JOINT_LIMITS_ERROR)
    loads = tuple(int(_finite_float(obj.get(key), key)) for key in LOAD_KEYS)
    torque_raw = tuple(_finite_float(obj.get(key), key) for key in TORQUE_KEYS)
    if any(value not in (0.0, 1.0) for value in torque_raw):
        raise SafetyError("invalid torque status")
    voltage = _finite_float(obj.get("v"), "v") / 100.0
    return Feedback(
        positions=positions,
        loads=loads,
        torque_enabled=tuple(bool(value) for value in torque_raw),
        voltage=voltage,
        received_at=time.monotonic() if now is None else now,
        raw=obj,
    )


def ordered_positions(
    names: Sequence[str], positions: Sequence[float]
) -> tuple[float, ...]:
    if len(names) != len(positions):
        raise SafetyError("joint name/position lengths differ")
    if len(set(names)) != len(names):
        raise SafetyError("duplicate joint name")
    if set(names) != set(JOINT_NAMES):
        raise SafetyError("command must contain exactly the six URDF joint names")
    by_name = dict(zip(names, positions))
    return tuple(_finite_float(by_name[name], name) for name in JOINT_NAMES)


def validate_command_age(stamp_s: float, now_s: float, max_age_s: float = 0.5) -> None:
    stamp = _finite_float(stamp_s, "command timestamp")
    now = _finite_float(now_s, "current timestamp")
    if stamp <= 0.0:
        raise SafetyError("command timestamp is required")
    age = now - stamp
    if age < -0.1 or age > max_age_s:
        raise SafetyError("command timestamp is not fresh")


def validate_target(
    target: Sequence[float],
    measured: Sequence[float],
    max_delta_rad: float,
) -> tuple[float, ...]:
    if len(target) != 6 or len(measured) != 6:
        raise SafetyError("expected six joints")
    checked = tuple(
        _finite_float(value, f"target[{index}]") for index, value in enumerate(target)
    )
    for index, (value, (lower, upper)) in enumerate(zip(checked, JOINT_LIMITS)):
        if not lower <= value <= upper:
            raise SafetyError(f"{JOINT_NAMES[index]} outside joint limit")
        if abs(value - measured[index]) > max_delta_rad + 1e-12:
            raise SafetyError(f"{JOINT_NAMES[index]} exceeds max delta")
    return checked


def _checked_profile(
    values: Sequence[object],
    field: str,
    maxima: Sequence[float],
    *,
    integer: bool,
) -> tuple:
    if len(values) != len(JOINT_NAMES):
        raise SafetyError(f"{field} must contain exactly six values")
    checked = []
    for index, (value, maximum) in enumerate(zip(values, maxima)):
        if integer:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SafetyError(f"{field}[{index}] must be an integer")
            result = value
        else:
            result = _finite_float(value, f"{field}[{index}]")
        if result <= 0 or result > maximum + 1e-12:
            raise SafetyError(
                f"{field}[{index}] must be greater than zero and no more than {maximum}"
            )
        checked.append(result)
    return tuple(checked)


def validate_joint_profiles(
    speed_steps_s: Sequence[object],
    acc_registers: Sequence[object],
    max_delta_rad: Sequence[object],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    speeds = _checked_profile(
        speed_steps_s,
        "joint_speed_steps_s",
        MAX_JOINT_SPEED_STEPS_S,
        integer=True,
    )
    accelerations = _checked_profile(
        acc_registers,
        "joint_acc_registers",
        MAX_JOINT_ACC_REGISTERS,
        integer=True,
    )
    deltas = _checked_profile(
        max_delta_rad,
        "joint_max_delta_rad",
        MAX_JOINT_DELTA_RAD,
        integer=False,
    )
    return speeds, accelerations, deltas


def validate_single_joint_target(
    target: Sequence[float],
    measured: Sequence[float],
    max_delta_rad: Sequence[float],
    match_tolerance_rad: float = ENCODER_STEP_RAD,
) -> tuple[tuple[float, ...], int]:
    """Require exactly one logical joint to differ meaningfully from feedback."""
    if len(target) != len(JOINT_NAMES) or len(measured) != len(JOINT_NAMES):
        raise SafetyError("expected six joints")
    if len(max_delta_rad) != len(JOINT_NAMES):
        raise SafetyError("joint_max_delta_rad must contain exactly six values")
    checked = tuple(
        _finite_float(value, f"target[{index}]") for index, value in enumerate(target)
    )
    current = tuple(
        _finite_float(value, f"measured[{index}]")
        for index, value in enumerate(measured)
    )
    tolerance = _finite_float(match_tolerance_rad, "match tolerance")
    if tolerance < 0.0:
        raise SafetyError("match tolerance cannot be negative")
    deltas = tuple(abs(value - present) for value, present in zip(checked, current))
    meaningful = [
        index for index, delta in enumerate(deltas) if delta > tolerance + 1e-12
    ]
    if len(meaningful) != 1:
        raise SafetyError("command must change exactly one logical joint")
    selected = meaningful[0]
    lower, upper = T101_COMMAND_LIMITS[selected]
    if not lower <= checked[selected] <= upper:
        raise SafetyError(f"{JOINT_NAMES[selected]} outside T101 command limit")
    allowed_delta = _finite_float(max_delta_rad[selected], f"max delta[{selected}]")
    if deltas[selected] > allowed_delta + 1e-12:
        raise SafetyError(f"{JOINT_NAMES[selected]} exceeds max delta")
    return checked, selected


def validate_multi_joint_target(
    target: Sequence[float],
    measured: Sequence[float],
    max_delta_rad: Sequence[float],
    match_tolerance_rad: float = ENCODER_STEP_RAD,
) -> tuple[tuple[float, ...], list[int]]:
    """Require at least one joint to differ meaningfully from feedback.

    Unlike the single-joint variant, this accepts multiple joints changing
    simultaneously.  Every moving joint is checked against T101 limits and
    its per-joint max delta.  Returns (checked_positions, moving_indices).
    """
    if len(target) != len(JOINT_NAMES) or len(measured) != len(JOINT_NAMES):
        raise SafetyError("expected six joints")
    if len(max_delta_rad) != len(JOINT_NAMES):
        raise SafetyError("joint_max_delta_rad must contain exactly six values")
    checked = tuple(
        _finite_float(value, f"target[{index}]") for index, value in enumerate(target)
    )
    current = tuple(
        _finite_float(value, f"measured[{index}]")
        for index, value in enumerate(measured)
    )
    tolerance = _finite_float(match_tolerance_rad, "match tolerance")
    if tolerance < 0.0:
        raise SafetyError("match tolerance cannot be negative")
    deltas = tuple(abs(value - present) for value, present in zip(checked, current))
    moving = [index for index, delta in enumerate(deltas) if delta > tolerance + 1e-12]
    if not moving:
        raise SafetyError("command must change at least one logical joint")
    for index in moving:
        lower, upper = T101_COMMAND_LIMITS[index]
        if not lower <= checked[index] <= upper:
            raise SafetyError(f"{JOINT_NAMES[index]} outside T101 command limit")
        allowed_delta = _finite_float(max_delta_rad[index], f"max delta[{index}]")
        if deltas[index] > allowed_delta + 1e-12:
            raise SafetyError(f"{JOINT_NAMES[index]} exceeds max delta")
    return checked, moving


def count_moving_joints(
    target: Sequence[float],
    measured: Sequence[float],
    match_tolerance_rad: float = ENCODER_STEP_RAD,
) -> int:
    """Count joints whose target differs meaningfully from feedback.

    Used by the ROS node to dispatch between the strict single-joint T101
    path (exactly one moving joint, non-target-drift convergence) and the
    multi-joint T102 path (two or more moving joints).
    """
    if len(target) != len(JOINT_NAMES) or len(measured) != len(JOINT_NAMES):
        raise SafetyError("expected six joints")
    tolerance = _finite_float(match_tolerance_rad, "match tolerance")
    if tolerance < 0.0:
        raise SafetyError("match tolerance cannot be negative")
    moving = 0
    for index, (value, present) in enumerate(zip(target, measured)):
        checked = _finite_float(value, f"target[{index}]")
        current = _finite_float(present, f"measured[{index}]")
        if abs(checked - current) > tolerance + 1e-12:
            moving += 1
    return moving


def build_t102(
    target_positions: Sequence[float],
    speed_steps_s: int,
    acc_register: int,
) -> bytes:
    """Build a T:102 multi-joint absolute-position command.

    Firmware expects named keys: base, shoulder, elbow, wrist, roll, hand.
    Speed and acceleration are shared across all six joints (firmware
    limitation).  Gripper uses firmware convention: firmware_rad = pi - ros_rad.
    """
    if len(target_positions) != len(JOINT_NAMES):
        raise SafetyError("T102 target must contain exactly six positions")
    checked = tuple(
        _finite_float(value, f"T102 target[{index}]")
        for index, value in enumerate(target_positions)
    )
    for index, (value, (lower, upper)) in enumerate(zip(checked, T101_COMMAND_LIMITS)):
        if not lower <= value <= upper:
            raise SafetyError(f"{JOINT_NAMES[index]} outside T101 command limit")
    # T102 uses a single shared speed/acc for all joints.  Validate against
    # the global max to ensure the value is reasonable, but don't require
    # it to satisfy every per-joint ceiling (the firmware handles joint-level
    # speed limiting internally).
    if isinstance(speed_steps_s, bool) or not isinstance(speed_steps_s, int):
        raise SafetyError("T102 speed must be an integer")
    if not 1 <= speed_steps_s <= max(MAX_JOINT_SPEED_STEPS_S):
        raise SafetyError(
            f"T102 speed must be 1–{max(MAX_JOINT_SPEED_STEPS_S)} steps/s"
        )
    if isinstance(acc_register, bool) or not isinstance(acc_register, int):
        raise SafetyError("T102 acceleration register must be an integer")
    if not 1 <= acc_register <= max(MAX_JOINT_ACC_REGISTERS):
        raise SafetyError(
            f"T102 acceleration register must be 1–{max(MAX_JOINT_ACC_REGISTERS)}"
        )
    firmware_gripper = math.pi - checked[5]
    payload = {
        "T": 102,
        "base": checked[0],
        "shoulder": checked[1],
        "elbow": checked[2],
        "wrist": checked[3],
        "roll": checked[4],
        "hand": firmware_gripper,
        "spd": int(speed_steps_s),
        "acc": int(acc_register),
    }
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "ascii"
    )


def build_joint_pid(joint_id: int, *, p: int, i: int) -> bytes:
    """Build the vendor T:108 per-logical-joint PID command.

    Dynamic precontact uses only the reviewed P=16/I=8 profile.  Keeping this
    builder exact prevents a launch file or caller from turning the temporary
    load-error correction into an arbitrary servo-tuning interface.
    """
    if isinstance(joint_id, bool) or not isinstance(joint_id, int):
        raise SafetyError("PID joint id must be an integer")
    if joint_id not in DYNAMIC_PRECONTACT_PID_JOINT_IDS:
        raise SafetyError("PID joint id is outside Dynamic task joints")
    if p != DYNAMIC_PRECONTACT_PID_P or i != DYNAMIC_PRECONTACT_PID_I:
        raise SafetyError("PID profile must be the reviewed Dynamic P16 I8 values")
    payload = {"T": 108, "joint": joint_id, "p": p, "i": i}
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "ascii"
    )


def build_reset_joint_pid() -> bytes:
    """Reset every logical joint to the firmware default P=16/I=0 profile."""
    return b'{"T":109}\n'


def build_t101(
    joint_index: int,
    target_rad: float,
    speed_steps_s: int,
    acc_register: int,
) -> bytes:
    if isinstance(joint_index, bool) or not isinstance(joint_index, int):
        raise SafetyError("joint index must be an integer")
    if not 0 <= joint_index < len(JOINT_NAMES):
        raise SafetyError("joint index is outside 0..5")
    target = _finite_float(target_rad, f"{JOINT_NAMES[joint_index]} target")
    lower, upper = T101_COMMAND_LIMITS[joint_index]
    if not lower <= target <= upper:
        raise SafetyError(f"{JOINT_NAMES[joint_index]} outside T101 command limit")
    _checked_profile(
        (speed_steps_s,) * len(JOINT_NAMES),
        "speed_steps_s",
        (MAX_JOINT_SPEED_STEPS_S[joint_index],) * len(JOINT_NAMES),
        integer=True,
    )
    _checked_profile(
        (acc_register,) * len(JOINT_NAMES),
        "acc_register",
        (MAX_JOINT_ACC_REGISTERS[joint_index],) * len(JOINT_NAMES),
        integer=True,
    )
    # T1051 feedback is exposed in ROS convention as pi-firmware_g, so the
    # inverse transform is required only for the gripper T101 command.
    firmware_target = math.pi - target if joint_index == 5 else target
    payload = {
        "T": 101,
        "joint": T101_JOINT_IDS[joint_index],
        "rad": firmware_target,
        "spd": int(speed_steps_s),
        "acc": int(acc_register),
    }
    return (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "ascii"
    )


def _round_half_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def firmware_base_goal_step(target_rad: float) -> int:
    """Mirror the firmware's 2047-centred base target conversion."""
    target = _finite_float(target_rad, "base target")
    offset = _round_half_away_from_zero(target / ENCODER_STEP_RAD)
    return 2047 - offset


def feedback_base_step(base_rad: float) -> int:
    """Recover the 2048-centred present-position step from feedback radians."""
    value = _finite_float(base_rad, "base feedback")
    return _round_half_away_from_zero(2048.0 - value / ENCODER_STEP_RAD)


def expected_motion_time_s(
    delta_rad: float, speed_steps_s: int, acc_register: int
) -> float:
    """Trapezoidal/triangular estimate; one acceleration unit is 100 steps/s²."""
    distance_steps = abs(_finite_float(delta_rad, "delta")) / ENCODER_STEP_RAD
    if isinstance(speed_steps_s, bool) or not isinstance(speed_steps_s, int):
        raise SafetyError("speed must be an integer")
    if speed_steps_s <= 0 or speed_steps_s > max(MAX_JOINT_SPEED_STEPS_S):
        raise SafetyError("speed must be between 1 and 34 steps/s")
    if isinstance(acc_register, bool) or not isinstance(acc_register, int):
        raise SafetyError("acceleration register must be an integer")
    # T101 deployments stay on acc=1; T102 diagnostic profiles may use the
    # vendor-documented 1..10 range.  The time model scales linearly with the
    # register (100 steps/s² per unit).
    if not 1 <= acc_register <= max(MAX_JOINT_ACC_REGISTERS):
        raise SafetyError(
            f"acceleration register must be 1–{max(MAX_JOINT_ACC_REGISTERS)}"
        )
    if distance_steps == 0.0:
        return 0.0
    speed = float(speed_steps_s)
    acceleration = float(acc_register * 100)
    accel_and_decel_distance = speed * speed / acceleration
    if distance_steps >= accel_and_decel_distance:
        return distance_steps / speed + speed / acceleration
    return 2.0 * math.sqrt(distance_steps / acceleration)


class SafeController:
    """State machine for arming and one-shot, single-logical-joint T101 commands."""

    def __init__(
        self,
        write: Callable[[bytes], None],
        *,
        dry_run: bool = True,
        joint_speed_steps_s: Sequence[int] = DEFAULT_JOINT_SPEED_STEPS_S,
        joint_acc_registers: Sequence[int] = DEFAULT_JOINT_ACC_REGISTERS,
        joint_max_delta_rad: Sequence[float] = DEFAULT_JOINT_MAX_DELTA_RAD,
        feedback_timeout_s: float = 0.5,
        command_timeout_s: float = 5.0,
        tolerance_rad: float = math.radians(0.5),
        convergence_samples: int = TERMINAL_MIN_SETTLE_SAMPLES,
        min_voltage: float = 11.0,
        max_voltage: float = 13.0,
    ) -> None:
        speeds, accelerations, deltas = validate_joint_profiles(
            joint_speed_steps_s,
            joint_acc_registers,
            joint_max_delta_rad,
        )
        if (
            isinstance(convergence_samples, bool)
            or not isinstance(convergence_samples, int)
            or convergence_samples < TERMINAL_MIN_SETTLE_SAMPLES
        ):
            raise SafetyError(
                "convergence_samples must be an integer no less than "
                f"{TERMINAL_MIN_SETTLE_SAMPLES}"
            )
        self._write = write
        self.dry_run = dry_run
        self.joint_speed_steps_s = speeds
        self.joint_acc_registers = accelerations
        self.joint_max_delta_rad = deltas
        self.feedback_timeout_s = feedback_timeout_s
        self.command_timeout_s = command_timeout_s
        self.tolerance_rad = tolerance_rad
        self.convergence_samples = convergence_samples
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.armed = False
        self.fault: Optional[str] = None
        self.feedback: Optional[Feedback] = None
        self.target: Optional[tuple[float, ...]] = None
        self.active_joint_index: Optional[int] = None
        self.pre_command_positions: Optional[tuple[float, ...]] = None
        self.arm_command_deadline_at: Optional[float] = None
        self.command_started_at: Optional[float] = None
        self.command_deadline_at: Optional[float] = None
        self.in_tolerance_count = 0
        self._convergence_candidate: list[Feedback] = []
        self._convergence_candidate_result = "CONVERGED_DISARMED"
        self._multi_joint_moving: frozenset[int] = frozenset()
        self.coarse_progress_endpoint: Optional[tuple[float, ...]] = None
        self.coarse_progress_active = False
        self.dynamic_exact_precontact_active = False
        self.exact_plan_no_reissue_active = False
        self.exact_plan_lease_purpose = ""
        self.dynamic_exact_accepted_command_sequence = 0
        self.dynamic_exact_accepted_target_positions: tuple[float, ...] = ()
        self.last_result = "DISARMED"
        self.telemetry: dict = {}

    def reset_dynamic_exact_command_acceptance(self) -> None:
        """Bind command acknowledgement state to one Dynamic lease lifetime."""
        self.dynamic_exact_accepted_command_sequence = 0
        self.dynamic_exact_accepted_target_positions = ()

    def dynamic_exact_advance_ready(self, now: float) -> bool:
        """Return the controller-authoritative smooth handoff predicate."""
        if (
            not self.exact_plan_no_reissue_active
            or not self.armed
            or self.target is None
            or self.feedback is None
            or not self.healthy(now)
        ):
            return False
        return max(
            abs(measured - expected)
            for measured, expected in zip(self.feedback.positions, self.target)
        ) <= DYNAMIC_PRECONTACT_TOLERANCE_RAD

    def accept_feedback(self, feedback: Feedback) -> None:
        self.feedback = feedback
        if not self.min_voltage <= feedback.voltage <= self.max_voltage:
            self.latch_fault(f"voltage {feedback.voltage:.2f} V outside safe range")
            return
        if not all(feedback.torque_enabled):
            self.latch_fault(TORQUE_SWITCH_OFF_FAULT)
            return
        # Heal prior sticky latches once measured feedback is healthy again.
        # Covers transient implausible-joint frames and a recovered torque-switch
        # latch after all six direct flags are true.  Never clears other faults.
        if (
            self.fault in RECOVERABLE_STICKY_FAULTS
            and not self.armed
            and self.target is None
        ):
            self.fault = None
            self.last_result = "FAULT_CLEARED_DISARMED"
        if self.target is None or self.pre_command_positions is None:
            return

        desired_result = "CONVERGED_DISARMED"
        if self._multi_joint_moving:
            # Multi-joint T102 convergence: all moving joints must be within
            # tolerance; non-target drift check is meaningless because every
            # moving joint is a target.
            if self.active_joint_index is not None:
                self.latch_fault("multi-joint command has a single selected joint")
                return
            joint_errors = {
                JOINT_NAMES[idx]: feedback.positions[idx] - self.target[idx]
                for idx in self._multi_joint_moving
            }
            max_error = max(abs(e) for e in joint_errors.values())
            samples = self.telemetry.setdefault("feedback_samples", [])
            samples.append(
                {
                    "received_at": feedback.received_at,
                    "positions": list(feedback.positions),
                    "mode": "multi_joint",
                    "joint_errors_rad": joint_errors,
                    "max_joint_error_rad": max_error,
                }
            )
            if len(samples) > 256:
                del samples[:-256]
            self.telemetry["max_joint_error_rad"] = max_error
            if self.coarse_progress_active:
                if self.coarse_progress_endpoint is None:
                    self.latch_fault("coarse progress endpoint is not configured")
                    return
                signed_progress = {}
                for idx in self._multi_joint_moving:
                    command_delta = self.target[idx] - self.pre_command_positions[idx]
                    direction = 1.0 if command_delta > 0.0 else -1.0
                    signed_progress[JOINT_NAMES[idx]] = (
                        feedback.positions[idx] - self.pre_command_positions[idx]
                    ) * direction
                max_useful = max(signed_progress.values(), default=0.0)
                max_wrong = max((-value for value in signed_progress.values()), default=0.0)
                initial_endpoint_distance = math.sqrt(
                    sum(
                        (endpoint - before) ** 2
                        for endpoint, before in zip(
                            self.coarse_progress_endpoint, self.pre_command_positions
                        )
                    )
                )
                endpoint_distance = math.sqrt(
                    sum(
                        (endpoint - current) ** 2
                        for endpoint, current in zip(
                            self.coarse_progress_endpoint, feedback.positions
                        )
                    )
                )
                self.telemetry.update(
                    {
                        "coarse_progress_active": True,
                        "coarse_signed_progress_rad": signed_progress,
                        "coarse_max_useful_motion_rad": max_useful,
                        "coarse_max_wrong_direction_rad": max_wrong,
                        "coarse_initial_endpoint_distance_rad": initial_endpoint_distance,
                        "coarse_endpoint_distance_rad": endpoint_distance,
                        "coarse_endpoint_distance_reduction_rad": (
                            initial_endpoint_distance - endpoint_distance
                        ),
                    }
                )
                if max_wrong > COARSE_MAX_WRONG_DIRECTION_RAD + 1e-12:
                    self.latch_fault("coarse progress moved in the wrong direction")
                    return
                exact = max_error <= self.tolerance_rad
                useful_partial = (
                    max_useful >= COARSE_MIN_USEFUL_MOTION_RAD - 1e-12
                    and endpoint_distance < initial_endpoint_distance - 1e-12
                )
                if exact:
                    desired_result = "CONVERGED_DISARMED"
                elif useful_partial:
                    desired_result = "COARSE_PROGRESS_DISARMED"
                else:
                    self._cancel_convergence_candidate("coarse_progress_not_yet_useful")
                    return
            elif max_error > self.tolerance_rad:
                self._cancel_convergence_candidate("multi_joint_target_left_tolerance")
                return
        elif self.active_joint_index is not None:
            # Single-joint T101 convergence: existing behaviour.
            selected = self.active_joint_index
            joint_error = feedback.positions[selected] - self.target[selected]
            non_target_drift_by_name = {
                JOINT_NAMES[index]: abs(
                    feedback.positions[index] - self.pre_command_positions[index]
                )
                for index in range(len(JOINT_NAMES))
                if index != selected
            }
            non_target_drift = max(non_target_drift_by_name.values(), default=0.0)
            samples = self.telemetry.setdefault("feedback_samples", [])
            samples.append(
                {
                    "received_at": feedback.received_at,
                    "positions": list(feedback.positions),
                    "joint_error_rad": joint_error,
                    "non_target_drift_rad": non_target_drift_by_name,
                    "max_non_target_drift_rad": non_target_drift,
                }
            )
            if len(samples) > 256:
                del samples[:-256]
            self.telemetry["latest_joint_error_rad"] = joint_error
            self.telemetry["max_non_target_drift_rad"] = max(
                self.telemetry.get("max_non_target_drift_rad", 0.0),
                non_target_drift,
            )
            if non_target_drift > self.tolerance_rad:
                self.latch_fault("non-target joint drift exceeded tolerance")
                return
            if abs(joint_error) > self.tolerance_rad:
                self._cancel_convergence_candidate("target_left_tolerance")
                return
        else:
            self.latch_fault("active command lost its selected joint(s)")
            return

        if not self._convergence_candidate:
            self._start_convergence_candidate(feedback, desired_result)
        elif self._convergence_candidate_result != desired_result:
            self._restart_convergence_candidate(
                feedback, "terminal_classification_changed", desired_result
            )
        else:
            gap_s = feedback.received_at - self._convergence_candidate[-1].received_at
            if gap_s <= 0.0 or gap_s > TERMINAL_MAX_FEEDBACK_GAP_S + 1e-12:
                self._restart_convergence_candidate(
                    feedback, "feedback_gap", desired_result
                )
            else:
                self._convergence_candidate.append(feedback)
                spans = self._candidate_spans()
                if max(spans, default=0.0) > TERMINAL_MAX_SETTLE_SPAN_RAD + 1e-12:
                    self._restart_convergence_candidate(
                        feedback, "joint_not_stable", desired_result
                    )

        self._update_convergence_telemetry()
        candidate_duration_s = (
            self._convergence_candidate[-1].received_at
            - self._convergence_candidate[0].received_at
        )
        if (
            len(self._convergence_candidate) >= self.convergence_samples
            and candidate_duration_s >= TERMINAL_SETTLE_WINDOW_S - 1e-12
        ):
            self._finish(self._convergence_candidate_result)

    def _candidate_spans(self) -> tuple[float, ...]:
        return tuple(
            max(sample.positions[index] for sample in self._convergence_candidate)
            - min(sample.positions[index] for sample in self._convergence_candidate)
            for index in range(len(JOINT_NAMES))
        )

    def _start_convergence_candidate(
        self, feedback: Feedback, result: str = "CONVERGED_DISARMED"
    ) -> None:
        self._convergence_candidate = [feedback]
        self._convergence_candidate_result = result
        self.in_tolerance_count = 1

    def _cancel_convergence_candidate(self, reason: str) -> None:
        if self._convergence_candidate:
            self.telemetry["convergence_candidate_reset_count"] = (
                self.telemetry.get("convergence_candidate_reset_count", 0) + 1
            )
        self._convergence_candidate = []
        self._convergence_candidate_result = "CONVERGED_DISARMED"
        self.in_tolerance_count = 0
        self.telemetry["convergence_candidate_last_reset_reason"] = reason
        self._update_convergence_telemetry()

    def _restart_convergence_candidate(
        self,
        feedback: Feedback,
        reason: str,
        result: str = "CONVERGED_DISARMED",
    ) -> None:
        self._cancel_convergence_candidate(reason)
        self._start_convergence_candidate(feedback, result)

    def _update_convergence_telemetry(self) -> None:
        self.in_tolerance_count = len(self._convergence_candidate)
        if not self._convergence_candidate:
            self.telemetry["convergence_candidate_started_at"] = None
            self.telemetry["convergence_candidate_sample_count"] = 0
            self.telemetry["convergence_candidate_duration_s"] = 0.0
            self.telemetry["convergence_candidate_span_rad"] = [0.0] * len(JOINT_NAMES)
            return
        self.telemetry["convergence_candidate_started_at"] = (
            self._convergence_candidate[0].received_at
        )
        self.telemetry["convergence_candidate_sample_count"] = len(
            self._convergence_candidate
        )
        self.telemetry["convergence_candidate_duration_s"] = (
            self._convergence_candidate[-1].received_at
            - self._convergence_candidate[0].received_at
        )
        self.telemetry["convergence_candidate_span_rad"] = list(self._candidate_spans())

    def healthy(self, now: float) -> bool:
        return (
            self.fault is None
            and self.feedback is not None
            and now - self.feedback.received_at <= self.feedback_timeout_s
            and self.min_voltage <= self.feedback.voltage <= self.max_voltage
            and all(self.feedback.torque_enabled)
        )

    def arm(self, now: float) -> None:
        if self.armed:
            raise SafetyError("driver is already armed")
        if self.target is not None:
            raise SafetyError("command already active")
        if not self.healthy(now):
            raise SafetyError(
                self.fault
                or "fresh healthy feedback with all torque enabled is required"
            )
        self.armed = True
        self.arm_command_deadline_at = now + ARM_COMMAND_DEADLINE_S
        self.last_result = "ARMED"

    def _write_joint_hold_if_possible(self, now: Optional[float]) -> Optional[bytes]:
        """Replace an active T101/T102 goal with one fresh measured T102 hold."""
        if now is None or self.target is None or not self.healthy(now):
            return None
        try:
            payload = build_t102(
                self.feedback.positions,
                min(self.joint_speed_steps_s),
                min(self.joint_acc_registers),
            )
        except SafetyError:
            return None
        if not self.dry_run:
            try:
                self._write(payload)
            except Exception:  # noqa: BLE001 - disarm must still clear authority
                return None
        return payload

    def disarm(self, reason: str = "operator", now: Optional[float] = None) -> None:
        hold_payload = self._write_joint_hold_if_possible(now)
        self.armed = False
        self.target = None
        self.active_joint_index = None
        self.pre_command_positions = None
        self.arm_command_deadline_at = None
        self.command_started_at = None
        self.command_deadline_at = None
        self.in_tolerance_count = 0
        self._multi_joint_moving = frozenset()
        self._convergence_candidate = []
        self._convergence_candidate_result = "CONVERGED_DISARMED"
        self.last_result = f"DISARMED:{reason}"
        if self.telemetry:
            self.telemetry["terminal_reason"] = self.last_result
            self.telemetry["stop_hold_outgoing"] = (
                None if hold_payload is None else hold_payload.decode("ascii").strip()
            )

    def command(self, target: Sequence[float], now: float) -> bytes:
        if not self.armed:
            raise SafetyError("driver is disarmed")
        if self.target is not None:
            raise SafetyError("one command is already active")
        if self.arm_command_deadline_at is None:
            self.latch_fault("armed state missing arm-command deadline")
            raise SafetyError("armed state missing arm-command deadline")
        if now >= self.arm_command_deadline_at:
            self._finish("ARM_COMMAND_TIMEOUT_DISARMED")
            raise SafetyError("arm command deadline expired")
        if not self.healthy(now):
            raise SafetyError(self.fault or "feedback is unhealthy or stale")
        checked, selected = validate_single_joint_target(
            target,
            self.feedback.positions,
            self.joint_max_delta_rad,
        )
        speed = self.joint_speed_steps_s[selected]
        acceleration = self.joint_acc_registers[selected]
        payload = build_t101(selected, checked[selected], speed, acceleration)
        self.target = checked
        self.active_joint_index = selected
        self.pre_command_positions = self.feedback.positions
        self.arm_command_deadline_at = None
        self.command_started_at = now
        expected_time = expected_motion_time_s(
            checked[selected] - self.feedback.positions[selected],
            speed,
            acceleration,
        )
        self.command_deadline_at = now + max(
            self.command_timeout_s, expected_time + 2.0
        )
        self.in_tolerance_count = 0
        self._convergence_candidate = []
        self._convergence_candidate_result = "CONVERGED_DISARMED"
        self.last_result = "DRY_RUN_ACTIVE" if self.dry_run else "COMMAND_ACTIVE"
        self.telemetry = {
            "command_started_at": now,
            "command_deadline_at": self.command_deadline_at,
            "expected_motion_time_s": expected_time,
            "selected_joint_index": selected,
            "selected_joint_id": T101_JOINT_IDS[selected],
            "selected_joint_name": JOINT_NAMES[selected],
            "selected_speed_steps_s": speed,
            "selected_acc_register": acceleration,
            "selected_max_delta_rad": self.joint_max_delta_rad[selected],
            "pre_command_positions": list(self.pre_command_positions),
            "target_positions": list(checked),
            "outgoing": payload.decode("ascii").strip(),
            "feedback_samples": [],
            "latest_joint_error_rad": checked[selected]
            - self.feedback.positions[selected],
            "max_non_target_drift_rad": 0.0,
            "terminal_settle_policy": {
                "window_s": TERMINAL_SETTLE_WINDOW_S,
                "min_samples": self.convergence_samples,
                "max_feedback_gap_s": TERMINAL_MAX_FEEDBACK_GAP_S,
                "max_joint_span_rad": TERMINAL_MAX_SETTLE_SPAN_RAD,
            },
            "convergence_candidate_started_at": None,
            "convergence_candidate_sample_count": 0,
            "convergence_candidate_duration_s": 0.0,
            "convergence_candidate_span_rad": [0.0] * len(JOINT_NAMES),
            "convergence_candidate_reset_count": 0,
            "convergence_candidate_last_reset_reason": "",
            "terminal_reason": "",
        }
        if not self.dry_run:
            self._write(payload)
        return payload

    def command_multi(self, target: Sequence[float], now: float) -> bytes:
        """Arm-gated single-shot multi-joint T102 command.

        Same arm deadline, health gate, and terminal settle policy as the
        single-joint path.  All moving joints must converge simultaneously;
        there is no non-target drift check because every moving joint is a
        target.
        """
        if not self.armed:
            raise SafetyError("driver is disarmed")
        if self.target is not None:
            raise SafetyError("one command is already active")
        if self.arm_command_deadline_at is None:
            self.latch_fault("armed state missing arm-command deadline")
            raise SafetyError("armed state missing arm-command deadline")
        if now >= self.arm_command_deadline_at:
            self._finish("ARM_COMMAND_TIMEOUT_DISARMED")
            raise SafetyError("arm command deadline expired")
        if not self.healthy(now):
            raise SafetyError(self.fault or "feedback is unhealthy or stale")
        return self._activate_multi_joint_t102(target, now, replace=False)

    def command_multi_replace(self, target: Sequence[float], now: float) -> bytes:
        """Replace the active command with a multi-joint T102 without disarming.

        Used by hold teleop streaming: validate the new absolute goal against
        current measured joints (same delta ceilings), rewrite T102, and reset
        convergence / deadline state.  Works after an initial T102 (preferred)
        or a T101 single-joint command.
        """
        if self.exact_plan_no_reissue_active:
            raise SafetyError("dynamic exact precontact forbids target replace")
        if not self.armed:
            raise SafetyError("driver is disarmed")
        if self.target is None:
            raise SafetyError("no active command to replace")
        if not self.healthy(now):
            raise SafetyError(self.fault or "feedback is unhealthy or stale")
        return self._activate_multi_joint_t102(target, now, replace=True)

    def command_dynamic_exact_advance(
        self, target: Sequence[float], now: float
    ) -> bytes:
        """Advance once to a different frozen waypoint without disarming.

        The sole executor publisher remains responsible for immutable-plan
        identity.  The driver permits the handoff only after measured feedback
        is inside the normal Dynamic terminal tolerance for the current exact
        target, and rejects duplicate-target reissue.
        """
        if not self.exact_plan_no_reissue_active:
            raise SafetyError("dynamic exact advance requires its active lease")
        if not self.armed or self.target is None or self.feedback is None:
            raise SafetyError("dynamic exact advance requires an active command")
        if not self.healthy(now):
            raise SafetyError(self.fault or "feedback is unhealthy or stale")
        current_error = max(
            abs(measured - expected)
            for measured, expected in zip(self.feedback.positions, self.target)
        )
        if current_error > DYNAMIC_PRECONTACT_TOLERANCE_RAD:
            raise SafetyError("dynamic exact advance before waypoint tolerance")
        requested = tuple(float(value) for value in target)
        if max(abs(a-b) for a,b in zip(requested, self.target)) <= ENCODER_STEP_RAD / 2.0:
            raise SafetyError("dynamic exact waypoint reissue rejected")
        return self._activate_multi_joint_t102(requested, now, replace=True)

    def _activate_multi_joint_t102(
        self, target: Sequence[float], now: float, *, replace: bool
    ) -> bytes:
        checked, moving = validate_multi_joint_target(
            target,
            self.feedback.positions,  # type: ignore[arg-type]
            self.joint_max_delta_rad,  # type: ignore[arg-type]
        )
        # Snap held joints to the exact measured pose and quantize movers onto
        # the encoder grid.  Tiny float residuals on "held" joints inside a
        # T102 absolute payload have caused partial undertravel on hardware.
        measured = self.feedback.positions  # type: ignore[union-attr]
        snapped: list[float] = []
        for index, value in enumerate(checked):
            if index not in moving:
                # Hold measured pose, but keep the absolute T102 field inside
                # command limits (gripper feedback often sits one encoder step
                # below 0.0 and would otherwise fail build_t102).
                lower, upper = T101_COMMAND_LIMITS[index]
                snapped.append(min(upper, max(lower, float(measured[index]))))
                continue
            steps = round(float(value) / ENCODER_STEP_RAD)
            quantized = steps * ENCODER_STEP_RAD
            lower, upper = T101_COMMAND_LIMITS[index]
            snapped.append(min(upper, max(lower, quantized)))
        checked = tuple(snapped)
        checked, moving = validate_multi_joint_target(
            checked,
            measured,  # type: ignore[arg-type]
            self.joint_max_delta_rad,  # type: ignore[arg-type]
        )
        # Shared T102 speed stays conservative (min of movers).  Acceleration
        # follows the configured per-joint registers so a diagnostic profile
        # can use the vendor-documented acc=10 instead of a hard-coded 1.
        speed = (
            min(self.joint_speed_steps_s[idx] for idx in moving)
            if moving
            else min(self.joint_speed_steps_s)
        )
        acceleration = (
            max(int(self.joint_acc_registers[idx]) for idx in moving)
            if moving
            else int(self.joint_acc_registers[0])
        )
        payload = build_t102(checked, speed, acceleration)
        self.target = checked
        self.active_joint_index = None  # multi-joint: no single selected joint
        self._multi_joint_moving = frozenset(moving)
        self.pre_command_positions = self.feedback.positions
        self.arm_command_deadline_at = None
        self.command_started_at = now
        # Estimate motion time from the largest joint delta.
        max_delta = (
            max(abs(checked[idx] - self.feedback.positions[idx]) for idx in moving)
            if moving
            else 0.0
        )
        expected_time = expected_motion_time_s(max_delta, speed, acceleration)
        self.command_deadline_at = now + max(
            self.command_timeout_s, expected_time + 2.0
        )
        self.in_tolerance_count = 0
        self._convergence_candidate = []
        self.last_result = "DRY_RUN_ACTIVE" if self.dry_run else "COMMAND_ACTIVE"
        self.telemetry = {
            "command_started_at": now,
            "command_deadline_at": self.command_deadline_at,
            "expected_motion_time_s": expected_time,
            "mode": "multi_joint_t102_replace" if replace else "multi_joint_t102",
            "moving_joint_indices": list(moving),
            "moving_joint_names": [JOINT_NAMES[idx] for idx in moving],
            "speed_steps_s": speed,
            "acc_register": acceleration,
            "max_joint_delta_rad": [self.joint_max_delta_rad[idx] for idx in moving],
            "pre_command_positions": list(self.pre_command_positions),
            "target_positions": list(checked),
            "outgoing": payload.decode("ascii").strip(),
            "feedback_samples": [],
            "joint_errors_rad": {
                JOINT_NAMES[idx]: checked[idx] - self.feedback.positions[idx]
                for idx in moving
            },
            "terminal_settle_policy": {
                "window_s": TERMINAL_SETTLE_WINDOW_S,
                "min_samples": self.convergence_samples,
                "max_feedback_gap_s": TERMINAL_MAX_FEEDBACK_GAP_S,
                "max_joint_span_rad": TERMINAL_MAX_SETTLE_SPAN_RAD,
            },
            "convergence_candidate_started_at": None,
            "convergence_candidate_sample_count": 0,
            "convergence_candidate_duration_s": 0.0,
            "convergence_candidate_span_rad": [0.0] * len(JOINT_NAMES),
            "convergence_candidate_reset_count": 0,
            "convergence_candidate_last_reset_reason": "",
            "terminal_reason": "",
            "t102_reissue_count": 0,
            "t102_reissue_at": None,
            "replaced": bool(replace),
            "coarse_progress_active": self.coarse_progress_active,
            "dynamic_exact_precontact_active": self.dynamic_exact_precontact_active,
            "exact_plan_no_reissue_active": self.exact_plan_no_reissue_active,
            "exact_plan_lease_purpose": self.exact_plan_lease_purpose,
            "dynamic_exact_no_reissue": self.exact_plan_no_reissue_active,
            "coarse_progress_endpoint_rad": (
                None
                if self.coarse_progress_endpoint is None
                else list(self.coarse_progress_endpoint)
            ),
        }
        if not self.dry_run:
            self._write(payload)
        if self.exact_plan_no_reissue_active:
            self.dynamic_exact_accepted_command_sequence += 1
            self.dynamic_exact_accepted_target_positions = tuple(checked)
        return payload

    def _maybe_reissue_stalled_t102(self, now: float) -> None:
        """Rewrite one stalled multi-joint command before the deadline faults.

        Live Quick Control evidence showed T102 sometimes leaving a mover
        unmoved until convergence timeout.  A single reissue keeps the same
        armed target and does not open a second concurrent command.
        """
        if self.coarse_progress_active or self.exact_plan_no_reissue_active:
            return
        if not self._multi_joint_moving or self.target is None or self.dry_run:
            return
        if int(self.telemetry.get("t102_reissue_count", 0) or 0) >= 1:
            return
        started = self.command_started_at
        expected = float(self.telemetry.get("expected_motion_time_s") or 0.0)
        if started is None or now < started + expected + 1.0:
            return
        max_error = self.telemetry.get("max_joint_error_rad")
        if not isinstance(max_error, (int, float)) or max_error <= self.tolerance_rad:
            return
        samples = self.telemetry.get("feedback_samples") or []
        if len(samples) < 8:
            return
        # Decide from the current settle window, not from the beginning of the
        # command.  A normal long T102 can make substantial early progress and
        # then stall just outside tolerance; comparing against sample zero
        # incorrectly classifies that terminal stall as still progressing and
        # suppresses the one permitted same-target reissue forever.
        recent_samples = samples[-8:]
        first_error = recent_samples[0].get("max_joint_error_rad", max_error)
        if not isinstance(first_error, (int, float)):
            return
        # Still progressing by more than one encoder step → do not reissue.
        if float(first_error) - float(max_error) > ENCODER_STEP_RAD:
            return
        speed = int(
            self.telemetry.get("speed_steps_s") or min(self.joint_speed_steps_s)
        )
        acceleration = int(self.telemetry.get("acc_register") or 1)
        payload = build_t102(self.target, speed, acceleration)
        self._write(payload)
        self.telemetry["t102_reissue_count"] = 1
        self.telemetry["t102_reissue_at"] = now
        # The initial deadline is based on the first T102's expected motion.
        # A permitted reissue happens near that deadline, while soft-settle
        # deliberately waits 1.5 s after the write.  Extend only this active
        # command's observation window; the reissue count and target remain
        # unchanged.
        post_reissue_deadline = now + T102_REISSUE_POST_SETTLE_DEADLINE_S
        if (
            self.command_deadline_at is None
            or self.command_deadline_at < post_reissue_deadline
        ):
            self.command_deadline_at = post_reissue_deadline
            self.telemetry["command_deadline_at"] = self.command_deadline_at
        self.telemetry["t102_reissue_post_settle_deadline_at"] = self.command_deadline_at

    def _maybe_soft_converge_stalled_t102(self, now: float) -> bool:
        """Accept a stalled multi-joint settle when residual is modest.

        Live Quick Control T102 evidence: shoulder often stops ~1.0–1.5° short
        of the absolute goal and never re-enters hard tolerance, so the driver
        latches ``command convergence timeout`` even though the arm is idle.
        After one stalled reissue, allow soft converge up to 2× tolerance so
        Cartesian jog can continue without clearing a false hard fault.
        Hold-teleop replace streams use a slightly wider soft band (3.5×) and
        may soft-settle after expected motion time even if release stops the
        stream before the reissue path finishes.  Larger residuals still fail
        closed on the command deadline.
        """
        if self.coarse_progress_active or self.exact_plan_no_reissue_active:
            return False
        if not self._multi_joint_moving or self.target is None:
            return False
        replaced = bool(self.telemetry.get("replaced"))
        soft_tol = (3.5 if replaced else 2.0) * self.tolerance_rad
        started = self.command_started_at
        expected = float(self.telemetry.get("expected_motion_time_s") or 0.0)
        if replaced:
            if started is None or now < float(started) + expected + 1.5:
                return False
        else:
            if int(self.telemetry.get("t102_reissue_count", 0) or 0) < 1:
                return False
            reissue_at = self.telemetry.get("t102_reissue_at")
            if not isinstance(reissue_at, (int, float)):
                return False
            if now < float(reissue_at) + 1.5:
                return False
        max_error = self.telemetry.get("max_joint_error_rad")
        if not isinstance(max_error, (int, float)) or float(max_error) > soft_tol:
            return False
        samples = self.telemetry.get("feedback_samples") or []
        if len(samples) < 8:
            return False
        recent_errors = [sample.get("max_joint_error_rad") for sample in samples[-8:]]
        if any(not isinstance(error, (int, float)) for error in recent_errors):
            return False
        if (
            max(float(error) for error in recent_errors)
            - min(float(error) for error in recent_errors)
            > ENCODER_STEP_RAD
        ):
            return False
        self.telemetry["soft_converge"] = True
        self.telemetry["soft_converge_max_joint_error_rad"] = float(max_error)
        self.telemetry["soft_converge_tolerance_rad"] = soft_tol
        self._finish("CONVERGED_DISARMED")
        return True

    def tick(self, now: float) -> None:
        if (
            self.feedback is not None
            and now - self.feedback.received_at > self.feedback_timeout_s
        ):
            self.latch_fault("stale feedback watchdog")
            return
        if (
            self.armed
            and self.target is None
            and self.arm_command_deadline_at is not None
            and now >= self.arm_command_deadline_at
        ):
            self._finish("ARM_COMMAND_TIMEOUT_DISARMED")
            return
        # Deadline is authoritative: a late feedback sample must not turn into
        # a success merely because the soft-settle predicate runs first.
        if self.target is not None and self.command_deadline_at is not None:
            if now > self.command_deadline_at:
                self.latch_fault("command convergence timeout")
                return
        if self.target is not None and self._multi_joint_moving:
            self._maybe_reissue_stalled_t102(now)
            if self._maybe_soft_converge_stalled_t102(now):
                return

    def _finish(self, result: str) -> None:
        self.armed = False
        self.target = None
        self.active_joint_index = None
        self.pre_command_positions = None
        self.arm_command_deadline_at = None
        self.command_started_at = None
        self.command_deadline_at = None
        self.in_tolerance_count = 0
        self._convergence_candidate = []
        self._convergence_candidate_result = "CONVERGED_DISARMED"
        self._multi_joint_moving = frozenset()
        self.last_result = result
        self.telemetry["terminal_reason"] = result

    def latch_fault(self, reason: str) -> None:
        self.fault = reason
        self._finish(f"FAULT:{reason}")

    def clear_fault(self) -> None:
        self.fault = None
        self.armed = False
        self.target = None
        self.active_joint_index = None
        self.pre_command_positions = None
        self.arm_command_deadline_at = None
        self.command_started_at = None
        self.command_deadline_at = None
        self.in_tolerance_count = 0
        self._convergence_candidate = []
        self._convergence_candidate_result = "CONVERGED_DISARMED"
        self._multi_joint_moving = frozenset()
        self.last_result = "FAULT_CLEARED_DISARMED"

    def status_snapshot(self) -> dict:
        now = time.monotonic()
        return {
            "state": self.last_result,
            "armed": self.armed,
            "dry_run": self.dry_run,
            "fault": self.fault or "",
            "joint_speed_steps_s": list(self.joint_speed_steps_s),
            "joint_acc_registers": list(self.joint_acc_registers),
            "joint_acceleration_steps_s2": [
                value * 100 for value in self.joint_acc_registers
            ],
            "joint_max_delta_rad": list(self.joint_max_delta_rad),
            "tolerance_rad": self.tolerance_rad,
            "arm_command_deadline_at": self.arm_command_deadline_at,
            "coarse_progress_active": self.coarse_progress_active,
            "dynamic_exact_precontact_active": self.dynamic_exact_precontact_active,
            "exact_plan_no_reissue_active": self.exact_plan_no_reissue_active,
            "exact_plan_lease_purpose": self.exact_plan_lease_purpose,
            "dynamic_exact_no_reissue": self.exact_plan_no_reissue_active,
            "dynamic_exact_accepted_command_sequence": int(
                self.dynamic_exact_accepted_command_sequence
            ),
            "dynamic_exact_accepted_target_positions": list(
                self.dynamic_exact_accepted_target_positions
            ),
            "dynamic_exact_advance_ready": bool(
                self.dynamic_exact_advance_ready(now)
            ),
            "coarse_progress_endpoint_rad": (
                None
                if self.coarse_progress_endpoint is None
                else list(self.coarse_progress_endpoint)
            ),
            "telemetry": copy.deepcopy(self.telemetry),
        }


class PrecontactSpeedLease:
    """Single-owner, non-replayable speed lease for exactly one T102 action.

    This is a pure state machine (no ROS, no transport): the ROS node passes
    the controller and a live ``foreign_publishers`` flag, and the lease
    applies or restores the exact speed tuple while holding the controller
    lock.  The Dynamic exact-plan purpose also applies its reviewed terminal
    tolerance; delta caps, retry/reissue counts, acceleration registers and
    generic Quick Control speeds are untouched.

    Enable fails closed unless: non-dry-run healthy fresh feedback, disarmed
    idle controller (``armed`` false and ``target`` None), all six torque
    flags true, no foreign ``/roarm_safe/joint_target`` publisher, no existing
    lease, the exact accepted purpose, and a fresh lease id.  Release requires
    the same purpose/lease identity and a disarmed idle controller, then
    restores the tuple captured before enable.  A failed release leaves the
    lease active (safe failure) so the operator/driver never silently drops
    the elevated speed back to a profile the lease no longer owns.

    Lease ids are one-use: a bounded SHA-256 history rejects replays so a
    captured enable request cannot reopen a released lease.
    """

    def __init__(self, *, used_history_limit: int = 64) -> None:
        if (
            isinstance(used_history_limit, bool)
            or not isinstance(used_history_limit, int)
            or used_history_limit <= 0
        ):
            raise SafetyError("lease used-id history limit must be a positive integer")
        self.active = False
        self.purpose = ""
        self.lease_id = ""
        self.original_speed_steps_s: Optional[tuple[int, ...]] = None
        self.original_tolerance_rad: Optional[float] = None
        self._used_hashes: deque[str] = deque(maxlen=used_history_limit)
        self._used_set: set[str] = set()

    def enable(
        self,
        controller: SafeController,
        *,
        purpose: str,
        lease_id: str,
        foreign_publishers: bool,
    ) -> tuple[bool, str]:
        """Open the lease and apply the fixed 34-steps/s profile."""
        if self.active:
            return False, "precontact speed lease already active"
        if purpose not in (
            PRECONTACT_LEASE_PURPOSE,
            COARSE_PRECONTACT_LEASE_PURPOSE,
            *EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES,
        ):
            return False, "lease purpose is not accepted"
        if not isinstance(lease_id, str) or not lease_id or len(lease_id) > 256:
            return False, "lease_id is required"
        digest = hashlib.sha256(lease_id.encode("utf-8")).hexdigest()
        if digest in self._used_set:
            return False, "lease_id replay rejected"
        if foreign_publishers:
            return False, "foreign /roarm_safe/joint_target publisher present"
        if controller.dry_run:
            return False, "precontact speed lease requires non-dry-run driver"
        now = time.monotonic()
        if not controller.healthy(now):
            return False, "precontact speed lease requires healthy fresh feedback"
        if controller.armed or controller.target is not None:
            return False, "precontact speed lease requires disarmed idle driver"
        if controller.feedback is None or not all(controller.feedback.torque_enabled):
            return False, "all six torque flags must be enabled"
        if (
            purpose == COARSE_PRECONTACT_LEASE_PURPOSE
            and controller.coarse_progress_endpoint is None
        ):
            return False, "coarse progress endpoint is not configured"
        self.active = True
        self.purpose = purpose
        self.lease_id = lease_id
        self.original_speed_steps_s = tuple(controller.joint_speed_steps_s)
        self.original_tolerance_rad = float(controller.tolerance_rad)
        controller.joint_speed_steps_s = PRECONTACT_LEASE_SPEED_STEPS_S
        controller.coarse_progress_active = (
            purpose == COARSE_PRECONTACT_LEASE_PURPOSE
        )
        controller.exact_plan_no_reissue_active = (
            purpose in EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES
        )
        controller.exact_plan_lease_purpose = (
            purpose if controller.exact_plan_no_reissue_active else ""
        )
        controller.dynamic_exact_precontact_active = (
            purpose == DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE
        )
        if controller.exact_plan_no_reissue_active:
            controller.reset_dynamic_exact_command_acceptance()
            controller.tolerance_rad = DYNAMIC_PRECONTACT_TOLERANCE_RAD
        self._remember(digest)
        return True, "PRECONTACT_SPEED_LEASE_ACTIVE"

    def release(
        self,
        controller: SafeController,
        *,
        purpose: str,
        lease_id: str,
    ) -> tuple[bool, str]:
        """Close the lease and restore the exact tuple captured at enable."""
        if not self.active:
            return False, "no active precontact speed lease"
        if (
            not secrets.compare_digest(str(purpose), self.purpose)
            or not secrets.compare_digest(str(lease_id), self.lease_id)
        ):
            return False, "lease identity does not match"
        if controller.armed or controller.target is not None:
            return False, "lease release requires disarmed idle driver"
        if self.original_speed_steps_s is None:
            return False, "lease lost its original speed profile"
        if self.original_tolerance_rad is None:
            return False, "lease lost its original tolerance"
        controller.joint_speed_steps_s = self.original_speed_steps_s
        controller.tolerance_rad = self.original_tolerance_rad
        controller.coarse_progress_active = False
        controller.dynamic_exact_precontact_active = False
        controller.exact_plan_no_reissue_active = False
        controller.exact_plan_lease_purpose = ""
        controller.reset_dynamic_exact_command_acceptance()
        self.active = False
        self.purpose = ""
        self.lease_id = ""
        self.original_speed_steps_s = None
        self.original_tolerance_rad = None
        return True, "PRECONTACT_SPEED_LEASE_RELEASED"

    def release_precheck(
        self,
        controller: SafeController,
        *,
        purpose: str,
        lease_id: str,
    ) -> tuple[bool, str]:
        """Prove release identity/state before any node-side PID transport write."""

        if not self.active:
            return False, "no active precontact speed lease"
        if (
            not secrets.compare_digest(str(purpose), self.purpose)
            or not secrets.compare_digest(str(lease_id), self.lease_id)
        ):
            return False, "lease identity does not match"
        if controller.armed or controller.target is not None:
            return False, "lease release requires disarmed idle driver"
        if self.original_speed_steps_s is None:
            return False, "lease lost its original speed profile"
        if self.original_tolerance_rad is None:
            return False, "lease lost its original tolerance"
        return True, "PRECONTACT_SPEED_LEASE_RELEASE_READY"

    def snapshot(self) -> dict:
        """Lease-safe state exposure for driver status telemetry."""
        return {
            "active": self.active,
            "purpose": self.purpose if self.active else "",
            "lease_id": self.lease_id if self.active else "",
            "speed_steps_s": (
                list(PRECONTACT_LEASE_SPEED_STEPS_S) if self.active else None
            ),
            "original_speed_steps_s": (
                list(self.original_speed_steps_s) if self.active else None
            ),
            "dynamic_exact_no_reissue": (
                self.active
                and self.purpose in EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES
            ),
            "dynamic_exact_tolerance_rad": (
                DYNAMIC_PRECONTACT_TOLERANCE_RAD
                if self.active
                and self.purpose in EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES
                else None
            ),
        }

    def _remember(self, digest: str) -> None:
        if digest in self._used_set:
            return
        if len(self._used_hashes) >= self._used_hashes.maxlen:
            dropped = self._used_hashes.popleft()
            self._used_set.discard(dropped)
        self._used_hashes.append(digest)
        self._used_set.add(digest)
