"""ROS-agnostic one-click XY correction contract.

The production policy currently disables runtime correction because both
projected centers are outside the wrist frame and calibration is diagnostic.
The pure state machine remains testable with explicit candidate joint targets;
it never computes IK, publishes, retries, or creates motion resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence


class XYState(str, Enum):
    NEED_MEASURE = "NEED_MEASURE"
    READY_ONE = "READY_ONE"
    ACTION_IN_FLIGHT = "ACTION_IN_FLIGHT"
    NEED_NEW_FRAME = "NEED_NEW_FRAME"
    ALIGNED = "NO_CONTACT_XY_ALIGNED"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class XYDecision:
    accepted: bool
    code: str
    state: XYState
    band: str = ""
    target_rad: tuple[float, ...] = ()


def _finite(values: Sequence[float], n: int, name: str) -> tuple[float, ...]:
    if len(values) != n:
        raise ValueError(f"{name} must have {n} values")
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return out


class DiscreteXYCore:
    def __init__(self, session_id: str, policy: Mapping, *, runtime_enabled: bool = False) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.policy = policy
        self.runtime_enabled = bool(runtime_enabled)
        self.state = XYState.NEED_MEASURE
        self.correction_count = 0
        self.last_frame_stamp_ns = 0
        self._before_error = (0.0, 0.0)
        self._before_q: tuple[float, ...] = ()
        self._requested_target: tuple[float, ...] = ()

    def measure(
        self, *, session_id: str, frame_stamp_ns: int, frame_age_s: float,
        button_available: bool, tool_center_available: bool,
        error_px: Sequence[float], feedback_age_s: float, current_q: Sequence[float]
    ) -> XYDecision:
        if session_id != self.session_id:
            return XYDecision(False, "SESSION_CHANGED", XYState.TERMINAL)
        if not button_available:
            return XYDecision(False, "BUTTON_ROI_UNAVAILABLE", self.state)
        if not tool_center_available:
            return XYDecision(False, "TOOL_CENTER_UNAVAILABLE", self.state)
        if frame_age_s > float(self.policy["maximum_frame_age_s"]):
            return XYDecision(False, "WRIST_FRAME_STALE", self.state)
        if feedback_age_s > float(self.policy["maximum_feedback_age_s"]):
            return XYDecision(False, "FEEDBACK_STALE", self.state)
        if frame_stamp_ns <= self.last_frame_stamp_ns:
            return XYDecision(False, "NEW_FRAME_REQUIRED", self.state)
        error = _finite(error_px, 2, "error_px")
        q = _finite(current_q, 6, "current_q")
        bands = self.policy["simulation_only_pixel_bands"]
        magnitude = max(abs(error[0]), abs(error[1]))
        self.last_frame_stamp_ns = int(frame_stamp_ns)
        self._before_error, self._before_q = error, q
        if magnitude <= float(bands["aligned_max_abs_px"]):
            self.state = XYState.ALIGNED
            return XYDecision(True, "NO_CONTACT_XY_ALIGNED", self.state, "ALIGNED")
        if self.correction_count >= int(self.policy["maximum_corrections_per_session"]):
            self.state = XYState.TERMINAL
            return XYDecision(False, "CORRECTION_LIMIT_REACHED", self.state)
        self.state = XYState.READY_ONE
        return XYDecision(True, "MEASURED", self.state, "MEDIUM" if magnitude <= float(bands["medium_max_abs_px"]) else "LARGE")

    def apply_one(self, *, session_id: str, candidate_target_rad: Sequence[float],
                  predicted_cross_axis_drift_px: float, predicted_normal_drift_m: float) -> XYDecision:
        if session_id != self.session_id:
            return XYDecision(False, "SESSION_CHANGED", XYState.TERMINAL)
        if self.state is not XYState.READY_ONE:
            return XYDecision(False, "MEASURE_REQUIRED", self.state)
        if not self.runtime_enabled:
            return XYDecision(False, "TOOL_CENTER_UNAVAILABLE", self.state)
        target = _finite(candidate_target_rad, 6, "candidate_target_rad")
        hard = self.policy.get("hard_joint_limits_rad")
        if hard and any(not (float(lo) <= q <= float(hi)) for q, (lo, hi) in zip(target, hard)):
            return XYDecision(False, "CORRECTION_ENVELOPE_REJECTED", self.state)
        delta = tuple(b-a for a, b in zip(self._before_q, target))
        caps = _finite(self.policy["maximum_joint_delta_rad"], 6, "joint caps")
        if any(abs(d) > cap + 1e-12 for d, cap in zip(delta, caps)):
            return XYDecision(False, "CORRECTION_ENVELOPE_REJECTED", self.state)
        useful = max(abs(d) for d in delta)
        if useful < float(self.policy["minimum_useful_joint_delta_rad"]) - 1e-12:
            return XYDecision(False, "DEADBAND_UNRESOLVED", self.state)
        if abs(float(predicted_cross_axis_drift_px)) > float(self.policy["maximum_cross_axis_drift_px"]):
            return XYDecision(False, "CORRECTION_ENVELOPE_REJECTED", self.state)
        if abs(float(predicted_normal_drift_m)) > float(self.policy["maximum_normal_drift_m"]):
            return XYDecision(False, "CORRECTION_ENVELOPE_REJECTED", self.state)
        self._requested_target = target
        self.correction_count += 1
        self.state = XYState.ACTION_IN_FLIGHT
        return XYDecision(True, "APPLY_ONE_XY", self.state, target_rad=target)

    def record_terminal(self, *, measured_q: Sequence[float], new_frame_stamp_ns: int,
                        new_error_px: Sequence[float]) -> XYDecision:
        if self.state is not XYState.ACTION_IN_FLIGHT:
            return XYDecision(False, "NO_CORRECTION_IN_FLIGHT", self.state)
        if int(new_frame_stamp_ns) <= self.last_frame_stamp_ns:
            self.state = XYState.NEED_NEW_FRAME
            return XYDecision(False, "NEW_FRAME_REQUIRED", self.state)
        measured = _finite(measured_q, 6, "measured_q")
        after_error = _finite(new_error_px, 2, "new_error_px")
        requested = tuple(b-a for a, b in zip(self._before_q, self._requested_target))
        actual = tuple(b-a for a, b in zip(self._before_q, measured))
        progress = sum(a*b for a, b in zip(actual, requested)) / max(math.sqrt(sum(v*v for v in requested)), 1e-12)
        if progress < -float(self.policy["maximum_wrong_direction_rad"]):
            self.state = XYState.TERMINAL
            return XYDecision(False, "WRONG_DIRECTION", self.state)
        if progress < float(self.policy["minimum_measured_progress_rad"]):
            self.state = XYState.TERMINAL
            return XYDecision(False, "NO_PROGRESS", self.state)
        before_norm = math.hypot(*self._before_error)
        after_norm = math.hypot(*after_error)
        if after_norm >= before_norm:
            self.state = XYState.TERMINAL
            return XYDecision(False, "NO_PROGRESS", self.state)
        self.last_frame_stamp_ns = int(new_frame_stamp_ns)
        self.state = XYState.NEED_MEASURE
        return XYDecision(True, "CORRECTION_COMPLETE_MEASURE_AGAIN", self.state)
