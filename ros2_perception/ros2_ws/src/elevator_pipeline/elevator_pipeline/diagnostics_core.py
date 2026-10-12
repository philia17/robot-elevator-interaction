"""Bounded, ROS-independent health aggregation for Phase C."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class HealthSnapshot:
    overall_status: str
    input_fps: float
    detector_latency_ms: float
    vlm_latency_ms: float
    fusion_latency_ms: float
    total_latency_ms: float
    detected_count: int
    labeled_count: int
    observation_count: int
    target_valid: bool
    task_active: bool
    current_state: str
    last_frame_age_sec: float
    last_frame_accepted: object
    timeout_count: int
    failure_count: int
    last_error: str
    warnings: tuple


class DiagnosticsTracker:
    def __init__(self, history_size=30, stale_frame_sec=2.5):
        if int(history_size) <= 1:
            raise ValueError('history_size must be greater than one')
        self._frame_times = deque(maxlen=int(history_size))
        self.stale_frame_sec = float(stale_frame_sec)
        self.last_frame_accepted = None
        self.detector_latency_ms = -1.0
        self.vlm_latency_ms = -1.0
        self.fusion_latency_ms = -1.0
        self.total_latency_ms = -1.0
        self.detected_count = 0
        self.labeled_count = 0
        self.observation_count = 0
        self.target_valid = False
        self.task_active = False
        self.current_state = 'IDLE'
        self.timeout_count = 0
        self.failure_count = 0
        self.last_error = ''

    @property
    def frame_history_size(self):
        return len(self._frame_times)

    def record_frame(self, accepted, now_monotonic):
        self._frame_times.append(float(now_monotonic))
        self.last_frame_accepted = bool(accepted)

    def record_detections(self, count, latency_ms):
        self.detected_count = max(0, int(count))
        self.detector_latency_ms = float(latency_ms)

    def record_semantics(self, count, latency_ms):
        self.labeled_count = max(0, int(count))
        self.vlm_latency_ms = float(latency_ms)

    def record_observations(self, count):
        self.observation_count = max(0, int(count))

    def record_target(self, valid):
        self.target_valid = bool(valid)

    def record_task(self, active):
        self.task_active = bool(active)

    def record_state(self, state):
        next_state = str(state or 'unknown')
        if next_state != self.current_state:
            if next_state == 'TIMEOUT':
                self.timeout_count += 1
            elif next_state == 'FAILED':
                self.failure_count += 1
        self.current_state = next_state

    def record_error(self, error):
        self.last_error = str(error or '')

    def _input_fps(self):
        if len(self._frame_times) < 2:
            return -1.0
        duration = self._frame_times[-1] - self._frame_times[0]
        if duration <= 0.0:
            return -1.0
        return (len(self._frame_times) - 1) / duration

    def snapshot(self, now_monotonic):
        if self._frame_times:
            frame_age = max(0.0, float(now_monotonic) - self._frame_times[-1])
        else:
            frame_age = -1.0
        warnings = []
        status = 'OK'
        if not self._frame_times:
            warnings.append('no_frame_received')
            status = 'WARN'
        elif frame_age > self.stale_frame_sec:
            warnings.append('frame_stale')
            status = 'WARN'
        if self.last_frame_accepted is False:
            warnings.append('last_frame_rejected')
            status = 'WARN'
        if self.current_state in {'FAILED', 'TIMEOUT'}:
            warnings.append('task_{}'.format(self.current_state.lower()))
            status = 'ERROR'
        if self.last_error:
            warnings.append('last_error_present')
            status = 'ERROR'
        return HealthSnapshot(
            overall_status=status,
            input_fps=self._input_fps(),
            detector_latency_ms=self.detector_latency_ms,
            vlm_latency_ms=self.vlm_latency_ms,
            fusion_latency_ms=self.fusion_latency_ms,
            total_latency_ms=self.total_latency_ms,
            detected_count=self.detected_count,
            labeled_count=self.labeled_count,
            observation_count=self.observation_count,
            target_valid=self.target_valid,
            task_active=self.task_active,
            current_state=self.current_state,
            last_frame_age_sec=frame_age,
            last_frame_accepted=self.last_frame_accepted,
            timeout_count=self.timeout_count,
            failure_count=self.failure_count,
            last_error=self.last_error,
            warnings=tuple(warnings),
        )
