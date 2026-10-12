"""ROS-independent Action task state machine for Phase C."""

from dataclasses import dataclass
import math
import uuid

from .floor_map import FloorMapError


IDLE = 'IDLE'
WAITING_FRAME = 'WAITING_FRAME'
DETECTING = 'DETECTING'
LABELING = 'LABELING'
FUSING = 'FUSING'
SELECTING = 'SELECTING'
STABILIZING = 'STABILIZING'
SUCCESS = 'SUCCESS'
FAILED = 'FAILED'
TIMEOUT = 'TIMEOUT'
CANCELLED = 'CANCELLED'

ACTIVE_STATES = frozenset({
    WAITING_FRAME,
    DETECTING,
    LABELING,
    FUSING,
    SELECTING,
    STABILIZING,
})
TERMINAL_STATES = frozenset({SUCCESS, FAILED, TIMEOUT, CANCELLED})

STATE_PROGRESS = {
    IDLE: 0.0,
    WAITING_FRAME: 0.05,
    DETECTING: 0.20,
    LABELING: 0.35,
    FUSING: 0.50,
    SELECTING: 0.65,
    STABILIZING: 0.80,
    SUCCESS: 1.0,
    FAILED: 1.0,
    TIMEOUT: 1.0,
    CANCELLED: 1.0,
}


class GoalValidationError(ValueError):
    """Raised when an Action goal cannot start."""


class GoalAlreadyActiveError(RuntimeError):
    """Raised when a second goal tries to replace an active task."""


@dataclass(frozen=True)
class GoalSpec:
    building_id: str
    requested_floor: str
    timeout_sec: float


@dataclass(frozen=True)
class ManagerSnapshot:
    state: str
    active: bool
    terminal: bool
    request_id: str
    building_id: str
    requested_floor: str
    timeout_sec: float
    detected_count: int
    labeled_count: int
    current_candidate: str
    progress: float
    failure_reason: str
    target: object


class ManagerStateMachine:
    """Coordinate one task without duplicating perception-stage logic."""

    def __init__(self, floor_map, id_factory=None, maximum_timeout_sec=120.0):
        self._floor_map = floor_map
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._maximum_timeout_sec = float(maximum_timeout_sec)
        if self._maximum_timeout_sec <= 0.0:
            raise ValueError('maximum_timeout_sec must be positive')
        self.reset()

    @property
    def is_active(self):
        return self.state in ACTIVE_STATES

    @property
    def is_terminal(self):
        return self.state in TERMINAL_STATES

    def validate_goal(self, building_id, requested_floor, timeout_sec):
        building = str(building_id or '').strip()
        floor = str(requested_floor or '').strip()
        if not building:
            raise GoalValidationError('building_id must not be empty')
        if not floor:
            raise GoalValidationError('requested_floor must not be empty')
        try:
            timeout = float(timeout_sec)
        except (TypeError, ValueError) as exc:
            raise GoalValidationError('timeout_sec must be numeric') from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise GoalValidationError('timeout_sec must be positive and finite')
        if timeout > self._maximum_timeout_sec:
            raise GoalValidationError(
                'timeout_sec exceeds configured maximum {}'.format(
                    self._maximum_timeout_sec
                )
            )
        try:
            self._floor_map.get_floor(building, floor)
        except FloorMapError as exc:
            raise GoalValidationError(str(exc)) from exc
        return GoalSpec(building, floor, timeout)

    def start_goal(
            self,
            building_id,
            requested_floor,
            timeout_sec,
            now_monotonic,
            request_id=None):
        if self.is_active:
            raise GoalAlreadyActiveError('a perception task is already active')
        goal = self.validate_goal(building_id, requested_floor, timeout_sec)
        self.reset()
        self.request_id = str(request_id or self._id_factory())
        if not self.request_id:
            raise GoalValidationError('request_id generation returned an empty value')
        self.building_id = goal.building_id
        self.requested_floor = goal.requested_floor
        self.timeout_sec = goal.timeout_sec
        self.started_monotonic = float(now_monotonic)
        self.deadline_monotonic = self.started_monotonic + goal.timeout_sec
        self.state = WAITING_FRAME
        return self.request_id

    def record_frame(self, accepted):
        if not self.is_active:
            return False
        if accepted:
            self.state = DETECTING
        return True

    def record_detections(self, detected_count):
        if not self.is_active:
            return False
        self.detected_count = max(0, int(detected_count))
        self.state = LABELING
        return True

    def record_semantics(self, labeled_count):
        if not self.is_active:
            return False
        self.labeled_count = max(0, int(labeled_count))
        self.state = FUSING
        return True

    def record_observations(self):
        if not self.is_active:
            return False
        self.state = SELECTING
        return True

    def record_candidate(self, candidate):
        if not self.is_active:
            return False
        if getattr(candidate, 'request_id', '') != self.request_id:
            return False
        if bool(getattr(candidate, 'found', False)):
            button = getattr(candidate, 'button', None)
            self.current_candidate = getattr(button, 'detection_id', '')
            self.state = STABILIZING
        else:
            self.current_candidate = ''
            self.state = SELECTING
        return True

    def record_target(self, target):
        if not self.is_active:
            return False
        if getattr(target, 'request_id', '') != self.request_id:
            return False
        if bool(getattr(target, 'valid', False)):
            self.target = target
            button = getattr(target, 'button', None)
            self.current_candidate = getattr(button, 'detection_id', '')
            self.state = SUCCESS
        else:
            self.state = STABILIZING
        return True

    def tick(self, now_monotonic):
        if not self.is_active:
            return False
        if float(now_monotonic) >= self.deadline_monotonic:
            self.failure_reason = 'perception task timed out'
            self.state = TIMEOUT
            return True
        return False

    def cancel(self, reason='goal cancelled'):
        if not self.is_active:
            return False
        self.failure_reason = str(reason)
        self.state = CANCELLED
        return True

    def fail(self, reason):
        if not self.is_active:
            return False
        self.failure_reason = str(reason or 'perception task failed')
        self.state = FAILED
        return True

    def snapshot(self):
        return ManagerSnapshot(
            state=self.state,
            active=self.is_active,
            terminal=self.is_terminal,
            request_id=self.request_id,
            building_id=self.building_id,
            requested_floor=self.requested_floor,
            timeout_sec=self.timeout_sec,
            detected_count=self.detected_count,
            labeled_count=self.labeled_count,
            current_candidate=self.current_candidate,
            progress=STATE_PROGRESS[self.state],
            failure_reason=self.failure_reason,
            target=self.target,
        )

    def reset(self):
        self.state = IDLE
        self.request_id = ''
        self.building_id = ''
        self.requested_floor = ''
        self.timeout_sec = 0.0
        self.started_monotonic = 0.0
        self.deadline_monotonic = 0.0
        self.detected_count = 0
        self.labeled_count = 0
        self.current_candidate = ''
        self.failure_reason = ''
        self.target = None
