"""Pure no-motion M0-B target-freeze and occlusion validation state machine."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time


ACQUIRING = 'ACQUIRING'
READY_TO_LOCK = 'READY_TO_LOCK'
TARGET_LOCKED = 'TARGET_LOCKED'
OCCLUSION_EXPECTED = 'OCCLUSION_EXPECTED'
REACQUIRING = 'REACQUIRING'
FLOW_PASS_NO_LIGHT_EXPECTED = 'FLOW_PASS_NO_LIGHT_EXPECTED'
FAILED = 'FAILED'


def _canonical_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _bbox_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _position(target):
    point = target.get('camera_point_m') or {}
    values = tuple(float(point[key]) for key in ('x', 'y', 'z'))
    if not all(math.isfinite(value) for value in values):
        raise ValueError('camera_point_m must contain finite x/y/z')
    return values


def _bbox(target):
    values = tuple(float(value) for value in target.get('bbox', ()))
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError('bbox must contain four finite values')
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError('bbox must have positive area')
    return values


def _ready_target(target, target_label):
    if not target:
        return False
    return (
        target.get('status') == 'READY'
        and bool(target.get('perception_ready'))
        and str(target.get('target_label', '')).strip() == target_label
        and str(target.get('effective_label', '')).strip() == target_label
        and target.get('motion_authorized') is False
    )


def _compatible_stabilizing_target(target, target_label):
    if not target:
        return False
    return (
        target.get('status') == 'STABILIZING'
        and int(target.get('matching_label_count', 0)) == 1
        and str(target.get('target_label', '')).strip() == target_label
        and str(target.get('effective_label', '')).strip() == target_label
        and target.get('motion_authorized') is False
    )


def _reacquire_candidate(target, target_label):
    return _ready_target(
        target,
        target_label,
    ) or _compatible_stabilizing_target(target, target_label)


@dataclass(frozen=True)
class M0BTrialConfig:
    target_label: str = '1'
    required_reacquire_frames: int = 3
    minimum_reacquire_bbox_iou: float = 0.45
    maximum_reacquire_position_delta_m: float = 0.025
    maximum_lock_age_sec: float = 1.0
    maximum_postcheck_duration_sec: float = 10.0

    def __post_init__(self):
        if not self.target_label.strip():
            raise ValueError('target_label must not be empty')
        if self.required_reacquire_frames < 2:
            raise ValueError('required_reacquire_frames must be at least 2')
        if not 0.0 <= self.minimum_reacquire_bbox_iou <= 1.0:
            raise ValueError('minimum_reacquire_bbox_iou must be in [0, 1]')
        if self.maximum_reacquire_position_delta_m <= 0.0:
            raise ValueError(
                'maximum_reacquire_position_delta_m must be positive'
            )
        if self.maximum_lock_age_sec <= 0.0:
            raise ValueError('maximum_lock_age_sec must be positive')
        if self.maximum_postcheck_duration_sec <= 0.0:
            raise ValueError(
                'maximum_postcheck_duration_sec must be positive'
            )


class M0BTrial:
    """Freeze one READY target and prove it survives an occlusion interval."""

    def __init__(self, config=None, clock=None):
        self.config = config or M0BTrialConfig()
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self):
        self.state = ACQUIRING
        self.failure_reason = ''
        self.latest_ready = None
        self.frozen_target = None
        self.frozen_target_sha256 = ''
        self.reacquired_target = None
        self.reacquire_count = 0
        self._last_reacquire_stamp_ns = None
        self._postcheck_start_stamp_ns = None
        self._postcheck_start_monotonic_sec = None
        self.ignored_live_updates = 0
        self.transitions = []

    def _transition(self, state, event, source_stamp_ns=0):
        self.state = state
        self.transitions.append(
            {
                'state': state,
                'event': event,
                'source_stamp_ns': int(source_stamp_ns),
            }
        )

    def _snapshot(self):
        return {
            'schema': 'm0b_target_freeze_trial_v1',
            'mode': 'NO_MOTION',
            'state': self.state,
            'failure_reason': self.failure_reason,
            'target_label': self.config.target_label,
            'frozen_target': deepcopy(self.frozen_target),
            'frozen_target_sha256': self.frozen_target_sha256,
            'reacquired_target': deepcopy(self.reacquired_target),
            'reacquire_count': self.reacquire_count,
            'required_reacquire_frames': (
                self.config.required_reacquire_frames
            ),
            'ignored_live_updates': self.ignored_live_updates,
            'result': (
                FLOW_PASS_NO_LIGHT_EXPECTED
                if self.state == FLOW_PASS_NO_LIGHT_EXPECTED
                else ''
            ),
            'motion_authorized': False,
            'transitions': deepcopy(self.transitions),
        }

    def _expire_postcheck_if_needed(self, source_stamp_ns=0):
        if (
            self.state != REACQUIRING
            or self._postcheck_start_monotonic_sec is None
        ):
            return False
        elapsed_sec = max(
            0.0,
            float(self._clock()) - self._postcheck_start_monotonic_sec,
        )
        if elapsed_sec <= self.config.maximum_postcheck_duration_sec:
            return False
        self.failure_reason = 'postcheck TTL expired'
        self._transition(FAILED, 'postcheck_ttl_expired', source_stamp_ns)
        return True

    def observe(self, target):
        target = deepcopy(target) if target else None
        source_stamp_ns = (
            int(target.get('source_stamp_ns', 0)) if target else 0
        )
        if self._expire_postcheck_if_needed(source_stamp_ns):
            return self._snapshot()
        if self.state in (TARGET_LOCKED, OCCLUSION_EXPECTED):
            self.ignored_live_updates += 1
            return self._snapshot()

        if self.state == REACQUIRING:
            return self._observe_reacquire(target)

        if self.state in (FLOW_PASS_NO_LIGHT_EXPECTED, FAILED):
            return self._snapshot()

        if _ready_target(target, self.config.target_label):
            try:
                _bbox(target)
                _position(target)
            except (KeyError, TypeError, ValueError):
                self.latest_ready = None
                self.failure_reason = 'READY target has invalid bbox or XYZ'
                self.state = ACQUIRING
            else:
                self.latest_ready = target
                self.failure_reason = ''
                self.state = READY_TO_LOCK
        elif (
            self.state == READY_TO_LOCK
            and self.latest_ready is not None
            and _compatible_stabilizing_target(
                target,
                self.config.target_label,
            )
        ):
            source_stamp_ns = int(target.get('source_stamp_ns', 0))
            ready_stamp_ns = int(
                self.latest_ready.get('source_stamp_ns', 0)
            )
            if (
                source_stamp_ns >= ready_stamp_ns
                and (source_stamp_ns - ready_stamp_ns) / 1e9
                <= self.config.maximum_lock_age_sec
            ):
                self.failure_reason = 'holding recent READY snapshot'
                return self._snapshot()
            self.latest_ready = None
            self.failure_reason = 'recent READY snapshot expired'
            self.state = ACQUIRING
        else:
            self.latest_ready = None
            self.failure_reason = 'waiting for stable unique READY target'
            self.state = ACQUIRING
        return self._snapshot()

    def lock(self, current_source_stamp_ns=None):
        if self.state != READY_TO_LOCK or self.latest_ready is None:
            self.failure_reason = 'lock requires a stable unique READY target'
            return self._snapshot()
        ready_stamp_ns = int(self.latest_ready.get('source_stamp_ns', 0))
        current_stamp_ns = (
            ready_stamp_ns
            if current_source_stamp_ns is None
            else int(current_source_stamp_ns)
        )
        if (
            ready_stamp_ns <= 0
            or current_stamp_ns < ready_stamp_ns
            or (current_stamp_ns - ready_stamp_ns) / 1e9
            > self.config.maximum_lock_age_sec
        ):
            self.latest_ready = None
            self.state = ACQUIRING
            self.failure_reason = 'READY target is stale at lock time'
            return self._snapshot()
        self.frozen_target = deepcopy(self.latest_ready)
        self.frozen_target_sha256 = _canonical_hash(self.frozen_target)
        self.failure_reason = ''
        self._transition(
            TARGET_LOCKED,
            'lock_target',
            self.frozen_target.get('source_stamp_ns', 0),
        )
        return self._snapshot()

    def enter_occlusion(self):
        if self.state != TARGET_LOCKED:
            self.failure_reason = 'occlusion requires TARGET_LOCKED'
            return self._snapshot()
        self.failure_reason = ''
        self._transition(
            OCCLUSION_EXPECTED,
            'simulate_occlusion',
            self.frozen_target.get('source_stamp_ns', 0),
        )
        return self._snapshot()

    def clear_occlusion(self, source_stamp_ns=0):
        if self.state != OCCLUSION_EXPECTED:
            self.failure_reason = 'clear requires OCCLUSION_EXPECTED'
            return self._snapshot()
        self.reacquire_count = 0
        self.reacquired_target = None
        self._last_reacquire_stamp_ns = None
        self._postcheck_start_stamp_ns = int(source_stamp_ns)
        self._postcheck_start_monotonic_sec = float(self._clock())
        self.failure_reason = ''
        self._transition(
            REACQUIRING,
            'clear_occlusion',
            self._postcheck_start_stamp_ns,
        )
        return self._snapshot()

    def _observe_reacquire(self, target):
        if not _reacquire_candidate(target, self.config.target_label):
            self.reacquire_count = 0
            self._last_reacquire_stamp_ns = None
            self.failure_reason = (
                'waiting to reacquire a unique stable floor-1 target'
            )
            return self._snapshot()
        try:
            frozen_bbox = _bbox(self.frozen_target)
            candidate_bbox = _bbox(target)
            frozen_position = _position(self.frozen_target)
            candidate_position = _position(target)
        except (KeyError, TypeError, ValueError):
            self.reacquire_count = 0
            self._last_reacquire_stamp_ns = None
            self.failure_reason = 'reacquired target has invalid bbox or XYZ'
            return self._snapshot()

        bbox_iou = _bbox_iou(frozen_bbox, candidate_bbox)
        position_delta = math.dist(frozen_position, candidate_position)
        stamp_ns = int(target.get('source_stamp_ns', 0))
        stamp_increasing = (
            self._last_reacquire_stamp_ns is None
            or stamp_ns > self._last_reacquire_stamp_ns
        )
        if (
            bbox_iou < self.config.minimum_reacquire_bbox_iou
            or position_delta
            > self.config.maximum_reacquire_position_delta_m
            or not stamp_increasing
        ):
            self.reacquire_count = 0
            self._last_reacquire_stamp_ns = None
            self.failure_reason = (
                'candidate does not match frozen image/3D geometry'
            )
            return self._snapshot()

        self.reacquire_count += 1
        self._last_reacquire_stamp_ns = stamp_ns
        self.reacquired_target = deepcopy(target)
        self.failure_reason = ''
        if self.reacquire_count >= self.config.required_reacquire_frames:
            self._transition(
                FLOW_PASS_NO_LIGHT_EXPECTED,
                'same_target_reacquired',
                stamp_ns,
            )
        return self._snapshot()

    def snapshot(self):
        self._expire_postcheck_if_needed()
        return self._snapshot()

    def abort(self, reason):
        self.failure_reason = str(reason).strip() or 'aborted'
        self._transition(FAILED, 'abort')
        return self._snapshot()
