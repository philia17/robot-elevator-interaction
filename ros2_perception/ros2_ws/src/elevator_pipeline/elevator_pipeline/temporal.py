"""Reusable bounded-history temporal stabilization for 2D targets."""

from collections import deque
from dataclasses import dataclass
from math import hypot


def bbox_iou(first, second):
    first_box = _bbox(first)
    second_box = _bbox(second)
    intersection_left = max(first_box[0], second_box[0])
    intersection_top = max(first_box[1], second_box[1])
    intersection_right = min(first_box[2], second_box[2])
    intersection_bottom = min(first_box[3], second_box[3])
    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_box[2] - first_box[0]) * max(
        0.0, first_box[3] - first_box[1]
    )
    second_area = max(0.0, second_box[2] - second_box[0]) * max(
        0.0, second_box[3] - second_box[1]
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _bbox(item):
    bbox = getattr(item, 'bbox', None)
    if bbox is not None:
        return bbox
    return (item.x_min, item.y_min, item.x_max, item.y_max)


def center_distance(first, second):
    return hypot(
        first.center_x - second.center_x,
        first.center_y - second.center_y,
    )


@dataclass(frozen=True)
class TrackedTarget:
    valid: bool
    request_id: str
    building_id: str
    requested_floor: str
    effective_label: str
    button: object
    temporal_confidence: float
    status: str
    failure_reason: str
    confirmation_count: int


class TemporalTracker:
    def __init__(
            self,
            required_confirmations=2,
            history_size=5,
            minimum_iou=0.5,
            maximum_center_distance=20.0,
            minimum_candidate_confidence=0.0):
        if int(required_confirmations) <= 0:
            raise ValueError('required_confirmations must be positive')
        if int(history_size) < int(required_confirmations):
            raise ValueError('history_size must cover required_confirmations')
        self.required_confirmations = int(required_confirmations)
        self.minimum_iou = float(minimum_iou)
        self.maximum_center_distance = float(maximum_center_distance)
        self.minimum_candidate_confidence = float(minimum_candidate_confidence)
        self._history = deque(maxlen=int(history_size))
        self._task_key = None

    def reset(self):
        self._history.clear()
        self._task_key = None

    def _unstable(self, candidate, status, reason):
        return TrackedTarget(
            valid=False,
            request_id=candidate.request_id,
            building_id=candidate.building_id,
            requested_floor=candidate.requested_floor,
            effective_label=candidate.effective_label,
            button=candidate.button,
            temporal_confidence=0.0,
            status=status,
            failure_reason=reason,
            confirmation_count=len(self._history),
        )

    def update(self, candidate):
        task_key = (
            candidate.request_id,
            candidate.building_id,
            candidate.requested_floor,
        )
        if self._task_key != task_key:
            self._history.clear()
            self._task_key = task_key
        if (
                not candidate.found
                or candidate.button is None
                or candidate.candidate_confidence < self.minimum_candidate_confidence):
            self._history.clear()
            return self._unstable(candidate, 'SEARCHING', candidate.reason)

        if self._history:
            previous = self._history[-1]
            consistent = (
                previous.effective_label == candidate.effective_label
                and previous.button.detection_id == candidate.button.detection_id
                and bbox_iou(previous.button, candidate.button) >= self.minimum_iou
                and center_distance(previous.button, candidate.button)
                <= self.maximum_center_distance
            )
            if not consistent:
                self._history.clear()
        self._history.append(candidate)

        if len(self._history) < self.required_confirmations:
            return self._unstable(
                candidate,
                'TRACKING',
                'awaiting_confirmations',
            )
        recent = list(self._history)[-self.required_confirmations:]
        confidence = sum(item.candidate_confidence for item in recent) / len(recent)
        return TrackedTarget(
            valid=True,
            request_id=candidate.request_id,
            building_id=candidate.building_id,
            requested_floor=candidate.requested_floor,
            effective_label=candidate.effective_label,
            button=candidate.button,
            temporal_confidence=confidence,
            status='STABLE',
            failure_reason='',
            confirmation_count=len(recent),
        )
