"""Pure fail-closed target selection for M0 no-motion validation."""

from dataclasses import dataclass
import math
import re
import statistics
from collections.abc import Mapping


FRESH = 'FRESH'
CACHED = 'CACHED'
NONE = 'NONE'
SEMANTIC_GRACE = 'SEMANTIC_GRACE'
ALLOWED_TARGET_STATUSES = (FRESH, CACHED)
COMPLETE_SET_STATUSES = (FRESH, CACHED, NONE)
SEMANTIC_SCOPE_STRICT_ALL_VISIBLE = 'STRICT_ALL_VISIBLE'
SEMANTIC_SCOPE_TARGET_ONLY_DIAGNOSTIC = 'TARGET_ONLY_DIAGNOSTIC'


def _exact_label(value):
    return str(value).strip().lower()


def _selected_label(value):
    label = _exact_label(value)
    if label in {'', '?', 'unknown', 'none', 'null', 'n/a', 'nan', 'unrecognized'}:
        return ''
    if label in {'open', 'close', 'up', 'down', 'alarm', 'help', 'stop'}:
        return label
    return label if re.fullmatch(r'(?:-?\d+[a-z]?|[a-z]{1,3}\d*)', label) else ''


def _finite_xyz(item):
    return all(math.isfinite(float(item[key])) for key in ('x_m', 'y_m', 'z_m'))


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


def _position_spread(points):
    if len(points) < 2:
        return 0.0
    return max(
        math.dist(points[first], points[second])
        for first in range(len(points))
        for second in range(first + 1, len(points))
    )


@dataclass(frozen=True)
class M0TargetConfig:
    target_label: str = '1'
    required_consecutive_frames: int = 5
    maximum_semantic_age_sec: float = 2.0
    maximum_non_target_semantic_age_sec: float = 30.0
    maximum_frame_gap_sec: float = 0.50
    minimum_detector_confidence: float = 0.50
    minimum_depth_valid_ratio: float = 0.50
    minimum_bbox_iou: float = 0.65
    maximum_position_spread_m: float = 0.015
    # Diagnostic target-only mode may briefly retain an already-confirmed
    # window while the selected target's semantic entry is missing or
    # REVALIDATING/PENDING.  The formal default never enters grace, and the
    # TTL is explicitly bounded so stale semantics can never READY/freeze.
    target_semantic_grace_ttl_sec: float = 10.0
    # Formal M0 freezes require every visible semantic label to be settled.
    # A separately named diagnostic observer may set this false so transient
    # non-target labels cannot prevent a read-only, target-only preview.
    require_complete_semantic_set: bool = True

    def __post_init__(self):
        if not self.target_label.strip():
            raise ValueError('target_label must not be empty')
        if self.required_consecutive_frames < 2:
            raise ValueError('required_consecutive_frames must be at least 2')
        if self.maximum_semantic_age_sec <= 0.0:
            raise ValueError('maximum_semantic_age_sec must be positive')
        if self.maximum_non_target_semantic_age_sec <= 0.0:
            raise ValueError(
                'maximum_non_target_semantic_age_sec must be positive'
            )
        if self.maximum_frame_gap_sec <= 0.0:
            raise ValueError('maximum_frame_gap_sec must be positive')
        if not 0.0 <= self.minimum_detector_confidence <= 1.0:
            raise ValueError('minimum_detector_confidence must be in [0, 1]')
        if not 0.0 <= self.minimum_depth_valid_ratio <= 1.0:
            raise ValueError('minimum_depth_valid_ratio must be in [0, 1]')
        if not 0.0 <= self.minimum_bbox_iou <= 1.0:
            raise ValueError('minimum_bbox_iou must be in [0, 1]')
        if self.maximum_position_spread_m <= 0.0:
            raise ValueError('maximum_position_spread_m must be positive')
        if not 0.0 < self.target_semantic_grace_ttl_sec <= 10.0:
            raise ValueError(
                'target_semantic_grace_ttl_sec must be in (0, 10]'
            )
        if not isinstance(self.require_complete_semantic_set, bool):
            raise ValueError('require_complete_semantic_set must be a bool')


class M0TargetTracker:
    """Require one fresh-enough semantic target and stable camera-frame XYZ."""

    def __init__(self, config=None):
        self.config = config or M0TargetConfig()
        self._last_stamp_ns = None
        self._last_bbox = None
        self._positions = []
        self._last_complete_semantic_stamp_ns = None
        # None preserves the legacy configured-label observer. An explicit
        # clear is a real empty selection and must never fall back to label 1.
        self._selection = None

    @property
    def target_label(self):
        if self._selection is not None:
            return self._selection['label']
        return self.config.target_label

    def select_target(self, selection):
        """Change perception-only identity and discard all previous READY evidence."""
        self.reset()
        self._selection = {'label': ''}
        if not isinstance(selection, Mapping):
            raise ValueError('target selection must be an object')
        raw_label = str(selection.get('label') or '').strip()
        if not raw_label:
            return
        label = _selected_label(raw_label)
        if not label:
            raise ValueError('selected button label is unknown')
        for key in ('selection_id', 'detection_id'):
            if not str(selection.get(key) or '').strip():
                raise ValueError('target selection missing ' + key)
        after_stamp_ns = int(selection.get('after_stamp_ns') or 0)
        if after_stamp_ns <= 0:
            raise ValueError('target selection requires its source timestamp')
        self._selection = {
            'selection_id': str(selection['selection_id']), 'label': label,
            'detection_id': str(selection['detection_id']),
            'semantic_track_id': str(selection.get('semantic_track_id') or ''),
            'after_stamp_ns': after_stamp_ns,
        }

    def _is_target(self, semantic):
        if _exact_label(semantic.get('label', '')) != _exact_label(self.target_label):
            return False
        if self._selection is None:
            return True
        # Detector IDs are only frame-local. A selected, newly observed
        # target requires the stable semantic track, not the old numeric ID.
        track = str(self._selection.get('semantic_track_id') or '')
        return bool(track) and str(semantic.get('track_id') or '') == track

    def reset(self):
        self._last_stamp_ns = None
        self._last_bbox = None
        self._positions = []
        self._last_complete_semantic_stamp_ns = None

    def _valid_observation(self, observation):
        return (
            bool(observation.get('depth_valid'))
            and _finite_xyz(observation)
            and float(observation.get('detector_confidence', 0.0))
            >= self.config.minimum_detector_confidence
            and float(observation.get('depth_valid_ratio', 0.0))
            >= self.config.minimum_depth_valid_ratio
        )

    def _decision(
        self,
        status,
        failure_reason,
        source_stamp_ns,
        frame_id,
        matches,
        selected=None,
        non_target_semantic_incomplete=(),
        semantic_grace_active=False,
        semantic_grace_reason='',
    ):
        spread = _position_spread(self._positions)
        ready = status == 'READY'
        grace_remaining_sec = None
        if (
            semantic_grace_active
            and self._last_complete_semantic_stamp_ns is not None
        ):
            elapsed_sec = (
                int(source_stamp_ns) - self._last_complete_semantic_stamp_ns
            ) / 1e9
            grace_remaining_sec = max(
                0.0,
                self.config.target_semantic_grace_ttl_sec - elapsed_sec,
            )
        return {
            'schema': 'm0_target_3d_v1',
            'mode': 'NO_MOTION',
            'status': status,
            'failure_reason': failure_reason,
            'perception_ready': ready,
            'target_label': self.target_label,
            'selection_id': (self._selection or {}).get('selection_id', ''),
            'effective_label': (
                selected['label'] if selected is not None else ''
            ),
            'source_stamp_ns': int(source_stamp_ns),
            'frame_id': str(frame_id),
            'matching_label_count': len(matches),
            'required_confirmations': (
                self.config.required_consecutive_frames
            ),
            'confirmation_count': len(self._positions),
            'position_spread_m': spread,
            'selected': selected,
            # This is provenance only.  In TARGET_ONLY_DIAGNOSTIC mode it
            # explains why a formal all-visible freeze would have rejected
            # the frame without weakening the selected target's gates.
            'semantic_scope': (
                SEMANTIC_SCOPE_STRICT_ALL_VISIBLE
                if self.config.require_complete_semantic_set
                else SEMANTIC_SCOPE_TARGET_ONLY_DIAGNOSTIC
            ),
            'non_target_semantic_incomplete': list(
                non_target_semantic_incomplete
            ),
            'semantic_grace_active': bool(semantic_grace_active),
            'semantic_grace_reason': str(semantic_grace_reason),
            'semantic_grace_ttl_sec': (
                self.config.target_semantic_grace_ttl_sec
            ),
            'semantic_grace_remaining_sec': grace_remaining_sec,
            'last_complete_semantic_stamp_ns': (
                int(self._last_complete_semantic_stamp_ns)
                if self._last_complete_semantic_stamp_ns is not None
                else 0
            ),
            'motion_authorized': False,
        }

    def _semantic_issue(self, item, target):
        """Return diagnostic provenance for an incomplete semantic item."""
        status = item.get('semantic_status')
        reason = ''
        if status not in COMPLETE_SET_STATUSES:
            reason = 'status_not_complete'
        elif status in ALLOWED_TARGET_STATUSES:
            age = float(item.get('semantic_age_sec', math.nan))
            maximum_age = (
                self.config.maximum_semantic_age_sec
                if _exact_label(item.get('label', '')) == target
                else self.config.maximum_non_target_semantic_age_sec
            )
            if not math.isfinite(age):
                reason = 'semantic_age_not_finite'
            elif age > maximum_age:
                reason = 'semantic_age_exceeded'
        if not reason:
            return None
        return {
            'detection_id': str(item.get('detection_id', '')),
            'label': str(item.get('label', '')),
            'semantic_status': str(status),
            'semantic_age_sec': float(item.get('semantic_age_sec', math.nan)),
            'reason': reason,
        }

    def _grace_hold(
        self,
        observation,
        source_stamp_ns,
        frame_id,
        matches,
        non_target_semantic_incomplete,
        grace_reason,
    ):
        """Retain a confirmed window across one bounded semantic gap.

        Grace is a diagnostic target-only allowance: the selected target's
        semantic entry is missing or not complete this frame, so the frame is
        never READY and never contributes a new confirmation.  Every
        observation-side gate still applies (depth/confidence/bbox/position/
        frame gap), and the retained window is only usable again once a fresh,
        complete semantic target frame arrives.
        """
        if (
            self._last_complete_semantic_stamp_ns is None
            or self._last_bbox is None
            or not self._positions
        ):
            return None
        stamp_ns = int(source_stamp_ns)
        if stamp_ns <= self._last_complete_semantic_stamp_ns:
            return None
        if (
            stamp_ns - self._last_complete_semantic_stamp_ns
        ) / 1e9 > self.config.target_semantic_grace_ttl_sec:
            return None
        if not self._valid_observation(observation):
            return None
        bbox = tuple(
            float(observation[key])
            for key in ('x_min', 'y_min', 'x_max', 'y_max')
        )
        if _bbox_iou(bbox, self._last_bbox) < self.config.minimum_bbox_iou:
            return None
        position = tuple(
            float(observation[key]) for key in ('x_m', 'y_m', 'z_m')
        )
        if (
            _position_spread(self._positions + [position])
            > self.config.maximum_position_spread_m
        ):
            return None
        if (
            self._last_stamp_ns is None
            or stamp_ns <= self._last_stamp_ns
            or (stamp_ns - self._last_stamp_ns) / 1e9
            > self.config.maximum_frame_gap_sec
        ):
            return None

        # Keep the confirmation window intact; do not append a grace frame.
        self._last_stamp_ns = stamp_ns
        self._last_bbox = bbox
        selected = {
            'detection_id': str(observation.get('detection_id', '')),
            'track_id': '',
            'label': '',
            'semantic_status': '',
            'semantic_age_sec': math.nan,
            'pressable': False,
            'detector_confidence': float(
                observation.get('detector_confidence', 0.0)
            ),
            'depth_valid': bool(observation.get('depth_valid')),
            'depth_valid_ratio': float(
                observation.get('depth_valid_ratio', 0.0)
            ),
            'bbox': bbox,
            'camera_point_m': {
                'x': float(observation.get('x_m', math.nan)),
                'y': float(observation.get('y_m', math.nan)),
                'z': float(observation.get('z_m', math.nan)),
            },
            'surface_geometry': observation.get('surface_geometry'),
        }
        selected['camera_point_m'] = {
            axis: statistics.median(
                point[index] for point in self._positions
            )
            for index, axis in enumerate(('x', 'y', 'z'))
        }
        return self._decision(
            SEMANTIC_GRACE,
            'target semantic missing or revalidating; confirmed window '
            'retained within bounded grace (no READY/freeze)',
            stamp_ns,
            frame_id,
            matches,
            selected,
            non_target_semantic_incomplete,
            semantic_grace_active=True,
            semantic_grace_reason=grace_reason,
        )

    def _continuous_grace_candidates(self, observations):
        """Same-frame observations keeping the last target spatially stable."""
        candidates = []
        for item in observations:
            if not self._valid_observation(item):
                continue
            bbox = tuple(
                float(item[key])
                for key in ('x_min', 'y_min', 'x_max', 'y_max')
            )
            if (
                self._last_bbox is None
                or _bbox_iou(bbox, self._last_bbox)
                < self.config.minimum_bbox_iou
            ):
                continue
            position = tuple(
                float(item[key]) for key in ('x_m', 'y_m', 'z_m')
            )
            if (
                not self._positions
                or _position_spread(self._positions + [position])
                > self.config.maximum_position_spread_m
            ):
                continue
            candidates.append(item)
        return candidates

    def _grace_for_incomplete_target(
        self,
        semantic_item,
        observations_by_id,
        source_stamp_ns,
        frame_id,
        matches,
        non_target_semantic_incomplete,
    ):
        # The semantic entry still carries the exact target label but is not
        # complete (e.g. REVALIDATING).  Keep the same-stamp observation
        # join and the association gate; pressable is re-required on READY.
        if not bool(semantic_item.get('association_valid')):
            return None
        observation = observations_by_id.get(
            str(semantic_item.get('detection_id', ''))
        )
        if observation is None:
            return None
        return self._grace_hold(
            observation,
            source_stamp_ns,
            frame_id,
            matches,
            non_target_semantic_incomplete,
            'target_semantic_incomplete',
        )

    def _grace_for_missing_target(
        self,
        observations,
        source_stamp_ns,
        frame_id,
        matches,
        non_target_semantic_incomplete,
    ):
        candidates = self._continuous_grace_candidates(observations)
        if len(candidates) != 1:
            return None
        return self._grace_hold(
            candidates[0],
            source_stamp_ns,
            frame_id,
            matches,
            non_target_semantic_incomplete,
            'target_semantic_missing',
        )

    def update(self, observations, semantics, source_stamp_ns, frame_id):
        if self._selection is not None:
            reason = ''
            if not self.target_label:
                status, reason = 'NO_SELECTION', 'select a detected button'
            elif not self._selection.get('semantic_track_id'):
                status, reason = 'SELECTION_TRACK_REQUIRED', 'frame-local detection ID cannot identify a new target frame'
            elif int(source_stamp_ns) <= self._selection['after_stamp_ns']:
                status, reason = 'SELECTION_WAIT_FRESH_FRAME', 'selected target requires a newer source frame'
            if reason:
                self.reset()
                return self._decision(status, reason, source_stamp_ns, frame_id, [])
        observations_by_id = {
            str(item['detection_id']): item for item in observations
        }
        target = _exact_label(self.target_label)
        matches = [
            item
            for item in semantics
            if self._is_target(item)
        ]

        # Keep the formal default fail-closed over the whole visible set.  The
        # diagnostic mode is intentionally narrower: it still requires the
        # selected target itself to be complete, associated, pressable, and
        # geometrically stable, but ignores a transient *non-target* label.
        semantic_issues = [
            (item, self._semantic_issue(item, target))
            for item in semantics
        ]
        non_target_semantic_incomplete = [
            issue
            for item, issue in semantic_issues
            if issue is not None
            and not self._is_target(item)
        ]
        incomplete = [
            item
            for item, issue in semantic_issues
            if issue is not None
            and (
                self.config.require_complete_semantic_set
                or self._is_target(item)
            )
        ]
        if incomplete:
            if (
                not self.config.require_complete_semantic_set
                and len(matches) == 1
            ):
                grace = self._grace_for_incomplete_target(
                    matches[0],
                    observations_by_id,
                    source_stamp_ns,
                    frame_id,
                    matches,
                    non_target_semantic_incomplete,
                )
                if grace is not None:
                    return grace
            self.reset()
            return self._decision(
                'SEMANTIC_SET_INCOMPLETE',
                'one or more visible labels are pending, stale, errored, or too old',
                source_stamp_ns,
                frame_id,
                matches,
                non_target_semantic_incomplete=non_target_semantic_incomplete,
            )
        if not matches:
            if not self.config.require_complete_semantic_set and self._selection is None:
                grace = self._grace_for_missing_target(
                    observations,
                    source_stamp_ns,
                    frame_id,
                    matches,
                    non_target_semantic_incomplete,
                )
                if grace is not None:
                    return grace
            self.reset()
            return self._decision(
                'NOT_FOUND',
                'target label is not visible',
                source_stamp_ns,
                frame_id,
                matches,
                non_target_semantic_incomplete=non_target_semantic_incomplete,
            )
        if len(matches) != 1:
            self.reset()
            return self._decision(
                'AMBIGUOUS',
                'more than one exact target label is visible',
                source_stamp_ns,
                frame_id,
                matches,
                non_target_semantic_incomplete=non_target_semantic_incomplete,
            )

        semantic = matches[0]
        observation = observations_by_id.get(str(semantic['detection_id']))
        if observation is None:
            self.reset()
            return self._decision(
                'UNMATCHED_3D',
                'semantic detection has no same-stamp 3D observation',
                source_stamp_ns,
                frame_id,
                matches,
                non_target_semantic_incomplete=non_target_semantic_incomplete,
            )

        selected = {
            'detection_id': str(semantic['detection_id']),
            'track_id': str(semantic.get('track_id', '')),
            'label': str(semantic.get('label', '')),
            'semantic_status': str(semantic.get('semantic_status', '')),
            'semantic_age_sec': float(
                semantic.get('semantic_age_sec', math.nan)
            ),
            'pressable': bool(semantic.get('pressable')),
            'detector_confidence': float(
                observation.get('detector_confidence', 0.0)
            ),
            'depth_valid': bool(observation.get('depth_valid')),
            'depth_valid_ratio': float(
                observation.get('depth_valid_ratio', 0.0)
            ),
            'bbox': [
                float(observation[key])
                for key in ('x_min', 'y_min', 'x_max', 'y_max')
            ],
            'camera_point_m': {
                'x': float(observation.get('x_m', math.nan)),
                'y': float(observation.get('y_m', math.nan)),
                'z': float(observation.get('z_m', math.nan)),
            },
            'surface_geometry': observation.get('surface_geometry'),
        }
        semantic_allowed = (
            semantic.get('semantic_status') in ALLOWED_TARGET_STATUSES
            and bool(semantic.get('pressable'))
            and bool(semantic.get('association_valid'))
        )
        if not semantic_allowed or not self._valid_observation(observation):
            self.reset()
            return self._decision(
                'INVALID_TARGET',
                'target failed semantic association or 3D quality gates',
                source_stamp_ns,
                frame_id,
                matches,
                selected,
                non_target_semantic_incomplete,
            )

        bbox = tuple(selected['bbox'])
        position = tuple(
            float(observation[key]) for key in ('x_m', 'y_m', 'z_m')
        )
        stamp_increasing = (
            self._last_stamp_ns is not None
            and int(source_stamp_ns) > self._last_stamp_ns
        )
        gap_allowed = (
            stamp_increasing
            and (int(source_stamp_ns) - self._last_stamp_ns) / 1e9
            <= self.config.maximum_frame_gap_sec
        )
        bbox_stable = (
            self._last_bbox is not None
            and _bbox_iou(bbox, self._last_bbox)
            >= self.config.minimum_bbox_iou
        )
        candidate_positions = self._positions + [position]
        position_stable = _position_spread(candidate_positions) <= (
            self.config.maximum_position_spread_m
        )

        # Legacy identity is anchored on the exact target label (single-match gate
        # above) plus spatial continuity, not on the detector/semantic ID:
        # the detector can alternate button_004/button_003 for the same
        # physical button, and that must not erase a stable confirmation
        # window.  A real bbox/XYZ jump still fails bbox_stable or
        # position_stable and restarts the count. Explicit selections also
        # require the exact stable semantic track in _is_target above.
        if gap_allowed and bbox_stable and position_stable:
            self._positions = candidate_positions[
                -self.config.required_consecutive_frames:
            ]
        else:
            self._positions = [position]
        self._last_complete_semantic_stamp_ns = int(source_stamp_ns)
        self._last_stamp_ns = int(source_stamp_ns)
        self._last_bbox = bbox

        status = (
            'READY'
            if len(self._positions)
            >= self.config.required_consecutive_frames
            else 'STABILIZING'
        )
        selected['camera_point_m'] = {
            axis: statistics.median(
                point[index] for point in self._positions
            )
            for index, axis in enumerate(('x', 'y', 'z'))
        }
        reason = '' if status == 'READY' else 'collecting stable target frames'
        return self._decision(
            status,
            reason,
            source_stamp_ns,
            frame_id,
            matches,
            selected,
            non_target_semantic_incomplete,
        )
