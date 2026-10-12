"""Display-only layout helpers for the Qwen semantic debug image."""

from dataclasses import dataclass, replace
import math

import cv2
import numpy as np


PENDING = 'PENDING'
FRESH = 'FRESH'
CACHED = 'CACHED'
NONE = 'NONE'
STALE = 'STALE'
ERROR = 'ERROR'
DISPLAY_STATUSES = (PENDING, FRESH, CACHED, NONE, STALE, ERROR)
LIT_CONTEXT_RATIO = 0.45
LIT_RED_DELTA_THRESHOLD = 12.0
LIT_RED_FRACTION_THRESHOLD = 0.055
LIT_UNCERTAIN_DELTA_THRESHOLD = 8.5
LIT_UNCERTAIN_FRACTION_THRESHOLD = 0.04


@dataclass(frozen=True)
class SemanticResultView:
    detection_id: str
    label: str
    latency_ms: float
    error: str
    received_monotonic: float


@dataclass(frozen=True)
class PanelEntry:
    display_id: str
    detection_id: str
    semantic_cache_key: str
    bbox: tuple
    detector_confidence: float
    depth_valid: bool
    x_m: float
    y_m: float
    z_m: float
    semantic: str
    semantic_status: str
    qwen_latency_ms: float
    active: bool = True
    lit_state: bool | None = None


@dataclass(frozen=True)
class SystemStatus:
    qwen_loaded: bool
    semantic_fps: float
    last_latency_ms: float
    cache_hits: int
    cache_misses: int
    queue_length: int
    dropped_tasks: int
    mode_label: str = 'CABIN'


class StableDisplayRegistry:
    """Assign Bxx IDs, preserving them when an upstream cache key rotates."""

    def __init__(self, stale_timeout_sec=1.5, reassociation_timeout_sec=30.0):
        self.stale_timeout_sec = float(stale_timeout_sec)
        self.reassociation_timeout_sec = max(
            self.stale_timeout_sec,
            float(reassociation_timeout_sec),
        )
        self._next_number = 1
        self._records = {}

    @staticmethod
    def _bbox_iou(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(
            0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(
            0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    @staticmethod
    def _center_distance(first, second):
        first_center = (
            (first[0] + first[2]) * 0.5,
            (first[1] + first[3]) * 0.5,
        )
        second_center = (
            (second[0] + second[2]) * 0.5,
            (second[1] + second[3]) * 0.5,
        )
        return math.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )

    def _migrate_matching_record(self, cache_key, entry, now):
        """Move a recent display record to a rotated cache key by geometry."""
        candidates = []
        for previous_key, record in self._records.items():
            previous_entry = record['entry']
            if previous_entry is None:
                continue
            age = float(now) - record['last_seen']
            if age > self.reassociation_timeout_sec:
                continue
            iou = self._bbox_iou(previous_entry.bbox, entry.bbox)
            distance = self._center_distance(previous_entry.bbox, entry.bbox)
            previous_width = max(
                1.0, previous_entry.bbox[2] - previous_entry.bbox[0])
            previous_height = max(
                1.0, previous_entry.bbox[3] - previous_entry.bbox[1])
            distance_limit = 0.35 * max(previous_width, previous_height)
            if iou >= 0.55 or (iou >= 0.25 and distance <= distance_limit):
                candidates.append((iou, -distance, previous_key, record))
        if not candidates:
            return None
        _, _, previous_key, record = max(candidates, key=lambda item: item[:2])
        del self._records[previous_key]
        self._records[cache_key] = record
        return record

    def update(self, cache_key, entry, now):
        record = self._records.get(cache_key)
        if record is None:
            record = self._migrate_matching_record(cache_key, entry, now)
        if record is None:
            display_id = 'B{:02d}'.format(self._next_number)
            self._next_number += 1
            record = {
                'display_id': display_id,
                'last_seen': float(now),
                'entry': None,
            }
            self._records[cache_key] = record
        record['last_seen'] = float(now)
        record['entry'] = replace(entry, display_id=record['display_id'], active=True)
        return record['entry']

    def frame_entries(self, active_cache_keys, now):
        active_cache_keys = set(active_cache_keys)
        entries = []
        expired = []
        for cache_key, record in self._records.items():
            entry = record['entry']
            if entry is None:
                continue
            if cache_key in active_cache_keys:
                entries.append(entry)
                continue
            age = float(now) - record['last_seen']
            if age <= self.stale_timeout_sec:
                entries.append(replace(
                    entry,
                    semantic='',
                    semantic_status=STALE,
                    qwen_latency_ms=math.nan,
                    active=False,
                ))
            elif age > self.reassociation_timeout_sec:
                expired.append(cache_key)
        for cache_key in expired:
            del self._records[cache_key]
        return sorted(entries, key=lambda item: int(item.display_id[1:]))

    def display_id_for(self, cache_key):
        record = self._records.get(cache_key)
        return record['display_id'] if record else None


def semantic_display_state(
    cache_label,
    result,
    detection_id,
    semantic_enabled,
    worker_available,
    cache_ttl_sec,
    fresh_duration_sec,
    now,
    hard_cache_ttl_sec=None,
    consecutive_failures=0,
    maximum_consecutive_failures=3,
    association_valid=True,
    return_reason=False,
):
    """Resolve display state with soft refresh and a separate hard expiry."""

    def resolved(semantic, status, latency, reason):
        values = (semantic, status, latency)
        return values + (reason,) if return_reason else values

    hard_ttl = (
        max(float(cache_ttl_sec), float(hard_cache_ttl_sec))
        if hard_cache_ttl_sec is not None else float(cache_ttl_sec)
    )
    if not semantic_enabled:
        return resolved('', ERROR, math.nan, 'semantic_disabled')
    if result is None:
        if cache_label == 'PROCESSING' and worker_available:
            return resolved('', PENDING, math.nan, 'new_target_processing')
        return resolved(
            '',
            ERROR if not worker_available else PENDING,
            math.nan,
            'worker_unavailable_no_cache' if not worker_available
            else 'waiting_for_result',
        )
    if not association_valid:
        return resolved('', STALE, math.nan, 'association_failure')
    latency = float(result.latency_ms)
    if result.error or result.label == 'UNKNOWN':
        return resolved(
            '',
            ERROR,
            latency,
            'consecutive_inference_failures'
            if consecutive_failures >= int(maximum_consecutive_failures)
            else 'inference_error',
        )
    if result.label == 'none':
        return resolved('', NONE, latency, 'semantic_none')
    age = max(0.0, float(now) - float(result.received_monotonic))
    if age >= hard_ttl:
        return resolved('', STALE, math.nan, 'hard_ttl_expired')
    if age <= float(fresh_duration_sec):
        return resolved(result.label, FRESH, latency, 'fresh_result')
    if not worker_available:
        return resolved(
            result.label,
            CACHED,
            latency,
            'worker_unavailable_cached',
        )
    if age >= float(cache_ttl_sec):
        return resolved(
            result.label,
            CACHED,
            latency,
            'soft_ttl_refreshing',
        )
    return resolved(result.label, CACHED, latency, 'cached_result')


def _clip_box(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
    return x1, y1, x2, y2


def red_local_contrast_features(rgb_image, box, context_ratio=LIT_CONTEXT_RATIO):
    """Exact red feature extraction from the legacy lit_v1 implementation."""
    image_array = np.asarray(rgb_image, dtype=np.float32)
    if image_array.ndim != 3 or image_array.shape[2] < 3:
        raise ValueError('red_local_contrast_v1 requires an RGB image')
    image_h, image_w = image_array.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, image_w, image_h)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    ix1 = max(0, int(round(x1)))
    iy1 = max(0, int(round(y1)))
    ix2 = min(image_w, max(ix1 + 1, int(round(x2))))
    iy2 = min(image_h, max(iy1 + 1, int(round(y2))))
    ex1 = max(0, int(round(x1 - context_ratio * box_w)))
    ey1 = max(0, int(round(y1 - context_ratio * box_h)))
    ex2 = min(image_w, int(round(x2 + context_ratio * box_w)))
    ey2 = min(image_h, int(round(y2 + context_ratio * box_h)))

    inside = image_array[iy1:iy2, ix1:ix2]
    expanded = image_array[ey1:ey2, ex1:ex2]
    if inside.size == 0 or expanded.size == 0:
        raise ValueError('Empty illumination crop for box {!r}'.format(box))

    red = inside[..., 0]
    green = inside[..., 1]
    blue = inside[..., 2]
    red_excess = red - 0.5 * (green + blue)
    red_mask = (
        (red > 45.0)
        & (red > 1.16 * green)
        & (red > 1.10 * blue)
    )

    outer_red = expanded[..., 0]
    outer_green = expanded[..., 1]
    outer_blue = expanded[..., 2]
    outer_excess = outer_red - 0.5 * (outer_green + outer_blue)
    outer_mask = np.ones(expanded.shape[:2], dtype=bool)
    outer_mask[
        max(0, iy1 - ey1):min(expanded.shape[0], iy2 - ey1),
        max(0, ix1 - ex1):min(expanded.shape[1], ix2 - ex1),
    ] = False
    context_values = outer_excess[outer_mask]
    if context_values.size == 0:
        context_values = outer_excess.reshape(-1)

    inside_p90 = float(np.quantile(red_excess, 0.90))
    context_p90 = float(np.quantile(context_values, 0.90))
    red_fraction = float(np.mean(red_mask))
    return {
        'inside_red_excess_p90': round(inside_p90, 4),
        'context_red_excess_p90': round(context_p90, 4),
        'red_delta_p90': round(inside_p90 - context_p90, 4),
        'red_fraction': round(red_fraction, 6),
        'context_ratio': context_ratio,
    }


def illumination_probability(red_delta, red_fraction):
    """Exact legacy confidence transform; not used as a control signal."""
    signal = (
        (float(red_delta) - 10.0) / 3.5
        + (float(red_fraction) - 0.045) / 0.04
    )
    signal = max(-20.0, min(20.0, signal))
    return 1.0 / (1.0 + math.exp(-signal))


def red_local_contrast_v1(rgb_image, box):
    """Return the legacy lit/uncertain/unlit result for one current bbox."""
    features = red_local_contrast_features(
        rgb_image,
        box,
        LIT_CONTEXT_RATIO,
    )
    red_delta = features['red_delta_p90']
    red_fraction = features['red_fraction']
    probability = illumination_probability(red_delta, red_fraction)

    if (
        red_delta >= LIT_RED_DELTA_THRESHOLD
        and red_fraction >= LIT_RED_FRACTION_THRESHOLD
    ):
        state = 'lit'
        confidence = max(0.70, probability)
    elif (
        red_delta >= LIT_UNCERTAIN_DELTA_THRESHOLD
        and red_fraction >= LIT_UNCERTAIN_FRACTION_THRESHOLD
    ):
        state = 'uncertain'
        confidence = 0.50 + 0.20 * abs(probability - 0.50)
    else:
        state = 'unlit'
        confidence = max(0.55, 1.0 - probability)

    return {
        'eligible': True,
        'state': state,
        'confidence': round(min(0.999, confidence), 4),
        'lit_probability': round(probability, 4),
        'method': 'red_local_contrast_v1',
        'features': features,
    }


def _overlap(first, second):
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _clamp(value, low, high):
    return max(low, min(high, value))


RED = (0, 0, 255)
ORANGE = (0, 165, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
VALID_SEMANTIC_LABELS = {
    'alarm', 'b1', '4', '3', '2', '1', 'open', 'close', 'up', 'down', 'g',
}


def _intersection_area(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _inside_image(region, width, height):
    return (
        region[0] >= 0 and region[1] >= 0
        and region[2] <= width and region[3] <= height
    )


def _semantic_is_valid(entry):
    normalized_label = str(entry.semantic or '').strip().lower()
    return (
        entry.active
        and entry.semantic_status in (FRESH, CACHED)
        and normalized_label in VALID_SEMANTIC_LABELS
    )


def _entry_style(entry):
    if entry.active and entry.semantic_status == PENDING:
        return WHITE, 'PROCESSING', False, False, True
    valid = _semantic_is_valid(entry)
    lit = entry.active and entry.lit_state is True
    result = str(entry.semantic).strip() if valid else 'UNKNOWN'
    if lit:
        color = ORANGE
    elif valid:
        color = GREEN
    else:
        color = RED
    if lit and valid:
        result = '{} [LIT]'.format(result)
    return color, result, valid, lit, False


def _entry_lines(entry, result):
    xyz = (
        'XYZ: {:.3f}, {:.3f}, {:.3f} m'.format(
            entry.x_m, entry.y_m, entry.z_m)
        if entry.depth_valid else 'XYZ: INVALID'
    )
    return [
        'Conf: {:.2f}'.format(entry.detector_confidence),
        xyz,
        'Result: {}'.format(result),
    ]


def _text_block_geometry(lines, font, font_scale, thickness, padding):
    sizes = [
        cv2.getTextSize(line, font, font_scale, thickness)
        for line in lines
    ]
    line_height = max(size[0][1] + size[1] for size in sizes) + 3
    width = max(size[0][0] for size in sizes) + 2 * padding
    height = len(lines) * line_height + 2 * padding
    return width, height, line_height


def _raw_text_candidates(bbox, block_width, block_height, gap):
    left, top, right, bottom = bbox
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    return [
        (right + gap, center_y - block_height * 0.5),
        (left - gap - block_width, center_y - block_height * 0.5),
        (center_x - block_width * 0.5, top - gap - block_height),
        (center_x - block_width * 0.5, bottom + gap),
    ]


def _candidate_region(x, y, block_width, block_height):
    left = int(round(x))
    top = int(round(y))
    return (left, top, left + block_width, top + block_height)


def _place_text_block(
    image_shape,
    bbox,
    block_width,
    block_height,
    all_boxes,
    used_regions,
    gap=8,
):
    height, width = image_shape[:2]
    primary = [
        _candidate_region(x, y, block_width, block_height)
        for x, y in _raw_text_candidates(
            bbox, block_width, block_height, gap)
    ]
    for region in primary:
        if not _inside_image(region, width, height):
            continue
        if any(_overlap(region, box) for box in all_boxes):
            continue
        if any(_overlap(region, used) for used in used_regions):
            continue
        return region

    # Search along each preferred side while keeping the block outside the
    # current bbox. This handles image edges and vertically stacked buttons.
    expanded = []
    vertical_step = max(8, block_height // 3)
    horizontal_step = max(8, block_width // 4)
    for offset in range(0, height + block_height, vertical_step):
        shifts = (0,) if offset == 0 else (-offset, offset)
        for shift in shifts:
            expanded.append(_candidate_region(
                bbox[2] + gap,
                (bbox[1] + bbox[3] - block_height) * 0.5 + shift,
                block_width,
                block_height,
            ))
            expanded.append(_candidate_region(
                bbox[0] - gap - block_width,
                (bbox[1] + bbox[3] - block_height) * 0.5 + shift,
                block_width,
                block_height,
            ))
    for offset in range(0, width + block_width, horizontal_step):
        shifts = (0,) if offset == 0 else (-offset, offset)
        for shift in shifts:
            expanded.append(_candidate_region(
                (bbox[0] + bbox[2] - block_width) * 0.5 + shift,
                bbox[1] - gap - block_height,
                block_width,
                block_height,
            ))
            expanded.append(_candidate_region(
                (bbox[0] + bbox[2] - block_width) * 0.5 + shift,
                bbox[3] + gap,
                block_width,
                block_height,
            ))

    bounded = []
    seen = set()
    for region in primary + expanded:
        x = int(_clamp(region[0], 0, max(0, width - block_width)))
        y = int(_clamp(region[1], 0, max(0, height - block_height)))
        candidate = (x, y, x + block_width, y + block_height)
        if candidate in seen or _overlap(candidate, bbox):
            continue
        seen.add(candidate)
        bounded.append(candidate)
        if (
            not any(_overlap(candidate, box) for box in all_boxes)
            and not any(_overlap(candidate, used) for used in used_regions)
        ):
            return candidate

    # If the scene is too dense, choose the bounded region with the smallest
    # total collision area. The current button is never eligible for overlap.
    if not bounded:
        return None
    return min(
        bounded,
        key=lambda region: (
            sum(_intersection_area(region, box) for box in all_boxes)
            + sum(_intersection_area(region, used) for used in used_regions),
            region[0],
            region[1],
        ),
    )


def render_semantic_display(rgb_image, entries, system_status=None):
    """Render only Conf/XYZ/Result beside each active bbox."""
    rgb_canvas = np.array(rgb_image, copy=True)
    canvas = cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    # ── Mode label in top-left ──
    if system_status is not None:
        mode_text = 'Mode: {}'.format(system_status.mode_label)
        mode_font = cv2.FONT_HERSHEY_SIMPLEX
        mode_scale = max(0.65, min(width / 1920.0, height / 1080.0) * 0.75)
        mode_thickness = max(1, int(round(1.5 * max(0.75, min(width / 1920.0, height / 1080.0)))))
        (mode_w, mode_h), mode_baseline = cv2.getTextSize(
            mode_text, mode_font, mode_scale, mode_thickness)
        mode_pad = max(6, int(round(8 * max(0.75, min(width / 1920.0, height / 1080.0)))))
        cv2.rectangle(canvas, (0, 0),
                      (mode_w + 2 * mode_pad, mode_h + 2 * mode_pad + mode_baseline),
                      (0, 0, 0), -1)
        cv2.putText(canvas, mode_text,
                    (mode_pad, mode_h + mode_pad),
                    mode_font, mode_scale, (255, 255, 255), mode_thickness, cv2.LINE_AA)
    active_entries = [entry for entry in entries if entry.active]
    active_boxes = [
        tuple(int(round(value)) for value in entry.bbox)
        for entry in active_entries
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    ui_scale = max(0.75, min(width / 1920.0, height / 1080.0))
    font_scale = 0.54 * ui_scale
    thickness = 1
    bbox_thickness = max(2, int(round(2 * ui_scale)))
    padding = max(4, int(round(5 * ui_scale)))

    text_regions = []
    rendered_lines = []
    box_colors = []
    recognized_count = 0
    unknown_count = 0
    orange_count = 0
    processing_count = 0
    unplaced_count = 0

    for entry, bbox in zip(active_entries, active_boxes):
        color, result, valid, lit, processing = _entry_style(entry)
        lines = _entry_lines(entry, result)
        block_width, block_height, line_height = _text_block_geometry(
            lines, font, font_scale, thickness, padding)
        region = _place_text_block(
            canvas.shape,
            bbox,
            block_width,
            block_height,
            active_boxes,
            text_regions,
            gap=max(6, int(round(8 * ui_scale))),
        )
        cv2.rectangle(
            canvas,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            bbox_thickness,
        )
        box_colors.append(color)
        recognized_count += int(valid and not lit)
        unknown_count += int(not valid and not processing)
        orange_count += int(lit)
        processing_count += int(processing)
        if region is None:
            unplaced_count += 1
            continue
        cv2.rectangle(
            canvas,
            (region[0], region[1]),
            (region[2], region[3]),
            (0, 0, 0),
            -1,
        )
        cv2.rectangle(
            canvas,
            (region[0], region[1]),
            (region[2], region[3]),
            color,
            1,
        )
        for line_index, line in enumerate(lines):
            baseline = (
                region[1] + padding
                + (line_index + 1) * line_height - 3
            )
            cv2.putText(
                canvas,
                line,
                (region[0] + padding, baseline),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        text_regions.append(region)
        rendered_lines.append(tuple(lines))

    text_overlap = any(
        _overlap(text_regions[first], text_regions[second])
        for first in range(len(text_regions))
        for second in range(first + 1, len(text_regions))
    )
    button_content_occluded = any(
        _overlap(region, bbox)
        for region in text_regions
        for bbox in active_boxes
    )
    text_outside_image = any(
        not _inside_image(region, width, height)
        for region in text_regions
    )
    diagnostics = {
        'panel_width': 0,
        'output_width': width,
        'columns': 0,
        'compact': False,
        'text_regions': text_regions,
        'rendered_lines': rendered_lines,
        'box_colors': box_colors,
        'recognized_count': recognized_count,
        'orange_count': orange_count,
        'unknown_count': unknown_count,
        'processing_count': processing_count,
        'unplaced_count': unplaced_count,
        'text_overlap': text_overlap,
        'text_outside_image': text_outside_image,
        'button_content_occluded': button_content_occluded,
        'panel_overflow': False,
    }
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), diagnostics
