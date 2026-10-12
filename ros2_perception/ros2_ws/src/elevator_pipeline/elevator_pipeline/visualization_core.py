"""Testable overlay planning with detector-owned geometry."""

from dataclasses import dataclass

from .floor_map import normalize_label


@dataclass(frozen=True)
class OverlayItem:
    detection_id: str
    marker: str
    bbox: tuple
    lines: tuple
    color: tuple
    thickness: int


def _stable_order(detections):
    return sorted(
        detections,
        key=lambda item: (item.center_y, item.center_x, item.detection_id),
    )


def build_overlay_plan(
        detections,
        observations=None,
        candidate=None,
        target=None):
    """Build annotations without changing geometry or pipeline decisions."""

    observation_by_id = {
        item.detection_id: item for item in (observations or [])
    }
    candidate_id = ''
    if candidate is not None and bool(getattr(candidate, 'found', False)):
        candidate_id = getattr(getattr(candidate, 'button', None), 'detection_id', '')
    target_id = ''
    if target is not None and bool(getattr(target, 'valid', False)):
        target_id = getattr(getattr(target, 'button', None), 'detection_id', '')

    items = []
    for index, detection in enumerate(_stable_order(detections or [])):
        marker = chr(ord('A') + index) if index < 26 else '?'
        observation = observation_by_id.get(detection.detection_id)
        if observation is None:
            label = '?'
            button_type = getattr(detection, 'coarse_type', 'unknown') or 'unknown'
            pressable = 'unknown'
            semantic_confidence = -1.0
            overall_confidence = -1.0
        else:
            label = normalize_label(getattr(observation, 'label', '')) or '?'
            button_type = getattr(observation, 'button_type', '') or 'unknown'
            pressable = str(bool(getattr(observation, 'pressable', False))).lower()
            semantic_confidence = float(
                getattr(observation, 'semantic_confidence', -1.0)
            )
            overall_confidence = float(
                getattr(observation, 'overall_confidence', -1.0)
            )
        detector_confidence = float(
            getattr(detection, 'detector_confidence', -1.0)
        )
        color = (0, 220, 255)
        thickness = 2
        if detection.detection_id == candidate_id:
            color = (255, 165, 0)
            thickness = 3
        if detection.detection_id == target_id:
            color = (0, 255, 0)
            thickness = 4
        lines = (
            '{} {} {}'.format(marker, detection.detection_id, label),
            '{} pressable={}'.format(button_type, pressable),
            'det={:.2f} sem={:.2f} all={:.2f}'.format(
                detector_confidence,
                semantic_confidence,
                overall_confidence,
            ),
        )
        items.append(OverlayItem(
            detection_id=detection.detection_id,
            marker=marker,
            bbox=(
                detection.x_min,
                detection.y_min,
                detection.x_max,
                detection.y_max,
            ),
            lines=lines,
            color=color,
            thickness=thickness,
        ))
    return tuple(items)


def render_debug_image(image, overlay_items, state='IDLE'):
    """Render a plan on an RGB image; no GUI or display server is used."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('debug image rendering requires OpenCV') from exc
    output = image.copy()
    height, width = output.shape[:2]
    cv2.putText(
        output,
        'state={}'.format(state),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for item in overlay_items:
        left = max(0, min(width - 1, int(round(item.bbox[0]))))
        top = max(0, min(height - 1, int(round(item.bbox[1]))))
        right = max(0, min(width - 1, int(round(item.bbox[2]))))
        bottom = max(0, min(height - 1, int(round(item.bbox[3]))))
        cv2.rectangle(
            output,
            (left, top),
            (right, bottom),
            item.color,
            item.thickness,
        )
        text_y = max(42, top - 8)
        for line in item.lines:
            cv2.putText(
                output,
                line,
                (left, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                item.color,
                1,
                cv2.LINE_AA,
            )
            text_y += 16
    return output
