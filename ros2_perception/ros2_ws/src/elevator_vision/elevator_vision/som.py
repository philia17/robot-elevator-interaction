"""Deterministic Set-of-Mark (SoM) annotation and marker mapping."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropMetadata:
    marker: str
    detection_id: str
    detector_bbox: tuple
    crop_bounds: tuple


@dataclass(frozen=True)
class SomResult:
    annotated_image: object
    marker_to_detection_id: dict
    crops: tuple


_GLYPHS = {
    'A': ('01110', '10001', '10001', '11111', '10001', '10001', '10001'),
    'B': ('11110', '10001', '10001', '11110', '10001', '10001', '11110'),
    'C': ('01111', '10000', '10000', '10000', '10000', '10000', '01111'),
    'D': ('11110', '10001', '10001', '10001', '10001', '10001', '11110'),
}


def stable_detection_order(detections):
    return sorted(
        detections,
        key=lambda item: (item.center_y, item.center_x, item.detection_id),
    )


def _marker_name(index):
    """Return deterministic spreadsheet-style markers: A..Z, AA..AZ, ..."""
    marker = ''
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        marker = chr(ord('A') + remainder) + marker
    return marker


def _paint_rectangle(image, left, top, right, bottom, color, thickness=2):
    height, width = image.shape[:2]
    left = max(0, min(width - 1, left))
    right = max(0, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        return
    image[top:min(top + thickness, bottom), left:right] = color
    image[max(top, bottom - thickness):bottom, left:right] = color
    image[top:bottom, left:min(left + thickness, right)] = color
    image[top:bottom, max(left, right - thickness):right] = color


def _paint_marker(image, marker, left, top):
    height, width = image.shape[:2]
    box_width = max(15, 9 * len(marker) + 6)
    box_height = 18
    left = max(0, min(width - box_width, left))
    top = max(0, min(height - box_height, top))
    image[top:top + box_height, left:left + box_width] = (255, 220, 0)
    try:
        import cv2
        cv2.putText(
            image,
            marker,
            (left + 2, top + box_height - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return
    except ImportError:
        pass
    glyph = _GLYPHS.get(marker)
    if glyph is None:
        return
    for row_index, row in enumerate(glyph):
        for column_index, pixel in enumerate(row):
            if pixel == '1':
                y_start = top + 2 + row_index * 2
                x_start = left + 2 + column_index * 2
                image[y_start:y_start + 2, x_start:x_start + 2] = (0, 0, 0)


def generate_som(image, detections):
    """Annotate a copy of an RGB image without mutating detection geometry."""

    if image is None or not hasattr(image, 'shape') or len(image.shape) != 3:
        raise ValueError('SoM input must be an HxWxC image')
    if image.shape[2] < 3:
        raise ValueError('SoM input must have at least three color channels')

    ordered = stable_detection_order(detections)

    annotated = np.array(image, copy=True)
    marker_map = {}
    crop_metadata = []
    height, width = annotated.shape[:2]
    for index, detection in enumerate(ordered):
        marker = _marker_name(index)
        left = int(round(detection.x_min))
        top = int(round(detection.y_min))
        right = int(round(detection.x_max))
        bottom = int(round(detection.y_max))
        clipped = (
            max(0, min(width, left)),
            max(0, min(height, top)),
            max(0, min(width, right)),
            max(0, min(height, bottom)),
        )
        _paint_rectangle(annotated, left, top, right, bottom, (255, 220, 0))
        _paint_marker(annotated, marker, left, max(0, top - 18))
        marker_map[marker] = detection.detection_id
        crop_metadata.append(CropMetadata(
            marker=marker,
            detection_id=detection.detection_id,
            detector_bbox=detection.bbox,
            crop_bounds=clipped,
        ))

    return SomResult(
        annotated_image=annotated,
        marker_to_detection_id=marker_map,
        crops=tuple(crop_metadata),
    )
