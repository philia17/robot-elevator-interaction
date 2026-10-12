"""Detection-semantic association and validation by detection_id only."""

from dataclasses import dataclass


class FusionValidationError(ValueError):
    """Raised when global fusion input invariants are violated."""


@dataclass(frozen=True)
class Observation:
    detection_id: str
    label: str
    button_type: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center_x: float
    center_y: float
    detector_confidence: float
    semantic_confidence: float
    overall_confidence: float
    pressable: bool
    valid: bool
    validation_reason: str

    @property
    def bbox(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)


def _confidence(value, field_name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FusionValidationError('{} must be numeric'.format(field_name))
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise FusionValidationError('{} must be in [0, 1]'.format(field_name))
    return value


def validate_bbox(detection):
    coordinates = (
        detection.x_min,
        detection.y_min,
        detection.x_max,
        detection.y_max,
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in coordinates):
        raise FusionValidationError('bbox coordinates must be numeric')
    if detection.x_min < 0.0 or detection.y_min < 0.0:
        raise FusionValidationError('bbox minimum coordinates must be non-negative')
    if detection.x_max <= detection.x_min or detection.y_max <= detection.y_min:
        raise FusionValidationError('bbox must have positive width and height')


def validate_image_bounds(detection, image_width, image_height):
    if image_width <= 0 or image_height <= 0:
        raise FusionValidationError('image dimensions must be positive')
    if detection.x_max > image_width or detection.y_max > image_height:
        raise FusionValidationError(
            'bbox for {} exceeds image bounds'.format(detection.detection_id)
        )
    if not 0.0 <= detection.center_x <= image_width:
        raise FusionValidationError('center_x exceeds image bounds')
    if not 0.0 <= detection.center_y <= image_height:
        raise FusionValidationError('center_y exceeds image bounds')


def _unique_by_detection_id(items, kind):
    indexed = {}
    for item in items:
        detection_id = getattr(item, 'detection_id', '')
        if not isinstance(detection_id, str) or not detection_id.strip():
            raise FusionValidationError('{} detection_id must not be empty'.format(kind))
        if detection_id in indexed:
            raise FusionValidationError(
                'duplicate {} detection_id: {}'.format(kind, detection_id)
            )
        indexed[detection_id] = item
    return indexed


def normalize_weights(detector_weight, semantic_weight):
    detector_weight = float(detector_weight)
    semantic_weight = float(semantic_weight)
    if detector_weight < 0.0 or semantic_weight < 0.0:
        raise FusionValidationError('fusion weights must be non-negative')
    total = detector_weight + semantic_weight
    if total <= 0.0:
        raise FusionValidationError('at least one fusion weight must be positive')
    return detector_weight / total, semantic_weight / total


def fuse_detections_and_semantics(
        detections,
        semantics,
        image_width,
        image_height,
        detector_weight=0.5,
        semantic_weight=0.5):
    """Fuse matching records while preserving detector geometry exactly."""

    detector_by_id = _unique_by_detection_id(detections, 'detector')
    semantic_by_id = _unique_by_detection_id(semantics, 'semantic')
    unknown_ids = sorted(set(semantic_by_id).difference(detector_by_id))
    if unknown_ids:
        raise FusionValidationError(
            'unknown semantic detection_id: {}'.format(', '.join(unknown_ids))
        )
    detector_weight, semantic_weight = normalize_weights(
        detector_weight,
        semantic_weight,
    )

    observations = []
    for detection in detections:
        validate_bbox(detection)
        validate_image_bounds(detection, image_width, image_height)
        detector_confidence = _confidence(
            detection.detector_confidence,
            'detector_confidence',
        )
        semantic = semantic_by_id.get(detection.detection_id)
        if semantic is None:
            observations.append(Observation(
                detection_id=detection.detection_id,
                label='',
                button_type='',
                x_min=detection.x_min,
                y_min=detection.y_min,
                x_max=detection.x_max,
                y_max=detection.y_max,
                center_x=detection.center_x,
                center_y=detection.center_y,
                detector_confidence=detector_confidence,
                semantic_confidence=0.0,
                overall_confidence=detector_weight * detector_confidence,
                pressable=False,
                valid=False,
                validation_reason='missing_semantic',
            ))
            continue

        semantic_confidence = _confidence(
            semantic.semantic_confidence,
            'semantic_confidence',
        )
        label = semantic.label.strip() if isinstance(semantic.label, str) else ''
        valid = bool(label)
        reason = 'valid' if valid else 'empty_label'
        observations.append(Observation(
            detection_id=detection.detection_id,
            label=label,
            button_type=semantic.button_type,
            x_min=detection.x_min,
            y_min=detection.y_min,
            x_max=detection.x_max,
            y_max=detection.y_max,
            center_x=detection.center_x,
            center_y=detection.center_y,
            detector_confidence=detector_confidence,
            semantic_confidence=semantic_confidence,
            overall_confidence=(
                detector_weight * detector_confidence
                + semantic_weight * semantic_confidence
            ),
            pressable=bool(semantic.pressable),
            valid=valid,
            validation_reason=reason,
        ))
    return observations
