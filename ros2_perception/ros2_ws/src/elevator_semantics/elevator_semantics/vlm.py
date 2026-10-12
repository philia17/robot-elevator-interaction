"""VLM backend contracts, deterministic Mock output, and strict parser."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import json
from pathlib import Path


SUPPORTED_BUTTON_TYPES = frozenset({
    'floor_button',
    'door_open',
    'door_close',
})


class VlmConfigurationError(ValueError):
    """Raised when a VLM backend is configured incorrectly."""


class VlmUnavailableError(RuntimeError):
    """Raised when an optional VLM backend cannot be used."""


class VlmParseError(ValueError):
    """Raised when VLM output violates the semantic response contract."""


@dataclass(frozen=True)
class SemanticResult:
    detection_id: str
    label: str
    button_type: str
    pressable: bool
    semantic_confidence: float
    raw_response: str = ''


class BaseVlm(ABC):
    """Common interface for semantic labeling backends."""

    @abstractmethod
    def label(self, detection_ids, som_image=None):
        """Return semantic results referring only to the supplied IDs."""


class MockVlm(BaseVlm):
    """Deterministic semantic backend for the four standard Mock detections."""

    _RESULTS = {
        'button_01': ('1', 'floor_button', True, 0.96),
        'button_02': ('2', 'floor_button', True, 0.95),
        'button_03': ('3B', 'floor_button', True, 0.90),
        'button_04': ('door_open', 'door_open', True, 0.94),
    }

    def label(self, detection_ids, som_image=None):
        known_ids = list(detection_ids)
        if len(set(known_ids)) != len(known_ids):
            raise ValueError('detection IDs supplied to MockVlm must be unique')
        results = []
        for detection_id in known_ids:
            values = self._RESULTS.get(detection_id)
            if values is None:
                continue
            label, button_type, pressable, confidence = values
            raw_response = json.dumps({
                'detection_id': detection_id,
                'label': label,
                'button_type': button_type,
                'pressable': pressable,
                'semantic_confidence': confidence,
            }, sort_keys=True)
            results.append(SemanticResult(
                detection_id=detection_id,
                label=label,
                button_type=button_type,
                pressable=pressable,
                semantic_confidence=confidence,
                raw_response=raw_response,
            ))
        return results


def parse_vlm_response(raw_response, known_detection_ids):
    """Parse and validate the Phase B logical JSON response contract."""

    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VlmParseError('VLM response is not valid JSON') from exc
    if not isinstance(payload, dict) or not isinstance(payload.get('items'), list):
        raise VlmParseError("VLM response must contain an 'items' array")

    known_ids = set(known_detection_ids)
    seen_ids = set()
    results = []
    bbox_fields = {
        'bbox', 'box', 'x_min', 'y_min', 'x_max', 'y_max',
        'center_x', 'center_y',
    }
    for index, item in enumerate(payload['items']):
        if not isinstance(item, dict):
            raise VlmParseError('item {} must be an object'.format(index))
        unexpected_bbox_fields = bbox_fields.intersection(item)
        if unexpected_bbox_fields:
            raise VlmParseError(
                'VLM geometry fields are not allowed: {}'.format(
                    ', '.join(sorted(unexpected_bbox_fields))
                )
            )

        detection_id = item.get('detection_id')
        label = item.get('label')
        button_type = item.get('button_type')
        pressable = item.get('pressable')
        confidence = item.get('semantic_confidence')
        if not isinstance(detection_id, str) or not detection_id.strip():
            raise VlmParseError('detection_id must be a non-empty string')
        detection_id = detection_id.strip()
        if detection_id in seen_ids:
            raise VlmParseError('duplicate semantic detection_id: {}'.format(detection_id))
        if detection_id not in known_ids:
            raise VlmParseError('unknown semantic detection_id: {}'.format(detection_id))
        if not isinstance(label, str) or not label.strip():
            raise VlmParseError('label must be a non-empty string')
        if button_type not in SUPPORTED_BUTTON_TYPES:
            raise VlmParseError('unsupported button_type: {}'.format(button_type))
        if not isinstance(pressable, bool):
            raise VlmParseError('pressable must be a boolean')
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise VlmParseError('semantic_confidence must be numeric')
        if not 0.0 <= float(confidence) <= 1.0:
            raise VlmParseError('semantic_confidence must be in [0, 1]')

        seen_ids.add(detection_id)
        results.append(SemanticResult(
            detection_id=detection_id,
            label=label.strip(),
            button_type=button_type,
            pressable=pressable,
            semantic_confidence=float(confidence),
            raw_response=raw_response,
        ))
    return results


class LlamaCppVlm(BaseVlm):
    """Validated placeholder for a future optional llama.cpp VLM backend."""

    def __init__(self, model_path, mmproj_path):
        self.model_path = str(model_path or '')
        self.mmproj_path = str(mmproj_path or '')
        self.validate_configuration()

    def validate_configuration(self):
        if not self.model_path:
            raise VlmConfigurationError('llama.cpp model_path must not be empty')
        if not Path(self.model_path).is_file():
            raise VlmConfigurationError(
                'llama.cpp model file does not exist: {}'.format(self.model_path)
            )
        if not self.mmproj_path:
            raise VlmConfigurationError('llama.cpp mmproj_path must not be empty')
        if not Path(self.mmproj_path).is_file():
            raise VlmConfigurationError(
                'llama.cpp mmproj file does not exist: {}'.format(self.mmproj_path)
            )

    @staticmethod
    def build_prompt(detection_ids):
        ids = ', '.join(sorted(detection_ids))
        return (
            'Return JSON only with shape '
            '{"items":[{"detection_id":"button_03","label":"3B",'
            '"button_type":"floor_button","pressable":true,'
            '"semantic_confidence":0.90}]}. '
            'Use only these detection IDs: {}. Do not return bounding boxes.'.format(ids)
        )

    @staticmethod
    def parse_response(raw_response, known_detection_ids):
        return parse_vlm_response(raw_response, known_detection_ids)

    def label(self, detection_ids, som_image=None):
        try:
            importlib.import_module('llama_cpp')
        except ImportError as exc:
            raise VlmUnavailableError(
                'llama.cpp backend unavailable: optional llama_cpp package is not installed'
            ) from exc
        raise VlmUnavailableError(
            'llama.cpp backend unavailable in Phase B: real inference is not implemented'
        )
