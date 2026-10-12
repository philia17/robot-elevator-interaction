"""CPU-only tests for Mock VLM and parser contracts."""

import json

import pytest

from elevator_semantics.vlm import MockVlm, VlmParseError, parse_vlm_response


def _valid_payload():
    return {
        'items': [{
            'detection_id': 'button_03',
            'label': '3B',
            'button_type': 'floor_button',
            'pressable': True,
            'semantic_confidence': 0.90,
        }]
    }


def test_mock_vlm_output_matches_standard_ids():
    results = MockVlm().label([
        'button_01', 'button_02', 'button_03', 'button_04'
    ])
    assert [(item.detection_id, item.label, item.button_type, item.pressable) for item in results] == [
        ('button_01', '1', 'floor_button', True),
        ('button_02', '2', 'floor_button', True),
        ('button_03', '3B', 'floor_button', True),
        ('button_04', 'door_open', 'door_open', True),
    ]


def test_parser_accepts_contract():
    result = parse_vlm_response(json.dumps(_valid_payload()), {'button_03'})
    assert result[0].detection_id == 'button_03'
    assert result[0].semantic_confidence == 0.90


def test_invalid_vlm_json_is_rejected():
    with pytest.raises(VlmParseError, match='valid JSON'):
        parse_vlm_response('{invalid', {'button_03'})


def test_duplicate_semantic_id_is_rejected():
    payload = _valid_payload()
    payload['items'].append(dict(payload['items'][0]))
    with pytest.raises(VlmParseError, match='duplicate'):
        parse_vlm_response(json.dumps(payload), {'button_03'})


def test_unknown_semantic_id_is_rejected():
    with pytest.raises(VlmParseError, match='unknown'):
        parse_vlm_response(json.dumps(_valid_payload()), {'button_01'})


def test_vlm_bbox_fields_are_rejected():
    payload = _valid_payload()
    payload['items'][0]['bbox'] = [1, 2, 3, 4]
    with pytest.raises(VlmParseError, match='geometry'):
        parse_vlm_response(json.dumps(payload), {'button_03'})
