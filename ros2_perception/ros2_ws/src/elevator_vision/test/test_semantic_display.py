import math

import numpy as np

from elevator_vision.semantic_display import (
    CACHED,
    ERROR,
    FRESH,
    NONE,
    PENDING,
    STALE,
    PanelEntry,
    SemanticResultView,
    StableDisplayRegistry,
    SystemStatus,
    illumination_probability,
    red_local_contrast_features,
    red_local_contrast_v1,
    render_semantic_display,
    semantic_display_state,
)


def _entry(
    index,
    bbox,
    status=CACHED,
    semantic='3',
    active=True,
    lit_state=None,
):
    return PanelEntry(
        display_id='B{:02d}'.format(index),
        detection_id='button_{:03d}'.format(index),
        semantic_cache_key='semantic_track_{:04d}'.format(index),
        bbox=bbox,
        detector_confidence=0.90 + index / 1000.0,
        depth_valid=True,
        x_m=0.1,
        y_m=-0.02,
        z_m=0.45,
        semantic=semantic,
        semantic_status=status,
        qwen_latency_ms=620.0,
        active=active,
        lit_state=lit_state,
    )


def _system():
    return SystemStatus(True, 1.8, 620.0, 120, 8, 0, 0)


def test_stable_display_ids_survive_jitter_and_are_not_reused():
    registry = StableDisplayRegistry(stale_timeout_sec=1.5)
    first = registry.update(
        'track_a', _entry(99, (100, 100, 180, 180)), now=1.0)
    jittered = registry.update(
        'track_a', _entry(99, (103, 101, 183, 181)), now=1.2)
    assert first.display_id == 'B01'
    assert jittered.display_id == 'B01'
    stale = registry.frame_entries([], now=2.0)
    assert stale[0].display_id == 'B01'
    assert stale[0].semantic_status == STALE
    assert not stale[0].active
    assert registry.frame_entries([], now=3.0) == []
    new_target = registry.update(
        'track_b', _entry(99, (300, 100, 380, 180)), now=3.1)
    assert new_target.display_id == 'B02'


def test_stable_display_id_survives_upstream_cache_key_rotation():
    registry = StableDisplayRegistry(stale_timeout_sec=1.5)
    first = registry.update(
        'semantic_track_0043', _entry(1, (100, 100, 180, 180)), now=1.0)
    rotated = registry.update(
        'semantic_track_0049', _entry(1, (102, 101, 182, 181)), now=1.2)
    assert first.display_id == 'B01'
    assert rotated.display_id == 'B01'
    assert registry.display_id_for('semantic_track_0043') is None
    assert registry.display_id_for('semantic_track_0049') == 'B01'
    entries = registry.frame_entries(['semantic_track_0049'], now=1.3)
    assert [entry.display_id for entry in entries] == ['B01']


def test_hidden_tombstone_reassociates_after_stale_panel_entry_is_removed():
    registry = StableDisplayRegistry(
        stale_timeout_sec=1.5, reassociation_timeout_sec=30.0)
    first = registry.update(
        'old_key', _entry(1, (100, 100, 180, 180)), now=1.0)
    assert registry.frame_entries([], now=3.0) == []
    rotated = registry.update(
        'new_key', _entry(1, (102, 101, 182, 181)), now=9.0)
    assert first.display_id == rotated.display_id == 'B01'


def test_cache_key_rotation_does_not_swap_adjacent_buttons():
    registry = StableDisplayRegistry(stale_timeout_sec=1.5)
    left = registry.update(
        'old_left', _entry(1, (100, 100, 150, 150)), now=1.0)
    right = registry.update(
        'old_right', _entry(2, (175, 100, 225, 150)), now=1.0)
    new_right = registry.update(
        'new_right', _entry(2, (173, 101, 223, 151)), now=1.2)
    new_left = registry.update(
        'new_left', _entry(1, (102, 99, 152, 149)), now=1.2)
    assert left.display_id == new_left.display_id == 'B01'
    assert right.display_id == new_right.display_id == 'B02'


def test_semantic_states_never_reuse_unreliable_or_invalid_label():
    now = 20.0
    valid = SemanticResultView('button_001', 'close', 610.0, '', 19.5)
    semantic, status, latency = semantic_display_state(
        'close', valid, 'button_001', True, True, 30.0, 1.5, now)
    assert (semantic, status, latency) == ('close', FRESH, 610.0)
    assert semantic_display_state(
        'close', valid, 'button_001', True, True, 30.0, 0.1, now)[1] == CACHED
    mismatch = semantic_display_state(
        'close', valid, 'button_002', True, True, 30.0, 1.5, now,
        association_valid=False)
    assert mismatch[0] == '' and mismatch[1] == STALE
    none_result = SemanticResultView('button_001', 'none', 600.0, '', 19.5)
    assert semantic_display_state(
        'none', none_result, 'button_001', True, True, 30.0, 1.5, now
    )[:2] == ('', NONE)
    error_result = SemanticResultView(
        'button_001', 'UNKNOWN', 650.0, 'strict_parse_failed', 19.5)
    assert semantic_display_state(
        'UNKNOWN', error_result, 'button_001', True, True, 30.0, 1.5, now
    )[:2] == ('', ERROR)
    assert semantic_display_state(
        'PROCESSING', None, 'button_001', True, True, 30.0, 1.5, now
    )[:2] == ('', PENDING)
    assert semantic_display_state(
        'close', valid, 'button_001', False, False, 30.0, 1.5, now
    )[:2] == ('', ERROR)


def test_six_button_inline_layout_preserves_content_and_original_resolution():
    image = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    boxes = [
        (900, 120, 980, 200),
        (900, 270, 980, 350),
        (900, 420, 980, 500),
        (900, 570, 980, 650),
        (820, 760, 900, 840),
        (950, 760, 1030, 840),
    ]
    entries = [_entry(index + 1, box) for index, box in enumerate(boxes)]
    canvas, diagnostics = render_semantic_display(image, entries, _system())
    assert canvas.shape == image.shape
    assert diagnostics['panel_width'] == 0
    assert diagnostics['output_width'] == image.shape[1]
    assert not diagnostics['button_content_occluded']
    assert not diagnostics['text_overlap']
    assert not diagnostics['text_outside_image']
    assert diagnostics['unplaced_count'] == 0
    assert len(diagnostics['text_regions']) == 6
    assert all(
        lines[0].startswith('Conf: ')
        and lines[1].startswith('XYZ: ')
        and lines[2] == 'Result: 3'
        for lines in diagnostics['rendered_lines']
    )
    assert not any(
        line.startswith('B') or 'Semantic FPS' in line or 'latency' in line
        for lines in diagnostics['rendered_lines']
        for line in lines
    )
    # The button interior stays untouched; only bbox borders are drawn.
    for left, top, right, bottom in boxes:
        assert np.array_equal(
            canvas[top + 4:bottom - 4, left + 4:right - 4],
            image[top + 4:bottom - 4, left + 4:right - 4],
        )


def test_green_red_and_orange_priority_colors_and_result_text():
    image = np.zeros((480, 900, 3), dtype=np.uint8)
    entries = [
        _entry(1, (100, 80, 160, 140), status=FRESH, semantic='4'),
        _entry(2, (350, 80, 410, 140), status=ERROR, semantic=''),
        _entry(
            3, (600, 80, 660, 140), status=CACHED, semantic='1',
            lit_state=True),
    ]
    _canvas, diagnostics = render_semantic_display(image, entries, _system())
    assert diagnostics['box_colors'] == [
        (0, 255, 0), (0, 0, 255), (0, 165, 255)]
    assert diagnostics['rendered_lines'][0][2] == 'Result: 4'
    assert diagnostics['rendered_lines'][1][2] == 'Result: UNKNOWN'
    assert diagnostics['rendered_lines'][2][2] == 'Result: 1 [LIT]'
    assert diagnostics['recognized_count'] == 1
    assert diagnostics['unknown_count'] == 1
    assert diagnostics['orange_count'] == 1


def test_pending_is_white_processing_and_none_is_red_unknown():
    image = np.zeros((480, 700, 3), dtype=np.uint8)
    entries = [
        _entry(1, (100, 200, 160, 260), status=PENDING, semantic=''),
        _entry(2, (400, 200, 460, 260), status=NONE, semantic='none'),
    ]
    _canvas, diagnostics = render_semantic_display(image, entries, _system())
    assert diagnostics['box_colors'] == [
        (255, 255, 255), (0, 0, 255)]
    assert diagnostics['rendered_lines'][0][2] == 'Result: PROCESSING'
    assert diagnostics['rendered_lines'][1][2] == 'Result: UNKNOWN'
    assert diagnostics['processing_count'] == 1
    assert diagnostics['unknown_count'] == 1


def test_soft_ttl_keeps_valid_label_until_hard_ttl():
    result = SemanticResultView('button_001', '4', 620.0, '', 1.0)
    soft = semantic_display_state(
        '4',
        result,
        'button_001',
        True,
        True,
        30.0,
        1.5,
        32.0,
        hard_cache_ttl_sec=120.0,
        return_reason=True,
    )
    assert soft[:3] == ('4', CACHED, 620.0)
    assert soft[3] == 'soft_ttl_refreshing'
    hard = semantic_display_state(
        '4',
        result,
        'button_001',
        True,
        True,
        30.0,
        1.5,
        121.1,
        hard_cache_ttl_sec=120.0,
        return_reason=True,
    )
    assert hard[0] == ''
    assert hard[1] == STALE
    assert hard[3] == 'hard_ttl_expired'


def test_valid_labels_are_explicit_case_normalized_and_unknown_text_is_red():
    image = np.zeros((480, 1100, 3), dtype=np.uint8)
    entries = [
        _entry(1, (100, 80, 160, 140), status=FRESH, semantic='G'),
        _entry(2, (350, 80, 410, 140), status=CACHED, semantic='UP'),
        _entry(3, (600, 80, 660, 140), status=FRESH, semantic='not_a_label'),
        _entry(
            4, (850, 80, 910, 140), status=ERROR, semantic='',
            lit_state=True),
    ]
    _canvas, diagnostics = render_semantic_display(image, entries, _system())
    assert diagnostics['box_colors'] == [
        (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 165, 255)]
    assert diagnostics['rendered_lines'][0][2] == 'Result: G'
    assert diagnostics['rendered_lines'][1][2] == 'Result: UP'
    assert diagnostics['rendered_lines'][2][2] == 'Result: UNKNOWN'
    assert diagnostics['rendered_lines'][3][2] == 'Result: UNKNOWN'
    assert diagnostics['recognized_count'] == 2
    assert diagnostics['unknown_count'] == 2
    assert diagnostics['orange_count'] == 1


def test_legacy_red_local_contrast_thresholds_and_lit_state_mapping():
    image = np.full((160, 320, 3), 25, dtype=np.uint8)
    lit_box = (40, 40, 120, 120)
    unlit_box = (200, 40, 280, 120)
    image[40:120, 40:120] = (180, 25, 20)
    lit = red_local_contrast_v1(image, lit_box)
    unlit = red_local_contrast_v1(image, unlit_box)
    assert lit['state'] == 'lit'
    assert lit['features']['red_delta_p90'] >= 12.0
    assert lit['features']['red_fraction'] >= 0.055
    assert unlit['state'] == 'unlit'
    assert unlit['features']['red_fraction'] == 0.0


def test_legacy_feature_formula_matches_direct_reference_on_same_crop():
    image = np.full((120, 180, 3), 30, dtype=np.uint8)
    image[30:90, 50:130] = (100, 40, 35)
    bbox = (50, 30, 130, 90)
    features = red_local_contrast_features(image, bbox, context_ratio=0.45)

    array = np.asarray(image, dtype=np.float32)
    inside = array[30:90, 50:130]
    expanded = array[3:117, 14:166]
    mask = np.ones(expanded.shape[:2], dtype=bool)
    mask[27:87, 36:116] = False
    inside_excess = inside[..., 0] - 0.5 * (
        inside[..., 1] + inside[..., 2])
    outer_excess = expanded[..., 0] - 0.5 * (
        expanded[..., 1] + expanded[..., 2])
    expected_delta = round(
        float(np.quantile(inside_excess, 0.90))
        - float(np.quantile(outer_excess[mask], 0.90)),
        4,
    )
    expected_fraction = round(float(np.mean(
        (inside[..., 0] > 45.0)
        & (inside[..., 0] > 1.16 * inside[..., 1])
        & (inside[..., 0] > 1.10 * inside[..., 2])
    )), 6)
    assert features['red_delta_p90'] == expected_delta
    assert features['red_fraction'] == expected_fraction
    assert illumination_probability(
        features['red_delta_p90'],
        features['red_fraction'],
    ) == illumination_probability(expected_delta, expected_fraction)


def test_edge_buttons_and_adjacent_labels_stay_inside_without_overlap():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    entries = [
        _entry(1, (0, 0, 60, 60)),
        _entry(2, (1860, 0, 1919, 60)),
        _entry(3, (0, 1020, 60, 1079)),
        _entry(4, (1860, 1020, 1919, 1079)),
        _entry(5, (850, 700, 910, 760)),
        _entry(6, (930, 700, 990, 760)),
    ]
    _canvas, diagnostics = render_semantic_display(image, entries, _system())
    assert diagnostics['unplaced_count'] == 0
    assert not diagnostics['text_outside_image']
    assert not diagnostics['text_overlap']
    assert not diagnostics['button_content_occluded']


def test_invalid_depth_and_stale_entries_show_unknown_without_stale_bbox():
    invalid = _entry(1, (100, 100, 180, 180), status=ERROR, semantic='')
    invalid = PanelEntry(**{
        **invalid.__dict__, 'depth_valid': False, 'x_m': math.nan,
        'y_m': math.nan, 'z_m': math.nan,
    })
    stale = _entry(2, (300, 100, 380, 180), status=STALE, semantic='', active=False)
    _canvas, diagnostics = render_semantic_display(
        np.zeros((480, 640, 3), dtype=np.uint8), [invalid, stale], _system())
    assert diagnostics['rendered_lines'] == [
        ('Conf: 0.90', 'XYZ: INVALID', 'Result: UNKNOWN')]
    assert diagnostics['box_colors'] == [(0, 0, 255)]
