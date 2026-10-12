import math

import pytest

from elevator_vision.m0_target_core import (
    M0TargetConfig,
    M0TargetTracker,
    SEMANTIC_GRACE,
)


def observation(detection_id='button_004', x=0.09, y=0.0, z=0.70):
    return {
        'detection_id': detection_id,
        'detector_confidence': 0.90,
        'depth_valid': True,
        'depth_valid_ratio': 1.0,
        'x_min': 100.0,
        'y_min': 200.0,
        'x_max': 160.0,
        'y_max': 260.0,
        'x_m': x,
        'y_m': y,
        'z_m': z,
    }


def semantic(
    detection_id='button_004',
    label='1',
    track_id='semantic_track_0004',
    status='FRESH',
    age=0.2,
):
    return {
        'detection_id': detection_id,
        'label': label,
        'pressable': True,
        'track_id': track_id,
        'semantic_status': status,
        'semantic_age_sec': age,
        'association_valid': True,
    }


def selected_button(label='2', track='semantic_track_chosen', after=100_000_000):
    return {
        'selection_id': 'selection-' + str(after), 'label': label,
        'detection_id': 'old-frame-id', 'semantic_track_id': track,
        'after_stamp_ns': after,
    }


@pytest.mark.parametrize('label', ['2', 'open', 'close', 'up', 'down', 'b1', 'g'])
def test_selected_target_uses_track_across_changing_frame_local_ids(label):
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.select_target(selected_button(label))
    decisions = []
    for index, detection_id in enumerate(('new-frame-7', 'new-frame-3')):
        decisions.append(tracker.update(
            [observation(detection_id, x=.12), observation('other', x=.25)],
            [semantic(detection_id, label, 'semantic_track_chosen'),
             semantic('other', label, 'different-physical-button')],
            (index + 2) * 100_000_000, 'cam',
        ))
    assert [decision['status'] for decision in decisions] == ['STABILIZING', 'READY']
    assert decisions[-1]['selected']['track_id'] == 'semantic_track_chosen'
    assert decisions[-1]['selected']['camera_point_m']['x'] == pytest.approx(.12)
    assert decisions[-1]['target_label'] == label
    assert decisions[-1]['motion_authorized'] is False


def test_selected_target_does_not_reuse_old_frame_or_switch_to_same_label_neighbor():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.select_target(selected_button())
    chosen = semantic('old-frame-id', '2', 'semantic_track_chosen')
    stale = tracker.update([observation('old-frame-id')], [chosen], 100_000_000, 'cam')
    assert stale['status'] == 'SELECTION_WAIT_FRESH_FRAME'
    wrong = tracker.update(
        [observation('old-frame-id')],
        [semantic('old-frame-id', '2', 'some-other-track')], 200_000_000, 'cam',
    )
    assert wrong['status'] == 'NOT_FOUND'
    assert wrong['confirmation_count'] == 0
    assert wrong['perception_ready'] is False


def test_selection_without_stable_track_never_promotes_reused_detector_id():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.select_target(selected_button(track=''))
    for stamp in (200_000_000, 300_000_000):
        decision = tracker.update(
            [observation('old-frame-id')], [semantic('old-frame-id', '2')], stamp, 'cam',
        )
        assert decision['status'] == 'SELECTION_TRACK_REQUIRED'
        assert not decision['perception_ready']


def test_reselect_and_clear_discard_ready_without_falling_back_to_button_one():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.select_target(selected_button())
    for stamp in (200_000_000, 300_000_000):
        ready = tracker.update(
            [observation('new')], [semantic('new', '2', 'semantic_track_chosen')], stamp, 'cam',
        )
    assert ready['perception_ready']
    tracker.select_target(selected_button(after=300_000_000))
    first = tracker.update(
        [observation('new')], [semantic('new', '2', 'semantic_track_chosen')], 400_000_000, 'cam',
    )
    assert first['status'] == 'STABILIZING'
    assert first['confirmation_count'] == 1
    tracker.select_target({'label': ''})
    for stamp in (500_000_000, 600_000_000):
        cleared = tracker.update([observation()], [semantic()], stamp, 'cam')
        assert cleared['status'] == 'NO_SELECTION'
        assert cleared['target_label'] == ''
        assert cleared['confirmation_count'] == 0


@pytest.mark.parametrize('label', ['unknown', '', '?'])
def test_unknown_selection_cannot_restore_a_previous_ready_target(label):
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    for stamp in (100_000_000, 200_000_000):
        tracker.update([observation()], [semantic()], stamp, 'cam')
    if label:
        with pytest.raises(ValueError, match='unknown'):
            tracker.select_target(selected_button(label))
    else:
        tracker.select_target({'label': ''})
    rejected = tracker.update([observation()], [semantic()], 300_000_000, 'cam')
    assert rejected['status'] == 'NO_SELECTION'
    assert not rejected['perception_ready']


def test_selected_nonpressable_button_fails_semantic_gate():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.select_target(selected_button())
    nonpressable = semantic('new', '2', 'semantic_track_chosen')
    nonpressable['pressable'] = False
    decision = tracker.update([observation('new')], [nonpressable], 200_000_000, 'cam')
    assert decision['status'] == 'INVALID_TARGET'
    assert not decision['perception_ready']


def test_unique_target_becomes_ready_after_required_stable_frames():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=3))
    decisions = [
        tracker.update(
            [observation(x=0.09 + index * 0.001)],
            [semantic()],
            (index + 1) * 100_000_000,
            'cam',
        )
        for index in range(3)
    ]
    assert [item['status'] for item in decisions] == [
        'STABILIZING',
        'STABILIZING',
        'READY',
    ]
    assert decisions[-1]['motion_authorized'] is False
    assert decisions[-1]['selected']['camera_point_m']['x'] == pytest.approx(
        0.091
    )
    assert decisions[-1]['selected']['camera_point_m']['z'] == pytest.approx(0.70)


def test_two_floor_one_labels_fail_ambiguous_and_reset_stability():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    decision = tracker.update(
        [observation(), observation('button_007')],
        [semantic(), semantic('button_007', track_id='track_7')],
        200_000_000,
        'cam',
    )
    assert decision['status'] == 'AMBIGUOUS'
    assert decision['confirmation_count'] == 0


def test_detection_id_switch_with_stable_bbox_keeps_confirmation_window():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=3))
    decisions = []
    for index, detection_id in enumerate(
        ('button_004', 'button_003', 'button_004')
    ):
        decisions.append(
            tracker.update(
                [observation(detection_id=detection_id)],
                [
                    semantic(
                        detection_id=detection_id,
                        track_id='semantic_track_000{}'.format(index + 3),
                    )
                ],
                (index + 1) * 100_000_000,
                'cam',
            )
        )
    assert [item['status'] for item in decisions] == [
        'STABILIZING',
        'STABILIZING',
        'READY',
    ]
    assert decisions[-1]['confirmation_count'] == 3
    assert decisions[-1]['selected']['detection_id'] == 'button_004'
    assert decisions[-1]['selected']['camera_point_m']['x'] == pytest.approx(
        0.09
    )


def test_ambiguous_same_label_on_jitter_ids_resets_window():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    decision = tracker.update(
        [observation(), observation('button_003')],
        [
            semantic(),
            semantic(
                detection_id='button_003',
                track_id='semantic_track_0003',
            ),
        ],
        200_000_000,
        'cam',
    )
    assert decision['status'] == 'AMBIGUOUS'
    assert decision['matching_label_count'] == 2
    assert decision['confirmation_count'] == 0


def test_detection_id_switch_with_position_jump_resets_window():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=3))
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    tracker.update([observation()], [semantic()], 200_000_000, 'cam')
    decision = tracker.update(
        [observation(detection_id='button_003', x=0.20)],
        [
            semantic(
                detection_id='button_003',
                track_id='semantic_track_0003',
            )
        ],
        300_000_000,
        'cam',
    )
    assert decision['status'] == 'STABILIZING'
    assert decision['confirmation_count'] == 1


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('depth_valid', False),
        ('depth_valid_ratio', 0.1),
        ('detector_confidence', 0.1),
        ('z_m', math.nan),
    ],
)
def test_invalid_3d_never_becomes_ready(field, value):
    item = observation()
    item[field] = value
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    decision = tracker.update([item], [semantic()], 100_000_000, 'cam')
    assert decision['status'] == 'INVALID_TARGET'
    assert decision['perception_ready'] is False


def test_cached_semantic_is_allowed_but_stale_is_rejected():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    cached = tracker.update(
        [observation()],
        [semantic(status='CACHED')],
        100_000_000,
        'cam',
    )
    stale = tracker.update(
        [observation()],
        [semantic(status='STALE')],
        200_000_000,
        'cam',
    )
    assert cached['status'] == 'STABILIZING'
    assert stale['status'] == 'SEMANTIC_SET_INCOMPLETE'


def test_position_jump_restarts_consecutive_count():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=3))
    tracker.update([observation(x=0.09)], [semantic()], 100_000_000, 'cam')
    tracker.update([observation(x=0.091)], [semantic()], 200_000_000, 'cam')
    decision = tracker.update(
        [observation(x=0.20)],
        [semantic()],
        300_000_000,
        'cam',
    )
    assert decision['status'] == 'STABILIZING'
    assert decision['confirmation_count'] == 1


def test_exact_label_does_not_treat_b1_as_floor_one():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    decision = tracker.update(
        [observation()],
        [semantic(label='b1')],
        100_000_000,
        'cam',
    )
    assert decision['status'] == 'NOT_FOUND'


def test_old_semantic_set_fails_closed():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    old = semantic()
    old['semantic_age_sec'] = 3.0
    decision = tracker.update(
        [observation()],
        [old],
        100_000_000,
        'cam',
    )
    assert decision['status'] == 'SEMANTIC_SET_INCOMPLETE'


def test_cached_non_target_can_be_older_while_target_must_be_fresh():
    tracker = M0TargetTracker(
        M0TargetConfig(
            required_consecutive_frames=2,
            maximum_semantic_age_sec=2.0,
            maximum_non_target_semantic_age_sec=30.0,
        )
    )
    target_item = semantic()
    other = semantic(
        detection_id='button_005',
        label='2',
        track_id='semantic_track_0005',
        status='CACHED',
    )
    other['semantic_age_sec'] = 20.0
    decision = tracker.update(
        [observation(), observation('button_005')],
        [target_item, other],
        100_000_000,
        'cam',
    )
    assert decision['status'] == 'STABILIZING'

    target_item['semantic_age_sec'] = 3.0
    decision = tracker.update(
        [observation(), observation('button_005')],
        [target_item, other],
        200_000_000,
        'cam',
    )
    assert decision['status'] == 'SEMANTIC_SET_INCOMPLETE'


def test_diagnostic_target_only_mode_ignores_pending_non_target():
    tracker = M0TargetTracker(
        M0TargetConfig(
            required_consecutive_frames=2,
            require_complete_semantic_set=False,
        )
    )
    pending_other = semantic(
        detection_id='button_005', label='close', status='PENDING'
    )
    first = tracker.update(
        [observation(), observation('button_005')],
        [semantic(), pending_other],
        100_000_000,
        'cam',
    )
    second = tracker.update(
        [observation(), observation('button_005')],
        [semantic(), pending_other],
        200_000_000,
        'cam',
    )
    assert first['status'] == 'STABILIZING'
    assert second['status'] == 'READY'
    assert second['semantic_scope'] == 'TARGET_ONLY_DIAGNOSTIC'
    assert second['non_target_semantic_incomplete'] == [
        {
            'detection_id': 'button_005',
            'label': 'close',
            'semantic_status': 'PENDING',
            'semantic_age_sec': pytest.approx(0.2),
            'reason': 'status_not_complete',
        }
    ]


def test_strict_mode_pending_non_target_resets_target_stability():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    pending_other = semantic(
        detection_id='button_005', label='close', status='PENDING'
    )
    decision = tracker.update(
        [observation(), observation('button_005')],
        [semantic(), pending_other],
        200_000_000,
        'cam',
    )
    assert decision['status'] == 'SEMANTIC_SET_INCOMPLETE'
    assert decision['confirmation_count'] == 0
    assert decision['semantic_scope'] == 'STRICT_ALL_VISIBLE'
    assert decision['non_target_semantic_incomplete'][0]['label'] == 'close'


@pytest.mark.parametrize('status, age', [('STALE', 0.2), ('FRESH', 2.1)])
def test_target_only_mode_still_rejects_stale_or_pending_target(status, age):
    tracker = M0TargetTracker(
        M0TargetConfig(
            require_complete_semantic_set=False,
            maximum_semantic_age_sec=2.0,
        )
    )
    target = semantic(status=status)
    target['semantic_age_sec'] = age
    decision = tracker.update([observation()], [target], 100_000_000, 'cam')
    assert decision['status'] == 'SEMANTIC_SET_INCOMPLETE'
    assert decision['perception_ready'] is False
    assert decision['motion_authorized'] is False


def test_target_only_mode_is_explicitly_no_motion_even_when_ready():
    tracker = M0TargetTracker(
        M0TargetConfig(
            required_consecutive_frames=2,
            require_complete_semantic_set=False,
        )
    )
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    ready = tracker.update([observation()], [semantic()], 200_000_000, 'cam')
    assert ready['status'] == 'READY'
    assert ready['mode'] == 'NO_MOTION'
    assert ready['motion_authorized'] is False


def test_diagnostic_target_only_mode_still_rejects_pending_target():
    tracker = M0TargetTracker(
        M0TargetConfig(require_complete_semantic_set=False)
    )
    decision = tracker.update(
        [observation()], [semantic(status='PENDING')], 100_000_000, 'cam'
    )
    assert decision['status'] == 'SEMANTIC_SET_INCOMPLETE'


def test_large_frame_gap_restarts_confirmation_window():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=3))
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    tracker.update([observation()], [semantic()], 200_000_000, 'cam')
    decision = tracker.update(
        [observation()],
        [semantic()],
        900_000_000,
        'cam',
    )
    assert decision['status'] == 'STABILIZING'
    assert decision['confirmation_count'] == 1


def test_8s_observation_gap_policy_reaches_ready_and_resets_beyond_bound():
    # Mirror the live RGB-D launch policy: 5 confirmations, 30 s semantic
    # freshness, and an 8.0 s bounded observation gap covering the measured
    # 4.0-7.3 s Qwen/RGB-D refresh pauses. This is an observation-continuity allowance,
    # not a motion gate.
    config = M0TargetConfig(
        target_label='1',
        required_consecutive_frames=5,
        maximum_semantic_age_sec=30.0,
        maximum_non_target_semantic_age_sec=30.0,
        maximum_frame_gap_sec=8.0,
        require_complete_semantic_set=False,
    )
    assert config.maximum_frame_gap_sec == 8.0
    assert config.required_consecutive_frames == 5

    tracker = M0TargetTracker(config)
    stamps_ns = [
        int((0.1 + 7.0 * index) * 1_000_000_000) for index in range(5)
    ]
    decisions = [
        tracker.update(
            [observation()],
            [semantic()],
            stamp_ns,
            'cam',
        )
        for stamp_ns in stamps_ns
    ]
    assert [item['status'] for item in decisions] == [
        'STABILIZING',
        'STABILIZING',
        'STABILIZING',
        'STABILIZING',
        'READY',
    ]
    ready = decisions[-1]
    assert ready['confirmation_count'] == 5
    assert ready['perception_ready'] is True
    assert ready['motion_authorized'] is False
    assert ready['effective_label'] == '1'

    beyond_bound_ns = stamps_ns[-1] + int(8.1 * 1_000_000_000)
    reset = tracker.update(
        [observation()],
        [semantic()],
        beyond_bound_ns,
        'cam',
    )
    assert reset['status'] == 'STABILIZING'
    assert reset['confirmation_count'] == 1
    assert reset['perception_ready'] is False
    assert reset['motion_authorized'] is False


def _stabilized_tracker(required=3, **config_overrides):
    tracker = M0TargetTracker(
        M0TargetConfig(
            required_consecutive_frames=required,
            require_complete_semantic_set=False,
            **config_overrides
        )
    )
    first = tracker.update(
        [observation()], [semantic()], 100_000_000, 'cam'
    )
    second = tracker.update(
        [observation()], [semantic()], 200_000_000, 'cam'
    )
    assert [first['status'], second['status']] == [
        'STABILIZING',
        'STABILIZING',
    ]
    assert second['confirmation_count'] == 2
    return tracker


def test_diagnostic_grace_keeps_window_while_target_revalidating():
    tracker = _stabilized_tracker()
    grace = tracker.update(
        [observation()],
        [semantic(status='REVALIDATING')],
        300_000_000,
        'cam',
    )
    assert grace['status'] == SEMANTIC_GRACE
    assert grace['confirmation_count'] == 2
    assert grace['perception_ready'] is False
    assert grace['semantic_grace_active'] is True
    assert grace['semantic_grace_reason'] == 'target_semantic_incomplete'
    assert grace['semantic_grace_remaining_sec'] == pytest.approx(9.9)

    ready = tracker.update(
        [observation()], [semantic()], 400_000_000, 'cam'
    )
    assert ready['status'] == 'READY'
    assert ready['confirmation_count'] == 3
    assert ready['perception_ready'] is True


def test_live_refresh_gap_retains_five_frame_window_until_fresh_semantic():
    tracker = M0TargetTracker(
        M0TargetConfig(
            required_consecutive_frames=5,
            maximum_semantic_age_sec=30.0,
            maximum_frame_gap_sec=8.0,
            target_semantic_grace_ttl_sec=10.0,
            require_complete_semantic_set=False,
        )
    )
    stamps = [0.1, 4.0, 4.6, 4.8, 6.7]
    decisions = [
        tracker.update(
            [observation()],
            [semantic(age=29.0)],
            int(stamp * 1_000_000_000),
            'cam',
        )
        for stamp in stamps
    ]
    assert decisions[-1]['status'] == 'READY'
    assert decisions[-1]['confirmation_count'] == 5

    grace = tracker.update(
        [observation()],
        [semantic(age=33.0)],
        int(10.9 * 1_000_000_000),
        'cam',
    )
    assert grace['status'] == SEMANTIC_GRACE
    assert grace['confirmation_count'] == 5
    assert grace['perception_ready'] is False

    resumed = tracker.update(
        [observation()],
        [semantic(age=3.4)],
        int(14.9 * 1_000_000_000),
        'cam',
    )
    assert resumed['status'] == 'READY'
    assert resumed['confirmation_count'] == 5
    assert resumed['perception_ready'] is True


def test_diagnostic_grace_keeps_window_while_target_missing_and_id_rotates():
    tracker = M0TargetTracker(
        M0TargetConfig(
            required_consecutive_frames=3,
            require_complete_semantic_set=False,
        )
    )
    tracker.update(
        [observation('button_004')],
        [semantic('button_004')],
        100_000_000,
        'cam',
    )
    tracker.update(
        [observation('button_003')],
        [semantic('button_003', track_id='semantic_track_0003')],
        200_000_000,
        'cam',
    )
    grace = tracker.update(
        [observation('button_004')],
        [],
        300_000_000,
        'cam',
    )
    assert grace['status'] == SEMANTIC_GRACE
    assert grace['confirmation_count'] == 2
    assert grace['semantic_grace_reason'] == 'target_semantic_missing'

    ready = tracker.update(
        [observation('button_004')],
        [semantic('button_004')],
        400_000_000,
        'cam',
    )
    assert ready['status'] == 'READY'
    assert ready['confirmation_count'] == 3


def test_diagnostic_pending_target_uses_spatial_candidate_for_grace():
    tracker = _stabilized_tracker()
    pending = semantic(status='PENDING')
    pending['label'] = ''
    grace = tracker.update(
        [observation('button_003')],
        [pending],
        300_000_000,
        'cam',
    )
    assert grace['status'] == SEMANTIC_GRACE
    assert grace['confirmation_count'] == 2
    assert grace['matching_label_count'] == 0


def test_grace_ttl_expiry_resets_window():
    tracker = _stabilized_tracker(
        target_semantic_grace_ttl_sec=1.0,
        maximum_frame_gap_sec=2.0,
    )
    tracker.update([observation()], [], 300_000_000, 'cam')
    expired = tracker.update(
        [observation()],
        [],
        1_300_000_000,
        'cam',
    )
    assert expired['status'] == 'NOT_FOUND'
    assert expired['confirmation_count'] == 0
    assert expired['semantic_grace_active'] is False


def test_grace_requires_increasing_stamps_within_frame_gap():
    tracker = _stabilized_tracker()
    duplicate = tracker.update(
        [observation()],
        [],
        200_000_000,
        'cam',
    )
    assert duplicate['status'] == 'NOT_FOUND'
    assert duplicate['confirmation_count'] == 0


def test_ambiguous_target_resets_even_after_grace():
    tracker = _stabilized_tracker()
    tracker.update([observation()], [], 300_000_000, 'cam')
    ambiguous = tracker.update(
        [observation(), observation('button_007')],
        [
            semantic(),
            semantic(
                detection_id='button_007',
                label='1',
                track_id='semantic_track_0007',
            ),
        ],
        400_000_000,
        'cam',
    )
    assert ambiguous['status'] == 'AMBIGUOUS'
    assert ambiguous['confirmation_count'] == 0


def test_spatially_ambiguous_missing_target_resets_window():
    tracker = _stabilized_tracker()
    decision = tracker.update(
        [
            observation('button_004'),
            observation('button_003'),
        ],
        [],
        300_000_000,
        'cam',
    )
    assert decision['status'] == 'NOT_FOUND'
    assert decision['confirmation_count'] == 0


def test_stale_target_cannot_ready_or_freeze_after_grace():
    tracker = _stabilized_tracker(
        target_semantic_grace_ttl_sec=1.0,
        maximum_frame_gap_sec=2.0,
    )
    tracker.update([observation()], [], 300_000_000, 'cam')
    stale = tracker.update(
        [observation()],
        [semantic(status='STALE')],
        1_500_000_000,
        'cam',
    )
    assert stale['status'] == 'SEMANTIC_SET_INCOMPLETE'
    assert stale['perception_ready'] is False
    assert stale['confirmation_count'] == 0


def test_grace_never_changes_motion_fields():
    tracker = _stabilized_tracker()
    decisions = [
        tracker.update(
            [observation()],
            [semantic(status='REVALIDATING')],
            300_000_000,
            'cam',
        ),
        tracker.update(
            [observation()],
            [semantic()],
            400_000_000,
            'cam',
        ),
    ]
    for decision in decisions:
        assert decision['mode'] == 'NO_MOTION'
        assert decision['motion_authorized'] is False
        assert 'execution_ready' not in decision
    assert decisions[-1]['status'] == 'READY'
    assert decisions[-1]['motion_authorized'] is False


def test_grace_requires_observation_quality_gates():
    tracker = _stabilized_tracker()
    bad = observation()
    bad['depth_valid'] = False
    decision = tracker.update([bad], [], 300_000_000, 'cam')
    assert decision['status'] == 'NOT_FOUND'
    assert decision['confirmation_count'] == 0


def test_grace_requires_association_valid_when_target_entry_present():
    tracker = _stabilized_tracker()
    revalidating = semantic(status='REVALIDATING')
    revalidating['association_valid'] = False
    decision = tracker.update(
        [observation()],
        [revalidating],
        300_000_000,
        'cam',
    )
    assert decision['status'] == 'SEMANTIC_SET_INCOMPLETE'
    assert decision['confirmation_count'] == 0


def test_formal_mode_never_enters_grace():
    tracker = M0TargetTracker(M0TargetConfig(required_consecutive_frames=2))
    tracker.update([observation()], [semantic()], 100_000_000, 'cam')
    pending = tracker.update(
        [observation()],
        [semantic(status='PENDING')],
        200_000_000,
        'cam',
    )
    assert pending['status'] == 'SEMANTIC_SET_INCOMPLETE'
    assert pending['confirmation_count'] == 0
    assert pending['semantic_grace_active'] is False

    missing = tracker.update(
        [observation()],
        [],
        300_000_000,
        'cam',
    )
    assert missing['status'] == 'NOT_FOUND'
    assert missing['confirmation_count'] == 0
    assert missing['semantic_grace_active'] is False


@pytest.mark.parametrize('ttl', [0.0, 10.01, 30.0])
def test_config_rejects_out_of_bounds_grace_ttl(ttl):
    with pytest.raises(ValueError):
        M0TargetConfig(target_semantic_grace_ttl_sec=ttl)


def test_config_accepts_maximum_grace_ttl_ten_seconds():
    config = M0TargetConfig(target_semantic_grace_ttl_sec=10.0)
    assert config.target_semantic_grace_ttl_sec == 10.0
