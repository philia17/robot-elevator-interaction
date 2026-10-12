from elevator_vision.m0b_trial_core import (
    ACQUIRING,
    FAILED,
    FLOW_PASS_NO_LIGHT_EXPECTED,
    OCCLUSION_EXPECTED,
    READY_TO_LOCK,
    REACQUIRING,
    TARGET_LOCKED,
    M0BTrial,
    M0BTrialConfig,
)


class FakeClock:
    def __init__(self, now=0.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def target(
    stamp_ns=100_000_000,
    detection_id='button_004',
    x=0.09,
    bbox=None,
):
    return {
        'status': 'READY',
        'perception_ready': True,
        'target_label': '1',
        'effective_label': '1',
        'source_stamp_ns': stamp_ns,
        'frame_id': 'camera_optical_frame',
        'matching_label_count': 1,
        'detection_id': detection_id,
        'semantic_track_id': 'semantic_track_0004',
        'bbox': bbox or [100.0, 200.0, 160.0, 260.0],
        'camera_point_m': {'x': x, 'y': 0.0, 'z': 0.70},
        'motion_authorized': False,
    }


def stabilizing(stamp_ns=200_000_000):
    item = target(stamp_ns=stamp_ns)
    item['status'] = 'STABILIZING'
    item['perception_ready'] = False
    return item


def test_ready_target_can_be_frozen_but_never_authorizes_motion():
    trial = M0BTrial()
    observed = trial.observe(target())
    assert observed['state'] == READY_TO_LOCK
    locked = trial.lock()
    assert locked['state'] == TARGET_LOCKED
    assert locked['motion_authorized'] is False
    assert len(locked['frozen_target_sha256']) == 64


def test_live_updates_cannot_mutate_target_after_lock_or_during_occlusion():
    trial = M0BTrial()
    trial.observe(target())
    locked = trial.lock()
    frozen = locked['frozen_target']
    digest = locked['frozen_target_sha256']

    trial.observe(target(stamp_ns=200_000_000, x=0.50))
    occluded = trial.enter_occlusion()
    trial.observe(target(stamp_ns=300_000_000, x=0.80))
    snapshot = trial.snapshot()

    assert occluded['state'] == OCCLUSION_EXPECTED
    assert snapshot['frozen_target'] == frozen
    assert snapshot['frozen_target_sha256'] == digest
    assert snapshot['ignored_live_updates'] == 2


def test_same_geometry_must_reappear_for_three_increasing_frames():
    trial = M0BTrial(M0BTrialConfig(required_reacquire_frames=3))
    trial.observe(target())
    trial.lock()
    trial.enter_occlusion()
    assert trial.clear_occlusion()['state'] == REACQUIRING

    for index in range(2):
        decision = trial.observe(
            target(stamp_ns=(index + 2) * 100_000_000, x=0.091)
        )
        assert decision['state'] == REACQUIRING

    passed = trial.observe(target(stamp_ns=400_000_000, x=0.091))
    assert passed['state'] == FLOW_PASS_NO_LIGHT_EXPECTED
    assert passed['result'] == FLOW_PASS_NO_LIGHT_EXPECTED


def test_reacquisition_accepts_three_unique_compatible_stabilizing_frames():
    trial = M0BTrial(M0BTrialConfig(required_reacquire_frames=3))
    trial.observe(target())
    trial.lock()
    trial.enter_occlusion()
    trial.clear_occlusion()

    for index in range(2):
        decision = trial.observe(
            stabilizing(stamp_ns=(index + 2) * 100_000_000)
        )
        assert decision['state'] == REACQUIRING
        assert decision['reacquire_count'] == index + 1

    passed = trial.observe(stabilizing(stamp_ns=400_000_000))
    assert passed['state'] == FLOW_PASS_NO_LIGHT_EXPECTED


def test_ambiguous_stabilizing_frame_resets_reacquisition_count():
    trial = M0BTrial(M0BTrialConfig(required_reacquire_frames=3))
    trial.observe(target())
    trial.lock()
    trial.enter_occlusion()
    trial.clear_occlusion()
    trial.observe(stabilizing(stamp_ns=200_000_000))

    ambiguous = stabilizing(stamp_ns=300_000_000)
    ambiguous['matching_label_count'] = 2
    decision = trial.observe(ambiguous)

    assert decision['state'] == REACQUIRING
    assert decision['reacquire_count'] == 0
    assert 'unique stable' in decision['failure_reason']


def test_wrong_geometry_never_completes_reacquisition():
    trial = M0BTrial(M0BTrialConfig(required_reacquire_frames=2))
    trial.observe(target())
    trial.lock()
    trial.enter_occlusion()
    trial.clear_occlusion()

    wrong = trial.observe(
        target(
            stamp_ns=200_000_000,
            x=0.30,
            bbox=[400.0, 200.0, 460.0, 260.0],
        )
    )
    assert wrong['state'] == REACQUIRING
    assert wrong['reacquire_count'] == 0
    assert 'does not match' in wrong['failure_reason']


def test_lock_fails_closed_without_ready_target_and_reset_returns_acquiring():
    trial = M0BTrial()
    failed = trial.lock()
    assert failed['state'] == ACQUIRING
    assert failed['frozen_target'] is None

    trial.observe(target())
    trial.lock()
    trial.reset()
    assert trial.snapshot()['state'] == ACQUIRING
    assert trial.snapshot()['frozen_target'] is None


def test_stale_ready_target_cannot_be_locked():
    trial = M0BTrial(M0BTrialConfig(maximum_lock_age_sec=1.0))
    trial.observe(target(stamp_ns=100_000_000))
    stale = trial.lock(current_source_stamp_ns=1_200_000_000)
    assert stale['state'] == ACQUIRING
    assert stale['frozen_target'] is None
    assert 'stale' in stale['failure_reason']


def test_recent_ready_snapshot_survives_compatible_stabilizing_frame():
    trial = M0BTrial(M0BTrialConfig(maximum_lock_age_sec=1.0))
    trial.observe(target(stamp_ns=100_000_000))
    held = trial.observe(stabilizing(stamp_ns=500_000_000))
    assert held['state'] == READY_TO_LOCK
    locked = trial.lock(current_source_stamp_ns=500_000_000)
    assert locked['state'] == TARGET_LOCKED


def test_postcheck_ttl_expires_fail_closed():
    clock = FakeClock(10.0)
    trial = M0BTrial(
        M0BTrialConfig(
            required_reacquire_frames=2,
            maximum_postcheck_duration_sec=1.0,
        ),
        clock=clock,
    )
    trial.observe(target(stamp_ns=100_000_000))
    trial.lock()
    trial.enter_occlusion()
    trial.clear_occlusion(source_stamp_ns=200_000_000)
    clock.advance(1.001)
    expired = trial.observe(target(stamp_ns=1_300_000_000))
    assert expired['state'] == FAILED
    assert expired['motion_authorized'] is False


def test_postcheck_ttl_expires_deterministically_during_input_gap():
    clock = FakeClock(20.0)
    trial = M0BTrial(
        M0BTrialConfig(maximum_postcheck_duration_sec=1.0),
        clock=clock,
    )
    trial.observe(target())
    trial.lock()
    trial.enter_occlusion()
    trial.clear_occlusion(source_stamp_ns=200_000_000)

    clock.advance(1.0)
    assert trial.snapshot()['state'] == REACQUIRING
    clock.advance(0.001)
    expired = trial.snapshot()

    assert expired['state'] == FAILED
    assert expired['failure_reason'] == 'postcheck TTL expired'
    assert expired['transitions'][-1]['event'] == 'postcheck_ttl_expired'
    assert expired['motion_authorized'] is False


def test_source_stamp_gap_does_not_spuriously_expire_fresh_postcheck():
    clock = FakeClock(30.0)
    trial = M0BTrial(
        M0BTrialConfig(
            required_reacquire_frames=2,
            maximum_postcheck_duration_sec=1.0,
        ),
        clock=clock,
    )
    trial.observe(target(stamp_ns=100_000_000))
    trial.lock()
    trial.enter_occlusion()
    trial.clear_occlusion(source_stamp_ns=200_000_000)

    clock.advance(0.1)
    fresh = trial.observe(target(stamp_ns=20_000_000_000, x=0.091))

    assert fresh['state'] == REACQUIRING
    assert fresh['reacquire_count'] == 1
    assert fresh['failure_reason'] == ''
