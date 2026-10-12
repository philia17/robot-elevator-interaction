import math
import os
from pathlib import Path

import pytest

from press.wrist_button_diagnostic import (
    CompletedPrecontactEvidence,
    POST_PRE_SUCCESS_REASON,
    ProjectionVerification,
    WristButtonCandidate,
    WristDetectorObservation,
    WristTargetLock,
    WristTargetTracker,
    admit_candidates_in_acquisition_roi,
    admit_button_sized_candidates,
    boot_completed_precontact_evidence,
    completed_precontact_gate,
    exclude_touch_overlapping_candidates,
    exclude_touch_overlapping_circles,
    extract_circle_candidates,
    extract_shape_candidates,
    merge_shape_candidates,
    provisional_nearest_candidate,
    projected_acquisition_roi,
    verify_candidate_against_touch_geometry,
    verify_glyph_one,
    verify_candidate_against_projection,
)


def candidate(center, confidence=0.9, half_size=10.0):
    u, v = center
    return WristButtonCandidate(
        (u - half_size, v - half_size, u + half_size, v + half_size),
        confidence,
    )


def completed_gate_inputs():
    positions = (0.1, -0.2, 0.3, 0.4, -0.5, 0.0)
    evidence = CompletedPrecontactEvidence(
        object(), {}, 10.0, "cart", "session", "plan", "target", "cal",
        positions, "executor", False,
    )
    qc = {
        "phase": "TERMINAL",
        "terminal_reason": POST_PRE_SUCCESS_REASON,
        "cart_pose_epoch": "cart",
        "session_id": "session",
        "plan_hash": "plan",
        "prepared_snapshot_sha256": "target",
        "calibration_hash": "cal",
    }
    executor = {
        "state": "IDLE",
        "instance_nonce": "executor",
        "precontact_lease_active": False,
        "terminal_armed": False,
        "joint_target_publishers": 0,
        "resource_counts": {
            "action_call": 0, "arm_client": 0, "lease_client": 0,
            "publisher": 0, "safe_driver_client": 0, "service_call": 0,
        },
    }
    feedback = {
        "state": "CONVERGED_DISARMED", "armed": False, "fault": "",
        "torque_enabled": [True] * 6, "positions_rad": list(positions),
    }
    recent = (positions, positions, positions)
    return evidence, qc, executor, feedback, recent


def boot_adoption_inputs():
    _evidence, qc, executor, feedback, recent = completed_gate_inputs()
    qc["frozen_plan_summary"] = {
        "plan_hash": "plan", "calibration_hash": "cal",
        "p_button_base_m": [0.2, 0.1, 0.3],
        "p_pre_base_m": [0.13, 0.1, 0.3],
        "normal_out_base": [-1.0, 0.0, 0.0],
    }
    qc["executor_instance_nonce"] = "executor"
    return qc, executor, feedback, recent


def adopt_boot(qc, executor, feedback, recent, **kwargs):
    return boot_completed_precontact_evidence(
        qc, executor, feedback, 0.1, recent,
        boot_started_monotonic_s=10.0,
        **kwargs,
    )


def test_boot_adoption_accepts_three_stationary_samples_display_only():
    qc, executor, feedback, recent = boot_adoption_inputs()
    evidence, reason = adopt_boot(qc, executor, feedback, recent)
    assert reason == "BOOT_QC_SUMMARY_DISPLAY_ONLY_ADOPTED"
    assert evidence is not None
    assert evidence.frozen_package is None
    assert evidence.motion_authorized is False
    assert evidence.adoption_source == "QC_SUMMARY_BOOT_ADOPTION"
    assert evidence.display_target_base_m == (0.2, 0.1, 0.3)
    assert evidence.completed_monotonic_s == 10.0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda values: values["qc"].update(prepared_snapshot_sha256=""), "BOOT_TERMINAL_IDENTITY_MISSING"),
        (lambda values: values["qc"].update(session_id=""), "BOOT_TERMINAL_IDENTITY_MISSING"),
        (lambda values: values["qc"].update(cart_pose_epoch=""), "BOOT_TERMINAL_IDENTITY_MISSING"),
        (lambda values: values["qc"]["frozen_plan_summary"].update(plan_hash="wrong"), "BOOT_PLAN_HASH_MISMATCH"),
        (lambda values: values["qc"]["frozen_plan_summary"].update(calibration_hash="wrong"), "BOOT_CALIBRATION_MISMATCH"),
        (lambda values: values["qc"]["frozen_plan_summary"].pop("p_button_base_m"), "BOOT_SUMMARY_P_BUTTON_BASE_M_INVALID"),
        (lambda values: values["qc"]["frozen_plan_summary"].update(p_button_base_m=[0.1, 0.2]), "BOOT_SUMMARY_P_BUTTON_BASE_M_INVALID"),
        (lambda values: values["qc"]["frozen_plan_summary"].update(p_button_base_m=[0.1, float("nan"), 0.3]), "BOOT_SUMMARY_P_BUTTON_BASE_M_NONFINITE"),
        (lambda values: values["qc"].update(phase="IDLE"), "LEFT_PRE_TERMINAL"),
        (lambda values: values["qc"].update(terminal_reason="wrong"), "PRE_TERMINAL_NOT_ACCEPTED"),
        (lambda values: values["qc"].update(executor_instance_nonce="other"), "BOOT_EXECUTOR_IDENTITY_MISMATCH"),
        (lambda values: values["executor"].update(state="RUNNING"), "EXECUTOR_NOT_IDLE"),
        (lambda values: values["executor"].update(precontact_lease_active=True), "EXECUTOR_LEASE_ACTIVE"),
        (lambda values: values["executor"].update(terminal_armed=True), "EXECUTOR_ARMED"),
        (lambda values: values["executor"].update(joint_target_publishers=1), "JOINT_TARGET_PUBLISHER"),
        (lambda values: values["executor"].update(resource_counts={"publisher": 1}), "EXECUTOR_RESOURCES_ACTIVE"),
        (lambda values: values["feedback"].update(state="MOVING"), "SAFE_NOT_CONVERGED_DISARMED"),
        (lambda values: values["feedback"].update(armed=True), "SAFE_ARMED"),
        (lambda values: values["feedback"].update(fault="FAULT"), "SAFE_FAULT"),
        (lambda values: values["feedback"].update(torque_enabled=[True] * 5 + [False]), "SAFE_TORQUE_NOT_6_OF_6"),
    ),
)
def test_boot_adoption_fails_closed_on_identity_and_safety_mismatch(mutation, expected):
    qc, executor, feedback, recent = boot_adoption_inputs()
    values = {
        "qc": qc, "executor": executor, "feedback": feedback, "recent": recent,
    }
    mutation(values)
    evidence, reason = boot_completed_precontact_evidence(
        values["qc"], values["executor"], values["feedback"],
        0.1, values["recent"], boot_started_monotonic_s=10.0,
    )
    assert evidence is None
    assert reason == expected


def test_boot_adoption_requires_fresh_three_sample_stationary_window():
    qc, executor, feedback, recent = boot_adoption_inputs()
    evidence, reason = boot_completed_precontact_evidence(
        qc, executor, feedback, 6.0, recent,
        boot_started_monotonic_s=10.0,
    )
    assert evidence is None and reason == "SAFE_FEEDBACK_STALE"
    evidence, reason = adopt_boot(qc, executor, feedback, recent[:2])
    assert evidence is None and reason == "WAIT_STATIONARY_SAMPLES"
    moving = list(recent)
    moving[-1] = tuple(value + 0.01 for value in moving[-1])
    evidence, reason = adopt_boot(qc, executor, feedback, tuple(moving))
    assert evidence is None and reason == "POSTURE_NOT_STATIONARY"


def test_completed_precontact_gate_requires_terminal_cleanup_and_stationary_feedback():
    evidence, qc, executor, feedback, recent = completed_gate_inputs()
    assert completed_precontact_gate(
        evidence, qc, executor, feedback, 0.1, recent,
    ) == (True, "ACCEPTED_PRE_TERMINAL_STATIONARY", False)

    assert completed_precontact_gate(
        evidence, qc, executor, feedback, 6.0, recent,
    ) == (False, "SAFE_FEEDBACK_STALE", False)

    moving = list(recent)
    moving[-1] = tuple(value + 0.01 for value in recent[-1])
    assert completed_precontact_gate(
        evidence, qc, executor, feedback, 0.1, tuple(moving),
    ) == (False, "POSTURE_NOT_STATIONARY", False)


@pytest.mark.parametrize(
    ("group", "key", "value", "reason", "expected_invalidate"),
    (
        ("qc", "session_id", "other", "PRE_IDENTITY_CHANGED", True),
        ("qc", "phase", "IDLE", "LEFT_PRE_TERMINAL", True),
        ("executor", "state", "RUNNING", "EXECUTOR_NOT_IDLE", False),
        ("executor", "precontact_lease_active", True, "EXECUTOR_LEASE_ACTIVE", False),
        ("executor", "terminal_armed", True, "EXECUTOR_ARMED", False),
        ("feedback", "state", "MOVING", "SAFE_NOT_CONVERGED_DISARMED", True),
        ("feedback", "armed", True, "SAFE_ARMED", True),
        ("feedback", "fault", "FAULT", "SAFE_FAULT", True),
    ),
)
def test_completed_precontact_identity_and_safety_loss_invalidates(
    group, key, value, reason, expected_invalidate,
):
    evidence, qc, executor, feedback, recent = completed_gate_inputs()
    values = {"qc": qc, "executor": executor, "feedback": feedback}
    values[group][key] = value
    enabled, observed_reason, invalidate = completed_precontact_gate(
        evidence, qc, executor, feedback, 0.1, recent,
    )
    assert enabled is False
    assert observed_reason == reason
    assert invalidate is expected_invalidate


def test_completed_precontact_posture_drift_invalidates():
    evidence, qc, executor, feedback, recent = completed_gate_inputs()
    feedback["positions_rad"][2] += 0.02
    assert completed_precontact_gate(
        evidence, qc, executor, feedback, 0.1, recent,
    ) == (False, "POSTURE_DRIFT", True)


def observation(sequence, candidates, received=10.0):
    return WristDetectorObservation(
        sequence,
        received,
        (640, 480),
        candidates=tuple(candidates),
    )


def test_provisional_rule_uses_nearest_geometry_not_button_number():
    selected, reason = provisional_nearest_candidate(
        (candidate((300, 200)), candidate((300, 350))),
        (305, 390),
        maximum_distance_px=220,
        runner_up_margin_px=40,
    )
    assert selected.center_px == (300, 350)
    assert reason == "PROVISIONAL_NEAREST_TO_TOUCH"


def test_near_tie_is_ambiguous():
    selected, reason = provisional_nearest_candidate(
        (candidate((280, 300)), candidate((320, 300))),
        (300, 360),
        maximum_distance_px=220,
        runner_up_margin_px=40,
    )
    assert selected is None
    assert reason == "AMBIGUOUS_NEAREST_TO_TOUCH"


def test_five_stable_frames_expose_w_target_and_pixel_error():
    tracker = WristTargetTracker()
    result = None
    for sequence, center in enumerate(
        ((300, 350), (301, 350), (300, 351), (299, 350), (300, 349)), start=1,
    ):
        result = tracker.update(
            observation(sequence, [candidate(center)]),
            (300.0, 400.0),
            now_monotonic_s=10.1,
        )
    assert result.available is True
    assert result.center_px == (300, 350)
    assert result.dx_px == 0.0
    assert result.dy_px == -50.0
    assert result.stable_count == 5
    assert result.spread_px <= 3.0


def test_no_detection_and_stale_result_clear_previous_target():
    tracker = WristTargetTracker(stable_required=1)
    available = tracker.update(
        observation(1, [candidate((300, 350))]),
        (300.0, 400.0),
        now_monotonic_s=10.1,
    )
    assert available.available is True

    missing = tracker.update(
        observation(2, []),
        (300.0, 400.0),
        now_monotonic_s=10.1,
    )
    assert missing.available is False
    assert missing.reason == "NO_DETECTION"

    stale = tracker.update(
        observation(3, [candidate((300, 350))]),
        (300.0, 400.0),
        now_monotonic_s=11.0,
    )
    assert stale.available is False
    assert stale.reason == "INFERENCE_STALE"


def test_touch_overlapping_tool_circle_is_excluded_before_association():
    button_2 = WristButtonCandidate((290, 40, 402, 152), None, source="CIRCLE", radius_px=56)
    button_1 = WristButtonCandidate((299, 201, 393, 295), None, source="CIRCLE", radius_px=47)
    tool = WristButtonCandidate((239, 311, 369, 441), None, source="CIRCLE", radius_px=65)
    usable, excluded = exclude_touch_overlapping_circles(
        (button_2, button_1, tool), (304.0, 397.0),
    )
    assert usable == (button_2, button_1)
    assert excluded == (tool,)
    selected, reason = provisional_nearest_candidate(
        usable,
        (304.0, 397.0),
        maximum_distance_px=220,
        runner_up_margin_px=40,
    )
    assert selected is button_1
    assert reason == "PROVISIONAL_NEAREST_TO_TOUCH"


def test_user_screenshot_proxy_circle_debug_is_stable_at_deployed_thresholds():
    cv2 = pytest.importorskip("cv2")
    screenshot_path = os.environ.get("WRIST_SCREENSHOT_PROXY")
    if not screenshot_path:
        pytest.skip("WRIST_SCREENSHOT_PROXY not provided")
    screenshot = cv2.imread(str(Path(screenshot_path)))
    assert screenshot is not None
    wrist_proxy = cv2.resize(
        screenshot[480:1350, 1635:2795], (640, 480), interpolation=cv2.INTER_AREA,
    )
    touch = (304.0, 397.0)
    selected_centers = []
    for threshold in (34.0, 38.0):
        circles = extract_circle_candidates(
            wrist_proxy, accumulator_threshold=threshold,
        )
        usable, excluded = exclude_touch_overlapping_circles(circles, touch)
        assert len(circles) == 3
        assert len(excluded) == 1
        assert len(usable) == 2
        selected, reason = provisional_nearest_candidate(
            usable,
            touch,
            maximum_distance_px=220,
            runner_up_margin_px=40,
        )
        assert reason == "PROVISIONAL_NEAREST_TO_TOUCH"
        assert selected is not None
        selected_centers.append(selected.center_px)
    assert all(abs(center[0] - 347.4) <= 3.0 for center in selected_centers)
    assert all(abs(center[1] - 246.6) <= 3.0 for center in selected_centers)

    tracker = WristTargetTracker()
    result = None
    for sequence in range(1, 6):
        result = tracker.update(
            WristDetectorObservation(
                sequence, 10.0 + sequence * 0.1, (640, 480),
                candidates=(selected,), circle_candidates=tuple(usable),
            ),
            touch,
            now_monotonic_s=10.0 + sequence * 0.1,
        )
    assert result is not None and result.available is True
    assert result.stable_count == 5


def test_shape_candidates_support_circle_square_and_rounded_rectangle():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((480, 640, 3), 35, dtype=np.uint8)
    cv2.circle(frame, (320, 95), 48, (235, 235, 235), 5)
    cv2.rectangle(frame, (270, 195), (370, 285), (235, 235, 235), 5)
    cv2.rectangle(frame, (258, 350), (382, 472), (235, 235, 235), -1)
    candidates = extract_shape_candidates(frame)
    centers = [item.center_px for item in candidates]
    assert any(abs(x - 320) < 5 and abs(y - 95) < 5 for x, y in centers)
    assert any(abs(x - 320) < 5 and abs(y - 240) < 10 for x, y in centers)
    assert any(abs(x - 320) < 5 and abs(y - 411) < 10 for x, y in centers)
    usable, excluded = exclude_touch_overlapping_candidates(
        candidates, (320.0, 410.0),
    )
    assert len(excluded) == 1
    assert len(usable) >= 2


def test_yolo_and_contour_candidates_are_deduplicated():
    yolo = WristButtonCandidate((270, 200, 370, 290), 0.8, source="YOLO")
    contour = WristButtonCandidate(
        (268, 198, 372, 292), None, source="CONTOUR", shape="RECT",
    )
    other = WristButtonCandidate(
        (270, 60, 370, 160), None, source="CONTOUR", shape="ROUND",
    )
    merged = merge_shape_candidates((yolo,), (contour, other))
    assert len(merged) == 2
    assert yolo in merged


def test_a_projection_is_primary_and_wrong_side_or_ambiguity_fails():
    button_2 = candidate((320, 100), half_size=45)
    button_1 = candidate((320, 245), half_size=45)
    verified = verify_candidate_against_projection(
        (button_2, button_1), (320, 400), (322, 247),
    )
    assert verified.candidate is button_1
    assert verified.reason == "A_PROJ_VERIFIED"

    wrong_side = verify_candidate_against_projection(
        (candidate((320, 395)),), (320, 400), (320, 395),
    )
    assert wrong_side.candidate is None
    assert wrong_side.reason == "WRONG_SIDE_OF_TOUCH"

    ambiguous = verify_candidate_against_projection(
        (candidate((300, 245)), candidate((320, 245))),
        (310, 400), (310, 245), projection_runner_up_margin_px=18,
    )
    assert ambiguous.candidate is None
    assert ambiguous.reason == "A_PROJ_AMBIGUOUS"


def test_out_of_image_fallback_uses_touch_geometry_and_latches_three_of_five():
    touch = (320.0, 400.0)
    tool = WristButtonCandidate(
        (255.0, 335.0, 385.0, 465.0), None,
        source="CIRCLE", radius_px=65.0, shape="CIRCLE",
    )
    button_1 = candidate((320.0, 250.0), half_size=48.0)
    verification = verify_candidate_against_touch_geometry(
        (tool, button_1), touch,
    )
    assert verification.candidate is button_1
    assert verification.reason == "TOUCH_GEOMETRY_PROVISIONAL"
    assert verification.projection_residual_px == 0.0

    tracker = WristTargetLock()
    identity = ("panel", "acquire-1")
    contact_reference = (345.0, 300.0)
    for sequence in range(1, 4):
        locked = tracker.observe_verified(
            verification,
            sequence=sequence,
            frame_stamp_ns=sequence,
            # Display fallback may bind its legacy projection marker to the
            # candidate, but the frozen motion reference must remain the
            # independently projected panel hit.
            projected_target_px=button_1.center_px,
            contact_reference_px=contact_reference,
            identity=identity,
            target_base_m=(0.2, 0.1, 0.3),
            feedback_positions_rad=(0.0,) * 6,
        )
    assert locked is not None and locked.stage == "PRE-LOCK"
    assert locked.acquired_projection_px == button_1.center_px
    assert locked.contact_reference_px == contact_reference
    assert locked.center_px != locked.contact_reference_px


@pytest.mark.parametrize(
    ("items", "expected"),
    (
        ((candidate((320.0, 385.0)),), "WRONG_SIDE_OF_TOUCH"),
        ((candidate((320.0, 250.0)), candidate((350.0, 255.0))), "AMBIGUOUS_NEAREST_TO_TOUCH"),
        ((candidate((320.0, 100.0)),), "NO_CANDIDATE_NEAR_TOUCH"),
    ),
)
def test_touch_geometry_fallback_rejects_wrong_side_ambiguity_and_far_candidate(
    items, expected,
):
    verification = verify_candidate_against_touch_geometry(items, (320.0, 400.0))
    assert verification.candidate is None
    assert verification.reason == expected
    tracker = WristTargetLock()
    assert tracker.observe_verified(
        verification,
        sequence=1,
        frame_stamp_ns=1,
        projected_target_px=(320.0, 400.0),
        identity=("panel", "acquire"),
        target_base_m=(0.2, 0.1, 0.3),
        feedback_positions_rad=(0.0,) * 6,
    ) is None
    assert tracker.stable_count == 0


def test_three_of_five_latches_median_and_detection_miss_cannot_retarget():
    lock = WristTargetLock()
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    centers = ((320, 245), (321, 245), (500, 100), (319, 246), (318, 244))
    result = None
    for sequence, center in enumerate(centers, start=1):
        item = candidate(center)
        result = lock.observe_verified(
            ProjectionVerification(item, "A_PROJ_VERIFIED", 2.0),
            sequence=sequence,
            frame_stamp_ns=sequence * 100,
            projected_target_px=(320.0, 245.0),
            identity=identity,
            target_base_m=target_base,
            feedback_positions_rad=(0.0,) * 6,
        )
    assert result is not None
    assert result.stage == "PRE-LOCK"
    assert result.center_px == (320.0, 245.0)
    frozen = result
    missed = lock.observe_verified(
        ProjectionVerification(None, "NO_CANDIDATE"),
        sequence=6,
        frame_stamp_ns=600,
        projected_target_px=(400.0, 100.0),
        identity=identity,
        target_base_m=(9.0, 9.0, 9.0),
        feedback_positions_rad=(0.0,) * 6,
    )
    assert missed is frozen
    assert lock.locked is frozen
    assert frozen.target_base_m == target_base


def test_three_of_five_source_switch_does_not_reset_acquisition_window():
    """YOLO/circle/contour evidence for one identity shares the same 3-of-5 lock."""
    lock = WristTargetLock()
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    result = None
    for sequence, source in enumerate(("YOLO", "CIRCLE", "CONTOUR"), start=1):
        item = WristButtonCandidate(
            (270.0, 195.0, 370.0, 295.0), None,
            source=source, shape=source,
        )
        result = lock.observe_verified(
            ProjectionVerification(item, "A_PROJ_VERIFIED", 1.0),
            sequence=sequence,
            frame_stamp_ns=sequence,
            projected_target_px=(320.0, 245.0),
            identity=identity,
            target_base_m=target_base,
            feedback_positions_rad=(0.0,) * 6,
        )
    assert result is not None and result.stage == "PRE-LOCK"
    assert result.center_px == (320.0, 245.0)


def test_three_of_five_counts_failed_observations_in_the_window():
    lock = WristTargetLock()
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    valid = ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.0)
    missing = ProjectionVerification(None, "NO_CANDIDATE")
    sequence_items = (valid, valid, missing, missing, missing, valid)
    result = None
    for sequence, item in enumerate(sequence_items, start=1):
        result = lock.observe_verified(
            item, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=(0.20, 0.10, 0.30),
            feedback_positions_rad=(0.0,) * 6,
        )
    assert result is None
    assert lock.locked is None
    assert lock.stable_count == 2


def test_duplicate_sequence_does_not_count_twice_toward_three_of_five():
    lock = WristTargetLock()
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    verified = ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.0)
    assert lock.observe_verified(
        verified, sequence=1, frame_stamp_ns=1,
        projected_target_px=(320, 245), identity=identity,
        target_base_m=(0.20, 0.10, 0.30), feedback_positions_rad=(0.0,) * 6,
    ) is None
    assert lock.observe_verified(
        verified, sequence=1, frame_stamp_ns=99,
        projected_target_px=(320, 245), identity=identity,
        target_base_m=(9.0, 9.0, 9.0), feedback_positions_rad=(0.0,) * 6,
    ) is None
    assert lock.stable_count == 1
    assert lock.observe_verified(
        verified, sequence=2, frame_stamp_ns=2,
        projected_target_px=(320, 245), identity=identity,
        target_base_m=(0.20, 0.10, 0.30), feedback_positions_rad=(0.0,) * 6,
    ) is None
    locked = lock.observe_verified(
        verified, sequence=3, frame_stamp_ns=3,
        projected_target_px=(320, 245), identity=identity,
        target_base_m=(0.20, 0.10, 0.30), feedback_positions_rad=(0.0,) * 6,
    )
    assert locked is not None and locked.stage == "PRE-LOCK"


def test_future_press_lock_requires_pre_lock_and_is_immutable():
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    lock = WristTargetLock()
    verified = ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.0)
    assert lock.acquire_press_lock(
        verified, sequence=1, frame_stamp_ns=1,
        projected_target_px=(320, 245), identity=identity,
        target_base_m=target_base,
        feedback_positions_rad=(0.0,) * 6,
    ) is None
    for sequence in range(1, 4):
        lock.observe_verified(
            verified, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base,
            feedback_positions_rad=(0.0,) * 6,
        )
    assert lock.begin_press_reacquire("press-click-1") is True
    pressed = None
    for sequence in range(4, 7):
        pressed = lock.acquire_press_lock(
            verified, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base,
            feedback_positions_rad=(0.0,) * 6,
        )
        if sequence < 6:
            assert pressed is None
            assert lock.locked is not None
            assert lock.locked.identity == identity
    assert pressed is not None and pressed.stage == "PRE-LOCK"
    assert pressed.identity == ("press-click-1", *identity)
    assert pressed.motion_authorized is False
    assert pressed.target_base_m == target_base
    assert lock.observe_verified(
        ProjectionVerification(candidate((500, 100)), "A_PROJ_VERIFIED", 0.0),
        sequence=7, frame_stamp_ns=7,
        projected_target_px=(500, 100), identity=identity,
        target_base_m=(9.0, 9.0, 9.0),
        feedback_positions_rad=(0.0,) * 6,
    ) is pressed
    assert lock.mark_future_motion_active().future_motion_active is True
    assert lock.locked.motion_authorized is False


def test_press_reacquire_failed_samples_count_in_independent_window():
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    lock = WristTargetLock()
    valid = ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.0)
    missing = ProjectionVerification(None, "NO_CANDIDATE")
    for sequence in range(1, 4):
        lock.observe_verified(
            valid, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )
    assert lock.begin_press_reacquire("press-click-2") is True
    result = None
    for sequence, item in enumerate(
        (valid, valid, missing, missing, missing, valid), start=10,
    ):
        result = lock.acquire_press_lock(
            item, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )
    assert result is None
    assert lock.locked is not None and lock.locked.stage == "PRE-LOCK"
    assert lock.last_reason == "PRESS_PENDING_3_OF_5"


def test_press_lock_pairs_latest_center_with_latest_projection_pose_and_contact():
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target = (.20, .10, .30)
    lock = WristTargetLock()
    for seq in range(1, 4):
        lock.observe_verified(
            ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.),
            sequence=seq, frame_stamp_ns=seq, projected_target_px=(330, 260),
            identity=identity, target_base_m=target, feedback_positions_rad=(0.,) * 6,
            contact_reference_px=(300, 290), projection_context_json='{"pose":"old"}',
        )
    assert lock.begin_press_reacquire("fresh-click")
    for seq, center in enumerate(((320, 245), (321, 246), (319, 244)), start=10):
        acquired = lock.acquire_press_lock(
            ProjectionVerification(candidate(center), "A_PROJ_VERIFIED", 1.),
            sequence=seq, frame_stamp_ns=seq, projected_target_px=(331, 261),
            identity=identity, target_base_m=target, feedback_positions_rad=(.001,) * 6,
            contact_reference_px=(301, 291), projection_context_json='{"pose":"fresh"}',
        )
    assert acquired.center_px == (319., 244.)
    assert acquired.acquired_projection_px == (331., 261.)
    assert acquired.feedback_positions_rad == (.001,) * 6
    assert acquired.contact_reference_px == (301., 291.)
    assert acquired.projection_context_json == '{"pose":"fresh"}'


def test_press_reacquire_rejects_target_drift_and_preserves_frozen_target():
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    drifted_target = (0.40, 0.10, 0.30)
    lock = WristTargetLock()
    valid = ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.0)
    for sequence in range(1, 4):
        lock.observe_verified(
            valid, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )
    frozen = lock.locked
    assert frozen is not None and frozen.target_base_m == target_base
    assert lock.begin_press_reacquire("press-click-3") is True
    for sequence in range(10, 13):
        assert lock.acquire_press_lock(
            valid, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(500, 100), identity=identity,
            target_base_m=drifted_target, feedback_positions_rad=(0.0,) * 6,
        ) is None
    assert lock.locked is frozen
    assert lock.locked.target_base_m == target_base


def test_press_reacquire_rejects_button2_and_keeps_departure_button1_anchor():
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    lock = WristTargetLock()
    button1 = ProjectionVerification(candidate((320, 245)), "A_PROJ_VERIFIED", 1.0)
    for sequence in range(1, 4):
        lock.observe_verified(
            button1, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )
    departure = lock.locked
    assert departure is not None and departure.center_px == (320.0, 245.0)
    assert lock.begin_press_reacquire("press-click-button1") is True

    button2 = ProjectionVerification(candidate((320, 150.6)), "A_PROJ_VERIFIED", 1.0)
    for sequence in range(10, 13):
        assert lock.acquire_press_lock(
            button2, sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 150.6), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        ) is None

    assert lock.locked is departure
    assert lock.locked.center_px == (320.0, 245.0)
    assert lock.last_reason == "PRESS_CANDIDATE_OUTSIDE_DEPARTURE_ANCHOR"
    assert lock.stable_count == 0


def test_press_reacquire_accepts_fresh_same_button_jitter_and_binds_click_id():
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    target_base = (0.20, 0.10, 0.30)
    lock = WristTargetLock()
    for sequence, center in enumerate(((320, 245), (321, 244), (319, 246)), start=1):
        lock.observe_verified(
            ProjectionVerification(candidate(center), "A_PROJ_VERIFIED", 1.0),
            sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )
    assert lock.begin_press_reacquire("press-click-current") is True

    acquired = None
    for sequence, center in enumerate(((321, 245), (320, 246), (319, 244)), start=10):
        acquired = lock.acquire_press_lock(
            ProjectionVerification(candidate(center), "A_PROJ_VERIFIED", 1.0),
            sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )

    assert acquired is not None and acquired.stage == "PRE-LOCK"
    assert acquired.identity == ("press-click-current", *identity)
    assert math.dist(acquired.center_px, (320.0, 245.0)) <= 2.0
    frozen = acquired
    assert lock.acquire_press_lock(
        ProjectionVerification(candidate((320, 150)), "A_PROJ_VERIFIED", 1.0),
        sequence=20, frame_stamp_ns=20,
        projected_target_px=(320, 150), identity=identity,
        target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
    ) is frozen

    assert lock.begin_press_reacquire("press-click-next") is True
    assert lock.stable_count == 0
    second = None
    for sequence, center in enumerate(((320, 245), (321, 246), (319, 244)), start=30):
        second = lock.acquire_press_lock(
            ProjectionVerification(candidate(center), "A_PROJ_VERIFIED", 1.0),
            sequence=sequence, frame_stamp_ns=sequence,
            projected_target_px=(320, 245), identity=identity,
            target_base_m=target_base, feedback_positions_rad=(0.0,) * 6,
        )
    assert second is not None
    assert second.identity == ("press-click-next", *identity)
    assert "press-click-current" not in second.identity


def test_user_screenshot_proxy_shape_candidate_finds_button1():
    cv2 = pytest.importorskip("cv2")
    screenshot_path = os.environ.get("WRIST_SCREENSHOT_PROXY")
    if not screenshot_path:
        pytest.skip("WRIST_SCREENSHOT_PROXY not provided")
    screenshot = cv2.imread(str(Path(screenshot_path)))
    assert screenshot is not None
    wrist_proxy = cv2.resize(
        screenshot[483:1348, 1642:2790], (640, 480),
        interpolation=cv2.INTER_AREA,
    )
    candidates = extract_shape_candidates(wrist_proxy)
    assert any(
        abs(item.center_px[0] - 349.0) <= 5.0
        and abs(item.center_px[1] - 252.0) <= 5.0
        for item in candidates
    )


def test_projected_roi_and_admission_reject_partial_tape_tool_and_button2():
    projected = (371.0, 327.0)
    touch = (303.0, 379.0)
    roi = projected_acquisition_roi((640, 480), projected, touch)
    assert roi == (261, 147, 481, 361)
    button_1 = WristButtonCandidate(
        (294.0, 194.0, 400.0, 300.0), None,
        source="CIRCLE", radius_px=53.0, shape="CIRCLE",
    )
    button_2 = WristButtonCandidate(
        (290.0, 34.0, 402.0, 146.0), None,
        source="CIRCLE", radius_px=56.0, shape="CIRCLE",
    )
    partial = WristButtonCandidate(
        (260.0, 210.0, 330.0, 285.0), None,
        source="CONTOUR", shape="ROUNDED_RECT",
    )
    tape = WristButtonCandidate(
        (300.0, 230.0, 450.0, 255.0), None,
        source="CONTOUR", shape="RECT",
    )
    tool = WristButtonCandidate(
        (240.0, 330.0, 366.0, 456.0), None,
        source="CIRCLE", radius_px=63.0, shape="CIRCLE",
    )
    admitted = admit_candidates_in_acquisition_roi(
        (button_1, button_2, partial, tape, tool), roi, touch, projected,
    )
    assert admitted == (button_1,)
    assert verify_candidate_against_projection(
        admitted, touch, projected,
    ).reason == "A_PROJ_ACQUISITION_VERIFIED"


def test_first_projected_roi_contains_full_current_button1_bbox_only():
    projected = (360.0, 330.0)
    touch = (303.0, 405.0)
    button_1 = WristButtonCandidate(
        (292.0, 171.0, 416.0, 336.0), None,
        source="CIRCLE", radius_px=62.0, shape="CIRCLE",
    )
    button_2 = WristButtonCandidate(
        (294.0, 35.0, 414.0, 155.0), None,
        source="CIRCLE", radius_px=60.0, shape="CIRCLE",
    )
    partial = WristButtonCandidate(
        (260.0, 190.0, 330.0, 265.0), None,
        source="CONTOUR", shape="ROUNDED_RECT",
    )
    tape = WristButtonCandidate(
        (280.0, 220.0, 450.0, 250.0), None,
        source="CONTOUR", shape="RECT",
    )
    touch_ring = WristButtonCandidate(
        (268.0, 370.0, 338.0, 440.0), None,
        source="CIRCLE", radius_px=35.0, shape="CIRCLE",
    )
    tool = WristButtonCandidate(
        (240.0, 335.0, 366.0, 461.0), None,
        source="CIRCLE", radius_px=63.0, shape="CIRCLE",
    )
    candidates = (button_1, button_2, partial, tape, touch_ring, tool)

    old_roi = (250, 190, 470, 375)
    assert admit_candidates_in_acquisition_roi(
        candidates, old_roi, touch, projected,
    ) == ()

    roi = projected_acquisition_roi((640, 480), projected, touch)
    assert roi == (250, 150, 470, 375)
    assert button_1.center_px == (354.0, 253.5)
    assert math.dist(button_1.center_px, projected) == pytest.approx(76.735)
    admitted = admit_candidates_in_acquisition_roi(
        candidates, roi, touch, projected,
    )
    assert admitted == (button_1,)
    assert verify_candidate_against_projection(
        admitted, touch, projected,
    ).reason == "A_PROJ_ACQUISITION_VERIFIED"


def test_departure_pre_lock_touch_fallback_handles_screenshot_projection_bias():
    projected = (305.3, 311.8)
    touch = (301.3, 399.8)
    button_1 = WristButtonCandidate(
        (214.0, 157.0, 338.0, 284.0), None,
        source="CIRCLE", radius_px=62.0, shape="CIRCLE",
    )
    button_2 = WristButtonCandidate(
        (214.0, 8.0, 338.0, 135.0), None,
        source="CIRCLE", radius_px=62.0, shape="CIRCLE",
    )
    roi = projected_acquisition_roi((640, 480), projected, touch)
    assert roi == (195, 131, 416, 357)
    assert math.dist(button_1.center_px, projected) == pytest.approx(95.8863)
    assert admit_candidates_in_acquisition_roi(
        (button_1, button_2), roi, touch, projected,
    ) == ()

    fallback_candidates = admit_candidates_in_acquisition_roi(
        (button_1, button_2),
        roi,
        touch,
        projected,
        maximum_residual_px=float("inf"),
    )
    assert fallback_candidates == (button_1,)
    verification = verify_candidate_against_touch_geometry(
        fallback_candidates, touch,
    )
    assert verification.candidate is button_1
    assert verification.reason == "TOUCH_GEOMETRY_PROVISIONAL"

    tracker = WristTargetLock()
    locked = None
    for sequence in range(1, 4):
        locked = tracker.observe_verified(
            verification,
            sequence=sequence,
            frame_stamp_ns=sequence,
            projected_target_px=button_1.center_px,
            contact_reference_px=(309.0, 308.0),
            identity=("cart", "session", "plan", "target", "cal", "executor"),
            target_base_m=(0.2, 0.1, 0.3),
            feedback_positions_rad=(0.0,) * 6,
        )
    assert locked is not None
    assert locked.stage == "PRE-LOCK"
    assert locked.center_px == button_1.center_px

    button_2_only = verify_candidate_against_touch_geometry((button_2,), touch)
    assert button_2_only.candidate is None
    assert button_2_only.reason == "NO_CANDIDATE_NEAR_TOUCH"


def test_touch_fallback_size_gate_keeps_only_complete_button_proposals():
    button = candidate((350.0, 250.0), half_size=50.0)
    partial = WristButtonCandidate(
        (300.0, 220.0, 365.0, 252.0), None, source="CONTOUR", shape="ROUND",
    )
    wide_tape = WristButtonCandidate(
        (250.0, 220.0, 440.0, 270.0), None, source="CONTOUR", shape="RECT",
    )
    assert admit_button_sized_candidates((button, partial, wide_tape)) == (button,)


def _synthetic_button(shape, glyph):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((480, 640, 3), 28, dtype=np.uint8)
    if shape == "CIRCLE":
        cv2.circle(frame, (320, 240), 55, (225, 225, 225), 5)
        candidate_item = WristButtonCandidate(
            (265, 185, 375, 295), None, source="CIRCLE",
            radius_px=55, shape="CIRCLE",
        )
    else:
        candidate_item = WristButtonCandidate(
            (265, 185, 375, 295), None, source="CONTOUR", shape=shape,
        )
        radius = 18 if shape == "ROUNDED_RECT" else 1
        cv2.rectangle(frame, (265, 185), (375, 295), (225, 225, 225), 5)
        if radius > 1:
            cv2.circle(frame, (283, 203), radius, (225, 225, 225), 5)
    if glyph == "1":
        cv2.line(frame, (320, 208), (320, 270), (245, 245, 245), 9)
        cv2.line(frame, (307, 218), (320, 208), (245, 245, 245), 7)
        cv2.line(frame, (307, 270), (334, 270), (245, 245, 245), 7)
    elif glyph == "2":
        cv2.putText(
            frame, "2", (296, 270), cv2.FONT_HERSHEY_SIMPLEX,
            2.2, (245, 245, 245), 8, cv2.LINE_AA,
        )
    elif glyph == "GLARE":
        cv2.rectangle(frame, (295, 205), (345, 275), (245, 245, 245), -1)
    return frame, candidate_item


@pytest.mark.parametrize("shape", ("CIRCLE", "RECT", "ROUNDED_RECT"))
def test_lightweight_glyph_one_accepts_supported_outer_shapes(shape):
    frame, item = _synthetic_button(shape, "1")
    result = verify_glyph_one(frame, item)
    assert result.passed is True
    assert result.reason == "GLYPH_1_VERIFIED"


@pytest.mark.parametrize("glyph", ("2", "", "GLARE"))
def test_lightweight_glyph_one_rejects_button2_blank_and_glare(glyph):
    frame, item = _synthetic_button("CIRCLE", glyph)
    assert verify_glyph_one(frame, item).passed is False


def test_six_1635_screenshot_proxies_uniquely_admit_button1_and_can_lock():
    cv2 = pytest.importorskip("cv2")
    paths = [
        Path(value) for value in os.environ.get("WRIST_1635_PROXIES", "").split(os.pathsep)
        if value
    ]
    if not paths:
        pytest.skip("WRIST_1635_PROXIES not provided")
    assert len(paths) == 6
    projected = (371.0, 327.0)
    touch = (303.0, 379.0)
    tracker = WristTargetLock()
    locked = None
    for sequence, path in enumerate(paths, start=1):
        screenshot = cv2.imread(str(path))
        assert screenshot is not None
        frame = cv2.resize(
            screenshot[514:1217, 1424:2358], (640, 480),
            interpolation=cv2.INTER_AREA,
        )
        candidates = merge_shape_candidates(
            extract_circle_candidates(frame), extract_shape_candidates(frame),
        )
        usable, _excluded = exclude_touch_overlapping_candidates(candidates, touch)
        admitted = admit_candidates_in_acquisition_roi(
            usable,
            projected_acquisition_roi((640, 480), projected, touch),
            touch,
            projected,
        )
        assert len(admitted) == 1
        verification = verify_candidate_against_projection(
            admitted, touch, projected,
        )
        assert verification.candidate is admitted[0]
        glyph = verify_glyph_one(frame, admitted[0])
        assert glyph.passed is True
        locked = tracker.observe_verified(
            verification,
            sequence=sequence,
            frame_stamp_ns=sequence,
            projected_target_px=projected,
            identity=("cart", "session", "plan", "target", "cal", "executor"),
            target_base_m=(0.2, 0.1, 0.3),
            feedback_positions_rad=(0.0,) * 6,
        )
    assert locked is not None and locked.stage == "PRE-LOCK"


def test_glyph_failure_cannot_accumulate_three_of_five_lock_samples():
    tracker = WristTargetLock()
    valid_candidate = candidate((348, 248), half_size=53)
    identity = ("cart", "session", "plan", "target", "cal", "executor")
    for sequence in range(1, 6):
        verification = ProjectionVerification(
            valid_candidate if sequence in (1, 3) else None,
            "A_PROJ_VERIFIED" if sequence in (1, 3) else "GLYPH_NOT_1",
            3.0 if sequence in (1, 3) else None,
        )
        assert tracker.observe_verified(
            verification,
            sequence=sequence,
            frame_stamp_ns=sequence,
            projected_target_px=(371.0, 327.0),
            identity=identity,
            target_base_m=(0.2, 0.1, 0.3),
            feedback_positions_rad=(0.0,) * 6,
        ) is None
    assert tracker.locked is None
