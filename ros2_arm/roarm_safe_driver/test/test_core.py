import json
import math
import time

import pytest

from roarm_safe_driver.core import (
    ARM_COMMAND_DEADLINE_S,
    COARSE_MAX_WRONG_DIRECTION_RAD,
    COARSE_MIN_USEFUL_MOTION_RAD,
    COARSE_PRECONTACT_LEASE_PURPOSE,
    DEFAULT_JOINT_ACC_REGISTERS,
    DEFAULT_JOINT_MAX_DELTA_RAD,
    DEFAULT_JOINT_SPEED_STEPS_S,
    DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
    POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE,
    DYNAMIC_PRECONTACT_PID_I,
    DYNAMIC_PRECONTACT_PID_JOINT_IDS,
    DYNAMIC_PRECONTACT_PID_P,
    DYNAMIC_PRECONTACT_TOLERANCE_RAD,
    ENCODER_STEP_RAD,
    FEEDBACK_REQUEST,
    Feedback,
    IMPLAUSIBLE_JOINT_LIMITS_ERROR,
    INVALID_FEEDBACK_JOINT_LIMITS_FAULT,
    JOINT_NAMES,
    PRECONTACT_LEASE_PURPOSE,
    PRECONTACT_LEASE_SPEED_STEPS_S,
    PrecontactSpeedLease,
    SafeController,
    SafetyError,
    TERMINAL_MAX_SETTLE_SPAN_RAD,
    TERMINAL_MIN_SETTLE_SAMPLES,
    TERMINAL_SETTLE_WINDOW_S,
    T102_REISSUE_POST_SETTLE_DEADLINE_S,
    TORQUE_SWITCH_OFF_FAULT,
    build_t101,
    build_t102,
    build_joint_pid,
    build_reset_joint_pid,
    count_moving_joints,
    expected_motion_time_s,
    feedback_base_step,
    firmware_base_goal_step,
    ordered_positions,
    parse_feedback_line,
    validate_command_age,
    validate_joint_profiles,
    validate_multi_joint_target,
    validate_single_joint_target,
    validate_single_joint_target,
    validate_target,
)


SAMPLE = (
    '{"T":1051,"x":344.0635947,"y":-7.918,"z":206.33,"tit":0.05,'
    '"b":-0.023009712,"s":-0.003067962,"e":1.619883712,'
    '"t":0.013805827,"r":-0.003067962,"g":3.132388769,'
    '"tB":75,"tS":0,"tE":133,"tT":51,"tR":24,"tG":-16,'
    '"torswitchB":1,"torswitchS":1,"torswitchE":1,"torswitchT":1,'
    '"torswitchR":1,"torswitchG":1,"v":1212}'
)


def test_feedback_request_is_exactly_one_read_only_t105_line():
    assert FEEDBACK_REQUEST == b'{"T":105}\n'
    assert json.loads(FEEDBACK_REQUEST) == {"T": 105}
    assert b'101' not in FEEDBACK_REQUEST
    assert b'102' not in FEEDBACK_REQUEST


def feedback(now=1.0):
    return parse_feedback_line(SAMPLE, now=now)


def make_feedback(positions, now=1.0, torque=True, voltage=12.1):
    return Feedback(
        positions=tuple(positions),
        loads=(0, 0, 0, 0, 0, 0),
        torque_enabled=(torque,) * 6,
        voltage=voltage,
        received_at=now,
        raw={},
    )


def joint_target(start, index=0, degrees=2.0):
    result = list(start)
    result[index] += math.radians(degrees)
    return tuple(result)


def feed_stable_window(ctl, positions, start_at=1.2, period_s=0.05):
    """Feed exactly one full terminal-settle window, including both endpoints."""
    count = int(round(TERMINAL_SETTLE_WINDOW_S / period_s)) + 1
    for index in range(count):
        ctl.accept_feedback(
            make_feedback(positions, now=start_at + index * period_s)
        )
    return start_at + (count - 1) * period_s


def test_recorded_feedback_units_order_and_gripper_conversion():
    value = feedback()
    assert value.positions[:5] == pytest.approx(
        [-0.023009712, -0.003067962, 1.619883712, 0.013805827, -0.003067962]
    )
    assert value.positions[5] == pytest.approx(math.pi - 3.132388769)
    assert value.voltage == pytest.approx(12.12)
    assert all(value.torque_enabled)


@pytest.mark.parametrize("line", ["not json", "[]", '{"T":1051,"b":NaN}'])
def test_malformed_feedback_rejected(line):
    with pytest.raises(SafetyError):
        parse_feedback_line(line)


def test_unsolicited_unrelated_valid_json_is_ignored():
    assert parse_feedback_line('{"T":114,"ok":true}') is None


def test_command_names_are_mapped_not_assumed_ordered():
    names = list(reversed(JOINT_NAMES))
    positions = list(reversed(range(6)))
    assert ordered_positions(names, positions) == tuple(range(6))


def test_missing_duplicate_and_nan_command_rejected():
    with pytest.raises(SafetyError):
        ordered_positions(JOINT_NAMES[:-1], [0] * 5)
    with pytest.raises(SafetyError):
        ordered_positions([JOINT_NAMES[0]] * 6, [0] * 6)
    with pytest.raises(SafetyError):
        ordered_positions(JOINT_NAMES, [0, 0, math.nan, 0, 0, 0])


def test_command_timestamp_must_be_fresh_and_finite():
    validate_command_age(10.0, 10.4)
    for stamp, now in ((0.0, 1.0), (10.0, 10.6), (10.2, 10.0), (math.nan, 10.0)):
        with pytest.raises(SafetyError):
            validate_command_age(stamp, now)


def test_joint_limit_and_max_delta_rejected():
    measured = feedback().positions
    with pytest.raises(SafetyError, match="max delta"):
        validate_target((measured[0] + 0.2,) + measured[1:], measured, math.radians(5))
    bad = list(measured)
    bad[1] = 2.0
    with pytest.raises(SafetyError, match="joint limit"):
        validate_target(bad, measured, 3.0)


@pytest.mark.parametrize("selected", range(6))
def test_exactly_one_logical_joint_must_change(selected):
    measured = feedback().positions
    valid = list(joint_target(measured, selected, degrees=1.0))
    other = (selected + 1) % 6
    valid[other] += ENCODER_STEP_RAD
    checked, index = validate_single_joint_target(
        valid, measured, DEFAULT_JOINT_MAX_DELTA_RAD
    )
    assert checked == pytest.approx(valid)
    assert index == selected

    no_meaningful_change = list(measured)
    no_meaningful_change[selected] += ENCODER_STEP_RAD
    with pytest.raises(SafetyError, match="exactly one"):
        validate_single_joint_target(
            no_meaningful_change, measured, DEFAULT_JOINT_MAX_DELTA_RAD
        )

    multi_joint = list(valid)
    multi_joint[other] += ENCODER_STEP_RAD * 0.01
    with pytest.raises(SafetyError, match="exactly one"):
        validate_single_joint_target(
            multi_joint, measured, DEFAULT_JOINT_MAX_DELTA_RAD
        )


@pytest.mark.parametrize(
    "index,target,expected_rad",
    [
        (0, -0.05045092359971647, -0.05045092359971647),
        (1, 0.1, 0.1),
        (2, 1.5, 1.5),
        (3, -0.1, -0.1),
        (4, 0.2, 0.2),
        (5, 0.4, math.pi - 0.4),
    ],
)
def test_exact_t101_mapping_for_all_six_logical_joints(index, target, expected_rad):
    raw = build_t101(
        index,
        target,
        DEFAULT_JOINT_SPEED_STEPS_S[index],
        DEFAULT_JOINT_ACC_REGISTERS[index],
    )
    payload = json.loads(raw)
    assert raw.endswith(b"\n")
    assert list(payload) == ["T", "joint", "rad", "spd", "acc"]
    assert payload == {
        "T": 101,
        "joint": index + 1,
        "rad": pytest.approx(expected_rad),
        "spd": DEFAULT_JOINT_SPEED_STEPS_S[index],
        "acc": 1,
    }
    assert b'"T":102' not in raw


def test_exact_base_payload_is_unchanged():
    target = -0.05045092359971647
    raw = build_t101(0, target, 34, 1)
    assert raw.endswith(b"\n")
    assert raw == (
        b'{"T":101,"joint":1,"rad":-0.05045092359971647,'
        b'"spd":34,"acc":1}\n'
    )


def test_shoulder_is_one_logical_t101_and_physical_pair_stays_firmware_internal():
    payload = json.loads(build_t101(1, 0.1, 17, 1))
    assert payload["joint"] == 2
    assert set(payload) == {"T", "joint", "rad", "spd", "acc"}
    assert all(key not in payload for key in ("servo", "servo_id", "joint2", "joint3"))


def test_acceleration_register_units_and_zero_rejected():
    assert expected_motion_time_s(56 * ENCODER_STEP_RAD, 34, 1) == pytest.approx(
        56 / 34 + 34 / 100
    )
    with pytest.raises(SafetyError, match="register"):
        build_t101(0, 0.0, 34, 0)
    with pytest.raises(SafetyError, match="register"):
        expected_motion_time_s(ENCODER_STEP_RAD, 34, 0)
    with pytest.raises(SafetyError, match="no more than"):
        build_t101(0, 0.0, 35, 1)
    with pytest.raises(SafetyError, match="no more than"):
        build_t101(0, 0.0, 34, 11)


def test_per_joint_profile_length_type_and_hard_ceilings():
    assert validate_joint_profiles(
        DEFAULT_JOINT_SPEED_STEPS_S,
        DEFAULT_JOINT_ACC_REGISTERS,
        DEFAULT_JOINT_MAX_DELTA_RAD,
    ) == (
        DEFAULT_JOINT_SPEED_STEPS_S,
        DEFAULT_JOINT_ACC_REGISTERS,
        DEFAULT_JOINT_MAX_DELTA_RAD,
    )

    with pytest.raises(SafetyError, match="exactly six"):
        validate_joint_profiles((1,), DEFAULT_JOINT_ACC_REGISTERS,
                                DEFAULT_JOINT_MAX_DELTA_RAD)
    speeds = list(DEFAULT_JOINT_SPEED_STEPS_S)
    speeds[1] += 1
    with pytest.raises(SafetyError, match="no more than"):
        validate_joint_profiles(speeds, DEFAULT_JOINT_ACC_REGISTERS,
                                DEFAULT_JOINT_MAX_DELTA_RAD)
    accelerations = list(DEFAULT_JOINT_ACC_REGISTERS)
    accelerations[0] = 0
    with pytest.raises(SafetyError, match="greater than zero"):
        validate_joint_profiles(DEFAULT_JOINT_SPEED_STEPS_S, accelerations,
                                DEFAULT_JOINT_MAX_DELTA_RAD)


def test_base90_diagnostic_expands_only_joint1_and_keeps_t101():
    expanded = (math.radians(95.0),) + DEFAULT_JOINT_MAX_DELTA_RAD[1:]
    _, _, checked = validate_joint_profiles(
        DEFAULT_JOINT_SPEED_STEPS_S, DEFAULT_JOINT_ACC_REGISTERS, expanded
    )
    assert checked == pytest.approx(expanded)
    assert build_t101(0, -math.pi / 2.0, 34, 1) == (
        b'{"T":101,"joint":1,"rad":-1.5707963267948966,"spd":34,"acc":1}\n'
    )
    invalid_shoulder = list(DEFAULT_JOINT_MAX_DELTA_RAD)
    invalid_shoulder[1] = math.radians(10.1)
    with pytest.raises(SafetyError, match="no more than"):
        validate_joint_profiles(
            DEFAULT_JOINT_SPEED_STEPS_S,
            DEFAULT_JOINT_ACC_REGISTERS,
            invalid_shoulder,
        )


def test_base90_timeout_math_and_no_t102_payload():
    travel = math.radians(90.0)
    expected = expected_motion_time_s(travel, 34, 1)
    assert expected == pytest.approx(1024.0 / 34.0 + 34.0 / 100.0)
    assert max(40.0, expected + 2.0) == 40.0
    assert b'"T":102' not in build_t101(0, -math.pi / 2.0, 34, 1)


@pytest.mark.parametrize(
    "index,delta_deg,speed,target,joint_id",
    [
        (4, 30.0, 34, math.radians(30.0), 5),
        (3, 15.0, 23, math.radians(15.0), 4),
        (5, 15.0, 17, math.radians(15.0), 6),
        (2, 10.0, 17, math.radians(10.0), 3),
        (1, 10.0, 17, math.radians(10.0), 2),
    ],
)
def test_characterization_caps_and_exact_t101_mapping(
    index, delta_deg, speed, target, joint_id
):
    configured = list(DEFAULT_JOINT_MAX_DELTA_RAD)
    configured[index] = math.radians(delta_deg)
    _, _, checked = validate_joint_profiles(
        DEFAULT_JOINT_SPEED_STEPS_S,
        DEFAULT_JOINT_ACC_REGISTERS,
        configured,
    )
    assert checked[index] == pytest.approx(math.radians(delta_deg))
    raw = build_t101(index, target, speed, 1)
    payload = json.loads(raw)
    assert payload["T"] == 101
    assert payload["joint"] == joint_id
    expected_rad = math.pi - target if index == 5 else target
    assert payload["rad"] == pytest.approx(expected_rad)
    assert payload["spd"] == speed
    assert payload["acc"] == 1
    assert b'"T":102' not in raw


def test_characterization_profile_does_not_expand_base():
    configured = (
        math.radians(5.0), math.radians(10.0), math.radians(10.0),
        math.radians(15.0), math.radians(30.0), math.radians(15.0),
    )
    _, _, checked = validate_joint_profiles(
        DEFAULT_JOINT_SPEED_STEPS_S,
        DEFAULT_JOINT_ACC_REGISTERS,
        configured,
    )
    assert checked[0] == pytest.approx(math.radians(5.0))


def test_directional_calibration_profile_caps():
    configured = tuple(math.radians(x) for x in (5, 9, 9, 11, 5, 2))
    _, _, checked = validate_joint_profiles(
        DEFAULT_JOINT_SPEED_STEPS_S,
        DEFAULT_JOINT_ACC_REGISTERS,
        configured,
    )
    assert checked == pytest.approx(configured)


def test_directional_staging_envelope_plus_landing_residual_boundary():
    staging_envelope = (8, 8, 10)
    command_guards = (9, 9, 11)
    assert all(stage + 1.0 <= guard for stage, guard in zip(staging_envelope, command_guards))


def test_ros_degree_parameter_round_trip_does_not_trip_exact_ceiling():
    round_tripped = tuple(
        math.radians(math.degrees(value)) for value in DEFAULT_JOINT_MAX_DELTA_RAD
    )
    _, _, checked = validate_joint_profiles(
        DEFAULT_JOINT_SPEED_STEPS_S,
        DEFAULT_JOINT_ACC_REGISTERS,
        round_tripped,
    )
    assert checked == pytest.approx(DEFAULT_JOINT_MAX_DELTA_RAD)


@pytest.mark.parametrize("index", range(6))
def test_per_joint_delta_ceiling_is_enforced(index):
    measured = feedback().positions
    over = math.degrees(DEFAULT_JOINT_MAX_DELTA_RAD[index]) + 0.01
    with pytest.raises(SafetyError, match="max delta"):
        validate_single_joint_target(
            joint_target(measured, index, over),
            measured,
            DEFAULT_JOINT_MAX_DELTA_RAD,
        )


def test_elbow_t101_rejects_firmware_clamped_negative_target():
    measured = list(feedback().positions)
    measured[2] = 0.01
    target = list(measured)
    target[2] = -0.01
    with pytest.raises(SafetyError, match="T101 command limit"):
        validate_single_joint_target(target, measured, DEFAULT_JOINT_MAX_DELTA_RAD)


def test_phase2c_quantization_and_2047_2048_asymmetry():
    start = 0.036815539
    target = -0.05045092359971647
    final = -0.012271846
    assert feedback_base_step(start) == 2024
    assert firmware_base_goal_step(target) == 2080
    assert feedback_base_step(final) == 2056
    assert feedback_base_step(final) - feedback_base_step(start) == 32
    assert firmware_base_goal_step(target) - feedback_base_step(final) == 24
    assert firmware_base_goal_step(0.0) == 2047
    assert feedback_base_step(0.0) == 2048
    half = ENCODER_STEP_RAD / 2
    assert firmware_base_goal_step(half) == 2046
    assert firmware_base_goal_step(-half) == 2048


def test_disarmed_gate_and_dry_run_never_writes():
    writes = []
    ctl = SafeController(writes.append, dry_run=True)
    start = feedback()
    target = joint_target(start.positions)
    ctl.accept_feedback(start)
    with pytest.raises(SafetyError, match="disarmed"):
        ctl.command(target, 1.1)
    ctl.arm(1.1)
    raw = ctl.command(target, 1.1)
    assert json.loads(raw)["T"] == 101
    assert writes == []


def test_armed_without_first_command_auto_disarms_at_hard_deadline():
    writes = []
    ctl = SafeController(writes.append, feedback_timeout_s=5.0)
    ctl.accept_feedback(feedback(now=1.0))
    ctl.arm(1.1)
    assert ctl.armed
    assert ctl.arm_command_deadline_at == pytest.approx(1.1 + ARM_COMMAND_DEADLINE_S)
    ctl.tick(1.1 + ARM_COMMAND_DEADLINE_S - 1e-6)
    assert ctl.armed
    ctl.tick(1.1 + ARM_COMMAND_DEADLINE_S)
    assert not ctl.armed
    assert ctl.target is None
    assert ctl.fault is None
    assert ctl.last_result == "ARM_COMMAND_TIMEOUT_DISARMED"
    assert ctl.status_snapshot()["arm_command_deadline_at"] is None
    assert writes == []


def test_duplicate_arm_cannot_extend_first_command_deadline():
    ctl = SafeController(lambda _: None, feedback_timeout_s=5.0)
    ctl.accept_feedback(feedback(now=1.0))
    ctl.arm(1.1)
    first_deadline = ctl.arm_command_deadline_at
    with pytest.raises(SafetyError, match="already armed"):
        ctl.arm(1.5)
    assert ctl.arm_command_deadline_at == first_deadline


def test_first_command_clears_arm_deadline_before_motion_is_active():
    ctl = SafeController(lambda _: None, feedback_timeout_s=5.0)
    start = feedback(now=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(joint_target(start.positions), 1.2)
    assert ctl.armed
    assert ctl.target is not None
    assert ctl.arm_command_deadline_at is None


def test_late_first_command_is_rejected_even_before_timer_tick():
    writes = []
    ctl = SafeController(writes.append, dry_run=False, feedback_timeout_s=5.0)
    start = feedback(now=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    with pytest.raises(SafetyError, match="deadline expired"):
        ctl.command(joint_target(start.positions), 1.1 + ARM_COMMAND_DEADLINE_S)
    assert not ctl.armed
    assert ctl.target is None
    assert ctl.fault is None
    assert ctl.last_result == "ARM_COMMAND_TIMEOUT_DISARMED"
    assert writes == []


def test_real_mode_writes_one_t101_and_busy_gate_rejects():
    writes = []
    ctl = SafeController(writes.append, dry_run=False)
    start = feedback()
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(joint_target(start.positions), 1.1)
    assert len(writes) == 1
    assert b'"T":101' in writes[0]
    assert b'"T":102' not in writes[0]
    with pytest.raises(SafetyError, match="already active"):
        ctl.command(joint_target(start.positions), 1.2)


def test_watchdog_stale_feedback_faults_and_disarms():
    ctl = SafeController(lambda _: None)
    ctl.accept_feedback(feedback())
    ctl.arm(1.1)
    ctl.tick(1.6)
    assert ctl.fault == "stale feedback watchdog"
    assert not ctl.armed


def test_torque_off_feedback_latches_fault():
    ctl = SafeController(lambda _: None)
    ctl.accept_feedback(make_feedback(feedback().positions, torque=False))
    assert ctl.fault == TORQUE_SWITCH_OFF_FAULT
    assert not ctl.armed


def test_torque_switch_fault_self_heals_when_all_flags_recover():
    ctl = SafeController(lambda _: None)
    ctl.latch_fault(TORQUE_SWITCH_OFF_FAULT)
    ctl.accept_feedback(feedback())
    assert ctl.fault is None
    assert ctl.last_result == "FAULT_CLEARED_DISARMED"


def test_implausible_joint_limits_fault_self_heals_on_healthy_feedback():
    ctl = SafeController(lambda _: None)
    ctl.latch_fault(INVALID_FEEDBACK_JOINT_LIMITS_FAULT)
    assert ctl.fault == INVALID_FEEDBACK_JOINT_LIMITS_FAULT
    assert ctl.last_result.startswith("FAULT:")
    ctl.accept_feedback(feedback())
    assert ctl.fault is None
    assert ctl.last_result == "FAULT_CLEARED_DISARMED"


def test_implausible_joint_limits_self_heal_does_not_clear_other_faults():
    ctl = SafeController(lambda _: None)
    ctl.latch_fault("serial reader: boom")
    ctl.accept_feedback(feedback())
    assert ctl.fault == "serial reader: boom"


def test_implausible_joint_limits_error_constant_matches_parser():
    payload = json.loads(
        '{"T":1051,"b":0,"s":0,"e":0,"t":0,"r":0,"g":0,'
        '"tB":0,"tS":0,"tE":0,"tT":0,"tR":0,"tG":0,'
        '"torswitchB":1,"torswitchS":1,"torswitchE":1,'
        '"torswitchT":1,"torswitchR":1,"torswitchG":1,"v":1210}'
    )
    payload["s"] = 10.0  # far outside shoulder limit
    with pytest.raises(SafetyError, match=IMPLAUSIBLE_JOINT_LIMITS_ERROR):
        parse_feedback_line(json.dumps(payload))
    assert INVALID_FEEDBACK_JOINT_LIMITS_FAULT == (
        f"invalid feedback: {IMPLAUSIBLE_JOINT_LIMITS_ERROR}"
    )


@pytest.mark.parametrize("selected", range(6))
def test_original_each_joint_convergence_path_requires_full_stable_window(selected):
    ctl = SafeController(lambda _: None, dry_run=True)
    start = feedback()
    target = joint_target(start.positions, selected, degrees=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    for index in range(20):
        now = 1.2 + index * 0.05
        ctl.accept_feedback(make_feedback(target, now=now))
        assert ctl.target is not None
    ctl.accept_feedback(make_feedback(target, now=2.2))
    assert ctl.target is None
    assert not ctl.armed
    assert ctl.last_result == "CONVERGED_DISARMED"
    assert ctl.telemetry["terminal_reason"] == "CONVERGED_DISARMED"
    assert ctl.telemetry["selected_joint_index"] == selected
    assert len(ctl.telemetry["feedback_samples"]) == 21
    assert ctl.telemetry["convergence_candidate_sample_count"] == 21
    assert ctl.telemetry["convergence_candidate_duration_s"] == pytest.approx(1.0)
    assert ctl.telemetry["terminal_settle_policy"] == {
        "window_s": 1.0,
        "min_samples": TERMINAL_MIN_SETTLE_SAMPLES,
        "max_feedback_gap_s": 0.1,
        "max_joint_span_rad": TERMINAL_MAX_SETTLE_SPAN_RAD,
    }


def test_brief_entry_then_p5_overshoot_does_not_publish_converged():
    writes = []
    start = (
        0.541495218,
        0.050621366,
        1.624485654,
        0.173339829,
        3.008136325,
        0.04141748158979297,
    )
    target = list(start)
    target[1] = 0.190247706
    target = tuple(target)
    first_in_tolerance = list(target)
    first_in_tolerance[1] = 0.187145656
    overshot = list(target)
    overshot[1] = 0.213223330

    ctl = SafeController(
        writes.append,
        dry_run=False,
        joint_max_delta_rad=(
            math.radians(5),
            math.radians(10),
            math.radians(10),
            math.radians(15),
            math.radians(30),
            math.radians(15),
        ),
        command_timeout_s=5.0,
    )
    ctl.accept_feedback(make_feedback(start, now=1.0))
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    for now in (1.2, 1.25, 1.3):
        ctl.accept_feedback(make_feedback(first_in_tolerance, now=now))
    assert ctl.target is not None
    assert ctl.last_result == "COMMAND_ACTIVE"

    ctl.accept_feedback(make_feedback(overshot, now=1.35))
    assert ctl.target is not None
    assert ctl.telemetry["convergence_candidate_sample_count"] == 0
    assert (
        ctl.telemetry["convergence_candidate_last_reset_reason"]
        == "target_left_tolerance"
    )
    timeout_at = ctl.command_deadline_at
    ctl.accept_feedback(make_feedback(overshot, now=timeout_at))
    ctl.tick(timeout_at + 0.2)
    assert ctl.last_result == "FAULT:command convergence timeout"
    assert len(writes) == 1


def test_leaving_tolerance_cancels_candidate_and_requires_new_full_window():
    ctl = SafeController(lambda _: None)
    start = feedback()
    target = joint_target(start.positions, 0, degrees=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(target, 1.1)

    for index in range(10):
        ctl.accept_feedback(make_feedback(target, now=1.2 + index * 0.05))
    outside = list(target)
    outside[0] += math.radians(0.6)
    ctl.accept_feedback(make_feedback(outside, now=1.7))
    assert ctl.telemetry["convergence_candidate_sample_count"] == 0
    assert ctl.telemetry["convergence_candidate_reset_count"] == 1

    for index in range(20):
        ctl.accept_feedback(make_feedback(target, now=1.75 + index * 0.05))
        assert ctl.target is not None
    ctl.accept_feedback(make_feedback(target, now=2.75))
    assert ctl.last_result == "CONVERGED_DISARMED"


def test_feedback_gap_restarts_measured_terminal_settle_window():
    ctl = SafeController(lambda _: None)
    start = feedback()
    target = joint_target(start.positions, 0, degrees=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    for index in range(10):
        ctl.accept_feedback(make_feedback(target, now=1.2 + index * 0.05))

    ctl.accept_feedback(make_feedback(target, now=1.9))
    assert ctl.target is not None
    assert ctl.telemetry["convergence_candidate_sample_count"] == 1
    assert ctl.telemetry["convergence_candidate_last_reset_reason"] == "feedback_gap"

    for index in range(1, 21):
        ctl.accept_feedback(make_feedback(target, now=1.9 + index * 0.05))
    assert ctl.last_result == "CONVERGED_DISARMED"


def test_stable_final_position_outside_tolerance_times_out_fail_closed():
    ctl = SafeController(lambda _: None, command_timeout_s=5.0)
    start = feedback()
    target = joint_target(start.positions, 0, degrees=2.0)
    outside = list(target)
    outside[0] += math.radians(0.75)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    for index in range(96):
        ctl.accept_feedback(make_feedback(outside, now=1.2 + index * 0.05))
    assert ctl.target is not None
    assert ctl.telemetry["convergence_candidate_sample_count"] == 0
    ctl.accept_feedback(make_feedback(outside, now=6.0))
    ctl.tick(6.2)
    assert ctl.last_result == "FAULT:command convergence timeout"


def test_small_non_target_motion_restarts_terminal_settle_window():
    ctl = SafeController(lambda _: None)
    start = feedback()
    target = joint_target(start.positions, 0, degrees=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    for index in range(10):
        ctl.accept_feedback(make_feedback(target, now=1.2 + index * 0.05))

    moved = list(target)
    moved[1] += TERMINAL_MAX_SETTLE_SPAN_RAD + ENCODER_STEP_RAD
    ctl.accept_feedback(make_feedback(moved, now=1.7))
    assert ctl.target is not None
    assert ctl.fault is None
    assert ctl.telemetry["convergence_candidate_sample_count"] == 1
    assert (
        ctl.telemetry["convergence_candidate_last_reset_reason"]
        == "joint_not_stable"
    )

    for index in range(1, 21):
        ctl.accept_feedback(make_feedback(moved, now=1.7 + index * 0.05))
    assert ctl.last_result == "CONVERGED_DISARMED"


def test_terminal_settle_policy_rejects_old_three_sample_configuration():
    with pytest.raises(SafetyError, match="no less than 20"):
        SafeController(lambda _: None, convergence_samples=3)


def test_non_target_drift_above_half_degree_faults():
    ctl = SafeController(lambda _: None)
    start = feedback()
    target = joint_target(start.positions, 2, degrees=1.0)
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    drifted = list(start.positions)
    drifted[1] += math.radians(0.51)
    ctl.accept_feedback(make_feedback(drifted, now=1.2))
    assert ctl.fault == "non-target joint drift exceeded tolerance"
    assert not ctl.armed


def test_recorded_phase2c_undertravel_replay_times_out_without_second_write():
    writes = []
    start = (0.036815539, 0.009203885, 1.622951674, -0.003067962,
             -0.003067962, 0.009203884589793265)
    target = (-0.05045092359971647,) + start[1:]
    final = (-0.012271846, 0.013805827, 1.629087597, -0.003067962,
             -0.003067962, 0.009203884589793265)
    ctl = SafeController(writes.append, dry_run=False)
    ctl.accept_feedback(make_feedback(start, now=1.0))
    ctl.arm(1.1)
    ctl.command(target, 1.1)
    for now in (2.0, 3.0, 6.0):
        ctl.accept_feedback(make_feedback(final, now=now))
    ctl.tick(6.2)
    assert ctl.fault == "command convergence timeout"
    assert len(writes) == 1
    assert ctl.telemetry["latest_joint_error_rad"] == pytest.approx(final[0] - target[0])
    assert ctl.telemetry["max_non_target_drift_rad"] == pytest.approx(
        math.radians(0.3515624913172499)
    )
    assert ctl.telemetry["terminal_reason"] == "FAULT:command convergence timeout"


def test_status_snapshot_reports_per_joint_units_and_timestamped_telemetry():
    ctl = SafeController(lambda _: None)
    start = feedback()
    ctl.accept_feedback(start)
    ctl.arm(1.1)
    ctl.command(joint_target(start.positions), 1.1)
    ctl.accept_feedback(make_feedback(start.positions, now=1.2))
    status = ctl.status_snapshot()
    assert status["joint_acc_registers"] == [1, 1, 1, 1, 1, 1]
    assert status["joint_acceleration_steps_s2"] == [100] * 6
    assert status["joint_speed_steps_s"] == [34, 17, 17, 23, 34, 17]
    assert status["telemetry"]["command_started_at"] == 1.1
    assert status["telemetry"]["feedback_samples"][0]["received_at"] == 1.2


# ------------------------------------------------------------------
# Multi-joint T102 tests
# ------------------------------------------------------------------

# Reusable starting positions for multi-joint tests
_multi_start = feedback().positions


class TestMultiJointTargetValidation:
    def test_two_joints_moving_passes(self):
        target = list(_multi_start)
        target[0] += 0.05  # base (max 5° = 0.087 rad)
        target[2] += 0.03  # elbow (max 2° = 0.035 rad)
        checked, moving = validate_multi_joint_target(
            target, _multi_start, DEFAULT_JOINT_MAX_DELTA_RAD,
        )
        assert checked == tuple(target)
        assert set(moving) == {0, 2}

    def test_zero_joints_moving_raises(self):
        with pytest.raises(SafetyError, match="at least one"):
            validate_multi_joint_target(
                _multi_start, _multi_start, DEFAULT_JOINT_MAX_DELTA_RAD,
            )

    def test_outside_t101_limit_raises(self):
        target = list(_multi_start)
        target[2] = -0.1  # elbow negative — forbidden by T101 limits
        with pytest.raises(SafetyError, match="outside T101"):
            validate_multi_joint_target(
                target, _multi_start, DEFAULT_JOINT_MAX_DELTA_RAD,
            )

    def test_exceeds_max_delta_raises(self):
        target = list(_multi_start)
        # wrist_tilt max delta is 3° (0.05236 rad).  Set target far past that.
        target[3] = _multi_start[3] + 0.2
        with pytest.raises(SafetyError, match="exceeds max delta"):
            validate_multi_joint_target(
                target, _multi_start, DEFAULT_JOINT_MAX_DELTA_RAD,
            )

    def test_all_five_task_joints_moving_passes(self):
        target = list(_multi_start)
        for idx in range(5):
            target[idx] = _multi_start[idx] + 0.01
        checked, moving = validate_multi_joint_target(
            target, _multi_start, DEFAULT_JOINT_MAX_DELTA_RAD,
        )
        assert set(moving) == {0, 1, 2, 3, 4}


class TestBuildT102:
    def test_payload_format(self):
        target = list(_multi_start)
        payload = build_t102(target, 17, 1)
        decoded = json.loads(payload)
        assert decoded["T"] == 102
        assert decoded["base"] == target[0]
        assert decoded["shoulder"] == target[1]
        assert decoded["elbow"] == target[2]
        assert decoded["wrist"] == target[3]
        assert decoded["roll"] == target[4]
        assert decoded["spd"] == 17
        assert decoded["acc"] == 1
        # Gripper convention: firmware = pi - ros
        assert decoded["hand"] == pytest.approx(math.pi - target[5])

    def test_t102_not_t101(self):
        payload = build_t102(list(_multi_start), 17, 1)
        assert b'"T":102' in payload
        assert b'"T":101' not in payload

    def test_vendor_documented_acceleration_register_is_accepted(self):
        decoded = json.loads(build_t102(list(_multi_start), 17, 10))
        assert decoded["T"] == 102
        assert decoded["spd"] == 17
        assert decoded["acc"] == 10

    def test_outside_t101_limit_raises(self):
        target = list(_multi_start)
        target[2] = -0.5  # elbow negative
        with pytest.raises(SafetyError, match="outside T101"):
            build_t102(target, 17, 1)


class TestMultiJointCommand:
    def test_full_cycle_arm_command_converge_disarm(self):
        writes: list[bytes] = []
        ctl = SafeController(writes.append, dry_run=False)
        now = 100.0
        ctl.accept_feedback(feedback(now=now))
        ctl.arm(now)
        assert ctl.armed

        target = list(feedback().positions)
        target[0] += 0.05  # base (max 5°)
        target[3] += 0.04  # wrist_tilt (max 3°)
        payload = ctl.command_multi(target, now)
        decoded = json.loads(payload)
        assert decoded["T"] == 102
        assert ctl.last_result == "COMMAND_ACTIVE"
        assert len(writes) == 1
        assert b'"T":102' in writes[0]

        # Feed converging feedback
        settle_start = now + 0.1
        for i in range(25):
            ctl.accept_feedback(
                make_feedback(tuple(target), now=settle_start + i * 0.05)
            )
        assert ctl.last_result == "CONVERGED_DISARMED"
        assert not ctl.armed

    def test_multi_joint_requires_arm(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback())
        with pytest.raises(SafetyError, match="disarmed"):
            ctl.command_multi(list(feedback().positions), 1.0)

    def test_second_command_rejected_while_active(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.03
        target[1] += 0.02
        ctl.command_multi(target, 1.0)
        with pytest.raises(SafetyError, match="already active"):
            ctl.command_multi(target, 1.0)

    def test_command_multi_replace_succeeds_while_active(self):
        writes: list[bytes] = []
        ctl = SafeController(writes.append, dry_run=False)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        first = list(feedback().positions)
        first[0] += 0.03
        first[1] += 0.02
        ctl.command_multi(first, 1.0)
        assert len(writes) == 1
        # Progress feedback toward first target, then replace with a farther goal
        # still within delta of the *measured* pose.
        mid = list(feedback().positions)
        mid[0] += 0.02
        mid[1] += 0.01
        ctl.accept_feedback(make_feedback(tuple(mid), now=1.05))
        second = list(mid)
        second[0] += 0.03
        second[1] += 0.02
        payload = ctl.command_multi_replace(second, 1.06)
        assert json.loads(payload)["T"] == 102
        assert ctl.armed
        assert ctl.target is not None
        assert ctl.telemetry.get("replaced") is True
        assert ctl.telemetry.get("mode") == "multi_joint_t102_replace"
        assert len(writes) == 2
        assert ctl.last_result == "COMMAND_ACTIVE"

    def test_command_multi_replace_rejects_over_delta(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        first = list(feedback().positions)
        first[0] += 0.03
        first[1] += 0.02
        ctl.command_multi(first, 1.0)
        far = list(feedback().positions)
        far[3] = feedback().positions[3] + 0.2  # exceeds wrist_tilt max delta
        with pytest.raises(SafetyError, match="exceeds max delta"):
            ctl.command_multi_replace(far, 1.05)
        assert ctl.armed
        assert ctl.target is not None

    def test_command_multi_replace_requires_active_multi(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.03
        target[1] += 0.02
        with pytest.raises(SafetyError, match="no active command"):
            ctl.command_multi_replace(target, 1.0)

    def test_disarm_clears_replaceable_target(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.03
        target[1] += 0.02
        ctl.command_multi(target, 1.0)
        ctl.disarm("operator")
        assert ctl.target is None
        assert not ctl._multi_joint_moving
        with pytest.raises(SafetyError, match="disarmed"):
            ctl.command_multi_replace(target, 1.1)

    def test_multi_joint_rejects_no_movement(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        with pytest.raises(SafetyError, match="at least one"):
            ctl.command_multi(list(feedback().positions), 1.0)

    def test_dry_run_never_writes(self):
        writes: list[bytes] = []
        ctl = SafeController(writes.append, dry_run=True)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.05
        ctl.command_multi(target, 1.0)
        assert len(writes) == 0
        assert ctl.last_result == "DRY_RUN_ACTIVE"

    def test_multi_joint_convergence_all_joints_must_be_within_tolerance(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.03
        target[1] += 0.02
        ctl.command_multi(target, 1.0)

        # Feed feedback where one joint is converged but the other is not
        partial = list(target)
        partial[1] = feedback().positions[1]  # shoulder NOT at target
        feed_multi_settle(ctl, partial, 1.1, count=25)
        # Should NOT converge — shoulder is still off
        assert ctl.last_result != "CONVERGED_DISARMED"

    def test_multi_joint_times_out(self):
        ctl = SafeController(lambda _: None, command_timeout_s=0.5)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.05
        ctl.command_multi(target, 1.0)
        # Feed one sample, then tick past timeout
        ctl.accept_feedback(make_feedback(feedback().positions, now=1.1))
        ctl.tick(2.0)
        assert "FAULT" in ctl.last_result

    def test_multi_joint_soft_converges_after_stalled_reissue(self):
        """T102 undertravel within 2× tolerance settles instead of hard timeout."""
        writes: list[bytes] = []
        diagnostic_deltas = tuple(
            math.radians(value) for value in (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
        )
        ctl = SafeController(
            writes.append,
            dry_run=False,
            tolerance_rad=math.radians(1.0),
            command_timeout_s=12.0,
            joint_acc_registers=(10, 10, 10, 10, 10, 10),
            joint_max_delta_rad=diagnostic_deltas,
        )
        started_at = time.monotonic()
        start = feedback(now=started_at)
        ctl.accept_feedback(start)
        ctl.arm(1.0)
        target = list(start.positions)
        target[1] += math.radians(5.0)  # shoulder mover
        ctl.command_multi(target, 1.0)
        # Stall ~1.4° short of target (above hard 1.0°, below soft 2.0°).
        stalled = list(target)
        stalled[1] -= math.radians(1.4)
        expected = float(ctl.telemetry["expected_motion_time_s"])
        for index in range(12):
            now = 1.0 + expected + 1.1 + index * 0.05
            ctl.accept_feedback(make_feedback(stalled, now=now))
            ctl.tick(now)
        assert ctl.telemetry.get("t102_reissue_count") == 1
        reissue_at = float(ctl.telemetry["t102_reissue_at"])
        for index in range(10):
            now = reissue_at + 1.6 + index * 0.05
            ctl.accept_feedback(make_feedback(stalled, now=now))
            ctl.tick(now)
        assert ctl.last_result == "CONVERGED_DISARMED"
        assert ctl.telemetry.get("soft_converge") is True
        assert ctl.fault is None
        assert len(writes) == 2  # original T102 + one reissue

    def test_multi_joint_reissues_after_early_progress_then_terminal_stall(self):
        """Early progress must not hide a recent just-outside-tolerance stall."""
        writes: list[bytes] = []
        diagnostic_deltas = tuple(
            math.radians(value) for value in (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
        )
        ctl = SafeController(
            writes.append,
            dry_run=False,
            tolerance_rad=math.radians(1.0),
            command_timeout_s=12.0,
            joint_acc_registers=(10, 10, 10, 10, 10, 10),
            joint_max_delta_rad=diagnostic_deltas,
        )
        start = feedback(now=1.0)
        ctl.accept_feedback(start)
        ctl.arm(1.0)
        target = list(start.positions)
        target[1] += math.radians(5.0)
        ctl.command_multi(target, 1.0)
        expected = float(ctl.telemetry["expected_motion_time_s"])

        # Retain meaningful early progress in telemetry, followed by eight
        # stable samples 1.05 degrees short of the target.  The old all-history
        # comparison never reissued this real Standby-return shape.
        halfway = list(target)
        halfway[1] -= math.radians(3.0)
        ctl.accept_feedback(make_feedback(halfway, now=1.1))
        stalled = list(target)
        stalled[1] -= math.radians(1.05)
        for index in range(8):
            now = 1.0 + expected + 1.1 + index * 0.05
            ctl.accept_feedback(make_feedback(stalled, now=now))
            ctl.tick(now)

        assert ctl.telemetry.get("t102_reissue_count") == 1
        assert len(writes) == 2

    def test_stalled_reissue_extends_only_the_settle_window_and_deadline_wins(self):
        """A modest post-reissue residual can settle, but never after deadline."""
        def stalled_controller():
            writes: list[bytes] = []
            diagnostic_deltas = tuple(
                math.radians(value)
                for value in (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
            )
            controller = SafeController(
                writes.append,
                dry_run=False,
                tolerance_rad=math.radians(1.0),
                command_timeout_s=0.5,
                joint_acc_registers=(10, 10, 10, 10, 10, 10),
                joint_max_delta_rad=diagnostic_deltas,
            )
            start = feedback(now=1.0)
            controller.accept_feedback(start)
            controller.arm(1.0)
            target = list(start.positions)
            target[1] += math.radians(5.0)
            controller.command_multi(target, 1.0)
            original_deadline = float(controller.command_deadline_at)
            stalled = list(target)
            stalled[1] -= math.radians(1.4)
            expected = float(controller.telemetry["expected_motion_time_s"])
            for index in range(8):
                now = 1.0 + expected + 1.1 + index * 0.05
                controller.accept_feedback(make_feedback(stalled, now=now))
                controller.tick(now)
            reissue_at = float(controller.telemetry["t102_reissue_at"])
            return controller, stalled, original_deadline, reissue_at

        ctl, stalled, original_deadline, reissue_at = stalled_controller()
        revised_deadline = float(ctl.command_deadline_at)
        assert revised_deadline == pytest.approx(
            reissue_at + T102_REISSUE_POST_SETTLE_DEADLINE_S
        )
        settle_at = reissue_at + 1.6
        assert original_deadline < settle_at < revised_deadline
        ctl.accept_feedback(make_feedback(stalled, now=settle_at))
        ctl.tick(settle_at)
        assert ctl.last_result == "CONVERGED_DISARMED"

        expired, stalled, _, _ = stalled_controller()
        expired_at = float(expired.command_deadline_at) + 0.01
        expired.accept_feedback(make_feedback(stalled, now=expired_at))
        expired.tick(expired_at)
        assert expired.fault == "command convergence timeout"

    def test_replace_soft_converges_without_reissue_on_modest_undertravel(self):
        """Hold-teleop release often stops before reissue; soft-settle replace."""
        writes: list[bytes] = []
        diagnostic_deltas = tuple(
            math.radians(value) for value in (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
        )
        ctl = SafeController(
            writes.append,
            dry_run=False,
            tolerance_rad=math.radians(1.0),
            command_timeout_s=12.0,
            joint_acc_registers=(10, 10, 10, 10, 10, 10),
            joint_max_delta_rad=diagnostic_deltas,
        )
        start = feedback(now=1.0)
        ctl.accept_feedback(start)
        ctl.arm(1.0)
        first = list(start.positions)
        first[1] += math.radians(4.0)
        ctl.command_multi(first, 1.0)
        mid = list(start.positions)
        mid[1] += math.radians(2.0)
        ctl.accept_feedback(make_feedback(mid, now=1.05))
        second = list(mid)
        second[1] += math.radians(3.0)
        ctl.command_multi_replace(second, 1.06)
        assert ctl.telemetry.get("replaced") is True
        # Stall ~2.5° short (above 2× hard band, inside 3.5× replace soft band).
        stalled = list(second)
        stalled[1] -= math.radians(2.5)
        expected = float(ctl.telemetry["expected_motion_time_s"])
        for index in range(16):
            now = 1.06 + expected + 1.6 + index * 0.05
            ctl.accept_feedback(make_feedback(stalled, now=now))
            ctl.tick(now)
        assert ctl.last_result == "CONVERGED_DISARMED"
        assert ctl.telemetry.get("soft_converge") is True
        assert ctl.fault is None

    def test_multi_joint_still_faults_when_stalled_far_from_target(self):
        writes: list[bytes] = []
        diagnostic_deltas = tuple(
            math.radians(value) for value in (5.0, 10.0, 10.0, 15.0, 30.0, 15.0)
        )
        ctl = SafeController(
            writes.append,
            dry_run=False,
            tolerance_rad=math.radians(1.0),
            command_timeout_s=2.0,
            joint_acc_registers=(10, 10, 10, 10, 10, 10),
            joint_max_delta_rad=diagnostic_deltas,
        )
        start = feedback(now=1.0)
        ctl.accept_feedback(start)
        ctl.arm(1.0)
        target = list(start.positions)
        target[1] += math.radians(5.0)
        ctl.command_multi(target, 1.0)
        stalled = list(start.positions)  # barely moved
        expected = float(ctl.telemetry["expected_motion_time_s"])
        for index in range(12):
            now = 1.0 + expected + 1.1 + index * 0.05
            ctl.accept_feedback(make_feedback(stalled, now=now))
            ctl.tick(now)
        deadline = float(ctl.command_deadline_at)
        ctl.accept_feedback(make_feedback(stalled, now=deadline))
        ctl.tick(deadline + 0.1)
        assert ctl.fault == "command convergence timeout"

    def test_multi_joint_telemetry(self):
        ctl = SafeController(lambda _: None)
        ctl.accept_feedback(feedback(now=1.0))
        ctl.arm(1.0)
        target = list(feedback().positions)
        target[0] += 0.04
        target[1] += 0.02
        ctl.command_multi(target, 1.0)
        assert ctl.telemetry["mode"] == "multi_joint_t102"
        assert len(ctl.telemetry["moving_joint_indices"]) == 2
        assert "moving_joint_names" in ctl.telemetry

    def test_single_and_multi_joint_can_interleave(self):
        """After a multi-joint command converges, a single-joint command works."""
        writes: list[bytes] = []
        ctl = SafeController(writes.append, dry_run=False)
        now = 100.0

        # Multi-joint
        ctl.accept_feedback(feedback(now=now))
        ctl.arm(now)
        target_multi = list(feedback().positions)
        target_multi[0] += 0.04
        target_multi[1] += 0.02
        ctl.command_multi(target_multi, now)
        feed_multi_settle(ctl, target_multi, now + 0.1, count=25)
        assert ctl.last_result == "CONVERGED_DISARMED"
        assert b'"T":102' in writes[0]

        # Single-joint
        now += 50
        ctl.accept_feedback(make_feedback(tuple(target_multi), now=now))
        ctl.arm(now)
        single = list(target_multi)
        single[0] += 0.03  # base only
        payload = ctl.command(single, now)
        decoded = json.loads(payload)
        assert decoded["T"] == 101
        feed_multi_settle(ctl, single, now + 0.1, count=25)
        assert ctl.last_result == "CONVERGED_DISARMED"


# ------------------------------------------------------------------
# Helpers for multi-joint tests
# ------------------------------------------------------------------


def feed_multi_settle(ctl, positions, start_time, count=25):
    """Feed count samples at target positions, 50 ms apart."""
    for i in range(count):
        ctl.accept_feedback(
            make_feedback(tuple(positions), now=start_time + i * 0.05)
        )


class TestCountMovingJoints:
    def test_zero_when_target_matches_measured(self):
        measured = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert count_moving_joints(measured, measured) == 0

    def test_encoder_step_noise_is_not_moving(self):
        measured = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        target = list(measured)
        target[2] += ENCODER_STEP_RAD * 0.5
        assert count_moving_joints(target, measured) == 0

    def test_single_moving_joint(self):
        measured = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        target = list(measured)
        target[1] += 0.02
        assert count_moving_joints(target, measured) == 1

    def test_multiple_moving_joints(self):
        measured = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        target = list(measured)
        target[0] += 0.02
        target[1] -= 0.02
        target[3] += 0.03
        assert count_moving_joints(target, measured) == 3

    def test_rejects_wrong_length(self):
        with pytest.raises(SafetyError, match="six joints"):
            count_moving_joints((0.0,) * 5, (0.0,) * 6)

    def test_rejects_non_finite(self):
        measured = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        target = list(measured)
        target[0] = float("nan")
        with pytest.raises(SafetyError):
            count_moving_joints(target, measured)

    def test_rejects_negative_tolerance(self):
        measured = (0.0,) * 6
        with pytest.raises(SafetyError, match="tolerance"):
            count_moving_joints(measured, measured, match_tolerance_rad=-1.0)


class TestPrecontactSpeedLease:
    """Pure fail-closed gates for the T102 precontact speed lease."""

    @staticmethod
    def _driver(dry_run=False):
        writes: list[bytes] = []
        controller = SafeController(
            writes.append,
            dry_run=dry_run,
            tolerance_rad=math.radians(0.5),
        )
        controller.accept_feedback(feedback(now=time.monotonic()))
        return controller

    def test_fixed_profile_is_exactly_all_34(self):
        assert PRECONTACT_LEASE_SPEED_STEPS_S == (34, 34, 34, 34, 34, 34)
        assert PRECONTACT_LEASE_PURPOSE == "T102_OUTBOUND_PRECONTACT_ONLY"
        assert DEFAULT_JOINT_SPEED_STEPS_S != PRECONTACT_LEASE_SPEED_STEPS_S

    def test_post_precontact_xy_has_distinct_exact_no_reissue_lease(self):
        assert (
            POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE
            == "T102_POST_PRECONTACT_XY_EXACT_PLAN_NO_REISSUE"
        )
        controller = self._driver()
        lease = PrecontactSpeedLease()
        ok, message = lease.enable(
            controller,
            purpose=POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="post-pre-xy-one-shot",
            foreign_publishers=False,
        )
        assert ok, message
        assert controller.exact_plan_no_reissue_active is True
        assert controller.dynamic_exact_precontact_active is False
        assert controller.exact_plan_lease_purpose == POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE
        assert lease.snapshot()["dynamic_exact_no_reissue"] is True
        assert lease.snapshot()["purpose"] == POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE
        assert lease.release(
            controller,
            purpose=POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="post-pre-xy-one-shot",
        )[0]

    def test_dynamic_pid_commands_are_exact_and_not_arbitrary_tuning(self):
        assert DYNAMIC_PRECONTACT_PID_P == 16
        assert DYNAMIC_PRECONTACT_PID_I == 8
        assert DYNAMIC_PRECONTACT_PID_JOINT_IDS == (1, 2, 3, 4, 5)
        assert build_joint_pid(3, p=16, i=8) == b'{"T":108,"joint":3,"p":16,"i":8}\n'
        assert build_reset_joint_pid() == b'{"T":109}\n'
        with pytest.raises(SafetyError, match="reviewed Dynamic"):
            build_joint_pid(3, p=16, i=16)
        with pytest.raises(SafetyError, match="outside Dynamic"):
            build_joint_pid(6, p=16, i=8)
        with pytest.raises(SafetyError, match="integer"):
            build_joint_pid(True, p=16, i=8)

    def test_enable_sets_all_34_and_release_restores_exact_original_tuple(self):
        controller = self._driver()
        lease = PrecontactSpeedLease()
        original = controller.joint_speed_steps_s
        ok, message = lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="nonce-1",
            foreign_publishers=False,
        )
        assert ok, message
        assert lease.active
        assert controller.joint_speed_steps_s == (34, 34, 34, 34, 34, 34)
        assert lease.snapshot()["active"] is True
        assert lease.snapshot()["original_speed_steps_s"] == list(original)
        ok, message = lease.release(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="nonce-1",
        )
        assert ok, message
        assert not lease.active
        assert controller.joint_speed_steps_s == original
        assert lease.snapshot()["active"] is False

    def test_enable_rejects_armed_target_dry_run_foreign_publisher_bad_purpose(self):
        armed = self._driver()
        armed.arm(1.0)
        assert lease_reject(armed, armed=True)

        active = self._driver()
        active.target = (0.0,) * 6
        assert lease_reject(active, active_target=True)

        dry = self._driver(dry_run=True)
        assert lease_reject(dry, dry_run=True)

        foreign = self._driver()
        assert lease_reject(foreign, foreign_publisher=True)

        bad_purpose = self._driver()
        assert lease_reject(bad_purpose, bad_purpose=True)

    def test_enable_rejects_unhealthy_torque_or_stale_feedback(self):
        torque_off = self._driver()
        torque_off.accept_feedback(
            make_feedback((0.0,) * 6, now=time.monotonic(), torque=False)
        )
        assert lease_reject(torque_off, torque_off=True)

        stale = self._driver()
        stale.feedback = make_feedback(
            stale.feedback.positions, now=time.monotonic() - 10.0
        )
        assert lease_reject(stale, stale=True)

    def test_lease_is_single_owner_and_non_replayable(self):
        controller = self._driver()
        lease = PrecontactSpeedLease()
        ok, _ = lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
            foreign_publishers=False,
        )
        assert ok
        # A second owner (or a replayed enable of the same id) is rejected
        # while the lease is active.
        assert not lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-b",
            foreign_publishers=False,
        )[0]
        assert not lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
            foreign_publishers=False,
        )[0]
        assert lease.active
        assert controller.joint_speed_steps_s == (34,) * 6

        ok, _ = lease.release(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
        )
        assert ok
        # Replaying the already-used lease id must not reopen the lease.
        ok, message = lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
            foreign_publishers=False,
        )
        assert not ok
        assert "replay" in message
        # A fresh id opens the lease again after the old one was released.
        ok, _ = lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-c",
            foreign_publishers=False,
        )
        assert ok

    def test_release_requires_exact_identity_and_disarmed_idle_driver(self):
        controller = self._driver()
        lease = PrecontactSpeedLease()
        lease.enable(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
            foreign_publishers=False,
        )
        ok, message = lease.release(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="attacker-b",
        )
        assert not ok
        assert "identity" in message
        assert lease.active
        assert controller.joint_speed_steps_s == (34,) * 6

        controller.arm(1.0)
        ok, message = lease.release(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
        )
        assert not ok
        assert "disarmed" in message
        controller.disarm("test")
        ok, _ = lease.release(
            controller,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="owner-a",
        )
        assert ok
        assert controller.joint_speed_steps_s == DEFAULT_JOINT_SPEED_STEPS_S

    def test_release_precheck_rejects_wrong_exact_identity_without_state_change(self):
        controller = self._driver()
        lease = PrecontactSpeedLease()
        assert lease.enable(
            controller,
            purpose=POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="xy-owner",
            foreign_publishers=False,
        )[0]
        before = (
            lease.snapshot(), controller.joint_speed_steps_s,
            controller.tolerance_rad, controller.exact_plan_no_reissue_active,
            controller.exact_plan_lease_purpose,
        )
        for purpose, lease_id in (
            (DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE, "xy-owner"),
            (POST_PRECONTACT_XY_EXACT_PLAN_LEASE_PURPOSE, "wrong-id"),
        ):
            ok, message = lease.release_precheck(
                controller, purpose=purpose, lease_id=lease_id
            )
            assert not ok and "identity" in message
            assert (
                lease.snapshot(), controller.joint_speed_steps_s,
                controller.tolerance_rad, controller.exact_plan_no_reissue_active,
                controller.exact_plan_lease_purpose,
            ) == before

    def test_release_without_active_lease_fails_closed(self):
        lease = PrecontactSpeedLease()
        ok, message = lease.release(
            self._driver(),
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="never-enabled",
        )
        assert not ok
        assert "no active" in message

    def test_failed_enable_leaves_speeds_and_lease_unchanged(self):
        controller = self._driver()
        lease = PrecontactSpeedLease()
        original = controller.joint_speed_steps_s
        for kwargs in (
            {"foreign_publishers": True},
            {"purpose": "WRONG_PURPOSE"},
        ):
            assert not lease.enable(
                controller,
                purpose=kwargs.get("purpose", PRECONTACT_LEASE_PURPOSE),
                lease_id="nonce-new",
                foreign_publishers=kwargs.get("foreign_publishers", False),
            )[0]
            assert not lease.active
            assert controller.joint_speed_steps_s == original

    def test_dynamic_exact_plan_stall_never_reissues_or_soft_converges(self):
        writes: list[bytes] = []
        controller = SafeController(
            writes.append,
            dry_run=False,
            tolerance_rad=math.radians(1.0),
            command_timeout_s=12.0,
            joint_acc_registers=(10,) * 6,
            joint_max_delta_rad=tuple(
                math.radians(value) for value in (5, 10, 10, 15, 30, 15)
            ),
        )
        started_at = time.monotonic()
        start = feedback(now=started_at)
        controller.accept_feedback(start)
        lease = PrecontactSpeedLease()
        ok, message = lease.enable(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-exact-stall",
            foreign_publishers=False,
        )
        assert ok, message
        assert controller.dynamic_exact_precontact_active is True
        assert controller.coarse_progress_active is False
        assert controller.tolerance_rad == pytest.approx(
            DYNAMIC_PRECONTACT_TOLERANCE_RAD
        )
        assert lease.snapshot()["dynamic_exact_no_reissue"] is True
        assert lease.snapshot()["dynamic_exact_tolerance_rad"] == pytest.approx(
            DYNAMIC_PRECONTACT_TOLERANCE_RAD
        )
        assert controller.status_snapshot()["dynamic_exact_no_reissue"] is True
        assert controller.status_snapshot()["tolerance_rad"] == pytest.approx(
            DYNAMIC_PRECONTACT_TOLERANCE_RAD
        )

        controller.arm(started_at)
        target = list(start.positions)
        target[1] += math.radians(5.0)
        controller.command_multi(target, started_at)
        stalled = list(target)
        stalled[1] -= math.radians(1.4)
        expected = float(controller.telemetry["expected_motion_time_s"])
        for index in range(12):
            now = started_at + expected + 1.1 + index * 0.05
            controller.accept_feedback(make_feedback(stalled, now=now))
            controller.tick(now)
        assert controller.target is not None
        assert controller.telemetry["t102_reissue_count"] == 0
        assert controller.telemetry.get("soft_converge") is not True
        assert len(writes) == 1

        expired_at = float(controller.command_deadline_at) + 0.01
        controller.accept_feedback(make_feedback(stalled, now=expired_at))
        controller.tick(expired_at)
        assert controller.last_result == "FAULT:command convergence timeout"
        assert controller.armed is False
        assert controller.target is None
        assert len(writes) == 1

        ok, message = lease.release(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-exact-stall",
        )
        assert ok, message
        assert controller.dynamic_exact_precontact_active is False
        assert controller.joint_speed_steps_s == DEFAULT_JOINT_SPEED_STEPS_S
        assert controller.tolerance_rad == pytest.approx(math.radians(1.0))

    def test_dynamic_exact_plan_rejects_replace_stream_without_write(self):
        writes: list[bytes] = []
        controller = SafeController(writes.append, dry_run=False)
        started_at = time.monotonic()
        start = feedback(now=started_at)
        controller.accept_feedback(start)
        lease = PrecontactSpeedLease()
        assert lease.enable(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-exact-replace",
            foreign_publishers=False,
        )[0]
        controller.arm(started_at)
        target = list(start.positions)
        target[0] += math.radians(2.0)
        controller.command_multi(target, started_at)
        with pytest.raises(
            SafetyError, match="dynamic exact precontact forbids target replace"
        ):
            controller.command_multi_replace(target, 1.1)
        assert len(writes) == 1

    def test_dynamic_exact_advance_requires_current_tolerance_and_rejects_reissue(self):
        writes: list[bytes] = []
        controller = SafeController(writes.append, dry_run=False)
        started_at = time.monotonic()
        start = feedback(now=started_at)
        controller.accept_feedback(start)
        lease = PrecontactSpeedLease()
        assert lease.enable(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-exact-advance",
            foreign_publishers=False,
        )[0]
        controller.arm(started_at)
        first = list(start.positions)
        first[0] += math.radians(2.0)
        controller.command_multi(first, started_at)
        accepted_first = tuple(controller.target)
        status = controller.status_snapshot()
        assert status["dynamic_exact_accepted_command_sequence"] == 1
        assert tuple(status["dynamic_exact_accepted_target_positions"]) == accepted_first
        assert status["dynamic_exact_advance_ready"] is False
        second = list(first)
        second[0] += math.radians(2.0)
        with pytest.raises(SafetyError, match="before waypoint tolerance"):
            controller.command_dynamic_exact_advance(second, started_at + 0.1)
        at_first = make_feedback(tuple(first), now=started_at + 0.2)
        controller.accept_feedback(at_first)
        assert controller.status_snapshot()["dynamic_exact_advance_ready"] is True
        with pytest.raises(SafetyError, match="reissue rejected"):
            controller.command_dynamic_exact_advance(first, started_at + 0.21)
        assert controller.dynamic_exact_accepted_command_sequence == 1
        controller.command_dynamic_exact_advance(second, started_at + 0.22)
        assert len(writes) == 2
        assert controller.dynamic_exact_accepted_command_sequence == 2
        assert tuple(controller.dynamic_exact_accepted_target_positions) == tuple(
            controller.target
        )

    def test_dynamic_exact_acceptance_defaults_and_lease_lifetime_reset(self):
        controller = self._driver()
        inactive = controller.status_snapshot()
        assert inactive["dynamic_exact_accepted_command_sequence"] == 0
        assert inactive["dynamic_exact_accepted_target_positions"] == []
        assert inactive["dynamic_exact_advance_ready"] is False

        lease = PrecontactSpeedLease()
        assert lease.enable(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-ack-first",
            foreign_publishers=False,
        )[0]
        now = time.monotonic()
        controller.accept_feedback(make_feedback(controller.feedback.positions, now=now))
        controller.arm(now)
        target = list(controller.feedback.positions)
        target[0] += math.radians(2.0)
        controller.command_multi(target, now)
        assert controller.dynamic_exact_accepted_command_sequence == 1
        controller.disarm("test", now=now + 0.1)
        assert lease.release(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-ack-first",
        )[0]
        assert controller.dynamic_exact_accepted_command_sequence == 0
        assert controller.dynamic_exact_accepted_target_positions == ()

        controller.accept_feedback(
            make_feedback(controller.feedback.positions, now=time.monotonic())
        )
        assert lease.enable(
            controller,
            purpose=DYNAMIC_PRECONTACT_EXACT_PLAN_LEASE_PURPOSE,
            lease_id="dynamic-ack-second",
            foreign_publishers=False,
        )[0]
        assert controller.dynamic_exact_accepted_command_sequence == 0
        assert controller.status_snapshot()["dynamic_exact_advance_ready"] is False


class TestCoarsePrecontactProgress:
    START = (0.0, -1.0, 2.0, 0.0, -1.0, 0.1)
    ENDPOINT = (0.20, -0.60, 1.80, -0.30, -1.0, 0.1)
    TARGET = (
        math.radians(4.0),
        -1.0 + math.radians(8.0),
        2.0 - math.radians(5.0),
        -math.radians(10.0),
        -1.0,
        0.1,
    )

    @classmethod
    def _active(cls):
        writes = []
        ctl = SafeController(
            writes.append,
            dry_run=False,
            joint_speed_steps_s=DEFAULT_JOINT_SPEED_STEPS_S,
            joint_acc_registers=(10,) * 6,
            joint_max_delta_rad=tuple(
                math.radians(value) for value in (5, 10, 10, 15, 30, 15)
            ),
            command_timeout_s=3.0,
            tolerance_rad=math.radians(1.0),
        )
        ctl.coarse_progress_endpoint = cls.ENDPOINT
        now = time.monotonic()
        ctl.accept_feedback(make_feedback(cls.START, now=now))
        lease = PrecontactSpeedLease()
        ok, message = lease.enable(
            ctl,
            purpose=COARSE_PRECONTACT_LEASE_PURPOSE,
            lease_id="coarse-test",
            foreign_publishers=False,
        )
        assert ok, message
        ctl.arm(now + 0.01)
        ctl.command_multi(cls.TARGET, now + 0.02)
        return ctl, lease, writes

    @staticmethod
    def _settle(ctl, positions):
        return feed_stable_window(
            ctl, positions, start_at=ctl.command_started_at + 0.2
        )

    def test_useful_partial_progress_is_explicit_coarse_completion(self):
        ctl, lease, writes = self._active()
        actual = list(self.START)
        actual[0] += 0.040
        actual[1] += 0.050
        self._settle(ctl, actual)
        assert ctl.last_result == "COARSE_PROGRESS_DISARMED"
        assert ctl.telemetry["coarse_max_useful_motion_rad"] >= 0.035
        assert ctl.telemetry["coarse_endpoint_distance_reduction_rad"] > 0.0
        assert ctl.telemetry["t102_reissue_count"] == 0
        assert len(writes) == 1
        assert lease.release(
            ctl,
            purpose=COARSE_PRECONTACT_LEASE_PURPOSE,
            lease_id="coarse-test",
        )[0]
        assert ctl.coarse_progress_active is False

    def test_natural_exact_convergence_remains_exact(self):
        ctl, _, _ = self._active()
        self._settle(ctl, self.TARGET)
        assert ctl.last_result == "CONVERGED_DISARMED"

    @pytest.mark.parametrize("motion_rad", [0.0, 0.020])
    def test_zero_and_sub_two_degree_motion_never_complete(self, motion_rad):
        ctl, _, writes = self._active()
        actual = list(self.START)
        actual[0] += motion_rad
        self._settle(ctl, actual)
        assert ctl.target is not None
        assert ctl.last_result == "COMMAND_ACTIVE"
        ctl.accept_feedback(
            make_feedback(actual, now=ctl.command_deadline_at + 0.005)
        )
        ctl.tick(ctl.command_deadline_at + 0.01)
        assert ctl.last_result == "FAULT:command convergence timeout"
        assert len(writes) == 1

    def test_wrong_direction_above_half_degree_faults_immediately(self):
        ctl, _, _ = self._active()
        actual = list(self.START)
        actual[0] -= COARSE_MAX_WRONG_DIRECTION_RAD + 0.001
        ctl.accept_feedback(
            make_feedback(actual, now=ctl.command_started_at + 0.2)
        )
        assert "wrong direction" in ctl.last_result

    def test_unstable_useful_progress_does_not_complete(self):
        ctl, _, _ = self._active()
        for index in range(24):
            actual = list(self.START)
            actual[0] += 0.040 + (0.004 if index % 2 else 0.0)
            actual[1] += 0.050
            ctl.accept_feedback(
                make_feedback(
                    actual, now=ctl.command_started_at + 0.2 + index * 0.05
                )
            )
        assert ctl.target is not None
        assert ctl.last_result == "COMMAND_ACTIVE"
        assert ctl.telemetry["convergence_candidate_last_reset_reason"] == "joint_not_stable"

    def test_torque_loss_faults_and_foreign_owner_blocks_lease(self):
        ctl, _, _ = self._active()
        ctl.accept_feedback(
            make_feedback(
                self.START, now=ctl.command_started_at + 0.2, torque=False
            )
        )
        assert ctl.fault == TORQUE_SWITCH_OFF_FAULT

        fresh = SafeController(lambda _: None, dry_run=False)
        fresh.coarse_progress_endpoint = self.ENDPOINT
        fresh.accept_feedback(make_feedback(self.START, now=time.monotonic()))
        lease = PrecontactSpeedLease()
        assert not lease.enable(
            fresh,
            purpose=COARSE_PRECONTACT_LEASE_PURPOSE,
            lease_id="foreign-owner",
            foreign_publishers=True,
        )[0]

    def test_coarse_lease_requires_bound_endpoint_and_default_lease_stays_precise(self):
        ctl = SafeController(lambda _: None, dry_run=False)
        ctl.accept_feedback(make_feedback(self.START, now=time.monotonic()))
        lease = PrecontactSpeedLease()
        ok, message = lease.enable(
            ctl,
            purpose=COARSE_PRECONTACT_LEASE_PURPOSE,
            lease_id="missing-endpoint",
            foreign_publishers=False,
        )
        assert not ok
        assert "endpoint" in message
        ok, message = lease.enable(
            ctl,
            purpose=PRECONTACT_LEASE_PURPOSE,
            lease_id="ordinary-purpose",
            foreign_publishers=False,
        )
        assert ok, message
        assert ctl.coarse_progress_active is False
        assert COARSE_MIN_USEFUL_MOTION_RAD == 0.035


def lease_reject(controller, **flags) -> bool:
    """Run enable through every failing gate and return True if it failed."""
    purpose = "WRONG" if flags.get("bad_purpose") else PRECONTACT_LEASE_PURPOSE
    lease = PrecontactSpeedLease()
    ok, message = lease.enable(
        controller,
        purpose=purpose,
        lease_id="nonce-gate-check",
        foreign_publishers=bool(flags.get("foreign_publisher")),
    )
    assert not ok, message
    assert not lease.active
    assert controller.joint_speed_steps_s == DEFAULT_JOINT_SPEED_STEPS_S
    return True
