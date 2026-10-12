from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from press.t102_post_precontact_xy_core import (
    AlignmentBlocked,
    AlignmentIssuanceRegistry,
    ArrivalReceipt,
    FakeAlignmentLifecycle,
    FormalArrivalReceiptLatch,
    MAX_CUMULATIVE_TANGENT_CORRECTION_M,
    MAX_PER_MOVE_TANGENT_CORRECTION_M,
    RayPlaneMeasurement,
    build_alignment_plan,
    canonical_hash,
    canonical_json,
    verify_alignment_plan,
)


class State:
    def __init__(self, tcp, axis=(0.0, 0.0, -1.0)):
        self.tcp = tuple(tcp)
        self.axis = tuple(axis)


class FakeModel:
    hashes = {
        "kinematic_model_hash": "d" * 64,
        "collision_proxy_hash": "e" * 64,
        "tool_model_hash": "f" * 64,
        "calibration_hash": "c" * 64,
    }

    def fk(self, q):
        return State((q[0], q[1], 0.2))

    def solve(self, seed, target, desired_axis):
        q = list(seed)
        q[0], q[1] = target[0], target[1]
        return type("Audit", (), {"q": tuple(q)})()


def receipt(**changes) -> ArrivalReceipt:
    unsigned = dict(
        cycle_id="future-cycle-1",
        terminal_code="DYNAMIC_PRECONTACT_STOPPED_DISARMED",
        session_id="session-1",
        cart_pose_epoch="cart-1",
        target_snapshot_sha256="a" * 64,
        stage_a_plan_hash="b" * 64,
        p_button_base_m=(0.0, 0.0, 0.2),
        p_pre_base_m=(0.0, 0.0, 0.13),
        panel_normal_base=(0.0, 0.0, -1.0),
        wall_surface_point_base_m=(0.0, 0.0, 0.2),
        kinematic_model_hash=FakeModel.hashes["kinematic_model_hash"],
        collision_proxy_hash=FakeModel.hashes["collision_proxy_hash"],
        tool_model_hash=FakeModel.hashes["tool_model_hash"],
        calibration_hash=FakeModel.hashes["calibration_hash"],
        tool_identity="touch-v1",
        executor_instance_id="executor-1",
        wrist_camera_calibration_identity="wrist-motion-cal-1",
        wrist_camera_calibration_hash="9" * 64,
        terminal_q_rad=(0.0,) * 6,
        feedback_stamp_ns=100,
        feedback_fresh=True,
        stationary_sample_count=3,
        stationary_span_rad=0.0001,
        cleanup_proven=True,
        joint_target_publishers=0,
        lease_active=False,
        armed=False,
        teardown_complete=True,
        terminal_resources={
            "publisher": 0, "lease_client": 0, "arm_client": 0,
            "service_call": 0, "action_call": 0,
        },
    )
    unsigned.update(changes)
    digest = canonical_hash(unsigned)
    return ArrivalReceipt(receipt_id=digest, receipt_sha256=digest, **unsigned)


def measurement(**changes) -> RayPlaneMeasurement:
    unsigned = dict(
        pre_lock_id="pre-lock-1",
        target_pixel=(370.0, 240.0),
        touch_pixel=(320.0, 240.0),
        frame_stamp_ns=101,
        fx=700.0,
        fy=700.0,
        cx=320.0,
        cy=240.0,
        distortion_model="plumb_bob",
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
        camera_origin_base_m=(0.0, 0.0, 0.0),
        rotation_base_from_camera=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        camera_calibration_identity="wrist-motion-cal-1",
        camera_calibration_hash="9" * 64,
        camera_calibration_execution_ready=True,
    )
    unsigned.update(changes)
    digest = canonical_hash(unsigned)
    return RayPlaneMeasurement(
        measurement_id=digest, measurement_sha256=digest, **unsigned
    )


def build(**changes):
    args = dict(
        alignment_id="align-1",
        receipt=receipt(),
        measurement=measurement(),
        model=FakeModel(),
        per_move_joint_caps_rad=(0.1,) * 6,
        joint_hard_limits_rad=((-1.0, 1.0),) * 6,
        path_validator=lambda _a, _b: None,
    )
    args.update(changes)
    return build_alignment_plan(**args)


def test_ray_plane_geometry_uses_panel_intersections_not_70mm_depth():
    frozen = build()
    assert "depth_m" not in frozen.plan
    assert "raw_requested_camera_m" not in frozen.plan
    assert frozen.plan["requested_tangent_norm_m"] == pytest.approx(0.2 * 50 / 700)
    assert frozen.plan["predicted_normal_drift_m"] == pytest.approx(0.0, abs=1e-12)


def test_ray_plane_requires_so3_and_execution_ready_bound_calibration():
    with pytest.raises(AlignmentBlocked, match="WRIST_CALIBRATION_NOT_EXECUTION_READY"):
        build(measurement=measurement(camera_calibration_execution_ready=False))
    with pytest.raises(AlignmentBlocked, match="SO3"):
        build(measurement=measurement(rotation_base_from_camera=((1, 0, 0), (0, 2, 0), (0, 0, 1))))
    with pytest.raises(AlignmentBlocked, match="WRIST_CALIBRATION_NOT_EXECUTION_READY"):
        build(measurement=measurement(camera_calibration_identity="diagnostic-only"))


def test_per_move_and_cumulative_caps_are_distinct_and_frozen():
    frozen = build(measurement=measurement(target_pixel=(600.0, 240.0)))
    assert frozen.plan["requested_tangent_norm_m"] > MAX_PER_MOVE_TANGENT_CORRECTION_M
    assert frozen.plan["capped_tangent_norm_m"] == pytest.approx(0.015)
    assert frozen.plan["maximum_per_move_tangent_correction_m"] == 0.015
    assert frozen.plan["maximum_cumulative_tangent_correction_m"] == 0.015
    with pytest.raises(AlignmentBlocked, match="CUMULATIVE"):
        build(cumulative_tangent_before_m=MAX_CUMULATIVE_TANGENT_CORRECTION_M)


def test_plan_holds_joints_4_and_5_checks_path_and_model_identity():
    calls = []
    frozen = build(path_validator=lambda a, b: calls.append((tuple(a), tuple(b))))
    assert frozen.plan["candidate_target_rad"][4:] == [0.0, 0.0]
    assert len(calls) == 1

    class WrongModel(FakeModel):
        hashes = {**FakeModel.hashes, "kinematic_model_hash": "0" * 64}

    with pytest.raises(AlignmentBlocked, match="MODEL_IDENTITY_MISMATCH"):
        build(model=WrongModel())


def test_executor_revalidates_canonical_plan_identity_start_and_collision():
    frozen = build()
    text = canonical_json(frozen.plan)
    verified = verify_alignment_plan(
        text, frozen.plan_sha256, expected_receipt=receipt(),
        expected_alignment_id="align-1", current_q_rad=(0.0,) * 6,
        path_validator=lambda _a, _b: None,
        expected_model_hashes=FakeModel.hashes,
    )
    assert verified["no_contact"] is True
    with pytest.raises(AlignmentBlocked, match="PLAN_INVALIDATED"):
        verify_alignment_plan(text, "0" * 64, expected_receipt=receipt(), expected_alignment_id="align-1", current_q_rad=(0.0,) * 6, path_validator=lambda _a, _b: None)
    with pytest.raises(AlignmentBlocked, match="START_POSTURE_DRIFT"):
        verify_alignment_plan(text, frozen.plan_sha256, expected_receipt=receipt(), expected_alignment_id="align-1", current_q_rad=(0.1, 0, 0, 0, 0, 0), path_validator=lambda _a, _b: None)
    with pytest.raises(RuntimeError, match="collision"):
        verify_alignment_plan(text, frozen.plan_sha256, expected_receipt=receipt(), expected_alignment_id="align-1", current_q_rad=(0.0,) * 6, path_validator=lambda _a, _b: (_ for _ in ()).throw(RuntimeError("collision")))


@pytest.mark.parametrize(
    "changes",
    [
        {"cleanup_proven": False}, {"joint_target_publishers": 1},
        {"lease_active": True}, {"armed": True}, {"teardown_complete": False},
        {"terminal_resources": {"publisher": 1}},
    ],
)
def test_arrival_receipt_binds_cleanup_resources(changes):
    with pytest.raises(AlignmentBlocked, match="ARRIVAL_RECEIPT_UNSAFE"):
        receipt(**changes).validate()


@pytest.mark.parametrize(
    "changes",
    [
        {"terminal_code": "MOVE_ENDPOINT_ERROR"},
        {"feedback_fresh": False},
        {"stationary_sample_count": 2},
        {"stationary_span_rad": float("inf")},
    ],
)
def test_arrival_receipt_binds_terminal_and_stationary_evidence(changes):
    with pytest.raises(AlignmentBlocked):
        receipt(**changes).validate()


def test_receipt_digest_detects_geometry_cycle_and_model_drift():
    valid = receipt()
    valid.validate()
    for field, value in (
        ("cycle_id", "other-cycle"),
        ("p_button_base_m", (0.01, 0, 0.2)),
        ("panel_normal_base", (1, 0, 0)),
        ("kinematic_model_hash", "1" * 64),
    ):
        with pytest.raises(AlignmentBlocked, match="canonical digest"):
            replace(valid, **{field: value}).validate()


def test_future_cycle_latch_and_issuance_remain_nonreplayable_after_clear():
    latch = FormalArrivalReceiptLatch("boot-cycle")
    with pytest.raises(AlignmentBlocked, match="NEW_GO_PRE_REQUIRED"):
        latch.record(receipt(cycle_id="boot-cycle"))
    accepted = latch.record(receipt())
    registry = AlignmentIssuanceRegistry()
    plan = "1" * 64
    assert registry.issue(receipt=accepted, alignment_id="align-1", plan_sha256=plan) == (accepted.receipt_id, "align-1", plan)
    registry.verify(receipt_id=accepted.receipt_id, alignment_id="align-1", plan_sha256=plan)
    registry.clear_active(); latch.clear()
    with pytest.raises(AlignmentBlocked, match="ARRIVAL_RECEIPT_REPLAY"):
        registry.issue(receipt=accepted, alignment_id="align-2", plan_sha256="2" * 64)


def test_external_progress_code_cannot_bypass_measured_failure():
    machine = FakeAlignmentLifecycle((0.01, 0, 0, 0, 0, 0))
    machine.begin(); machine.terminal((0, 0, 0, 0, 0, 0), progress_code="OK")
    assert machine.code == "NO_PROGRESS" and machine.teardown_complete


@pytest.mark.parametrize(
    ("target", "measured", "code"),
    [
        ((0.01, 0.01, 0, 0, 0, 0), (0.01, 0, 0, 0, 0, 0), "NO_PROGRESS"),
        ((0.01, 0, 0, 0, 0, 0), (0.01, 0.002, 0, 0, 0, 0), "LOCKED_JOINT_DRIFT"),
        ((0.01, 0, 0, 0, 0, 0), (-0.001, 0, 0, 0, 0, 0), "WRONG_DIRECTION"),
        ((0.02, 0, 0, 0, 0, 0), (0.01, 0, 0, 0, 0, 0), "UNDERTRAVEL"),
    ],
)
def test_per_joint_progress_failures(target, measured, code):
    machine = FakeAlignmentLifecycle(target)
    machine.begin(); machine.terminal(measured)
    assert machine.code == code and machine.teardown_complete


def test_sub_half_encoder_step_plan_has_no_moving_joint():
    machine = FakeAlignmentLifecycle((0.0005, 0, 0, 0, 0, 0))
    machine.begin()
    with pytest.raises(AlignmentBlocked, match="no moving joint"):
        machine.terminal((0.0005, 0, 0, 0, 0, 0))


def test_success_and_external_failure_both_teardown_without_retry():
    success = FakeAlignmentLifecycle((0.01, 0, 0, 0, 0, 0))
    success.begin(); success.terminal((0.01, 0, 0, 0, 0, 0))
    assert success.state == "NEED_PRESS_LOCK" and success.publish_count == 1
    forced = FakeAlignmentLifecycle((0.01, 0, 0, 0, 0, 0))
    forced.begin(); forced.terminal((0.01, 0, 0, 0, 0, 0), progress_code="UNDERTRAVEL")
    assert forced.state == "FAILED" and forced.teardown_complete


def test_plan_and_policy_contain_only_no_contact_default_off_semantics():
    frozen = build()
    assert frozen.plan["no_contact"] is True
    assert frozen.plan["automatic_retry"] is False
    assert frozen.plan["automatic_compensation"] is False
    assert frozen.plan["press"] is False and frozen.plan["retract"] is False
    policy = json.loads(
        (Path(__file__).resolve().parents[2] / "config/press/releases/t102_post_precontact_xy_policy_20260821_v1.json").read_text()
    )
    for flag in ("runtime_enabled", "motion_authorized", "contact_authorized", "press_authorized", "normal_motion_enabled", "press", "retract", "automatic_retry", "automatic_compensation"):
        assert policy[flag] is False
    assert policy["no_contact"] is True


def test_prepare_service_and_move_action_have_split_authority_schema():
    root = Path(__file__).resolve().parents[2]
    action = (root / "ros2_perception/ros2_ws/src/elevator_interfaces/action/PrecontactXY.action").read_text()
    request = action.split("---", 1)[0]
    fields = [line.strip() for line in request.splitlines() if line.strip() and not line.startswith("#")]
    assert fields == [
        "string operation", "string authorization_id", "string alignment_id",
        "string arrival_receipt_id", "string alignment_plan_sha256",
        "string confirmation_token",
    ]
    field_text = "\n".join(fields)
    for forbidden in ("pixel", "candidate_target", "measurement_json", "plan_json"):
        assert forbidden not in field_text
    service = (root / "ros2_perception/ros2_ws/src/elevator_interfaces/srv/PreparePrecontactXY.srv").read_text()
    assert "canonical_arrival_receipt_json" in service
    assert "canonical_ray_plane_measurement_json" in service
    assert "canonical_alignment_plan_json" in service
    assert '"srv/PreparePrecontactXY.srv"' in (
        root / "ros2_perception/ros2_ws/src/elevator_interfaces/CMakeLists.txt"
    ).read_text()
