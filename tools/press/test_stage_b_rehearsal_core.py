from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from press.stage_b_rehearsal_core import (
    FORWARD_DISTANCE_M,
    APPROACH_GEOMETRY_TOLERANCE_M,
    DEFAULT_ENCODER_STEP_RAD,
    LEASE_PURPOSE,
    PRESS_OVERTRAVEL_M,
    PreTerminalReceipt,
    StageBBlocked,
    StageBExactExecution,
    StageBExecutionConfig,
    StageBSafetySample,
    canonical_hash,
    canonical_json,
    prepare_stage_b_plan,
    verify_stage_b_plan,
    validate_command_acceptance,
    validate_model_path,
    _audit_runtime_normal_path,
    _contact_depth_from_endpoint,
    _quantize_runtime_waypoint,
    _select_runtime_waypoints,
)
from press.t102_dynamic_precontact_core import FrozenModel


class Model:
    def __init__(self) -> None:
        self.hashes = {
            "kinematic_model_hash": "a" * 64,
            "collision_proxy_hash": "b" * 64,
            "tool_model_hash": "c" * 64,
            "calibration_hash": "d" * 64,
        }
        self.policy = {
            "cartesian_sample_count": 3,
            "segment_sample_count": 3,
            "per_command_caps_rad": [0.2] * 6,
            "joint_hard_limits_rad": [[-1.0, 1.0]] * 6,
            "maximum_joint_jump_rad": 0.45,
            "position_residual_limit_m": 0.0025,
            "axis_residual_limit_deg": 8.0,
            "q0_drift_limit_rad": 0.003,
            "encoder_step_rad": DEFAULT_ENCODER_STEP_RAD,
            "move_waypoint_tolerance_rad": 0.008726646259971648,
            "maximum_q0_age_s": 1.0,
            "wall_margin_m": 0.001,
            "capsules": [{"name": "tool", "joint_a": 0, "joint_b": 0, "radius_m": 0.0}],
            "self_pairs": [],
        }

    def fk(self, q):
        x = float(q[0])
        return SimpleNamespace(
            tcp=(x, 0.0, 0.0),
            axis=(-1.0, 0.0, 0.0),
            points=((x, 0.0, 0.0),),
        )

    def solve(self, seed, target_tcp, desired_axis):
        q = list(seed)
        q[0] = float(target_tcp[0])
        return SimpleNamespace(
            q=tuple(q),
            seed_hash=canonical_hash(list(seed)),
            position_residual_m=0.0,
            axis_residual_deg=0.0,
            condition=10.0,
            manipulability=0.1,
        )

    def _validate_limits(self, q):
        if any(value < -0.99 or value > 0.99 for value in q):
            raise ValueError("outside limits")


class SmallCapModel(Model):
    """Model with a small unchanged per-command cap for endpoint regression."""

    def __init__(self) -> None:
        super().__init__()
        self.policy["per_command_caps_rad"] = [0.061] * 6


class TangentModel(Model):
    def fk(self, q):
        x = float(q[0])
        y = float(q[1])
        return SimpleNamespace(
            tcp=(x, y, 0.0),
            axis=(-1.0, 0.0, 0.0),
            points=((x, y, 0.0),),
        )

    def solve(self, seed, target_tcp, desired_axis):
        q = list(seed)
        q[0] = float(target_tcp[0])
        q[1] = float(target_tcp[1])
        return SimpleNamespace(
            q=tuple(q),
            seed_hash=canonical_hash(list(seed)),
            position_residual_m=0.0,
            axis_residual_deg=0.0,
            condition=10.0,
            manipulability=0.1,
        )


class ToolOffsetModel(Model):
    """Tool centerline is ahead of TCP along the panel normal."""

    def __init__(self) -> None:
        super().__init__()
        self.policy["capsules"] = [{"name": "tool", "joint_a": 0, "joint_b": 1, "radius_m": 0.0}]

    def fk(self, q):
        x = float(q[0])
        return SimpleNamespace(
            tcp=(x, 0.0, 0.0),
            axis=(-1.0, 0.0, 0.0),
            points=((x - 0.002, 0.0, 0.0), (x - 0.001, 0.0, 0.0)),
        )


class ExcessiveToolOffsetModel(ToolOffsetModel):
    """The tool would exceed fixed overtravel plus IK residual authority."""

    def fk(self, q):
        x = float(q[0])
        return SimpleNamespace(
            tcp=(x, 0.0, 0.0),
            axis=(-1.0, 0.0, 0.0),
            points=((x - 0.003, 0.0, 0.0), (x - 0.002, 0.0, 0.0)),
        )


class ToolAndGuardModel(ToolOffsetModel):
    def __init__(self) -> None:
        super().__init__()
        self.policy["capsules"] = [
            {"name": "tool", "joint_a": 0, "joint_b": 1, "radius_m": 0.0},
            {"name": "guard", "joint_a": 2, "joint_b": 2, "radius_m": 0.0},
        ]

    def fk(self, q):
        x = float(q[0])
        return SimpleNamespace(
            tcp=(x, 0.0, 0.0),
            axis=(-1.0, 0.0, 0.0),
            points=(
                (x - 0.02, 0.0, 0.0),
                (x - 0.01, 0.0, 0.0),
                (x - 0.05, 0.0, 0.0),
            ),
        )


def make_receipt() -> PreTerminalReceipt:
    raw = {
        "schema": "current_panel_press_receipt_v1",
        "panel_geometry_sha256": "e" * 64,
        "terminal_q_rad": [0.10, 0.0, 0.0, 0.0, 0.2, 0.3],
        "p_button_base_m": [0.03, 0.0, 0.0],
        "normal_out_base": [1.0, 0.0, 0.0],
        "wall_surface_point_base_m": [0.03, 0.0, 0.0],
        "kinematic_model_hash": "a" * 64,
        "collision_proxy_hash": "b" * 64,
        "tool_model_hash": "c" * 64,
        "calibration_hash": "d" * 64,
        "feedback_stamp_ns": 1_000_000_000,
        "feedback_fresh": True,
        "stationary_sample_count": 3,
        "stationary_span_rad": 0.0,
        "cleanup_proven": True,
        "joint_target_publishers": 0,
        "lease_active": False,
        "armed": False,
        "teardown_complete": True,
        "terminal_resources": {"publisher": 0, "lease": 0, "armed": 0},
        "wrist_acquisition_id": "press-click-1",
        "wrist_lock_identity": ["press-click-1", "wrist-cal", "640x480"],
        "wrist_lock_stage": "PRE-LOCK",
        "wrist_lock_center_px": [320.0, 240.0],
        "wrist_lock_observation_count": 3,
    }
    digest = canonical_hash(raw)
    raw["receipt_id"] = digest
    raw["receipt_sha256"] = digest
    return PreTerminalReceipt.from_mapping(raw)


def changed_receipt(**changes) -> PreTerminalReceipt:
    raw = dict(make_receipt().unsigned_mapping())
    raw.update(changes)
    digest = canonical_hash(raw)
    raw.update(receipt_id=digest, receipt_sha256=digest)
    return PreTerminalReceipt.from_mapping(raw)


def make_wrist_corrected_receipt(
    delta=(0.0, 0.005, 0.0),
) -> PreTerminalReceipt:
    raw = dict(make_receipt().unsigned_mapping())
    delta = [float(value) for value in delta]
    raw.update(
        schema="current_panel_press_receipt_v2",
        gemini_p_button_base_m=[0.03, 0.0, 0.0],
        gemini_wall_surface_point_base_m=[0.03, 0.0, 0.0],
        wrist_tangent_delta_base_m=delta,
        wrist_tangent_calibration_sha256="f" * 64,
        wrist_image_error_px=[-10.0, 0.0],
        wrist_correction_applied=True,
        p_button_base_m=[0.03 + delta[0], delta[1], delta[2]],
        wall_surface_point_base_m=[0.03 + delta[0], delta[1], delta[2]],
    )
    digest = canonical_hash(raw)
    raw.update(receipt_id=digest, receipt_sha256=digest)
    return PreTerminalReceipt.from_mapping(raw)


def test_wrist_corrected_receipt_accepts_thirty_five_mm_and_rejects_more():
    demonstrated = make_wrist_corrected_receipt((0.0, 0.0309382, 0.0))
    demonstrated_plan = prepare_stage_b_plan(
        demonstrated,
        TangentModel(),
        now_ns=1_100_000_000,
        contact_authorized=True,
        press_authorized=True,
        force_control=False,
        zero_dwell=True,
        require_wrist_tangent_correction=True,
    )
    assert demonstrated_plan.plan["wrist_tangent_delta_base_m"] == pytest.approx(
        [0.0, 0.0309382, 0.0]
    )
    accepted = make_wrist_corrected_receipt((0.0, 0.035, 0.0))
    assert accepted.wrist_tangent_delta_base_m == pytest.approx(
        (0.0, 0.035, 0.0)
    )
    with pytest.raises(StageBBlocked, match="WRIST_CORRECTION_TOO_LARGE"):
        make_wrist_corrected_receipt((0.0, 0.035001, 0.0))


def test_wrist_corrected_receipt_keeps_schema_hash_provenance_and_normal_gates():
    unsigned = dict(make_wrist_corrected_receipt().unsigned_mapping())

    def parse(changes, *, valid_hash=True):
        value = dict(unsigned)
        value.update(changes)
        digest = canonical_hash(value) if valid_hash else "0" * 64
        value.update(receipt_id=digest, receipt_sha256=digest)
        return PreTerminalReceipt.from_mapping(value)

    with pytest.raises(StageBBlocked, match="ARRIVAL_RECEIPT_INVALID: schema"):
        parse({"schema": "current_panel_press_receipt_v3"})
    with pytest.raises(StageBBlocked, match="ARRIVAL_RECEIPT_INVALID: canonical digest"):
        parse({}, valid_hash=False)
    with pytest.raises(
        StageBBlocked,
        match="ARRIVAL_RECEIPT_INVALID: wrist_tangent_calibration_sha256",
    ):
        parse({"wrist_tangent_calibration_sha256": ""})
    with pytest.raises(StageBBlocked, match="WRIST_CORRECTION_NORMAL_COMPONENT"):
        parse({
            "wrist_tangent_delta_base_m": [0.001, 0.005, 0.0],
            "p_button_base_m": [0.031, 0.005, 0.0],
            "wall_surface_point_base_m": [0.031, 0.005, 0.0],
        })


def prepare(receipt=None, model=None):
    return prepare_stage_b_plan(
        receipt or make_receipt(), model or Model(),
        now_ns=1_100_000_000, contact_authorized=True,
        press_authorized=True, force_control=False, zero_dwell=True,
    )


def test_wrist_tangent_delta_changes_actual_stage_b_endpoint_and_preserves_normal_depth():
    receipt = make_wrist_corrected_receipt()
    plan = prepare_stage_b_plan(
        receipt,
        TangentModel(),
        now_ns=1_100_000_000,
        contact_authorized=True,
        press_authorized=True,
        force_control=False,
        zero_dwell=True,
        require_wrist_tangent_correction=True,
    )
    payload = plan.plan
    assert payload["wrist_correction_applied"] is True
    assert payload["wrist_tangent_delta_base_m"] == [0.0, 0.005, 0.0]
    assert payload["forward_target_tcp_base_m"][1] == pytest.approx(0.005)
    assert payload["forward_sample_tcp_base_m"][-1][1] == pytest.approx(0.005, abs=0.0025)
    assert sum(
        a * b
        for a, b in zip(
            payload["wrist_tangent_delta_base_m"], payload["normal_out_base"]
        )
    ) == pytest.approx(0.0, abs=1e-12)
    assert payload["current_panel_gap_m"] == pytest.approx(0.07)
    verified = verify_stage_b_plan(
        canonical_json(payload),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=TangentModel(),
        current_q_rad=receipt.terminal_q_rad,
        now_ns=1_100_000_000,
        now_monotonic=0.1,
        expected_press_authorized=True,
    )
    assert verified["forward_target_tcp_base_m"][1] == pytest.approx(0.005)


def test_final_stage_b_path_rejects_legacy_receipt_without_wrist_correction():
    with pytest.raises(StageBBlocked, match="WRIST_CORRECTION_REQUIRED"):
        prepare_stage_b_plan(
            make_receipt(),
            Model(),
            now_ns=1_100_000_000,
            contact_authorized=True,
            press_authorized=True,
            force_control=False,
            zero_dwell=True,
            require_wrist_tangent_correction=True,
        )


def make_field_receipt(model) -> PreTerminalReceipt:
    raw = {
        "schema": "current_panel_press_receipt_v1",
        "panel_geometry_sha256": "4ae0d58303db86126490de6a7d6bb48d4d6df0c1d2e76a456f8c1218e0a0b5d4",
        "terminal_q_rad": [
            0.340543735, -0.60438843, 1.823903157, 0.335941793,
            -1.510971076, 0.001533980589793149,
        ],
        "p_button_base_m": [
            0.35967248671249197, 0.11086854196087337, 0.2844202327023364,
        ],
        "normal_out_base": [
            -0.9969275160393365, -0.06968065310800552, -0.03577896513428543,
        ],
        "wall_surface_point_base_m": [
            0.35967248671249197, 0.11086854196087337, 0.2844202327023364,
        ],
        "kinematic_model_hash": model.hashes["kinematic_model_hash"],
        "collision_proxy_hash": model.hashes["collision_proxy_hash"],
        "tool_model_hash": model.hashes["tool_model_hash"],
        "calibration_hash": model.hashes["calibration_hash"],
        "feedback_stamp_ns": 1_000_000_000,
        "feedback_fresh": True,
        "stationary_sample_count": 3,
        "stationary_span_rad": 0.0,
        "cleanup_proven": True,
        "joint_target_publishers": 0,
        "lease_active": False,
        "armed": False,
        "teardown_complete": True,
        "terminal_resources": {"publisher": 0, "lease": 0, "armed": 0},
        "wrist_acquisition_id": "field-press-20260824",
        "wrist_lock_identity": ["field-press-20260824", "wrist-cal", "640x480"],
        "wrist_lock_stage": "PRE-LOCK",
        "wrist_lock_center_px": [320.0, 240.0],
        "wrist_lock_observation_count": 3,
    }
    digest = canonical_hash(raw)
    raw.update(receipt_id=digest, receipt_sha256=digest)
    return PreTerminalReceipt.from_mapping(raw)


def make_field_receipt_at_gap(model, gap_m: float) -> PreTerminalReceipt:
    raw = dict(make_field_receipt(model).unsigned_mapping())
    q0 = tuple(raw["terminal_q_rad"])
    normal = tuple(raw["normal_out_base"])
    start_tcp = tuple(model.fk(q0).tcp)
    p_button = [
        start_tcp[index] - normal[index] * gap_m
        for index in range(3)
    ]
    raw.update(
        p_button_base_m=p_button,
        wall_surface_point_base_m=p_button,
        wrist_acquisition_id="field-press-observed-gap",
        wrist_lock_identity=["field-press-observed-gap", "wrist-cal", "640x480"],
    )
    digest = canonical_hash(raw)
    raw.update(receipt_id=digest, receipt_sha256=digest)
    return PreTerminalReceipt.from_mapping(raw)


def test_field_frozen_model_selects_quantized_q4_q8_q12_q16_and_validates_four_out_four_back():
    root = Path(__file__).resolve().parents[2]
    model = FrozenModel.load_once(
        root,
        root / "config/press/releases/t102_dynamic_precontact_policy_release_20260818.json",
    )
    receipt = make_field_receipt(model)
    plan = prepare(receipt, model)
    payload = plan.plan
    q0 = tuple(receipt.terminal_q_rad)
    assert len(payload["offline_forward_validation_waypoints_rad"]) == 16
    assert payload["runtime_source_waypoint_indices"] == [4, 8, 12, 16]
    expected_quantized = tuple(
        _quantize_runtime_waypoint(
            payload["offline_forward_validation_waypoints_rad"][index - 1],
            q0, DEFAULT_ENCODER_STEP_RAD,
        )
        for index in (4, 8, 12, 16)
    )
    assert plan.forward_waypoints_rad == expected_quantized
    qq4, qq8, qq12, qq16 = plan.forward_waypoints_rad
    assert plan.return_waypoints_rad == (qq12, qq8, qq4, q0)
    assert payload["forward_endpoint_q_rad"] == list(qq16)
    assert payload["command_count"] == 8
    assert payload["press_overtravel_m"] == PRESS_OVERTRAVEL_M
    assert payload["commanded_forward_distance_m"] == pytest.approx(
        payload["current_panel_gap_m"] + PRESS_OVERTRAVEL_M,
        abs=1e-12,
    )
    assert payload["forward_target_tcp_base_m"] == pytest.approx([
        receipt.p_button_base_m[i] - receipt.normal_out_base[i] * PRESS_OVERTRAVEL_M
        for i in range(3)
    ])
    residual_limit = model.policy["position_residual_limit_m"]
    assert payload["positional_contact_depth_limit_m"] <= PRESS_OVERTRAVEL_M + residual_limit
    assert payload["endpoint_target_residual_m"] <= residual_limit
    assert abs(payload["endpoint_projection_error_m"]) <= residual_limit

    caps = tuple(payload["per_command_caps_rad"])
    first_buffer = payload["runtime_first_segment_cap_buffer_rad"]
    later_buffer = payload["runtime_subsequent_segment_cap_buffer_rad"]
    assert any(
        abs(value - start) + first_buffer > cap
        for value, start, cap in zip(qq16, q0, caps)
    )
    commands = plan.forward_waypoints_rad + plan.return_waypoints_rad
    previous = q0
    for index, point in enumerate(commands):
        buffer = first_buffer if index == 0 else later_buffer
        assert all(
            abs(value - prior) + buffer <= cap + 1e-12
            for value, prior, cap in zip(point, previous, caps)
        )
        previous = point

    start_tcp = tuple(payload["start_tcp_base_m"])
    direction = tuple(payload["forward_direction_base"])
    previous_tcp = start_tcp
    forward_segment_projections_m = []
    for waypoint in plan.forward_waypoints_rad:
        tcp = tuple(model.fk(waypoint).tcp)
        forward_segment_projections_m.append(sum(
            (tcp[axis] - previous_tcp[axis]) * direction[axis]
            for axis in range(3)
        ))
        previous_tcp = tcp
    assert min(forward_segment_projections_m) > 0.015
    tangent_offset = tuple(
        receipt.p_button_base_m[i] - start_tcp[i]
        - direction[i] * payload["current_panel_gap_m"]
        for i in range(3)
    )
    out_audit = _audit_runtime_normal_path(
        model, q0, plan.forward_waypoints_rad,
        reference_tcp_base_m=start_tcp,
        forward_direction_base=direction,
        current_panel_gap_m=payload["commanded_forward_distance_m"],
        residual_limit_m=model.policy["position_residual_limit_m"],
        retreat=False,
        sample_count=1001,
        tangent_offset_base_m=tangent_offset,
    )
    back_audit = _audit_runtime_normal_path(
        model, qq16, plan.return_waypoints_rad,
        reference_tcp_base_m=start_tcp,
        forward_direction_base=direction,
        current_panel_gap_m=payload["commanded_forward_distance_m"],
        residual_limit_m=model.policy["position_residual_limit_m"],
        retreat=True,
        sample_count=1001,
        tangent_offset_base_m=tangent_offset,
    )
    assert out_audit["reverse_step_count"] == back_audit["reverse_step_count"] == 0
    assert out_audit["maximum_lateral_m"] <= residual_limit
    assert back_audit["maximum_lateral_m"] <= residual_limit

    stress_model = SimpleNamespace(policy=deepcopy(model.policy), fk=model.fk)
    stress_model.policy["segment_sample_count"] = 1001
    for start, waypoints in ((q0, plan.forward_waypoints_rad), (qq16, plan.return_waypoints_rad)):
        validate_model_path(
            stress_model,
            start,
            waypoints,
            wall_surface_point_base_m=receipt.wall_surface_point_base_m,
            normal_out_base=receipt.normal_out_base,
            positional_contact_capsules=("tool",),
            positional_contact_depth_limit_m=payload["positional_contact_depth_limit_m"],
        )
    assert verify_stage_b_plan(
        canonical_json(payload),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=model,
        current_q_rad=q0,
        now_ns=1_100_000_000,
        expected_press_authorized=True,
    )["command_count"] == 8


def test_observed_gap_adds_exactly_one_4mm_overtravel_and_keeps_four_out_four_back():
    root = Path(__file__).resolve().parents[2]
    model = FrozenModel.load_once(
        root,
        root / "config/press/releases/t102_dynamic_precontact_policy_release_20260818.json",
    )
    receipt = make_field_receipt_at_gap(model, 0.070604)
    plan = prepare(receipt, model)
    payload = plan.plan
    direction = tuple(payload["forward_direction_base"])
    normal = tuple(payload["normal_out_base"])
    start_tcp = tuple(payload["start_tcp_base_m"])
    p_button = tuple(payload["p_button_base_m"])
    target = tuple(payload["forward_target_tcp_base_m"])

    assert payload["current_panel_gap_m"] == pytest.approx(0.070604, abs=1e-12)
    assert payload["press_overtravel_m"] == PRESS_OVERTRAVEL_M
    for field in (
        "requested_forward_distance_m",
        "commanded_forward_distance_m",
        "approach_distance_m",
    ):
        assert payload[field] == pytest.approx(0.074604, abs=1e-12)
    assert target == pytest.approx(
        tuple(start_tcp[index] + direction[index] * 0.074604 for index in range(3)),
        abs=1e-12,
    )
    assert sum(
        normal[index] * (target[index] - p_button[index])
        for index in range(3)
    ) == pytest.approx(-PRESS_OVERTRAVEL_M, abs=APPROACH_GEOMETRY_TOLERANCE_M)
    assert payload["runtime_source_waypoint_indices"] == [4, 8, 12, 16]
    assert len(plan.forward_waypoints_rad) == len(plan.return_waypoints_rad) == 4
    assert plan.return_waypoints_rad == tuple(
        reversed(plan.forward_waypoints_rad[:-1])
    ) + (tuple(receipt.terminal_q_rad),)
    assert payload["command_count"] == 8
    assert payload["positional_contact_depth_limit_m"] <= (
        PRESS_OVERTRAVEL_M + model.policy["position_residual_limit_m"]
    )
    assert verify_stage_b_plan(
        canonical_json(payload),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=model,
        current_q_rad=receipt.terminal_q_rad,
        now_ns=1_100_000_000,
        expected_press_authorized=True,
    )["command_count"] == 8


def test_verify_rejects_stale_3mm_plan_after_four_mm_update():
    model = Model()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    stale = deepcopy(plan.plan)
    stale["press_overtravel_m"] = 0.003

    with pytest.raises(StageBBlocked, match="CURRENT_PANEL_GAP_INVALID"):
        verify_stage_b_plan(
            canonical_json(stale),
            canonical_hash(stale),
            expected_receipt=receipt,
            model=model,
            current_q_rad=receipt.terminal_q_rad,
            now_ns=1_100_000_000,
            expected_press_authorized=True,
        )


@pytest.mark.parametrize(("rejected_counts", "expected_count"), [
    (frozenset(), 4),
    (frozenset({4}), 3),
    (frozenset({4, 3}), 2),
])
def test_runtime_selector_prefers_four_then_falls_back_to_three_then_two(
    rejected_counts, expected_count,
):
    q0 = (0.0,) * 6
    dense = tuple((value, 0.0, 0.0, 0.0, 0.0, 0.0) for value in (0.25, 0.5, 0.75, 1.0))
    seen = []

    def validate(forward, _back):
        seen.append(len(forward))
        if len(forward) in rejected_counts:
            raise StageBBlocked("TEST_REJECT")

    forward, back, indices, _score = _select_runtime_waypoints(
        dense,
        q0,
        (0.7,) * 6,
        first_segment_buffer_rad=0.05,
        subsequent_segment_buffer_rad=0.02,
        minimum_command_delta_rad=0.01,
        validate_candidate=validate,
    )
    assert len(forward) == expected_count
    assert indices[-1] == 4
    assert back == tuple(reversed(forward[:-1])) + (q0,)
    assert seen[0] == 4
    assert 1 not in seen


def test_runtime_selector_fails_closed_after_four_without_dense_fallback():
    dense = tuple((value, 0.0, 0.0, 0.0, 0.0, 0.0) for value in (0.25, 0.5, 0.75, 1.0))
    seen = []

    def reject(forward, _back):
        seen.append(len(forward))
        raise StageBBlocked("TEST_REJECT")

    with pytest.raises(StageBBlocked, match="RUNTIME_WAYPOINT_SELECTION_FAILED"):
        _select_runtime_waypoints(
            dense,
            (0.0,) * 6,
            (0.7,) * 6,
            first_segment_buffer_rad=0.05,
            subsequent_segment_buffer_rad=0.02,
            minimum_command_delta_rad=0.01,
            validate_candidate=reject,
        )
    assert set(seen) == {2, 3, 4}


def test_runtime_quantization_matches_safe_driver_round_and_command_limit_clamp():
    q0 = (0.0, 0.0, 1.0, 0.0, 0.2, 0.3)
    point = (4.0, -2.0, -0.1, 2.0, -0.9, -0.8)
    quantized = _quantize_runtime_waypoint(point, q0, DEFAULT_ENCODER_STEP_RAD)
    assert quantized[:4] == pytest.approx(
        (3.141592653589793, -1.5707963267948966, 0.0, 1.5707963267948966)
    )
    assert quantized[4:] == (0.2, 0.3)


def test_prepare_freezes_quantized_gap_plus_overtravel_and_largest_mirrored_return():
    model = Model()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    payload = plan.plan
    assert LEASE_PURPOSE == "T102_DYNAMIC_PRECONTACT_EXACT_PLAN_NO_REISSUE"
    assert payload["lease_purpose"] == LEASE_PURPOSE
    assert payload["requested_forward_distance_m"] == FORWARD_DISTANCE_M
    assert payload["requested_forward_distance_m"] == pytest.approx(0.074)
    assert payload["press_overtravel_m"] == PRESS_OVERTRAVEL_M == 0.004
    assert payload["approach_geometry_tolerance_m"] == APPROACH_GEOMETRY_TOLERANCE_M
    assert payload["forward_direction_base"] == [-1.0, 0.0, 0.0]
    assert payload["p_button_base_m"] == list(receipt.p_button_base_m)
    assert payload["forward_target_tcp_base_m"] == pytest.approx([0.026, 0.0, 0.0])
    assert payload["endpoint_target_residual_m"] <= model.policy["position_residual_limit_m"]
    assert payload["return_waypoints_rad"][-1] == list(receipt.terminal_q_rad)
    # This pure model exposes only three dense Cartesian samples; the field
    # release exposes sixteen and therefore selects four runtime waypoints.
    assert len(payload["forward_waypoints_rad"]) == 3
    assert len(payload["return_waypoints_rad"]) == 3
    assert payload["forward_waypoints_rad"][-1] == payload["forward_endpoint_q_rad"]
    assert payload["return_waypoints_rad"] == list(
        reversed(payload["forward_waypoints_rad"][:-1])
    ) + [list(receipt.terminal_q_rad)]
    assert payload["command_count"] == (
        len(payload["forward_waypoints_rad"]) + len(payload["return_waypoints_rad"])
    )
    assert payload["command_count"] == 6
    assert payload["runtime_quantization_step_rad"] == DEFAULT_ENCODER_STEP_RAD
    assert payload["operation"] == "CONTACT_APPROACH_OUT_BACK"
    assert payload["contact_authorized"] is True
    assert payload["press_authorized"] is True
    assert payload["press"] is False
    assert payload["force_control"] is False
    assert payload["zero_dwell"] is True
    projections = [payload["start_tcp_base_m"][0] - point[0] for point in payload["forward_sample_tcp_base_m"]]
    assert projections == sorted(projections)
    for point in payload["forward_waypoints_rad"] + payload["return_waypoints_rad"]:
        assert point[4] == receipt.terminal_q_rad[4]
        assert point[5] == receipt.terminal_q_rad[5]


def test_receipt_requires_click_scoped_fresh_wrist_lock():
    for changes in (
        {"wrist_acquisition_id": ""},
        {"wrist_lock_identity": []},
        {"wrist_lock_stage": "VIS-LOCK"},
        {"wrist_lock_observation_count": 2},
    ):
        with pytest.raises(StageBBlocked, match="WRIST_LOCK_INVALID|ARRIVAL_RECEIPT_INVALID"):
            changed_receipt(**changes)


def test_prepare_uses_fresh_actual_fk_measured_gap_and_returns_exact_actual_q0():
    actual_q = [0.095, 0.0, 0.0, 0.0, 0.2, 0.3]
    receipt = changed_receipt(
        terminal_q_rad=actual_q,
    )
    plan = prepare(receipt)

    assert plan.plan["start_tcp_base_m"] == pytest.approx([0.095, 0.0, 0.0])
    assert plan.plan["current_panel_gap_m"] == pytest.approx(0.065)
    assert plan.plan["forward_target_tcp_base_m"] == pytest.approx([0.026, 0.0, 0.0])
    assert plan.plan["endpoint_target_residual_m"] <= Model().policy["position_residual_limit_m"]
    assert plan.plan["return_waypoints_rad"][-1] == actual_q
    assert plan.plan["command_count"] == len(plan.forward_waypoints_rad) + len(plan.return_waypoints_rad)
    verified = verify_stage_b_plan(
        canonical_json(plan.plan),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=Model(),
        current_q_rad=actual_q,
        now_ns=1_100_000_000,
        expected_press_authorized=True,
    )
    assert verified["return_endpoint_exact_frozen_start_q"] is True


def test_tool_contact_depth_is_locked_to_frozen_endpoint_centerline():
    model = ToolOffsetModel()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    payload = plan.plan
    assert payload["forward_target_tcp_base_m"] == pytest.approx([0.026, 0.0, 0.0])
    assert payload["positional_contact_depth_limit_m"] == pytest.approx(
        0.03 - (payload["forward_sample_tcp_base_m"][-1][0] - 0.002)
    )
    assert payload["positional_contact_depth_limit_m"] <= (
        PRESS_OVERTRAVEL_M + model.policy["position_residual_limit_m"]
    )
    verified = verify_stage_b_plan(
        canonical_json(payload),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=model,
        current_q_rad=receipt.terminal_q_rad,
        now_ns=1_100_000_000,
        expected_press_authorized=True,
    )
    assert verified["command_count"] == len(plan.forward_waypoints_rad) + len(plan.return_waypoints_rad)


def test_path_exceeding_frozen_endpoint_tool_depth_is_rejected():
    model = ToolOffsetModel()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    q0 = tuple(receipt.terminal_q_rad)
    endpoint = tuple(plan.plan["forward_endpoint_q_rad"])
    with pytest.raises(StageBBlocked, match="CONTACT_WALL_OVERSHOOT.*tool"):
        validate_model_path(
            model,
            q0,
            ((0.0, 0.0, 0.0, 0.0, 0.2, 0.3), endpoint),
            wall_surface_point_base_m=receipt.wall_surface_point_base_m,
            normal_out_base=receipt.normal_out_base,
            positional_contact_capsules=("tool",),
            positional_contact_depth_limit_m=plan.plan["positional_contact_depth_limit_m"],
        )
    altered = dict(plan.plan)
    altered["positional_contact_depth_limit_m"] = _contact_depth_from_endpoint(
        model,
        altered["offline_forward_validation_waypoints_rad"][-1],
        wall_surface_point_base_m=receipt.wall_surface_point_base_m,
        normal_out_base=receipt.normal_out_base,
        positional_contact_capsules=("tool",),
    )
    with pytest.raises(StageBBlocked, match="PLAN_INVALIDATED.*endpoint contact depth"):
        verify_stage_b_plan(
            canonical_json(altered),
            canonical_hash(altered),
            expected_receipt=receipt,
            model=model,
            current_q_rad=receipt.terminal_q_rad,
            now_ns=1_100_000_000,
            expected_press_authorized=True,
        )


def test_non_tool_capsule_wall_gate_remains_rejecting():
    with pytest.raises(StageBBlocked, match="WALL_HALFSPACE_REJECTED.*guard"):
        validate_model_path(
            ToolAndGuardModel(),
            (0.10, 0.0, 0.0, 0.0, 0.2, 0.3),
            ((0.03, 0.0, 0.0, 0.0, 0.2, 0.3),),
            wall_surface_point_base_m=(0.03, 0.0, 0.0),
            normal_out_base=(1.0, 0.0, 0.0),
            positional_contact_capsules=("tool",),
            positional_contact_depth_limit_m=0.02,
        )


def test_prepare_rejects_tool_depth_beyond_overtravel_plus_existing_residual():
    model = ExcessiveToolOffsetModel()
    with pytest.raises(StageBBlocked, match="CONTACT_DEPTH_LIMIT_EXCEEDED"):
        prepare(make_receipt(), model)


def test_runtime_uses_largest_cap_valid_quantized_subsequence_and_exact_mirror():
    model = SmallCapModel()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    payload = plan.plan
    q0 = tuple(receipt.terminal_q_rad)
    endpoint = tuple(payload["forward_endpoint_q_rad"])
    caps = tuple(model.policy["per_command_caps_rad"])

    # The direct endpoint delta exceeds one command cap. Runtime chooses the
    # largest cap-valid ordered subsequence, not the full offline partition.
    assert any(abs(value - start) > cap for value, start, cap in zip(endpoint, q0, caps))
    assert len(payload["forward_waypoints_rad"]) == 3
    assert len(payload["offline_forward_validation_waypoints_rad"]) >= 3
    assert payload["forward_waypoints_rad"] != payload["offline_forward_validation_waypoints_rad"]
    assert payload["return_waypoints_rad"] == list(
        reversed(payload["forward_waypoints_rad"][:-1])
    ) + [list(q0)]
    assert payload["command_count"] == len(payload["forward_waypoints_rad"]) + len(payload["return_waypoints_rad"])

    first_buffer = payload["runtime_first_segment_cap_buffer_rad"]
    later_buffer = payload["runtime_subsequent_segment_cap_buffer_rad"]
    previous = q0
    for index, point in enumerate(payload["forward_waypoints_rad"]):
        buffer = first_buffer if index == 0 else later_buffer
        assert all(
            abs(value - prior) + buffer <= cap + 1e-12
            for value, prior, cap in zip(point, previous, caps)
        )
        previous = tuple(point)
    previous = endpoint
    for point in payload["return_waypoints_rad"]:
        assert all(
            abs(value - prior) + later_buffer <= cap + 1e-12
            for value, prior, cap in zip(point, previous, caps)
        )
        previous = tuple(point)
    assert previous == q0

    verified = verify_stage_b_plan(
        canonical_json(payload),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=model,
        current_q_rad=q0,
        now_ns=1_100_000_000,
        expected_press_authorized=True,
    )
    assert verified["command_count"] == payload["command_count"]


def test_verify_rejects_an_over_cap_offline_audit_segment():
    model = SmallCapModel()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    altered = dict(plan.plan)
    offline = [list(point) for point in altered["offline_forward_validation_waypoints_rad"]]
    offline[0][0] = receipt.terminal_q_rad[0] - 0.062
    altered["offline_forward_validation_waypoints_rad"] = offline

    with pytest.raises(StageBBlocked, match="WAYPOINT_CAP_REJECTED"):
        verify_stage_b_plan(
            canonical_json(altered),
            canonical_hash(altered),
            expected_receipt=receipt,
            model=model,
            current_q_rad=receipt.terminal_q_rad,
            now_ns=1_100_000_000,
            expected_press_authorized=True,
        )


def test_verify_rejects_runtime_path_that_differs_from_recomputed_minimum_selection():
    model = SmallCapModel()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    altered = dict(plan.plan)
    altered["forward_waypoints_rad"] = altered["forward_waypoints_rad"][1:]
    altered["return_waypoints_rad"] = list(
        reversed(altered["forward_waypoints_rad"][:-1])
    ) + [list(receipt.terminal_q_rad)]
    altered["command_count"] = len(altered["forward_waypoints_rad"]) + len(
        altered["return_waypoints_rad"]
    )

    with pytest.raises(StageBBlocked, match="PLAN_INVALIDATED.*runtime waypoint selection"):
        verify_stage_b_plan(
            canonical_json(altered),
            canonical_hash(altered),
            expected_receipt=receipt,
            model=model,
            current_q_rad=receipt.terminal_q_rad,
            now_ns=1_100_000_000,
            expected_press_authorized=True,
        )


def test_verify_rejects_runtime_quantization_shape_and_identity_tampering():
    model = Model()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    variants = []

    missing = deepcopy(plan.plan)
    missing["forward_waypoints_rad"] = missing["forward_waypoints_rad"][1:]
    missing["return_waypoints_rad"] = [list(receipt.terminal_q_rad)]
    missing["forward_endpoint_q_rad"] = list(missing["forward_waypoints_rad"][-1])
    missing["command_count"] = 2
    variants.append(missing)

    reordered = deepcopy(plan.plan)
    reordered["forward_waypoints_rad"].reverse()
    reordered["forward_endpoint_q_rad"] = list(reordered["forward_waypoints_rad"][-1])
    reordered["return_waypoints_rad"] = list(
        reversed(reordered["forward_waypoints_rad"][:-1])
    ) + [list(receipt.terminal_q_rad)]
    variants.append(reordered)

    non_mirror = deepcopy(plan.plan)
    non_mirror["return_waypoints_rad"][0][0] += DEFAULT_ENCODER_STEP_RAD
    variants.append(non_mirror)

    wrong_count = deepcopy(plan.plan)
    wrong_count["command_count"] += 1
    variants.append(wrong_count)

    wrong_indices = deepcopy(plan.plan)
    wrong_indices["runtime_source_waypoint_indices"][0] += 1
    variants.append(wrong_indices)

    wrong_step = deepcopy(plan.plan)
    wrong_step["runtime_quantization_step_rad"] += 1e-6
    variants.append(wrong_step)

    wrong_endpoint = deepcopy(plan.plan)
    wrong_endpoint["forward_waypoints_rad"][-1][0] += DEFAULT_ENCODER_STEP_RAD
    wrong_endpoint["forward_endpoint_q_rad"] = list(wrong_endpoint["forward_waypoints_rad"][-1])
    wrong_endpoint["return_waypoints_rad"] = list(
        reversed(wrong_endpoint["forward_waypoints_rad"][:-1])
    ) + [list(receipt.terminal_q_rad)]
    variants.append(wrong_endpoint)

    wrong_depth = deepcopy(plan.plan)
    wrong_depth["positional_contact_depth_limit_m"] += 1e-4
    variants.append(wrong_depth)

    excessive_depth = deepcopy(plan.plan)
    excessive_depth["positional_contact_depth_limit_m"] = (
        PRESS_OVERTRAVEL_M + model.policy["position_residual_limit_m"] + 1e-6
    )
    variants.append(excessive_depth)

    wrong_overtravel = deepcopy(plan.plan)
    wrong_overtravel["press_overtravel_m"] = PRESS_OVERTRAVEL_M - 1e-4
    variants.append(wrong_overtravel)

    wrong_stroke = deepcopy(plan.plan)
    wrong_stroke["commanded_forward_distance_m"] += 1e-4
    variants.append(wrong_stroke)

    wrong_cartesian_target = deepcopy(plan.plan)
    wrong_cartesian_target["forward_target_tcp_base_m"][0] += 1e-4
    variants.append(wrong_cartesian_target)

    for altered in variants:
        with pytest.raises(StageBBlocked):
            verify_stage_b_plan(
                canonical_json(altered),
                canonical_hash(altered),
                expected_receipt=receipt,
                model=model,
                current_q_rad=receipt.terminal_q_rad,
                now_ns=1_100_000_000,
                expected_press_authorized=True,
            )


def test_safe_terminal_actual_q_at_point_527_deg_error_uses_actual_fk_start():
    actual_q = [0.10306796, -0.009203885, 0.00153398, 0.00306796, 0.2, 0.3]
    receipt = changed_receipt(
        terminal_q_rad=actual_q,
    )

    plan = prepare(receipt)
    assert plan.plan["start_tcp_base_m"] == pytest.approx([actual_q[0], 0.0, 0.0])
    assert plan.plan["current_panel_gap_m"] == pytest.approx(actual_q[0] - 0.030)
    assert plan.plan["forward_target_tcp_base_m"] == pytest.approx([0.026, 0.0, 0.0])
    assert plan.plan["return_waypoints_rad"][-1] == actual_q


@pytest.mark.parametrize("gap_m", [0.070, 0.095])
def test_prepare_adapts_stroke_to_current_pre_gap(gap_m):
    start_x = 0.030 + gap_m
    actual_q = [start_x, 0.0, 0.0, 0.0, 0.2, 0.3]
    receipt = changed_receipt(
        terminal_q_rad=actual_q,
    )

    plan = prepare(receipt)

    assert plan.plan["current_panel_gap_m"] == pytest.approx(gap_m)
    assert plan.plan["requested_forward_distance_m"] == pytest.approx(
        gap_m + PRESS_OVERTRAVEL_M
    )
    assert plan.plan["forward_target_tcp_base_m"] == pytest.approx([0.026, 0.0, 0.0])
    assert plan.plan["endpoint_projection_error_m"] <= Model().policy["position_residual_limit_m"]
    assert plan.plan["return_waypoints_rad"][-1] == actual_q
    assert plan.plan["forward_waypoints_rad"][-1] == plan.plan["forward_endpoint_q_rad"]
    assert plan.plan["return_waypoints_rad"] == list(
        reversed(plan.plan["forward_waypoints_rad"][:-1])
    ) + [actual_q]
    assert plan.plan["command_count"] == len(plan.forward_waypoints_rad) + len(plan.return_waypoints_rad)


def test_minimum_panel_gap_still_applies_full_overtravel_without_clamping():
    receipt = changed_receipt(terminal_q_rad=[0.031, 0.0, 0.0, 0.0, 0.2, 0.3])
    model = Model()
    # This synthetic model maps one radian to one metre. Use a fine encoder
    # for this geometry-only boundary test; physical encoder limits stay fixed.
    model.policy["encoder_step_rad"] = 0.0001
    plan = prepare(receipt, model)
    assert plan.plan["current_panel_gap_m"] == pytest.approx(0.001)
    assert plan.plan["commanded_forward_distance_m"] == pytest.approx(0.005)
    assert plan.plan["forward_target_tcp_base_m"] == pytest.approx([0.026, 0.0, 0.0])


def test_prepare_rejects_99mm_gap_instead_of_clamping_total_stroke_above_100mm():
    receipt = changed_receipt(terminal_q_rad=[0.129, 0.0, 0.0, 0.0, 0.2, 0.3])
    with pytest.raises(StageBBlocked, match="CURRENT_PANEL_GAP_TOO_LARGE"):
        prepare(receipt)


@pytest.mark.parametrize("gap_m", [-0.001, 0.0, 0.1001])
def test_prepare_rejects_current_pre_gap_outside_adaptive_range(gap_m):
    start_x = 0.030 + gap_m
    receipt = changed_receipt(
        terminal_q_rad=[start_x, 0.0, 0.0, 0.0, 0.2, 0.3],
    )
    expected = "CURRENT_PANEL_GAP_TOO_LARGE" if gap_m > 0.100 else "CURRENT_PANEL_GAP_INVALID"
    with pytest.raises(StageBBlocked, match=expected):
        prepare(receipt)


def test_execution_verify_still_rejects_current_q_drift_from_fresh_actual_q0():
    receipt = changed_receipt(
        terminal_q_rad=[0.10306796, -0.009203885, 0.00153398, 0.00306796, 0.2, 0.3],
    )
    plan = prepare(receipt)
    drifted = list(receipt.terminal_q_rad)
    drifted[0] += 0.0031
    with pytest.raises(StageBBlocked, match="START_Q_DRIFT"):
        verify_stage_b_plan(
            canonical_json(plan.plan),
            plan.plan_sha256,
            expected_receipt=receipt,
            model=Model(),
            current_q_rad=drifted,
            now_ns=1_100_000_000,
        )


def test_receipt_requires_panel_geometry_artifact_hash():
    raw = dict(make_receipt().unsigned_mapping())
    raw.pop("panel_geometry_sha256")
    digest = canonical_hash(raw)
    raw.update(receipt_id=digest, receipt_sha256=digest)
    with pytest.raises(StageBBlocked):
        PreTerminalReceipt.from_mapping(raw)


def test_default_contact_authority_is_closed():
    with pytest.raises(StageBBlocked, match="CONTACT_AUTHORIZATION_DISABLED"):
        prepare_stage_b_plan(make_receipt(), Model(), now_ns=1_100_000_000)


def test_plan_hash_and_identity_are_rechecked():
    model = Model()
    receipt = make_receipt()
    plan = prepare(receipt, model)
    checked = verify_stage_b_plan(
        canonical_json(plan.plan),
        plan.plan_sha256,
        expected_receipt=receipt,
        model=model,
        current_q_rad=receipt.terminal_q_rad,
        now_ns=1_100_000_000,
        expected_press_authorized=True,
    )
    assert checked["panel_press_receipt_sha256"] == receipt.receipt_sha256
    altered = dict(plan.plan)
    altered["panel_geometry_sha256"] = "f" * 64
    with pytest.raises(StageBBlocked):
        verify_stage_b_plan(
            canonical_json(altered),
            canonical_hash(altered),
            expected_receipt=receipt,
            model=model,
            current_q_rad=receipt.terminal_q_rad,
            now_ns=1_100_000_000,
        )


def sample(q, *, pubs=0, lease=False, state="CONVERGED_DISARMED", armed=False,
           sequence=0, accepted=(), advance=False, received=1.0,
           feedback_received=None, status_received=None, feedback_age=0.01,
           feedback_stamp_age=0.01):
    if feedback_received is None:
        feedback_received = received
    if status_received is None:
        status_received = received
    return StageBSafetySample(
        tuple(q), feedback_age, state, armed, False, "", (True,) * 6, 12.0,
        pubs, lease, LEASE_PURPOSE if lease else "", lease, 0.01,
        feedback_received, feedback_stamp_age, sequence, tuple(accepted), advance, status_received,
    )


def safe_accepted_target(target, boundary):
    result = []
    for value, measured in zip(target, boundary):
        if abs(value - measured) > DEFAULT_ENCODER_STEP_RAD + 1e-12:
            result.append(round(value / DEFAULT_ENCODER_STEP_RAD) * DEFAULT_ENCODER_STEP_RAD)
        else:
            result.append(measured)
    return tuple(result)


def drive_command(machine, target, phase, sequence, now, boundary, *, final=False):
    machine.mark_published(
        phase, target, prior_sequence=sequence,
        publish_boundary_positions=boundary, now_monotonic=now,
    )
    accepted_target = safe_accepted_target(target, boundary)
    active = sample(accepted_target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
                    sequence=sequence + 1, accepted=accepted_target, advance=not final, received=now + 0.01)
    machine.accept_command(active, now_monotonic=now + 0.01)
    terminal = (
        sample(accepted_target, pubs=1, lease=True, sequence=sequence + 1, accepted=accepted_target, received=now + 0.02)
        if final else active
    )
    machine.observe(terminal, now_monotonic=now + 0.02)
    return accepted_target


def test_exact_machine_publishes_every_frozen_waypoint_once_and_zero_dwell_return():
    plan = prepare()
    q0 = plan.receipt.terminal_q_rad
    machine = StageBExactExecution(plan, StageBExecutionConfig(), now_monotonic=1.0)
    machine.precheck(sample(q0))
    machine.lease_acquired(sample(q0, lease=True))
    machine.publisher_created(sample(q0, pubs=1, lease=True))
    machine.armed(True)
    sequence = 4
    now = 1.1
    boundary = tuple(q0)
    expected = [("FORWARD", point) for point in plan.forward_waypoints_rad] + [
        ("RETURN", point) for point in plan.return_waypoints_rad
    ]
    assert len(plan.forward_waypoints_rad) == len(plan.return_waypoints_rad) == 3
    assert len(expected) == 6
    for index, (phase, target) in enumerate(expected):
        boundary = drive_command(
            machine, target, phase, sequence, now, boundary,
            final=index + 1 == len(expected),
        )
        sequence += 1
        now += 0.02
    assert machine.state == "TEARDOWN_SUCCESS"
    assert machine.publish_log == expected
    assert len(machine.publish_log) == len(set((phase, target) for phase, target in expected))
    assert machine.first_return_published_monotonic == machine.forward_endpoint_converged_monotonic
    assert machine.publish_log[-1][1] == tuple(q0)


def test_forward_failure_never_opens_return_and_reissue_is_blocked():
    plan = prepare()
    forward = plan.forward_waypoints_rad[0]
    q0 = plan.receipt.terminal_q_rad
    machine = StageBExactExecution(plan, now_monotonic=1.0)
    machine.precheck(sample(q0)); machine.lease_acquired(sample(q0, lease=True))
    machine.publisher_created(sample(q0, pubs=1, lease=True)); machine.armed(True)
    machine.mark_published(
        "FORWARD", forward, prior_sequence=0,
        publish_boundary_positions=q0, now_monotonic=1.1,
    )
    with pytest.raises(StageBBlocked, match="ACK_INVALID"):
        machine.accept_command(
            sample(forward, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
                   sequence=2, accepted=forward, received=1.2),
            now_monotonic=1.2,
        )
    assert [phase for phase, _target in machine.publish_log] == ["FORWARD"]
    with pytest.raises(StageBBlocked):
        machine.mark_published(
            "RETURN", plan.return_waypoints_rad[0], prior_sequence=0,
            publish_boundary_positions=forward, now_monotonic=1.3,
        )


def _machine_waiting_for_ack(plan):
    q0 = plan.receipt.terminal_q_rad
    machine = StageBExactExecution(plan, now_monotonic=1.0)
    machine.precheck(sample(q0))
    machine.lease_acquired(sample(q0, lease=True))
    machine.publisher_created(sample(q0, pubs=1, lease=True))
    machine.armed(True)
    return machine, q0


def _machine_at_first_post_arm_publish_boundary(plan):
    q0 = plan.receipt.terminal_q_rad
    machine = StageBExactExecution(plan, now_monotonic=1.0)
    machine.precheck(sample(q0))
    machine.lease_acquired(sample(q0, lease=True))
    machine.publisher_created(sample(q0, pubs=1, lease=True))
    machine.armed(True)
    return machine, q0


def _machine_at_advance_publish_boundary(plan):
    machine, q0 = _machine_waiting_for_ack(plan)
    accepted = drive_command(
        machine,
        plan.forward_waypoints_rad[0],
        "FORWARD",
        4,
        1.1,
        q0,
    )
    assert machine.state == "PUBLISH_WAYPOINT"
    return machine, accepted, 5


def test_first_publish_boundary_requires_fresh_post_arm_health_and_frozen_q0_drift():
    plan = prepare()
    machine, q0 = _machine_at_first_post_arm_publish_boundary(plan)
    boundary = sample(q0, pubs=1, lease=True, state="ARMED", armed=True)
    machine.validate_publish_boundary(boundary, first_command=True)

    drifted = list(q0)
    drifted[0] += machine.q0_drift_limit_rad + 1e-6
    with pytest.raises(StageBBlocked, match="START_Q_DRIFT"):
        machine.validate_publish_boundary(
            sample(drifted, pubs=1, lease=True, state="ARMED", armed=True),
            first_command=True,
        )
    unhealthy = StageBSafetySample(**{**boundary.__dict__, "feedback_age_s": 1.0})
    with pytest.raises(StageBBlocked, match="MOVE_FEEDBACK_STALE"):
        machine.validate_publish_boundary(unhealthy, first_command=True)
    with pytest.raises(StageBBlocked, match="fresh post-arm ARMED"):
        machine.validate_publish_boundary(
            sample(q0, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True),
            first_command=True,
        )


def test_advance_publish_boundary_accepts_exact_fresh_bound_ack():
    machine, accepted, sequence = _machine_at_advance_publish_boundary(prepare())
    machine.validate_publish_boundary(
        sample(
            accepted,
            pubs=1,
            lease=True,
            state="COMMAND_ACTIVE",
            armed=True,
            sequence=sequence,
            accepted=accepted,
            advance=True,
            received=1.2,
        ),
        first_command=False,
    )


def test_advance_publish_boundary_rejects_stale_owner_ack_and_position_drift():
    machine, accepted, sequence = _machine_at_advance_publish_boundary(prepare())

    def boundary(**changes):
        value = sample(
            accepted,
            pubs=1,
            lease=True,
            state="COMMAND_ACTIVE",
            armed=True,
            sequence=sequence,
            accepted=accepted,
            advance=True,
            received=1.2,
        )
        return StageBSafetySample(**{**value.__dict__, **changes})

    wrong_target = list(accepted)
    wrong_target[0] += 1e-6
    far_position = list(accepted)
    far_position[0] += machine.config.endpoint_tolerance_rad + 1e-6
    rejected = (
        (boundary(feedback_age_s=1.0), "MOVE_FEEDBACK_STALE"),
        (boundary(joint_target_publishers=2), "MOVE_OWNER_DRIFT"),
        (boundary(accepted_command_sequence=sequence + 1), "MOVE_COMMAND_ACK_INVALID"),
        (boundary(accepted_target_positions=tuple(wrong_target)), "MOVE_COMMAND_ACK_INVALID"),
        (boundary(positions_rad=tuple(far_position)), "MOVE_ENDPOINT_ERROR"),
        (boundary(dynamic_exact_advance_ready=False), "MOVE_OWNER_DRIFT"),
    )
    for evidence, code in rejected:
        with pytest.raises(StageBBlocked, match=code):
            machine.validate_publish_boundary(evidence, first_command=False)


def test_command_ack_uses_fresh_status_with_still_fresh_pre_publish_feedback():
    plan = prepare()
    machine, boundary = _machine_waiting_for_ack(plan)
    target = plan.forward_waypoints_rad[0]
    accepted_target = safe_accepted_target(target, boundary)
    machine.mark_published(
        "FORWARD", target, prior_sequence=0,
        publish_boundary_positions=boundary, now_monotonic=1.0,
    )
    # The physical sample was received before publish but remains within the
    # physical freshness window; only the controller status is post-publish.
    accepted = sample(
        accepted_target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
        sequence=1, accepted=accepted_target, feedback_received=0.99,
        status_received=1.01,
    )
    machine.accept_command(accepted, now_monotonic=1.01)
    assert machine.state == "WAIT_WAYPOINT_CONVERGENCE"


def test_command_ack_rejects_stale_status_even_when_feedback_is_fresh():
    plan = prepare()
    machine, boundary = _machine_waiting_for_ack(plan)
    target = plan.forward_waypoints_rad[0]
    accepted_target = safe_accepted_target(target, boundary)
    machine.mark_published(
        "FORWARD", target, prior_sequence=0,
        publish_boundary_positions=boundary, now_monotonic=1.0,
    )
    stale_status = sample(
        accepted_target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
        sequence=1, accepted=accepted_target, feedback_received=1.01,
        status_received=0.99,
    )
    with pytest.raises(StageBBlocked, match="MOVE_COMMAND_ACK_MISSING"):
        machine.accept_command(stale_status, now_monotonic=1.01)


def test_command_ack_still_rejects_stale_physical_feedback():
    plan = prepare()
    machine, boundary = _machine_waiting_for_ack(plan)
    target = plan.forward_waypoints_rad[0]
    accepted_target = safe_accepted_target(target, boundary)
    machine.mark_published(
        "FORWARD", target, prior_sequence=0,
        publish_boundary_positions=boundary, now_monotonic=1.0,
    )
    stale_feedback = sample(
        accepted_target, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
        sequence=1, accepted=accepted_target, feedback_received=0.99,
        status_received=1.01, feedback_age=1.0,
    )
    with pytest.raises(StageBBlocked, match="MOVE_FEEDBACK_STALE"):
        machine.accept_command(stale_feedback, now_monotonic=1.01)


@pytest.mark.parametrize("held_steps", [0.0, 1.0])
def test_ack_matches_safe_driver_quantized_movers_and_boundary_held_joints(held_steps):
    boundary = (0.10, 0.0005, 0.0, 0.0, 0.2, 0.3)
    requested = (0.03, 0.0, 0.0, 0.0, 0.2, 0.3)
    expected = list(safe_accepted_target(requested, boundary))
    expected[1] = boundary[1] + held_steps * DEFAULT_ENCODER_STEP_RAD
    expected = tuple(expected)
    accepted = sample(
        expected, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
        sequence=8, accepted=expected, received=1.2,
    )
    assert validate_command_acceptance(
        accepted,
        prior_sequence=7,
        requested_target=requested,
        publish_boundary_positions=boundary,
        encoder_step_rad=DEFAULT_ENCODER_STEP_RAD,
    ) == expected
    wrong_held = list(expected)
    wrong_held[1] = boundary[1] + DEFAULT_ENCODER_STEP_RAD + 1e-6
    with pytest.raises(StageBBlocked, match="accepted held target"):
        validate_command_acceptance(
            sample(expected, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
                   sequence=8, accepted=wrong_held, received=1.2),
            prior_sequence=7,
            requested_target=requested,
            publish_boundary_positions=boundary,
            encoder_step_rad=DEFAULT_ENCODER_STEP_RAD,
        )
    wrong_mover = list(expected)
    wrong_mover[0] += 1e-6
    with pytest.raises(StageBBlocked, match="accepted mover target"):
        validate_command_acceptance(
            sample(expected, pubs=1, lease=True, state="COMMAND_ACTIVE", armed=True,
                   sequence=8, accepted=wrong_mover, received=1.2),
            prior_sequence=7,
            requested_target=requested,
            publish_boundary_positions=boundary,
            encoder_step_rad=DEFAULT_ENCODER_STEP_RAD,
        )
    with pytest.raises(StageBBlocked, match="accepted sequence"):
        validate_command_acceptance(
            accepted,
            prior_sequence=6,
            requested_target=requested,
            publish_boundary_positions=boundary,
            encoder_step_rad=DEFAULT_ENCODER_STEP_RAD,
        )


def test_stage_b_owner_gate_accepts_only_two_exact_safe_idle_states():
    plan = prepare()
    q0 = plan.receipt.terminal_q_rad
    for state in ("CONVERGED_DISARMED", "DISARMED:operator"):
        machine = StageBExactExecution(plan, now_monotonic=1.0)
        machine.precheck(sample(q0, state=state))
        assert machine.state == "ACQUIRE_LEASE"
    for state in ("DISARMED", "DISARMED:stop", "disarmed:operator", "FAILED_DISARMED"):
        machine = StageBExactExecution(plan, now_monotonic=1.0)
        with pytest.raises(StageBBlocked, match="MOVE_SAFETY_GATE_FAILED"):
            machine.precheck(sample(q0, state=state))


def test_operator_disarmed_owner_gate_keeps_all_other_safety_proofs():
    plan = prepare()
    q0 = plan.receipt.terminal_q_rad
    unsafe = (
        sample(q0, state="DISARMED:operator", armed=True),
        StageBSafetySample(**{**sample(q0, state="DISARMED:operator").__dict__, "fault": "fault"}),
        StageBSafetySample(**{**sample(q0, state="DISARMED:operator").__dict__, "torque_enabled": (True,) * 5 + (False,)}),
        sample(q0, state="DISARMED:operator", lease=True),
        sample(q0, state="DISARMED:operator", pubs=1),
        StageBSafetySample(**{**sample(q0, state="DISARMED:operator").__dict__, "feedback_age_s": 1.0}),
    )
    for evidence in unsafe:
        machine = StageBExactExecution(plan, now_monotonic=1.0)
        with pytest.raises(StageBBlocked):
            machine.precheck(evidence)
