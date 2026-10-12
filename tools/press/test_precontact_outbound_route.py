"""Focused tests for the isolated outbound-only pre-contact route layer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from press.precontact_outbound_route import (
    DIAGNOSTIC_SCHEMA,
    PHASE,
    ROUTE_SCHEMA,
    RouteBlocked,
    build_outbound_route,
    canonical_route_artifact_hash,
    seal_manifest,
    seal_release,
    sha256_canonical,
    validate_outbound_route_artifact,
)


NOW_NS = 2_000_000_000
REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_R6 = REPO_ROOT / "config/calibration/accepted/gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
CURRENT_FN = REPO_ROOT / "config/calibration/candidates/section_c_fingertip_c0_operator_click_best_20260806_v1.json"
CURRENT_STANDBY = REPO_ROOT / "config/press/poses/archive/press_standby_manual_20260803_current.yaml"
COMMAND_HASH_KEYS = (
    "schema",
    "command_mode",
    "driver_profile_release_id",
    "driver_profile_artifact_sha256",
    "joint_order",
    "joint_order_selection_policy",
    "joint_names",
    "encoder_step_rad",
    "joint_max_delta_rad",
    "joint_max_speed_rad_s",
    "joint_max_acceleration_rad_s2",
    "joint_command_timeout_s",
    "joint_limits_rad",
    "desired_waypoints_rad",
    "atomic_steps",
)


class ExactIK:
    """Synthetic solver that proves builder geometry without a robot planner."""

    def __init__(self, positions_rad=(0.13, -0.07, 0.04, 0.0, 0.02, -0.01)):
        self.positions_rad = positions_rad
        self.request = None

    def __call__(self, request):
        self.request = request
        return {
            "status": "converged",
            "positions_rad": self.positions_rad,
            "achieved_hand_position_m": request.hand_target_position_m,
            "achieved_hand_rotation_xyzw": request.hand_target_rotation_xyzw,
        }


class ExactCollision:
    """Synthetic checker that binds its evidence to the supplied model hash."""

    def __init__(self, *, status="clear", clearance_m=0.030):
        self.status = status
        self.clearance_m = clearance_m
        self.request = None
        self.requests = []

    def __call__(self, request):
        self.request = request
        self.requests.append(request)
        return {
            "status": self.status,
            "collision_model_sha256": request.collision_model_sha256,
            "checked_waypoint_count": len(request.path_setpoints_rad),
            "segments_checked": len(request.path_setpoints_rad) - 1,
            "continuous_corridor_status": "clear",
            "swept_segments_status": "clear",
            "minimum_clearance_m": self.clearance_m,
            "keepout_model": "synthetic-static-keepout",
        }


class ExactKinematics:
    """Synthetic FK evidence for the final, already quantized setpoint."""

    def __init__(self, ik):
        self.ik = ik
        self.request = None
        self.requests = []

    def __call__(self, request):
        self.request = request
        self.requests.append(request)
        assert self.ik.request is not None
        return {
            "status": "validated",
            "checked_positions_rad": request.joint_positions_rad,
            "achieved_hand_position_m": self.ik.request.hand_target_position_m,
            "achieved_hand_rotation_xyzw": self.ik.request.hand_target_rotation_xyzw,
        }


def _released_manifest():
    """One entirely synthetic, sealed set of execution-grade inputs."""

    r6 = seal_release(
        "r6-synthetic-release",
        "released",
        {
            "schema": "camera_base_release_v1",
            "relative": False,
            "topology_closed": True,
            "tf_status": "released_execution",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )
    m0 = seal_release(
        "m0-synthetic-frozen",
        "released",
        {
            "schema": "m0_frozen_target_release_v1",
            "complete": True,
            "immutable": True,
            "frame_id": "camera_optical_frame",
            "source_stamp_ns": NOW_NS - 100_000_000,
            "expires_at_ns": NOW_NS + 500_000_000,
            "ttl_ms": 1000,
            "confidence": 0.95,
            "rgb_depth_skew_ms": 2.0,
            "target_id": "button-1",
            "detection_id": "det-1",
            "calibration_id": r6["release_id"],
            "point_camera_m": [0.0, 0.0, 0.500],
            "normal_camera": [0.0, 0.0, 1.0],
        },
    )
    releases = {
        "m0_frozen": m0,
        "r6_bundle": r6,
        "tool_release": seal_release(
            "fn-contact-synthetic-release",
            "released",
            {
                "schema": "accepted_tool_frame_release_v1",
                "parent_frame": "hand_tcp",
                "child_frame": "press_contact",
                "transform_complete": True,
                "contact_surface_accepted": True,
                "rotation_status": "accepted",
                "contact_axis_child": [1.0, 0.0, 0.0],
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        ),
        "standby_release": seal_release(
            "standby-synthetic-release",
            "released",
            {
                "schema": "standby_execution_release_v1",
                "route_start_accepted": True,
                "all_joints_inside_limits": True,
                "corridor_clearance_accepted": True,
                "joint_positions_rad": [0.0] * 6,
            },
        ),
        "urdf_release": seal_release(
            "urdf-synthetic-release",
            "released",
            {"schema": "urdf_release_v1", "urdf_id": "roarm-synthetic-urdf"},
        ),
        "driver_profile_release": seal_release(
            "driver-synthetic-release",
            "released",
            {
                "schema": "driver_profile_release_v1",
                "joint_names": [
                    "base_link_to_link1",
                    "link1_to_link2",
                    "link2_to_link3",
                    "link3_to_link4",
                    "link4_to_link5",
                    "link5_to_gripper_link"
                ],
                "encoder_step_rad": 0.01,
                "joint_max_delta_rad": [0.05] * 6,
                "joint_max_speed_rad_s": [0.20] * 6,
                "joint_max_acceleration_rad_s2": [0.40] * 6,
                "joint_command_timeout_s": [1.0] * 6,
                "joint_order_allowlist": [
                    [0, 1, 2, 3, 4, 5],
                    [5, 4, 3, 2, 1, 0]
                ],
                "joint_limits_rad": [[-1.0, 1.0]] * 6,
            },
        ),
        "normal_decision": seal_release(
            "normal-synthetic-release",
            "released",
            {
                "schema": "normal_decision_release_v1",
                "target_id": "button-1",
                "source_stamp_ns": NOW_NS - 100_000_000,
                "status": "outward_verified",
                "ambiguous": False,
                "input_normal_was_flipped": False,
                "dot_n_out_p_minus_a": 0.10,
                "camera_side_dot": 0.10,
                "normal_out_base": [0.0, 0.0, 1.0],
            },
        ),
        "collision_model_release": seal_release(
            "collision-synthetic-release",
            "released",
            {
                "schema": "collision_model_release_v1",
                "model_status": "released",
                "required_minimum_clearance_m": 0.020,
            },
        ),
    }
    return seal_manifest(
        {
            "schema": "precontact_outbound_release_manifest_v1",
            "manifest_id": "synthetic-precontact-outbound-v1",
            "releases": releases,
            "route_policy": {
                "allowed_phase": PHASE,
                "no_contact": True,
                "automatic_retry": False,
                "precontact_distance_m": 0.050,
                "roll_reference_base": [0.0, 1.0, 0.0],
                "position_tolerance_m": 0.001,
                "orientation_tolerance_deg": 0.5,
            },
        }
    )


def _replace_release(manifest, name, *, payload=None, status=None):
    candidate = copy.deepcopy(manifest)
    previous = candidate["releases"][name]
    candidate["releases"][name] = seal_release(
        previous["release_id"],
        previous["status"] if status is None else status,
        previous["payload"] if payload is None else payload,
    )
    return seal_manifest(candidate)


def _build(manifest=None, *, ik=None, kinematics=None, collision=None, diagnostic=False):
    selected_ik = ExactIK() if ik is None else ik
    return build_outbound_route(
        _released_manifest() if manifest is None else manifest,
        ik_solver=selected_ik,
        kinematics_checker=ExactKinematics(selected_ik) if kinematics is None else kinematics,
        collision_checker=ExactCollision() if collision is None else collision,
        now_ns=NOW_NS,
        diagnostic_on_error=diagnostic,
    )


def _assert_diagnostic(artifact, expected_reason):
    assert artifact["schema"] == DIAGNOSTIC_SCHEMA
    assert artifact["status"] == "blocked_diagnostic_only"
    assert artifact["executable"] is False
    assert artifact["adapter_consumable"] is False
    assert artifact["phases"] == []
    assert expected_reason in artifact["blocked_reason"]
    unsigned = dict(artifact)
    digest = unsigned.pop("canonical_payload_sha256")
    assert digest == sha256_canonical(unsigned)


def _reseal_command_aliases(artifact):
    command = artifact["clearance_evidence"]["command_contract"]
    command["command_sequence_sha256"] = sha256_canonical(
        {key: command[key] for key in COMMAND_HASH_KEYS}
    )
    artifact["command_sequence_sha256"] = command["command_sequence_sha256"]
    artifact["canonical_payload_sha256"] = canonical_route_artifact_hash(artifact)


def _diagnostic(manifest, expected_reason, *, ik=None, kinematics=None, collision=None):
    _assert_diagnostic(
        _build(
            manifest,
            ik=ik,
            kinematics=kinematics,
            collision=collision,
            diagnostic=True,
        ),
        expected_reason,
    )


def test_released_outbound_route_is_exact_quantized_and_no_contact():
    ik = ExactIK()
    kinematics = ExactKinematics(ik)
    collision = ExactCollision()
    artifact = _build(ik=ik, kinematics=kinematics, collision=collision)

    assert artifact["schema"] == ROUTE_SCHEMA
    assert artifact["phase"] == PHASE
    assert artifact["phases"] == [PHASE]
    assert artifact["executable"] is True
    assert artifact["no_contact"] is True
    assert artifact["automatic_retry"] is False
    assert artifact["contact_authorized"] is False
    assert artifact["press_authorized"] is False
    assert artifact["required_execution_interface"] == "SAFE_ADAPTER_ACTION_SERVER"
    assert not ({"PRESS", "RETRACT", "RETURN"} & set(artifact["phases"]))

    geometry = artifact["geometry"]
    assert math.dist(
        geometry["target_surface_base_m"], geometry["precontact_contact_point_base_m"]
    ) == pytest.approx(0.050, abs=1e-12)
    assert geometry["exact_precontact_distance_m"] == 0.050
    assert geometry["contact_axis_base"] == pytest.approx([0.0, 0.0, -1.0])
    assert ik.request.contact_target_position_m == pytest.approx((0.0, 0.0, 0.550))

    emitted = artifact["quantization"]["emitted_setpoints_rad"]
    assert emitted[-1] == pytest.approx([0.13, -0.07, 0.04, 0.0, 0.02, -0.01])
    previous = artifact["quantization"]["start_quantized_rad"]
    for waypoint in emitted:
        for before, after in zip(previous, waypoint):
            assert abs(after - before) <= 0.04 + 1e-12  # cap minus one encoder step
            assert after / 0.01 == pytest.approx(round(after / 0.01))
        previous = waypoint
    selected_path = tuple(
        tuple(point) for point in artifact["clearance_evidence"]["quantized_path_setpoints_rad"]
    )
    assert selected_path[0] == tuple(artifact["quantization"]["start_quantized_rad"])
    assert selected_path[1:] == tuple(tuple(point) for point in emitted)
    assert artifact["clearance_evidence"]["checked_waypoint_count"] == len(emitted) + 1
    assert artifact["clearance_evidence"]["segments_checked"] == len(emitted)
    assert artifact["clearance_evidence"]["continuous_corridor_status"] == "clear"
    assert artifact["clearance_evidence"]["swept_segments_status"] == "clear"
    assert kinematics.request.joint_positions_rad == tuple(emitted[-1])
    assert artifact["clearance_evidence"]["minimum_clearance_m"] == 0.030
    command = artifact["clearance_evidence"]["command_contract"]
    assert artifact["command_mode"] == "T101_SINGLE_JOINT_ONLY"
    assert artifact["command_sequence_sha256"] == command["command_sequence_sha256"]
    assert command["joint_order"] == [0, 1, 2, 3, 4, 5]
    assert command["joint_order_selection_policy"] == (
        "MAXIMUM_MINIMUM_CLEARANCE_THEN_LEXICOGRAPHIC"
    )
    assert command["encoder_step_rad"] == 0.01
    assert command["joint_max_speed_rad_s"] == [0.20] * 6
    assert command["joint_max_acceleration_rad_s2"] == [0.40] * 6
    assert command["joint_command_timeout_s"] == [1.0] * 6
    assert command["joint_limits_rad"] == [[-1.0, 1.0]] * 6
    assert len(command["order_selection_evidence"]) == 2
    previous_atomic = artifact["quantization"]["start_quantized_rad"]
    for index, (step, post) in enumerate(zip(command["atomic_steps"], emitted)):
        assert step["sequence_index"] == index
        assert step["previous_pose_rad"] == pytest.approx(previous_atomic)
        assert step["post_pose_rad"] == pytest.approx(post)
        changed = [
            joint_index
            for joint_index, (before, after) in enumerate(zip(previous_atomic, post))
            if abs(after - before) > 1e-12
        ]
        assert changed == [step["joint_index"]]
        assert step["t101_target_rad"] == pytest.approx(post[step["joint_index"]])
        previous_atomic = post
    assert all(
        len(record["fk_path_evidence"]) == record["collision_evidence"]["checked_waypoint_count"]
        for record in command["order_selection_evidence"]
    )
    assert len(kinematics.requests) == sum(
        record["collision_evidence"]["checked_waypoint_count"]
        for record in command["order_selection_evidence"]
    )
    for joint_order in {(request.joint_order) for request in kinematics.requests}:
        order_requests = [request for request in kinematics.requests if request.joint_order == joint_order]
        assert order_requests[0].path_index == 0
        assert order_requests[-1].is_final is True
        assert [request.path_index for request in order_requests] == list(range(len(order_requests)))
    assert {request.joint_order for request in collision.requests} == {
        (0, 1, 2, 3, 4, 5),
        (5, 4, 3, 2, 1, 0),
    }

    assert artifact["target_binding"] == {
        "target_id": "button-1",
        "detection_id": "det-1",
        "source_stamp_ns": NOW_NS - 100_000_000,
        "source_frame_id": "camera_optical_frame",
        "m0_frozen_artifact_sha256": artifact["release_bindings"]["m0_frozen"]["artifact_sha256"],
        "calibration_release_id": "r6-synthetic-release",
        "r6_bundle_artifact_sha256": artifact["release_bindings"]["r6_bundle"]["artifact_sha256"],
    }
    assert geometry["A_base_m"] == pytest.approx([0.0, 0.0, 0.5])
    assert geometry["N_out_base"] == pytest.approx([0.0, 0.0, 1.0])
    assert geometry["P_pre_base_m"] == pytest.approx([0.0, 0.0, 0.55])

    unsigned = dict(artifact)
    digest = unsigned.pop("canonical_payload_sha256")
    assert digest == sha256_canonical(unsigned)
    assert digest == canonical_route_artifact_hash(artifact)
    validate_outbound_route_artifact(artifact)
    assert artifact["route_id"] == f"precontact-outbound-{artifact['route_hash'][:16]}"


def test_joint_order_uses_highest_clearance_and_stable_lexicographic_tie_break():
    class OrderSensitiveCollision(ExactCollision):
        def __call__(self, request):
            result = dict(super().__call__(request))
            result["minimum_clearance_m"] = (
                0.040 if request.joint_order == (5, 4, 3, 2, 1, 0) else 0.030
            )
            return result

    collision = OrderSensitiveCollision()
    artifact = _build(collision=collision)
    command = artifact["clearance_evidence"]["command_contract"]
    assert command["joint_order"] == [5, 4, 3, 2, 1, 0]
    assert artifact["clearance_evidence"]["minimum_clearance_m"] == 0.040
    assert len(collision.requests) == 2


def test_adapter_consumable_artifact_rejects_non_atomic_or_unbound_command_contract():
    artifact = _build()
    command = artifact["clearance_evidence"]["command_contract"]
    step = command["atomic_steps"][0]
    other_joint = (step["joint_index"] + 1) % 6
    step["post_pose_rad"][other_joint] += 0.01
    artifact["quantization"]["emitted_setpoints_rad"][0][other_joint] += 0.01
    artifact["clearance_evidence"]["quantized_path_setpoints_rad"][1][other_joint] += 0.01
    _reseal_command_aliases(artifact)
    with pytest.raises(RouteBlocked, match="exactly one joint"):
        validate_outbound_route_artifact(artifact)

    artifact = _build()
    command = artifact["clearance_evidence"]["command_contract"]
    command["joint_command_timeout_s"][0] = 1.1
    for atomic_step in command["atomic_steps"]:
        if atomic_step["joint_index"] == 0:
            atomic_step["timeout_s"] = 1.1
    _reseal_command_aliases(artifact)
    with pytest.raises(RouteBlocked, match="route hash does not bind"):
        validate_outbound_route_artifact(artifact)


@pytest.mark.parametrize("release_name", ["tool_release", "r6_bundle", "standby_release"])
def test_diagnostic_fn_r6_and_standby_releases_are_never_adapter_consumable(release_name):
    manifest = _released_manifest()
    _diagnostic(_replace_release(manifest, release_name, status="diagnostic"), release_name)


def test_actual_style_relative_r6_and_diagnostic_fn_are_blocked():
    manifest = _released_manifest()
    r6 = copy.deepcopy(manifest["releases"]["r6_bundle"]["payload"])
    r6["relative"] = True
    _diagnostic(_replace_release(manifest, "r6_bundle", payload=r6), "relative")
    _diagnostic(_replace_release(manifest, "tool_release", status="diagnostic"), "tool_release")


def test_checked_in_r6_fn_and_standby_are_rejected_before_any_route_exists():
    """Bind the current diagnostic-only project evidence to a fail-closed outcome."""

    r6 = json.loads(CURRENT_R6.read_text())
    fn = json.loads(CURRENT_FN.read_text())
    standby_text = CURRENT_STANDBY.read_text()
    assert r6["status"] == "b_section_accepted_relative_unpublished"
    assert r6["execution_ready"] is False
    assert fn["status"] == "diagnostic_overlay_default"
    assert fn["candidate_transform"]["transform_complete"] is False
    assert "status: recorded_candidate" in standby_text

    manifest = _released_manifest()
    _diagnostic(_replace_release(manifest, "r6_bundle", status=r6["status"]), "r6_bundle")
    _diagnostic(_replace_release(manifest, "tool_release", status=fn["status"]), "tool_release")
    _diagnostic(_replace_release(manifest, "standby_release", status="recorded_candidate"), "standby_release")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda p: p.update(frame_id="base_link"), "M0 frame"),
        (lambda p: p.update(expires_at_ns=NOW_NS - 1), "expired"),
        (lambda p: p.update(ttl_ms=1001), "TTL"),
    ],
)
def test_wrong_m0_frame_stamp_or_ttl_fails_closed(mutate, reason):
    manifest = _released_manifest()
    payload = copy.deepcopy(manifest["releases"]["m0_frozen"]["payload"])
    mutate(payload)
    _diagnostic(_replace_release(manifest, "m0_frozen", payload=payload), reason)


def test_m0_detection_id_is_required_and_artifact_target_binding_is_tamper_evident():
    manifest = _released_manifest()
    payload = copy.deepcopy(manifest["releases"]["m0_frozen"]["payload"])
    payload.pop("detection_id")
    _diagnostic(_replace_release(manifest, "m0_frozen", payload=payload), "detection_id")

    artifact = _build()
    tampered = copy.deepcopy(artifact)
    tampered["target_binding"]["detection_id"] = "det-other"
    with pytest.raises(RouteBlocked, match="canonical payload hash mismatch"):
        validate_outbound_route_artifact(tampered)
    tampered["canonical_payload_sha256"] = canonical_route_artifact_hash(tampered)
    with pytest.raises(RouteBlocked, match="route hash does not bind"):
        validate_outbound_route_artifact(tampered)
    tampered = copy.deepcopy(artifact)
    tampered["geometry"]["P_pre_base_m"][2] = 0.551
    with pytest.raises(RouteBlocked, match="canonical payload hash mismatch"):
        validate_outbound_route_artifact(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_id", "button-other"),
        ("source_stamp_ns", NOW_NS - 200_000_000),
    ],
)
def test_m0_target_and_stamp_must_bind_the_normal_decision(field, value):
    manifest = _released_manifest()
    payload = copy.deepcopy(manifest["releases"]["m0_frozen"]["payload"])
    payload[field] = value
    _diagnostic(
        _replace_release(manifest, "m0_frozen", payload=payload),
        "normal decision does not bind",
    )


def test_tampered_release_hash_and_manifest_hash_fail_closed():
    manifest = _released_manifest()
    manifest["releases"]["m0_frozen"]["payload"]["point_camera_m"] = [0.0, 0.0, 0.501]
    _diagnostic(seal_manifest(manifest), "payload hash mismatch")

    manifest = _released_manifest()
    manifest["route_policy"]["precontact_distance_m"] = 0.051
    _diagnostic(manifest, "manifest hash mismatch")


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda p: p.update(status="inward_verified"), "ambiguous, inward"),
        (lambda p: p.update(dot_n_out_p_minus_a=-0.1), "does not point"),
        (lambda p: p.update(input_normal_was_flipped=True), "never silently flips"),
    ],
)
def test_invalid_or_inward_normal_never_silently_flips(mutate, reason):
    manifest = _released_manifest()
    payload = copy.deepcopy(manifest["releases"]["normal_decision"]["payload"])
    mutate(payload)
    _diagnostic(_replace_release(manifest, "normal_decision", payload=payload), reason)


def test_quantized_delta_requires_a_full_encoder_step_of_margin():
    manifest = _released_manifest()
    driver = copy.deepcopy(manifest["releases"]["driver_profile_release"]["payload"])
    driver["joint_max_delta_rad"] = [0.015] * 6
    _diagnostic(
        _replace_release(manifest, "driver_profile_release", payload=driver),
        "leaves no encoder-step safety margin",
    )


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("joint_max_speed_rad_s", "joint_max_speed_rad_s"),
        ("joint_max_acceleration_rad_s2", "joint_max_acceleration_rad_s2"),
        ("joint_command_timeout_s", "joint_command_timeout_s"),
        ("joint_order_allowlist", "joint_order_allowlist"),
    ],
)
def test_driver_release_requires_speed_acceleration_timeout_and_order_allowlist(field, reason):
    manifest = _released_manifest()
    driver = copy.deepcopy(manifest["releases"]["driver_profile_release"]["payload"])
    driver.pop(field)
    _diagnostic(
        _replace_release(manifest, "driver_profile_release", payload=driver),
        reason,
    )


def test_quantized_endpoint_not_the_raw_ik_endpoint_is_revalidated():
    ik = ExactIK(positions_rad=(0.131, -0.069, 0.041, 0.0, 0.019, -0.011))
    kinematics = ExactKinematics(ik)
    artifact = _build(ik=ik, kinematics=kinematics)
    assert ik.positions_rad != kinematics.request.joint_positions_rad
    assert kinematics.request.joint_positions_rad == pytest.approx(
        (0.13, -0.07, 0.04, 0.0, 0.02, -0.01)
    )
    assert artifact["quantized_endpoint_evidence"]["checked_positions_rad"] == pytest.approx(
        list(kinematics.request.joint_positions_rad)
    )


@pytest.mark.parametrize(
    ("collision", "reason"),
    [
        (ExactCollision(status="blocked"), "no allowlisted T101 joint order"),
        (ExactCollision(clearance_m=0.019), "no allowlisted T101 joint order"),
    ],
)
def test_collision_keepout_evidence_must_clear_all_quantized_waypoints(collision, reason):
    _diagnostic(_released_manifest(), reason, collision=collision)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("checked_waypoint_count", 0, "standby plus every atomic pose"),
        ("segments_checked", 0, "every atomic swept segment"),
        ("continuous_corridor_status", "pending", "no allowlisted T101 joint order"),
        ("swept_segments_status", "pending", "no allowlisted T101 joint order"),
    ],
)
def test_collision_evidence_requires_start_and_every_swept_segment(field, value, reason):
    class IncompleteCollision(ExactCollision):
        def __call__(self, request):
            result = dict(super().__call__(request))
            result[field] = value
            return result

    _diagnostic(_released_manifest(), reason, collision=IncompleteCollision())


def test_orientation_and_injected_planning_dependencies_are_required():
    class WrongOrientationIK(ExactIK):
        def __call__(self, request):
            result = dict(super().__call__(request))
            result["achieved_hand_rotation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
            return result

    with pytest.raises(RouteBlocked, match="no offline IK solver"):
        build_outbound_route(
            _released_manifest(),
            kinematics_checker=ExactKinematics(ExactIK()),
            collision_checker=ExactCollision(),
            now_ns=NOW_NS,
        )
    ik = ExactIK()
    with pytest.raises(RouteBlocked, match="no offline quantized-path kinematics checker"):
        build_outbound_route(
            _released_manifest(), ik_solver=ik, collision_checker=ExactCollision(), now_ns=NOW_NS
        )
    with pytest.raises(RouteBlocked, match="no offline collision checker"):
        ik = ExactIK()
        build_outbound_route(
            _released_manifest(),
            ik_solver=ik,
            kinematics_checker=ExactKinematics(ik),
            now_ns=NOW_NS,
        )
    _diagnostic(_released_manifest(), "contact axis", ik=WrongOrientationIK())
