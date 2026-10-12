from __future__ import annotations

import pytest

from press.static_installation_measurement_bundle import (
    StaticBundleBlocked,
    seal_bundle,
    validate_bundle,
)


def _sha(char: str) -> str:
    return char * 64


def _source(kind: str = "operator_measurement", char: str = "a", *, synthetic: bool = False) -> dict:
    return {
        "kind": kind,
        "synthetic": synthetic,
        "artifact_sha256": _sha(char),
        "measurement_method": "ruler_and_scaled_photo_fixture",
    }


def _capsule() -> dict:
    return {
        "primitive": "capsule",
        "endpoint_a_hand_tcp_m": [0.0, 0.0, 0.0],
        "endpoint_b_hand_tcp_m": [0.0, 0.0, 0.12],
    }


def _part(kind: str, char: str) -> dict:
    return {
        "present": True,
        "part_kind": kind,
        "body_id": f"press_{kind}",
        "evidence_class": "CURRENT_BOUNDED_CONSERVATIVE",
        "source": _source(char=char),
        "length_units": "m",
        "geometry": _capsule(),
        "conservative_radius_m": 0.01,
        "uncertainty_m": 0.002,
        "conservative_inflation_m": 0.001,
        "invalidation_conditions": ["physical remount or shape change"],
        "notes": "fixture",
    }


def _bundle(**changes) -> dict:
    value = {
        "schema": "static_installation_measurement_bundle_v1",
        "mode": "OFFLINE_STATIC_MEASUREMENT_EVIDENCE_ONLY",
        "bundle_id": "fixture-bundle-001",
        "created_at": "2026-08-14T00:00:00+08:00",
        "installation": {
            "installation_id": "fixture-install",
            "tool_id": "fixture-stirring-rod",
            "identity_proven": True,
            "identity_source_sha256": _sha("1"),
            "invalidation_conditions": ["physical re-mount or tool replacement"],
        },
        "evidence_inventory": [
            {
                "id": "fixture-source",
                "path": "temp/fixture.json",
                "sha256": _sha("2"),
                "classification": "CURRENT_VERIFIED",
                "role": "fixture source evidence",
            }
        ],
        "raw_operator_evidence": {
            "received": True,
            "source_manifest_path": "config/press/diagnostic/fixture.json",
            "source_manifest_sha256": _sha("3"),
            "statements": [
                {
                    "fact_id": "fixture-statement",
                    "statement": "fixture operator measurement",
                    "status": "FIELD_OBSERVED_ONLY",
                }
            ],
            "photos_with_scale": [],
        },
        "hand_tcp_to_tool": {
            "present": True,
            "evidence_class": "CURRENT_BOUNDED_CONSERVATIVE",
            "source": _source(char="4"),
            "length_units": "m",
            "angle_units": "rad",
            "translation_m": [0.0, 0.0, 0.12],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "translation_uncertainty_m": [0.002, 0.002, 0.002],
            "orientation_uncertainty_rad": 0.02,
            "coaxial_with_hand_tcp_axis": True,
            "invalidation_conditions": ["tool re-mount or axis re-alignment"],
            "notes": "fixture",
        },
        "tool_parts": {
            "shaft": _part("shaft", "5"),
            "soft_tip": _part("soft_tip", "6"),
            "adapters": {
                "present": False,
                "count": 0,
                "evidence_class": "CURRENT_VERIFIED",
                "source": _source(char="7"),
                "length_units": "m",
                "geometry": None,
                "uncertainty_m": None,
                "conservative_inflation_m": None,
                "invalidation_conditions": ["adapter addition or change"],
                "notes": "no adapter extends beyond shaft/tip bounds",
            },
            "cable": {
                "present": True,
                "fixed_configuration": True,
                "free_length_m": 0.35,
                "bow_max_excursion_m": 0.03,
                "permitted_bow_direction": "rearward_away_from_panel",
                "evidence_class": "CURRENT_BOUNDED_CONSERVATIVE",
                "source": _source(char="8"),
                "length_units": "m",
                "geometry": {
                    "primitive": "capsule",
                    "endpoint_a_hand_tcp_m": [0.0, 0.0, 0.1],
                    "endpoint_b_hand_tcp_m": [0.0, 0.0, 0.35],
                    "route_description": "fixed anchors with bounded rearward bow",
                },
                "uncertainty_m": 0.005,
                "conservative_inflation_m": 0.002,
                "invalidation_conditions": ["cable reroute or anchor change"],
                "notes": "fixture",
            },
        },
        "rear_obstacles": {
            "camera_bracket": {
                "present": True,
                "obstacle_id": "rear-camera-bracket",
                "primitive": "box",
                "frame_id": "roarm_base_link",
                "evidence_class": "CURRENT_BOUNDED_CONSERVATIVE",
                "source": _source(char="9"),
                "length_units": "m",
                "angle_units": "rad",
                "pose_base": {
                    "translation_m": [0.0, -0.2, 0.3],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "size_m": [0.05, 0.05, 0.4],
                "translation_uncertainty_m": [0.01, 0.01, 0.01],
                "orientation_uncertainty_rad": 0.05,
                "invalidation_conditions": ["bracket movement or camera re-mount"],
                "notes": "fixture",
            },
            "other_rear_structure_checked": True,
            "other_rear_obstacles": [],
        },
        "robot_model": {
            "urdf_or_xacro_path": "RoArm-M3/.../roarm_m3.xacro",
            "urdf_or_xacro_sha256": _sha("b"),
            "collision_mesh_hashes": [_sha("c"), _sha("d")],
            "body_ids": ["base_link", "link1", "link2", "link3", "link4", "link5", "gripper_link", "tool_shaft", "tool_soft_tip", "tool_cable"],
            "identity_accepted": True,
            "model_status": "released_static",
            "self_collision_policy": {
                "source_path": "config/press/fixture_policy.json",
                "source_sha256": _sha("e"),
                "reviewed": True,
            },
        },
        "margin_policy": {
            "status": "REVIEWED",
            "source_path": "config/press/fixture_margin.json",
            "source_sha256": _sha("f"),
            "value_m": 0.01,
            "non_authoritative_candidate_m": None,
            "notes": "fixture",
        },
        "layer_d_boundary": {
            "statement": "Layer D values are never sealed into Layer S.",
            "session_fresh_inputs": ["A_base_m", "N_out_base", "q0"],
            "sealed_values_present": False,
        },
        "authority": {
            "execution_ready": False,
            "motion_authorized": False,
            "contact_authorized": False,
            "adapter_consumable": False,
            "deployment_authorized": False,
        },
        "status": {
            "measurement_bundle_complete": True,
            "release_status": "blocked_static",
            "review_status": "pending_independent_review",
            "raw_operator_evidence_received": True,
            "blockers": [],
        },
    }
    value.update(changes)
    return seal_bundle(value)


def test_complete_static_fixture_never_releases_or_authorizes_motion():
    receipt = validate_bundle(_bundle())
    assert receipt["result"] == "measurement_complete"
    assert receipt["measurement_bundle_complete"] is True
    assert receipt["release_status"] == "blocked_static"
    assert receipt["execution_ready"] is False
    assert receipt["motion_authorized"] is False
    assert receipt["contact_authorized"] is False
    assert receipt["adapter_consumable"] is False
    assert receipt["deployment_authorized"] is False


def test_missing_translation_or_orientation_uncertainty_blocks_completeness():
    value = _bundle()
    value["hand_tcp_to_tool"]["translation_uncertainty_m"] = None
    receipt = validate_bundle(seal_bundle(value))
    assert receipt["result"] == "blocked"
    assert "HAND_TCP_TO_TOOL_TRANSLATION_UNCERTAINTY_MISSING" in receipt["missing_reasons"]

    value = _bundle()
    value["hand_tcp_to_tool"]["orientation_uncertainty_rad"] = None
    receipt = validate_bundle(seal_bundle(value))
    assert "HAND_TCP_TO_TOOL_ORIENTATION_UNCERTAINTY_MISSING" in receipt["missing_reasons"]


def test_missing_shaft_tip_adapter_or_cable_blocks_completeness():
    value = _bundle()
    value["tool_parts"]["shaft"]["present"] = False
    assert "SHAFT_GEOMETRY_MISSING" in validate_bundle(seal_bundle(value))["missing_reasons"]

    value = _bundle()
    value["tool_parts"]["soft_tip"]["geometry"] = None
    assert "SOFT_TIP_GEOMETRY_MISSING" in validate_bundle(seal_bundle(value))["missing_reasons"]

    value = _bundle()
    value["tool_parts"]["adapters"]["present"] = True
    value["tool_parts"]["adapters"]["count"] = 1
    assert "ADAPTER_GEOMETRY_MISSING" in validate_bundle(seal_bundle(value))["missing_reasons"]

    value = _bundle()
    value["tool_parts"]["cable"]["present"] = False
    assert "CABLE_GEOMETRY_MISSING" in validate_bundle(seal_bundle(value))["missing_reasons"]


def test_unbounded_cable_routing_blocks_completeness():
    value = _bundle()
    value["tool_parts"]["cable"]["fixed_configuration"] = False
    receipt = validate_bundle(seal_bundle(value))
    assert "CABLE_ROUTING_UNBOUNDED" in receipt["missing_reasons"]

    value = _bundle()
    value["tool_parts"]["cable"]["bow_max_excursion_m"] = None
    receipt = validate_bundle(seal_bundle(value))
    assert "CABLE_ROUTING_UNBOUNDED" in receipt["missing_reasons"]


def test_rear_bracket_without_base_registration_blocks_completeness():
    value = _bundle()
    value["rear_obstacles"]["camera_bracket"]["frame_id"] = "camera_optical_frame"
    receipt = validate_bundle(seal_bundle(value))
    assert "REAR_CAMERA_BRACKET_NOT_BASE_FRAME" in receipt["missing_reasons"]

    value = _bundle()
    value["rear_obstacles"]["camera_bracket"]["translation_uncertainty_m"] = None
    receipt = validate_bundle(seal_bundle(value))
    assert "REAR_CAMERA_BRACKET_REGISTRATION_UNCERTAINTY_MISSING" in receipt["missing_reasons"]


def test_non_finite_wrong_unit_malformed_quaternion_and_incomplete_geometry_fail():
    with pytest.raises(StaticBundleBlocked):
        value = _bundle()
        value["hand_tcp_to_tool"]["translation_m"] = [0.0, 0.0, float("nan")]
        seal_bundle(value)

    value = _bundle()
    value["hand_tcp_to_tool"]["length_units"] = "mm"
    assert "WRONG_UNITS" in validate_bundle(seal_bundle(value))["missing_reasons"]

    value = _bundle()
    value["hand_tcp_to_tool"]["quaternion_xyzw"] = [1.0, 1.0, 0.0, 0.0]
    receipt = validate_bundle(seal_bundle(value))
    assert receipt["result"] == "blocked"
    assert "unit length" in receipt["missing_reasons"][0]

    value = _bundle()
    value["tool_parts"]["shaft"]["geometry"] = None
    assert "SHAFT_GEOMETRY_MISSING" in validate_bundle(seal_bundle(value))["missing_reasons"]


def test_source_evidence_or_installation_change_invalidates_canonical_hash():
    value = _bundle()
    value["installation"]["identity_source_sha256"] = _sha("9")
    receipt = validate_bundle(value)
    assert receipt["result"] == "blocked"
    assert "bundle hash mismatch" in receipt["missing_reasons"]

    value = _bundle()
    value["hand_tcp_to_tool"]["source"]["artifact_sha256"] = _sha("9")
    receipt = validate_bundle(value)
    assert "bundle hash mismatch" in receipt["missing_reasons"]


def test_synthetic_evidence_cannot_be_promoted():
    value = _bundle()
    value["hand_tcp_to_tool"]["source"]["synthetic"] = True
    receipt = validate_bundle(seal_bundle(value))
    assert receipt["result"] == "blocked"
    assert "SYNTHETIC_EVIDENCE_PROMOTED" in receipt["missing_reasons"]


def test_static_and_session_fresh_fields_cannot_be_silently_mixed():
    value = _bundle()
    value["layer_d_boundary"]["sealed_values_present"] = True
    receipt = validate_bundle(seal_bundle(value))
    assert "LAYER_D_STATIC_MIX" in receipt["missing_reasons"]


def test_release_status_and_authority_flags_remain_false():
    value = _bundle()
    value["status"]["release_status"] = "released_static"
    receipt = validate_bundle(seal_bundle(value))
    assert "RELEASE_STATUS_NOT_BLOCKED" in receipt["missing_reasons"]
    assert receipt["release_status"] == "blocked_static"

    value = _bundle()
    value["authority"]["motion_authorized"] = True
    receipt = validate_bundle(seal_bundle(value))
    assert "AUTHORITY_FLAG_RAISED" in receipt["missing_reasons"]


def test_complete_fixture_with_wrong_completeness_flag_fails_closed():
    value = _bundle()
    value["status"]["measurement_bundle_complete"] = False
    receipt = validate_bundle(seal_bundle(value))
    assert receipt["result"] == "blocked"
    assert "COMPLETENESS_FLAG_INCONSISTENT" in receipt["missing_reasons"]
