from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODULE_DIR))

from coordinate_contract import (  # noqa: E402
    ContractError,
    load_contract,
    quaternion_to_matrix,
    validate_contract,
    validate_target,
)


CONTRACT_PATH = PROJECT_ROOT / "config" / "coordinate_contract_v1.yaml"


def pending_contract():
    return load_contract(CONTRACT_PATH)


def ready_contract():
    contract = copy.deepcopy(pending_contract())
    calibration_id = "test-calibration-bundle-v1"
    contract["calibration"]["execution_enabled"] = True
    contract["calibration"]["calibration_id"] = calibration_id
    for transform in contract["transforms"]:
        if transform["id"] in contract["calibration"]["required_static_transform_ids"]:
            transform["status"] = "calibrated"
            transform["calibration_id"] = calibration_id
            transform["translation_m"] = [0.1, -0.2, 0.3]
            transform["rotation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    for key in contract["calibration"]["required_component_status"]:
        contract["calibration"]["required_component_status"][key] = "calibrated"
    hp60c = contract["sensor_adapters"]["hp60c"]
    hp60c["optical_axis_validation_status"] = "calibrated"
    hp60c["depth_scale_status"] = "calibrated"
    hp60c["depth_scale_m_per_unit"] = 0.001
    assert validate_contract(contract) is True
    return contract


def valid_target(stamp_ns=1_000_000_000):
    return {
        "schema_version": 1,
        "coordinate_contract_version": "1.0.0",
        "target_id": "target-1",
        "detection_id": "detection-1",
        "pose": {
            "header": {
                "frame_id": "camera_optical_frame",
                "stamp_ns": stamp_ns,
            },
            "position_m": [0.1, -0.03, 0.6],
            "orientation_xyzw": [0.0, 2**-0.5, 0.0, 2**-0.5],
        },
        "panel_outward_normal": {
            "frame_id": "camera_optical_frame",
            "xyz": [0.0, 0.0, -1.0],
        },
        "source_timestamp_ns": stamp_ns,
        "confidence": 0.95,
        "calibration_id": "test-calibration-bundle-v1",
        "ttl_ms": 1000,
    }


def transform(contract, transform_id):
    return next(item for item in contract["transforms"] if item["id"] == transform_id)


def test_committed_contract_is_valid_but_not_execution_ready():
    assert validate_contract(pending_contract()) is False


def test_gemini335_adapter_is_fail_closed_and_canonical():
    contract = pending_contract()
    gemini = contract["sensor_adapters"]["gemini335"]
    assert gemini["observed_raw_frame_id"] == "gemini335_color_optical_frame"
    assert gemini["canonical_frame_id"] == "camera_optical_frame"
    assert gemini["vendor_tf_allowed_for_execution"] is False
    assert gemini["optical_axis_validation_status"] == "pending"
    assert gemini["depth_scale_status"] == "pending"
    assert gemini["depth_scale_m_per_unit"] is None


def test_gemini335_pending_depth_cannot_use_numeric_placeholder():
    contract = pending_contract()
    contract["sensor_adapters"]["gemini335"]["depth_scale_m_per_unit"] = 0.001
    with pytest.raises(ContractError, match="pending Gemini 335 depth scale"):
        validate_contract(contract)


def test_gemini335_vendor_tf_is_rejected_for_execution():
    contract = pending_contract()
    contract["sensor_adapters"]["gemini335"]["vendor_tf_allowed_for_execution"] = True
    with pytest.raises(ContractError, match="vendor Gemini 335 TF"):
        validate_contract(contract)


def test_exact_rep103_axes_are_enforced():
    contract = pending_contract()
    contract["axes"]["body"]["x"] = "right"
    with pytest.raises(ContractError, match="REP-103"):
        validate_contract(contract)


def test_frame_names_are_frozen():
    contract = pending_contract()
    contract["frames"]["camera_link_wrong"] = contract["frames"].pop("camera_link")
    with pytest.raises(ContractError, match="frame set"):
        validate_contract(contract)


def test_transform_parent_child_direction_is_enforced():
    contract = pending_contract()
    edge = transform(contract, "cart_base_to_camera_link")
    edge["parent"], edge["child"] = edge["child"], edge["parent"]
    with pytest.raises(ContractError, match="direction/mode"):
        validate_contract(contract)


def test_edge_change_that_would_create_multiple_parents_is_rejected():
    contract = pending_contract()
    transform(contract, "map_to_odom")["child"] = "cart_base"
    with pytest.raises(ContractError, match="direction/mode"):
        validate_contract(contract)


def test_external_units_are_frozen():
    contract = pending_contract()
    contract["units"]["external_length"] = "mm"
    with pytest.raises(ContractError, match="external_length"):
        validate_contract(contract)


def test_body_to_optical_quaternion_maps_axes():
    matrix = quaternion_to_matrix([-0.5, 0.5, -0.5, 0.5])
    assert matrix == ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))


def test_illegal_quaternion_is_rejected():
    contract = pending_contract()
    transform(contract, "camera_link_to_camera_optical_frame")["rotation_xyzw"] = [0, 0, 0, 2]
    with pytest.raises(ContractError, match="quaternion"):
        validate_contract(contract)


def test_pending_calibration_cannot_hide_identity_placeholder():
    contract = pending_contract()
    transform(contract, "cart_base_to_camera_link")["translation_m"] = [0, 0, 0]
    with pytest.raises(ContractError, match="placeholder"):
        validate_contract(contract)


def test_execution_flag_fails_when_bundle_is_incomplete():
    contract = pending_contract()
    contract["calibration"]["execution_enabled"] = True
    contract["calibration"]["calibration_id"] = "incomplete"
    with pytest.raises(ContractError, match="incomplete"):
        validate_contract(contract)


def test_execution_ready_requires_matching_calibration_ids():
    contract = ready_contract()
    transform(contract, "hand_tcp_to_press_contact")["calibration_id"] = "stale"
    with pytest.raises(ContractError, match="incomplete or mismatched"):
        validate_contract(contract)


def test_vendor_camera_tf_cannot_be_enabled_for_execution():
    contract = pending_contract()
    contract["sensor_adapters"]["hp60c"]["vendor_tf_allowed_for_execution"] = True
    with pytest.raises(ContractError, match="vendor HP60C TF"):
        validate_contract(contract)


def test_target_required_field_set_is_frozen():
    contract = pending_contract()
    contract["target_interface"]["required_fields"].remove("calibration_id")
    with pytest.raises(ContractError, match="required_fields"):
        validate_contract(contract)


def test_valid_target_passes_with_complete_bundle():
    validate_target(valid_target(), ready_contract(), now_ns=1_500_000_000)


def test_target_rejects_camera_link_for_optical_coordinates():
    target = valid_target()
    target["pose"]["header"]["frame_id"] = "camera_link"
    target["panel_outward_normal"]["frame_id"] = "camera_link"
    with pytest.raises(ContractError, match="not allowed"):
        validate_target(target, ready_contract(), now_ns=1_500_000_000)


def test_target_requires_calibration_id():
    target = valid_target()
    target["calibration_id"] = ""
    with pytest.raises(ContractError, match="calibration_id"):
        validate_target(target, ready_contract(), now_ns=1_500_000_000)


def test_target_rejects_pending_contract_even_with_plausible_numbers():
    with pytest.raises(ContractError, match="not execution-ready"):
        validate_target(valid_target(), pending_contract(), now_ns=1_500_000_000)


def test_target_requires_unit_panel_normal():
    target = valid_target()
    target["panel_outward_normal"]["xyz"] = [0.0, 0.0, -2.0]
    with pytest.raises(ContractError, match="unit length"):
        validate_target(target, ready_contract(), now_ns=1_500_000_000)


def test_target_pose_x_axis_must_match_outward_normal():
    target = valid_target()
    target["pose"]["orientation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ContractError, match=r"pose \+X axis"):
        validate_target(target, ready_contract(), now_ns=1_500_000_000)


def test_target_requires_one_observation_timestamp():
    target = valid_target()
    target["source_timestamp_ns"] += 1
    with pytest.raises(ContractError, match="source_timestamp_ns"):
        validate_target(target, ready_contract(), now_ns=1_500_000_000)


def test_stale_target_is_rejected():
    with pytest.raises(ContractError, match="stale"):
        validate_target(valid_target(), ready_contract(), now_ns=2_100_000_001)


def test_mixed_target_units_are_rejected():
    target = valid_target()
    target["debug_offset_mm"] = 3.0
    with pytest.raises(ContractError, match="forbidden external unit"):
        validate_target(target, ready_contract(), now_ns=1_500_000_000)
