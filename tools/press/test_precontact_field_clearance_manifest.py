import copy
import json
from pathlib import Path

import pytest

from precontact_field_clearance_manifest import (
    FieldClearanceBlocked,
    confirmation_sha256,
    manifest_sha256,
    validate_field_clearance_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "config/press/diagnostic/button1_direct_precontact_field_clearance_20260811_v1.json"
)


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def reseal(value):
    source = value.get("operator_confirmation_source")
    if isinstance(source, dict):
        source["record_sha256"] = confirmation_sha256(source)
    value["canonical_manifest_sha256"] = manifest_sha256(value)
    return value


def test_checked_in_manifest_is_field_pass_but_prepare_blocked():
    report = validate_field_clearance_manifest(load_manifest(), repository_root=ROOT)
    assert report["valid"] is True
    assert report["field_observed_verdict"] == "PASS_FIELD_OBSERVATION_ONLY"
    assert report["prepare_result"] == "BLOCKED_DIAGNOSTIC_CLEARANCE_ONLY"
    assert report["execution_ready"] is False
    assert report["motion_authorized"] is False
    assert report["adapter_consumable"] is False
    assert report["deployment_frozen"] is True
    assert "NO_COLLISION_RELEASE" in report["blocked_reasons"]


def test_missing_source_fails_closed_even_after_manifest_reseal():
    value = load_manifest()
    value.pop("operator_confirmation_source")
    value["canonical_manifest_sha256"] = manifest_sha256(value)
    with pytest.raises(FieldClearanceBlocked, match="unknown or missing fields"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_operator_source_tamper_fails_without_source_reseal():
    value = load_manifest()
    value["operator_confirmation_source"]["facts"][2] = "collision-free"
    value["canonical_manifest_sha256"] = manifest_sha256(value)
    with pytest.raises(FieldClearanceBlocked, match="facts are incomplete"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_missing_provenance_hash_fails_closed():
    value = load_manifest()
    value["provenance"][0].pop("sha256")
    reseal(value)
    with pytest.raises(FieldClearanceBlocked, match="unknown or missing fields"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_tip_diameter_cannot_be_resealed_as_unconfirmed_geometry():
    value = load_manifest()
    value["tool_observation"]["front_tip_nominal_diameter_m"] = 0.02
    reseal(value)
    with pytest.raises(FieldClearanceBlocked, match="0.030 m"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


@pytest.mark.parametrize(
    "section,field,bad_value",
    [
        ("front_corridor_observation", "coordinate_bound_geometry", True),
        ("rear_observation", "continuous_swept_volume_checked", True),
        ("rear_observation", "rear_keepout_release_status", "released"),
    ],
)
def test_observation_cannot_be_resealed_as_collision_release(section, field, bad_value):
    value = load_manifest()
    value[section][field] = bad_value
    reseal(value)
    with pytest.raises(FieldClearanceBlocked):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_prepare_cannot_issue_ticket_after_reseal():
    value = load_manifest()
    value["prepare_disposition"]["ticket_may_be_issued"] = True
    reseal(value)
    with pytest.raises(FieldClearanceBlocked, match="cannot issue a ticket"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_missing_block_reason_fails_closed_after_reseal():
    value = load_manifest()
    value["prepare_disposition"]["blocked_reasons"].remove("NO_COLLISION_RELEASE")
    reseal(value)
    with pytest.raises(FieldClearanceBlocked, match="blocked reasons"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_coordinate_or_release_field_injection_fails_after_full_reseal():
    value = load_manifest()
    value["rear_observation"]["camera_bracket_pose_base_m"] = [0.0, 0.0, 0.0]
    value["collision_release_id"] = "invented-release"
    reseal(value)
    with pytest.raises(FieldClearanceBlocked, match="unknown or missing fields"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_resealed_operator_fact_substitution_fails_closed():
    value = load_manifest()
    value["operator_confirmation_source"]["facts"][2] = "collision-free"
    reseal(value)
    with pytest.raises(FieldClearanceBlocked, match="facts are incomplete"):
        validate_field_clearance_manifest(value, repository_root=ROOT)


def test_module_has_no_control_or_deployment_imports():
    source = (ROOT / "tools/press/precontact_field_clearance_manifest.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "import rclpy",
        "ActionClient",
        "create_publisher",
        "create_client",
        "serial.Serial",
        "roarm_safe_driver",
        "subprocess",
    ):
        assert forbidden not in source
