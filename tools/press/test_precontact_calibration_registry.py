import copy
import json
from pathlib import Path

import pytest

from precontact_calibration_registry import (
    RegistryError,
    embedded_record_sha256,
    registry_payload_sha256,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config/calibration/precontact_candidate_registry_v1.json"


def load_registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def reseal(registry):
    confirmation = registry.get("operator_confirmation_source")
    if isinstance(confirmation, dict):
        confirmation["record_sha256"] = embedded_record_sha256(confirmation)
    registry["registry_payload_sha256"] = registry_payload_sha256(registry)
    return registry


def test_checked_in_registry_validates_and_remains_inert():
    report = validate_registry(load_registry(), repository_root=ROOT)
    assert report["valid"] is True
    assert report["release_status"] == "not_released"
    assert report["execution_ready"] is False
    assert report["motion_authorized"] is False
    assert report["deployment_frozen"] is True
    assert report["validated_artifact_count"] == 5


def test_missing_provenance_hash_fails_closed_even_after_registry_reseal():
    value = load_registry()
    value["provenance"][0].pop("sha256")
    reseal(value)
    with pytest.raises(RegistryError, match="lowercase SHA-256"):
        validate_registry(value, repository_root=ROOT)


def test_wrong_normal_policy_fails_closed_even_after_registry_reseal():
    value = load_registry()
    value["normal_policy"]["name"] = "camera_facing_guess"
    reseal(value)
    with pytest.raises(RegistryError, match="button_panel_outward"):
        validate_registry(value, repository_root=ROOT)


def test_missing_operator_confirmation_source_fails_closed():
    value = load_registry()
    value.pop("operator_confirmation_source")
    value["registry_payload_sha256"] = registry_payload_sha256(value)
    with pytest.raises(RegistryError, match="operator_confirmation_source"):
        validate_registry(value, repository_root=ROOT)


def test_operator_confirmation_hash_tamper_fails_closed():
    value = load_registry()
    value["operator_confirmation_source"]["summary"] = "motion authorized"
    value["registry_payload_sha256"] = registry_payload_sha256(value)
    with pytest.raises(RegistryError, match="confirmation source hash mismatch"):
        validate_registry(value, repository_root=ROOT)


def test_provenance_source_drift_fails_closed(tmp_path):
    value = load_registry()
    source = ROOT / value["provenance"][0]["path"]
    altered = tmp_path / "altered-r6.json"
    altered.write_bytes(source.read_bytes() + b"\n")
    value["provenance"][0]["path"] = str(altered.relative_to(tmp_path))
    reseal(value)
    with pytest.raises(RegistryError, match="provenance hash mismatch"):
        validate_registry(value, repository_root=tmp_path)


@pytest.mark.parametrize(
    "field,bad_value",
    [("execution_ready", True), ("motion_authorized", True), ("deployment_frozen", False)],
)
def test_safety_boundary_cannot_be_resealed(field, bad_value):
    value = load_registry()
    value["safety_flags"][field] = bad_value
    reseal(value)
    with pytest.raises(RegistryError, match=field):
        validate_registry(value, repository_root=ROOT)


def test_registry_module_has_no_control_runtime_imports():
    source = (ROOT / "tools/press/precontact_calibration_registry.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "import rclpy",
        "ActionClient",
        "create_publisher",
        "create_client",
        "serial.Serial",
        "roarm_safe_driver",
    ):
        assert forbidden not in source
