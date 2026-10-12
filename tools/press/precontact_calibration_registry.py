#!/usr/bin/env python3
"""Fail-closed validation for the planning-only pre-contact candidate registry.

This module reads local evidence only.  It has no ROS imports, publishers,
services, action clients, serial access, or motion-authority output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "precontact_calibration_candidate_registry_v1"
EXPECTED_ARTIFACT_ROLES = {
    "section_b_relative_bundle",
    "fn_active_pointer",
    "fn_diagnostic_candidate",
    "normal_policy_implementation",
    "coordinate_contract",
}
HEX = frozenset("0123456789abcdef")


class RegistryError(ValueError):
    """The registry or one of its bound local artifacts failed validation."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def registry_payload_sha256(registry: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(registry))
    payload.pop("registry_payload_sha256", None)
    return canonical_sha256(payload)


def embedded_record_sha256(record: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(record))
    payload.pop("record_sha256", None)
    return canonical_sha256(payload)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{where} must be an object")
    return value


def _sha(value: Any, where: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in HEX for character in digest):
        raise RegistryError(f"{where} must be a lowercase SHA-256")
    return digest


def _local_path(repository_root: Path, value: Any, where: str) -> Path:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute():
        raise RegistryError(f"{where} must be a repository-relative path")
    root = repository_root.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise RegistryError(f"{where} escapes the repository")
    if not resolved.is_file():
        raise RegistryError(f"{where} does not exist: {relative}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"{where} is not valid JSON: {exc}") from exc
    return _mapping(value, where)


def validate_registry(
    registry: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    """Validate registry semantics and every provenance hash, without writes."""

    registry = _mapping(registry, "registry")
    root = Path(repository_root)
    if registry.get("schema") != SCHEMA:
        raise RegistryError("registry schema is invalid")
    if registry.get("calibration_status") != "candidate_release_ready":
        raise RegistryError("calibration_status must remain candidate_release_ready")
    if registry.get("release_status") != "not_released":
        raise RegistryError("candidate must not claim released status")
    if registry.get("review_status") != "planning_domain_reviewed_not_release_reviewed":
        raise RegistryError("review_status is missing or overclaims release review")

    flags = _mapping(registry.get("safety_flags"), "safety_flags")
    expected_flags = {
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "press_authorized": False,
        "publish_authorized": False,
        "deployment_frozen": True,
    }
    for field, expected in expected_flags.items():
        if flags.get(field) is not expected:
            raise RegistryError(f"safety_flags.{field} must be {expected}")

    reference = _mapping(registry.get("precontact_reference"), "precontact_reference")
    if reference.get("name") != "fingertip/Fn":
        raise RegistryError("precontact_reference must be fingertip/Fn")
    if reference.get("geometry_authority") is not False:
        raise RegistryError("diagnostic Fn cannot grant geometry authority")
    if reference.get("status") != "diagnostic_only":
        raise RegistryError("fingertip/Fn must remain diagnostic_only")

    normal = _mapping(registry.get("normal_policy"), "normal_policy")
    if normal.get("name") != "button_panel_outward":
        raise RegistryError("normal_policy must be button_panel_outward")
    if normal.get("sign_handling") != "explicit_validate_or_reject_no_silent_flip":
        raise RegistryError("normal sign policy must reject silent flips")
    if normal.get("precontact_equation") != "P_pre = A + 0.050 * N_out":
        raise RegistryError("normal policy has an unexpected pre-contact equation")

    environment = _mapping(registry.get("environment_evidence"), "environment_evidence")
    if environment.get("front_corridor") != "field_observed_only":
        raise RegistryError("front corridor must remain field_observed_only")
    if environment.get("rear_keepout") != "offline_synthetic_pending":
        raise RegistryError("rear keepout must remain offline_synthetic_pending")

    confirmation = _mapping(
        registry.get("operator_confirmation_source"), "operator_confirmation_source"
    )
    if confirmation.get("source_kind") != "conversation_instruction":
        raise RegistryError("operator confirmation source is missing")
    if confirmation.get("scope") != "planning_domain_registry_only":
        raise RegistryError("operator confirmation has an invalid scope")
    for field in ("release_authorized", "motion_authorized", "deployment_authorized"):
        if confirmation.get(field) is not False:
            raise RegistryError(f"operator confirmation must keep {field}=false")
    confirmation_hash = _sha(
        confirmation.get("record_sha256"), "operator_confirmation_source.record_sha256"
    )
    if embedded_record_sha256(confirmation) != confirmation_hash:
        raise RegistryError("operator confirmation source hash mismatch")

    provenance = registry.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise RegistryError("provenance must be a non-empty list")
    by_role: dict[str, tuple[Mapping[str, Any], Path]] = {}
    for index, item in enumerate(provenance):
        artifact = _mapping(item, f"provenance[{index}]")
        role = str(artifact.get("role") or "")
        if role in by_role:
            raise RegistryError(f"duplicate provenance role: {role}")
        path = _local_path(root, artifact.get("path"), f"provenance[{index}].path")
        expected_hash = _sha(artifact.get("sha256"), f"provenance[{index}].sha256")
        if _file_sha256(path) != expected_hash:
            raise RegistryError(f"provenance hash mismatch for {role}")
        if artifact.get("release_claim") != "none":
            raise RegistryError(f"provenance {role} must not claim release")
        by_role[role] = (artifact, path)
    if set(by_role) != EXPECTED_ARTIFACT_ROLES:
        missing = sorted(EXPECTED_ARTIFACT_ROLES - set(by_role))
        extra = sorted(set(by_role) - EXPECTED_ARTIFACT_ROLES)
        raise RegistryError(f"provenance roles mismatch; missing={missing}, extra={extra}")

    r6 = _load_json(by_role["section_b_relative_bundle"][1], "r6 bundle")
    if r6.get("status") != "b_section_accepted_relative_unpublished":
        raise RegistryError("r6 must remain relative and unpublished")
    if r6.get("execution_ready") is not False or r6.get("publish_authorized") is not False:
        raise RegistryError("r6 must not grant execution or publication")

    pointer = _load_json(by_role["fn_active_pointer"][1], "Fn active pointer")
    fn_candidate = _load_json(by_role["fn_diagnostic_candidate"][1], "Fn candidate")
    if pointer.get("status") != "diagnostic_overlay_default":
        raise RegistryError("Fn pointer must remain diagnostic")
    if pointer.get("active_candidate_path") != by_role["fn_diagnostic_candidate"][0].get("path"):
        raise RegistryError("Fn pointer does not bind the registered candidate")
    execution_flags = _mapping(fn_candidate.get("execution_flags"), "Fn execution_flags")
    if any(execution_flags.get(field) is not False for field in expected_flags if field != "deployment_frozen"):
        raise RegistryError("Fn candidate must not grant execution, contact, motion, press, or publish")
    transform = _mapping(fn_candidate.get("candidate_transform"), "Fn candidate_transform")
    if transform.get("rotation_status") != "unmeasured" or transform.get("transform_complete") is not False:
        raise RegistryError("Fn must remain an incomplete translation-only diagnostic")

    expected_registry_hash = _sha(
        registry.get("registry_payload_sha256"), "registry_payload_sha256"
    )
    if registry_payload_sha256(registry) != expected_registry_hash:
        raise RegistryError("registry payload hash mismatch")

    return {
        "schema": "precontact_calibration_registry_validation_v1",
        "valid": True,
        "calibration_status": "candidate_release_ready",
        "release_status": "not_released",
        "execution_ready": False,
        "motion_authorized": False,
        "deployment_frozen": True,
        "validated_artifact_count": len(by_role),
        "registry_payload_sha256": expected_registry_hash,
    }


def load_and_validate_registry(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    registry_path = Path(path)
    return validate_registry(
        _load_json(registry_path, "registry"), repository_root=repository_root
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = load_and_validate_registry(
            args.registry, repository_root=args.repository_root
        )
    except (OSError, RegistryError) as exc:
        report = {
            "schema": "precontact_calibration_registry_validation_v1",
            "valid": False,
            "execution_ready": False,
            "motion_authorized": False,
            "deployment_frozen": True,
            "blocked": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
