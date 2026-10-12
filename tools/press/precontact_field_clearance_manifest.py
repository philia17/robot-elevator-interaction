#!/usr/bin/env python3
"""Validate a field-observed, non-executable pre-contact clearance manifest.

This pure module reads local JSON evidence. It has no ROS, driver, serial,
publisher, service, action, trajectory, or motion-authority behavior.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "field_observed_precontact_clearance_manifest_v1"
PASS_REASONS = {
    "TOOL_KIND_OPERATOR_CONFIRMED_STIRRING_ROD",
    "FRONT_TIP_OPERATOR_CONFIRMED_SILICONE_30MM_NOMINAL",
    "FRONT_EXPECTED_PATH_REPORTED_CLEAR",
    "REAR_VERTICAL_CAMERA_BRACKET_REPORTED_OUTSIDE_EXPECTED_PATH",
}
BLOCKED_REASONS = {
    "FIELD_OBSERVATION_HAS_NO_COORDINATE_BOUND_CLEARANCE",
    "HAND_TCP_TO_INSTALLED_TOOL_TRANSFORM_UNMEASURED",
    "FULL_INSTALLED_TOOL_ENVELOPE_NOT_MODELED",
    "REAR_CAMERA_BRACKET_GEOMETRY_UNMEASURED",
    "REAR_KEEPOUT_RELEASE_OFFLINE_SYNTHETIC_PENDING",
    "CONTINUOUS_FULL_BODY_SWEPT_VOLUME_NOT_CERTIFIED",
    "NO_COLLISION_RELEASE",
}
PROVENANCE_ROLES = {
    "precontact_calibration_candidate_registry",
    "installed_tool_envelope_schema",
    "keepout_preview_schema",
}
SOURCE_FACTS = {
    "installed tool is a stirring rod",
    "front silicone tip nominal diameter is 30 mm",
    "front side was reported clear",
    "rear side has only a vertical camera bracket, reported outside the expected path",
}
TOP_LEVEL_FIELDS = {
    "schema",
    "mode",
    "observation_id",
    "observation_date",
    "observation_time_status",
    "operator_confirmation_source",
    "tool_observation",
    "front_corridor_observation",
    "rear_observation",
    "diagnostic_pass_reasons",
    "prepare_disposition",
    "provenance",
    "review_status",
    "release_status",
    "execution_ready",
    "motion_authorized",
    "contact_authorized",
    "adapter_consumable",
    "deployment_frozen",
    "canonical_manifest_sha256",
}
HEX = frozenset("0123456789abcdef")


class FieldClearanceBlocked(ValueError):
    """The diagnostic manifest is missing evidence or overclaims clearance."""


def canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FieldClearanceBlocked(f"canonical JSON failed: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(manifest))
    value.pop("canonical_manifest_sha256", None)
    return canonical_sha256(value)


def confirmation_sha256(record: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(record))
    value.pop("record_sha256", None)
    return canonical_sha256(value)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FieldClearanceBlocked(f"{field} must be an object")
    return value


def _sha(value: Any, field: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(character not in HEX for character in result):
        raise FieldClearanceBlocked(f"{field} must be a lowercase SHA-256")
    return result


def _repository_file(root: Path, value: Any, field: str) -> Path:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute():
        raise FieldClearanceBlocked(f"{field} must be repository-relative")
    base = root.resolve()
    resolved = (base / relative).resolve()
    if resolved != base and base not in resolved.parents:
        raise FieldClearanceBlocked(f"{field} escapes repository root")
    if not resolved.is_file():
        raise FieldClearanceBlocked(f"{field} does not exist")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_field_clearance_manifest(
    manifest: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    value = _mapping(manifest, "manifest")
    if set(value) != TOP_LEVEL_FIELDS:
        raise FieldClearanceBlocked("manifest has unknown or missing fields")
    if value.get("schema") != SCHEMA:
        raise FieldClearanceBlocked("unexpected manifest schema")
    if value.get("mode") != "DIAGNOSTIC_FIELD_OBSERVATION_ONLY":
        raise FieldClearanceBlocked("manifest mode must remain diagnostic-only")
    if value.get("observation_time_status") != "date_only_no_timestamp":
        raise FieldClearanceBlocked("observation must not invent an exact timestamp")
    if value.get("review_status") != "field_observation_recorded_diagnostic_not_release_reviewed":
        raise FieldClearanceBlocked("review_status must not claim release review")
    if value.get("release_status") != "not_released":
        raise FieldClearanceBlocked("manifest must remain not_released")
    expected_flags = {
        "execution_ready": False,
        "motion_authorized": False,
        "contact_authorized": False,
        "adapter_consumable": False,
        "deployment_frozen": True,
    }
    for field, expected in expected_flags.items():
        if value.get(field) is not expected:
            raise FieldClearanceBlocked(f"{field} must be {expected}")

    source = _mapping(value.get("operator_confirmation_source"), "operator_confirmation_source")
    if set(source) != {
        "source_kind",
        "source_id",
        "scope",
        "facts",
        "coordinate_measurement_authorized",
        "collision_release_authorized",
        "deployment_authorized",
        "motion_authorized",
        "record_sha256",
    }:
        raise FieldClearanceBlocked("operator confirmation has unknown or missing fields")
    if source.get("source_kind") != "conversation_instruction":
        raise FieldClearanceBlocked("operator confirmation source is missing")
    if source.get("scope") != "diagnostic_clearance_facts_only":
        raise FieldClearanceBlocked("operator confirmation scope is invalid")
    for field in (
        "coordinate_measurement_authorized",
        "collision_release_authorized",
        "deployment_authorized",
        "motion_authorized",
    ):
        if source.get(field) is not False:
            raise FieldClearanceBlocked(f"operator confirmation must keep {field}=false")
    facts = source.get("facts")
    if not isinstance(facts, list) or set(facts) != SOURCE_FACTS:
        raise FieldClearanceBlocked("operator confirmation facts are incomplete or unexpected")
    supplied_source_hash = _sha(source.get("record_sha256"), "operator record_sha256")
    if confirmation_sha256(source) != supplied_source_hash:
        raise FieldClearanceBlocked("operator confirmation source hash mismatch")

    tool = _mapping(value.get("tool_observation"), "tool_observation")
    if set(tool) != {
        "tool_kind",
        "front_tip_material",
        "front_tip_nominal_diameter_m",
        "diameter_evidence",
        "hand_tcp_attachment_transform",
        "full_installed_tool_envelope",
    }:
        raise FieldClearanceBlocked("tool observation has unknown or missing fields")
    if tool.get("tool_kind") != "stirring_rod":
        raise FieldClearanceBlocked("tool must remain operator-confirmed stirring_rod")
    if tool.get("front_tip_material") != "silicone":
        raise FieldClearanceBlocked("front tip material must remain silicone")
    diameter = tool.get("front_tip_nominal_diameter_m")
    if isinstance(diameter, bool) or not isinstance(diameter, (int, float)) or not math.isfinite(diameter):
        raise FieldClearanceBlocked("front tip nominal diameter must be finite")
    if abs(float(diameter) - 0.030) > 1e-12:
        raise FieldClearanceBlocked("front tip nominal diameter must be 0.030 m")
    if tool.get("diameter_evidence") != "operator_confirmed_nominal_not_independently_measured":
        raise FieldClearanceBlocked("diameter evidence must not claim independent measurement")
    if tool.get("hand_tcp_attachment_transform") != "unmeasured":
        raise FieldClearanceBlocked("hand_tcp attachment must remain unmeasured")
    if tool.get("full_installed_tool_envelope") != "not_modeled":
        raise FieldClearanceBlocked("full installed tool envelope must remain not_modeled")

    front = _mapping(value.get("front_corridor_observation"), "front_corridor_observation")
    rear = _mapping(value.get("rear_observation"), "rear_observation")
    if set(front) != {
        "status",
        "reported_condition",
        "coordinate_bound_geometry",
        "continuous_swept_volume_checked",
        "diagnostic_result",
    }:
        raise FieldClearanceBlocked("front observation has unknown or missing fields")
    if set(rear) != {
        "status",
        "obstacle_kind",
        "reported_relation",
        "coordinate_bound_geometry",
        "continuous_swept_volume_checked",
        "rear_keepout_release_status",
        "diagnostic_result",
    }:
        raise FieldClearanceBlocked("rear observation has unknown or missing fields")
    for name, observation in (("front", front), ("rear", rear)):
        if observation.get("status") != "field_observed_only":
            raise FieldClearanceBlocked(f"{name} observation must remain field_observed_only")
        if observation.get("coordinate_bound_geometry") is not False:
            raise FieldClearanceBlocked(f"{name} observation cannot claim coordinate geometry")
        if observation.get("continuous_swept_volume_checked") is not False:
            raise FieldClearanceBlocked(f"{name} observation cannot claim swept-volume checking")
        if observation.get("diagnostic_result") != "PASS_FIELD_OBSERVATION_ONLY":
            raise FieldClearanceBlocked(f"{name} observation has invalid diagnostic result")
    if front.get("reported_condition") != "clear_in_front_of_expected_no_contact_path":
        raise FieldClearanceBlocked("front reported condition is invalid")
    if rear.get("obstacle_kind") != "vertical_camera_bracket":
        raise FieldClearanceBlocked("rear obstacle must be vertical_camera_bracket")
    if rear.get("reported_relation") != "outside_expected_path":
        raise FieldClearanceBlocked("rear relation must remain an observation only")
    if rear.get("rear_keepout_release_status") != "offline_synthetic_pending":
        raise FieldClearanceBlocked("rear keepout must remain offline_synthetic_pending")

    if set(value.get("diagnostic_pass_reasons") or []) != PASS_REASONS:
        raise FieldClearanceBlocked("diagnostic pass reasons are incomplete or unexpected")
    disposition = _mapping(value.get("prepare_disposition"), "prepare_disposition")
    if set(disposition) != {
        "prepare_readable",
        "execution_admissible",
        "result",
        "ticket_may_be_issued",
        "blocked_reasons",
    }:
        raise FieldClearanceBlocked("prepare disposition has unknown or missing fields")
    if disposition.get("prepare_readable") is not True:
        raise FieldClearanceBlocked("manifest must remain prepare-readable")
    if disposition.get("execution_admissible") is not False:
        raise FieldClearanceBlocked("diagnostic manifest cannot be execution-admissible")
    if disposition.get("result") != "BLOCKED_DIAGNOSTIC_CLEARANCE_ONLY":
        raise FieldClearanceBlocked("prepare disposition must remain blocked")
    if disposition.get("ticket_may_be_issued") is not False:
        raise FieldClearanceBlocked("diagnostic manifest cannot issue a ticket")
    if set(disposition.get("blocked_reasons") or []) != BLOCKED_REASONS:
        raise FieldClearanceBlocked("prepare blocked reasons are incomplete or unexpected")

    provenance = value.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise FieldClearanceBlocked("provenance is required")
    root = Path(repository_root)
    seen: set[str] = set()
    for index, item in enumerate(provenance):
        record = _mapping(item, f"provenance[{index}]")
        if set(record) != {"role", "path", "sha256"}:
            raise FieldClearanceBlocked(
                f"provenance[{index}] has unknown or missing fields"
            )
        role = str(record.get("role") or "")
        if role in seen:
            raise FieldClearanceBlocked(f"duplicate provenance role: {role}")
        path = _repository_file(root, record.get("path"), f"provenance[{index}].path")
        supplied_hash = _sha(record.get("sha256"), f"provenance[{index}].sha256")
        if _file_sha256(path) != supplied_hash:
            raise FieldClearanceBlocked(f"provenance hash mismatch for {role}")
        seen.add(role)
    if seen != PROVENANCE_ROLES:
        raise FieldClearanceBlocked("provenance roles are incomplete or unexpected")

    supplied_manifest_hash = _sha(
        value.get("canonical_manifest_sha256"), "canonical_manifest_sha256"
    )
    if manifest_sha256(value) != supplied_manifest_hash:
        raise FieldClearanceBlocked("canonical manifest hash mismatch")
    return {
        "schema": "field_observed_precontact_clearance_validation_v1",
        "valid": True,
        "field_observed_verdict": "PASS_FIELD_OBSERVATION_ONLY",
        "prepare_result": "BLOCKED_DIAGNOSTIC_CLEARANCE_ONLY",
        "blocked_reasons": sorted(BLOCKED_REASONS),
        "execution_ready": False,
        "motion_authorized": False,
        "adapter_consumable": False,
        "deployment_frozen": True,
        "canonical_manifest_sha256": supplied_manifest_hash,
    }


def load_and_validate(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FieldClearanceBlocked(f"manifest is not valid JSON: {exc}") from exc
    return validate_field_clearance_manifest(value, repository_root=repository_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = load_and_validate(args.manifest, repository_root=args.repository_root)
    except FieldClearanceBlocked as exc:
        report = {
            "schema": "field_observed_precontact_clearance_validation_v1",
            "valid": False,
            "prepare_result": "BLOCKED",
            "execution_ready": False,
            "motion_authorized": False,
            "adapter_consumable": False,
            "deployment_frozen": True,
            "blocked": str(exc),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
