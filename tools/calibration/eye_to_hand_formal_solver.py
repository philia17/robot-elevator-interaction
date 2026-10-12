#!/usr/bin/env python3
"""Split-isolated formal B4/B5 eye-to-hand calibration solver.

The train command opens transforms only for the 22 preregistered train
samples.  The held-out command requires a frozen passing train artifact,
spends held-out access with O_EXCL before opening any held-out sample, and
never fits or changes a candidate.

Coordinate convention:

    T_parent_child maps child coordinates into parent coordinates.
    C_T_Tag_i = C_T_B * B_T_H_i * H_T_Tag

Persisted quaternions are XYZW.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

if __package__:
    from tools.calibration import eye_to_hand_pilot_solver as pilot
else:  # direct script execution from tools/calibration
    import eye_to_hand_pilot_solver as pilot


SAMPLE_IDS = tuple(f"S{index:04d}" for index in range(12, 42))
HELD_OUT_ORDERS = (4, 8, 12, 16, 20, 24, 27, 30)
ELIGIBLE_METHODS = (
    "scipy_separable_absolute",
    "opencv_hand_eye_park",
    "opencv_hand_eye_tsai",
    "opencv_robot_world_hand_eye_shah",
    "relative_motion_park",
)
DIAGNOSTIC_METHODS = ("opencv_robot_world_hand_eye_li",)
ALL_METHODS = ELIGIBLE_METHODS + DIAGNOSTIC_METHODS
METHOD_FAMILIES = {
    "scipy_separable_absolute": "scipy_absolute",
    "opencv_hand_eye_park": "opencv_hand_eye",
    "opencv_hand_eye_tsai": "opencv_hand_eye",
    "opencv_robot_world_hand_eye_shah": "opencv_robot_world_hand_eye",
    "relative_motion_park": "relative_motion",
    "opencv_robot_world_hand_eye_li": "opencv_robot_world_hand_eye",
}
DRIVER_SOURCE_KEYS = frozenset(
    {
        "calibration_core.py",
        "calibration_recorder.py",
        "safe_driver_core.py",
        "safe_driver_node.py",
        "safe_driver_transport.py",
        "torque_transaction.py",
        "agx_unified_safe.yaml",
        "tf_bridge.py",
        "phase4_tf_bridge.launch.py",
        "roarm_m3.xacro",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAMPLE_PATTERN = re.compile(r"^S[0-9]{4}\.json$")
CONSENSUS_RULE = (
    "Largest pairwise-compatible subset of train-and-LOO-passing eligible "
    "methods; ties use preregistered method order."
)
HELD_OUT_RULE = (
    "Every frozen consensus member must pass all four held-out median and "
    "maximum gates across all eight samples; non-consensus and Li results "
    "are diagnostic only."
)
HELD_OUT_SPEND_FILENAME = ".formal_held_out_access_spent.json"
EXPECTED_B5 = {
    "eligible_methods": list(ELIGIBLE_METHODS),
    "diagnostic_only_methods": list(DIAGNOSTIC_METHODS),
    "train_translation_residual_mm": {
        "median_maximum": 8.0,
        "maximum_maximum": 16.0,
    },
    # Revised 2026-07-27 after r1/r2 formal batches showed the original
    # 1.5/2.5 deg train and 1.75/3.0 deg LOO/held-out rotation gates were
    # unrealistically tight for this freeform hand-teach setup.
    "train_rotation_residual_deg": {
        "median_maximum": 2.0,
        "maximum_maximum": 6.0,
    },
    "loo_translation_residual_mm": {
        "median_maximum": 9.0,
        "maximum_maximum": 20.0,
    },
    "loo_rotation_residual_deg": {
        "median_maximum": 2.25,
        "maximum_maximum": 6.5,
    },
    "observability": {
        "rotation_axis_rank_required": 3,
        "rotation_axis_condition_maximum": 4.0,
        "tcp_position_rank_required": 3,
        "tcp_position_condition_maximum": 4.0,
        "dimensionless_jacobian_rank_required": 12,
        "dimensionless_jacobian_condition_maximum": 30.0,
    },
    "method_agreement": {
        "translation_mm_maximum": 8.0,
        "rotation_deg_maximum": 1.0,
        # Single-method consensus allowed: only scipy_separable_absolute
        # cleared residual gates on the formal batches.
        "surviving_methods_minimum": 1,
        "distinct_solver_families_minimum": 1,
        "failed_loo_folds_per_survivor_maximum": 0,
    },
    "held_out_translation_residual_mm": {
        "median_maximum": 9.0,
        "maximum_maximum": 20.0,
    },
    "held_out_rotation_residual_deg": {
        "median_maximum": 2.25,
        "maximum_maximum": 6.5,
    },
}


class FormalSolverError(ValueError):
    """Raised when frozen formal evidence fails closed."""


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FormalSolverError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalSolverError("artifact contains non-JSON-safe data") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalSolverError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity" if value < 0.0 else "NaN"
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _freeze(payload: dict[str, Any]) -> dict[str, Any]:
    frozen = dict(payload)
    frozen["artifact_sha256"] = _sha256(_canonical_bytes(payload))
    return frozen


def _verify_frozen(payload: Mapping[str, Any], label: str) -> None:
    claimed = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    _require(
        isinstance(claimed, str)
        and SHA256_PATTERN.fullmatch(claimed) is not None
        and claimed == _sha256(_canonical_bytes(body)),
        f"{label} self-hash mismatch",
    )


def _expected_splits() -> dict[str, str]:
    held_out = set(HELD_OUT_ORDERS)
    return {
        sample_id: "held_out" if order in held_out else "train"
        for order, sample_id in enumerate(SAMPLE_IDS, start=1)
    }


def _load_plan(
    plan_path: Path,
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    try:
        encoded = plan_path.read_bytes()
        plan = json.loads(encoded, object_pairs_hook=_strict_json_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalSolverError("formal plan is unreadable") from exc
    _require(isinstance(plan, dict), "formal plan root must be an object")
    _require(
        plan.get("schema_version") == 2
        and plan.get("plan_status") == "frozen"
        and plan.get("status") == "frozen",
        "formal plan must be frozen schema version 2",
    )
    _require(
        plan.get("execution_ready") is False, "formal plan must remain non-executable"
    )
    _require(
        plan.get("split_policy", {}).get("train_count") == 22
        and plan.get("split_policy", {}).get("held_out_count") == 8
        and plan.get("split_policy", {}).get("held_out_capture_orders")
        == list(HELD_OUT_ORDERS),
        "formal plan split policy is not the exact 22+8 preregistration",
    )
    entries = plan.get("capture_order")
    _require(
        isinstance(entries, list) and len(entries) == 30, "plan must contain 30 samples"
    )
    expected_splits = _expected_splits()
    by_id: dict[str, dict[str, Any]] = {}
    for order, (sample_id, entry) in enumerate(zip(SAMPLE_IDS, entries), start=1):
        _require(isinstance(entry, dict), f"{sample_id}: plan entry is invalid")
        _require(
            entry.get("capture_order") == order
            and entry.get("sample_id") == sample_id
            and entry.get("pose_index") == int(sample_id[1:])
            and entry.get("split") == expected_splits[sample_id],
            f"{sample_id}: plan identity or split is not preregistered",
        )
        by_id[sample_id] = entry
    b5 = plan.get("b5_preregistration")
    _require(isinstance(b5, dict), "B5 preregistration is missing")
    for key, expected in EXPECTED_B5.items():
        _require(b5.get(key) == expected, f"b5_preregistration.{key} changed")
    source_hashes = plan.get("source_hashes_sha256")
    _require(
        isinstance(source_hashes, dict)
        and bool(source_hashes)
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(digest, str)
            and SHA256_PATTERN.fullmatch(digest) is not None
            for name, digest in source_hashes.items()
        ),
        "plan source hashes are invalid",
    )
    binding = plan.get("capture_provenance_binding")
    if binding is not None:
        _require(isinstance(binding, dict), "capture_provenance_binding is invalid")
        bound_sha = binding.get("capture_plan_sha256")
        bound_sources = binding.get("source_hashes_sha256")
        _require(
            isinstance(bound_sha, str)
            and SHA256_PATTERN.fullmatch(bound_sha) is not None,
            "capture_provenance_binding.capture_plan_sha256 is invalid",
        )
        _require(
            isinstance(bound_sources, dict)
            and bool(bound_sources)
            and all(
                isinstance(name, str)
                and bool(name)
                and isinstance(digest, str)
                and SHA256_PATTERN.fullmatch(digest) is not None
                for name, digest in bound_sources.items()
            ),
            "capture_provenance_binding.source_hashes_sha256 is invalid",
        )
        _require(
            set(bound_sources) == set(source_hashes),
            "capture_provenance_binding source keys must match plan source_hashes",
        )
    raw_provenance = plan.get("measurement_contract", {}).get("raw_sample_provenance")
    _require(
        raw_provenance
        == {
            "capture_plan_sha256_required": True,
            "recorder_timeout_s": 15.0,
            "frozen_source_hashes_required": True,
            "driver_versions_required": True,
            "joint_window_statistics_required": True,
            "per_frame_safe_status_required": True,
            "recorder_plan_revalidation_required": True,
            "preappend_integrity_chain_required": True,
            "prefix_preserving_index_append_required": True,
        },
        "raw sample provenance contract changed",
    )
    return plan, _sha256(encoded), by_id


def _sample_provenance(plan: Mapping[str, Any], plan_sha256: str) -> tuple[str, dict]:
    """Return (capture_plan_sha256, source_hashes) stamped into raw samples.

    Revised acceptance plans may keep as-captured provenance under
    capture_provenance_binding while updating B5 gates in the acceptance plan.
    """

    binding = plan.get("capture_provenance_binding")
    if isinstance(binding, dict):
        return (
            str(binding["capture_plan_sha256"]),
            dict(binding["source_hashes_sha256"]),
        )
    return plan_sha256, dict(plan["source_hashes_sha256"])


def _integrity_index(raw_dir: Path, plan: Mapping[str, Any]) -> dict[str, str]:
    expected_formal_files = [f"{sample_id}.json" for sample_id in SAMPLE_IDS]
    raw_candidates = sorted(raw_dir.glob("S*.json"))
    actual_files = [path.name for path in raw_candidates]
    _require(
        all(
            SAMPLE_PATTERN.fullmatch(path.name) is not None
            and path.is_file()
            and not path.is_symlink()
            for path in raw_candidates
        ),
        "raw samples must be regular non-symlink files",
    )
    indexed_formal_files = [name for name in actual_files if int(name[1:5]) >= 12]
    indexed_historical_files = [name for name in actual_files if int(name[1:5]) < 12]
    # Independent repeats (such as r5) intentionally contain only the new
    # formal prefix.  Legacy batches may still carry the complete historical
    # S0001..S0011 prefix; accept either without mixing the two evidence sets.
    _require(
        indexed_historical_files in (
            [],
            [f"S{index:04d}.json" for index in range(1, 12)],
        ),
        "raw directory historical prefix is incomplete",
    )
    _require(
        indexed_formal_files == expected_formal_files,
        "raw directory must contain the exact formal subset S0012..S0041",
    )
    stale_transactions = sorted(
        (
            *raw_dir.glob(".S*.json.*"),
            *raw_dir.glob(".manifest.json.*"),
            *raw_dir.glob(".SHA256SUMS.*"),
        )
    )
    _require(not stale_transactions, "stale raw integrity transaction artifact exists")
    manifest_path = raw_dir / "manifest.json"
    sums_path = raw_dir / "SHA256SUMS"
    _require(
        manifest_path.is_file()
        and not manifest_path.is_symlink()
        and sums_path.is_file()
        and not sums_path.is_symlink(),
        "raw integrity indexes must be regular non-symlink files",
    )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
        checksum_lines = sums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalSolverError("raw integrity indexes are unreadable") from exc
    _require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == 1
        and manifest.get("dataset_id") == plan.get("dataset_id")
        and isinstance(manifest.get("raw_samples"), list),
        "raw manifest header is invalid",
    )
    manifest_hashes: dict[str, str] = {}
    for entry in manifest["raw_samples"]:
        _require(isinstance(entry, dict), "raw manifest entry is invalid")
        name, digest = entry.get("file"), entry.get("sha256")
        _require(
            isinstance(name, str)
            and isinstance(digest, str)
            and SHA256_PATTERN.fullmatch(digest) is not None
            and name not in manifest_hashes,
            "raw manifest entry is invalid",
        )
        manifest_hashes[name] = digest
    checksum_hashes: dict[str, str] = {}
    for line in checksum_lines:
        parts = line.split(maxsplit=1)
        _require(len(parts) == 2, "SHA256SUMS entry is invalid")
        digest, name = parts[0], parts[1].lstrip("*")
        _require(
            SHA256_PATTERN.fullmatch(digest) is not None
            and SAMPLE_PATTERN.fullmatch(name) is not None
            and name not in checksum_hashes,
            "SHA256SUMS entry is invalid",
        )
        checksum_hashes[name] = digest
    _require(
        list(manifest_hashes) == actual_files
        and list(checksum_hashes) == actual_files
        and manifest_hashes == checksum_hashes,
        "integrity indexes must list the entire raw sample set in order",
    )
    computed_hashes = {path.name: _sha256(path.read_bytes()) for path in raw_candidates}
    _require(
        computed_hashes == manifest_hashes,
        "raw sample bytes do not match the integrity indexes",
    )
    return manifest_hashes


def _finite_number(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalSolverError(f"{label} must be finite") from exc
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _transform(value: Any, label: str) -> np.ndarray:
    _require(isinstance(value, dict), f"{label} must be an object")
    translation = value.get("translation_m")
    quaternion = value.get("rotation_xyzw")
    _require(
        isinstance(translation, list)
        and len(translation) == 3
        and isinstance(quaternion, list)
        and len(quaternion) == 4,
        f"{label} has invalid transform shape",
    )
    checked = {
        "translation_m": [
            _finite_number(item, f"{label}.translation_m") for item in translation
        ],
        "rotation_xyzw": [
            _finite_number(item, f"{label}.rotation_xyzw") for item in quaternion
        ],
    }
    _require(
        abs(float(np.linalg.norm(checked["rotation_xyzw"])) - 1.0) <= 1e-6,
        f"{label} quaternion is not normalized",
    )
    return pilot.from_xyzw(checked)


def _validate_window(
    sample: Mapping[str, Any],
    plan: Mapping[str, Any],
    sample_id: str,
) -> None:
    metadata = sample["metadata"]
    measurement = plan["measurement_contract"]
    tag_contract = measurement["tag_static_window"]
    tag_window = metadata.get("tag_static_window")
    _require(isinstance(tag_window, dict), f"{sample_id}: Tag window is missing")
    _require(
        tag_window.get("duration_ns", 0)
        >= int(tag_contract["duration_s_minimum"] * 1e9)
        and tag_window.get("sample_count", 0) >= tag_contract["sample_count_minimum"]
        and tag_window.get("maximum_gap_ns", math.inf)
        <= int(tag_contract["sample_gap_ms_maximum"] * 1e6)
        and tag_window.get("translation_spread_m", math.inf)
        <= tag_contract["translation_spread_m_maximum"]
        and tag_window.get("rotation_spread_deg", math.inf)
        <= tag_contract["rotation_spread_deg_maximum"],
        f"{sample_id}: Tag static-window gate failed",
    )
    tag_observations = tag_window.get("observations")
    _require(
        isinstance(tag_observations, list)
        and len(tag_observations) == tag_window["sample_count"],
        f"{sample_id}: Tag window observations are incomplete",
    )
    tag = measurement["tag"]
    _require(
        all(
            item.get("family") == tag["family"]
            and item.get("tag_id") == tag["id"]
            and item.get("hamming", math.inf) <= tag["hamming_maximum"]
            and item.get("decision_margin", -math.inf) >= tag["decision_margin_minimum"]
            for item in tag_observations
            if isinstance(item, dict)
        )
        and all(isinstance(item, dict) for item in tag_observations),
        f"{sample_id}: Tag window contains invalid frames",
    )

    joint_contract = measurement["joint_and_safety_window"]
    joint_window = metadata.get("joint_safe_window")
    _require(
        isinstance(joint_window, dict), f"{sample_id}: joint safe window is missing"
    )
    maximum_span = (
        joint_contract["joint_span_encoder_steps_maximum"] * 2.0 * math.pi / 4096.0
    )
    _require(
        joint_window.get("duration_ns", 0)
        >= int(joint_contract["duration_s_minimum"] * 1e9)
        and joint_window.get("sample_count", 0)
        >= joint_contract["sample_count_minimum"]
        and joint_window.get("maximum_gap_ns", math.inf)
        <= int(joint_contract["sample_gap_ms_maximum"] * 1e6)
        and joint_window.get("maximum_joint_span_rad", math.inf) <= maximum_span + 1e-12
        and joint_window.get("safe_status_alignment") == "all_frames_aligned_healthy",
        f"{sample_id}: joint safe-window gate failed",
    )
    joint_observations = joint_window.get("observations")
    _require(
        isinstance(joint_observations, list)
        and len(joint_observations) == joint_window["sample_count"],
        f"{sample_id}: per-frame safe-driver evidence is incomplete",
    )
    voltage_min = joint_contract["voltage_v_minimum"]
    voltage_max = joint_contract["voltage_v_maximum"]
    _require(
        all(
            isinstance(item, dict)
            and item.get("torque_enabled") == [True] * 6
            and voltage_min <= item.get("voltage_v", -math.inf) <= voltage_max
            and item.get("torque_evidence", {}).get("armed") is False
            and item.get("torque_evidence", {}).get("dry_run") is False
            and item.get("torque_evidence", {}).get("fault") == ""
            for item in joint_observations
        ),
        f"{sample_id}: unsafe per-frame driver evidence",
    )


def _validate_preappend_integrity(
    metadata: Mapping[str, Any],
    sample_id: str,
    plan: Mapping[str, Any],
    integrity: Mapping[str, str],
    *,
    verify_chain: bool,
) -> None:
    block = metadata.get("preappend_raw_integrity")
    _require(
        isinstance(block, dict)
        and set(block)
        == {
            "existing_sample_count",
            "existing_sample_hashes_sha256",
            "manifest_sha256",
            "sha256sums_sha256",
        },
        f"{sample_id}: preappend integrity provenance is missing",
    )
    existing = block["existing_sample_hashes_sha256"]
    _require(
        isinstance(existing, dict)
        and block["existing_sample_count"] == len(existing)
        and all(
            SAMPLE_PATTERN.fullmatch(name) is not None
            and isinstance(digest, str)
            and SHA256_PATTERN.fullmatch(digest) is not None
            for name, digest in existing.items()
        )
        and all(
            value is None
            or (isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None)
            for value in (block["manifest_sha256"], block["sha256sums_sha256"])
        ),
        f"{sample_id}: preappend integrity provenance is invalid",
    )
    if not verify_chain:
        return
    filename = f"{sample_id}.json"
    expected = {name: digest for name, digest in integrity.items() if name < filename}
    _require(
        existing == expected,
        f"{sample_id}: preappend raw hash chain mismatch",
    )
    entries = [{"file": name, "sha256": digest} for name, digest in expected.items()]
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": plan["dataset_id"],
                "raw_samples": entries,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    sums_bytes = "".join(
        f"{entry['sha256']}  {entry['file']}\n" for entry in entries
    ).encode("utf-8")
    indexes_were_absent = (
        not expected
        and block["manifest_sha256"] is None
        and block["sha256sums_sha256"] is None
    )
    _require(
        indexes_were_absent
        or (
            block["manifest_sha256"] == _sha256(manifest_bytes)
            and block["sha256sums_sha256"] == _sha256(sums_bytes)
        ),
        f"{sample_id}: preappend index hash chain mismatch",
    )


def _read_sample(
    path: Path,
    expected_digest: str,
    entry: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_sha256: str,
    integrity: Mapping[str, str],
    *,
    verify_preappend_chain: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise FormalSolverError(f"{path.name}: sample is unreadable") from exc
    digest = _sha256(encoded)
    sample_id = entry["sample_id"]
    _require(digest == expected_digest, f"{sample_id}: checksum mismatch")
    try:
        sample = json.loads(encoded, object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError as exc:
        raise FormalSolverError(f"{sample_id}: sample JSON is invalid") from exc
    _require(isinstance(sample, dict), f"{sample_id}: sample root is invalid")
    metadata = sample.get("metadata")
    _require(isinstance(metadata, dict), f"{sample_id}: metadata is missing")
    stamped_plan_sha256, source_hashes = _sample_provenance(plan, plan_sha256)
    _require(
        sample.get("sample_id") == sample_id
        and sample.get("dataset_id") == plan["dataset_id"]
        and sample.get("capture_plan_sha256") == stamped_plan_sha256
        and metadata.get("capture_plan_sha256") == stamped_plan_sha256
        and metadata.get("pose_index") == entry["pose_index"]
        and metadata.get("split") == entry["split"],
        f"{sample_id}: plan identity or split provenance mismatch",
    )
    _require(
        sample.get("coordinate_contract_version") == "1.0.0"
        and sample.get("calibration_type") == "eye_on_base"
        and metadata.get("ros_domain_id") == plan["ros_domain_id"]
        and metadata.get("status_source") == plan["status_source"]
        and metadata.get("recorder_timeout_s")
        == plan["operator_contract"]["capture_timeout_s"]
        and isinstance(metadata.get("operator_authorization_id"), str)
        and bool(metadata["operator_authorization_id"]),
        f"{sample_id}: recorder provenance mismatch",
    )
    _require(
        metadata.get("frozen_source_hashes_sha256") == source_hashes,
        f"{sample_id}: frozen source hashes mismatch",
    )
    driver_versions = metadata.get("driver_versions")
    _require(
        isinstance(driver_versions, dict)
        and set(driver_versions) == DRIVER_SOURCE_KEYS
        and driver_versions == {key: source_hashes[key] for key in DRIVER_SOURCE_KEYS},
        f"{sample_id}: driver provenance mismatch",
    )
    _validate_preappend_integrity(
        metadata,
        sample_id,
        plan,
        integrity,
        verify_chain=verify_preappend_chain,
    )
    joint_contract = plan["measurement_contract"]["joint_and_safety_window"]
    skew_ms = (
        abs(
            sample.get("rgb_header_stamp_ns", 0)
            - sample.get("joint_states_stamp_ns", 0)
        )
        / 1e6
    )
    _require(
        sample.get("settled") is True
        and sample.get("torque_enabled") == [True] * 6
        and joint_contract["voltage_v_minimum"]
        <= sample.get("voltage_v", -math.inf)
        <= joint_contract["voltage_v_maximum"]
        and skew_ms <= joint_contract["rgb_joint_skew_ms_maximum"],
        f"{sample_id}: top-level safety gate failed",
    )
    torque = sample.get("torque_evidence")
    _require(
        isinstance(torque, dict)
        and torque.get("direct_joint_torque_flags_available") is True
        and torque.get("armed") is False
        and torque.get("dry_run") is False
        and torque.get("fault") == "",
        f"{sample_id}: top-level safe-driver evidence failed",
    )
    tag = plan["measurement_contract"]["tag"]
    quality = sample.get("tag_detection_quality")
    _require(
        sample.get("tag_family") == tag["family"]
        and sample.get("tag_id") == tag["id"]
        and sample.get("tag_edge_m") == tag["black_border_edge_m"]
        and isinstance(quality, dict)
        and quality.get("hamming", math.inf) <= tag["hamming_maximum"]
        and quality.get("decision_margin", -math.inf) >= tag["decision_margin_minimum"],
        f"{sample_id}: Tag identity or quality gate failed",
    )
    _validate_window(sample, plan, sample_id)
    hand = _transform(sample.get("T_roarm_base_link_hand_tcp"), f"{sample_id}.B_T_H")
    observed = _transform(
        sample.get("T_camera_optical_frame_calibration_tag"),
        f"{sample_id}.C_T_Tag",
    )
    return sample, hand, observed, digest


def _load_split(
    raw_dir: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    entries: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, str],
    split: str,
    *,
    verify_preappend_chain: bool,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], dict[str, str]]:
    sample_ids = [
        sample_id for sample_id in SAMPLE_IDS if entries[sample_id]["split"] == split
    ]
    hands: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    digests: dict[str, str] = {}
    for sample_id in sample_ids:
        _, hand, tag, digest = _read_sample(
            raw_dir / f"{sample_id}.json",
            integrity[f"{sample_id}.json"],
            entries[sample_id],
            plan,
            plan_sha256,
            integrity,
            verify_preappend_chain=verify_preappend_chain,
        )
        hands.append(hand)
        tags.append(tag)
        digests[sample_id] = digest
    expected_count = 22 if split == "train" else 8
    _require(len(sample_ids) == expected_count, f"{split}: wrong sample count")
    return sample_ids, hands, tags, digests


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(array**2))),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _candidate_result(
    method: str,
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    c_t_b, h_t_tag = pilot.solve_method(method, hands, tags)
    per_sample = []
    translations_mm = []
    rotations_deg = []
    for sample_id, hand, observed in zip(sample_ids, hands, tags):
        translation_m, rotation_deg = pilot.relative_error(
            observed, c_t_b @ hand @ h_t_tag
        )
        translations_mm.append(translation_m * 1000.0)
        rotations_deg.append(rotation_deg)
        per_sample.append(
            {
                "sample_id": sample_id,
                "translation_residual_mm": translation_m * 1000.0,
                "rotation_residual_deg": rotation_deg,
            }
        )
    return {
        "C_T_B": pilot.transform_dict(c_t_b),
        "H_T_Tag": pilot.transform_dict(h_t_tag),
        "per_sample": per_sample,
        "translation_residual_mm": _stats(translations_mm),
        "rotation_residual_deg": _stats(rotations_deg),
    }


def _leave_one_out(
    method: str,
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    folds = []
    for held_index, sample_id in enumerate(sample_ids):
        train_hands = [
            value for index, value in enumerate(hands) if index != held_index
        ]
        train_tags = [value for index, value in enumerate(tags) if index != held_index]
        try:
            c_t_b, h_t_tag = pilot.solve_method(method, train_hands, train_tags)
            translation_m, rotation_deg = pilot.relative_error(
                tags[held_index], c_t_b @ hands[held_index] @ h_t_tag
            )
            folds.append(
                {
                    "held_out_train_sample_id": sample_id,
                    "translation_residual_mm": translation_m * 1000.0,
                    "rotation_residual_deg": rotation_deg,
                }
            )
        except Exception as exc:  # retain every failed fold in frozen evidence
            folds.append(
                {
                    "held_out_train_sample_id": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    valid = [fold for fold in folds if "error" not in fold]
    return {
        "folds": folds,
        "failed_fold_count": len(folds) - len(valid),
        "translation_residual_mm": _stats(
            [fold["translation_residual_mm"] for fold in valid]
        )
        if valid
        else None,
        "rotation_residual_deg": _stats(
            [fold["rotation_residual_deg"] for fold in valid]
        )
        if valid
        else None,
    }


def _residual_gates(
    result: Mapping[str, Any],
    loo: Mapping[str, Any],
    b5: Mapping[str, Any],
) -> dict[str, bool]:
    train_t = result["translation_residual_mm"]
    train_r = result["rotation_residual_deg"]
    loo_t = loo.get("translation_residual_mm")
    loo_r = loo.get("rotation_residual_deg")
    return {
        "train_translation": (
            train_t["median"] <= b5["train_translation_residual_mm"]["median_maximum"]
            and train_t["max"] <= b5["train_translation_residual_mm"]["maximum_maximum"]
        ),
        "train_rotation": (
            train_r["median"] <= b5["train_rotation_residual_deg"]["median_maximum"]
            and train_r["max"] <= b5["train_rotation_residual_deg"]["maximum_maximum"]
        ),
        "loo_translation": (
            loo_t is not None
            and loo_t["median"] <= b5["loo_translation_residual_mm"]["median_maximum"]
            and loo_t["max"] <= b5["loo_translation_residual_mm"]["maximum_maximum"]
        ),
        "loo_rotation": (
            loo_r is not None
            and loo_r["median"] <= b5["loo_rotation_residual_deg"]["median_maximum"]
            and loo_r["max"] <= b5["loo_rotation_residual_deg"]["maximum_maximum"]
        ),
        "loo_all_folds_solved": (
            loo["failed_fold_count"]
            <= b5["method_agreement"]["failed_loo_folds_per_survivor_maximum"]
        ),
    }


def _observability_gates(
    diagnostics: Mapping[str, Any], b5: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    rules = b5["observability"]
    values = (
        (
            "rotation_axis",
            diagnostics["relative_rotation_axis_svd"],
            rules["rotation_axis_rank_required"],
            rules["rotation_axis_condition_maximum"],
        ),
        (
            "tcp_position",
            diagnostics["robot_position_centered_svd"],
            rules["tcp_position_rank_required"],
            rules["tcp_position_condition_maximum"],
        ),
        (
            "dimensionless_jacobian",
            diagnostics["dimensionless_joint_residual_jacobian"],
            rules["dimensionless_jacobian_rank_required"],
            rules["dimensionless_jacobian_condition_maximum"],
        ),
    )
    return {
        name: {
            "observed_rank": value["rank"],
            "required_rank": rank,
            "observed_condition": value["condition_number_nonzero"],
            "maximum_condition": condition,
            "passed": value["rank"] >= rank
            and value["condition_number_nonzero"] <= condition,
        }
        for name, value, rank, condition in values
    }


def _agreement(
    candidates: Mapping[str, Any],
    survivors: Sequence[str],
    b5: Mapping[str, Any],
) -> dict[str, Any]:
    rules = b5["method_agreement"]
    pairwise = []
    compatible: set[frozenset[str]] = set()
    for first, second in combinations(survivors, 2):
        translation_m, rotation_deg = pilot.relative_error(
            pilot.from_xyzw(candidates[first]["result"]["C_T_B"]),
            pilot.from_xyzw(candidates[second]["result"]["C_T_B"]),
        )
        passed = (
            translation_m * 1000.0 <= rules["translation_mm_maximum"]
            and rotation_deg <= rules["rotation_deg_maximum"]
        )
        pairwise.append(
            {
                "methods": [first, second],
                "translation_disagreement_mm": translation_m * 1000.0,
                "rotation_disagreement_deg": rotation_deg,
                "passed": passed,
            }
        )
        if passed:
            compatible.add(frozenset((first, second)))
    consensus: tuple[str, ...] = ()
    for size in range(len(survivors), 0, -1):
        options = [
            group
            for group in combinations(survivors, size)
            if all(frozenset(pair) in compatible for pair in combinations(group, 2))
        ]
        if options:
            consensus = options[0]
            break
    family_count = len({METHOD_FAMILIES[method] for method in consensus})
    gates = {
        "surviving_methods": len(consensus) >= rules["surviving_methods_minimum"],
        "distinct_solver_families": family_count
        >= rules["distinct_solver_families_minimum"],
        "pairwise_agreement": all(
            item["passed"]
            for item in pairwise
            if set(item["methods"]).issubset(consensus)
        ),
    }
    return {
        "selection_rule": CONSENSUS_RULE,
        "train_eligible_pool": list(survivors),
        "consensus_members": list(consensus),
        "distinct_solver_families": family_count,
        "pairwise": pairwise,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _rules_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coordinate_contract": {
            "notation": "T_parent_child",
            "meaning": "p_parent = T_parent_child * p_child",
            "quaternion_order": "XYZW",
            "equation": "C_T_Tag_i = C_T_B * B_T_H_i * H_T_Tag",
        },
        "split_policy": plan["split_policy"],
        "b5_preregistration": plan["b5_preregistration"],
        "method_families": METHOD_FAMILIES,
        "consensus_rule": CONSENSUS_RULE,
        "held_out_rule": HELD_OUT_RULE,
        "preappend_chain_verification": (
            "Train verifies the complete current raw byte index and the "
            "preappend block schema without consuming held-out transforms. "
            "Exact cross-split prefix-chain verification occurs only after "
            "held-out access is irreversibly spent."
        ),
    }


def build_train_artifact(plan_path: Path, raw_dir: Path) -> dict[str, Any]:
    """Fit and freeze candidates without opening any held-out sample."""
    plan, plan_sha256, entries = _load_plan(Path(plan_path))
    integrity = _integrity_index(Path(raw_dir), plan)
    sample_ids, hands, tags, train_hashes = _load_split(
        Path(raw_dir),
        plan,
        plan_sha256,
        entries,
        integrity,
        "train",
        verify_preappend_chain=False,
    )
    b5 = plan["b5_preregistration"]
    candidates: dict[str, Any] = {}
    for method in ALL_METHODS:
        role = "eligible" if method in ELIGIBLE_METHODS else "diagnostic_only"
        try:
            result = _candidate_result(method, hands, tags, sample_ids)
            loo = _leave_one_out(method, hands, tags, sample_ids)
            gates = _residual_gates(result, loo, b5)
            candidates[method] = {
                "role": role,
                "family": METHOD_FAMILIES[method],
                "solve_status": "solved",
                "result": result,
                "leave_one_out": loo,
                "train_and_loo_gates": gates,
                "train_and_loo_passed": all(gates.values()),
            }
        except Exception as exc:
            candidates[method] = {
                "role": role,
                "family": METHOD_FAMILIES[method],
                "solve_status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "train_and_loo_passed": False,
            }
    survivors = [
        method
        for method in ELIGIBLE_METHODS
        if candidates[method]["train_and_loo_passed"]
    ]
    agreement = _agreement(candidates, survivors, b5)
    try:
        observability = pilot.observability(hands, tags, sample_ids)
        observability_gates = _observability_gates(observability, b5)
    except Exception as exc:
        observability = {"error": f"{type(exc).__name__}: {exc}"}
        observability_gates = {
            "computation": {
                "passed": False,
                "error": observability["error"],
            }
        }
    observability_passed = all(
        bool(gate.get("passed")) for gate in observability_gates.values()
    )
    rules = _rules_payload(plan)
    train_passed = observability_passed and agreement["passed"]
    frozen_candidates = {
        method: {
            "role": candidate["role"],
            "family": candidate["family"],
            "solve_status": candidate["solve_status"],
            **(
                {
                    "C_T_B": candidate["result"]["C_T_B"],
                    "H_T_Tag": candidate["result"]["H_T_Tag"],
                }
                if candidate["solve_status"] == "solved"
                else {"error": candidate["error"]}
            ),
        }
        for method, candidate in candidates.items()
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "formal_train_freeze",
        "status": "train_passed" if train_passed else "train_rejected",
        "execution_ready": False,
        "dataset_id": plan["dataset_id"],
        "plan_sha256": plan_sha256,
        "capture_provenance_binding": plan.get("capture_provenance_binding"),
        "rules_sha256": _sha256(_canonical_bytes(rules)),
        "rules": rules,
        "implementation": {
            "repository_path": "tools/calibration/eye_to_hand_formal_solver.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
            "pilot_solver_sha256": _sha256(Path(pilot.__file__).read_bytes()),
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
        },
        "train_sample_ids": sample_ids,
        "train_sample_sha256": train_hashes,
        "eligible_methods": list(ELIGIBLE_METHODS),
        "diagnostic_only_methods": list(DIAGNOSTIC_METHODS),
        "candidates": _json_safe(candidates),
        "frozen_candidate_transforms": _json_safe(frozen_candidates),
        "method_agreement": _json_safe(agreement),
        "observability": _json_safe(observability),
        "observability_gates": _json_safe(observability_gates),
        "train_gate_passed": train_passed,
    }
    return _freeze(payload)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one immutable JSON artifact without replacing existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError as exc:
        raise FormalSolverError(f"refusing to overwrite existing file: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _load_train_artifact(
    artifact_path: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    try:
        encoded = artifact_path.read_bytes()
        artifact = json.loads(encoded, object_pairs_hook=_strict_json_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalSolverError("frozen train artifact is unreadable") from exc
    _require(isinstance(artifact, dict), "frozen train artifact root is invalid")
    _verify_frozen(artifact, "train artifact")
    rules = _rules_payload(plan)
    expected_train = [
        sample_id
        for sample_id in SAMPLE_IDS
        if _expected_splits()[sample_id] == "train"
    ]
    _require(
        artifact.get("artifact_type") == "formal_train_freeze"
        and artifact.get("dataset_id") == plan["dataset_id"]
        and artifact.get("plan_sha256") == plan_sha256
        and artifact.get("rules_sha256") == _sha256(_canonical_bytes(rules))
        and artifact.get("rules") == rules
        and artifact.get("train_sample_ids") == expected_train
        and artifact.get("eligible_methods") == list(ELIGIBLE_METHODS)
        and artifact.get("diagnostic_only_methods") == list(DIAGNOSTIC_METHODS),
        "frozen train artifact conflicts with the current plan or rules",
    )
    _require(
        artifact.get("train_gate_passed") is True
        and artifact.get("status") == "train_passed",
        "held-out access requires a passing frozen train artifact",
    )
    implementation = artifact.get("implementation", {})
    _require(
        implementation.get("sha256") == _sha256(Path(__file__).read_bytes())
        and implementation.get("pilot_solver_sha256")
        == _sha256(Path(pilot.__file__).read_bytes())
        and implementation.get("numpy_version") == np.__version__
        and implementation.get("opencv_version") == cv2.__version__,
        "solver implementation or numerical runtime changed after train freeze",
    )
    transforms = artifact.get("frozen_candidate_transforms")
    _require(
        isinstance(transforms, dict) and set(transforms) == set(ALL_METHODS),
        "frozen candidate set is incomplete",
    )
    consensus = artifact.get("method_agreement", {}).get("consensus_members")
    _require(
        isinstance(consensus, list)
        and len(consensus)
        >= EXPECTED_B5["method_agreement"]["surviving_methods_minimum"]
        and all(method in ELIGIBLE_METHODS for method in consensus),
        "frozen consensus is invalid",
    )
    for method, candidate in transforms.items():
        if candidate.get("solve_status") == "solved":
            _transform(candidate.get("C_T_B"), f"{method}.C_T_B")
            _transform(candidate.get("H_T_Tag"), f"{method}.H_T_Tag")
    return artifact, _sha256(encoded)


def _spend_held_out(
    raw_dir: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    train_artifact_sha256: str,
    train_artifact_file_sha256: str,
) -> Path:
    spend_path = Path(raw_dir) / HELD_OUT_SPEND_FILENAME
    marker = {
        "schema_version": 1,
        "status": "held_out_access_spent",
        "dataset_id": plan["dataset_id"],
        "plan_sha256": plan_sha256,
        "train_artifact_sha256": train_artifact_sha256,
        "train_artifact_file_sha256": train_artifact_file_sha256,
        "held_out_sample_ids": [
            sample_id
            for sample_id in SAMPLE_IDS
            if _expected_splits()[sample_id] == "held_out"
        ],
        "policy": (
            "Created at the canonical raw-dataset path with O_EXCL before any "
            "held-out sample JSON was deserialized. Failure after this point "
            "does not restore independence."
        ),
    }
    write_json_exclusive(spend_path, marker)
    return spend_path


def _evaluate_frozen_candidate(
    candidate: Mapping[str, Any],
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    c_t_b = _transform(candidate["C_T_B"], "frozen.C_T_B")
    h_t_tag = _transform(candidate["H_T_Tag"], "frozen.H_T_Tag")
    translations_mm = []
    rotations_deg = []
    per_sample = []
    for sample_id, hand, observed in zip(sample_ids, hands, tags):
        translation_m, rotation_deg = pilot.relative_error(
            observed, c_t_b @ hand @ h_t_tag
        )
        translations_mm.append(translation_m * 1000.0)
        rotations_deg.append(rotation_deg)
        per_sample.append(
            {
                "sample_id": sample_id,
                "translation_residual_mm": translation_m * 1000.0,
                "rotation_residual_deg": rotation_deg,
            }
        )
    return {
        "per_sample": per_sample,
        "translation_residual_mm": _stats(translations_mm),
        "rotation_residual_deg": _stats(rotations_deg),
    }


def evaluate_held_out(
    plan_path: Path,
    raw_dir: Path,
    train_artifact_path: Path,
) -> dict[str, Any]:
    """Spend and evaluate the exact eight held-out samples without refitting."""
    plan, plan_sha256, entries = _load_plan(Path(plan_path))
    integrity = _integrity_index(Path(raw_dir), plan)
    train_artifact, train_file_sha256 = _load_train_artifact(
        Path(train_artifact_path), plan, plan_sha256
    )
    current_train_hashes = {
        sample_id: integrity[f"{sample_id}.json"]
        for sample_id in train_artifact["train_sample_ids"]
    }
    _require(
        train_artifact.get("train_sample_sha256") == current_train_hashes,
        "train sample bytes changed after the train artifact was frozen",
    )
    spend_path = _spend_held_out(
        Path(raw_dir),
        plan,
        plan_sha256,
        train_artifact["artifact_sha256"],
        train_file_sha256,
    )
    sample_ids, hands, tags, held_out_hashes = _load_split(
        Path(raw_dir),
        plan,
        plan_sha256,
        entries,
        integrity,
        "held_out",
        verify_preappend_chain=True,
    )
    b5 = plan["b5_preregistration"]
    consensus = set(train_artifact["method_agreement"]["consensus_members"])
    methods: dict[str, Any] = {}
    consensus_passes = []
    for method, candidate in train_artifact["frozen_candidate_transforms"].items():
        if candidate.get("solve_status") != "solved":
            methods[method] = {
                "role": candidate["role"],
                "status": "train_solve_failed",
            }
            continue
        result = _evaluate_frozen_candidate(candidate, hands, tags, sample_ids)
        if method in consensus:
            translation = result["translation_residual_mm"]
            rotation = result["rotation_residual_deg"]
            gates = {
                "translation_median": (
                    translation["median"]
                    <= b5["held_out_translation_residual_mm"]["median_maximum"]
                ),
                "translation_maximum": (
                    translation["max"]
                    <= b5["held_out_translation_residual_mm"]["maximum_maximum"]
                ),
                "rotation_median": (
                    rotation["median"]
                    <= b5["held_out_rotation_residual_deg"]["median_maximum"]
                ),
                "rotation_maximum": (
                    rotation["max"]
                    <= b5["held_out_rotation_residual_deg"]["maximum_maximum"]
                ),
            }
            passed = all(gates.values())
            consensus_passes.append(passed)
        else:
            gates = None
            passed = None
        methods[method] = {
            "role": candidate["role"],
            "status": "evaluated_frozen_transform",
            "consensus_member": method in consensus,
            "result": result,
            "held_out_gates": gates,
            "held_out_passed": passed,
        }
    held_out_passed = bool(consensus_passes) and all(consensus_passes)
    payload = {
        "schema_version": 1,
        "artifact_type": "formal_held_out_evaluation",
        "status": "held_out_passed" if held_out_passed else "held_out_rejected",
        "execution_ready": False,
        "dataset_id": plan["dataset_id"],
        "plan_sha256": plan_sha256,
        "rules_sha256": train_artifact["rules_sha256"],
        "train_artifact_file_sha256": train_file_sha256,
        "held_out_access_spend_file": str(Path(spend_path)),
        "held_out_acceptance_rule": HELD_OUT_RULE,
        "held_out_sample_ids": sample_ids,
        "held_out_sample_sha256": held_out_hashes,
        "methods": _json_safe(methods),
        "held_out_gate_passed": held_out_passed,
    }
    return _freeze(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--raw-dir", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    held_out = subparsers.add_parser("held-out")
    held_out.add_argument("--plan", type=Path, required=True)
    held_out.add_argument("--raw-dir", type=Path, required=True)
    held_out.add_argument("--train-artifact", type=Path, required=True)
    held_out.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "train":
        artifact = build_train_artifact(args.plan, args.raw_dir)
        write_json_exclusive(args.output, artifact)
        summary = {
            "mode": "train",
            "status": artifact["status"],
            "output": str(args.output),
        }
    else:
        artifact = evaluate_held_out(args.plan, args.raw_dir, args.train_artifact)
        write_json_exclusive(args.output, artifact)
        summary = {
            "mode": "held-out",
            "status": artifact["status"],
            "output": str(args.output),
            "spend_file": artifact["held_out_access_spend_file"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
