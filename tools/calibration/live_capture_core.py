"""Pure, fail-closed helpers for the calibration LIVE one-shot Sample button.

This module deliberately has no ROS, OpenCV, or subprocess imports so its plan,
inventory, command, and local precheck behavior can be unit tested on a
development machine without touching calibration hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping


FORMAL_SAMPLE_IDS = tuple(f"S{index:04d}" for index in range(12, 42))
FORMAL_HELD_OUT_ORDERS = (4, 8, 12, 16, 20, 24, 27, 30)
FREEFORM_POSE_MODE = "freeform_diverse"
ALLOWED_SPLITS = frozenset(("train", "held_out"))
SAMPLE_ID_PATTERN = re.compile(r"^S(\d{4})$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RGB_MAX_AGE_S = 1.0
# The AGX calibration-only stack normally runs Tag at about 14 Hz, but a
# measured scheduler/USB gap can occasionally reach roughly 1.5 s. Keep the
# UI stable across
# that transient; the recorder still enforces its stricter 1 s static window
# and 250 ms maximum inter-observation gap before writing a sample.
TAG_MAX_AGE_S = 2.0
TF_MAX_AGE_S = 2.0
FEEDBACK_MAX_AGE_S = 1.0
MIN_DECISION_MARGIN = 50.0
AUTO_CLEAR_TORQUE_FAULT = "one or more joint torque switches are off"
AUTO_CLEAR_JOINT_LIMITS_FAULT = (
    "invalid feedback: feedback violates plausible joint limits"
)
AUTO_CLEAR_FAULTS = frozenset(
    (
        AUTO_CLEAR_TORQUE_FAULT,
        AUTO_CLEAR_JOINT_LIMITS_FAULT,
    )
)
CLEAR_FAULT_FEEDBACK_TIMEOUT_S = 2.0
REQUIRED_DRIVER_PROVENANCE_KEYS = frozenset(
    (
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
    )
)


class CapturePlanError(ValueError):
    """The formal capture plan is absent, malformed, or internally unsafe."""


@dataclass(frozen=True)
class CapturePlanEntry:
    sample_id: str
    pose_index: int
    split: str
    pose_mode: str
    pose_intent: str
    operator_prompt: str | None


@dataclass(frozen=True)
class CapturePlan:
    schema_version: str
    plan_status: str
    dataset_id: str
    dataset_dir: Path | None
    ros_domain_id: int
    status_source: str
    capture_enabled: bool
    user_approved: bool
    formal_capture_prerequisites_frozen: bool
    capture_timeout_s: float
    source_hashes_sha256: tuple[tuple[str, str], ...]
    capture_order: tuple[CapturePlanEntry, ...]
    # Kept with the loaded plan so retrospective inventory checks apply the
    # exact frozen capture contract rather than a legacy global default.
    tag_static_window_sample_count_minimum: int
    tag_static_window_maximum_gap_ns: int


@dataclass(frozen=True)
class Inventory:
    completed_sample_ids: frozenset[str]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ClearFaultVerification:
    response_received: float
    deadline: float
    pre_clear_feedback_stamp_ns: int

    @classmethod
    def from_successful_response(
        cls, response_received: float, pre_clear_feedback_stamp_ns: int
    ) -> "ClearFaultVerification":
        return cls(
            response_received=response_received,
            deadline=response_received + CLEAR_FAULT_FEEDBACK_TIMEOUT_S,
            pre_clear_feedback_stamp_ns=pre_clear_feedback_stamp_ns,
        )

    def feedback_verifies(
        self, feedback_received: float, feedback: object
    ) -> bool:
        return (
            feedback_received > self.response_received
            and isinstance(feedback, Mapping)
            and isinstance(feedback.get("feedback_stamp_ns"), int)
            and not isinstance(feedback.get("feedback_stamp_ns"), bool)
            and feedback["feedback_stamp_ns"] > self.pre_clear_feedback_stamp_ns
            and feedback.get("state") == "FAULT_CLEARED_DISARMED"
            and feedback.get("fault") == ""
            and isinstance(feedback.get("torque_enabled"), list)
            and len(feedback["torque_enabled"]) == 6
            and all(value is True for value in feedback["torque_enabled"])
            and feedback.get("armed") is False
            and feedback.get("dry_run") is False
            and isinstance(feedback.get("voltage"), (int, float))
            and not isinstance(feedback.get("voltage"), bool)
            and math.isfinite(float(feedback["voltage"]))
            and 11.0 <= float(feedback["voltage"]) <= 13.0
        )

    def expired(self, now: float) -> bool:
        return now >= self.deadline


def _required(mapping: Mapping, key: str):
    if key not in mapping:
        raise CapturePlanError(f"capture plan is missing {key!r}")
    return mapping[key]


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CapturePlanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _required_bool(mapping: Mapping, key: str) -> bool:
    value = _required(mapping, key)
    if not isinstance(value, bool):
        raise CapturePlanError(f"capture plan {key!r} must be a boolean")
    return value


def validate_capture_plan(payload: object) -> CapturePlan:
    """Validate the exact formal S0012-S0041 freeform capture plan schema.

    A proposed/disabled plan is valid data, but ``capture_block_reasons`` keeps
    it non-executable. This lets the LIVE view show the proposed next pose
    without accidentally treating proposal review as capture authorization.
    """

    if not isinstance(payload, Mapping):
        raise CapturePlanError("capture plan root must be a mapping")
    schema_version_raw = _required(payload, "schema_version")
    if (
        isinstance(schema_version_raw, bool)
        or not isinstance(schema_version_raw, (str, int))
        or (isinstance(schema_version_raw, str) and not schema_version_raw.strip())
        or (isinstance(schema_version_raw, int) and schema_version_raw < 1)
    ):
        raise CapturePlanError(
            "capture plan schema_version must be a positive integer or non-empty string"
        )
    schema_version = str(schema_version_raw).strip()
    plan_status = _required(payload, "plan_status")
    if plan_status not in ("proposed", "frozen"):
        raise CapturePlanError("capture plan plan_status must be proposed or frozen")
    dataset_id = _required(payload, "dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise CapturePlanError("capture plan dataset_id must be a non-empty string")
    dataset_dir_raw = payload.get("dataset_dir")
    if dataset_dir_raw is not None and (
        not isinstance(dataset_dir_raw, str) or not dataset_dir_raw.strip()
    ):
        raise CapturePlanError(
            "capture plan dataset_dir must be null or a non-empty string"
        )
    dataset_dir = (
        None
        if dataset_dir_raw is None
        else Path(dataset_dir_raw).expanduser().resolve()
    )
    ros_domain_id = _required(payload, "ros_domain_id")
    if (
        isinstance(ros_domain_id, bool)
        or not isinstance(ros_domain_id, int)
        or ros_domain_id < 0
    ):
        raise CapturePlanError(
            "capture plan ros_domain_id must be a non-negative integer"
        )
    status_source = _required(payload, "status_source")
    if status_source != "safe_driver":
        raise CapturePlanError("formal capture status_source must be safe_driver")

    capture_enabled = _required_bool(payload, "capture_enabled")
    user_approved = _required_bool(payload, "user_approved")
    prerequisites_frozen = _required_bool(
        payload, "formal_capture_prerequisites_frozen"
    )
    operator_contract = _required(payload, "operator_contract")
    if not isinstance(operator_contract, Mapping):
        raise CapturePlanError("capture plan operator_contract must be a mapping")
    capture_timeout_s = _required(operator_contract, "capture_timeout_s")
    if (
        isinstance(capture_timeout_s, bool)
        or not isinstance(capture_timeout_s, (int, float))
        or not math.isfinite(float(capture_timeout_s))
        or float(capture_timeout_s) != 15.0
    ):
        raise CapturePlanError(
            "capture plan operator_contract.capture_timeout_s must equal 15.0"
        )
    source_hashes = _required(payload, "source_hashes_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise CapturePlanError(
            "capture plan source_hashes_sha256 must be a non-empty mapping"
        )
    normalized_source_hashes = []
    for source_name, digest in source_hashes.items():
        if (
            not isinstance(source_name, str)
            or not source_name.strip()
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
        ):
            raise CapturePlanError(
                "capture plan source_hashes_sha256 entries must be named "
                "lowercase SHA-256 digests"
            )
        normalized_source_hashes.append((source_name, digest))
    missing_driver_sources = REQUIRED_DRIVER_PROVENANCE_KEYS - source_hashes.keys()
    if missing_driver_sources:
        raise CapturePlanError(
            "capture plan source hashes are missing required driver provenance"
        )

    # Historical unit-test plans predate this explicit contract.  Their
    # established 15-sample / 250-ms behaviour remains the fallback, while
    # formal plans carry their own frozen values (r6 uses 10 / 250 ms).
    tag_sample_minimum = 15
    tag_gap_maximum_ns = 250_000_000
    measurement_contract = payload.get("measurement_contract")
    if measurement_contract is not None:
        if not isinstance(measurement_contract, Mapping):
            raise CapturePlanError("capture plan measurement_contract must be a mapping")
        tag_window = measurement_contract.get("tag_static_window")
        if not isinstance(tag_window, Mapping):
            raise CapturePlanError("capture plan tag_static_window must be a mapping")
        raw_sample_minimum = tag_window.get("sample_count_minimum")
        raw_gap_ms = tag_window.get("sample_gap_ms_maximum")
        if (
            isinstance(raw_sample_minimum, bool)
            or not isinstance(raw_sample_minimum, int)
            or raw_sample_minimum < 1
        ):
            raise CapturePlanError(
                "capture plan tag_static_window.sample_count_minimum must be positive"
            )
        if (
            isinstance(raw_gap_ms, bool)
            or not isinstance(raw_gap_ms, (int, float))
            or not math.isfinite(float(raw_gap_ms))
            or float(raw_gap_ms) <= 0.0
        ):
            raise CapturePlanError(
                "capture plan tag_static_window.sample_gap_ms_maximum must be positive"
            )
        tag_sample_minimum = raw_sample_minimum
        tag_gap_maximum_ns = int(float(raw_gap_ms) * 1_000_000)
    entries_raw = _required(payload, "capture_order")
    if not isinstance(entries_raw, list):
        raise CapturePlanError("capture plan capture_order must be a list")

    entries: list[CapturePlanEntry] = []
    for order_index, item in enumerate(entries_raw):
        prefix = f"capture_order[{order_index}]"
        if not isinstance(item, Mapping):
            raise CapturePlanError(f"{prefix} must be a mapping")
        sample_id = _required(item, "sample_id")
        pose_index = _required(item, "pose_index")
        split = _required(item, "split")
        pose_mode = _required(item, "pose_mode")
        pose_intent = _required(item, "pose_intent")
        operator_prompt = item.get("operator_prompt")
        if operator_prompt is not None and (
            not isinstance(operator_prompt, str) or not operator_prompt.strip()
        ):
            raise CapturePlanError(
                f"{prefix}.operator_prompt must be null or a non-empty string"
            )
        for prohibited_target in ("image_target", "orientation_target"):
            if prohibited_target in item:
                raise CapturePlanError(
                    f"{prefix}.{prohibited_target} is prohibited in freeform mode"
                )
        if not isinstance(sample_id, str) or not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise CapturePlanError(f"{prefix}.sample_id must match Sdddd")
        if (
            isinstance(pose_index, bool)
            or not isinstance(pose_index, int)
            or pose_index < 1
        ):
            raise CapturePlanError(f"{prefix}.pose_index must be a positive integer")
        if split not in ALLOWED_SPLITS:
            raise CapturePlanError(f"{prefix}.split must be train or held_out")
        if pose_mode != FREEFORM_POSE_MODE:
            raise CapturePlanError(
                f"{prefix}.pose_mode must be {FREEFORM_POSE_MODE!r}"
            )
        if not isinstance(pose_intent, str) or not pose_intent.strip():
            raise CapturePlanError(f"{prefix}.pose_intent must be a non-empty string")
        entries.append(
            CapturePlanEntry(
                sample_id=sample_id,
                pose_index=pose_index,
                split=split,
                pose_mode=pose_mode,
                pose_intent=pose_intent.strip(),
                operator_prompt=(
                    None if operator_prompt is None else operator_prompt.strip()
                ),
            )
        )

    ids = [entry.sample_id for entry in entries]
    pose_indices = [entry.pose_index for entry in entries]
    if len(entries) != 30:
        raise CapturePlanError("formal capture_order must contain exactly 30 entries")
    if len(set(ids)) != len(ids):
        raise CapturePlanError(
            "formal capture_order contains duplicate sample_id values"
        )
    if ids != list(FORMAL_SAMPLE_IDS):
        raise CapturePlanError(
            "formal sample IDs must be ordered exactly S0012-S0041"
        )
    if len(set(pose_indices)) != len(pose_indices):
        raise CapturePlanError(
            "formal capture_order contains duplicate pose_index values"
        )
    train_count = sum(entry.split == "train" for entry in entries)
    held_out_count = sum(entry.split == "held_out" for entry in entries)
    if (train_count, held_out_count) != (22, 8):
        raise CapturePlanError(
            "formal capture_order must contain exactly 22 train and 8 held_out entries"
        )
    held_out_orders = tuple(
        index
        for index, entry in enumerate(entries, start=1)
        if entry.split == "held_out"
    )
    if held_out_orders != FORMAL_HELD_OUT_ORDERS:
        raise CapturePlanError(
            "held_out entries must be at capture orders 4, 8, 12, 16, "
            "20, 24, 27, and 30"
        )

    return CapturePlan(
        schema_version=schema_version,
        plan_status=plan_status,
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        ros_domain_id=ros_domain_id,
        status_source=status_source,
        capture_enabled=capture_enabled,
        user_approved=user_approved,
        formal_capture_prerequisites_frozen=prerequisites_frozen,
        capture_timeout_s=float(capture_timeout_s),
        source_hashes_sha256=tuple(sorted(normalized_source_hashes)),
        capture_order=tuple(entries),
        tag_static_window_sample_count_minimum=tag_sample_minimum,
        tag_static_window_maximum_gap_ns=tag_gap_maximum_ns,
    )


def load_capture_plan(path: Path) -> CapturePlan:
    """Load JSON or YAML, with YAML kept optional for non-ROS test hosts."""

    path = Path(path).expanduser().resolve()
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(raw, object_pairs_hook=_strict_json_object)
    elif path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CapturePlanError(
                "loading YAML capture plans requires PyYAML on this host"
            ) from exc
        payload = yaml.safe_load(raw)
    else:
        raise CapturePlanError("capture plan must use .json, .yaml, or .yml")
    return validate_capture_plan(payload)


def capture_block_reasons(
    plan: CapturePlan, dataset_dir: Path | None
) -> tuple[str, ...]:
    reasons = []
    if plan.plan_status != "frozen":
        reasons.append("plan is proposed, not frozen")
    if not plan.capture_enabled:
        reasons.append("capture_enabled is false")
    if not plan.user_approved:
        reasons.append("user_approved is false")
    if not plan.formal_capture_prerequisites_frozen:
        reasons.append("formal prerequisites remain pending")
    if dataset_dir is None:
        reasons.append("raw dataset directory is not configured")
    elif Path(dataset_dir).name != "raw":
        reasons.append("dataset directory must be the immutable raw directory")
    return tuple(reasons)


def _canonical_integrity_index_hashes(
    dataset_id: str,
    sample_hashes: Mapping[str, str],
) -> tuple[str, str]:
    entries = [
        {"file": filename, "sha256": digest}
        for filename, digest in sample_hashes.items()
    ]
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "raw_samples": entries,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    sums_bytes = "".join(
        f"{entry['sha256']}  {entry['file']}\n" for entry in entries
    ).encode("utf-8")
    return (
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(sums_bytes).hexdigest(),
    )


def _formal_sample_issue(
    document: Mapping,
    entry: CapturePlanEntry,
    plan: CapturePlan,
    capture_plan_sha256: str,
    expected_prior_hashes: Mapping[str, str],
) -> str | None:
    def valid_transform(value):
        if not isinstance(value, Mapping):
            return False
        translation = value.get("translation_m")
        rotation = value.get("rotation_xyzw")
        if (
            not isinstance(translation, list)
            or len(translation) != 3
            or not isinstance(rotation, list)
            or len(rotation) != 4
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                for number in (*translation, *rotation)
            )
        ):
            return False
        norm = math.sqrt(sum(float(number) ** 2 for number in rotation))
        return abs(norm - 1.0) <= 1e-3

    if document.get("capture_plan_sha256") != capture_plan_sha256:
        return "capture_plan_sha256 conflicts with the loaded plan"
    if (
        document.get("coordinate_contract_version") != "1.0.0"
        or document.get("calibration_type") != "eye_on_base"
    ):
        return "coordinate or calibration contract is missing"
    if document.get("settled") is not True:
        return "settled proof is missing"
    if document.get("torque_enabled") != [True] * 6:
        return "top-level six-axis Torque proof is missing"
    torque_evidence = document.get("torque_evidence")
    if (
        not isinstance(torque_evidence, Mapping)
        or torque_evidence.get("direct_joint_torque_flags_available") is not True
        or torque_evidence.get("armed") is not False
        or torque_evidence.get("dry_run") is not False
        or torque_evidence.get("fault") != ""
    ):
        return "top-level safe-driver evidence is invalid"
    rgb_stamp = document.get("rgb_header_stamp_ns")
    joint_stamp = document.get("joint_states_stamp_ns")
    if (
        isinstance(rgb_stamp, bool)
        or not isinstance(rgb_stamp, int)
        or rgb_stamp <= 0
        or isinstance(joint_stamp, bool)
        or not isinstance(joint_stamp, int)
        or joint_stamp <= 0
        or abs(rgb_stamp - joint_stamp) > 50_000_000
    ):
        return "RGB/joint timestamp evidence is invalid"
    if not valid_transform(document.get("T_roarm_base_link_hand_tcp")) or not valid_transform(
        document.get("T_camera_optical_frame_calibration_tag")
    ):
        return "calibration transform evidence is invalid"
    quality = document.get("tag_detection_quality")
    if (
        document.get("tag_family") != "tag36h11"
        or document.get("tag_id") != 0
        or not isinstance(quality, Mapping)
        or quality.get("hamming") != 0
        or not isinstance(quality.get("decision_margin"), (int, float))
        or float(quality["decision_margin"]) < 50.0
    ):
        return "Tag identity or quality evidence is invalid"

    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        return "formal sample metadata is missing"
    if metadata.get("pose_index") != entry.pose_index:
        return "pose_index conflicts with capture plan"
    if metadata.get("split") != entry.split:
        return "split conflicts with capture plan"
    if metadata.get("capture_plan_sha256") != capture_plan_sha256:
        return "metadata capture_plan_sha256 conflicts with the loaded plan"
    if metadata.get("recorder_timeout_s") != plan.capture_timeout_s:
        return "recorder timeout conflicts with capture plan"
    if metadata.get("status_source") != "safe_driver":
        return "safe-driver status provenance is missing"
    if metadata.get("ros_domain_id") != plan.ros_domain_id:
        return "ROS domain provenance conflicts with capture plan"
    authorization_id = metadata.get("operator_authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id:
        return "operator authorization provenance is missing"

    expected_sources = dict(plan.source_hashes_sha256)
    if metadata.get("frozen_source_hashes_sha256") != expected_sources:
        return "frozen source hashes conflict with capture plan"
    driver_versions = metadata.get("driver_versions")
    if not isinstance(driver_versions, Mapping):
        return "driver version provenance is missing"
    for key in REQUIRED_DRIVER_PROVENANCE_KEYS:
        if driver_versions.get(key) != expected_sources.get(key):
            return f"driver version provenance conflicts for {key}"

    preappend = metadata.get("preappend_raw_integrity")
    if not isinstance(preappend, Mapping):
        return "pre-append raw integrity proof is missing"
    prior_hashes = preappend.get("existing_sample_hashes_sha256")
    prior_count = preappend.get("existing_sample_count")
    if (
        isinstance(prior_count, bool)
        or not isinstance(prior_count, int)
        or prior_count < 0
        or not isinstance(prior_hashes, Mapping)
        or len(prior_hashes) != prior_count
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"S[0-9]{4}\.json", name)
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
            for name, digest in prior_hashes.items()
        )
    ):
        return "pre-append raw integrity proof is invalid"
    if prior_count != len(expected_prior_hashes) or dict(prior_hashes) != dict(
        expected_prior_hashes
    ):
        return "pre-append raw sample hash chain conflicts with the actual prefix"
    expected_manifest_sha256, expected_sums_sha256 = (
        _canonical_integrity_index_hashes(
            plan.dataset_id,
            expected_prior_hashes,
        )
    )
    # Before the first sample the two index files may legitimately not exist;
    # the recorder records that fact as a pair of nulls.  Once any prior sample
    # exists, both hashes remain mandatory and exact.
    indexes_were_absent = (
        not expected_prior_hashes
        and preappend.get("manifest_sha256") is None
        and preappend.get("sha256sums_sha256") is None
    )
    if not indexes_were_absent:
        if preappend.get("manifest_sha256") != expected_manifest_sha256:
            return "pre-append manifest hash conflicts with the actual prefix"
        if preappend.get("sha256sums_sha256") != expected_sums_sha256:
            return "pre-append SHA256SUMS hash conflicts with the actual prefix"

    tag_window = metadata.get("tag_static_window")
    if not isinstance(tag_window, Mapping):
        return "Tag static-window evidence is missing"
    tag_observations = tag_window.get("observations")
    if (
        not isinstance(tag_window.get("sample_count"), int)
        or tag_window["sample_count"] < plan.tag_static_window_sample_count_minimum
        or not isinstance(tag_observations, list)
        or len(tag_observations) != tag_window["sample_count"]
        or not isinstance(tag_window.get("duration_ns"), int)
        or tag_window["duration_ns"] < 1_000_000_000
        or not isinstance(tag_window.get("maximum_gap_ns"), int)
        or tag_window["maximum_gap_ns"] > plan.tag_static_window_maximum_gap_ns
        or not isinstance(tag_window.get("translation_spread_m"), (int, float))
        or float(tag_window["translation_spread_m"]) > 0.003
        or not isinstance(tag_window.get("rotation_spread_deg"), (int, float))
        or float(tag_window["rotation_spread_deg"]) > 1.5
    ):
        return "Tag static-window evidence is invalid"

    joint_window = metadata.get("joint_safe_window")
    if not isinstance(joint_window, Mapping):
        return "joint/safe-status window evidence is missing"
    joint_observations = joint_window.get("observations")
    if (
        not isinstance(joint_window.get("sample_count"), int)
        or joint_window["sample_count"] < 20
        or not isinstance(joint_observations, list)
        or len(joint_observations) != joint_window["sample_count"]
        or not isinstance(joint_window.get("duration_ns"), int)
        or joint_window["duration_ns"] < 1_000_000_000
        or not isinstance(joint_window.get("maximum_gap_ns"), int)
        or joint_window["maximum_gap_ns"] > 100_000_000
        or joint_window.get("safe_status_alignment")
        != "all_frames_aligned_healthy"
    ):
        return "joint/safe-status window evidence is invalid"
    for observation in joint_observations:
        if (
            not isinstance(observation, Mapping)
            or observation.get("torque_enabled") != [True] * 6
            or not isinstance(observation.get("positions_rad"), list)
            or len(observation["positions_rad"]) != 6
            or not isinstance(observation.get("voltage_v"), (int, float))
            or not 11.0 <= float(observation["voltage_v"]) <= 13.0
        ):
            return "per-frame safe-status evidence is invalid"
    return None


def _raw_integrity_state(
    dataset_dir: Path,
    plan: CapturePlan,
) -> tuple[tuple[str, ...], dict[str, str] | None]:
    sample_paths = sorted(path for path in dataset_dir.glob("S*.json"))
    if any(
        not re.fullmatch(r"S[0-9]{4}\.json", path.name)
        or not path.is_file()
        or path.is_symlink()
        for path in sample_paths
    ):
        return (("raw sample filename or file type is invalid",), None)
    if any(
        (
            *dataset_dir.glob(".S*.json.*"),
            *dataset_dir.glob(".manifest.json.*"),
            *dataset_dir.glob(".SHA256SUMS.*"),
        )
    ):
        return (("stale raw integrity transaction artifact exists",), None)
    manifest_path = dataset_dir / "manifest.json"
    sums_path = dataset_dir / "SHA256SUMS"
    # A separately frozen repeat starts with an empty raw directory. Requiring
    # r2's historical prefix here would mix evidence between independent runs.
    if not sample_paths and not manifest_path.exists() and not sums_path.exists():
        return ((), {})
    expected_files = [path.name for path in sample_paths]
    historical_files = [
        name for name in expected_files if int(name[1:5]) < 12
    ]
    formal_files = [
        name for name in expected_files if int(name[1:5]) >= 12
    ]
    complete_historical_prefix = [
        f"S{index:04d}.json" for index in range(1, 12)
    ]
    if historical_files not in ([], complete_historical_prefix):
        return (("historical S0001-S0011 integrity prefix is incomplete",), None)
    if formal_files != [
        f"{sample_id}.json" for sample_id in FORMAL_SAMPLE_IDS[: len(formal_files)]
    ]:
        return (("formal raw sample order is not a contiguous plan prefix",), None)
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not sums_path.is_file()
        or sums_path.is_symlink()
    ):
        return (("raw integrity indexes are missing",), None)
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, json.JSONDecodeError, CapturePlanError) as exc:
        return ((f"raw integrity index is unreadable ({exc})",), None)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != 1
        or manifest.get("dataset_id") != plan.dataset_id
        or not isinstance(manifest.get("raw_samples"), list)
    ):
        return (("raw manifest header is invalid",), None)

    manifest_hashes: dict[str, str] = {}
    for entry in manifest["raw_samples"]:
        if not isinstance(entry, Mapping):
            return (("raw manifest entry is invalid",), None)
        name = entry.get("file")
        digest = entry.get("sha256")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"S[0-9]{4}\.json", name)
            or not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
            or name in manifest_hashes
        ):
            return (("raw manifest entry is invalid",), None)
        manifest_hashes[name] = digest
    if list(manifest_hashes) != expected_files:
        return (
            ("raw manifest file set or order does not match raw samples",),
            None,
        )

    checksum_hashes: dict[str, str] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return (("raw SHA256SUMS entry is invalid",), None)
        digest, name = parts[0], parts[1].lstrip("*")
        if (
            not SHA256_PATTERN.fullmatch(digest)
            or not re.fullmatch(r"S[0-9]{4}\.json", name)
            or name in checksum_hashes
        ):
            return (("raw SHA256SUMS entry is invalid",), None)
        checksum_hashes[name] = digest
    if list(checksum_hashes) != expected_files:
        return (
            ("raw SHA256SUMS file set or order does not match raw samples",),
            None,
        )
    computed_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sample_paths
    }
    if manifest_hashes != checksum_hashes or manifest_hashes != computed_hashes:
        return (("raw sample digest does not match integrity indexes",), None)
    return ((), computed_hashes)


def inspect_inventory(
    paths: Iterable[Path],
    plan: CapturePlan,
    *,
    capture_plan_sha256: str | None = None,
    raw_hashes_sha256: Mapping[str, str] | None = None,
) -> Inventory:
    """Read raw S*.json files and fail closed on duplicate/conflicting evidence."""

    issues: list[str] = []
    completed: set[str] = set()
    source_by_sample: dict[str, Path] = {}
    entry_by_id = {entry.sample_id: entry for entry in plan.capture_order}
    formal_start = int(FORMAL_SAMPLE_IDS[0][1:])

    for path_raw in sorted((Path(path) for path in paths), key=lambda item: str(item)):
        path = path_raw.expanduser().resolve()
        filename_id = path.stem
        if not SAMPLE_ID_PATTERN.fullmatch(filename_id):
            continue
        try:
            document = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object,
            )
        except (OSError, json.JSONDecodeError, CapturePlanError) as exc:
            issues.append(f"{path.name}: unreadable calibration sample ({exc})")
            continue
        if not isinstance(document, Mapping):
            issues.append(f"{path.name}: sample document is not an object")
            continue
        document_id = document.get("sample_id")
        if document_id != filename_id:
            issues.append(
                f"{path.name}: filename/sample_id conflict ({filename_id!r} vs {document_id!r})"
            )
            continue
        prior = source_by_sample.get(filename_id)
        if prior is not None:
            issues.append(
                f"duplicate sample_id {filename_id} in {prior.name} and {path.name}"
            )
            continue
        source_by_sample[filename_id] = path
        if document.get("dataset_id") != plan.dataset_id:
            issues.append(f"{path.name}: dataset_id conflicts with capture plan")
            continue

        entry = entry_by_id.get(filename_id)
        if entry is None:
            match = SAMPLE_ID_PATTERN.fullmatch(filename_id)
            assert match is not None
            if int(match.group(1)) >= formal_start:
                issues.append(f"{path.name}: unlisted formal sample_id")
            continue
        if capture_plan_sha256 is None:
            metadata = document.get("metadata")
            if not isinstance(metadata, Mapping):
                issues.append(f"{path.name}: formal sample metadata is missing")
                continue
            if metadata.get("pose_index") != entry.pose_index:
                issues.append(f"{path.name}: pose_index conflicts with capture plan")
                continue
            if metadata.get("split") != entry.split:
                issues.append(f"{path.name}: split conflicts with capture plan")
                continue
        else:
            filename = f"{filename_id}.json"
            if raw_hashes_sha256 is None or filename not in raw_hashes_sha256:
                issues.append(
                    f"{path.name}: verified raw hash state is unavailable"
                )
                continue
            ordered_names = list(raw_hashes_sha256)
            current_index = ordered_names.index(filename)
            expected_prior_hashes = {
                name: raw_hashes_sha256[name]
                for name in ordered_names[:current_index]
            }
            sample_issue = _formal_sample_issue(
                document,
                entry,
                plan,
                capture_plan_sha256,
                expected_prior_hashes,
            )
            if sample_issue is not None:
                issues.append(f"{path.name}: {sample_issue}")
                continue
        completed.add(filename_id)

    first_missing_index = next(
        (
            index
            for index, entry in enumerate(plan.capture_order)
            if entry.sample_id not in completed
        ),
        len(plan.capture_order),
    )
    out_of_order = [
        entry.sample_id
        for entry in plan.capture_order[first_missing_index + 1 :]
        if entry.sample_id in completed
    ]
    if out_of_order:
        expected = plan.capture_order[first_missing_index].sample_id
        issues.append(
            f"capture-order conflict: {out_of_order} exist while {expected} is pending"
        )
    return Inventory(frozenset(completed), tuple(issues))


def inspect_raw_inventory(
    dataset_dir: Path,
    plan: CapturePlan,
    *,
    capture_plan_sha256: str,
) -> Inventory:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if dataset_dir.name != "raw":
        return Inventory(frozenset(), ("dataset directory is not named raw",))
    if not dataset_dir.is_dir():
        return Inventory(
            frozenset(), (f"raw dataset directory does not exist: {dataset_dir}",)
        )
    if not SHA256_PATTERN.fullmatch(capture_plan_sha256):
        return Inventory(frozenset(), ("loaded capture plan SHA-256 is invalid",))
    integrity_issues, raw_hashes = _raw_integrity_state(dataset_dir, plan)
    if integrity_issues:
        return Inventory(frozenset(), integrity_issues)
    assert raw_hashes is not None
    inventory = inspect_inventory(
        dataset_dir.glob("S*.json"),
        plan,
        capture_plan_sha256=capture_plan_sha256,
        raw_hashes_sha256=raw_hashes,
    )
    return inventory


def next_capture(plan: CapturePlan, inventory: Inventory) -> CapturePlanEntry | None:
    for entry in plan.capture_order:
        if entry.sample_id not in inventory.completed_sample_ids:
            return entry
    return None


def precheck_capture(
    *,
    now: float,
    image_received: float | None,
    detection_received: float | None,
    tf_received: float | None,
    feedback_received: float | None,
    decision_margin: float | None,
    feedback: object,
    base_tcp_available: bool,
) -> tuple[str, ...]:
    """Fast local UI check; the recorder remains the authoritative gate."""

    reasons: list[str] = []

    def require_fresh(label: str, received: float | None, limit: float):
        if received is None:
            reasons.append(f"{label} has not arrived")
            return
        age = now - received
        if not math.isfinite(age) or age < 0.0 or age >= limit:
            reasons.append(f"{label} is stale ({age:.2f}s)")

    require_fresh("RGB", image_received, RGB_MAX_AGE_S)
    require_fresh("Tag", detection_received, TAG_MAX_AGE_S)
    require_fresh("Tag TF", tf_received, TF_MAX_AGE_S)
    require_fresh("safe-driver feedback", feedback_received, FEEDBACK_MAX_AGE_S)
    # tf2_ros Buffer.can_transform returns Python bool on some hosts and int 0/1
    # on AGX Humble.  Accept only those exact true forms; reject everything else.
    if base_tcp_available is not True and base_tcp_available != 1:
        reasons.append("roarm_base_link -> hand_tcp TF is unavailable")
    if (
        decision_margin is None
        or not isinstance(decision_margin, (int, float))
        or not math.isfinite(float(decision_margin))
        or float(decision_margin) < MIN_DECISION_MARGIN
    ):
        reasons.append("Tag decision margin is below 50")
    if not isinstance(feedback, Mapping):
        reasons.append("safe-driver feedback payload is unavailable")
        return tuple(reasons)
    torque = feedback.get("torque_enabled")
    if (
        not isinstance(torque, list)
        or len(torque) != 6
        or any(value is not True for value in torque)
    ):
        reasons.append("all six torque flags are not true")
    if feedback.get("armed") is not False:
        reasons.append("safe driver is not explicitly disarmed")
    if feedback.get("dry_run") is not False:
        reasons.append("safe driver is not explicitly non-dry-run")
    if feedback.get("fault") != "":
        reasons.append("safe driver fault is not empty")
    voltage = feedback.get("voltage")
    if (
        isinstance(voltage, bool)
        or not isinstance(voltage, (int, float))
        or not math.isfinite(float(voltage))
        or not 11.0 <= float(voltage) <= 13.0
    ):
        reasons.append("safe-driver voltage is not within 11-13 V")
    return tuple(reasons)


def sample_auto_clear_block_reasons(
    *,
    now: float,
    feedback_received: float | None,
    feedback: object,
    joint_target_publisher_count: object,
) -> tuple[str, ...]:
    """Whitelist the one fault that a Sample click may clear exactly once."""

    reasons: list[str] = []
    if feedback_received is None:
        reasons.append("safe-driver feedback has not arrived")
    else:
        age = now - feedback_received
        if not math.isfinite(age) or age < 0.0 or age >= FEEDBACK_MAX_AGE_S:
            reasons.append(f"safe-driver feedback is stale ({age:.2f}s)")
    if not isinstance(feedback, Mapping):
        reasons.append("safe-driver feedback payload is unavailable")
        return tuple(reasons)
    feedback_stamp_ns = feedback.get("feedback_stamp_ns")
    if (
        isinstance(feedback_stamp_ns, bool)
        or not isinstance(feedback_stamp_ns, int)
        or feedback_stamp_ns <= 0
    ):
        reasons.append("safe-driver feedback_stamp_ns is unavailable")
    torque = feedback.get("torque_enabled")
    if (
        not isinstance(torque, list)
        or len(torque) != 6
        or any(value is not True for value in torque)
    ):
        reasons.append("all six torque flags are not true")
    if feedback.get("armed") is not False:
        reasons.append("safe driver is not explicitly disarmed")
    if feedback.get("dry_run") is not False:
        reasons.append("safe driver is not explicitly non-dry-run")
    # Some safe-driver builds keep an exact recoverable fault latched while
    # reporting TORQUE_ON_VERIFIED after all six direct feedback flags recover.
    # Accept only the narrowly-whitelisted faults; never a different fault or
    # a different state.
    allowed_states = {f"FAULT:{fault}" for fault in AUTO_CLEAR_FAULTS}
    allowed_states.add("TORQUE_ON_VERIFIED")
    if feedback.get("state") not in allowed_states:
        reasons.append("safe driver state is not the exact auto-clear fault state")
    voltage = feedback.get("voltage")
    if (
        isinstance(voltage, bool)
        or not isinstance(voltage, (int, float))
        or not math.isfinite(float(voltage))
        or not 11.0 <= float(voltage) <= 13.0
    ):
        reasons.append("safe-driver voltage is not within 11-13 V")
    if (
        isinstance(joint_target_publisher_count, bool)
        or not isinstance(joint_target_publisher_count, int)
        or joint_target_publisher_count != 0
    ):
        reasons.append("joint_target publisher count is not zero")
    if feedback.get("fault") not in AUTO_CLEAR_FAULTS:
        reasons.append("fault is not an exact auto-clear recoverable fault")
    return tuple(reasons)


def ui_click_authorization_id(sample_id: str, clicked_at_ns: int) -> str:
    if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise ValueError("authorization sample_id must match Sdddd")
    if (
        isinstance(clicked_at_ns, bool)
        or not isinstance(clicked_at_ns, int)
        or clicked_at_ns <= 0
    ):
        raise ValueError("clicked_at_ns must be a positive integer")
    clicked = datetime.fromtimestamp(clicked_at_ns / 1_000_000_000, tz=timezone.utc)
    timestamp = clicked.strftime("%Y%m%dT%H%M%S.%fZ")
    return f"ui-click-{sample_id}-{timestamp}"


def build_recorder_command(
    *,
    capture_plan_path: Path,
    dataset_dir: Path,
    plan: CapturePlan,
    entry: CapturePlanEntry,
    operator_authorization_id: str,
    capture_plan_sha256: str,
) -> list[str]:
    capture_plan_path = Path(capture_plan_path).expanduser().resolve()
    if not capture_plan_path.is_file():
        raise ValueError("capture_plan_path must be an existing file")
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if dataset_dir.name != "raw":
        raise ValueError("dataset_dir must be the immutable raw directory")
    if plan.dataset_dir is None or dataset_dir != plan.dataset_dir:
        raise ValueError("dataset_dir must equal the frozen capture-plan path")
    if entry not in plan.capture_order:
        raise ValueError("entry is not part of the loaded capture plan")
    if not operator_authorization_id:
        raise ValueError("operator authorization ID must be non-empty")
    if not SHA256_PATTERN.fullmatch(capture_plan_sha256):
        raise ValueError("capture_plan_sha256 must be a lowercase SHA-256 digest")
    return [
        "ros2",
        "run",
        "roarm_safe_driver",
        "calibration_recorder",
        "--",
        "--capture-plan",
        str(capture_plan_path),
        "--sample-id",
        entry.sample_id,
        "--operator-authorization-id",
        operator_authorization_id,
        "--capture-plan-sha256",
        capture_plan_sha256,
    ]
