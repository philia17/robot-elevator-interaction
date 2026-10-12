import hashlib
import json
from pathlib import Path

import pytest

from tools.calibration.live_capture_core import (
    AUTO_CLEAR_JOINT_LIMITS_FAULT,
    AUTO_CLEAR_TORQUE_FAULT,
    ClearFaultVerification,
    CapturePlanError,
    REQUIRED_DRIVER_PROVENANCE_KEYS,
    build_recorder_command,
    capture_block_reasons,
    inspect_inventory,
    inspect_raw_inventory,
    load_capture_plan,
    next_capture,
    precheck_capture,
    sample_auto_clear_block_reasons,
    ui_click_authorization_id,
    validate_capture_plan,
)
from tools.calibration.validate_formal_capture_plan import (
    SOURCE_HASH_PATHS,
    source_path_for_validation,
    validate as validate_formal_plan,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROPOSED_FORMAL_PLAN = (
    REPOSITORY
    / "config/calibration/formal/"
    "gemini335_roarm_20260722_r1_b4_formal_capture_plan.json"
)
LIVE_VIEW_SOURCE = REPOSITORY / "tools/apriltag_calibration_live_view.py"
HELD_OUT_ORDERS = {4, 8, 12, 16, 20, 24, 27, 30}


def _plan_payload(**overrides):
    entries = []
    for order, sample_number in enumerate(range(12, 42), start=1):
        entries.append(
            {
                "sample_id": f"S{sample_number:04d}",
                "pose_index": sample_number,
                "split": "held_out" if order in HELD_OUT_ORDERS else "train",
                "pose_mode": "freeform_diverse",
                "pose_intent": "Use a visibly distinct safe pose.",
                "operator_prompt": (
                    "Hand-teach a visibly distinct safe pose and keep the full Tag "
                    "visible."
                ),
            }
        )
    payload = {
        "schema_version": "1.0",
        "plan_status": "frozen",
        "dataset_id": "gemini335_roarm_20260722_r1",
        "dataset_dir": None,
        "ros_domain_id": 151,
        "status_source": "safe_driver",
        "capture_enabled": True,
        "user_approved": True,
        "formal_capture_prerequisites_frozen": True,
        "operator_contract": {"capture_timeout_s": 15.0},
        "source_hashes_sha256": {
            "test_source.py": "a" * 64,
            **{key: "b" * 64 for key in REQUIRED_DRIVER_PROVENANCE_KEYS},
        },
        "capture_order": entries,
    }
    payload.update(overrides)
    return payload


def _write_sample(path, plan, sample_id, *, pose_index=None, split=None):
    entry = next(item for item in plan.capture_order if item.sample_id == sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "dataset_id": plan.dataset_id,
                "metadata": {
                    "pose_index": (
                        entry.pose_index if pose_index is None else pose_index
                    ),
                    "split": entry.split if split is None else split,
                },
            }
        )
    )
    return path


def _write_integrity_indexes(raw, plan):
    entries = [
        {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(raw.glob("S*.json"))
    ]
    (raw / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": plan.dataset_id,
                "raw_samples": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (raw / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['file']}\n" for entry in entries)
    )


def _write_formal_sample(
    raw,
    plan,
    capture_plan_sha256,
    *,
    include_historical=True,
    **metadata_overrides,
):
    entry = plan.capture_order[0]
    sources = dict(plan.source_hashes_sha256)
    raw.mkdir(parents=True, exist_ok=True)
    if include_historical:
        for index in range(1, 12):
            historical = raw / f"S{index:04d}.json"
            if not historical.exists():
                historical.write_text(
                    json.dumps(
                        {
                            "sample_id": f"S{index:04d}",
                            "dataset_id": plan.dataset_id,
                        }
                    )
                    + "\n"
                )
    _write_integrity_indexes(raw, plan)
    prior_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw.glob("S*.json"))
    }
    preappend_manifest_sha256 = hashlib.sha256(
        (raw / "manifest.json").read_bytes()
    ).hexdigest()
    preappend_sums_sha256 = hashlib.sha256(
        (raw / "SHA256SUMS").read_bytes()
    ).hexdigest()
    metadata = {
        "pose_index": entry.pose_index,
        "split": entry.split,
        "operator_authorization_id": "ui-click-S0012-test",
        "status_source": "safe_driver",
        "ros_domain_id": plan.ros_domain_id,
        "capture_plan_sha256": capture_plan_sha256,
        "recorder_timeout_s": 15.0,
        "frozen_source_hashes_sha256": sources,
        "driver_versions": {
            key: sources[key] for key in REQUIRED_DRIVER_PROVENANCE_KEYS
        },
        "preappend_raw_integrity": {
            "existing_sample_count": len(prior_hashes),
            "existing_sample_hashes_sha256": prior_hashes,
            "manifest_sha256": preappend_manifest_sha256,
            "sha256sums_sha256": preappend_sums_sha256,
        },
        "tag_static_window": {
            "duration_ns": 1_000_000_000,
            "sample_count": 15,
            "maximum_gap_ns": 80_000_000,
            "translation_spread_m": 0.001,
            "rotation_spread_deg": 0.2,
            "observations": [{} for _ in range(15)],
        },
        "joint_safe_window": {
            "duration_ns": 1_000_000_000,
            "sample_count": 20,
            "maximum_gap_ns": 60_000_000,
            "safe_status_alignment": "all_frames_aligned_healthy",
            "observations": [
                {
                    "positions_rad": [0.0] * 6,
                    "voltage_v": 12.1,
                    "torque_enabled": [True] * 6,
                }
                for _ in range(20)
            ],
        },
    }
    metadata.update(metadata_overrides)
    path = raw / f"{entry.sample_id}.json"
    path.write_text(
        json.dumps(
            {
                "sample_id": entry.sample_id,
                "dataset_id": plan.dataset_id,
                "capture_plan_sha256": capture_plan_sha256,
                "coordinate_contract_version": "1.0.0",
                "calibration_type": "eye_on_base",
                "settled": True,
                "rgb_header_stamp_ns": 1_000_000_000,
                "joint_states_stamp_ns": 1_000_000_000,
                "torque_enabled": [True] * 6,
                "torque_evidence": {
                    "direct_joint_torque_flags_available": True,
                    "armed": False,
                    "dry_run": False,
                    "fault": "",
                },
                "T_roarm_base_link_hand_tcp": {
                    "translation_m": [0.0, 0.0, 0.2],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "T_camera_optical_frame_calibration_tag": {
                    "translation_m": [0.0, 0.0, 0.4],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "tag_family": "tag36h11",
                "tag_id": 0,
                "tag_detection_quality": {
                    "hamming": 0,
                    "decision_margin": 100.0,
                },
                "metadata": metadata,
            },
            sort_keys=True,
        )
        + "\n"
    )
    _write_integrity_indexes(raw, plan)
    return path


def test_plan_schema_accepts_disabled_proposal_but_blocks_capture():
    plan = validate_capture_plan(
        _plan_payload(
            plan_status="proposed",
            capture_enabled=False,
            user_approved=False,
            formal_capture_prerequisites_frozen=False,
        )
    )
    assert capture_block_reasons(plan, Path("/evidence/raw")) == (
        "plan is proposed, not frozen",
        "capture_enabled is false",
        "user_approved is false",
        "formal prerequisites remain pending",
    )


def test_plan_schema_requires_exact_30_ids_22_train_8_held_out():
    payload = _plan_payload()
    payload["capture_order"][-1]["sample_id"] = "S0012"
    with pytest.raises(CapturePlanError, match="duplicate sample_id"):
        validate_capture_plan(payload)

    payload = _plan_payload()
    payload["capture_order"][3]["split"] = "train"
    with pytest.raises(CapturePlanError, match="22 train and 8 held_out"):
        validate_capture_plan(payload)

    payload = _plan_payload()
    payload["capture_order"][26]["split"] = "train"
    payload["capture_order"][27]["split"] = "held_out"
    with pytest.raises(CapturePlanError, match="held_out entries must be"):
        validate_capture_plan(payload)

    payload = _plan_payload()
    payload["capture_order"][-1]["pose_index"] = payload["capture_order"][0][
        "pose_index"
    ]
    with pytest.raises(CapturePlanError, match="duplicate pose_index"):
        validate_capture_plan(payload)


def test_freeform_schema_rejects_numeric_targets_and_wrong_mode():
    for target_key in ("image_target", "orientation_target"):
        payload = _plan_payload()
        payload["capture_order"][0][target_key] = {"unexpected": True}
        with pytest.raises(CapturePlanError, match=f"{target_key} is prohibited"):
            validate_capture_plan(payload)

    payload = _plan_payload()
    payload["capture_order"][0]["pose_mode"] = "numeric_target"
    with pytest.raises(CapturePlanError, match="pose_mode must be"):
        validate_capture_plan(payload)


def test_plan_schema_locks_timeout_and_requires_source_hashes():
    payload = _plan_payload()
    payload["operator_contract"]["capture_timeout_s"] = 16.0
    with pytest.raises(CapturePlanError, match="capture_timeout_s must equal 15.0"):
        validate_capture_plan(payload)

    payload = _plan_payload(source_hashes_sha256={"broken.py": "not-a-digest"})
    with pytest.raises(CapturePlanError, match="lowercase SHA-256"):
        validate_capture_plan(payload)


def test_json_plan_loader_and_next_sample_selection(tmp_path):
    plan_path = tmp_path / "formal_plan.json"
    plan_path.write_text(json.dumps(_plan_payload()))
    plan = load_capture_plan(plan_path)
    first = plan.capture_order[0]
    second = plan.capture_order[1]
    raw = tmp_path / "raw"
    inventory = inspect_inventory([], plan)
    assert next_capture(plan, inventory) == first

    first_path = _write_sample(raw / f"{first.sample_id}.json", plan, first.sample_id)
    inventory = inspect_inventory([first_path], plan)
    assert next_capture(plan, inventory) == second


def test_repository_legacy_frozen_freeform_plan_remains_structurally_readable():
    payload = json.loads(PROPOSED_FORMAL_PLAN.read_text())
    plan = load_capture_plan(PROPOSED_FORMAL_PLAN)
    assert len(plan.capture_order) == 30
    assert [entry.sample_id for entry in plan.capture_order] == [
        f"S{sample_number:04d}" for sample_number in range(12, 42)
    ]
    assert [
        order
        for order, entry in enumerate(plan.capture_order, start=1)
        if entry.split == "held_out"
    ] == sorted(HELD_OUT_ORDERS)
    assert all(entry.pose_mode == "freeform_diverse" for entry in plan.capture_order)
    assert all("Freeform diverse pose" in entry.pose_intent for entry in plan.capture_order)
    assert all(
        "full tag visible" in entry.operator_prompt.lower()
        for entry in plan.capture_order
    )
    reasons = capture_block_reasons(plan, plan.dataset_dir)
    assert reasons == ()
    assert payload["operator_contract"]["capture_timeout_s"] == 15.0
    assert payload["measurement_contract"]["joint_and_safety_window"][
        "sample_count_minimum"
    ] == 20
    assert payload["measurement_contract"]["joint_and_safety_window"][
        "sample_gap_ms_maximum"
    ] == 100.0
    assert payload["measurement_contract"]["tag_static_window"][
        "duration_s_minimum"
    ] == 1.0
    assert payload["measurement_contract"]["tag_static_window"][
        "sample_count_minimum"
    ] == 15
    assert payload["measurement_contract"]["tag_static_window"][
        "sample_gap_ms_maximum"
    ] == 250.0
    assert payload["measurement_contract"]["tag_static_window"][
        "measurement_basis"
    ]["canonical_image_publish_limit_hz"] == 30.0
    assert payload["measurement_contract"]["tag_static_window"][
        "measurement_basis"
    ]["isolated_tag_detection_rate_hz"] == 17.7
    assert payload["measurement_contract"]["tag_static_window"][
        "measurement_basis"
    ]["observed_tag_detection_gap_ms_maximum"] == 192.0
    assert payload["source_hashes_sha256"].keys() == SOURCE_HASH_PATHS.keys()
    assert all(
        len(payload["source_hashes_sha256"][source_name]) == 64
        for source_name in SOURCE_HASH_PATHS
    )
    # Frozen historical plans remain readable, but must not silently become
    # active after the recorder/verifier source set changes.
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_formal_plan(PROPOSED_FORMAL_PLAN)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (
            ("measurement_contract", "tag_static_window", "sample_count_minimum"),
            14,
            "sample_count_minimum",
        ),
        (
            (
                "measurement_contract",
                "tag_static_window",
                "measurement_basis",
                "isolated_tag_detection_rate_hz",
            ),
            18.0,
            "isolated_tag_detection_rate_hz",
        ),
        (
            ("source_hashes_sha256", "calibration_core.py"),
            "0" * 64,
            "source hash mismatch",
        ),
    ],
)
def test_formal_plan_validator_rejects_policy_or_provenance_drift(
    tmp_path, path, replacement, message
):
    payload = json.loads(PROPOSED_FORMAL_PLAN.read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    candidate = tmp_path / "formal_plan.json"
    candidate.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        validate_formal_plan(candidate)


def test_inventory_rejects_duplicate_conflict_and_out_of_order(tmp_path):
    plan = validate_capture_plan(_plan_payload())
    duplicate_a = _write_sample(tmp_path / "a" / "S0012.json", plan, "S0012")
    duplicate_b = _write_sample(tmp_path / "b" / "S0012.json", plan, "S0012")
    inventory = inspect_inventory([duplicate_a, duplicate_b], plan)
    assert any("duplicate sample_id S0012" in issue for issue in inventory.issues)

    split_conflict = _write_sample(
        tmp_path / "split" / "S0012.json", plan, "S0012", split="held_out"
    )
    inventory = inspect_inventory([split_conflict], plan)
    assert inventory.completed_sample_ids == frozenset()
    assert inventory.issues == ("S0012.json: split conflicts with capture plan",)

    out_of_order = _write_sample(tmp_path / "order" / "S0013.json", plan, "S0013")
    inventory = inspect_inventory([out_of_order], plan)
    assert any("capture-order conflict" in issue for issue in inventory.issues)
    assert next_capture(plan, inventory).sample_id == "S0012"


def test_formal_raw_inventory_requires_plan_provenance_windows_and_hashes(tmp_path):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    path = _write_formal_sample(raw, plan, plan_sha256)
    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert inventory.issues == ()
    assert inventory.completed_sample_ids == frozenset(("S0012",))

    document = json.loads(path.read_text())
    del document["metadata"]["joint_safe_window"]
    path.write_text(json.dumps(document) + "\n")
    _write_integrity_indexes(raw, plan)
    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert inventory.completed_sample_ids == frozenset()
    assert any("joint/safe-status window evidence is missing" in issue for issue in inventory.issues)


def test_formal_raw_inventory_uses_frozen_tag_window_contract(tmp_path):
    payload = _plan_payload(
        measurement_contract={
            "tag_static_window": {
                "sample_count_minimum": 10,
                "sample_gap_ms_maximum": 250.0,
            }
        }
    )
    plan = validate_capture_plan(payload)
    raw = tmp_path / "raw"
    path = _write_formal_sample(raw, plan, "c" * 64)
    document = json.loads(path.read_text())
    window = document["metadata"]["tag_static_window"]
    window["sample_count"] = 10
    window["observations"] = [{} for _ in range(10)]
    window["maximum_gap_ns"] = 233_000_000
    path.write_text(json.dumps(document) + "\n")
    _write_integrity_indexes(raw, plan)

    inventory = inspect_raw_inventory(raw, plan, capture_plan_sha256="c" * 64)

    assert inventory.issues == ()
    assert inventory.completed_sample_ids == frozenset(("S0012",))


def test_independent_repeat_accepts_first_formal_sample_without_historical_prefix(
    tmp_path,
):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    _write_formal_sample(
        raw,
        plan,
        plan_sha256,
        include_historical=False,
    )

    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )

    assert inventory.issues == ()
    assert inventory.completed_sample_ids == frozenset(("S0012",))


def test_formal_raw_inventory_rejects_stale_plan_or_integrity_drift(tmp_path):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    path = _write_formal_sample(raw, plan, "d" * 64)
    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert inventory.completed_sample_ids == frozenset()
    assert any("capture_plan_sha256 conflicts" in issue for issue in inventory.issues)

    path = _write_formal_sample(raw, plan, plan_sha256)
    path.write_text(path.read_text().replace('"settled": true', '"settled": false'))
    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert any("digest does not match" in issue for issue in inventory.issues)

    _write_integrity_indexes(raw, plan)
    (raw / "SHA256SUMS").write_text(f"{'0' * 64}  S0012.json\n")
    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert any("SHA256SUMS" in issue for issue in inventory.issues)


def test_formal_inventory_rejects_rebaselined_historical_prefix(tmp_path):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    _write_formal_sample(raw, plan, plan_sha256)
    historical = raw / "S0001.json"
    historical.write_text(
        historical.read_text().replace(
            '"dataset_id"',
            '"mutated": true, "dataset_id"',
        )
    )
    _write_integrity_indexes(raw, plan)

    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert inventory.completed_sample_ids == frozenset()
    assert any(
        "pre-append raw sample hash chain conflicts with the actual prefix"
        in issue
        for issue in inventory.issues
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("manifest_sha256", "pre-append manifest hash conflicts"),
        ("sha256sums_sha256", "pre-append SHA256SUMS hash conflicts"),
    ],
)
def test_formal_inventory_rejects_rebaselined_preappend_index_digest(
    tmp_path,
    field,
    message,
):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    sample = _write_formal_sample(
        raw,
        plan,
        plan_sha256,
        include_historical=False,
    )
    document = json.loads(sample.read_text())
    document["metadata"]["preappend_raw_integrity"][field] = "0" * 64
    sample.write_text(json.dumps(document, sort_keys=True) + "\n")
    _write_integrity_indexes(raw, plan)

    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert inventory.completed_sample_ids == frozenset()
    assert any(message in issue for issue in inventory.issues)


def test_first_formal_sample_accepts_absent_preappend_indexes(tmp_path):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    sample = _write_formal_sample(
        raw,
        plan,
        plan_sha256,
        include_historical=False,
    )
    document = json.loads(sample.read_text())
    preappend = document["metadata"]["preappend_raw_integrity"]
    preappend["existing_sample_count"] = 0
    preappend["existing_sample_hashes_sha256"] = {}
    preappend["manifest_sha256"] = None
    preappend["sha256sums_sha256"] = None
    sample.write_text(json.dumps(document, sort_keys=True) + "\n")
    _write_integrity_indexes(raw, plan)

    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert inventory.completed_sample_ids == frozenset({"S0012"})
    assert inventory.issues == ()


def test_formal_raw_inventory_blocks_interrupted_unindexed_append(tmp_path):
    plan = validate_capture_plan(_plan_payload())
    plan_sha256 = "c" * 64
    raw = tmp_path / "raw"
    _write_formal_sample(raw, plan, plan_sha256)
    (raw / "manifest.json").unlink()
    inventory = inspect_raw_inventory(
        raw,
        plan,
        capture_plan_sha256=plan_sha256,
    )
    assert any("integrity indexes are missing" in issue for issue in inventory.issues)


def test_local_precheck_is_fail_closed_without_numeric_pose_gating():
    now = 100.0
    feedback = {
        "torque_enabled": [True] * 6,
        "armed": False,
        "dry_run": False,
        "fault": "",
        "voltage": 12.1,
    }
    assert (
        precheck_capture(
            now=now,
            image_received=99.9,
            detection_received=99.9,
            tf_received=99.9,
            feedback_received=99.9,
            decision_margin=50.0,
            feedback=feedback,
            base_tcp_available=True,
        )
        == ()
    )
    # AGX Humble tf2_ros returns int 1 from can_transform.
    assert (
        precheck_capture(
            now=now,
            image_received=99.9,
            detection_received=99.9,
            tf_received=99.9,
            feedback_received=99.9,
            decision_margin=50.0,
            feedback=feedback,
            base_tcp_available=1,
        )
        == ()
    )
    reasons = precheck_capture(
        now=now,
        image_received=98.0,
        detection_received=97.9,
        tf_received=None,
        feedback_received=98.0,
        decision_margin=49.9,
        feedback={
            "torque_enabled": [True] * 5 + [False],
            "armed": True,
            "dry_run": True,
            "fault": "latched",
            "voltage": 10.9,
        },
        base_tcp_available=False,
    )
    assert any("RGB is stale" in reason for reason in reasons)
    assert any("Tag is stale" in reason for reason in reasons)
    assert "Tag TF has not arrived" in reasons
    assert any("feedback is stale" in reason for reason in reasons)
    assert "Tag decision margin is below 50" in reasons
    assert "roarm_base_link -> hand_tcp TF is unavailable" in reasons
    assert "all six torque flags are not true" in reasons
    assert "safe driver is not explicitly disarmed" in reasons
    assert "safe driver is not explicitly non-dry-run" in reasons
    assert "safe driver fault is not empty" in reasons
    assert "safe-driver voltage is not within 11-13 V" in reasons
    assert not any("outside target" in reason for reason in reasons)


def _eligible_auto_clear_feedback(**overrides):
    feedback = {
        "feedback_stamp_ns": 123_000_000,
        "torque_enabled": [True] * 6,
        "armed": False,
        "dry_run": False,
        "state": f"FAULT:{AUTO_CLEAR_TORQUE_FAULT}",
        "fault": AUTO_CLEAR_TORQUE_FAULT,
        "voltage": 12.1,
    }
    feedback.update(overrides)
    return feedback


def test_sample_auto_clear_is_exactly_whitelisted_and_fail_closed():
    assert (
        sample_auto_clear_block_reasons(
            now=100.0,
            feedback_received=99.9,
            feedback=_eligible_auto_clear_feedback(),
            joint_target_publisher_count=0,
        )
        == ()
    )
    assert (
        sample_auto_clear_block_reasons(
            now=100.0,
            feedback_received=99.9,
            feedback=_eligible_auto_clear_feedback(state="TORQUE_ON_VERIFIED"),
            joint_target_publisher_count=0,
        )
        == ()
    )
    assert (
        sample_auto_clear_block_reasons(
            now=100.0,
            feedback_received=99.9,
            feedback=_eligible_auto_clear_feedback(
                state=f"FAULT:{AUTO_CLEAR_JOINT_LIMITS_FAULT}",
                fault=AUTO_CLEAR_JOINT_LIMITS_FAULT,
            ),
            joint_target_publisher_count=0,
        )
        == ()
    )
    assert (
        sample_auto_clear_block_reasons(
            now=100.0,
            feedback_received=99.9,
            feedback=_eligible_auto_clear_feedback(
                state="TORQUE_ON_VERIFIED",
                fault=AUTO_CLEAR_JOINT_LIMITS_FAULT,
            ),
            joint_target_publisher_count=0,
        )
        == ()
    )

    cases = (
        (
            {"feedback_received": 98.0},
            "safe-driver feedback is stale",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(feedback_stamp_ns=0)},
            "feedback_stamp_ns is unavailable",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(torque_enabled=[True] * 5)},
            "all six torque flags are not true",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(armed=True)},
            "safe driver is not explicitly disarmed",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(dry_run=True)},
            "safe driver is not explicitly non-dry-run",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(state="DISARMED")},
            "safe driver state is not the exact auto-clear fault state",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(voltage=10.9)},
            "safe-driver voltage is not within 11-13 V",
        ),
        (
            {"joint_target_publisher_count": 1},
            "joint_target publisher count is not zero",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(fault="some other fault")},
            "fault is not an exact auto-clear recoverable fault",
        ),
        (
            {"feedback": _eligible_auto_clear_feedback(fault="")},
            "fault is not an exact auto-clear recoverable fault",
        ),
    )
    baseline = {
        "now": 100.0,
        "feedback_received": 99.9,
        "feedback": _eligible_auto_clear_feedback(),
        "joint_target_publisher_count": 0,
    }
    for overrides, expected_reason in cases:
        arguments = {**baseline, **overrides}
        assert any(
            expected_reason in reason
            for reason in sample_auto_clear_block_reasons(**arguments)
        )


def test_clear_fault_verification_requires_newer_complete_safe_feedback():
    verification = ClearFaultVerification.from_successful_response(
        response_received=100.0,
        pre_clear_feedback_stamp_ns=123_000_000,
    )
    valid = _eligible_auto_clear_feedback(
        feedback_stamp_ns=123_000_001,
        state="FAULT_CLEARED_DISARMED",
        fault="",
    )
    assert verification.feedback_verifies(100.1, valid)
    assert not verification.feedback_verifies(100.0, valid)
    assert not verification.feedback_verifies(
        100.1,
        {**valid, "feedback_stamp_ns": 123_000_000},
    )
    assert not verification.feedback_verifies(
        100.1,
        {**valid, "state": "DISARMED"},
    )
    assert not verification.feedback_verifies(
        100.1,
        {**valid, "torque_enabled": [True] * 5 + [False]},
    )
    assert not verification.feedback_verifies(100.1, {**valid, "armed": True})
    assert not verification.feedback_verifies(100.1, {**valid, "dry_run": True})
    assert not verification.feedback_verifies(100.1, {**valid, "fault": "other"})
    assert not verification.feedback_verifies(100.1, {**valid, "voltage": 10.9})
    assert not verification.expired(101.9)
    assert verification.expired(102.0)


def test_authorization_and_recorder_command_are_explicit_argv(tmp_path):
    plan_payload = _plan_payload(dataset_dir=str(tmp_path / "raw"))
    plan_path = tmp_path / "formal_plan.json"
    plan_path.write_text(json.dumps(plan_payload))
    plan = validate_capture_plan(plan_payload)
    entry = plan.capture_order[0]
    authorization_id = ui_click_authorization_id(
        entry.sample_id, 1_785_072_194_123_456_000
    )
    assert authorization_id.startswith("ui-click-S0012-")
    assert authorization_id.endswith("Z")
    command = build_recorder_command(
        capture_plan_path=plan_path,
        dataset_dir=tmp_path / "raw",
        plan=plan,
        entry=entry,
        operator_authorization_id=authorization_id,
        capture_plan_sha256="b" * 64,
    )
    assert command[:5] == [
        "ros2",
        "run",
        "roarm_safe_driver",
        "calibration_recorder",
        "--",
    ]
    assert command[command.index("--capture-plan") + 1] == str(plan_path.resolve())
    assert command[command.index("--sample-id") + 1] == "S0012"
    assert command[command.index("--operator-authorization-id") + 1] == authorization_id
    assert command[command.index("--capture-plan-sha256") + 1] == "b" * 64
    for redundant in (
        "--dataset-dir",
        "--dataset-id",
        "--pose-index",
        "--split",
        "--ros-domain-id",
        "--status-source",
        "--timeout-s",
        "--frozen-source-hashes-json",
    ):
        assert redundant not in command


def test_live_view_has_freeform_guide_and_single_child_fail_safe_guards():
    source = LIVE_VIEW_SOURCE.read_text()
    assert "Freeform: use a visibly distinct pose; keep full tag visible" in source
    assert "subprocess.Popen(" in source
    assert "shell=True" not in source
    assert "if self._process is not None or self._result is not None:" in source
    assert "process.terminate()" in source
    assert "process.kill()" in source
    assert "capture_runner.shutdown(timeout_s=2.0)" in source
    assert "def running_snapshot(self, now=None):" in source
    assert '"--capture-timeout-s"' not in source
    assert "capture_plan.capture_timeout_s" in source
    assert "fresh 1s Tag + joint windows" in source
    assert "recorder gate timeout" in source
    assert "capture_progress[0]" in source
    assert "capture_running_sample_id or sample_id" in source
    assert "if result is not None:" in source
    assert "no evidence " in source
    assert "files were deleted. Manually review raw inventory" in source
    assert 'Trigger, "/roarm_safe/clear_fault"' in source
    assert 'message == "FAULT_CLEARED_DISARMED"' in source
    assert "AUTO_CLEAR_FAULTS" in source
    assert "click clears one exact recoverable fault" in source
    assert "SAMPLE PRECHECK BLOCKED: fault=" in source
    assert "_recorder_package_block_reason" in source
    assert "roarm_safe_driver is not on AMENT_PREFIX_PATH" in source
    assert "pre_clear_feedback_stamp_ns" in source
    assert "node.joint_target_publisher_count() != 0" in source
    assert "pending sample changed during clear-fault" in source
    assert "one attempt, no retry" in source
    assert "node.request_sample_fault_clear()" in source
    assert source.count("call_async(Trigger.Request())") == 1
    assert source.count("capture_runner.start(") == 1
    assert "validate_formal_plan(parsed.capture_plan)" in source
    snapshot_source = source.split("def _plan_snapshot_reason():", maxsplit=1)[1]
    snapshot_source = snapshot_source.split("rclpy.init()", maxsplit=1)[0]
    assert "validate_formal_plan(parsed.capture_plan)" in snapshot_source
    assert "capture plan or active source revalidation failed" in snapshot_source
    assert 'parser.add_argument(\n        "--dataset-dir"' not in source
    assert "def _plan_snapshot_reason():" in source
    assert "capture plan changed after LIVE startup" in source
    assert "exact_single_advance" in source
    assert "base_tcp_available=node.base_tcp_available()" in source
    assert "elif capture_plan is None or dataset_dir is None:" in source
    assert "Torque request blocked while a Sample operation is active" in source
