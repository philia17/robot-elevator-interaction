import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from roarm_safe_driver.calibration_core import (
    ACTIVE_SOURCE_RELATIVE_PATHS, AGX_ACTIVE_SOURCE_PATHS,
    JointSample, MAX_SETTLE_DELTA_RAD, MAX_TAG_SAMPLE_GAP_NS,
    MIN_TAG_SAMPLES, MONITOR_STATUS_SOURCE, TagObservation,
    SAFE_DRIVER_STATUS_SOURCE, SafetyError, TAG_WINDOW_NS, joint_sample,
    append_integrity_indexes, nearest_joint_sample, raw_directory_lock,
    rebuild_manifest, canonical_integrity_index_hashes, validate_settled,
    validate_transform,
    validate_active_source_hashes,
    validate_preappend_integrity_chain,
    validate_safe_status_window, validate_status_alignment,
    validate_status_payload, validate_raw_integrity, tag_observation, validate_tag_window,
    write_sample_exclusive,
)
from roarm_safe_driver.core import JOINT_NAMES


def _samples(*, span=0.0, gap_ns=50_000_000):
    base = 2_000_000_000
    return tuple(JointSample(base - 1_000_000_000 + index * gap_ns, (span if index == 0 else 0.0,) * 6) for index in range(21))


def test_nearest_joint_pair_accepts_49ms_and_rejects_51ms():
    rgb = 1_000_000_000
    accepted = nearest_joint_sample((JointSample(rgb - 49_000_000, (0.0,) * 6),), rgb)
    assert accepted.stamp_ns == rgb - 49_000_000
    with pytest.raises(SafetyError, match="skew"):
        nearest_joint_sample((JointSample(rgb - 51_000_000, (0.0,) * 6),), rgb)


def test_nearest_joint_pair_breaks_tie_toward_earlier_stamp():
    rgb = 1_000_000_000
    result = nearest_joint_sample((JointSample(rgb + 10, (0.0,) * 6), JointSample(rgb - 10, (0.0,) * 6)), rgb)
    assert result.stamp_ns == rgb - 10


def test_joint_names_are_reordered_and_exact():
    sample = joint_sample(1, tuple(reversed(JOINT_NAMES)), tuple(range(6)))
    assert sample.positions == (5.0, 4.0, 3.0, 2.0, 1.0, 0.0)


def test_settle_window_accepts_measured_contiguous_window():
    samples = _samples(span=MAX_SETTLE_DELTA_RAD)
    assert len(validate_settled(samples, samples[-1].stamp_ns)) == 21


def test_settle_window_accepts_30hz_samples_without_exact_one_second_boundary():
    base = 2_000_000_000
    samples = tuple(
        JointSample(base - 1_023_000_000 + index * 33_000_000, (0.0,) * 6)
        for index in range(33)
    )
    assert len(validate_settled(samples, samples[-1].stamp_ns)) == 32


@pytest.mark.parametrize("kwargs", [{"span": MAX_SETTLE_DELTA_RAD + 1e-5}, {"gap_ns": 101_000_000}])
def test_settle_window_rejects_motion_or_gap(kwargs):
    samples = _samples(**kwargs)
    with pytest.raises(SafetyError):
        validate_settled(samples, samples[-1].stamp_ns)


def test_transform_requires_finite_unit_quaternion():
    assert validate_transform((0, 0, 1), (0, 0, 0, 1), "tag") == ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])
    with pytest.raises(SafetyError):
        validate_transform((0, 0, 1), (0, 0, 0, 2), "tag")


def _yaw_quaternion(degrees):
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _tag_samples(
    *,
    count=21,
    gap_ns=50_000_000,
    translation_step_m=0.00005,
    rotation_step_deg=0.01,
):
    start = 1_000_000_000
    return tuple(
        tag_observation(
            start + index * gap_ns,
            (
                0.1 + (index - count // 2) * translation_step_m,
                -0.2,
                0.4,
            ),
            _yaw_quaternion((index - count // 2) * rotation_step_deg),
            family="tag36h11",
            tag_id=0,
            hamming=0,
            decision_margin=100.0 + index,
        )
        for index in range(count)
    )


def _tag_samples_with_gap():
    samples = _tag_samples(count=25)
    return tuple(
        TagObservation(
            stamp_ns=item.stamp_ns + (201_000_000 if index >= 22 else 0),
            translation_m=item.translation_m,
            rotation_xyzw=item.rotation_xyzw,
            family=item.family,
            tag_id=item.tag_id,
            hamming=item.hamming,
            decision_margin=item.decision_margin,
        )
        for index, item in enumerate(samples)
    )


def test_tag_window_aggregates_one_second_of_static_observations():
    samples = _tag_samples()
    result = validate_tag_window(samples, samples[-1].stamp_ns)
    assert result.observations == samples
    assert result.duration_ns == 1_000_000_000
    assert result.translation_m == pytest.approx((0.1, -0.2, 0.4))
    assert result.rotation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert result.translation_spread_m == pytest.approx(0.001)
    assert result.rotation_spread_deg == pytest.approx(0.2)
    assert result.aggregation_method == (
        "translation_component_median;"
        "quaternion_hemisphere_aligned_normalized_component_mean"
    )


def test_tag_window_accepts_measured_fifteen_hz_cadence():
    samples = _tag_samples(count=16, gap_ns=67_000_000)
    result = validate_tag_window(samples, samples[-1].stamp_ns)
    assert len(result.observations) == 16
    assert result.duration_ns == 1_005_000_000


def test_tag_window_policy_constants_match_current_r4_to_r6_contract():
    assert TAG_WINDOW_NS == 1_000_000_000
    assert MIN_TAG_SAMPLES == 10
    assert MAX_TAG_SAMPLE_GAP_NS == 250_000_000


def test_tag_window_accepts_exactly_250ms_gap():
    stamps = [1_000_000_000]
    for index in range(15):
        gap_ns = 250_000_000 if index == 7 else 55_000_000
        stamps.append(stamps[-1] + gap_ns)
    samples = tuple(
        TagObservation(
            stamp_ns=stamp,
            translation_m=(0.1, -0.2, 0.4),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            family="tag36h11",
            tag_id=0,
            hamming=0,
            decision_margin=100.0,
        )
        for stamp in stamps
    )
    assert len(validate_tag_window(samples, samples[-1].stamp_ns).observations) == 16


def test_tag_window_accepts_current_minimum_of_ten_observations():
    samples = _tag_samples(count=10, gap_ns=112_000_000)
    result = validate_tag_window(samples, samples[-1].stamp_ns)
    assert len(result.observations) == 10
    assert result.duration_ns == 1_008_000_000


def test_tag_window_rejects_full_duration_with_only_nine_observations():
    samples = _tag_samples(count=9, gap_ns=125_000_000)
    with pytest.raises(SafetyError, match="fewer than 10"):
        validate_tag_window(samples, samples[-1].stamp_ns)


def test_tag_window_hemisphere_aligns_equivalent_quaternions():
    samples = list(_tag_samples(rotation_step_deg=0.0))
    samples[::2] = [
        TagObservation(
            stamp_ns=item.stamp_ns,
            translation_m=item.translation_m,
            rotation_xyzw=tuple(-value for value in item.rotation_xyzw),
            family=item.family,
            tag_id=item.tag_id,
            hamming=item.hamming,
            decision_margin=item.decision_margin,
        )
        for item in samples[::2]
    ]
    result = validate_tag_window(samples, samples[-1].stamp_ns)
    assert result.rotation_xyzw == pytest.approx((0.0, 0.0, 0.0, -1.0))
    assert result.rotation_spread_deg == pytest.approx(0.0)


@pytest.mark.parametrize(
        ("samples", "message"),
    [
        (_tag_samples(count=20), "cover full"),
        (_tag_samples_with_gap(), "gap"),
        (_tag_samples(translation_step_m=0.00016), "translation spread"),
        (_tag_samples(rotation_step_deg=0.08), "rotation spread"),
    ],
)
def test_tag_window_rejects_incomplete_gapped_or_unstable_windows(samples, message):
    with pytest.raises(SafetyError, match=message):
        validate_tag_window(samples, samples[-1].stamp_ns)


@pytest.mark.parametrize(
    "overrides",
    [
        {"family": "tag25h9"},
        {"tag_id": 1},
        {"hamming": 1},
        {"decision_margin": 49.99},
        {"decision_margin": float("nan")},
    ],
)
def test_tag_observation_rejects_wrong_or_low_quality_detection(overrides):
    values = {
        "family": "tag36h11",
        "tag_id": 0,
        "hamming": 0,
        "decision_margin": 50.0,
    }
    values.update(overrides)
    with pytest.raises(SafetyError):
        tag_observation(
            1,
            (0.0, 0.0, 0.4),
            (0.0, 0.0, 0.0, 1.0),
            **values,
        )


def test_monitor_status_preserves_direct_torque_evidence():
    result = validate_status_payload(
        {
            "state": "MONITOR_ONLY_NO_WRITES",
            "torque_enabled": [True] * 6,
            "voltage": 12.12,
        },
        MONITOR_STATUS_SOURCE,
    )
    assert result["torque_enabled"] == [True] * 6
    assert result["torque_evidence"] == {
        "source_topic": "/roarm_monitor/status",
        "mode": "direct_t1051_joint_flags",
        "direct_joint_torque_flags_available": True,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"state": "MONITOR_ONLY_NO_WRITES", "torque_enabled": [True] * 5, "voltage": 12.1},
        {"state": "MONITOR_ONLY_NO_WRITES", "torque_enabled": [True] * 5 + [False], "voltage": 12.1},
        {"state": "MONITOR_FAULT", "torque_enabled": [True] * 6, "voltage": 12.1},
        {"state": "MONITOR_ONLY_NO_WRITES", "torque_enabled": [True] * 6, "voltage": 0.0},
    ],
)
def test_monitor_status_rejects_unhealthy_payload(payload):
    with pytest.raises(SafetyError):
        validate_status_payload(payload, MONITOR_STATUS_SOURCE)


def _safe_status(**overrides):
    payload = {
        "state": "CONVERGED_DISARMED",
        "armed": False,
        "dry_run": False,
        "fault": "",
        "feedback_stamp_ns": 1_784_878_598_537_000_000,
        "positions_rad": [0.0, 0.1, 1.6, -0.2, 3.0, 0.04],
        "torque_enabled": [True] * 6,
        "voltage": 12.10,
    }
    payload.update(overrides)
    return payload


def test_safe_driver_status_preserves_sample_aligned_direct_feedback():
    result = validate_status_payload(_safe_status(), SAFE_DRIVER_STATUS_SOURCE)
    assert result["voltage_v"] == 12.10
    assert result["torque_enabled"] == [True] * 6
    assert result["feedback_stamp_ns"] == 1_784_878_598_537_000_000
    assert result["positions_rad"] == [0.0, 0.1, 1.6, -0.2, 3.0, 0.04]
    evidence = result["torque_evidence"]
    assert evidence["source_topic"] == "/roarm_safe/feedback_status"
    assert evidence["mode"] == "sample_aligned_t1051_joint_flags"
    assert evidence["direct_joint_torque_flags_available"] is True
    assert evidence["safe_driver_state"] == "CONVERGED_DISARMED"


@pytest.mark.parametrize(
    "state",
    ["DISARMED", "DISARMED:operator", "CONVERGED_DISARMED", "FAULT_CLEARED_DISARMED"],
)
def test_safe_driver_status_accepts_only_explicit_disarmed_terminal_states(state):
    result = validate_status_payload(
        _safe_status(state=state), SAFE_DRIVER_STATUS_SOURCE
    )
    assert result["torque_evidence"]["safe_driver_state"] == state


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": "COMMAND_ACTIVE"},
        {"armed": True},
        {"dry_run": True},
        {"fault": "one or more joint torque switches are off"},
        {"feedback_stamp_ns": 0},
        {"positions_rad": [0.0] * 5},
        {"positions_rad": [0.0] * 5 + [float("nan")]},
        {"voltage": 10.99},
        {"torque_enabled": [True] * 5 + [False]},
    ],
)
def test_safe_driver_status_fails_closed_on_schema_or_health_change(overrides):
    with pytest.raises(SafetyError):
        validate_status_payload(_safe_status(**overrides), SAFE_DRIVER_STATUS_SOURCE)


def test_status_source_must_be_explicit_and_known():
    with pytest.raises(SafetyError, match="unknown"):
        validate_status_payload(_safe_status(), "auto")


def test_safe_status_alignment_requires_exact_stamp_and_positions():
    joint = JointSample(
        1_784_878_598_537_000_000,
        (0.0, 0.1, 1.6, -0.2, 3.0, 0.04),
    )
    evidence = validate_status_payload(
        _safe_status(), SAFE_DRIVER_STATUS_SOURCE
    )
    validate_status_alignment(evidence, joint)
    with pytest.raises(SafetyError, match="timestamp"):
        validate_status_alignment(
            {**evidence, "feedback_stamp_ns": joint.stamp_ns + 1}, joint
        )
    with pytest.raises(SafetyError, match="positions"):
        validate_status_alignment(
            {**evidence, "positions_rad": [0.0] * 6}, joint
        )


def test_hand_teach_settle_window_requires_healthy_status_for_every_frame():
    samples = _samples()
    payload_by_stamp = {
        sample.stamp_ns: _safe_status(
            feedback_stamp_ns=sample.stamp_ns,
            positions_rad=list(sample.positions),
        )
        for sample in samples
    }
    result = validate_safe_status_window(samples, payload_by_stamp)
    assert len(result) == len(samples)


def test_hand_teach_settle_window_rejects_torque_off_history_even_if_final_is_healthy():
    samples = _samples()
    payload_by_stamp = {
        sample.stamp_ns: _safe_status(
            feedback_stamp_ns=sample.stamp_ns,
            positions_rad=list(sample.positions),
        )
        for sample in samples
    }
    payload_by_stamp[samples[0].stamp_ns] = {
        **payload_by_stamp[samples[0].stamp_ns],
        "torque_enabled": [False] * 6,
        "fault": "one or more joint torque switches are off",
    }
    assert payload_by_stamp[samples[-1].stamp_ns]["torque_enabled"] == [True] * 6
    with pytest.raises(SafetyError, match="fault|torque"):
        validate_safe_status_window(samples, payload_by_stamp)


def test_hand_teach_settle_window_rejects_missing_status_frame():
    samples = _samples()
    payload_by_stamp = {
        sample.stamp_ns: _safe_status(
            feedback_stamp_ns=sample.stamp_ns,
            positions_rad=list(sample.positions),
        )
        for sample in samples[1:]
    }
    with pytest.raises(SafetyError, match="missing"):
        validate_safe_status_window(samples, payload_by_stamp)


def test_hand_teach_settle_window_rejects_misaligned_status_frame():
    samples = _samples()
    payload_by_stamp = {
        sample.stamp_ns: _safe_status(
            feedback_stamp_ns=sample.stamp_ns,
            positions_rad=list(sample.positions),
        )
        for sample in samples
    }
    payload_by_stamp[samples[5].stamp_ns] = {
        **payload_by_stamp[samples[5].stamp_ns],
        "positions_rad": [0.1] * 6,
    }
    with pytest.raises(SafetyError, match="positions"):
        validate_safe_status_window(samples, payload_by_stamp)


def _append_raw_sample(raw_dir, dataset_id, sample_id, document):
    with raw_directory_lock(raw_dir):
        existing = validate_raw_integrity(raw_dir, dataset_id=dataset_id)
        path = write_sample_exclusive(raw_dir, sample_id, document)
        manifest, sums = append_integrity_indexes(
            raw_dir,
            dataset_id=dataset_id,
            new_sample_path=path,
            existing_hashes=existing,
        )
    return path, manifest, sums


def test_raw_sample_is_exclusive_and_manifest_is_parseable(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    document = {"sample_id": "S0001", "value": 1}
    path, manifest, sums = _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        document,
    )
    with pytest.raises(SafetyError, match="overwrite"):
        write_sample_exclusive(tmp_path, "S0001", document)
    assert json.loads(manifest.read_text())["raw_samples"][0]["file"] == path.name
    assert path.name in sums.read_text()
    assert validate_raw_integrity(
        tmp_path, dataset_id=dataset_id
    ) == {path.name: json.loads(manifest.read_text())["raw_samples"][0]["sha256"]}


def test_raw_integrity_rejects_modified_or_unindexed_samples(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    first, _, _ = _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        {"sample_id": "S0001"},
    )
    original = first.read_bytes()
    first.write_text('{"sample_id":"S0001","modified":true}\n')
    with pytest.raises(SafetyError, match="digest"):
        validate_raw_integrity(tmp_path, dataset_id=dataset_id)
    with pytest.raises(SafetyError, match="established raw dataset"):
        rebuild_manifest(tmp_path, dataset_id=dataset_id)

    first.write_bytes(original)
    assert validate_raw_integrity(tmp_path, dataset_id=dataset_id)
    write_sample_exclusive(tmp_path, "S0002", {"sample_id": "S0002"})
    with pytest.raises(SafetyError, match="file set"):
        validate_raw_integrity(tmp_path, dataset_id=dataset_id)


def test_manifest_rebuild_refuses_existing_raw_even_if_indexes_are_removed(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        {"sample_id": "S0001"},
    )
    (tmp_path / "manifest.json").unlink()
    (tmp_path / "SHA256SUMS").unlink()

    with pytest.raises(SafetyError, match="established raw dataset"):
        rebuild_manifest(tmp_path, dataset_id=dataset_id)


def test_raw_integrity_rejects_manifest_and_checksum_disagreement(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    _, _, sums_path = _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        {"sample_id": "S0001"},
    )
    sums_path.write_text(f"{'0' * 64}  S0001.json\n")
    with pytest.raises(SafetyError, match="digest"):
        validate_raw_integrity(tmp_path, dataset_id=dataset_id)


def test_prefix_preserving_integrity_append_commits_one_new_sample(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    first, _, _ = _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        {"sample_id": "S0001"},
    )
    with raw_directory_lock(tmp_path):
        existing = validate_raw_integrity(tmp_path, dataset_id=dataset_id)
        second = write_sample_exclusive(
            tmp_path,
            "S0002",
            {"sample_id": "S0002"},
        )
        manifest, _ = append_integrity_indexes(
            tmp_path,
            dataset_id=dataset_id,
            new_sample_path=second,
            existing_hashes=existing,
        )
    entries = json.loads(manifest.read_text())["raw_samples"]
    assert [entry["file"] for entry in entries] == [first.name, second.name]
    assert entries[0]["sha256"] == existing[first.name]
    assert validate_raw_integrity(tmp_path, dataset_id=dataset_id)[
        first.name
    ] == existing[first.name]


def test_preappend_chain_rejects_rebaselined_prior_hash(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        {"sample_id": "S0001"},
    )
    existing = validate_raw_integrity(tmp_path, dataset_id=dataset_id)
    manifest_sha256, sums_sha256 = canonical_integrity_index_hashes(
        dataset_id,
        existing,
    )
    second, _, _ = _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0002",
        {
            "sample_id": "S0002",
            "metadata": {
                "preappend_raw_integrity": {
                    "existing_sample_count": 1,
                    "existing_sample_hashes_sha256": existing,
                    "manifest_sha256": manifest_sha256,
                    "sha256sums_sha256": sums_sha256,
                }
            },
        },
    )
    current = validate_raw_integrity(tmp_path, dataset_id=dataset_id)
    validate_preappend_integrity_chain(
        tmp_path,
        dataset_id=dataset_id,
        existing_hashes=current,
        formal_start=2,
    )

    document = json.loads(second.read_text())
    document["metadata"]["preappend_raw_integrity"][
        "existing_sample_hashes_sha256"
    ]["S0001.json"] = "0" * 64
    second.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    rebaselined = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tmp_path.glob("S*.json"))
    }
    entries = [
        {"file": filename, "sha256": digest}
        for filename, digest in rebaselined.items()
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "raw_samples": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{entry['sha256']}  {entry['file']}\n"
            for entry in entries
        )
    )
    rebaselined = validate_raw_integrity(tmp_path, dataset_id=dataset_id)
    with pytest.raises(SafetyError, match="pre-append raw hash chain"):
        validate_preappend_integrity_chain(
            tmp_path,
            dataset_id=dataset_id,
            existing_hashes=rebaselined,
            formal_start=2,
        )


def test_raw_integrity_rejects_duplicate_json_keys_and_stale_transaction(tmp_path):
    dataset_id = "gemini335_roarm_test"
    rebuild_manifest(tmp_path, dataset_id=dataset_id)
    _, manifest, _ = _append_raw_sample(
        tmp_path,
        dataset_id,
        "S0001",
        {"sample_id": "S0001"},
    )
    original_manifest = manifest.read_bytes()
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,'
        f'"dataset_id":"{dataset_id}","raw_samples":[]}}\n'
    )
    with pytest.raises(SafetyError, match="duplicate JSON key"):
        validate_raw_integrity(tmp_path, dataset_id=dataset_id)

    manifest.write_bytes(original_manifest)
    (tmp_path / ".S0002.json.interrupted").write_text("partial")
    with pytest.raises(SafetyError, match="stale"):
        validate_raw_integrity(tmp_path, dataset_id=dataset_id)


def test_active_source_hash_validation_fails_closed_on_drift(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "repository"
    plan_path = repository / "config/calibration/formal/plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}\n")
    source_path = repository / "source.py"
    source_path.write_text("VALUE = 1\n")
    monkeypatch.setattr(
        "roarm_safe_driver.calibration_core.ACTIVE_SOURCE_RELATIVE_PATHS",
        {"source.py": "source.py"},
    )
    monkeypatch.setattr(
        "roarm_safe_driver.calibration_core.AGX_ACTIVE_SOURCE_PATHS",
        {},
    )
    expected = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert validate_active_source_hashes(
        plan_path,
        {"source.py": expected},
    ) == {"source.py": source_path}

    source_path.write_text("VALUE = 2\n")
    with pytest.raises(SafetyError, match="active frozen source hash mismatch"):
        validate_active_source_hashes(
            plan_path,
            {"source.py": expected},
        )


def test_active_source_contract_matches_formal_validator():
    repository = Path(__file__).resolve().parents[3]
    validator_path = (
        repository / "tools/calibration/validate_formal_capture_plan.py"
    )
    if not validator_path.is_file():
        validator_path = Path(
            "/opt/robot-elevator-interaction/tools/calibration/"
            "validate_formal_capture_plan.py"
        )
    spec = importlib.util.spec_from_file_location(
        "formal_capture_plan_validator_contract",
        validator_path,
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)

    assert set(ACTIVE_SOURCE_RELATIVE_PATHS) == set(validator.SOURCE_HASH_PATHS)
    assert {
        name: str(path.relative_to(validator.REPOSITORY))
        for name, path in validator.SOURCE_HASH_PATHS.items()
    } == ACTIVE_SOURCE_RELATIVE_PATHS
    assert {
        name: str(path)
        for name, path in validator.AGX_SOURCE_FALLBACKS.items()
    } == AGX_ACTIVE_SOURCE_PATHS
