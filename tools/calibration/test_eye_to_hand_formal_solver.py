import copy
import hashlib
import json
from pathlib import Path

import pytest
from scipy.spatial.transform import Rotation

from tools.calibration import eye_to_hand_formal_solver as formal
from tools.calibration import eye_to_hand_pilot_solver as pilot


REPOSITORY = Path(__file__).resolve().parents[2]
FORMAL_PLAN = (
    REPOSITORY / "config/calibration/formal/"
    "gemini335_roarm_20260722_r1_b4_formal_capture_plan.json"
)


def _write_integrity(raw_dir, dataset_id):
    entries = []
    for path in sorted(raw_dir.glob("S*.json")):
        entries.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (raw_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": dataset_id,
                "raw_samples": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['file']}\n" for entry in entries),
        encoding="utf-8",
    )


def _safe_evidence(stamp_ns):
    return {
        "armed": False,
        "direct_joint_torque_flags_available": True,
        "dry_run": False,
        "fault": "",
        "feedback_stamp_ns": stamp_ns,
        "mode": "sample_aligned_t1051_joint_flags",
        "safe_driver_state": "FAULT_CLEARED_DISARMED",
        "source_topic": "/roarm_safe/feedback_status",
    }


def _sample_document(
    plan,
    plan_sha256,
    entry,
    hand,
    tag,
    preappend_integrity,
):
    stamp_ns = 2_000_000_000 + entry["capture_order"] * 10_000_000_000
    joint_stamps = [
        stamp_ns - 1_000_000_000 + round(index * 1_000_000_000 / 19)
        for index in range(20)
    ]
    tag_stamps = [
        stamp_ns - 1_000_000_000 + round(index * 1_000_000_000 / 14)
        for index in range(15)
    ]
    evidence = _safe_evidence(stamp_ns)
    driver_versions = {
        key: plan["source_hashes_sha256"][key] for key in formal.DRIVER_SOURCE_KEYS
    }
    return {
        "sample_id": entry["sample_id"],
        "dataset_id": plan["dataset_id"],
        "capture_plan_sha256": plan_sha256,
        "coordinate_contract_version": "1.0.0",
        "calibration_type": "eye_on_base",
        "timestamp_ns": stamp_ns,
        "rgb_header_stamp_ns": stamp_ns,
        "joint_states_stamp_ns": stamp_ns,
        "settled": True,
        "settle_reason": "Measured and safe.",
        "joint_positions_rad": [0.0] * 6,
        "joint_names": [f"joint_{index}" for index in range(6)],
        "voltage_v": 12.1,
        "torque_enabled": [True] * 6,
        "torque_evidence": evidence,
        "T_roarm_base_link_hand_tcp": pilot.transform_dict(hand),
        "T_camera_optical_frame_calibration_tag": pilot.transform_dict(tag),
        "tag_family": "tag36h11",
        "tag_id": 0,
        "tag_edge_m": 0.064,
        "tag_detection_quality": {
            "hamming": 0,
            "decision_margin": 120.0,
        },
        "metadata": {
            "ros_domain_id": plan["ros_domain_id"],
            "capture_plan_sha256": plan_sha256,
            "recorder_timeout_s": 15.0,
            "frozen_source_hashes_sha256": plan["source_hashes_sha256"],
            "driver_versions": driver_versions,
            "preappend_raw_integrity": preappend_integrity,
            "pose_index": entry["pose_index"],
            "split": entry["split"],
            "operator_authorization_id": "formal-test-click",
            "status_source": "safe_driver",
            "status_topic": "/roarm_safe/feedback_status",
            "tag_static_window": {
                "start_stamp_ns": tag_stamps[0],
                "end_stamp_ns": tag_stamps[-1],
                "duration_ns": 1_000_000_000,
                "sample_count": len(tag_stamps),
                "maximum_gap_ns": max(
                    current - previous
                    for previous, current in zip(tag_stamps, tag_stamps[1:])
                ),
                "translation_spread_m": 0.0,
                "rotation_spread_deg": 0.0,
                "aggregation_method": (
                    "translation_component_median;"
                    "quaternion_hemisphere_aligned_normalized_component_mean"
                ),
                "observations": [
                    {
                        "stamp_ns": item_stamp,
                        "translation_m": pilot.transform_dict(tag)["translation_m"],
                        "rotation_xyzw": pilot.transform_dict(tag)["rotation_xyzw"],
                        "family": "tag36h11",
                        "tag_id": 0,
                        "hamming": 0,
                        "decision_margin": 120.0,
                    }
                    for item_stamp in tag_stamps
                ],
            },
            "joint_safe_window": {
                "start_stamp_ns": joint_stamps[0],
                "end_stamp_ns": joint_stamps[-1],
                "duration_ns": 1_000_000_000,
                "sample_count": len(joint_stamps),
                "maximum_gap_ns": max(
                    current - previous
                    for previous, current in zip(joint_stamps, joint_stamps[1:])
                ),
                "per_joint_span_rad": {f"joint_{index}": 0.0 for index in range(6)},
                "maximum_joint_span_rad": 0.0,
                "safe_status_alignment": "all_frames_aligned_healthy",
                "observations": [
                    {
                        "stamp_ns": item_stamp,
                        "positions_rad": [0.0] * 6,
                        "voltage_v": 12.1,
                        "torque_enabled": [True] * 6,
                        "torque_evidence": _safe_evidence(item_stamp),
                    }
                    for item_stamp in joint_stamps
                ],
            },
        },
    }


def _dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    plan = json.loads(FORMAL_PLAN.read_text(encoding="utf-8"))
    plan["dataset_dir"] = str(raw_dir)
    plan_path = tmp_path / "formal_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for sample_number in range(1, 12):
        (raw_dir / f"S{sample_number:04d}.json").write_text(
            json.dumps(
                {
                    "sample_id": f"S{sample_number:04d}",
                    "dataset_id": plan["dataset_id"],
                    "metadata": {"split": "pilot"},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_integrity(raw_dir, plan["dataset_id"])

    known_x = pilot.transform(
        Rotation.from_euler("xyz", [17.0, -23.0, 31.0], degrees=True).as_matrix(),
        [0.42, -0.18, 0.27],
    )
    known_z = pilot.transform(
        Rotation.from_euler("xyz", [-11.0, 29.0, 7.0], degrees=True).as_matrix(),
        [0.035, -0.024, 0.081],
    )
    for index, entry in enumerate(plan["capture_order"]):
        hand = pilot.transform(
            Rotation.from_euler(
                "xyz",
                [
                    (37 * index) % 151 - 75,
                    (53 * index) % 121 - 60,
                    (71 * index) % 171 - 85,
                ],
                degrees=True,
            ).as_matrix(),
            [
                0.18 + 0.025 * ((index * 7) % 11),
                -0.13 + 0.023 * ((index * 5) % 13),
                0.12 + 0.019 * ((index * 9) % 12),
            ],
        )
        tag = known_x @ hand @ known_z
        existing_manifest = json.loads(
            (raw_dir / "manifest.json").read_text(encoding="utf-8")
        )
        existing_hashes = {
            item["file"]: item["sha256"] for item in existing_manifest["raw_samples"]
        }
        preappend_integrity = {
            "existing_sample_count": len(existing_hashes),
            "existing_sample_hashes_sha256": existing_hashes,
            "manifest_sha256": hashlib.sha256(
                (raw_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "sha256sums_sha256": hashlib.sha256(
                (raw_dir / "SHA256SUMS").read_bytes()
            ).hexdigest(),
        }
        document = _sample_document(
            plan,
            plan_sha256,
            entry,
            hand,
            tag,
            preappend_integrity,
        )
        (raw_dir / f"{entry['sample_id']}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_integrity(raw_dir, plan["dataset_id"])
    return plan_path, raw_dir, plan


def _rewrite_sample(raw_dir, sample_id, mutate):
    path = raw_dir / f"{sample_id}.json"
    sample = json.loads(path.read_text(encoding="utf-8"))
    mutate(sample)
    path.write_text(
        json.dumps(sample, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_train_artifact_is_held_out_transform_isolated(tmp_path):
    plan_path, raw_dir, plan = _dataset(tmp_path)
    first = formal.build_train_artifact(plan_path, raw_dir)
    assert first["train_gate_passed"] is True
    assert len(first["train_sample_ids"]) == 22
    assert not set(first["train_sample_ids"]).intersection(
        {
            entry["sample_id"]
            for entry in plan["capture_order"]
            if entry["split"] == "held_out"
        }
    )

    def alter_held_out_transform(sample):
        sample["T_camera_optical_frame_calibration_tag"]["translation_m"][0] += 0.25

    _rewrite_sample(raw_dir, "S0015", alter_held_out_transform)
    _write_integrity(raw_dir, plan["dataset_id"])
    second = formal.build_train_artifact(plan_path, raw_dir)
    assert second == first


def test_held_out_spends_once_and_never_refits(tmp_path, monkeypatch):
    plan_path, raw_dir, _ = _dataset(tmp_path)
    train_artifact = formal.build_train_artifact(plan_path, raw_dir)
    train_path = tmp_path / "train.json"
    formal.write_json_exclusive(train_path, train_artifact)
    spend_path = raw_dir / formal.HELD_OUT_SPEND_FILENAME

    def fail_if_refit(*_args, **_kwargs):
        raise AssertionError("held-out mode attempted to refit")

    original_read_sample = formal._read_sample

    def require_spend_before_held_out_deserialization(*args, **kwargs):
        entry = args[2]
        if entry["split"] == "held_out":
            assert spend_path.is_file()
        return original_read_sample(*args, **kwargs)

    monkeypatch.setattr(formal.pilot, "solve_method", fail_if_refit)
    monkeypatch.setattr(
        formal,
        "_read_sample",
        require_spend_before_held_out_deserialization,
    )
    result = formal.evaluate_held_out(plan_path, raw_dir, train_path)
    assert result["held_out_gate_passed"] is True
    assert len(result["held_out_sample_ids"]) == 8
    assert spend_path.is_file()
    marker = json.loads(spend_path.read_text())
    assert marker["plan_sha256"] == train_artifact["plan_sha256"]
    assert marker["train_artifact_sha256"] == train_artifact["artifact_sha256"]
    assert result["held_out_access_spend_file"] == str(spend_path)
    with pytest.raises(formal.FormalSolverError, match="refusing to overwrite"):
        formal.evaluate_held_out(plan_path, raw_dir, train_path)


def test_cli_rejects_alternate_spend_paths(tmp_path):
    plan_path, raw_dir, _ = _dataset(tmp_path)
    train = formal.build_train_artifact(plan_path, raw_dir)
    train_path = tmp_path / "train.json"
    formal.write_json_exclusive(train_path, train)
    alternate = tmp_path / "alternate-spend.json"
    with pytest.raises(SystemExit):
        formal.main(
            [
                "held-out",
                "--plan",
                str(plan_path),
                "--raw-dir",
                str(raw_dir),
                "--train-artifact",
                str(train_path),
                "--spend-file",
                str(alternate),
                "--output",
                str(tmp_path / "held-out.json"),
            ]
        )
    assert not alternate.exists()
    assert not (raw_dir / formal.HELD_OUT_SPEND_FILENAME).exists()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda raw_dir: _rewrite_sample(
                raw_dir,
                "S0012",
                lambda sample: sample["metadata"].update({"split": "held_out"}),
            ),
            "split provenance mismatch",
        ),
        (
            lambda raw_dir: (raw_dir / "S0041.json").unlink(),
            "exact formal subset S0012..S0041",
        ),
        (
            lambda raw_dir: _rewrite_sample(
                raw_dir,
                "S0012",
                lambda sample: sample["metadata"].update(
                    {"capture_plan_sha256": "0" * 64}
                ),
            ),
            "plan identity or split provenance mismatch",
        ),
        (
            lambda raw_dir: _rewrite_sample(
                raw_dir,
                "S0012",
                lambda sample: sample["metadata"].update(
                    {"frozen_source_hashes_sha256": {"wrong.py": "0" * 64}}
                ),
            ),
            "frozen source hashes mismatch",
        ),
    ],
)
def test_train_rejects_wrong_split_missing_id_and_provenance(tmp_path, change, match):
    plan_path, raw_dir, plan = _dataset(tmp_path)
    change(raw_dir)
    _write_integrity(raw_dir, plan["dataset_id"])
    with pytest.raises(formal.FormalSolverError, match=match):
        formal.build_train_artifact(plan_path, raw_dir)


def test_held_out_provenance_failure_remains_spent(tmp_path):
    plan_path, raw_dir, plan = _dataset(tmp_path)
    train = formal.build_train_artifact(plan_path, raw_dir)
    train_path = tmp_path / "train.json"
    formal.write_json_exclusive(train_path, train)

    _rewrite_sample(
        raw_dir,
        "S0015",
        lambda sample: sample["metadata"].update({"capture_plan_sha256": "f" * 64}),
    )
    _write_integrity(raw_dir, plan["dataset_id"])
    spend_path = raw_dir / formal.HELD_OUT_SPEND_FILENAME
    with pytest.raises(
        formal.FormalSolverError,
        match="plan identity or split provenance mismatch",
    ):
        formal.evaluate_held_out(plan_path, raw_dir, train_path)
    assert json.loads(spend_path.read_text())["status"] == "held_out_access_spent"


def test_train_hashes_every_historical_and_formal_raw_file(tmp_path):
    plan_path, raw_dir, _ = _dataset(tmp_path)
    historical_path = raw_dir / "S0007.json"
    historical_path.write_bytes(historical_path.read_bytes() + b" ")
    with pytest.raises(
        formal.FormalSolverError,
        match="raw sample bytes do not match the integrity indexes",
    ):
        formal.build_train_artifact(plan_path, raw_dir)


def test_plan_method_or_gate_change_is_rejected_before_sample_access(tmp_path):
    plan_path, raw_dir, _ = _dataset(tmp_path)
    plan = json.loads(plan_path.read_text())
    plan["b5_preregistration"] = copy.deepcopy(plan["b5_preregistration"])
    plan["b5_preregistration"]["held_out_rotation_residual_deg"]["maximum_maximum"] = (
        3.1
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(
        formal.FormalSolverError,
        match="b5_preregistration.held_out_rotation_residual_deg changed",
    ):
        formal.build_train_artifact(plan_path, raw_dir)
