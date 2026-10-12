import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.calibration import build_fingertip_c0_candidate as builder


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _axis_angle_xyzw(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    xyz = axis * math.sin(angle / 2.0)
    return [float(value) for value in xyz] + [math.cos(angle / 2.0)]


def _rotation_xyzw(quaternion):
    return builder.rotation_xyzw(quaternion, "test quaternion")


ROTATIONS = (
    _axis_angle_xyzw((1, 0, 0), 0.0),
    _axis_angle_xyzw((1, 0, 0), 0.35),
    _axis_angle_xyzw((0, 1, 0), -0.40),
    _axis_angle_xyzw((0, 0, 1), 0.50),
    _axis_angle_xyzw((1, 1, 0), 0.60),
    _axis_angle_xyzw((0, 1, 1), -0.55),
    _axis_angle_xyzw((1, 0, 1), 0.45),
    _axis_angle_xyzw((1, 2, 3), -0.70),
    _axis_angle_xyzw((2, 1, 0), 0.30),
    _axis_angle_xyzw((0, 2, 1), -0.30),
    _axis_angle_xyzw((1, 0, 2), 0.25),
    _axis_angle_xyzw((2, 3, 1), -0.25),
)


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _make_bundle(root: Path) -> Path:
    return _write_json(
        root / "section_b.json",
        {
            "calibration_id": "test-section-b-relative",
            "dataset_id": "test-b",
            "method": "scipy_separable_absolute",
            "status": "b_section_accepted_relative_unpublished",
            "execution_ready": False,
            "publish_authorized": False,
            "transforms": {
                "T_roarm_base_link_camera_optical_frame": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            },
        },
    )


def _make_sources(root: Path):
    paths = {}
    for name in ("plan.md", "capture.py", "solver.py"):
        path = root / name
        path.write_text(f"fixture {name}\n", encoding="utf-8")
        paths[name] = path
    return paths


def _make_session(
    root: Path,
    session_id: str,
    hand_tip,
    fixed_point,
    bundle_path: Path,
    *,
    held_out_offset_m=(0.0, 0.0, 0.0),
) -> Path:
    session = root / session_id
    session.mkdir(parents=True)
    bundle_sha = _sha(bundle_path)
    manifest = {
        "schema": "fixed_contact_pivot_v1",
        "calibration_id": "test-section-b-relative",
        "bundle_sha256": bundle_sha,
        "frozen_split": {
            "train_contact_indices": [1, 8],
            "held_out_contact_indices": [9, 12],
        },
    }
    manifest_path = _write_json(session / "session_manifest.json", manifest)

    anchor_rgb = session / "target_anchor_raw_rgb.png"
    anchor_rgb.write_bytes(f"anchor-{session_id}".encode())
    _write_json(
        session / "target_anchor.json",
        {
            "schema": "fixed_contact_pivot_target_anchor_v1",
            "operator_confirmed_tip_clear": True,
            "image": {
                "raw_rgb": anchor_rgb.name,
                "raw_rgb_sha256": _sha(anchor_rgb),
            },
        },
    )

    hand_tip = np.asarray(hand_tip, dtype=float)
    fixed_point = np.asarray(fixed_point, dtype=float)
    held_out_offset = np.asarray(held_out_offset_m, dtype=float)
    for index, quaternion in enumerate(ROTATIONS, start=1):
        sample_id = f"C{index:03d}"
        raw_rgb = session / "contact_samples" / f"{sample_id}_raw_rgb.png"
        raw_rgb.parent.mkdir(parents=True, exist_ok=True)
        raw_rgb.write_bytes(f"{session_id}-{sample_id}".encode())
        rotation = _rotation_xyzw(quaternion)
        translation = fixed_point - rotation @ hand_tip
        if index >= 9:
            translation = translation + held_out_offset
        _write_json(
            session / "contact_samples" / f"{sample_id}.json",
            {
                "schema": "fixed_contact_pivot_contact_sample_v1",
                "sample_id": sample_id,
                "split": "train" if index <= 8 else "held_out",
                "operator_contact_confirmed": True,
                "base_fixed_confirmed": True,
                "image": {
                    "raw_rgb": raw_rgb.name,
                    "raw_rgb_sha256": _sha(raw_rgb),
                },
                "T_roarm_base_link_hand_tcp": {
                    "translation_m": translation.tolist(),
                    "rotation_xyzw": quaternion,
                    "stamp_ns": index * 1_000_000_000,
                },
                "bundle": {
                    "calibration_id": "test-section-b-relative",
                    "sha256": bundle_sha,
                },
            },
        )
    _write_json(
        session / "pivot_result.json",
        {
            "schema": "fixed_contact_pivot_result_v1",
            "session_manifest_sha256": _sha(manifest_path),
            "acceptance": {"status": "diagnostic_only"},
        },
    )
    (session / "operator_notes.txt").write_text(
        f"complete inventory marker {session_id}\n", encoding="utf-8"
    )
    return session


def _build_fixture(tmp_path, *, held_out_offset=(0.0, 0.0, 0.0)):
    hand_tip = np.array([0.12697, 0.01054, 0.00496])
    bundle = _make_bundle(tmp_path)
    sources = _make_sources(tmp_path)
    sessions = {
        "r3": _make_session(
            tmp_path,
            "r3",
            hand_tip,
            [0.40, -0.10, 0.25],
            bundle,
            held_out_offset_m=held_out_offset,
        ),
        "r5": _make_session(
            tmp_path,
            "r5",
            hand_tip,
            [0.42, 0.08, 0.27],
            bundle,
            held_out_offset_m=held_out_offset,
        ),
    }
    kwargs = {
        "session_paths": sessions,
        "section_b_bundle": bundle,
        "plan_path": sources["plan.md"],
        "capture_script": sources["capture.py"],
        "single_session_solver": sources["solver.py"],
        "expected_translation_m": hand_tip,
        "expected_tolerance_m": 1e-9,
        "artifact_id": "test-c0-r3-r5-v1",
        "created_at": "2026-07-29T12:00:00+00:00",
        "repository_root": tmp_path,
    }
    return hand_tip, kwargs


def test_builds_complete_inventory_and_fail_closed_candidate(tmp_path):
    hand_tip, kwargs = _build_fixture(tmp_path)
    document = builder.build_document(**kwargs)

    assert document["candidate_transform"]["translation_m"] == pytest.approx(
        hand_tip
    )
    assert document["candidate_transform"]["rotation_xyzw"] is None
    assert document["candidate_transform"]["transform_complete"] is False
    assert document["calibration_id"] is None
    assert all(value is False for value in document["execution_flags"].values())
    assert document["derivation"]["fit_inputs"] == "train_only"
    assert document["derivation"]["recompute_matches_plan"] is True
    assert document["derivation"]["metrics"]["aggregate"]["held_out"][
        "max_mm"
    ] == pytest.approx(0.0, abs=1e-9)

    for session in document["source_sessions"]:
        paths = {item["relative_path"] for item in session["content_files"]}
        assert "operator_notes.txt" in paths
        assert "contact_samples/C001.json" in paths
        assert "contact_samples/C012_raw_rgb.png" in paths
        assert session["integrity_verified"] is True
        assert session["content_manifest_sha256"] == builder.canonical_sha256(
            session["content_files"]
        )


def test_held_out_samples_do_not_influence_train_only_solution(tmp_path):
    hand_tip, kwargs = _build_fixture(
        tmp_path, held_out_offset=(0.020, -0.010, 0.005)
    )
    document = builder.build_document(**kwargs)

    assert document["candidate_transform"]["translation_m"] == pytest.approx(
        hand_tip, abs=1e-10
    )
    held_out = document["derivation"]["metrics"]["aggregate"]["held_out"]
    assert held_out["rms_mm"] == pytest.approx(
        np.linalg.norm([0.020, -0.010, 0.005]) * 1000.0
    )
    assert document["derivation"]["metrics"]["aggregate"]["train"][
        "max_mm"
    ] == pytest.approx(0.0, abs=1e-9)


def test_rejects_embedded_rgb_hash_mismatch(tmp_path):
    _hand_tip, kwargs = _build_fixture(tmp_path)
    bad_rgb = (
        kwargs["session_paths"]["r5"]
        / "contact_samples"
        / "C010_raw_rgb.png"
    )
    bad_rgb.write_bytes(b"tampered")

    with pytest.raises(builder.CandidateError, match="raw_rgb_sha256"):
        builder.build_document(**kwargs)


def test_exclusive_pair_refuses_overwrite_and_cleans_partial_reservation(tmp_path):
    hand_tip, kwargs = _build_fixture(tmp_path)
    document = builder.build_document(**kwargs)
    output = tmp_path / "candidate" / "candidate.json"
    sums = tmp_path / "candidate" / "SHA256SUMS"

    artifact_sha, _sums_sha = builder.write_exclusive_pair(
        output, sums, document
    )
    original_output = output.read_bytes()
    original_sums = sums.read_bytes()
    assert artifact_sha == hashlib.sha256(original_output).hexdigest()
    assert original_sums.decode() == f"{artifact_sha}  {output.name}\n"

    with pytest.raises(FileExistsError):
        builder.write_exclusive_pair(output, sums, document)
    assert output.read_bytes() == original_output
    assert sums.read_bytes() == original_sums

    second_output = tmp_path / "other" / "candidate.json"
    second_sums = tmp_path / "other" / "SHA256SUMS"
    second_sums.parent.mkdir(parents=True)
    second_sums.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        builder.write_exclusive_pair(second_output, second_sums, document)
    assert not second_output.exists()
    assert second_sums.read_text(encoding="utf-8") == "existing\n"
