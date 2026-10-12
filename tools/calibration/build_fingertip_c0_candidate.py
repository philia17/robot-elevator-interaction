#!/usr/bin/env python3
"""Build an immutable, non-executable Section-C fingertip C0 candidate.

The builder reads two fixed-contact pivot sessions, verifies their complete
file inventories and embedded RGB hashes, and solves one train-only least
squares problem with a shared hand-frame fingertip point and one independent
base-frame fixed point per session.  It never publishes TF, connects to ROS, or
commands hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_SESSION_IDS = ("r3", "r5")
TRAIN_SAMPLE_IDS = tuple(f"C{index:03d}" for index in range(1, 9))
HELD_OUT_SAMPLE_IDS = tuple(f"C{index:03d}" for index in range(9, 13))
ALL_SAMPLE_IDS = TRAIN_SAMPLE_IDS + HELD_OUT_SAMPLE_IDS
EXPECTED_SESSION_SCHEMA = "fixed_contact_pivot_v1"
EXPECTED_SAMPLE_SCHEMA = "fixed_contact_pivot_contact_sample_v1"
EXPECTED_ANCHOR_SCHEMA = "fixed_contact_pivot_target_anchor_v1"
EXPECTED_RESULT_SCHEMA = "fixed_contact_pivot_result_v1"

# Backward-compatible alias used by existing tests/callers.
SESSION_IDS = DEFAULT_SESSION_IDS


def normalize_session_ids(session_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(str(item).strip() for item in session_ids)
    if len(ids) != 2 or any(not item for item in ids):
        raise CandidateError("joint solve requires exactly two non-empty session ids")
    if len(set(ids)) != 2:
        raise CandidateError("joint solve requires two distinct session ids")
    return ids


class CandidateError(ValueError):
    """A source artifact is incomplete, inconsistent, or unsafe to bind."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateError(f"{label} must be a JSON object: {path}")
    return value


def finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise CandidateError(f"{label} must contain exactly {length} values")
    if any(isinstance(item, bool) for item in value):
        raise CandidateError(f"{label} must contain finite numbers")
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"{label} must contain finite numbers") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise CandidateError(f"{label} must contain finite numbers")
    return vector


def rotation_xyzw(value: Any, label: str) -> np.ndarray:
    quaternion = finite_vector(value, 4, label)
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise CandidateError(f"{label} must be a unit XYZW quaternion")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-8)
    ):
        raise CandidateError(f"{label} does not form a proper rotation")
    return rotation


def checked_source_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise CandidateError(f"{label} must name a relative source file")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise CandidateError(f"{label} must be relative")
    root_resolved = root.resolve()
    candidate = (root_resolved / candidate_relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise CandidateError(f"{label} escapes the session directory") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise CandidateError(f"{label} does not resolve to a regular file")
    return candidate


def verify_embedded_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise CandidateError(f"{label} must contain a SHA-256 digest")
    actual = sha256_file(path)
    if actual != expected:
        raise CandidateError(f"{label} does not match {path}")
    return actual


def complete_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    root = root.resolve()
    if not root.is_dir():
        raise CandidateError(f"session directory does not exist: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise CandidateError(f"session inventory contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CandidateError(f"session inventory contains a special file: {path}")
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise CandidateError(f"session inventory is empty: {root}")
    return entries, canonical_sha256(entries)


def validate_section_b_bundle(path: Path) -> dict[str, Any]:
    document = load_json(path, "section-B bundle")
    calibration_id = document.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise CandidateError("section-B bundle calibration_id is required")
    if document.get("execution_ready") is not False:
        raise CandidateError("section-B bundle must remain execution_ready=false")
    if document.get("publish_authorized") is not False:
        raise CandidateError("section-B bundle must remain publish_authorized=false")
    transforms = document.get("transforms")
    if (
        not isinstance(transforms, dict)
        or "T_roarm_base_link_camera_optical_frame" not in transforms
    ):
        raise CandidateError("section-B bundle is missing T_roarm_base_link_camera_optical_frame")
    return {
        "calibration_id": calibration_id,
        "dataset_id": document.get("dataset_id"),
        "method": document.get("method"),
        "status": document.get("status"),
        "bundle_path": str(path),
        "bundle_sha256": sha256_file(path),
    }


def validate_session(
    session_id: str,
    root: Path,
    section_b: Mapping[str, Any],
    *,
    allowed_session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    allowed = tuple(allowed_session_ids or DEFAULT_SESSION_IDS)
    if session_id not in allowed:
        raise CandidateError(f"unexpected session id: {session_id}")
    root = root.resolve()
    inventory_before, inventory_sha_before = complete_inventory(root)
    manifest_path = root / "session_manifest.json"
    anchor_path = root / "target_anchor.json"
    result_path = root / "pivot_result.json"
    manifest = load_json(manifest_path, f"{session_id} manifest")
    if manifest.get("schema") != EXPECTED_SESSION_SCHEMA:
        raise CandidateError(f"{session_id} manifest schema is invalid")
    if manifest.get("calibration_id") != section_b["calibration_id"]:
        raise CandidateError(f"{session_id} manifest calibration_id mismatch")
    if manifest.get("bundle_sha256") != section_b["bundle_sha256"]:
        raise CandidateError(f"{session_id} manifest bundle SHA mismatch")
    split = manifest.get("frozen_split")
    if (
        not isinstance(split, dict)
        or split.get("train_contact_indices") != [1, 8]
        or split.get("held_out_contact_indices") != [9, 12]
    ):
        raise CandidateError(f"{session_id} frozen 8+4 split is invalid")

    anchor = load_json(anchor_path, f"{session_id} target anchor")
    if (
        anchor.get("schema") != EXPECTED_ANCHOR_SCHEMA
        or anchor.get("operator_confirmed_tip_clear") is not True
    ):
        raise CandidateError(f"{session_id} target anchor is invalid")
    anchor_image = anchor.get("image")
    if not isinstance(anchor_image, dict):
        raise CandidateError(f"{session_id} target anchor image is missing")
    anchor_rgb = checked_source_file(
        root, anchor_image.get("raw_rgb"), f"{session_id} anchor raw_rgb"
    )
    anchor_rgb_sha = verify_embedded_hash(
        anchor_rgb,
        anchor_image.get("raw_rgb_sha256"),
        f"{session_id} anchor raw_rgb_sha256",
    )

    sample_dir = root / "contact_samples"
    actual_sample_ids = tuple(
        path.stem for path in sorted(sample_dir.glob("C*.json"))
    )
    if actual_sample_ids != ALL_SAMPLE_IDS:
        raise CandidateError(
            f"{session_id} must contain exactly C001-C012 JSON samples"
        )
    samples: list[dict[str, Any]] = []
    for sample_id in ALL_SAMPLE_IDS:
        sample_path = sample_dir / f"{sample_id}.json"
        sample = load_json(sample_path, f"{session_id}/{sample_id}")
        expected_split = "train" if sample_id in TRAIN_SAMPLE_IDS else "held_out"
        if (
            sample.get("schema") != EXPECTED_SAMPLE_SCHEMA
            or sample.get("sample_id") != sample_id
            or sample.get("split") != expected_split
            or sample.get("operator_contact_confirmed") is not True
            or sample.get("base_fixed_confirmed") is not True
        ):
            raise CandidateError(f"{session_id}/{sample_id} contract is invalid")
        image = sample.get("image")
        if not isinstance(image, dict):
            raise CandidateError(f"{session_id}/{sample_id} image is missing")
        raw_rgb = checked_source_file(
            sample_dir,
            image.get("raw_rgb"),
            f"{session_id}/{sample_id} raw_rgb",
        )
        verify_embedded_hash(
            raw_rgb,
            image.get("raw_rgb_sha256"),
            f"{session_id}/{sample_id} raw_rgb_sha256",
        )
        transform = sample.get("T_roarm_base_link_hand_tcp")
        if not isinstance(transform, dict):
            raise CandidateError(f"{session_id}/{sample_id} transform is missing")
        translation = finite_vector(
            transform.get("translation_m"),
            3,
            f"{session_id}/{sample_id} translation_m",
        )
        rotation = rotation_xyzw(
            transform.get("rotation_xyzw"),
            f"{session_id}/{sample_id} rotation_xyzw",
        )
        stamp_ns = transform.get("stamp_ns")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns <= 0:
            raise CandidateError(f"{session_id}/{sample_id} TF stamp is invalid")
        bundle = sample.get("bundle")
        if (
            not isinstance(bundle, dict)
            or bundle.get("calibration_id") != section_b["calibration_id"]
            or bundle.get("sha256") != section_b["bundle_sha256"]
        ):
            raise CandidateError(f"{session_id}/{sample_id} bundle binding mismatch")
        samples.append(
            {
                "sample_id": sample_id,
                "split": expected_split,
                "rotation": rotation,
                "translation": translation,
            }
        )

    result = load_json(result_path, f"{session_id} pivot result")
    manifest_sha = sha256_file(manifest_path)
    if (
        result.get("schema") != EXPECTED_RESULT_SCHEMA
        or result.get("session_manifest_sha256") != manifest_sha
        or (result.get("acceptance") or {}).get("status") != "diagnostic_only"
    ):
        raise CandidateError(f"{session_id} pivot result binding is invalid")

    inventory, inventory_sha = complete_inventory(root)
    if inventory != inventory_before or inventory_sha != inventory_sha_before:
        raise CandidateError(f"{session_id} changed while it was being validated")
    return {
        "session_id": session_id,
        "uri": str(root),
        "session_manifest_sha256": manifest_sha,
        "target_anchor_json_sha256": sha256_file(anchor_path),
        "target_anchor_rgb_sha256": anchor_rgb_sha,
        "pivot_result_sha256": sha256_file(result_path),
        "content_manifest_sha256": inventory_sha,
        "content_files": inventory,
        "integrity_verified": True,
        "_samples": samples,
    }


def residual_summary(residuals: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not residuals:
        return {"count": 0}
    values = np.asarray([item["norm_mm"] for item in residuals], dtype=float)
    return {
        "count": int(len(values)),
        "mean_mm": float(np.mean(values)),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "median_mm": float(np.median(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(np.max(values)),
    }


def solve_joint_train_only(
    sessions: Sequence[dict[str, Any]],
    *,
    session_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    expected_ids = normalize_session_ids(session_ids or DEFAULT_SESSION_IDS)
    actual_ids = tuple(session["session_id"] for session in sessions)
    if actual_ids != expected_ids:
        raise CandidateError(
            f"joint solve requires sessions ordered exactly {', '.join(expected_ids)}"
        )
    variable_count = 3 + 3 * len(sessions)
    blocks: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for session_index, session in enumerate(sessions):
        train = [
            sample for sample in session["_samples"] if sample["split"] == "train"
        ]
        if tuple(item["sample_id"] for item in train) != TRAIN_SAMPLE_IDS:
            raise CandidateError(f"{session['session_id']} train split is incomplete")
        for sample in train:
            block = np.zeros((3, variable_count), dtype=float)
            block[:, :3] = sample["rotation"]
            start = 3 + 3 * session_index
            block[:, start : start + 3] = -np.eye(3)
            blocks.append(block)
            rhs.append(-sample["translation"])
    matrix = np.vstack(blocks)
    vector = np.hstack(rhs)
    rank = int(np.linalg.matrix_rank(matrix))
    label = "+".join(expected_ids)
    if rank != variable_count:
        raise CandidateError(
            f"joint {label} design is rank deficient ({rank}/{variable_count})"
        )
    solution, _residuals, _rank, singular_values = np.linalg.lstsq(
        matrix, vector, rcond=None
    )
    if not np.all(np.isfinite(solution)):
        raise CandidateError(f"joint {label} solution is non-finite")
    hand_tip = solution[:3]
    fixed_points = {
        session["session_id"]: solution[
            3 + 3 * index : 3 + 3 * index + 3
        ]
        for index, session in enumerate(sessions)
    }
    diagnostics = {
        "design_rank": rank,
        "design_dimension": variable_count,
        "design_condition": float(np.linalg.cond(matrix)),
        "design_singular_values": singular_values.tolist(),
    }
    return hand_tip, fixed_points, diagnostics


def evaluate_splits(
    sessions: Sequence[dict[str, Any]],
    hand_tip: np.ndarray,
    fixed_points: Mapping[str, np.ndarray],
    *,
    session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    ordered_ids = normalize_session_ids(
        session_ids or tuple(session["session_id"] for session in sessions)
    )
    output: dict[str, Any] = {"by_session": {}, "aggregate": {}}
    for session in sessions:
        session_id = session["session_id"]
        output["by_session"][session_id] = {}
        for split in ("train", "held_out"):
            residuals = []
            for sample in session["_samples"]:
                if sample["split"] != split:
                    continue
                vector = (
                    sample["rotation"] @ hand_tip
                    + sample["translation"]
                    - fixed_points[session_id]
                )
                residuals.append(
                    {
                        "sample_id": sample["sample_id"],
                        "vector_m": vector.tolist(),
                        "norm_mm": float(np.linalg.norm(vector) * 1000.0),
                    }
                )
            output["by_session"][session_id][split] = {
                "residuals": residuals,
                "summary": residual_summary(residuals),
            }
    for split in ("train", "held_out"):
        aggregate = [
            residual
            for session_id in ordered_ids
            for residual in output["by_session"][session_id][split]["residuals"]
        ]
        output["aggregate"][split] = residual_summary(aggregate)
    return output


def display_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_document(
    *,
    session_paths: Mapping[str, Path],
    section_b_bundle: Path,
    plan_path: Path,
    capture_script: Path,
    single_session_solver: Path,
    expected_translation_m: Sequence[float] | None,
    expected_tolerance_m: float,
    artifact_id: str,
    created_at: str,
    repository_root: Path,
    session_ids: Sequence[str] | None = None,
    accept_solved_translation: bool = False,
    excluded_sessions: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    if not artifact_id.strip():
        raise CandidateError("artifact_id is required")
    if expected_tolerance_m < 0.0 or not math.isfinite(expected_tolerance_m):
        raise CandidateError("expected_tolerance_m must be finite and non-negative")
    ordered_ids = normalize_session_ids(session_ids or tuple(session_paths.keys()))
    if set(session_paths.keys()) != set(ordered_ids):
        raise CandidateError("session_paths keys must match session_ids")
    for session_id in ordered_ids:
        if session_id not in session_paths:
            raise CandidateError(f"missing session path for {session_id}")
    if expected_translation_m is None and not accept_solved_translation:
        raise CandidateError(
            "expected_translation_m is required unless accept_solved_translation=True"
        )
    required_sources = {
        "plan_snapshot": plan_path,
        "capture_script": capture_script,
        "single_session_solver": single_session_solver,
        "builder": Path(__file__),
    }
    for label, path in required_sources.items():
        if path.is_symlink() or not path.is_file():
            raise CandidateError(f"{label} source file is missing: {path}")

    section_b = validate_section_b_bundle(section_b_bundle)
    sessions = [
        validate_session(
            session_id,
            session_paths[session_id],
            section_b,
            allowed_session_ids=ordered_ids,
        )
        for session_id in ordered_ids
    ]
    hand_tip, fixed_points, diagnostics = solve_joint_train_only(
        sessions, session_ids=ordered_ids
    )
    if accept_solved_translation and expected_translation_m is None:
        expected = hand_tip.copy()
        difference = np.zeros(3, dtype=float)
        difference_norm = 0.0
        recompute_matches_plan = True
    else:
        expected = finite_vector(
            list(expected_translation_m), 3, "expected_translation_m"
        )
        difference = hand_tip - expected
        difference_norm = float(np.linalg.norm(difference))
        if difference_norm > expected_tolerance_m + 1e-12:
            raise CandidateError(
                "recomputed "
                + "+".join(ordered_ids)
                + " translation does not match the frozen plan value"
            )
        recompute_matches_plan = True
    metrics = evaluate_splits(
        sessions, hand_tip, fixed_points, session_ids=ordered_ids
    )
    public_sessions = [
        {key: value for key, value in session.items() if not key.startswith("_")}
        for session in sessions
    ]
    source_code = {
        label: {
            "path": display_path(path, repository_root),
            "sha256": sha256_file(path),
        }
        for label, path in required_sources.items()
    }
    if excluded_sessions is None:
        if ordered_ids == DEFAULT_SESSION_IDS:
            excluded_sessions = [
                {
                    "session_id": "r4",
                    "reason": "weak rotation coverage; not pooled",
                }
            ]
        else:
            excluded_sessions = []
    return {
        "schema": "section_c_fingertip_translation_candidate_v1",
        "schema_version": 1,
        "artifact_id": artifact_id,
        "created_at": created_at,
        "status": "diagnostic_translation_only",
        "calibration_id": None,
        "coordinate_contract_version": "1.0.0",
        "notation": {
            "transform": "T_parent_child",
            "meaning": (
                "coordinates in child map into parent: "
                "p_parent = T_parent_child * p_child"
            ),
            "quaternion_order": "XYZW",
            "length_unit": "m",
        },
        "candidate_transform": {
            "id": "T_hand_tcp_fingertip_candidate",
            "parent_frame": "hand_tcp",
            "child_frame": "fingertip_contact_point_candidate",
            "translation_m": hand_tip.tolist(),
            "rotation_xyzw": None,
            "rotation_status": "unmeasured",
            "transform_complete": False,
            "tf_publishable": False,
            "point_equation": (
                "p_base_tip_i = R_base_hand_i * p_hand_tip + t_base_hand_i"
            ),
        },
        "derivation": {
            "method": (
                "joint_linear_least_squares_shared_hand_tip_"
                "separate_fixed_base_point_per_session"
            ),
            "fit_inputs": "train_only",
            "included_sessions": list(ordered_ids),
            "excluded_sessions": list(excluded_sessions),
            "frozen_split": {
                "train_sample_ids": list(TRAIN_SAMPLE_IDS),
                "held_out_sample_ids": list(HELD_OUT_SAMPLE_IDS),
            },
            "separate_fixed_points_base_m": {
                key: value.tolist() for key, value in fixed_points.items()
            },
            "diagnostics": diagnostics,
            "plan_recorded_translation_m": expected.tolist(),
            "recompute_minus_plan_m": difference.tolist(),
            "recompute_difference_norm_m": difference_norm,
            "recompute_tolerance_m": expected_tolerance_m,
            "recompute_matches_plan": recompute_matches_plan,
            "accept_solved_translation": bool(accept_solved_translation),
            "metrics": metrics,
        },
        "source_section_b": {
            **section_b,
            "status_scope": "accepted_relative_unpublished",
        },
        "source_sessions": public_sessions,
        "source_code": source_code,
        "missing_geometry": {
            "rotation_xyzw": "unmeasured",
            "contact_surface_normal_in_fingertip": "unmeasured",
            "contact_patch_extent": "unmeasured",
            "translation_covariance_m2": "not_estimated",
            "installation_repeatability": "unverified",
            "fingertip_installation_id": None,
        },
        "acceptance": {
            "status": "diagnostic_only",
            "translation_thresholds_preregistered": False,
            "rotation_accepted": False,
            "contact_surface_accepted": False,
            "section_c_complete": False,
        },
        "execution_flags": {
            "execution_ready": False,
            "publish_authorized": False,
            "tf_static_publish_allowed": False,
            "motion_authorized": False,
            "contact_authorized": False,
            "press_authorized": False,
        },
        "allowed_uses": [
            "no-motion overlay",
            "offline fingertip point projection",
            "offline reachability diagnostics",
        ],
        "prohibited_uses": [
            "publishing hand_tcp to press_contact TF",
            "constructing an executable contact pose",
            "arming or commanding motion",
            "claiming Section C acceptance",
            "setting execution_ready true",
        ],
    }


def json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_exclusive_pair(
    output_path: Path,
    sums_path: Path,
    document: Mapping[str, Any],
) -> tuple[str, str]:
    output_path = output_path.resolve()
    sums_path = sums_path.resolve()
    if output_path == sums_path:
        raise CandidateError("artifact and SHA256SUMS paths must differ")
    for path in (output_path, sums_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(document)
    artifact_sha = hashlib.sha256(payload).hexdigest()
    sums_payload = f"{artifact_sha}  {output_path.name}\n".encode("utf-8")

    reserved: list[tuple[Path, int]] = []
    try:
        for path in (output_path, sums_path):
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            reserved.append((path, descriptor))
    except Exception:
        for path, descriptor in reserved:
            os.close(descriptor)
            path.unlink(missing_ok=True)
        raise

    try:
        for (_path, descriptor), content in zip(
            reserved, (payload, sums_payload)
        ):
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        reserved.clear()
    except Exception:
        for path, descriptor in reserved:
            try:
                os.close(descriptor)
            except OSError:
                pass
            path.unlink(missing_ok=True)
        raise
    return artifact_sha, hashlib.sha256(sums_payload).hexdigest()


def reject_outputs_inside_sources(
    output_paths: Sequence[Path],
    source_roots: Sequence[Path],
) -> None:
    resolved_roots = [root.resolve() for root in source_roots]
    for output_path in output_paths:
        resolved_output = output_path.resolve()
        for root in resolved_roots:
            try:
                resolved_output.relative_to(root)
            except ValueError:
                continue
            raise CandidateError(
                f"output must not be written inside source session: {resolved_output}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-ids", nargs=2, default=None, metavar=("ID_A", "ID_B"))
    parser.add_argument(
        "--session-dirs",
        nargs=2,
        type=Path,
        default=None,
        metavar=("DIR_A", "DIR_B"),
    )
    parser.add_argument("--r3-dir", type=Path, default=None)
    parser.add_argument("--r5-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sums-output", type=Path, default=None)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--created-at",
        default=datetime.now(timezone.utc).isoformat(),
    )
    parser.add_argument(
        "--section-b-bundle",
        type=Path,
        default=(
            repository
            / "config/calibration/accepted/"
            "gemini335_roarm_20260727_r2_eye_on_base_relative_bundle.json"
        ),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=repository / "docs/camera_to_arm_press_calibration_plan.md",
    )
    parser.add_argument(
        "--capture-script",
        type=Path,
        default=repository / "tools/live_button_arm_coord_view.py",
    )
    parser.add_argument(
        "--single-session-solver",
        type=Path,
        default=repository / "tools/calibration/fixed_contact_pivot_solver.py",
    )
    parser.add_argument(
        "--expected-translation-m",
        nargs=3,
        type=float,
        default=None,
    )
    parser.add_argument("--expected-tolerance-mm", type=float, default=0.10)
    parser.add_argument(
        "--accept-solved-translation",
        action="store_true",
        help=(
            "Freeze plan_recorded_translation_m from this solve "
            "(for new session pairs such as r6+r7)."
        ),
    )
    return parser.parse_args(argv)


def resolve_session_paths(args: argparse.Namespace) -> tuple[tuple[str, str], dict[str, Path]]:
    if args.session_dirs is not None:
        ids = normalize_session_ids(args.session_ids or ("r6", "r7"))
        return ids, {ids[0]: args.session_dirs[0], ids[1]: args.session_dirs[1]}
    if args.r3_dir is None or args.r5_dir is None:
        raise SystemExit(
            "provide --r3-dir/--r5-dir, or --session-dirs with optional --session-ids"
        )
    ids = normalize_session_ids(args.session_ids or DEFAULT_SESSION_IDS)
    if ids != DEFAULT_SESSION_IDS:
        raise SystemExit("--r3-dir/--r5-dir require session ids r3 r5 (or omit --session-ids)")
    return ids, {"r3": args.r3_dir, "r5": args.r5_dir}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_tolerance_mm < 0.0:
        raise SystemExit("--expected-tolerance-mm must be non-negative")
    if args.expected_translation_m is None and not args.accept_solved_translation:
        # Preserve historical default for the frozen r3+r5 candidate workflow.
        if args.session_dirs is None and (
            args.session_ids is None or tuple(args.session_ids) == DEFAULT_SESSION_IDS
        ):
            args.expected_translation_m = (0.12697, 0.01054, 0.00496)
        else:
            raise SystemExit(
                "new session pairs require --expected-translation-m "
                "or --accept-solved-translation"
            )
    repository = Path(__file__).resolve().parents[2]
    session_ids, session_paths = resolve_session_paths(args)
    document = build_document(
        session_paths=session_paths,
        section_b_bundle=args.section_b_bundle,
        plan_path=args.plan,
        capture_script=args.capture_script,
        single_session_solver=args.single_session_solver,
        expected_translation_m=args.expected_translation_m,
        expected_tolerance_m=args.expected_tolerance_mm / 1000.0,
        artifact_id=args.artifact_id,
        created_at=args.created_at,
        repository_root=repository,
        session_ids=session_ids,
        accept_solved_translation=args.accept_solved_translation,
    )
    sums_path = args.sums_output or args.output.parent / "SHA256SUMS"
    reject_outputs_inside_sources(
        (args.output, sums_path),
        tuple(session_paths.values()),
    )
    artifact_sha, sums_sha = write_exclusive_pair(
        args.output, sums_path, document
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "artifact_sha256": artifact_sha,
                "sums_output": str(sums_path.resolve()),
                "sums_sha256": sums_sha,
                "status": document["status"],
                "translation_m": document["candidate_transform"]["translation_m"],
                "execution_ready": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
