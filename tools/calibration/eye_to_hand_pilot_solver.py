#!/usr/bin/env python3
"""Read-only pilot solver for Gemini 335 -> RoArm eye-to-hand calibration.

Coordinate convention (frozen by docs/coordinate_contract.md):

    T_parent_child maps child coordinates into parent coordinates.
    C_T_Tag_i = C_T_B * B_T_H_i * H_T_Tag

Quaternions in persisted input/output are XYZW.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


METHODS = (
    "opencv_hand_eye_tsai",
    "opencv_hand_eye_park",
    "opencv_robot_world_hand_eye_shah",
    "opencv_robot_world_hand_eye_li",
    "relative_motion_park",
    "scipy_separable_absolute",
)


def transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float)
    result[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return result


def from_xyzw(data: dict[str, Any]) -> np.ndarray:
    rotation = Rotation.from_quat(np.asarray(data["rotation_xyzw"], dtype=float)).as_matrix()
    return transform(rotation, data["translation_m"])


def inverse(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ value[:3, 3])
    return result


def transform_dict(value: np.ndarray) -> dict[str, list[float]]:
    quat = Rotation.from_matrix(value[:3, :3]).as_quat()
    if quat[3] < 0.0:
        quat = -quat
    return {
        "translation_m": value[:3, 3].tolist(),
        "rotation_xyzw": quat.tolist(),
    }


def rotation_angle_rad(value: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(value[:3, :3]).as_rotvec()))


def relative_error(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    delta = inverse(observed) @ predicted
    return float(np.linalg.norm(delta[:3, 3])), math.degrees(rotation_angle_rad(delta))


def mean_transform(values: Iterable[np.ndarray]) -> np.ndarray:
    values = list(values)
    rotations = Rotation.from_matrix(np.stack([value[:3, :3] for value in values]))
    return transform(rotations.mean().as_matrix(), np.mean([value[:3, 3] for value in values], axis=0))


def _opencv_robot_world_hand_eye(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    method: int,
) -> tuple[np.ndarray, np.ndarray]:
    # OpenCV solves A X = Z B.  The explicit mapping is:
    #   A = Tag_T_C = inverse(C_T_Tag)
    #   B = H_T_B   = inverse(B_T_H)
    #   X = C_T_B
    #   Z = Tag_T_H = inverse(H_T_Tag)
    tag_t_c = [inverse(value) for value in tags]
    h_t_b = [inverse(value) for value in hands]
    r_x, t_x, r_z, t_z = cv2.calibrateRobotWorldHandEye(
        [value[:3, :3] for value in tag_t_c],
        [value[:3, 3].reshape(3, 1) for value in tag_t_c],
        [value[:3, :3] for value in h_t_b],
        [value[:3, 3].reshape(3, 1) for value in h_t_b],
        method=method,
    )
    c_t_b = transform(r_x, t_x)
    tag_t_h = transform(r_z, t_z)
    return c_t_b, inverse(tag_t_h)


def _opencv_hand_eye(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    method: int,
) -> tuple[np.ndarray, np.ndarray]:
    # OpenCV's documented eye-to-hand substitution is:
    #   input "gripper2base" = H_T_B = inverse(B_T_H)
    #   input target2cam     = C_T_Tag
    #   output cam2gripper   = B_T_C
    # Thus C_T_B is the explicit inverse of the documented output.
    h_t_b = [inverse(value) for value in hands]
    r_b_t_c, t_b_t_c = cv2.calibrateHandEye(
        [value[:3, :3] for value in h_t_b],
        [value[:3, 3].reshape(3, 1) for value in h_t_b],
        [value[:3, :3] for value in tags],
        [value[:3, 3].reshape(3, 1) for value in tags],
        method=method,
    )
    c_t_b = inverse(transform(r_b_t_c, t_b_t_c))
    h_t_tags = [inverse(hand) @ inverse(c_t_b) @ tag for hand, tag in zip(hands, tags)]
    return c_t_b, mean_transform(h_t_tags)


def solve_relative_motion_park(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve pairwise L X = X R, then recover the constant H_T_Tag."""
    left_rotvecs: list[np.ndarray] = []
    right_rotvecs: list[np.ndarray] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            left = tags[i] @ inverse(tags[j])
            right = hands[i] @ inverse(hands[j])
            left_rotvecs.append(Rotation.from_matrix(left[:3, :3]).as_rotvec())
            right_rotvecs.append(Rotation.from_matrix(right[:3, :3]).as_rotvec())
            pairs.append((left, right))

    cross_covariance = sum(
        np.outer(left, right)
        for left, right in zip(left_rotvecs, right_rotvecs)
    )
    u, _, vt = np.linalg.svd(cross_covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    r_x = u @ correction @ vt

    lhs: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for left, right in pairs:
        lhs.append(left[:3, :3] - np.eye(3))
        rhs.append(r_x @ right[:3, 3] - left[:3, 3])
    t_x, _, _, _ = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)
    c_t_b = transform(r_x, t_x)
    h_t_tags = [inverse(hand) @ inverse(c_t_b) @ tag for hand, tag in zip(hands, tags)]
    return c_t_b, mean_transform(h_t_tags)


def _pack(c_t_b: np.ndarray, h_t_tag: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            Rotation.from_matrix(c_t_b[:3, :3]).as_rotvec(),
            c_t_b[:3, 3],
            Rotation.from_matrix(h_t_tag[:3, :3]).as_rotvec(),
            h_t_tag[:3, 3],
        )
    )


def _unpack(parameters: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(parameters, dtype=float)
    return (
        transform(Rotation.from_rotvec(values[0:3]).as_matrix(), values[3:6]),
        transform(Rotation.from_rotvec(values[6:9]).as_matrix(), values[9:12]),
    )


def joint_residuals(
    parameters: Sequence[float],
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
) -> np.ndarray:
    c_t_b, h_t_tag = _unpack(parameters)
    residuals: list[float] = []
    for hand, observed in zip(hands, tags):
        predicted = c_t_b @ hand @ h_t_tag
        delta = inverse(observed) @ predicted
        residuals.extend(delta[:3, 3])
        residuals.extend(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
    return np.asarray(residuals, dtype=float)


def solve_scipy_separable(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Solve absolute rotations on SO(3), then solve both translations linearly."""
    initial_x, initial_z = solve_relative_motion_park(hands, tags)

    def rotation_residuals(parameters: Sequence[float]) -> np.ndarray:
        r_x = Rotation.from_rotvec(parameters[:3]).as_matrix()
        r_z = Rotation.from_rotvec(parameters[3:6]).as_matrix()
        output: list[float] = []
        for hand, observed in zip(hands, tags):
            predicted = r_x @ hand[:3, :3] @ r_z
            delta = observed[:3, :3].T @ predicted
            output.extend(Rotation.from_matrix(delta).as_rotvec())
        return np.asarray(output)

    result = least_squares(
        rotation_residuals,
        np.concatenate(
            (
                Rotation.from_matrix(initial_x[:3, :3]).as_rotvec(),
                Rotation.from_matrix(initial_z[:3, :3]).as_rotvec(),
            )
        ),
        method="trf",
        x_scale="jac",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=4000,
    )
    r_x = Rotation.from_rotvec(result.x[:3]).as_matrix()
    r_z = Rotation.from_rotvec(result.x[3:6]).as_matrix()
    translation_lhs: list[np.ndarray] = []
    translation_rhs: list[np.ndarray] = []
    for hand, observed in zip(hands, tags):
        translation_lhs.append(np.hstack((np.eye(3), r_x @ hand[:3, :3])))
        translation_rhs.append(observed[:3, 3] - r_x @ hand[:3, 3])
    translations, _, _, _ = np.linalg.lstsq(
        np.vstack(translation_lhs), np.concatenate(translation_rhs), rcond=None
    )
    c_t_b = transform(r_x, translations[:3])
    h_t_tag = transform(r_z, translations[3:])
    return c_t_b, h_t_tag, result


def solve_method(
    method: str,
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if method == "opencv_hand_eye_tsai":
        return _opencv_hand_eye(hands, tags, cv2.CALIB_HAND_EYE_TSAI)
    if method == "opencv_hand_eye_park":
        return _opencv_hand_eye(hands, tags, cv2.CALIB_HAND_EYE_PARK)
    if method == "opencv_robot_world_hand_eye_shah":
        return _opencv_robot_world_hand_eye(
            hands, tags, cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )
    if method == "opencv_robot_world_hand_eye_li":
        return _opencv_robot_world_hand_eye(
            hands, tags, cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI
        )
    if method == "relative_motion_park":
        return solve_relative_motion_park(hands, tags)
    if method == "scipy_separable_absolute":
        c_t_b, h_t_tag, _ = solve_scipy_separable(hands, tags)
        return c_t_b, h_t_tag
    raise ValueError(f"unknown method: {method}")


def scalar_stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(array**2))),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def method_result(
    method: str,
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    c_t_b, h_t_tag = solve_method(method, hands, tags)
    per_sample: list[dict[str, Any]] = []
    translation: list[float] = []
    rotation: list[float] = []
    for sample_id, hand, observed in zip(sample_ids, hands, tags):
        predicted = c_t_b @ hand @ h_t_tag
        translation_m, rotation_deg = relative_error(observed, predicted)
        translation.append(translation_m)
        rotation.append(rotation_deg)
        per_sample.append(
            {
                "sample_id": sample_id,
                "predicted_C_T_Tag": transform_dict(predicted),
                "translation_residual_m": translation_m,
                "rotation_residual_deg": rotation_deg,
            }
        )
    return {
        "C_T_B": transform_dict(c_t_b),
        "B_T_C": transform_dict(inverse(c_t_b)),
        "H_T_Tag": transform_dict(h_t_tag),
        "per_sample": per_sample,
        "translation_residual_m": scalar_stats(translation),
        "rotation_residual_deg": scalar_stats(rotation),
    }


def leave_one_out(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in METHODS:
        folds: list[dict[str, Any]] = []
        for held_index, sample_id in enumerate(sample_ids):
            train_hands = [value for index, value in enumerate(hands) if index != held_index]
            train_tags = [value for index, value in enumerate(tags) if index != held_index]
            try:
                c_t_b, h_t_tag = solve_method(method, train_hands, train_tags)
                predicted = c_t_b @ hands[held_index] @ h_t_tag
                translation_m, rotation_deg = relative_error(tags[held_index], predicted)
                folds.append(
                    {
                        "held_out_sample_id": sample_id,
                        "translation_residual_m": translation_m,
                        "rotation_residual_deg": rotation_deg,
                    }
                )
            except Exception as exc:  # fail closed and retain the failed fold
                folds.append({"held_out_sample_id": sample_id, "error": repr(exc)})
        valid = [fold for fold in folds if "error" not in fold]
        output[method] = {
            "folds": folds,
            "translation_residual_m": scalar_stats(
                [fold["translation_residual_m"] for fold in valid]
            )
            if valid
            else None,
            "rotation_residual_deg": scalar_stats(
                [fold["rotation_residual_deg"] for fold in valid]
            )
            if valid
            else None,
            "failed_fold_count": len(folds) - len(valid),
        }
    return output


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    nonzero = singular_values[singular_values > tolerance]
    return {
        "shape": list(matrix.shape),
        "rank": int(np.sum(singular_values > tolerance)),
        "singular_values": singular_values.tolist(),
        "condition_number_nonzero": float(nonzero[0] / nonzero[-1])
        if len(nonzero)
        else math.inf,
    }


def observability(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    axes: list[np.ndarray] = []
    rotation_vectors: list[np.ndarray] = []
    translation_blocks: list[np.ndarray] = []
    kronecker_blocks: list[np.ndarray] = []
    near_duplicates: list[list[str]] = []
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            robot_relative = hands[i] @ inverse(hands[j])
            camera_relative = tags[i] @ inverse(tags[j])
            rotvec = Rotation.from_matrix(robot_relative[:3, :3]).as_rotvec()
            rotation_deg = math.degrees(float(np.linalg.norm(rotvec)))
            translation_m = float(np.linalg.norm(robot_relative[:3, 3]))
            axis = (
                rotvec / np.linalg.norm(rotvec)
                if np.linalg.norm(rotvec) > 1e-12
                else np.zeros(3)
            )
            pairs.append(
                {
                    "sample_ids": [sample_ids[i], sample_ids[j]],
                    "robot_relative_translation_m": translation_m,
                    "robot_relative_rotation_deg": rotation_deg,
                    "robot_relative_rotation_axis": axis.tolist(),
                }
            )
            if translation_m < 0.02 and rotation_deg < 3.0:
                near_duplicates.append([sample_ids[i], sample_ids[j]])
            if np.linalg.norm(axis) > 0.0:
                axes.append(axis)
            rotation_vectors.append(rotvec)
            translation_blocks.append(camera_relative[:3, :3] - np.eye(3))
            kronecker_blocks.append(
                np.kron(np.eye(3), camera_relative[:3, :3])
                - np.kron(robot_relative[:3, :3].T, np.eye(3))
            )

    positions = np.stack([hand[:3, 3] for hand in hands])
    centered_positions = positions - np.mean(positions, axis=0)
    x, z = solve_method("scipy_separable_absolute", hands, tags)
    characteristic_length = float(
        np.median([pair["robot_relative_translation_m"] for pair in pairs])
    )
    packed = _pack(x, z)
    normalized_parameters = packed.copy()
    normalized_parameters[3:6] /= characteristic_length
    normalized_parameters[9:12] /= characteristic_length

    def normalized_joint_residuals(parameters: Sequence[float]) -> np.ndarray:
        physical = np.asarray(parameters, dtype=float).copy()
        physical[3:6] *= characteristic_length
        physical[9:12] *= characteristic_length
        residual = joint_residuals(physical, hands, tags).reshape(-1, 6)
        residual[:, :3] /= characteristic_length
        return residual.reshape(-1)

    normalized = least_squares(
        normalized_joint_residuals,
        normalized_parameters,
        method="trf",
        max_nfev=1,
    )
    return {
        "pairwise": pairs,
        "pairwise_translation_m": scalar_stats(
            [pair["robot_relative_translation_m"] for pair in pairs]
        ),
        "pairwise_rotation_deg": scalar_stats(
            [pair["robot_relative_rotation_deg"] for pair in pairs]
        ),
        "near_duplicate_candidate_rule": {
            "translation_m_less_than": 0.02,
            "rotation_deg_less_than": 3.0,
            "status": "handoff candidate only, not a production acceptance threshold",
        },
        "near_duplicate_pairs": near_duplicates,
        "robot_position_centered_svd": _matrix_diagnostics(centered_positions),
        "relative_rotation_vector_svd": _matrix_diagnostics(
            np.vstack(rotation_vectors)
        ),
        "relative_rotation_axis_svd": _matrix_diagnostics(np.vstack(axes)),
        "relative_rotation_kronecker_system": {
            **_matrix_diagnostics(np.vstack(kronecker_blocks)),
            "ideal_noise_free_rank": 8,
            "ideal_noise_free_nullity": 1,
        },
        "relative_translation_linear_system": _matrix_diagnostics(
            np.vstack(translation_blocks)
        ),
        "dimensionless_joint_residual_jacobian": {
            **_matrix_diagnostics(normalized.jac),
            "translation_normalization_m": characteristic_length,
            "rotation_unit": "rad",
        },
    }


def disagreements(methods: dict[str, Any], field: str) -> dict[str, Any]:
    names = sorted(methods)
    pairwise: list[dict[str, Any]] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            first_t = from_xyzw(methods[first][field])
            second_t = from_xyzw(methods[second][field])
            translation_m, rotation_deg = relative_error(first_t, second_t)
            pairwise.append(
                {
                    "methods": [first, second],
                    "translation_disagreement_m": translation_m,
                    "rotation_disagreement_deg": rotation_deg,
                }
            )
    return {
        "pairwise": pairwise,
        "translation_disagreement_m": scalar_stats(
            [item["translation_disagreement_m"] for item in pairwise]
        ),
        "rotation_disagreement_deg": scalar_stats(
            [item["rotation_disagreement_deg"] for item in pairwise]
        ),
    }


def load_samples(raw_dir: Path) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    samples = [json.loads(path.read_text()) for path in sorted(raw_dir.glob("S*.json"))]
    hands = [from_xyzw(sample["T_roarm_base_link_hand_tcp"]) for sample in samples]
    tags = [from_xyzw(sample["T_camera_optical_frame_calibration_tag"]) for sample in samples]
    return samples, hands, tags


def run_synthetic_direction_test() -> dict[str, Any]:
    known_x = transform(
        Rotation.from_euler("xyz", [17.0, -23.0, 31.0], degrees=True).as_matrix(),
        [0.42, -0.18, 0.27],
    )
    known_z = transform(
        Rotation.from_euler("xyz", [-11.0, 29.0, 7.0], degrees=True).as_matrix(),
        [0.035, -0.024, 0.081],
    )
    hands: list[np.ndarray] = []
    for index in range(10):
        rotation = Rotation.from_euler(
            "xyz",
            [9.0 * index - 21.0, (-1) ** index * (8.0 + index), 13.0 * index - 37.0],
            degrees=True,
        ).as_matrix()
        translation = [
            0.22 + 0.018 * index,
            -0.11 + 0.009 * index * index,
            0.16 + 0.014 * ((index * 3) % 5),
        ]
        hands.append(transform(rotation, translation))
    tags = [known_x @ hand @ known_z for hand in hands]

    results: dict[str, Any] = {}
    for method in METHODS:
        solved_x, solved_z = solve_method(method, hands, tags)
        x_translation, x_rotation = relative_error(known_x, solved_x)
        z_translation, z_rotation = relative_error(known_z, solved_z)
        passed = (
            x_translation < 1e-6
            and x_rotation < 1e-5
            and z_translation < 1e-6
            and z_rotation < 1e-5
        )
        results[method] = {
            "passed": passed,
            "C_T_B_translation_error_m": x_translation,
            "C_T_B_rotation_error_deg": x_rotation,
            "H_T_Tag_translation_error_m": z_translation,
            "H_T_Tag_rotation_error_deg": z_rotation,
        }
        if not passed:
            raise AssertionError(f"synthetic direction test failed for {method}: {results[method]}")
    return {
        "equation": "C_T_Tag_i = C_T_B * B_T_H_i * H_T_Tag",
        "transform_direction": "T_parent_child maps child coordinates into parent",
        "quaternion_order": "XYZW",
        "methods": results,
        "passed": True,
    }


def build_output(raw_dir: Path, source_raw_dir: str | None = None) -> dict[str, Any]:
    synthetic = run_synthetic_direction_test()
    samples, hands, tags = load_samples(raw_dir)
    sample_ids = [sample["sample_id"] for sample in samples]
    methods = {
        method: method_result(method, hands, tags, sample_ids) for method in METHODS
    }
    timestamp_skews_ms = [
        abs(sample["rgb_header_stamp_ns"] - sample["joint_states_stamp_ns"]) / 1e6
        for sample in samples
    ]
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    expected_hashes = {
        entry["file"]: entry["sha256"] for entry in manifest["raw_samples"]
    }
    raw_files: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("S*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hashes.get(path.name) != digest:
            raise ValueError(f"manifest/checksum mismatch for {path.name}")
        stat = path.stat()
        raw_files.append(
            {
                "file": path.name,
                "size_bytes": stat.st_size,
                "mtime_ns_of_audit_copy": stat.st_mtime_ns,
                "sha256": digest,
            }
        )
    sample_audit = []
    for sample, skew_ms in zip(samples, timestamp_skews_ms):
        metadata = sample["metadata"]
        sample_audit.append(
            {
                "sample_id": sample["sample_id"],
                "pose_index": metadata["pose_index"],
                "split": metadata["split"],
                "operator_authorization_id": metadata["operator_authorization_id"],
                "settled": sample["settled"],
                "settle_reason": sample["settle_reason"],
                "tag_family": sample["tag_family"],
                "tag_id": sample["tag_id"],
                "tag_hamming": sample["tag_detection_quality"]["hamming"],
                "tag_decision_margin": sample["tag_detection_quality"]["decision_margin"],
                "torque_enabled": sample["torque_enabled"],
                "voltage_v": sample["voltage_v"],
                "rgb_joint_skew_ms": skew_ms,
                "joint_positions_rad": sample["joint_positions_rad"],
                "status_evidence_schema": (
                    "sample_aligned_safe_driver"
                    if metadata.get("status_source") == "safe_driver"
                    else "historical_recorder"
                ),
            }
        )
    return {
        "schema_version": 1,
        "status": "provisional_pilot_only",
        "execution_ready": False,
        "implementation": {
            "repository_path": "tools/calibration/eye_to_hand_pilot_solver.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "scipy_solver": "scipy.optimize.least_squares",
            "opencv_version": cv2.__version__,
        },
        "coordinate_contract": {
            "version": "1.0.0",
            "notation": "T_parent_child",
            "meaning": "p_parent = T_parent_child * p_child",
            "quaternion_order": "XYZW",
            "solve_equation": "C_T_Tag_i = C_T_B * B_T_H_i * H_T_Tag",
        },
        "synthetic_direction_test": synthetic,
        "dataset": {
            "source_raw_dir": source_raw_dir or str(raw_dir),
            "audit_copy_dir": str(raw_dir),
            "sample_ids": sample_ids,
            "count": len(samples),
            "split": sorted({sample["metadata"]["split"] for sample in samples}),
            "pose_indices": [sample["metadata"]["pose_index"] for sample in samples],
            "rgb_joint_skew_ms": {
                "per_sample": dict(zip(sample_ids, timestamp_skews_ms)),
                **scalar_stats(timestamp_skews_ms),
            },
            "raw_integrity": {
                "manifest_schema_version": manifest["schema_version"],
                "manifest_count": len(manifest["raw_samples"]),
                "all_manifest_hashes_match": True,
                "files": raw_files,
            },
            "sample_audit": sample_audit,
        },
        "methods": methods,
        "solver_disagreement": {
            "C_T_B": disagreements(methods, "C_T_B"),
            "H_T_Tag": disagreements(methods, "H_T_Tag"),
        },
        "leave_one_out": leave_one_out(hands, tags, sample_ids),
        "observability": observability(hands, tags, sample_ids),
        "threshold_context": {
            "formal_pilot_go_no_go_threshold": None,
            "handoff_full_dataset_draft": {
                "held_out_position_median_m_lte": 0.005,
                "held_out_position_p95_m_lte": 0.010,
                "held_out_rotation_p95": "pending; 2 deg is an example only",
            },
            "decision_status": "requires_user_decision",
        },
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "pilot_solver_results.json"
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    result_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (output_dir / "SHA256SUMS").write_text(f"{digest}  {result_path.name}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--source-raw-dir")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    synthetic = run_synthetic_direction_test()
    if args.synthetic_only:
        print(json.dumps(synthetic, indent=2, sort_keys=True))
        return 0
    if args.raw_dir is None or args.output_dir is None:
        parser.error("--raw-dir and --output-dir are required unless --synthetic-only is used")
    payload = build_output(args.raw_dir, args.source_raw_dir)
    write_outputs(args.output_dir, payload)
    print(json.dumps({"output_dir": str(args.output_dir), "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
