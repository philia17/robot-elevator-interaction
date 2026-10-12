#!/usr/bin/env python3
"""Diagnostic eye-in-hand solver for wrist RGB + world-fixed AprilTag.

Pipeline:
  1) Estimate rough camera intrinsics from Tag corners + known edge length.
  2) Re-solve per-sample T_cam_tag with that K.
  3) Run OpenCV calibrateHandEye Park + Tsai (eye-in-hand convention).
  4) Report train / held-out residuals. Never publish TF; all execution flags false.

Eye-in-hand OpenCV convention:
  gripper2base = T_base_hand
  target2cam   = T_cam_tag
  cam2gripper  = T_hand_cam   (= T_hand_tcp_wrist_cam)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from calibration.wrist_april_tag import (  # noqa: E402
    from_xyzw,
    provisional_vga_intrinsics,
    tag_object_corners_m,
    transform,
    transform_dict,
)


def inverse(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ value[:3, 3])
    return result


def rotation_angle_deg(value: np.ndarray) -> float:
    return float(math.degrees(np.linalg.norm(Rotation.from_matrix(value[:3, :3]).as_rotvec())))


def relative_error(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    delta = inverse(observed) @ predicted
    return float(np.linalg.norm(delta[:3, 3])), rotation_angle_deg(delta)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_samples(raw_dir: Path) -> list[dict[str, Any]]:
    samples = []
    for path in sorted(raw_dir.glob("W*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("calibration_type") != "eye_in_hand":
            raise ValueError(f"{path.name}: calibration_type must be eye_in_hand")
        payload["_path"] = str(path)
        samples.append(payload)
    if not samples:
        raise ValueError(f"no W*.json samples under {raw_dir}")
    return samples


def default_held_out_ids(sample_ids: Sequence[str], count: int = 5) -> list[str]:
    if len(sample_ids) <= count:
        raise ValueError("need more samples than held-out count")
    return list(sample_ids[-count:])


def solve_intrinsics(
    samples: Sequence[dict[str, Any]],
    *,
    hfov_deg: float = 90.0,
    mode: str = "wide_fov_fixed_f",
) -> dict[str, Any]:
    """Estimate wrist intrinsics with a wide-FOV prior.

    Free 4-corner Tag calibration on VGA often drifts to absurd fx (~2000).
    Default mode fixes focal length / aspect / principal-point from a 90° prior
    and only fits barrel terms k1/k2.
    """
    object_points = []
    image_points = []
    image_size = None
    edge = None
    for sample in samples:
        corners = np.asarray(sample["tag_corners_px"], dtype=float).reshape(4, 2)
        edge = float(sample["tag_edge_m"])
        width, height = sample["camera"]["resolution"]
        image_size = (int(width), int(height))
        object_points.append(tag_object_corners_m(edge).astype(np.float32).reshape(-1, 1, 3))
        image_points.append(corners.astype(np.float32).reshape(-1, 1, 2))
    assert image_size is not None and edge is not None
    init_k, init_dist = provisional_vga_intrinsics(
        image_size[0], image_size[1], hfov_deg=hfov_deg
    )

    if mode == "wide_fov_fixed_f":
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_FOCAL_LENGTH
            | cv2.CALIB_FIX_PRINCIPAL_POINT
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
            | cv2.CALIB_ZERO_TANGENT_DIST
        )
    elif mode == "wide_fov_soft":
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
            | cv2.CALIB_ZERO_TANGENT_DIST
        )
    else:
        raise ValueError(f"unknown intrinsics mode: {mode}")

    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        init_k.copy(),
        init_dist.copy(),
        flags=flags,
    )
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    # Guardrail: if soft mode still flies, fall back to fixed-f prior.
    fx = float(camera_matrix[0, 0])
    fallback = False
    if mode == "wide_fov_soft" and (fx > 1.5 * image_size[0] or fx < 0.25 * image_size[0]):
        fallback = True
        return solve_intrinsics(samples, hfov_deg=hfov_deg, mode="wide_fov_fixed_f") | {
            "fallback_from_soft": True,
            "rejected_soft_fx": fx,
        }

    return {
        "rms_px": float(rms),
        "image_size_wh": list(image_size),
        "tag_edge_m": edge,
        "hfov_deg_prior": float(hfov_deg),
        "mode": mode,
        "fallback_from_soft": fallback,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        "flags": str(flags),
        "execution_ready": False,
        "publish_tf": False,
        "diagnostic_only": True,
    }


def solve_pnp_tag(
    corners_px: np.ndarray,
    edge_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    object_points = tag_object_corners_m(edge_m)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        corners_px.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            corners_px.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation, _ = cv2.Rodrigues(rvec)
    return transform(rotation, tvec.reshape(3))


def calibrate_hand_eye_ei(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    method: int,
) -> np.ndarray:
    """Return T_hand_cam (cam2gripper) for eye-in-hand."""
    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        [value[:3, :3] for value in hands],
        [value[:3, 3].reshape(3, 1) for value in hands],
        [value[:3, :3] for value in tags],
        [value[:3, 3].reshape(3, 1) for value in tags],
        method=method,
    )
    return transform(r_cam2gripper, t_cam2gripper)


def evaluate_hand_eye(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    t_hand_cam: np.ndarray,
) -> dict[str, Any]:
    trans = []
    rots = []
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            a_mat = tags[i] @ inverse(tags[j])
            b_mat = inverse(hands[i]) @ hands[j]
            left = a_mat @ t_hand_cam
            right = t_hand_cam @ b_mat
            dt, dr = relative_error(left, right)
            trans.append(dt)
            rots.append(dr)
    if not trans:
        return {"count": 0}
    arr_t = np.asarray(trans)
    arr_r = np.asarray(rots)
    return {
        "pair_count": int(len(trans)),
        "translation_m": {
            "median": float(np.median(arr_t)),
            "max": float(np.max(arr_t)),
            "mean": float(np.mean(arr_t)),
        },
        "rotation_deg": {
            "median": float(np.median(arr_r)),
            "max": float(np.max(arr_r)),
            "mean": float(np.mean(arr_r)),
        },
    }


def base_tag_estimates(
    hands: Sequence[np.ndarray],
    tags: Sequence[np.ndarray],
    t_hand_cam: np.ndarray,
) -> list[np.ndarray]:
    return [hand @ t_hand_cam @ tag for hand, tag in zip(hands, tags)]


def held_out_base_tag_consistency(
    train_tags_base: Sequence[np.ndarray],
    held_hands: Sequence[np.ndarray],
    held_tags: Sequence[np.ndarray],
    t_hand_cam: np.ndarray,
) -> dict[str, Any]:
    if not train_tags_base:
        return {"count": 0}
    mean_rot = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in train_tags_base])
    ).mean()
    mean_t = np.mean([value[:3, 3] for value in train_tags_base], axis=0)
    t_base_tag = transform(mean_rot.as_matrix(), mean_t)
    errors_t = []
    errors_r = []
    for hand, tag in zip(held_hands, held_tags):
        predicted = inverse(t_hand_cam) @ inverse(hand) @ t_base_tag
        dt, dr = relative_error(tag, predicted)
        errors_t.append(dt)
        errors_r.append(dr)
    return {
        "count": len(errors_t),
        "translation_m": {
            "median": float(np.median(errors_t)),
            "max": float(np.max(errors_t)),
            "mean": float(np.mean(errors_t)),
        },
        "rotation_deg": {
            "median": float(np.median(errors_r)),
            "max": float(np.max(errors_r)),
            "mean": float(np.mean(errors_r)),
        },
        "reference_T_base_tag": transform_dict(t_base_tag),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--held-out-count", type=int, default=5)
    parser.add_argument(
        "--held-out-ids",
        nargs="*",
        default=None,
        help="Explicit held-out sample IDs; default = last N",
    )
    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=90.0,
        help="Wide-FOV horizontal prior for wrist VGA (deg)",
    )
    parser.add_argument(
        "--intrinsics-mode",
        choices=("wide_fov_fixed_f", "wide_fov_soft"),
        default="wide_fov_fixed_f",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir
    output_dir = args.output_dir or (raw_dir.parent / "solve")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(raw_dir)
    sample_ids = [sample["sample_id"] for sample in samples]
    held_out_ids = (
        list(args.held_out_ids)
        if args.held_out_ids
        else default_held_out_ids(sample_ids, args.held_out_count)
    )
    unknown = [sample_id for sample_id in held_out_ids if sample_id not in sample_ids]
    if unknown:
        raise SystemExit(f"held-out IDs not in raw dir: {unknown}")
    train_samples = [s for s in samples if s["sample_id"] not in set(held_out_ids)]
    held_samples = [s for s in samples if s["sample_id"] in set(held_out_ids)]
    if len(train_samples) < 8:
        raise SystemExit(f"need >=8 train samples, got {len(train_samples)}")

    intrinsics = solve_intrinsics(
        train_samples + held_samples,
        hfov_deg=args.hfov_deg,
        mode=args.intrinsics_mode,
    )
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=float)
    dist_coeffs = np.asarray(intrinsics["dist_coeffs"], dtype=float)

    def poses_for(subset: Sequence[dict[str, Any]]):
        hands = []
        tags = []
        ids = []
        for sample in subset:
            corners = np.asarray(sample["tag_corners_px"], dtype=float).reshape(4, 2)
            tag = solve_pnp_tag(
                corners,
                float(sample["tag_edge_m"]),
                camera_matrix,
                dist_coeffs,
            )
            hands.append(from_xyzw(sample["T_roarm_base_link_hand_tcp"]))
            tags.append(tag)
            ids.append(sample["sample_id"])
        return ids, hands, tags

    train_ids, train_hands, train_tags = poses_for(train_samples)
    held_ids, held_hands, held_tags = poses_for(held_samples)

    methods = {
        "opencv_hand_eye_park": cv2.CALIB_HAND_EYE_PARK,
        "opencv_hand_eye_tsai": cv2.CALIB_HAND_EYE_TSAI,
    }
    method_results = {}
    for name, flag in methods.items():
        t_hand_cam = calibrate_hand_eye_ei(train_hands, train_tags, flag)
        train_metrics = evaluate_hand_eye(train_hands, train_tags, t_hand_cam)
        train_base_tags = base_tag_estimates(train_hands, train_tags, t_hand_cam)
        held_metrics = held_out_base_tag_consistency(
            train_base_tags, held_hands, held_tags, t_hand_cam
        )
        method_results[name] = {
            "T_hand_tcp_wrist_cam": transform_dict(t_hand_cam),
            "train_pair_residuals": train_metrics,
            "held_out_T_cam_tag_residuals": held_metrics,
        }

    park = from_xyzw(method_results["opencv_hand_eye_park"]["T_hand_tcp_wrist_cam"])
    tsai = from_xyzw(method_results["opencv_hand_eye_tsai"]["T_hand_tcp_wrist_cam"])
    agree_t, agree_r = relative_error(park, tsai)

    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "execution_ready": False,
        "publish_tf": False,
        "press_authorized": False,
        "calibration_type": "eye_in_hand",
        "dataset_id": samples[0]["dataset_id"],
        "raw_dir": str(raw_dir),
        "sample_ids": sample_ids,
        "train_ids": train_ids,
        "held_out_ids": held_ids,
        "wrist_intrinsics_diagnostic": intrinsics,
        "methods": method_results,
        "park_tsai_agreement": {
            "translation_m": agree_t,
            "rotation_deg": agree_r,
        },
        "notes": [
            "Unpublished diagnostic. Do not load into runtime TF.",
            "Intrinsics from single AprilTag corners are weak vs chessboard.",
        ],
    }

    intrinsics_path = output_dir / "wrist_intrinsics_diagnostic.json"
    report_path = output_dir / "eye_in_hand_diagnostic_unpublished.json"
    intrinsics_path.write_text(
        json.dumps(intrinsics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{sha256_file(intrinsics_path)}  {intrinsics_path.name}\n"
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(f"wrote {intrinsics_path}")
    print(f"wrote {report_path}")
    print(
        "Park/Tsai agreement: "
        f"{agree_t * 1000:.2f} mm, {agree_r:.3f} deg"
    )
    print(
        "held-out (Park) translation median/max m: "
        f"{method_results['opencv_hand_eye_park']['held_out_T_cam_tag_residuals']['translation_m']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
