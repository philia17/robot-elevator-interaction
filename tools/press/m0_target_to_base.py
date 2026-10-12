#!/usr/bin/env python3
"""Transform a frozen M0 target into roarm_base_link using the accepted B bundle.

Offline math only — no /tf_static publish, no ROS, no motion.  Fail-closed on
calibration_id mismatch, incomplete geometry when required, or non-finite
inputs.  Provenance-incomplete targets may be accepted with
``allow_incomplete_provenance=True`` and are labeled ``diagnostic``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BUNDLE = (
    REPO_ROOT
    / "config"
    / "calibration"
    / "accepted"
    / "gemini335_roarm_20260803_r6_eye_on_base_relative_bundle.json"
)
EXPECTED_CALIBRATION_ID = (
    "gemini335_roarm_20260803_r6__scipy_separable_absolute__b5_tag_gap_250ms_v3"
)
TRANSFORM_KEY = "T_roarm_base_link_camera_optical_frame"
SCHEMA = "m0_target_to_base_v1"


class TransformError(ValueError):
    """Fail-closed transform rejection."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, where: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TransformError(f"{where} must be finite")
    return number


def _vec3(value: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 3:
                raise TransformError(f"{where} must have length 3")
            return (
                _finite(value[0], f"{where}[0]"),
                _finite(value[1], f"{where}[1]"),
                _finite(value[2], f"{where}[2]"),
            )
        raise TransformError(f"{where} must be an object or length-3 sequence")
    return (
        _finite(value.get("x", value.get(0)), f"{where}.x"),
        _finite(value.get("y", value.get(1)), f"{where}.y"),
        _finite(value.get("z", value.get(2)), f"{where}.z"),
    )


def _normalize(vector: Sequence[float], where: str) -> tuple[float, float, float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm < 1e-9:
        raise TransformError(f"{where} has near-zero length")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def quat_xyzw_to_rotation(quat: Sequence[float]) -> list[list[float]]:
    x, y, z, w = (_finite(v, "quat") for v in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise TransformError("rotation quaternion has near-zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def rotate(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def apply_transform(
    translation_m: Sequence[float],
    rotation_xyzw: Sequence[float],
    point_m: Sequence[float],
) -> tuple[float, float, float]:
    matrix = quat_xyzw_to_rotation(rotation_xyzw)
    rotated = rotate(matrix, point_m)
    return (
        rotated[0] + float(translation_m[0]),
        rotated[1] + float(translation_m[1]),
        rotated[2] + float(translation_m[2]),
    )


def load_relative_bundle(
    path: str | Path,
    *,
    expected_calibration_id: str = EXPECTED_CALIBRATION_ID,
) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise TransformError(f"B bundle not found: {bundle_path}")
    bundle = json.loads(bundle_path.read_text())
    if not isinstance(bundle, Mapping):
        raise TransformError("B bundle root must be an object")
    calibration_id = str(bundle.get("calibration_id", "")).strip()
    if calibration_id != expected_calibration_id:
        raise TransformError(
            f"calibration_id mismatch: got {calibration_id!r}, "
            f"expected {expected_calibration_id!r}"
        )
    if bundle.get("execution_ready") is True:
        raise TransformError("bundle claims execution_ready=true; rejected")
    if bundle.get("tf_static_published") is True:
        raise TransformError("bundle claims tf_static_published=true; rejected")
    transforms = bundle.get("transforms")
    if not isinstance(transforms, Mapping):
        raise TransformError("bundle.transforms missing")
    transform = transforms.get(TRANSFORM_KEY)
    if not isinstance(transform, Mapping):
        raise TransformError(f"bundle missing transforms.{TRANSFORM_KEY}")
    translation = _finite_sequence(transform.get("translation_m"), 3, "translation_m")
    rotation = _finite_sequence(transform.get("rotation_xyzw"), 4, "rotation_xyzw")
    return {
        "path": bundle_path,
        "sha256": sha256_file(bundle_path),
        "calibration_id": calibration_id,
        "translation_m": translation,
        "rotation_xyzw": rotation,
        "raw": bundle,
    }


def _finite_sequence(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TransformError(f"{where} must be a sequence")
    if len(value) != length:
        raise TransformError(f"{where} must have length {length}")
    return tuple(_finite(item, f"{where}[{index}]") for index, item in enumerate(value))


def extract_camera_geometry(
    frozen_target: Mapping[str, Any],
    *,
    allow_incomplete_provenance: bool = True,
) -> dict[str, Any]:
    if not isinstance(frozen_target, Mapping):
        raise TransformError("frozen_target must be an object")
    point = _vec3(frozen_target.get("camera_point_m"), "camera_point_m")
    surface = frozen_target.get("surface_geometry") or {}
    if not isinstance(surface, Mapping):
        raise TransformError("surface_geometry must be an object")
    normal_raw = surface.get("panel_outward_normal")
    if normal_raw is None:
        raise TransformError("panel_outward_normal missing")
    normal = _normalize(_vec3(normal_raw, "panel_outward_normal"), "panel_outward_normal")
    provenance_complete = bool(surface.get("provenance_complete", False))
    estimate_valid = bool(surface.get("estimate_valid", False))
    if not estimate_valid and not allow_incomplete_provenance:
        raise TransformError("surface estimate_valid is false")
    if not provenance_complete and not allow_incomplete_provenance:
        raise TransformError("provenance_complete is false")
    quality = "accepted" if provenance_complete and estimate_valid else "diagnostic"
    return {
        "camera_point_m": point,
        "panel_outward_normal_camera": normal,
        "provenance_complete": provenance_complete,
        "estimate_valid": estimate_valid,
        "quality": quality,
        "target_label": str(frozen_target.get("target_label", "")).strip(),
        "frame_id": str(frozen_target.get("frame_id", "")).strip(),
    }


def transform_frozen_target(
    frozen_target: Mapping[str, Any],
    bundle_path: str | Path = DEFAULT_BUNDLE,
    *,
    expected_calibration_id: str = EXPECTED_CALIBRATION_ID,
    allow_incomplete_provenance: bool = True,
) -> dict[str, Any]:
    """Return base-frame point + outward normal bound to the B calibration_id."""
    bundle = load_relative_bundle(
        bundle_path, expected_calibration_id=expected_calibration_id
    )
    geometry = extract_camera_geometry(
        frozen_target, allow_incomplete_provenance=allow_incomplete_provenance
    )
    point_base = apply_transform(
        bundle["translation_m"],
        bundle["rotation_xyzw"],
        geometry["camera_point_m"],
    )
    matrix = quat_xyzw_to_rotation(bundle["rotation_xyzw"])
    normal_base = _normalize(
        rotate(matrix, geometry["panel_outward_normal_camera"]),
        "panel_outward_normal_base",
    )
    artifact = {
        "schema": SCHEMA,
        "calibration_id": bundle["calibration_id"],
        "bundle": {
            "path": str(bundle["path"].relative_to(REPO_ROOT))
            if bundle["path"].is_relative_to(REPO_ROOT)
            else str(bundle["path"]),
            "sha256": bundle["sha256"],
            "transform_key": TRANSFORM_KEY,
        },
        "target_label": geometry["target_label"],
        "source_frame_id": geometry["frame_id"],
        "target_frame_id": "roarm_base_link",
        "quality": geometry["quality"],
        "provenance_complete": geometry["provenance_complete"],
        "estimate_valid": geometry["estimate_valid"],
        "camera_point_m": list(geometry["camera_point_m"]),
        "panel_outward_normal_camera": list(geometry["panel_outward_normal_camera"]),
        "point_roarm_base_m": list(point_base),
        "panel_outward_normal_roarm_base": list(normal_base),
        "tf_static_published": False,
        "execution_ready": False,
        "motion_authorized": False,
    }
    canonical = json.dumps(
        artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    artifact["canonical_payload_sha256"] = sha256_bytes(canonical)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-target", required=True, help="Frozen M0 target JSON")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument(
        "--expected-calibration-id", default=EXPECTED_CALIBRATION_ID
    )
    parser.add_argument(
        "--require-complete-provenance",
        action="store_true",
        help="Reject targets with incomplete surface provenance",
    )
    parser.add_argument("--output", help="Optional output path for the base target JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen_path = Path(args.frozen_target).expanduser().resolve()
    frozen = json.loads(frozen_path.read_text())
    if isinstance(frozen, Mapping) and "frozen_target" in frozen:
        frozen = frozen["frozen_target"]
    artifact = transform_frozen_target(
        frozen,
        args.bundle,
        expected_calibration_id=args.expected_calibration_id,
        allow_incomplete_provenance=not args.require_complete_provenance,
    )
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise TransformError(f"refusing to overwrite existing file: {out}")
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
