#!/usr/bin/env python3
"""Offline, display-only standby -> pre-contact geometry preview.

This module deliberately stops at a fingertip endpoint proposal.  It does
not solve IK, create a trajectory, publish TF/joint targets, call a ROS
service/action, or claim that the robot may move.  A future one-shot
pre-contact adapter must independently re-check the returned provenance and
all runtime safety gates before accepting any operator request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from press.m0_target_to_base import (
    DEFAULT_BUNDLE,
    EXPECTED_CALIBRATION_ID,
    TransformError,
    load_relative_bundle,
    transform_frozen_target,
)


SCHEMA = "precontact_preview_v1"
FROZEN_PACKAGE_SCHEMA = "m0_demo_freeze_v1"
DEFAULT_D_PRE_MM = 50.0
MIN_D_PRE_MM = 1.0
MAX_D_PRE_MM = 100.0
DEFAULT_MAX_SOURCE_AGE_S = 3.0
DEFAULT_REGISTRY = REPO_ROOT / "config/calibration/precontact_candidate_registry_v1.json"
REQUIRED_PROVENANCE_ROLES = (
    "section_b_relative_bundle",
    "fn_active_pointer",
    "fn_diagnostic_candidate",
)


def _finite(value: Any, where: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise TransformError(f"{where} must be finite")
    return number


def _vec3(value: Sequence[float], where: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise TransformError(f"{where} must have length 3")
    return tuple(_finite(item, f"{where}[{index}]") for index, item in enumerate(value))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _add_scaled(point: Sequence[float], direction: Sequence[float], distance_m: float) -> tuple[float, float, float]:
    return tuple(float(p) + distance_m * float(n) for p, n in zip(point, direction))


def _canonical_sha256(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit:
        payload.pop(omit, None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, where: str) -> str:
    value = str(value or "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise TransformError(f"{where} must be a lowercase SHA-256")
    return value


def _required_button1_label(value: Any) -> str:
    label = str(value or "").strip().lower().replace("_", " ")
    if label not in {"1", "button 1"}:
        raise TransformError("target identity must be Button 1")
    return "Button 1"


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise TransformError(f"{where} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{where} must be a positive integer") from exc
    if result <= 0:
        raise TransformError(f"{where} must be a positive integer")
    return result


def _read_frozen_package(
    package: Mapping[str, Any], *, now_unix_ns: int, max_source_age_s: float
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Validate a frozen Button-1 record before any geometry calculation.

    The package hash is a tamper-evident receipt for the *input* evidence. It
    is intentionally distinct from the preview hash computed below.
    """
    if not isinstance(package, Mapping):
        raise TransformError("frozen target package must be an object")
    if package.get("schema") != FROZEN_PACKAGE_SCHEMA:
        raise TransformError(f"frozen target package schema must be {FROZEN_PACKAGE_SCHEMA}")
    supplied_hash = _sha256(package.get("canonical_payload_sha256"), "canonical_payload_sha256")
    if _canonical_sha256(package, omit="canonical_payload_sha256") != supplied_hash:
        raise TransformError("frozen target package canonical hash mismatch")
    if package.get("motion_authorized") is not False:
        raise TransformError("frozen target package motion_authorized must be false")
    target = package.get("frozen_target")
    if not isinstance(target, Mapping):
        raise TransformError("frozen target package frozen_target must be an object")
    target_identity = _required_button1_label(target.get("target_label"))
    if "target_label" in package and _required_button1_label(package.get("target_label")) != target_identity:
        raise TransformError("package and frozen target Button 1 identity mismatch")
    frame_id = str(target.get("frame_id") or "").strip()
    stamp_ns = _positive_int(target.get("source_stamp_ns"), "frozen_target.source_stamp_ns")
    header = target.get("header")
    if header is not None:
        if not isinstance(header, Mapping):
            raise TransformError("frozen_target.header must be an object")
        if str(header.get("frame_id") or "").strip() != frame_id:
            raise TransformError("frozen target header frame_id mismatch")
        if _positive_int(header.get("stamp_ns"), "frozen_target.header.stamp_ns") != stamp_ns:
            raise TransformError("frozen target header stamp_ns mismatch")
    if not frame_id:
        raise TransformError("frozen target header frame_id is missing")
    max_age_ns = int(_finite(max_source_age_s, "max_source_age_s") * 1_000_000_000)
    age_ns = int(now_unix_ns) - stamp_ns
    if age_ns < 0:
        raise TransformError("frozen target header stamp is in the future")
    if age_ns > max_age_ns:
        raise TransformError("frozen target header is stale")
    return target, {
        "target_identity": target_identity,
        "header": {"frame_id": frame_id, "stamp_ns": stamp_ns},
        "frozen_package_sha256": supplied_hash,
        "source_age_s": age_ns / 1_000_000_000,
    }


def _bound_provenance(
    *, registry_path: str | Path, bundle_path: str | Path
) -> dict[str, dict[str, str]]:
    """Bind R6/Fn evidence hashes without treating diagnostic Fn as geometry."""
    path = Path(registry_path).expanduser().resolve()
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransformError(f"provenance registry is unreadable: {exc}") from exc
    if not isinstance(registry, Mapping):
        raise TransformError("provenance registry must be an object")
    records = registry.get("provenance")
    if not isinstance(records, list):
        raise TransformError("provenance registry provenance must be a list")
    root = REPO_ROOT.resolve()
    by_role: dict[str, dict[str, str]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        role = str(record.get("role") or "")
        if role not in REQUIRED_PROVENANCE_ROLES:
            continue
        relative = Path(str(record.get("path") or ""))
        if not str(relative) or relative.is_absolute():
            raise TransformError(f"provenance {role} path must be repository-relative")
        artifact = (root / relative).resolve()
        if root not in artifact.parents or not artifact.is_file():
            raise TransformError(f"provenance {role} artifact is missing")
        expected = _sha256(record.get("sha256"), f"provenance {role} sha256")
        if _sha256_file(artifact) != expected:
            raise TransformError(f"provenance hash mismatch for {role}")
        by_role[role] = {"path": str(relative), "sha256": expected}
    if set(by_role) != set(REQUIRED_PROVENANCE_ROLES):
        raise TransformError("provenance registry is missing required R6/Fn roles")
    actual_bundle = Path(bundle_path).expanduser().resolve()
    expected_bundle = (root / by_role["section_b_relative_bundle"]["path"]).resolve()
    if actual_bundle != expected_bundle:
        raise TransformError("bundle_path is not the provenance-bound R6 bundle")
    return by_role


def build_precontact_preview(
    frozen_target_package: Mapping[str, Any],
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    expected_calibration_id: str = EXPECTED_CALIBRATION_ID,
    d_pre_mm: float = DEFAULT_D_PRE_MM,
    registry_path: str | Path = DEFAULT_REGISTRY,
    now_unix_ns: int | None = None,
    max_source_age_s: float = DEFAULT_MAX_SOURCE_AGE_S,
) -> dict[str, Any]:
    """Return a provenance-bound endpoint proposal without motion semantics.

    The accepted input is the existing ``m0_demo_freeze_v1`` package, with a
    verified canonical receipt, a fresh header, and Button-1 identity.  The
    diagnostic Fn files are provenance-bound but never used to alter A/N.
    """

    distance_mm = _finite(d_pre_mm, "d_pre_mm")
    if not MIN_D_PRE_MM <= distance_mm <= MAX_D_PRE_MM:
        raise TransformError(
            f"d_pre_mm must be between {MIN_D_PRE_MM:g} and {MAX_D_PRE_MM:g}"
        )
    clock_ns = time.time_ns() if now_unix_ns is None else _positive_int(now_unix_ns, "now_unix_ns")
    target, package_evidence = _read_frozen_package(
        frozen_target_package, now_unix_ns=clock_ns, max_source_age_s=max_source_age_s
    )
    provenance = _bound_provenance(registry_path=registry_path, bundle_path=bundle_path)
    transformed = transform_frozen_target(
        target,
        bundle_path,
        expected_calibration_id=expected_calibration_id,
        allow_incomplete_provenance=False,
    )
    bundle = load_relative_bundle(bundle_path, expected_calibration_id=expected_calibration_id)
    target_base = _vec3(transformed["point_roarm_base_m"], "point_roarm_base_m")
    normal_base = _vec3(transformed["panel_outward_normal_roarm_base"], "panel_outward_normal_roarm_base")
    camera_base = _vec3(bundle["translation_m"], "camera_base")
    camera_side_dot = _dot(normal_base, _sub(camera_base, target_base))
    if camera_side_dot > 1e-6:
        normal_status = "VALID_OUTWARD"
    else:
        raise TransformError("panel outward normal is inward or unverified; refusing to flip N")
    distance_m = distance_mm / 1000.0
    precontact = _add_scaled(target_base, normal_base, distance_m)

    payload = {
        "schema": SCHEMA,
        "mode": "DISPLAY_ONLY",
        "decision": "DIAGNOSTIC_ACCEPTED_NO_MOTION",
        "decision_reasons": [
            "Button 1 frozen package, header, A/N, R6 and diagnostic Fn provenance verified.",
            "Fn_pre = A + 0.050 * N_out computed for display only.",
            "execution_ready=false; no IK, trajectory, command, or execution interface exists.",
        ],
        "target_identity": package_evidence["target_identity"],
        "target_label": transformed["target_label"],
        "source_header": package_evidence["header"],
        "source_frame_id": transformed["source_frame_id"],
        "source_age_s": package_evidence["source_age_s"],
        "frozen_package_sha256": package_evidence["frozen_package_sha256"],
        "target_frame_id": transformed["target_frame_id"],
        "target_point_base_m": list(target_base),
        "panel_outward_normal_base": list(normal_base),
        "precontact_fingertip_base_m": list(precontact),
        "precontact_distance_m": distance_m,
        "normal_camera_side_dot": camera_side_dot,
        "normal_direction_status": normal_status,
        "calibration_id": transformed["calibration_id"],
        "bundle_sha256": transformed["bundle"]["sha256"],
        "provenance": provenance,
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "ik_run": False,
        "action_server_required": True,
        "execution_interface": "NOT_IMPLEMENTED",
        "required_owner": "SAFE_ADAPTER_ACTION_SERVER",
        "legacy_route_used": False,
        "notes": [
            "Endpoint proposal only: no IK, trajectory, ROS command, or TF publish.",
            "The normal is never silently flipped; inward or unverified N is rejected.",
            "Active Fn is diagnostic-only provenance and is not used as a contact calibration here.",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    # Retained as a compatibility alias for earlier display-only consumers.
    payload["preview_hash"] = payload["canonical_payload_sha256"]
    payload["preview_id"] = f"precontact-preview-{payload['preview_hash'][:12]}"
    return payload


def _load_target(path: str | Path) -> Mapping[str, Any]:
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TransformError("frozen target package JSON must be an object")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-target", required=True, type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--expected-calibration-id", default=EXPECTED_CALIBRATION_ID)
    parser.add_argument("--d-pre-mm", type=float, default=DEFAULT_D_PRE_MM)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        preview = build_precontact_preview(
            _load_target(args.frozen_target),
            bundle_path=args.bundle,
            expected_calibration_id=args.expected_calibration_id,
            d_pre_mm=args.d_pre_mm,
        )
    except (OSError, json.JSONDecodeError, TransformError) as exc:
        blocked = {
            "schema": SCHEMA,
            "mode": "DISPLAY_ONLY",
            "decision": "REJECTED_NO_MOTION",
            "decision_reasons": [str(exc)],
            "execution_ready": False,
            "motion_authorized": False,
            "execution_interface": "NOT_IMPLEMENTED",
            "required_owner": "SAFE_ADAPTER_ACTION_SERVER",
        }
        encoded = json.dumps(blocked, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 2
    encoded = json.dumps(preview, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
