"""Pure offline synthetic keep-out and observed-field corridor preview.

This module is diagnostic geometry only.  It does not import ROS, a robot
model, IK/FK, a driver, or any execution/admission package.  A report can show
an observed obstruction or a synthetic rear-halfspace violation; it can never
show that a route is executable or collision-free.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "precontact_keepout_preview_request_v1"
REPORT_SCHEMA = "precontact_keepout_preview_report_v1"
MODE = "SIM_ONLY"
STATUS = "offline_synthetic_pending"
PRECONTACT_DISTANCE_M = 0.050
_REQUEST_FIELDS = {
    "schema", "mode", "status", "geometry", "rear_keepout",
    "front_corridor", "request_sha256",
}
_REPORT_FIELDS = {
    "schema", "mode", "status", "preview_verdict", "executable",
    "execution_ready", "adapter_consumable", "field_observed_only",
    "request_sha256", "geometry_binding", "rear_keepout_evidence",
    "front_corridor_evidence", "limitations", "canonical_report_sha256",
}


class KeepoutPreviewBlocked(ValueError):
    """The diagnostic request or report is malformed or self-inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise KeepoutPreviewBlocked(f"canonical JSON failed: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_report_sha256(report: Mapping[str, Any]) -> str:
    unsigned = dict(_mapping(report, "report"))
    unsigned.pop("canonical_report_sha256", None)
    return canonical_sha256(unsigned)


def seal_preview_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal deterministic offline fixtures; this is not a release signature."""

    request = dict(value)
    request.pop("request_sha256", None)
    request["request_sha256"] = canonical_sha256(request)
    return request


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KeepoutPreviewBlocked(f"{field} must be an object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise KeepoutPreviewBlocked(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise KeepoutPreviewBlocked(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise KeepoutPreviewBlocked(f"{field} must be finite")
    return result


def _vec(value: Any, length: int, field: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise KeepoutPreviewBlocked(f"{field} must contain exactly {length} finite values")
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, field: str) -> tuple[float, float, float]:
    vector = _vec(value, 3, field)
    magnitude = math.sqrt(sum(item * item for item in vector))
    if not 0.999 <= magnitude <= 1.001:
        raise KeepoutPreviewBlocked(f"{field} must be a unit vector")
    return tuple(item / magnitude for item in vector)  # type: ignore[return-value]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))  # type: ignore[return-value]


def _point_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> tuple[float, float]:
    segment = _sub(end, start)
    length_squared = _dot(segment, segment)
    if length_squared <= 1e-18:
        raise KeepoutPreviewBlocked("front corridor Fn and P_pre must be distinct")
    parameter = max(0.0, min(1.0, _dot(_sub(point, start), segment) / length_squared))
    projection = tuple(float(start[index]) + parameter * segment[index] for index in range(3))
    return _distance(point, projection), parameter


def _parse_request(request: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(_mapping(request, "request"))
    if set(value) != _REQUEST_FIELDS:
        raise KeepoutPreviewBlocked("preview request has unknown or missing fields")
    if value.get("schema") != REQUEST_SCHEMA:
        raise KeepoutPreviewBlocked("unexpected keep-out preview request schema")
    supplied_hash = value.get("request_sha256")
    if (
        not isinstance(supplied_hash, str)
        or len(supplied_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_hash)
    ):
        raise KeepoutPreviewBlocked("request_sha256 must be a lowercase SHA-256")
    unsigned = dict(value)
    unsigned.pop("request_sha256", None)
    if canonical_sha256(unsigned) != supplied_hash:
        raise KeepoutPreviewBlocked("preview request hash mismatch")
    if value.get("mode") != MODE or value.get("status") != STATUS:
        raise KeepoutPreviewBlocked("preview request must be SIM_ONLY/offline_synthetic_pending")

    geometry = _mapping(value.get("geometry"), "geometry")
    if set(geometry) != {"frame_id", "A_base_m", "N_out_base", "P_pre_base_m", "Fn_base_m"}:
        raise KeepoutPreviewBlocked("preview geometry has unknown or missing fields")
    if geometry.get("frame_id") != "roarm_base_link":
        raise KeepoutPreviewBlocked("preview geometry frame must be roarm_base_link")
    a = _vec(geometry.get("A_base_m"), 3, "geometry.A_base_m")
    n_out = _unit(geometry.get("N_out_base"), "geometry.N_out_base")
    p_pre = _vec(geometry.get("P_pre_base_m"), 3, "geometry.P_pre_base_m")
    fn = _vec(geometry.get("Fn_base_m"), 3, "geometry.Fn_base_m")
    expected = tuple(a[index] + PRECONTACT_DISTANCE_M * n_out[index] for index in range(3))
    if _distance(p_pre, expected) > 1e-12:
        raise KeepoutPreviewBlocked("P_pre must equal A + 0.050*N_out exactly")
    if _distance(fn, p_pre) <= 1e-9:
        raise KeepoutPreviewBlocked("Fn and P_pre must define a nonzero observed corridor")

    rear = _mapping(value.get("rear_keepout"), "rear_keepout")
    if set(rear) != {
        "model", "synthetic_only", "minimum_front_signed_distance_m", "proxy_points_base_m"
    }:
        raise KeepoutPreviewBlocked("rear keep-out has unknown or missing fields")
    if rear.get("model") != "synthetic_panel_rear_halfspace_v1":
        raise KeepoutPreviewBlocked("rear keep-out model is not the synthetic halfspace")
    if rear.get("synthetic_only") is not True:
        raise KeepoutPreviewBlocked("rear keep-out must remain synthetic_only")
    margin = _finite(rear.get("minimum_front_signed_distance_m"), "rear margin")
    if margin < 0.0:
        raise KeepoutPreviewBlocked("rear keep-out front margin cannot be negative")
    proxies_raw = rear.get("proxy_points_base_m")
    if not isinstance(proxies_raw, Sequence) or isinstance(proxies_raw, (str, bytes)):
        raise KeepoutPreviewBlocked("rear proxy points must be a list")
    proxies = tuple(
        _vec(point, 3, f"rear proxy point[{index}]")
        for index, point in enumerate(proxies_raw)
    )

    front = _mapping(value.get("front_corridor"), "front_corridor")
    if set(front) != {
        "model", "field_observed_only", "corridor_radius_m", "observed_points_base_m"
    }:
        raise KeepoutPreviewBlocked("front corridor has unknown or missing fields")
    if front.get("model") != "observed_cartesian_capsule_v1":
        raise KeepoutPreviewBlocked("front corridor model is not the observed capsule")
    if front.get("field_observed_only") is not True:
        raise KeepoutPreviewBlocked("front corridor must be field_observed_only")
    radius = _finite(front.get("corridor_radius_m"), "front corridor radius")
    if radius <= 0.0:
        raise KeepoutPreviewBlocked("front corridor radius must be positive")
    observed_raw = front.get("observed_points_base_m")
    if not isinstance(observed_raw, Sequence) or isinstance(observed_raw, (str, bytes)):
        raise KeepoutPreviewBlocked("front observed points must be a list")
    observed = tuple(
        _vec(point, 3, f"front observed point[{index}]")
        for index, point in enumerate(observed_raw)
    )
    return {
        "request": value,
        "request_sha256": supplied_hash,
        "A": a,
        "N": n_out,
        "P_pre": p_pre,
        "Fn": fn,
        "rear_margin": margin,
        "rear_proxies": (fn, p_pre, *proxies),
        "front_radius": radius,
        "front_observed": observed,
    }


def build_keepout_preview_report(request: Mapping[str, Any]) -> dict[str, Any]:
    """Build a hash-bound diagnostic report, never an execution artifact."""

    parsed = _parse_request(request)
    a = parsed["A"]
    n_out = parsed["N"]
    rear_samples = []
    rear_violations = []
    for index, point in enumerate(parsed["rear_proxies"]):
        signed_distance = _dot(_sub(point, a), n_out)
        sample = {
            "sample_index": index,
            "point_base_m": list(point),
            "signed_distance_from_panel_m": signed_distance,
        }
        rear_samples.append(sample)
        if signed_distance < parsed["rear_margin"]:
            rear_violations.append(sample)

    front_samples = []
    front_violations = []
    for index, point in enumerate(parsed["front_observed"]):
        distance, parameter = _point_segment_distance(point, parsed["Fn"], parsed["P_pre"])
        sample = {
            "sample_index": index,
            "point_base_m": list(point),
            "distance_to_fn_p_pre_segment_m": distance,
            "segment_parameter": parameter,
        }
        front_samples.append(sample)
        if distance < parsed["front_radius"]:
            front_violations.append(sample)

    if rear_violations or front_violations:
        preview_verdict = "BLOCKED_SYNTHETIC_OR_OBSERVED"
    else:
        preview_verdict = "PENDING_FIELD_VALIDATION"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "mode": MODE,
        "status": STATUS,
        "preview_verdict": preview_verdict,
        "executable": False,
        "execution_ready": False,
        "adapter_consumable": False,
        "field_observed_only": True,
        "request_sha256": parsed["request_sha256"],
        "geometry_binding": {
            "frame_id": "roarm_base_link",
            "A_base_m": list(a),
            "N_out_base": list(n_out),
            "P_pre_base_m": list(parsed["P_pre"]),
            "Fn_base_m": list(parsed["Fn"]),
            "exact_precontact_distance_m": PRECONTACT_DISTANCE_M,
        },
        "rear_keepout_evidence": {
            "model": "synthetic_panel_rear_halfspace_v1",
            "synthetic_only": True,
            "minimum_front_signed_distance_m": parsed["rear_margin"],
            "sample_count": len(rear_samples),
            "violation_count": len(rear_violations),
            "verdict": "SYNTHETIC_VIOLATION" if rear_violations else "SYNTHETIC_CLEAR_PENDING",
            "samples": rear_samples,
        },
        "front_corridor_evidence": {
            "model": "observed_cartesian_capsule_v1",
            "field_observed_only": True,
            "start_Fn_base_m": list(parsed["Fn"]),
            "end_P_pre_base_m": list(parsed["P_pre"]),
            "corridor_radius_m": parsed["front_radius"],
            "observed_sample_count": len(front_samples),
            "observed_violation_count": len(front_violations),
            "verdict": (
                "OBSERVED_BLOCKED"
                if front_violations
                else "OBSERVED_CLEAR_NOT_COMPLETE"
                if front_samples
                else "UNOBSERVED_PENDING"
            ),
            "samples": front_samples,
        },
        "limitations": [
            "NO_ROBOT_BODY_MODEL",
            "NO_JOINT_SPACE_PATH",
            "NO_SWEPT_VOLUME_PROOF",
            "NO_UNOBSERVED_FIELD_CLEARANCE_CLAIM",
            "NO_MOTION_AUTHORITY",
        ],
    }
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    validate_keepout_preview_report(report, request=request)
    return report


def validate_keepout_preview_report(
    report: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> None:
    value = _mapping(report, "report")
    if set(value) != _REPORT_FIELDS:
        raise KeepoutPreviewBlocked("preview report has unknown or missing fields")
    if value.get("schema") != REPORT_SCHEMA or value.get("mode") != MODE:
        raise KeepoutPreviewBlocked("unexpected keep-out preview report schema or mode")
    if value.get("status") != STATUS:
        raise KeepoutPreviewBlocked("keep-out preview report is not pending")
    for field in ("executable", "execution_ready", "adapter_consumable"):
        if value.get(field) is not False:
            raise KeepoutPreviewBlocked(f"preview report has unsafe {field}")
    if value.get("field_observed_only") is not True:
        raise KeepoutPreviewBlocked("preview report lost field_observed_only")
    supplied = value.get("canonical_report_sha256")
    if canonical_report_sha256(value) != supplied:
        raise KeepoutPreviewBlocked("preview report canonical hash mismatch")
    geometry = _mapping(value.get("geometry_binding"), "report geometry")
    if set(geometry) != {
        "frame_id", "A_base_m", "N_out_base", "P_pre_base_m", "Fn_base_m",
        "exact_precontact_distance_m",
    }:
        raise KeepoutPreviewBlocked("report geometry has unknown or missing fields")
    a = _vec(geometry.get("A_base_m"), 3, "report A")
    n_out = _unit(geometry.get("N_out_base"), "report N")
    p_pre = _vec(geometry.get("P_pre_base_m"), 3, "report P_pre")
    _vec(geometry.get("Fn_base_m"), 3, "report Fn")
    expected = tuple(a[index] + PRECONTACT_DISTANCE_M * n_out[index] for index in range(3))
    if _distance(p_pre, expected) > 1e-12:
        raise KeepoutPreviewBlocked("report P_pre is not A+0.050*N_out")
    if abs(_finite(geometry.get("exact_precontact_distance_m"), "report distance") - 0.050) > 1e-12:
        raise KeepoutPreviewBlocked("report precontact distance is not exactly 50 mm")
    rear = _mapping(value.get("rear_keepout_evidence"), "report rear evidence")
    front = _mapping(value.get("front_corridor_evidence"), "report front evidence")
    if set(rear) != {
        "model", "synthetic_only", "minimum_front_signed_distance_m", "sample_count",
        "violation_count", "verdict", "samples",
    }:
        raise KeepoutPreviewBlocked("report rear evidence has unknown or missing fields")
    if set(front) != {
        "model", "field_observed_only", "start_Fn_base_m", "end_P_pre_base_m",
        "corridor_radius_m", "observed_sample_count", "observed_violation_count",
        "verdict", "samples",
    }:
        raise KeepoutPreviewBlocked("report front evidence has unknown or missing fields")
    if rear.get("synthetic_only") is not True or front.get("field_observed_only") is not True:
        raise KeepoutPreviewBlocked("report promoted synthetic/observed evidence")
    if value.get("preview_verdict") not in {
        "BLOCKED_SYNTHETIC_OR_OBSERVED",
        "PENDING_FIELD_VALIDATION",
    }:
        raise KeepoutPreviewBlocked("report has an execution-like verdict")
    if request is not None:
        parsed = _parse_request(request)
        if value.get("request_sha256") != parsed["request_sha256"]:
            raise KeepoutPreviewBlocked("report does not bind the supplied request")
        bindings = (
            (geometry.get("A_base_m"), parsed["A"], "A"),
            (geometry.get("N_out_base"), parsed["N"], "N_out"),
            (geometry.get("P_pre_base_m"), parsed["P_pre"], "P_pre"),
            (geometry.get("Fn_base_m"), parsed["Fn"], "Fn"),
        )
        for actual, expected_value, field in bindings:
            if _distance(_vec(actual, 3, f"report {field}"), expected_value) > 1e-12:
                raise KeepoutPreviewBlocked(f"report {field} does not bind the request")


__all__ = [
    "KeepoutPreviewBlocked",
    "MODE",
    "PRECONTACT_DISTANCE_M",
    "REPORT_SCHEMA",
    "REQUEST_SCHEMA",
    "STATUS",
    "build_keepout_preview_report",
    "canonical_json_bytes",
    "canonical_report_sha256",
    "canonical_sha256",
    "seal_preview_request",
    "validate_keepout_preview_report",
]
