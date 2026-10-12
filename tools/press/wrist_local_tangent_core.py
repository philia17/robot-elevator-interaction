"""Local wrist-image correction constrained to the frozen panel tangent plane.

Gemini supplies button identity, the initial 3-D target, and the panel plane.
PRESS corrects that target using the departure wrist ray/plane intersection.
The reference-pose Jacobian solver remains for calibration/offline comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "wrist_local_tangent_calibration_v1"
MAX_CORRECTION_M = 0.035


class WristTangentBlocked(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(self.code if not detail else f"{self.code}: {detail}")


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise WristTangentBlocked("CALIBRATION_INVALID", "canonical JSON") from exc


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise WristTangentBlocked("CALIBRATION_INVALID", field)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WristTangentBlocked("CALIBRATION_INVALID", field) from exc
    if not math.isfinite(number):
        raise WristTangentBlocked("CALIBRATION_INVALID", field)
    return number


def _vector(value: Any, length: int, field: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise WristTangentBlocked("CALIBRATION_INVALID", field)
    return tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))


def _unit(value: Any, field: str) -> tuple[float, float, float]:
    vector = _vector(value, 3, field)
    length = math.sqrt(sum(item * item for item in vector))
    if length <= 1e-12:
        raise WristTangentBlocked("CALIBRATION_INVALID", field)
    return tuple(item / length for item in vector)  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _matrix_2x2(value: Any, field: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise WristTangentBlocked("CALIBRATION_INVALID", field)
    first = _vector(value[0], 2, f"{field}[0]")
    second = _vector(value[1], 2, f"{field}[1]")
    return (first, second)  # type: ignore[return-value]


def _condition_2x2(matrix: Sequence[Sequence[float]]) -> float:
    a, b = float(matrix[0][0]), float(matrix[0][1])
    c, d = float(matrix[1][0]), float(matrix[1][1])
    trace = a * a + b * b + c * c + d * d
    determinant_squared = (a * d - b * c) ** 2
    discriminant = max(0.0, trace * trace - 4.0 * determinant_squared)
    largest = 0.5 * (trace + math.sqrt(discriminant))
    smallest = 0.5 * (trace - math.sqrt(discriminant))
    if smallest <= 1e-18:
        return math.inf
    return math.sqrt(largest / smallest)


@dataclass(frozen=True)
class WristTangentCalibration:
    calibration_sha256: str
    accepted_for_press: bool
    wrist_lock_identity_suffix: tuple[str, ...]
    tool_apex_px: tuple[float, float]
    tangent_u_base: tuple[float, float, float]
    tangent_v_base: tuple[float, float, float]
    reference_normal_out_base: tuple[float, float, float]
    image_error_jacobian_px_per_m: tuple[tuple[float, float], tuple[float, float]]
    maximum_condition_number: float
    maximum_correction_m: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WristTangentCalibration":
        if not isinstance(raw, Mapping) or raw.get("schema") != SCHEMA:
            raise WristTangentBlocked("CALIBRATION_INVALID", "schema")
        supplied_hash = str(raw.get("calibration_sha256") or "")
        unsigned = dict(raw)
        unsigned.pop("calibration_sha256", None)
        if len(supplied_hash) != 64 or canonical_hash(unsigned) != supplied_hash:
            raise WristTangentBlocked("CALIBRATION_INVALID", "hash")
        suffix = raw.get("wrist_lock_identity_suffix")
        if (
            not isinstance(suffix, Sequence)
            or isinstance(suffix, (str, bytes))
            or not suffix
            or any(not str(item) for item in suffix)
        ):
            raise WristTangentBlocked("CALIBRATION_INVALID", "wrist identity")
        calibration = cls(
            calibration_sha256=supplied_hash,
            accepted_for_press=raw.get("accepted_for_press") is True,
            wrist_lock_identity_suffix=tuple(str(item) for item in suffix),
            tool_apex_px=_vector(raw.get("tool_apex_px"), 2, "tool_apex_px"),  # type: ignore[arg-type]
            tangent_u_base=_unit(raw.get("tangent_u_base"), "tangent_u_base"),
            tangent_v_base=_unit(raw.get("tangent_v_base"), "tangent_v_base"),
            reference_normal_out_base=_unit(
                raw.get("reference_normal_out_base"), "reference_normal_out_base"
            ),
            image_error_jacobian_px_per_m=_matrix_2x2(
                raw.get("image_error_jacobian_px_per_m"),
                "image_error_jacobian_px_per_m",
            ),
            maximum_condition_number=_finite(
                raw.get("maximum_condition_number", 10.0), "maximum_condition_number"
            ),
            maximum_correction_m=_finite(
                raw.get("maximum_correction_m", MAX_CORRECTION_M),
                "maximum_correction_m",
            ),
        )
        calibration.validate()
        return calibration

    def validate(self) -> None:
        if not self.accepted_for_press:
            raise WristTangentBlocked("CALIBRATION_NOT_ACCEPTED")
        if self.maximum_condition_number <= 1.0 or not (
            0.0 < self.maximum_correction_m <= MAX_CORRECTION_M
        ):
            raise WristTangentBlocked("CALIBRATION_INVALID", "limits")
        normal = self.reference_normal_out_base
        if (
            abs(_dot(self.tangent_u_base, normal)) > 1e-6
            or abs(_dot(self.tangent_v_base, normal)) > 1e-6
            or abs(_dot(self.tangent_u_base, self.tangent_v_base)) > 1e-3
        ):
            raise WristTangentBlocked("CALIBRATION_INVALID", "tangent frame")
        condition = _condition_2x2(self.image_error_jacobian_px_per_m)
        if not math.isfinite(condition) or condition > self.maximum_condition_number:
            raise WristTangentBlocked("CALIBRATION_SINGULAR", f"condition={condition:.4g}")


@dataclass(frozen=True)
class WristTangentCorrection:
    image_error_px: tuple[float, float]
    tangent_delta_m: tuple[float, float]
    delta_base_m: tuple[float, float, float]
    requested_norm_m: float
    applied_norm_m: float
    capped: bool
    calibration_sha256: str
    suppress_upward_correction: bool = False
    suppressed_upward_base_z_m: float = 0.0


def solve_projected_target_correction(
    calibration: WristTangentCalibration,
    *,
    wrist_lock_identity: Sequence[str],
    observed_button_center_px: Sequence[float],
    projected_target_px: Sequence[float],
    p_button_base_m: Sequence[float],
    normal_out_base: Sequence[float],
    projection_context: Mapping[str, Any],
    suppress_upward_correction: bool = False,
) -> WristTangentCorrection:
    """Correct Gemini's target, not the current tool pose.

    Intersect the departure VIS-LOCK ray with the frozen panel plane using
    that acquisition's camera pose/intrinsics. This is the exact perspective
    equivalent of VIS-LOCK minus PROJ-TGT, without a reference-pose Jacobian.
    The existing calibration identity and 35 mm tangent bound still apply.
    """
    import cv2
    import numpy as np

    calibration.validate()
    suffix = calibration.wrist_lock_identity_suffix
    if tuple(wrist_lock_identity)[-len(suffix):] != suffix:
        raise WristTangentBlocked("WRIST_IDENTITY_MISMATCH")
    # This solver uses the current frozen plane, not the reference-pose
    # Jacobian. Cart yaw must not be limited by that Jacobian's 5-degree
    # applicability gate (retained in solve_wrist_tangent_correction).
    normal = np.asarray(_unit(normal_out_base, "normal_out_base"))
    center = np.asarray(_vector(observed_button_center_px, 2, "VIS-LOCK"))
    projected = np.asarray(_vector(projected_target_px, 2, "PROJ-TGT"))
    button = np.asarray(_vector(p_button_base_m, 3, "p_button_base_m"))
    try:
        transform = np.asarray(projection_context["camera_from_base"], dtype=float)
        intrinsic = np.asarray(projection_context["camera_matrix"], dtype=float)
        distortion = np.asarray(projection_context["dist_coeffs"], dtype=float).reshape(-1)
        if (
            transform.shape != (4, 4) or intrinsic.shape != (3, 3)
            or distortion.size not in {4, 5, 8, 12, 14}
            or not all(np.all(np.isfinite(x)) for x in (transform, intrinsic, distortion))
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(transform[:3, :3])), 1.0, abs_tol=1e-6)
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-9)
        ):
            raise ValueError("invalid projection geometry")
        rotation, translation = transform[:3, :3], transform[:3, 3]
        camera_button = rotation @ button + translation
        if camera_button[2] <= 1e-6:
            raise ValueError("target behind camera")
        exact_projected, _ = cv2.projectPoints(
            camera_button.reshape(1, 3), np.zeros(3), np.zeros(3), intrinsic, distortion,
        )
        # Live's diagnostic projection is integer-rounded. A different
        # target/pose must not be mixed with this click's frozen pixel pair.
        if np.linalg.norm(exact_projected.reshape(2) - projected) > 1.0:
            raise WristTangentBlocked("PROJECTION_TARGET_MISMATCH")
        normalized = cv2.undistortPoints(center.reshape(1, 1, 2), intrinsic, distortion).reshape(2)
        origin = -rotation.T @ translation
        ray = rotation.T @ np.array([normalized[0], normalized[1], 1.0])
        denominator = float(normal @ ray)
        if abs(denominator) < 1e-8:
            raise ValueError("ray parallel to panel")
        distance = float(normal @ (button - origin)) / denominator
        if distance <= 0:
            raise ValueError("panel behind camera ray")
        delta = origin + distance * ray - button
    except WristTangentBlocked:
        raise
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        raise WristTangentBlocked("PROJECTION_GEOMETRY_INVALID", str(exc)) from exc
    requested = float(np.linalg.norm(delta))
    if not math.isfinite(requested) or abs(float(delta @ normal)) > 1e-9:
        raise WristTangentBlocked("NORMAL_COMPONENT_NONZERO")
    applied = min(requested, calibration.maximum_correction_m)
    if requested > 0:
        delta *= applied / requested
    suppressed_z = 0.0
    if suppress_upward_correction and delta[2] > 0.0:
        # Operator-requested A/B: remove only the upward panel-tangent
        # component. Simply zeroing base Z would add penetration on a
        # tilted panel. This orthogonal removal preserves the horizontal
        # tangent component and never increases the bounded correction.
        up_tangent = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
        up_norm_squared = float(up_tangent @ up_tangent)
        if up_norm_squared <= 1e-12:
            raise WristTangentBlocked("PANEL_UP_DIRECTION_UNDEFINED")
        suppressed_z = float(delta[2])
        delta -= up_tangent * (suppressed_z / up_norm_squared)
    return WristTangentCorrection(
        image_error_px=tuple(float(x) for x in center - projected),
        tangent_delta_m=(
            _dot(delta, calibration.tangent_u_base),
            _dot(delta, calibration.tangent_v_base),
        ),
        delta_base_m=tuple(float(x) for x in delta),
        requested_norm_m=requested,
        applied_norm_m=float(np.linalg.norm(delta)),
        capped=requested > calibration.maximum_correction_m,
        calibration_sha256=calibration.calibration_sha256,
        suppress_upward_correction=suppress_upward_correction,
        suppressed_upward_base_z_m=suppressed_z,
    )


def solve_wrist_tangent_correction(
    calibration: WristTangentCalibration | Mapping[str, Any],
    *,
    wrist_lock_identity: Sequence[str],
    observed_button_center_px: Sequence[float],
    normal_out_base: Sequence[float],
    contact_reference_px: Sequence[float] | None = None,
) -> WristTangentCorrection:
    """Solve ``e + J delta = 0`` and return one bounded base translation."""

    parsed = (
        calibration
        if isinstance(calibration, WristTangentCalibration)
        else WristTangentCalibration.from_mapping(calibration)
    )
    parsed.validate()
    identity = tuple(str(item) for item in wrist_lock_identity)
    suffix = parsed.wrist_lock_identity_suffix
    if len(identity) < len(suffix) or identity[-len(suffix):] != suffix:
        raise WristTangentBlocked("WRIST_IDENTITY_MISMATCH")
    normal = _unit(normal_out_base, "normal_out_base")
    if _dot(normal, parsed.reference_normal_out_base) < math.cos(math.radians(5.0)):
        raise WristTangentBlocked("PANEL_NORMAL_MISMATCH")
    tangent_u_projected = tuple(
        parsed.tangent_u_base[index]
        - _dot(parsed.tangent_u_base, normal) * normal[index]
        for index in range(3)
    )
    tangent_u_norm = math.sqrt(_dot(tangent_u_projected, tangent_u_projected))
    tangent_v_projected = tuple(
        parsed.tangent_v_base[index]
        - _dot(parsed.tangent_v_base, normal) * normal[index]
        for index in range(3)
    )
    if tangent_u_norm <= 1e-12:
        raise WristTangentBlocked("TANGENT_FRAME_MISMATCH")
    tangent_u = tuple(value / tangent_u_norm for value in tangent_u_projected)
    tangent_v_orthogonal = tuple(
        tangent_v_projected[index]
        - _dot(tangent_v_projected, tangent_u) * tangent_u[index]
        for index in range(3)
    )
    tangent_v_norm = math.sqrt(
        _dot(tangent_v_orthogonal, tangent_v_orthogonal)
    )
    if tangent_v_norm <= 1e-12:
        raise WristTangentBlocked("TANGENT_FRAME_MISMATCH")
    tangent_v = tuple(value / tangent_v_norm for value in tangent_v_orthogonal)
    if _dot(tangent_v, parsed.tangent_v_base) < 0.0:
        tangent_v = tuple(-value for value in tangent_v)
    center = _vector(observed_button_center_px, 2, "observed_button_center_px")
    reference = (
        parsed.tool_apex_px
        if contact_reference_px is None
        else _vector(contact_reference_px, 2, "contact_reference_px")
    )
    error = (center[0] - reference[0], center[1] - reference[1])
    matrix = parsed.image_error_jacobian_px_per_m
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a * d - b * c
    if abs(determinant) <= 1e-12:
        raise WristTangentBlocked("CALIBRATION_SINGULAR")
    # -inv(J) * image_error
    du = -(d * error[0] - b * error[1]) / determinant
    dv = -(-c * error[0] + a * error[1]) / determinant
    requested = math.hypot(du, dv)
    applied = min(requested, parsed.maximum_correction_m)
    scale = 1.0 if requested <= parsed.maximum_correction_m or requested == 0.0 else applied / requested
    du *= scale
    dv *= scale
    delta = tuple(
        tangent_u[index] * du + tangent_v[index] * dv
        for index in range(3)
    )
    normal_component = _dot(delta, normal)
    if abs(normal_component) > 1e-9:
        raise WristTangentBlocked("NORMAL_COMPONENT_NONZERO")
    return WristTangentCorrection(
        image_error_px=error,
        tangent_delta_m=(du, dv),
        delta_base_m=delta,  # type: ignore[arg-type]
        requested_norm_m=requested,
        applied_norm_m=math.sqrt(sum(value * value for value in delta)),
        capped=requested > parsed.maximum_correction_m,
        calibration_sha256=parsed.calibration_sha256,
    )


def build_calibration_candidate(
    *,
    wrist_lock_identity_suffix: Sequence[str],
    tool_apex_px: Sequence[float],
    tangent_u_base: Sequence[float],
    tangent_v_base: Sequence[float],
    reference_normal_out_base: Sequence[float],
    q0_button_centers_px: Sequence[Sequence[float]],
    plus_u_button_centers_px: Sequence[Sequence[float]],
    plus_v_button_centers_px: Sequence[Sequence[float]],
    probe_u_m: float,
    probe_v_m: float,
    accepted_for_press: bool = False,
) -> Mapping[str, Any]:
    """Build a calibration from median q0/+U/+V button-center observations."""

    def median_center(samples: Sequence[Sequence[float]], field: str) -> tuple[float, float]:
        values = tuple(_vector(item, 2, field) for item in samples)
        if len(values) < 3:
            raise WristTangentBlocked("CALIBRATION_INVALID", f"{field}: need >=3")
        ordered_x = sorted(item[0] for item in values)
        ordered_y = sorted(item[1] for item in values)
        middle = len(values) // 2
        if len(values) % 2:
            return ordered_x[middle], ordered_y[middle]
        return (
            0.5 * (ordered_x[middle - 1] + ordered_x[middle]),
            0.5 * (ordered_y[middle - 1] + ordered_y[middle]),
        )

    q0 = median_center(q0_button_centers_px, "q0_button_centers_px")
    plus_u = median_center(plus_u_button_centers_px, "plus_u_button_centers_px")
    plus_v = median_center(plus_v_button_centers_px, "plus_v_button_centers_px")
    du = _finite(probe_u_m, "probe_u_m")
    dv = _finite(probe_v_m, "probe_v_m")
    if du == 0.0 or dv == 0.0:
        raise WristTangentBlocked("CALIBRATION_INVALID", "probe distance")
    jacobian = [
        [(plus_u[0] - q0[0]) / du, (plus_v[0] - q0[0]) / dv],
        [(plus_u[1] - q0[1]) / du, (plus_v[1] - q0[1]) / dv],
    ]
    unsigned = {
        "schema": SCHEMA,
        "accepted_for_press": bool(accepted_for_press),
        "wrist_lock_identity_suffix": [str(item) for item in wrist_lock_identity_suffix],
        "tool_apex_px": list(_vector(tool_apex_px, 2, "tool_apex_px")),
        "tangent_u_base": list(_unit(tangent_u_base, "tangent_u_base")),
        "tangent_v_base": list(_unit(tangent_v_base, "tangent_v_base")),
        "reference_normal_out_base": list(
            _unit(reference_normal_out_base, "reference_normal_out_base")
        ),
        "image_error_jacobian_px_per_m": jacobian,
        "probe_u_m": du,
        "probe_v_m": dv,
        "q0_button_center_median_px": list(q0),
        "plus_u_button_center_median_px": list(plus_u),
        "plus_v_button_center_median_px": list(plus_v),
        "maximum_condition_number": 10.0,
        "maximum_correction_m": MAX_CORRECTION_M,
    }
    result = dict(unsigned)
    result["calibration_sha256"] = canonical_hash(unsigned)
    return result
