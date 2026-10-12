"""ROS-agnostic Dynamic-IK Prepare core for T102 coarse precontact.

The boot loader is the only file-I/O boundary.  ``prepare_dynamic_plan`` and
``verify_frozen_plan`` operate exclusively on immutable in-memory values and
cannot create publishers, clients, leases, actions, subprocesses or commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import secrets
import time
from typing import Any, Callable, Mapping, Sequence
from xml.etree import ElementTree as ET

from press.button_target import normalize_button_label, validate_target_selection

SCHEMA = "t102_dynamic_precontact_plan_v1"
SUCCESS_CODE = "SAMPLED_LIGHTWEIGHT_CONSTRAINT_PASS"
STOPPED_MODE = "OPERATOR_CONFIRMED_CART_PARKED_PROTOTYPE"
WRIST_XY_DISABLED = "WRIST_XY_DISABLED_DIAGNOSTIC_CALIBRATION"
MAX_DYNAMIC_SOURCE_STAMP_SKEW_NS = 150_000_000
MAX_CAMERA_INFO_STAMP_SKEW_NS = 500_000_000
START_STATE_ADMISSION_MODE = "FRESH_CONVERGED_DISARMED_LEGAL_Q0"


class DynamicPrepareBlocked(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {detail}" if detail else self.code)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def canonical_accepted_camera_record(
    calibration: Mapping[str, Any], *, calibration_sha256: str,
    source_path_relative: str, source_path_absolute: str,
) -> Mapping[str, Any]:
    """Validate and canonicalize the boot-trusted accepted base<-camera edge."""

    if not isinstance(calibration, Mapping):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera calibration mapping")
    digest = str(calibration_sha256 or "").strip().lower()
    relative = str(source_path_relative or "").strip()
    absolute = str(source_path_absolute or "").strip()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera calibration hash")
    if (
        not relative.startswith("config/calibration/accepted/")
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or not Path(absolute).is_absolute()
        or Path(absolute).name != Path(relative).name
    ):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera calibration path semantics")
    frames = calibration.get("frames")
    transform_key = "T_roarm_base_link_camera_optical_frame"
    transforms = calibration.get("transforms")
    if (
        not isinstance(frames, Mapping)
        or frames.get("arm_base") != "roarm_base_link"
        or frames.get("camera") != "camera_optical_frame"
        or not isinstance(transforms, Mapping)
        or not isinstance(transforms.get(transform_key), Mapping)
    ):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera calibration frames")
    transform = transforms[transform_key]
    translation = _vec(transform.get("translation_m", ()), 3, "accepted camera translation")
    rotation = _vec(transform.get("rotation_xyzw", ()), 4, "accepted camera quaternion")
    if abs(_norm(rotation) - 1.0) > 1e-6:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera quaternion unit")
    gates = calibration.get("gates_summary")
    if not isinstance(gates, Mapping):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera gates")
    calibration_id = str(calibration.get("calibration_id") or "")
    validation_identity = "|".join((
        calibration_id,
        str(calibration.get("dataset_id") or ""),
        str(calibration.get("method") or ""),
        str(gates.get("held_out_status") or ""),
    ))
    acceptance_artifact = str(calibration.get("acceptance_artifact") or "")
    acceptance_sha256 = str(calibration.get("acceptance_sha256") or "").strip().lower()
    if (
        not calibration_id
        or calibration.get("status") != "b_section_accepted_relative_unpublished"
        or calibration.get("artifact_type") != "eye_on_base_relative_bundle"
        or calibration.get("b_section_exit") is not True
        or gates.get("held_out_gate_passed") is not True
        or gates.get("held_out_status") != "held_out_passed"
        or calibration.get("execution_ready") is not False
        or calibration.get("publish_authorized") is not False
        or calibration.get("tf_static_published") is not False
        or calibration.get("quaternion_order") != "XYZW"
        or not str(calibration.get("section_b_scope") or "")
        or not Path(acceptance_artifact).is_absolute()
        or len(acceptance_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in acceptance_sha256)
        or acceptance_sha256 != str(calibration.get("held_out_artifact_sha256") or "")
    ):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted camera authority/provenance")
    return {
        "schema": "t102_accepted_camera_record_v1",
        "transform_key": transform_key,
        "parent_frame": "roarm_base_link",
        "child_frame": "camera_optical_frame",
        "translation_m": list(translation),
        "rotation_xyzw": list(rotation),
        "quaternion_order": "XYZW",
        "calibration_identity": calibration_id,
        "calibration_sha256": digest,
        "calibration_source_path_relative": relative,
        "calibration_source_path_absolute": absolute,
        "calibration_status": str(calibration.get("status")),
        "artifact_type": str(calibration.get("artifact_type")),
        "section_b_scope": str(calibration.get("section_b_scope")),
        "b_section_exit": True,
        "held_out_gate_passed": True,
        "held_out_status": "held_out_passed",
        "calibration_validation_identity": validation_identity,
        "acceptance_artifact": acceptance_artifact,
        "acceptance_sha256": acceptance_sha256,
        "execution_ready": False,
        "publish_authorized": False,
        "tf_static_published": False,
    }


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DynamicPrepareBlocked("NONFINITE_INPUT", field) from exc
    if isinstance(value, bool) or not math.isfinite(number):
        raise DynamicPrepareBlocked("NONFINITE_INPUT", field)
    return number


def _vec(value: Sequence[float], length: int, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise DynamicPrepareBlocked("NONFINITE_INPUT", field)
    return tuple(_finite(item, f"{field}[{i}]") for i, item in enumerate(value))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def _scale(a: Sequence[float], s: float) -> tuple[float, ...]:
    return tuple(float(x) * float(s) for x in a)


def _unit(a: Sequence[float], code: str) -> tuple[float, float, float]:
    value = _vec(a, 3, code)
    length = _norm(value)
    if length < 1e-12:
        raise DynamicPrepareBlocked(code, "zero vector")
    return tuple(x / length for x in value)  # type: ignore[return-value]


def _angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(_unit(a, "axis"), _unit(b, "axis"))))))


@dataclass(frozen=True)
class NormalProof:
    n_out: tuple[float, float, float]
    raw_normal: tuple[float, float, float]
    camera_reference_unit: tuple[float, float, float]
    robot_reference_unit: tuple[float, float, float]
    camera_dot: float
    robot_dot: float
    flipped: bool
    result: str = "OUTWARD_NORMAL_PROVEN"


def orient_outward_normal(
    raw_normal: Sequence[float], p_button: Sequence[float], camera_center: Sequence[float],
    robot_reference: Sequence[float], *, min_dot: float = 0.35, unit_tolerance: float = 0.02,
) -> NormalProof:
    raw = _vec(raw_normal, 3, "normal")
    raw_len = _norm(raw)
    if abs(raw_len - 1.0) > float(unit_tolerance):
        raise DynamicPrepareBlocked("NORMAL_DIRECTION_AMBIGUOUS", "candidate normal is not unit")
    button = _vec(p_button, 3, "p_button")
    camera_ref = _unit(_sub(camera_center, button), "camera reference")
    robot_ref = _unit(_sub(robot_reference, button), "robot reference")
    agreement = _dot(camera_ref, robot_ref)
    if agreement < float(min_dot):
        raise DynamicPrepareBlocked("NORMAL_DIRECTION_AMBIGUOUS", "camera/base references conflict")
    normal = tuple(x / raw_len for x in raw)
    camera_dot = _dot(normal, camera_ref)
    robot_dot = _dot(normal, robot_ref)
    if abs(camera_dot) < min_dot or abs(robot_dot) < min_dot or camera_dot * robot_dot <= 0:
        raise DynamicPrepareBlocked("NORMAL_DIRECTION_AMBIGUOUS", "normal sign references ambiguous")
    flipped = camera_dot < 0.0
    if flipped:
        normal = tuple(-x for x in normal)
        camera_dot, robot_dot = -camera_dot, -robot_dot
    return NormalProof(normal, raw, camera_ref, robot_ref, camera_dot, robot_dot, flipped)


@dataclass(frozen=True)
class FrozenSession:
    session_id: str
    live_boot_epoch: str
    qc_boot_epoch: str
    executor_boot_epoch: str
    cart_pose_epoch: str
    cart_confirmed_stamp_ns: int
    stopped_proof_mode: str
    target_label: str
    detection_id: str
    confidence: float
    target_stamp_ns: int
    source_frame: str
    canonical_frame: str
    p_button_base_m: tuple[float, float, float]
    wall_surface_point_base_m: tuple[float, float, float]
    candidate_normal_base: tuple[float, float, float]
    camera_center_base_m: tuple[float, float, float]
    robot_reference_base_m: tuple[float, float, float]
    rgbd_identity: str
    rgbd_stamp_ns: int
    camera_info_identity: str
    camera_info_hash: str
    camera_info_stamp_ns: int
    d2c_identity: str
    d2c_hash: str
    d2c_stamp_ns: int
    tf_hash: str
    tf_stamp_ns: int
    calibration_identity: str
    calibration_hash: str
    calibration_observation_path: str
    calibration_observation_status: str
    calibration_observation_validation_identity: str
    calibration_acceptance_artifact: str
    calibration_acceptance_sha256: str
    accepted_camera_record_hash: str
    calibration_source_path_relative: str
    calibration_source_path_absolute: str
    calibration_validation_identity: str
    fn_pointer_path_relative: str
    fn_pointer_path_absolute: str
    fn_artifact_path_relative: str
    fn_artifact_path_absolute: str
    fn_acceptance_path_relative: str
    fn_acceptance_path_absolute: str
    fn_candidate_sidecar_path_relative: str
    fn_candidate_sidecar_path_absolute: str
    fn_validation_identity: str
    fn_artifact_hash: str
    fn_acceptance_hash: str
    collision_proxy_path_relative: str
    collision_proxy_path_absolute: str
    collision_proxy_validation_identity: str
    standby_acceptance_path_relative: str
    standby_acceptance_path_absolute: str
    standby_validation_identity: str
    standby_acceptance_hash: str
    source_registry_hash: str
    kinematic_model_hash: str
    collision_proxy_hash: str
    tool_model_hash: str
    template_hash: str
    q0_rad: tuple[float, ...]
    q0_stamp_ns: int
    q0_hash: str
    target_selection: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenSession":
        required_text = (
            "session_id", "live_boot_epoch", "qc_boot_epoch", "executor_boot_epoch",
            "cart_pose_epoch", "detection_id", "source_frame", "canonical_frame",
            "rgbd_identity", "camera_info_identity", "camera_info_hash", "d2c_identity",
            "d2c_hash", "tf_hash", "calibration_identity", "calibration_hash",
            "calibration_observation_path", "calibration_observation_status",
            "calibration_observation_validation_identity",
            "calibration_acceptance_artifact", "calibration_acceptance_sha256",
            "accepted_camera_record_hash",
            "calibration_source_path_relative", "calibration_source_path_absolute",
            "calibration_validation_identity", "fn_pointer_path_relative",
            "fn_pointer_path_absolute", "fn_artifact_path_relative",
            "fn_artifact_path_absolute", "fn_acceptance_path_relative",
            "fn_acceptance_path_absolute", "fn_candidate_sidecar_path_relative",
            "fn_candidate_sidecar_path_absolute", "fn_validation_identity",
            "fn_artifact_hash", "fn_acceptance_hash", "collision_proxy_path_relative",
            "collision_proxy_path_absolute", "collision_proxy_validation_identity",
            "standby_acceptance_path_relative", "standby_acceptance_path_absolute",
            "standby_validation_identity", "standby_acceptance_hash",
            "source_registry_hash", "kinematic_model_hash",
            "collision_proxy_hash", "tool_model_hash", "template_hash",
        )
        for key in required_text:
            if not str(value.get(key, "")).strip():
                raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", f"missing {key}")
        for key in ("camera_info_hash","d2c_hash","tf_hash","calibration_hash","calibration_acceptance_sha256","accepted_camera_record_hash","fn_artifact_hash","fn_acceptance_hash","standby_acceptance_hash","source_registry_hash","kinematic_model_hash","collision_proxy_hash","tool_model_hash","template_hash"):
            digest=str(value[key])
            if len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH",f"invalid {key}")
        if value.get("stopped_proof_mode") != STOPPED_MODE:
            raise DynamicPrepareBlocked("CART_STOP_PROOF_MISSING")
        if value.get("canonical_frame") != "roarm_base_link":
            raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "canonical frame")
        target_label = normalize_button_label(value.get("target_label"))
        selection = value.get("target_selection")
        if not target_label:
            raise DynamicPrepareBlocked("TARGET_STALE", "unknown target label")
        if selection is None:
            if target_label != "1":
                raise DynamicPrepareBlocked("TARGET_STALE", "selected target binding is required")
        else:
            if not isinstance(selection, Mapping):
                raise DynamicPrepareBlocked("TARGET_STALE", "selected target binding is not an object")
            try:
                validate_target_selection(
                    selection, target_label=target_label,
                    source_stamp_ns=int(value.get("target_stamp_ns") or 0),
                )
            except (TypeError, ValueError) as exc:
                raise DynamicPrepareBlocked("TARGET_STALE", str(exc)) from exc
        q0 = _vec(value.get("q0_rad", ()), 6, "q0")
        supplied_q0_hash = str(value.get("q0_hash", ""))
        if supplied_q0_hash != canonical_hash(list(q0)):
            raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "q0 hash")
        target_stamp = int(value.get("target_stamp_ns", 0))
        rgbd_stamp = int(value.get("rgbd_stamp_ns", 0))
        camera_info_stamp = int(value.get("camera_info_stamp_ns", 0))
        d2c_stamp = int(value.get("d2c_stamp_ns", 0))
        tf_stamp = int(value.get("tf_stamp_ns", 0))
        bound_stamps = (rgbd_stamp, camera_info_stamp, d2c_stamp, tf_stamp)
        if target_stamp <= 0 or any(stamp <= 0 for stamp in bound_stamps):
            raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "missing bound timestamp")
        if any(abs(target_stamp - stamp) > MAX_DYNAMIC_SOURCE_STAMP_SKEW_NS for stamp in (rgbd_stamp,d2c_stamp,tf_stamp)):
            raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "RGB-D/D2C/TF stamps")
        # CameraInfo is exact surface-bound calibration evidence, but its
        # static intrinsics message can be scheduled separately from the
        # dynamic target frame. Preserve its real stamp and enforce a distinct
        # structural ceiling rather than rewriting it to target time.
        if abs(target_stamp-camera_info_stamp) > MAX_CAMERA_INFO_STAMP_SKEW_NS:
            raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "CameraInfo stamp")
        cart_stamp=int(value.get("cart_confirmed_stamp_ns",0)); q0_stamp=int(value.get("q0_stamp_ns",0))
        if cart_stamp<=0 or cart_stamp>target_stamp or q0_stamp<target_stamp:
            raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH","cart/target/q0 chronology")
        result = cls(
            **{key: str(value[key]) for key in required_text},
            cart_confirmed_stamp_ns=cart_stamp, stopped_proof_mode=STOPPED_MODE,
            target_label=str(value.get("target_label")), confidence=_finite(value.get("confidence"), "confidence"),
            target_stamp_ns=target_stamp,
            p_button_base_m=_vec(value.get("p_button_base_m", ()), 3, "p_button"),
            wall_surface_point_base_m=_vec(value.get("wall_surface_point_base_m", ()), 3, "wall surface point"),
            candidate_normal_base=_vec(value.get("candidate_normal_base", ()), 3, "normal"),
            camera_center_base_m=_vec(value.get("camera_center_base_m", ()), 3, "camera"),
            robot_reference_base_m=_vec(value.get("robot_reference_base_m", ()), 3, "robot reference"),
            rgbd_stamp_ns=rgbd_stamp, camera_info_stamp_ns=camera_info_stamp,
            d2c_stamp_ns=d2c_stamp, tf_stamp_ns=tf_stamp,
            q0_rad=q0, q0_stamp_ns=q0_stamp, q0_hash=supplied_q0_hash,
            target_selection=dict(selection) if selection is not None else None,
        )
        unsigned = dict(value)
        supplied_session_id = str(unsigned.pop("session_id", ""))
        if supplied_session_id != canonical_hash(unsigned):
            # asdict() adds the new optional field when replaying a legacy
            # Button-1 session. Its original byte-bound hash omits that field.
            if selection is None:
                unsigned.pop("target_selection", None)
            if supplied_session_id != canonical_hash(unsigned):
                raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "session_id hash")
        return result

    def validate_fresh(self, now_ns: int, *, max_target_age_ns: int = 2_000_000_000, max_cart_age_ns: int = 30_000_000_000, max_q0_age_ns: int = 1_000_000_000) -> None:
        if self.cart_confirmed_stamp_ns <= 0 or self.q0_stamp_ns <= 0:
            raise DynamicPrepareBlocked("CART_STOP_PROOF_MISSING")
        if now_ns - self.target_stamp_ns > max_target_age_ns or now_ns < self.target_stamp_ns:
            raise DynamicPrepareBlocked("TARGET_STALE")
        if now_ns-self.cart_confirmed_stamp_ns>max_cart_age_ns or now_ns<self.cart_confirmed_stamp_ns:
            raise DynamicPrepareBlocked("CART_STOP_PROOF_MISSING","cart confirmation stale")
        if now_ns-self.q0_stamp_ns>max_q0_age_ns or now_ns<self.q0_stamp_ns:
            raise DynamicPrepareBlocked("TARGET_STALE","q0 stale")


@dataclass(frozen=True)
class FKState:
    tcp: tuple[float, float, float]
    axis: tuple[float, float, float]
    points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class IKAudit:
    q: tuple[float, ...]
    position_residual_m: float
    axis_residual_deg: float
    condition: float
    manipulability: float
    iterations: int
    seed_hash: str


def _validate_standard_standby_acceptance(
    policy: Mapping[str, Any],
    standby: Mapping[str, Any],
    source_bytes_by_role: Mapping[str, bytes],
) -> str:
    """Validate the boot-frozen Standard Standby evidence and source contract."""

    if standby.get("schema") != "t102_dynamic_standard_standby_acceptance_v1":
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby acceptance schema")
    validation_identity = "|".join((
        str(standby.get("artifact_id") or ""),
        str(standby.get("scope") or ""),
        str(standby.get("status") or ""),
    ))
    if validation_identity != str(policy.get("standby_validation_identity") or ""):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby validation identity mismatch")

    expected_roles = {
        "operator_designated_standby_pose": (
            "config/press/poses/press_standby_manual_20260804_preflight.yaml", None,
        ),
        "active_agx_safe_driver_config": (
            "ros2_arm/roarm_safe_driver/config/agx_t102_smooth_precontact.yaml",
            "/opt/robot-elevator-interaction/config/agx_t102_smooth_precontact.yaml",
        ),
    }
    bindings = standby.get("source_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(expected_roles):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby source bindings")
    by_role = {str(item.get("role") or ""): item for item in bindings if isinstance(item, Mapping)}
    if set(by_role) != set(expected_roles) or set(source_bytes_by_role) != set(expected_roles):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby source roles")
    for role, (expected_path, expected_runtime_path) in expected_roles.items():
        binding = by_role[role]
        if str(binding.get("path") or "") != expected_path:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"Standard Standby source path {role}")
        if expected_runtime_path is not None and str(binding.get("agx_runtime_path") or "") != expected_runtime_path:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby AGX runtime path")
        if hashlib.sha256(source_bytes_by_role[role]).hexdigest() != str(binding.get("sha256") or ""):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"Standard Standby source drift {role}")

    config_text = source_bytes_by_role["active_agx_safe_driver_config"].decode("utf-8")
    tolerance_match = re.search(r"(?m)^\s*tolerance_deg:\s*([-+0-9.eE]+)\s*(?:#.*)?$", config_text)
    target_match = re.search(r"(?m)^\s*startup_restore_target_rad:\s*(\[[^\]\r\n]+\])\s*(?:#.*)?$", config_text)
    if tolerance_match is None or target_match is None:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "active safe-driver Standby fields missing")
    configured_target = _vec(json.loads(target_match.group(1)), 6, "configured standby target")
    artifact_configured = _vec(standby.get("configured_target_rad", ()), 6, "artifact configured standby")
    if max(abs(a-b) for a,b in zip(configured_target, artifact_configured)) > 1e-12:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "active safe-driver Standby target drift")
    tolerance = math.radians(_finite(tolerance_match.group(1), "safe-driver tolerance_deg"))
    artifact_tolerance = _finite(standby.get("admission_tolerance_rad"), "standby tolerance")
    policy_tolerance = _finite(policy.get("standard_standby_tolerance_rad"), "policy standby tolerance")
    if abs(tolerance-artifact_tolerance) > 1e-12 or abs(tolerance-policy_tolerance) > 1e-12:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "active safe-driver Standby tolerance drift")

    command_target = _vec(standby.get("command_quantized_target_rad", ()), 6, "Standby command target")
    policy_target = _vec(policy.get("standard_standby_rad", ()), 6, "policy Standby target")
    if command_target != policy_target:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby target mismatch")
    step = _finite(policy.get("encoder_step_rad"), "encoder step")
    expected_first_five = tuple(round(value/step)*step for value in configured_target[:5])
    if max(abs(a-b) for a,b in zip(command_target[:5], expected_first_five)) > 1e-12:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby quantized command mismatch")
    if abs(configured_target[5]) > 1e-12 or not 0.0 < command_target[5] <= step + 1e-6:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby gripper normalization mismatch")

    evidence = standby.get("operator_evidence", {})
    if (
        evidence.get("observed_terminal_state") != "CONVERGED_DISARMED"
        or evidence.get("observed_armed") is not False
        or evidence.get("observed_dry_run") is not False
        or str(evidence.get("observed_fault") or "")
        or evidence.get("observed_joint_target_publisher_count") != 0
        or evidence.get("observed_precontact_lease_active") is not False
    ):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby operator terminal evidence")
    voltage = _finite(evidence.get("observed_voltage_v"), "Standby voltage")
    if not 11.0 <= voltage <= 13.0:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby voltage evidence")
    observed_q = _vec(evidence.get("observed_q_rad", ()), 6, "observed Standby q")
    recomputed_error = max(abs(a-b) for a,b in zip(observed_q, command_target))
    declared_error = _finite(evidence.get("observed_max_command_error_rad"), "observed Standby error")
    terminal_window_error = _finite(evidence.get("terminal_window_max_command_error_rad"), "terminal window Standby error")
    if abs(recomputed_error-declared_error) > 1e-12 or declared_error > tolerance or terminal_window_error < declared_error or terminal_window_error > tolerance:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby terminal error evidence")
    if str(evidence.get("terminal_window_source") or "") != "safe-driver retained terminal convergence telemetry":
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby terminal window provenance")

    q0_contract = standby.get("q0_contract", {})
    if (
        str(q0_contract.get("source") or "") != "fresh feedback after terminal CONVERGED_DISARMED"
        or
        q0_contract.get("freeze_measured_q0_not_command_target") is not True
        or q0_contract.get("hash_bind_measured_q0") is not True
        or _finite(q0_contract.get("maximum_age_s"), "Standby q0 maximum age") != _finite(policy.get("maximum_q0_age_s"), "policy q0 maximum age")
    ):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby measured q0 contract")
    approval = standby.get("approval", {})
    if approval.get("dynamic_prepare_q0_admission_ready") is not True:
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby approval scope")
    for field in ("move_authorized", "contact_authorized", "press_authorized", "retract_authorized", "return_authorized"):
        if approval.get(field) is not False:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"Standard Standby forbidden approval {field}")
    return validation_identity


class FrozenModel:
    """Boot-frozen standard URDF chain, tool and compact proxy policy."""

    def __init__(self, policy: Mapping[str, Any], joints: Mapping[str, ET.Element], *, boot_epoch: str, hashes: Mapping[str, str], source_bindings: Mapping[str, str] | None = None, startup_load_ms: float, execution_admission_ready: bool = True, admission_failure_reason: str = "") -> None:
        self.policy = json.loads(canonical_json(policy))
        self.joints = dict(joints)
        self.boot_epoch = boot_epoch
        self.hashes = dict(hashes)
        self.source_bindings = dict(source_bindings or {})
        self.startup_load_ms = startup_load_ms
        self.model_loaded = True
        self.model_failure_reason = ""
        self.execution_admission_ready = bool(execution_admission_ready)
        self.admission_failure_reason = str(admission_failure_reason)

    @classmethod
    def load_once(cls, repo_root: str | Path, policy_path: str | Path) -> "FrozenModel":
        started = time.perf_counter_ns()
        root = Path(repo_root).resolve()
        policy_bytes = Path(policy_path).resolve().read_bytes()
        policy = json.loads(policy_bytes)
        if policy.get("schema") != "t102_dynamic_precontact_policy_v1" or policy.get("fixed_v2_fallback") is not False:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "dynamic policy schema/fallback")
        try:
            ik_axis_weight = _finite(policy.get("ik_axis_weight"), "ik axis weight")
            camera_info_skew_s = _finite(policy.get("maximum_camera_info_stamp_skew_s"), "CameraInfo stamp skew")
        except DynamicPrepareBlocked as exc:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", str(exc)) from exc
        if not 0.0 < ik_axis_weight <= 1.0:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "ik axis weight must be in (0, 1]")
        if not 0.0 < camera_info_skew_s <= MAX_CAMERA_INFO_STAMP_SKEW_NS/1e9:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "CameraInfo stamp skew must be in (0, 0.5]")
        stage_candidates = policy.get("adaptive_stage_candidates_m")
        adaptive_waypoints = policy.get("adaptive_effective_waypoint_count_enabled")
        if stage_candidates is not None or adaptive_waypoints is not None:
            if policy.get("start_state_admission_mode") != START_STATE_ADMISSION_MODE:
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "start-state admission mode")
            if adaptive_waypoints is not True or not isinstance(stage_candidates, list):
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "adaptive plan selection contract")
            try:
                stages = tuple(_finite(value, "adaptive stage candidate") for value in stage_candidates)
                d_pre = _finite(policy.get("d_pre_m"), "precontact distance")
                configured_stage = _finite(policy.get("d_stage_m"), "stage distance")
            except DynamicPrepareBlocked as exc:
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", str(exc)) from exc
            if (
                not stages
                or stages != tuple(sorted(stages, reverse=True))
                or len(set(stages)) != len(stages)
                or abs(stages[0] - configured_stage) > 1e-12
                or any(stage <= d_pre for stage in stages)
                or not 3 <= int(policy.get("minimum_waypoints", 0))
                <= int(policy.get("maximum_waypoints", 0)) <= 6
            ):
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "adaptive plan selection bounds")
        else:
            # Legacy offline prototype policies predate the explicit field;
            # release policies must declare and boot-validate it above.
            policy.setdefault("start_state_admission_mode", START_STATE_ADMISSION_MODE)
        model_path = root / str(policy["model_source"])
        tool_relative = str(policy["tool_source"])
        fn_acceptance_relative = str(policy["fn_acceptance_source"])
        calibration_relative = str(policy["calibration_source"])
        registry_relative = str(policy["source_registry"])
        collision_relative = str(policy["collision_proxy_source"])
        standby_relative = str(policy["standby_acceptance_source"])
        tool_path = (root / tool_relative).resolve()
        fn_acceptance_path = (root / fn_acceptance_relative).resolve()
        calibration_path = (root / calibration_relative).resolve()
        registry_path = (root / registry_relative).resolve()
        collision_path = (root / collision_relative).resolve()
        standby_path = (root / standby_relative).resolve()
        model_bytes = model_path.read_bytes()
        tool_bytes = tool_path.read_bytes()
        fn_acceptance_bytes = fn_acceptance_path.read_bytes()
        calibration_bytes = calibration_path.read_bytes()
        registry_bytes = registry_path.read_bytes()
        collision_bytes = collision_path.read_bytes()
        standby_bytes = standby_path.read_bytes()
        parsed = ET.fromstring(model_bytes)
        joints = {item.get("name"): item for item in parsed.findall("joint")}
        chain = ("base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5", "link5_to_hand_tcp")
        if any(name not in joints for name in chain):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "approved FK chain incomplete")
        tool = json.loads(tool_bytes)
        if tool.get("status") != "diagnostic_overlay_default" or len(tool.get("translation_m", ())) != 3:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "tool source malformed")
        policy["tool_translation_hand_m"] = tool["translation_m"]
        artifact_relative = str(tool.get("active_candidate_path") or "")
        artifact_path = (root / artifact_relative).resolve()
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)
        if str(artifact.get("artifact_id") or "") != str(tool.get("active_artifact_id") or ""):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn pointer/artifact identity mismatch")
        if artifact.get("acceptance", {}).get("applied_as_overlay_default") is not True or artifact.get("operator_preferred_for_overlay") is not True:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn validation record missing")
        candidate_translation = artifact.get("candidate_transform", {}).get("translation_m")
        if canonical_hash(candidate_translation) != canonical_hash(tool.get("translation_m")):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn pointer/artifact translation mismatch")
        fn_acceptance = json.loads(fn_acceptance_bytes)
        if fn_acceptance.get("schema") != "precontact_outbound_fn_acceptance_v1":
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn scoped acceptance schema")
        accepted_transform = fn_acceptance.get("accepted_transform", {})
        accepted_translation = accepted_transform.get("translation_m")
        if canonical_hash(accepted_translation) != canonical_hash(candidate_translation):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn acceptance translation mismatch")
        bindings = fn_acceptance.get("source_bindings", {})
        expected_fn_bindings = {
            "active_pointer_path": tool_relative,
            "active_pointer_sha256": hashlib.sha256(tool_bytes).hexdigest(),
            "candidate_path": artifact_relative,
            "candidate_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "candidate_artifact_id": str(artifact.get("artifact_id") or ""),
            "candidate_derivation_method": str(artifact.get("derivation", {}).get("method") or ""),
            "registry_path": registry_relative,
            "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        }
        if any(str(bindings.get(key) or "") != value for key, value in expected_fn_bindings.items()):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn scoped acceptance source mismatch")
        sidecar_relative = str(bindings.get("candidate_sidecar_path") or "")
        sidecar_path = (root / sidecar_relative).resolve()
        sidecar_tokens = sidecar_path.read_text(encoding="utf-8").strip().split()
        if sidecar_tokens != [hashlib.sha256(artifact_bytes).hexdigest(), artifact_path.name]:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn candidate sidecar mismatch")
        fn_validation_identity = "|".join((
            str(fn_acceptance.get("artifact_id") or ""),
            str(fn_acceptance.get("scope") or ""),
            str(fn_acceptance.get("status") or ""),
        ))
        if fn_validation_identity != str(policy.get("fn_validation_identity") or ""):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Fn validation identity mismatch")
        fn_geometry_ready = (
            fn_acceptance.get("approval", {}).get("precontact_outbound_geometry_ready") is True
            and fn_acceptance.get("approval", {}).get("dynamic_prepare_input_allowed") is True
            and fn_acceptance.get("approval", {}).get("contact_authorized") is False
            and fn_acceptance.get("approval", {}).get("press_authorized") is False
            and fn_acceptance.get("approval", {}).get("tf_publish_authorized") is False
        )
        policy["tool_translation_hand_m"] = accepted_translation
        calibration = json.loads(calibration_bytes)
        if str(calibration.get("calibration_id") or "") != str(policy.get("calibration_identity") or ""):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "calibration identity mismatch")
        calibration_validation_identity = "|".join((str(calibration.get("calibration_id") or ""), str(calibration.get("dataset_id") or ""), str(calibration.get("method") or ""), str(calibration.get("gates_summary", {}).get("held_out_status") or "")))
        if calibration.get("b_section_exit") is not True or calibration.get("gates_summary", {}).get("held_out_gate_passed") is not True or calibration_validation_identity != str(policy.get("calibration_validation_identity") or ""):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "accepted calibration validation identity mismatch")
        accepted_camera_record = canonical_accepted_camera_record(
            calibration,
            calibration_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
            source_path_relative=calibration_relative,
            source_path_absolute=str(calibration_path),
        )
        registry = json.loads(registry_bytes)
        provenance = registry.get("provenance")
        if not isinstance(provenance, list):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "source registry provenance missing")
        expected_sources = {
            "section_b_relative_bundle": (calibration_relative, hashlib.sha256(calibration_bytes).hexdigest()),
            "fn_active_pointer": (tool_relative, hashlib.sha256(tool_bytes).hexdigest()),
            "fn_diagnostic_candidate": (artifact_relative, hashlib.sha256(artifact_bytes).hexdigest()),
        }
        for role, expected in expected_sources.items():
            matches = [(str(item.get("path") or ""), str(item.get("sha256") or "")) for item in provenance if item.get("role") == role]
            if matches != [expected]:
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"ambiguous or stale source registry role {role}")
        collision = json.loads(collision_bytes)
        if collision.get("schema") != "t102_dynamic_collision_proxy_v1":
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "collision proxy schema")
        collision_keys = (
            "capsules", "self_pairs", "base_check_capsules", "bracket_check_capsules",
            "base_proxy", "camera_bracket_proxy",
        )
        for key in collision_keys:
            if canonical_hash(policy.get(key)) != canonical_hash(collision.get(key)):
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"collision proxy policy mismatch {key}")
        collision_bindings = collision.get("source_bindings")
        if not isinstance(collision_bindings, list) or not collision_bindings:
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "collision proxy source bindings")
        for binding in collision_bindings:
            source_relative = str(binding.get("path") or "")
            expected_hash = str(binding.get("sha256") or "")
            if not source_relative or len(expected_hash) != 64:
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "collision proxy source binding malformed")
            if hashlib.sha256((root / source_relative).resolve().read_bytes()).hexdigest() != expected_hash:
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"collision proxy source drift {binding.get('role')}")
        collision_geometry_ready = collision.get("precontact_outbound_geometry_ready") is True
        standby = json.loads(standby_bytes)
        standby_bindings = standby.get("source_bindings")
        if not isinstance(standby_bindings, list):
            raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby source bindings")
        standby_source_bytes = {}
        for binding in standby_bindings:
            if not isinstance(binding, Mapping):
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby source binding malformed")
            role = str(binding.get("role") or "")
            source_relative = str(binding.get("path") or "")
            if not role or not source_relative or role in standby_source_bytes:
                raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", "Standard Standby source binding malformed")
            standby_source_bytes[role] = (root / source_relative).resolve().read_bytes()
        standby_validation_identity = _validate_standard_standby_acceptance(
            policy, standby, standby_source_bytes,
        )
        hashes = {
            "kinematic_model_hash": hashlib.sha256(model_bytes).hexdigest(),
            "collision_proxy_hash": hashlib.sha256(collision_bytes).hexdigest(),
            "tool_model_hash": hashlib.sha256(tool_bytes).hexdigest(),
            "template_hash": hashlib.sha256(policy_bytes).hexdigest(),
            "calibration_hash": hashlib.sha256(calibration_bytes).hexdigest(),
            "accepted_camera_record_hash": canonical_hash(accepted_camera_record),
            "fn_artifact_hash": hashlib.sha256(artifact_bytes).hexdigest(),
            "fn_acceptance_hash": hashlib.sha256(fn_acceptance_bytes).hexdigest(),
            "standby_acceptance_hash": hashlib.sha256(standby_bytes).hexdigest(),
            "source_registry_hash": hashlib.sha256(registry_bytes).hexdigest(),
        }
        source_bindings = {
            "calibration_source_path_relative": calibration_relative,
            "calibration_source_path_absolute": str(calibration_path),
            "calibration_validation_identity": calibration_validation_identity,
            "fn_pointer_path_relative": tool_relative,
            "fn_pointer_path_absolute": str(tool_path),
            "fn_artifact_path_relative": artifact_relative,
            "fn_artifact_path_absolute": str(artifact_path),
            "fn_acceptance_path_relative": fn_acceptance_relative,
            "fn_acceptance_path_absolute": str(fn_acceptance_path),
            "fn_candidate_sidecar_path_relative": sidecar_relative,
            "fn_candidate_sidecar_path_absolute": str(sidecar_path),
            "fn_validation_identity": fn_validation_identity,
            "collision_proxy_path_relative": collision_relative,
            "collision_proxy_path_absolute": str(collision_path),
            "collision_proxy_validation_identity": "|".join((
                str(collision.get("artifact_id") or ""),
                str(collision.get("scope") or ""),
                str(collision.get("status") or ""),
            )),
            "standby_acceptance_path_relative": standby_relative,
            "standby_acceptance_path_absolute": str(standby_path),
            "standby_validation_identity": standby_validation_identity,
        }
        approved = bool(policy.get("execution_approved")) and fn_geometry_ready and collision_geometry_ready
        blockers = []
        if not fn_geometry_ready:
            blockers.append("FN_PRECONTACT_OUTBOUND_GEOMETRY_NOT_ACCEPTED")
        if not collision_geometry_ready:
            blockers.append(str(collision.get("approval_blocker") or "COLLISION_PROXY_NOT_ACCEPTED"))
        if not bool(policy.get("execution_approved")):
            blockers.append(str(policy.get("approval_blocker") or "POLICY_NOT_APPROVED"))
        reason = "" if approved else ";".join(dict.fromkeys(blockers))
        return cls(policy, joints, boot_epoch=secrets.token_hex(16), hashes=hashes, source_bindings=source_bindings, startup_load_ms=(time.perf_counter_ns()-started)/1e6, execution_admission_ready=approved, admission_failure_reason=reason)

    @staticmethod
    def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[sum(a[r][k]*b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]

    @staticmethod
    def _transform(r: Sequence[Sequence[float]], t: Sequence[float]) -> list[list[float]]:
        return [[r[i][j] if j < 3 else float(t[i]) for j in range(4)] for i in range(3)] + [[0.0,0.0,0.0,1.0]]

    @staticmethod
    def _rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
        cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
        return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]

    @staticmethod
    def _axis(axis: Sequence[float], angle: float) -> list[list[float]]:
        x,y,z=_unit(axis,"urdf axis"); c,s,o=math.cos(angle),math.sin(angle),1-math.cos(angle)
        return [[c+x*x*o,x*y*o-z*s,x*z*o+y*s],[y*x*o+z*s,c+y*y*o,y*z*o-x*s],[z*x*o-y*s,z*y*o+x*s,c+z*z*o]]

    def fk(self, q: Sequence[float]) -> FKState:
        qv = _vec(q, 6, "q")
        by_name = dict(zip(self.policy["joint_names"], qv))
        total = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
        points=[(0.,0.,0.)]
        chain=("base_link_to_link1","link1_to_link2","link2_to_link3","link3_to_link4","link4_to_link5","link5_to_hand_tcp")
        for name in chain:
            joint=self.joints[name]; origin=joint.find("origin")
            xyz=[float(v) for v in ((origin.get("xyz") if origin is not None and origin.get("xyz") else "0 0 0").split())]
            rpy=[float(v) for v in ((origin.get("rpy") if origin is not None and origin.get("rpy") else "0 0 0").split())]
            step=self._transform(self._rpy(*rpy),xyz)
            if joint.get("type") in {"revolute","continuous"}:
                node=joint.find("axis"); axis=[float(v) for v in ((node.get("xyz") if node is not None and node.get("xyz") else "1 0 0").split())]
                step=self._matmul(step,self._transform(self._axis(axis,by_name[name]),(0,0,0)))
            total=self._matmul(total,step); points.append(tuple(total[i][3] for i in range(3)))
        rotation=[row[:3] for row in total[:3]]
        translation=_vec(self.policy["tool_translation_hand_m"],3,"tool translation")
        offset=tuple(sum(rotation[i][j]*translation[j] for j in range(3)) for i in range(3))
        tcp=_add(points[-1],offset)
        axis_hand=_vec(self.policy["nominal_tool_axis_hand_unit"],3,"tool axis")
        axis=_unit(tuple(sum(rotation[i][j]*axis_hand[j] for j in range(3)) for i in range(3)),"tool axis")
        return FKState(tcp,axis,tuple(points)+(tcp,))

    @staticmethod
    def _linear(matrix: list[list[float]], rhs: list[float]) -> tuple[list[float], float]:
        n=len(rhs); aug=[matrix[i][:]+[rhs[i]] for i in range(n)]; pivots=[]
        for col in range(n):
            pivot=max(range(col,n),key=lambda r:abs(aug[r][col])); aug[col],aug[pivot]=aug[pivot],aug[col]
            if abs(aug[col][col])<1e-12: raise DynamicPrepareBlocked("IK_NEAR_SINGULAR")
            pivots.append(abs(aug[col][col]))
            for row in range(col+1,n):
                factor=aug[row][col]/aug[col][col]
                for k in range(col,n+1): aug[row][k]-=factor*aug[col][k]
        out=[0.]*n
        for row in range(n-1,-1,-1): out[row]=(aug[row][n]-sum(aug[row][k]*out[k] for k in range(row+1,n)))/aug[row][row]
        return out,max(pivots)/min(pivots)

    @staticmethod
    def _eigenvalues_symmetric(matrix: Sequence[Sequence[float]]) -> list[float]:
        """Jacobi eigenvalues for the small symmetric J^T J matrix."""
        a=[list(map(float,row)) for row in matrix]; n=len(a)
        for _ in range(80):
            p,q=max(((i,j) for i in range(n) for j in range(i+1,n)),key=lambda ij:abs(a[ij[0]][ij[1]]))
            if abs(a[p][q])<1e-14: break
            phi=.5*math.atan2(2*a[p][q],a[q][q]-a[p][p]); c,s=math.cos(phi),math.sin(phi)
            app,aqq,apq=a[p][p],a[q][q],a[p][q]
            a[p][p]=c*c*app-2*s*c*apq+s*s*aqq; a[q][q]=s*s*app+2*s*c*apq+c*c*aqq; a[p][q]=a[q][p]=0.
            for k in range(n):
                if k in (p,q): continue
                akp,akq=a[k][p],a[k][q]; a[k][p]=a[p][k]=c*akp-s*akq; a[k][q]=a[q][k]=s*akp+c*akq
        return sorted(max(0.,a[i][i]) for i in range(n))

    def _jacobian_audit(self, q: Sequence[float], *, eps: float = 0.0005) -> tuple[list[list[float]], list[list[float]], float, float]:
        """Return the single authoritative weighted TCP/axis Jacobian audit."""
        qv=list(_vec(q,6,"jacobian q")); state=self.fk(qv); cols=[]; axis_weight=float(self.policy["ik_axis_weight"])
        for joint_index in range(5):
            perturbed=qv[:]; perturbed[joint_index]+=eps; sample=self.fk(perturbed)
            cols.append([(sample.tcp[k]-state.tcp[k])/eps for k in range(3)]+[axis_weight*(sample.axis[k]-state.axis[k])/eps for k in range(3)])
        gram=[[sum(cols[a][row]*cols[b][row] for row in range(6)) for b in range(5)] for a in range(5)]
        singular=[math.sqrt(value) for value in self._eigenvalues_symmetric(gram)]
        condition=math.inf if singular[0]<1e-12 else singular[-1]/singular[0]
        manipulability=math.prod(singular)
        return cols,gram,condition,manipulability

    def jacobian_metrics(self, q: Sequence[float]) -> tuple[float, float]:
        """Recompute condition/manipulability for an exact quantized q."""
        _cols,_gram,condition,manipulability=self._jacobian_audit(q)
        return condition,manipulability

    def solve(self, seed: Sequence[float], target: Sequence[float], desired_axis: Sequence[float]) -> IKAudit:
        q=list(_vec(seed,6,"seed")); target=_vec(target,3,"target"); desired=_unit(desired_axis,"desired axis"); eps=0.0005; axis_weight=float(self.policy["ik_axis_weight"])
        # link4_to_link5 is the physical wrist-camera roll joint. Tool-axis
        # alignment alone leaves this degree of freedom underconstrained and
        # previously allowed the camera to rotate roughly 70 degrees during a
        # no-contact approach. Dynamic precontact therefore solves only the
        # four positioning/axis joints and holds camera roll exactly at the
        # fresh measured q0 seed. The gripper was already held the same way.
        wrist_roll_locked=bool(self.policy.get("wrist_roll_lock_enabled",False))
        solve_joint_indices=range(4) if wrist_roll_locked else range(5)
        # Retain a small deterministic reserve for encoder quantization; the
        # authoritative post-quantization gate still uses the declared limit.
        axis_convergence_deg=max(0.,float(self.policy["axis_residual_limit_deg"])-0.5)
        last_condition=math.inf; last_manip=0.; iteration=0
        for iteration in range(80):
            state=self.fk(q); pos=_sub(target,state.tcp); axis_err=(desired[0]-state.axis[0],desired[1]-state.axis[1],desired[2]-state.axis[2]); residual=list(pos)+[axis_weight*x for x in axis_err]
            pos_error=_norm(pos); axis_deg=_angle_deg(state.axis,desired)
            cols,raw_gram,last_condition,last_manip=self._jacobian_audit(q,eps=eps)
            if not math.isfinite(last_condition) or last_condition > float(self.policy["jacobian_condition_limit"]) or last_manip < float(self.policy["manipulability_floor"]): raise DynamicPrepareBlocked("IK_NEAR_SINGULAR")
            if pos_error <= float(self.policy["position_residual_limit_m"]) and axis_deg <= axis_convergence_deg: break
            # Pure bounded DLS with a monotone backtracking merit check.  The
            # previous tiny damping allowed the orientation rows to drive a
            # large wrist excursion even when the seed axis was already in
            # the accepted window.  The same boot-frozen axis weight used by
            # the authoritative Jacobian audit defines this merit function.
            gram=[[raw_gram[a][b]+(1e-3 if a==b else 0.) for b in solve_joint_indices] for a in solve_joint_indices]
            rhs=[sum(cols[a][r]*residual[r] for r in range(6)) for a in solve_joint_indices]
            delta,_=self._linear(gram,rhs)
            merit=sum(value*value for value in residual); accepted=False
            for scale in (1.,.5,.25,.125,.0625,.03125):
                candidate=q[:]
                for delta_index,ji in enumerate(solve_joint_indices): candidate[ji]+=scale*max(-0.05,min(0.05,delta[delta_index]))
                if wrist_roll_locked: candidate[4]=float(seed[4])
                candidate[5]=float(seed[5])
                try:
                    self._validate_limits(candidate)
                except DynamicPrepareBlocked:
                    continue
                sample=self.fk(candidate); sample_pos=_sub(target,sample.tcp); sample_axis=_sub(desired,sample.axis)
                candidate_merit=sum(value*value for value in sample_pos)+sum((axis_weight*value)**2 for value in sample_axis)
                if candidate_merit < merit-1e-15:
                    q=candidate; accepted=True; break
            if not accepted:
                break
        final=self.fk(q); pos_error=_norm(_sub(target,final.tcp)); axis_deg=_angle_deg(final.axis,desired)
        if pos_error > float(self.policy["position_residual_limit_m"]): raise DynamicPrepareBlocked("IK_RESIDUAL_EXCEEDED",f"{pos_error}")
        if axis_deg > float(self.policy["axis_residual_limit_deg"]): raise DynamicPrepareBlocked("ORIENTATION_WINDOW_MISS",f"{axis_deg}")
        if wrist_roll_locked and q[4] != float(seed[4]): raise DynamicPrepareBlocked("WRIST_ROLL_LOCK_FAILED")
        self._validate_limits(q)
        _cols,_gram,last_condition,last_manip=self._jacobian_audit(q,eps=eps)
        if not math.isfinite(last_condition) or last_condition > float(self.policy["jacobian_condition_limit"]) or last_manip < float(self.policy["manipulability_floor"]): raise DynamicPrepareBlocked("IK_NEAR_SINGULAR")
        return IKAudit(tuple(q),pos_error,axis_deg,last_condition,last_manip,iteration+1,canonical_hash(list(seed)))

    def _validate_limits(self, q: Sequence[float]) -> None:
        buffer=float(self.policy["joint_limit_buffer_rad"])
        for i,(value,limits) in enumerate(zip(q,self.policy["joint_hard_limits_rad"])):
            # The gripper is held exactly at the measured seed and is not an IK
            # degree of freedom; retain its firmware hard limits without a
            # fictitious arm-joint buffer that would reject Standard Standby.
            effective_buffer = 0.0 if i == 5 else buffer
            if not float(limits[0])+effective_buffer <= float(value) <= float(limits[1])-effective_buffer:
                raise DynamicPrepareBlocked("JOINT_LIMIT_REJECTED",str(i))


def _point_segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ab=_sub(b,a); denom=_dot(ab,ab); t=0.0 if denom<=1e-15 else max(0.,min(1.,_dot(_sub(p,a),ab)/denom))
    return _norm(_sub(p,_add(a,_scale(ab,t))))


def _segment_distance(a0: Sequence[float],a1: Sequence[float],b0: Sequence[float],b1: Sequence[float]) -> float:
    # Closest points on two finite 3-D segments (Ericson, Real-Time Collision Detection).
    u=_sub(a1,a0); v=_sub(b1,b0); w=_sub(a0,b0); a=_dot(u,u); b=_dot(u,v); c=_dot(v,v); d=_dot(u,w); e=_dot(v,w); eps=1e-15
    if a<=eps and c<=eps: return _norm(w)
    if a<=eps: return _point_segment_distance(a0,b0,b1)
    if c<=eps: return _point_segment_distance(b0,a0,a1)
    den=a*c-b*b
    s=0. if den<=eps else max(0.,min(1.,(b*e-c*d)/den))
    t=(b*s+e)/c
    if t<0.: t=0.; s=max(0.,min(1.,-d/a))
    elif t>1.: t=1.; s=max(0.,min(1.,(b-d)/a))
    return _norm(_sub(_add(a0,_scale(u,s)),_add(b0,_scale(v,t))))


def _quantize(q: Sequence[float], step: float) -> tuple[float,...]:
    return tuple(round(float(x)/step)*step for x in q)


def _select_bounded_waypoint_indices(
    audits: Sequence[IKAudit],
    q0: Sequence[float],
    *,
    stage_index: int,
    caps: Sequence[float],
    encoder_step_rad: float,
    minimum_useful_delta_rad: float,
    minimum_waypoints: int,
    maximum_waypoints: int,
    cartesian_sample_count: int,
    maximum_joint_reversal_rad: float,
) -> list[int]:
    """Choose the shortest valid subsequence while preserving stage/final.

    The old midpoint insertion could spend two slots where one farther
    accepted sample satisfied every cap, then reject a route that had a valid
    six-waypoint subsequence.  This bounded exhaustive search has at most 16
    accepted IK samples in the production policy, so it is deterministic and
    cheap while checking the exact quantized command deltas that would be
    audited later.
    """

    final_index = len(audits) - 1
    if not (0 <= stage_index < final_index):
        raise DynamicPrepareBlocked("WAYPOINT_CAP_REJECTED", "stage/final indices")
    lower = max(3, int(minimum_waypoints))
    upper = min(6, int(maximum_waypoints), len(audits))
    if lower > upper:
        raise DynamicPrepareBlocked("WAYPOINT_CAP_REJECTED", "invalid waypoint bounds")
    mandatory = {int(stage_index), int(final_index)}
    candidates = tuple(range(len(audits)))
    step = float(encoder_step_rad)
    deadband = float(minimum_useful_delta_rad)
    reversal_limit = float(maximum_joint_reversal_rad)
    saw_deadband_only = False
    saw_reversal_only = False

    for count in range(lower, upper + 1):
        valid_choices: list[tuple[tuple[float, float, tuple[int, ...]], tuple[int, ...]]] = []
        for chosen in combinations(candidates, count):
            if not mandatory.issubset(chosen):
                continue
            previous_raw = tuple(float(x) for x in q0)
            previous_quantized = previous_raw
            previous_delta = None
            valid = True
            rejection = ""
            for position, sample_index in enumerate(chosen):
                raw = tuple(float(x) for x in audits[sample_index].q)
                quantized = _quantize(raw, step)
                if any(
                    abs(raw[j] - previous_raw[j]) > float(caps[j]) + 1e-12
                    or abs(quantized[j] - previous_quantized[j])
                    > float(caps[j]) + 1e-12
                    for j in range(6)
                ):
                    valid = False
                    rejection = "cap"
                    break
                delta = tuple(
                    quantized[j] - previous_quantized[j] for j in range(6)
                )
                if max(abs(value) for value in delta) < deadband:
                    valid = False
                    rejection = "deadband"
                    break
                if previous_delta is not None:
                    previous_index = chosen[position - 1]
                    same_cartesian_leg = (
                        (previous_index <= stage_index and sample_index <= stage_index)
                        or (previous_index > stage_index and sample_index > stage_index)
                    )
                    previous_in_initial_settlement = (
                        previous_index < max(1, int(cartesian_sample_count) // 2)
                    )
                    if same_cartesian_leg and not previous_in_initial_settlement:
                        if any(
                            prior * current < 0
                            and min(abs(prior), abs(current)) > reversal_limit
                            for prior, current in zip(previous_delta, delta)
                        ):
                            valid = False
                            rejection = "reversal"
                            break
                previous_raw = raw
                previous_quantized = quantized
                previous_delta = delta
            if valid:
                first_quantized = _quantize(audits[chosen[0]].q, step)
                first_delta = tuple(
                    abs(first_quantized[j] - float(q0[j])) for j in range(6)
                )
                active = tuple(value for value in first_delta if value >= 0.5 * step)
                # Among equally short, fully valid routes, prefer the one with
                # the strongest first physical command.  The RoArm firmware is
                # known to under-execute tiny multi-joint deltas without a
                # reissue; Dynamic Move forbids reissue, so lexicographically
                # choosing the earliest IK sample can make the first waypoint
                # physically unreachable even though a later accepted sample
                # obeys the same caps, residuals, constraints and 3-6 bound.
                score = (
                    min(active) if active else 0.0,
                    sum(value * value for value in first_delta),
                    chosen,
                )
                valid_choices.append((score, chosen))
                continue
            if rejection == "deadband":
                saw_deadband_only = True
            elif rejection == "reversal":
                saw_reversal_only = True
        if valid_choices:
            return list(max(valid_choices, key=lambda item: item[0])[1])
    if saw_deadband_only:
        raise DynamicPrepareBlocked(
            "DEADBAND_SHAPE_REJECTED",
            "no bounded waypoint subsequence preserves useful command deltas",
        )
    if saw_reversal_only:
        raise DynamicPrepareBlocked(
            "IK_BRANCH_DISCONTINUITY",
            "no bounded waypoint subsequence preserves reversal constraints",
        )
    raise DynamicPrepareBlocked(
        "WAYPOINT_CAP_REJECTED",
        "no 3-6 waypoint subsequence satisfies raw/quantized caps",
    )


def _build_effective_joint_partition(
    q0: Sequence[float],
    final_q: Sequence[float],
    *,
    count: int,
    caps: Sequence[float],
    encoder_step_rad: float,
    minimum_useful_delta_rad: float,
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], list[float]]:
    """Repartition the exact Dynamic endpoint into larger effective commands.

    T102 is known to stop short when a long path is fragmented into tiny
    commands.  A physical release may therefore use a reviewed 3-6 monotonic
    joint-space partition to the already-solved, exact Dynamic endpoint.
    This changes neither the endpoint nor Move-time behavior: every effective
    waypoint is frozen during Prepare and later published exactly once.  The
    caller still applies the full joint-limit, orientation, Jacobian and
    sampled no-contact checks to the resulting segments.
    """

    if not 3 <= int(count) <= 6:
        raise DynamicPrepareBlocked(
            "WAYPOINT_CAP_REJECTED", "effective waypoint count must be 3-6"
        )
    step = float(encoder_step_rad)
    deadband = float(minimum_useful_delta_rad)
    raw_waypoints: list[tuple[float, ...]] = []
    waypoints: list[tuple[float, ...]] = []
    ratios: list[float] = []
    previous_raw = tuple(float(value) for value in q0)
    previous_quantized = previous_raw
    for index in range(1, int(count) + 1):
        ratio = index / float(count)
        raw = tuple(
            float(q0[j]) + ratio * (float(final_q[j]) - float(q0[j]))
            for j in range(6)
        )
        quantized = _quantize(raw, step)
        if any(
            abs(raw[j] - previous_raw[j]) > float(caps[j]) + 1e-12
            or abs(quantized[j] - previous_quantized[j])
            > float(caps[j]) + 1e-12
            for j in range(6)
        ):
            raise DynamicPrepareBlocked(
                "WAYPOINT_CAP_REJECTED",
                f"effective partition {index - 1} exceeds command cap",
            )
        if max(
            abs(quantized[j] - previous_quantized[j]) for j in range(6)
        ) < deadband:
            raise DynamicPrepareBlocked(
                "DEADBAND_SHAPE_REJECTED",
                f"effective partition {index - 1} is below useful delta",
            )
        raw_waypoints.append(raw)
        waypoints.append(quantized)
        ratios.append(ratio)
        previous_raw = raw
        previous_quantized = quantized
    return raw_waypoints, waypoints, ratios


@dataclass(frozen=True)
class PreparedPlan:
    plan: Mapping[str, Any]
    plan_hash: str
    expires_monotonic: float
    consumed: bool = False


def prepare_dynamic_plan(
    model: FrozenModel,
    session: FrozenSession,
    *,
    now_ns: int,
    now_monotonic: float,
    solver: Callable[[Sequence[float],Sequence[float],Sequence[float]],IKAudit] | None = None,
    initial_disarmed_state: str = "CONVERGED_DISARMED",
) -> PreparedPlan:
    if initial_disarmed_state not in {
        "CONVERGED_DISARMED", "DISARMED:operator",
        "FAULT_CLEARED_DISARMED", "TORQUE_ON_VERIFIED",
    }:
        raise DynamicPrepareBlocked("PLAN_INVALIDATED", "initial disarmed state")
    if not bool(getattr(model, "model_loaded", False)):
        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", str(getattr(model, "model_failure_reason", "boot model unavailable")))
    if not bool(getattr(model, "execution_admission_ready", False)):
        raise DynamicPrepareBlocked("TOOL_MODEL_OR_COLLISION_PROXY_NOT_APPROVED", str(getattr(model, "admission_failure_reason", "execution sources not approved")))
    base_policy=model.policy
    if base_policy.get("adaptive_effective_waypoint_count_enabled") is True:
        failures=[]
        invariant_failure_codes={
            "MODEL_LOAD_FAILED","TOOL_MODEL_OR_COLLISION_PROXY_NOT_APPROVED",
            "TARGET_STALE","CART_STOP_PROOF_MISSING","RGBD_TF_SESSION_MISMATCH",
            "PLAN_INVALIDATED","NONFINITE_INPUT","NORMAL_DIRECTION_AMBIGUOUS",
            "WRIST_ROLL_LOCK_FAILED",
        }
        minimum=int(base_policy["minimum_waypoints"])
        maximum=int(base_policy["maximum_waypoints"])
        candidates=tuple(float(value) for value in base_policy["adaptive_stage_candidates_m"])
        for stage_index,stage_distance in enumerate(candidates):
            stage_solver_cache={}
            stage_solver_reuse_hits=0
            uncached_solver=solver or model.solve

            def stage_solver(seed,target,desired):
                nonlocal stage_solver_reuse_hits
                if solver is not None:
                    return solver(seed,target,desired)
                key=(tuple(seed),tuple(target),tuple(desired))
                if key in stage_solver_cache:
                    stage_solver_reuse_hits+=1
                    return stage_solver_cache[key]
                # Do not cache solver exceptions: an injected solver retains
                # its existing call/failure behavior.  Only completed audits
                # are reused across waypoint-count candidates for this stage.
                audit=uncached_solver(seed,target,desired)
                stage_solver_cache[key]=audit
                return audit

            for waypoint_count in range(minimum,maximum+1):
                try:
                    ticket=_prepare_dynamic_plan_candidate(
                        model,session,now_ns=now_ns,now_monotonic=now_monotonic,
                        solver=stage_solver,_stage_distance_m=stage_distance,
                        _effective_waypoint_count=waypoint_count,
                        initial_disarmed_state=initial_disarmed_state,
                    )
                except DynamicPrepareBlocked as exc:
                    if exc.code in invariant_failure_codes:
                        raise
                    failures.append({"stage_m":stage_distance,"waypoint_count":waypoint_count,"code":exc.code})
                    continue
                payload=dict(ticket.plan)
                payload["adaptive_plan_selection"]={
                    "enabled":True,
                    "selection_order":"FARTHEST_STAGE_THEN_MINIMUM_WAYPOINT_COUNT",
                    "declared_stage_candidates_m":list(candidates),
                    "declared_waypoint_count_range":[minimum,maximum],
                    "selected_stage_candidate_index":stage_index,
                    "selected_stage_m":stage_distance,
                    "selected_effective_waypoint_count":waypoint_count,
                    "partial_candidate_reuse":stage_solver_reuse_hits>0,
                    "runtime_retry":False,
                }
                return PreparedPlan(
                    payload,canonical_hash(payload),ticket.expires_monotonic,ticket.consumed,
                )
        detail=",".join(
            f"{item['stage_m']:.6f}m/{item['waypoint_count']}:{item['code']}"
            for item in failures
        )
        raise DynamicPrepareBlocked("WAYPOINT_CAP_REJECTED",f"no adaptive candidate passed [{detail}]")
    return _prepare_dynamic_plan_candidate(
        model,session,now_ns=now_ns,now_monotonic=now_monotonic,solver=solver,
        initial_disarmed_state=initial_disarmed_state,
    )


def _prepare_dynamic_plan_candidate(
    model: FrozenModel,
    session: FrozenSession,
    *,
    now_ns: int,
    now_monotonic: float,
    solver: Callable[[Sequence[float],Sequence[float],Sequence[float]],IKAudit] | None = None,
    _stage_distance_m: float | None = None,
    _effective_waypoint_count: int | None = None,
    initial_disarmed_state: str = "CONVERGED_DISARMED",
) -> PreparedPlan:
    policy=dict(model.policy)
    if _stage_distance_m is not None:
        policy["d_stage_m"]=float(_stage_distance_m)
    if _effective_waypoint_count is not None:
        policy["effective_waypoint_count"]=int(_effective_waypoint_count)
    session.validate_fresh(now_ns,max_cart_age_ns=int(float(policy["maximum_cart_confirmation_age_s"])*1e9),max_q0_age_ns=int(float(policy["maximum_q0_age_s"])*1e9))
    camera_info_skew_limit_ns=int(float(policy["maximum_camera_info_stamp_skew_s"])*1e9)
    if abs(session.target_stamp_ns-session.camera_info_stamp_ns)>camera_info_skew_limit_ns:
        raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH","CameraInfo stamp policy")
    if session.confidence < float(policy["minimum_target_confidence"]): raise DynamicPrepareBlocked("TARGET_STALE","confidence below threshold")
    if session.executor_boot_epoch != model.boot_epoch: raise DynamicPrepareBlocked("PLAN_INVALIDATED","executor restart")
    identity_checks=((session.kinematic_model_hash,model.hashes["kinematic_model_hash"]),(session.collision_proxy_hash,model.hashes["collision_proxy_hash"]),(session.tool_model_hash,model.hashes["tool_model_hash"]),(session.template_hash,model.hashes["template_hash"]),(session.calibration_hash,model.hashes["calibration_hash"]),(session.accepted_camera_record_hash,model.hashes["accepted_camera_record_hash"]),(session.fn_artifact_hash,model.hashes["fn_artifact_hash"]),(session.fn_acceptance_hash,model.hashes["fn_acceptance_hash"]),(session.standby_acceptance_hash,model.hashes["standby_acceptance_hash"]),(session.source_registry_hash,model.hashes["source_registry_hash"]),(session.calibration_identity,str(policy["calibration_identity"])))
    identity_checks += tuple((str(getattr(session, key)), str(value)) for key,value in model.source_bindings.items())
    if any(a!=b for a,b in identity_checks): raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH","model/template/tool/proxy/calibration identity")
    model._validate_limits(session.q0_rad)
    proof=orient_outward_normal(session.candidate_normal_base,session.p_button_base_m,session.camera_center_base_m,session.robot_reference_base_m,min_dot=float(policy["normal_reference_min_dot"]),unit_tolerance=float(policy["normal_unit_tolerance"]))
    button_plane_offset=abs(_dot(proof.n_out,_sub(session.p_button_base_m,session.wall_surface_point_base_m)))
    if button_plane_offset>float(policy["maximum_button_plane_offset_m"]):
        raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH","button/wall plane mismatch")
    d_pre,d_stage=float(policy["d_pre_m"]),float(policy["d_stage_m"])
    if not d_stage>d_pre>0: raise DynamicPrepareBlocked("MODEL_LOAD_FAILED","distance policy")
    p_pre=_add(session.p_button_base_m,_scale(proof.n_out,d_pre)); p_stage=_add(session.p_button_base_m,_scale(proof.n_out,d_stage)); desired=tuple(-x for x in proof.n_out)
    solve=solver or model.solve; current=model.fk(session.q0_rad).tcp; cart=[]
    count=int(policy["cartesian_sample_count"])
    for a,b in ((current,p_stage),(p_stage,p_pre)):
        for i in range(1,count+1): cart.append(_add(a,_scale(_sub(b,a),i/count)))
    seed=session.q0_rad; audits=[]; previous_ik_delta=None
    for audit_index,target in enumerate(cart):
        audit=solve(seed,target,desired)
        if audit.seed_hash!=canonical_hash(list(seed)): raise DynamicPrepareBlocked("IK_BRANCH_DISCONTINUITY","seed chain")
        if not math.isfinite(audit.position_residual_m) or audit.position_residual_m>float(policy["position_residual_limit_m"]): raise DynamicPrepareBlocked("IK_RESIDUAL_EXCEEDED","solver audit")
        if not math.isfinite(audit.axis_residual_deg) or audit.axis_residual_deg>float(policy["axis_residual_limit_deg"]): raise DynamicPrepareBlocked("ORIENTATION_WINDOW_MISS","solver audit")
        if not math.isfinite(audit.condition) or audit.condition>float(policy["jacobian_condition_limit"]) or not math.isfinite(audit.manipulability) or audit.manipulability<float(policy["manipulability_floor"]): raise DynamicPrepareBlocked("IK_NEAR_SINGULAR","solver audit")
        model._validate_limits(audit.q)
        max_seed_jump=max(abs(a-b) for a,b in zip(audit.q,seed))
        if max_seed_jump>float(policy["maximum_joint_jump_rad"]):
            raise DynamicPrepareBlocked(
                "IK_BRANCH_DISCONTINUITY",
                f"IK sample {audit_index} seed jump {max_seed_jump:.9f} rad exceeds "
                f"{float(policy['maximum_joint_jump_rad']):.9f} rad",
            )
        ik_delta=tuple(a-b for a,b in zip(audit.q,seed))
        # The first half of the current->stage leg is a bounded initial axis
        # settlement region.  A single direction change there is not branch
        # switching: every sample still obeys seed-distance jump, hard-limit,
        # residual, axis and Jacobian gates.  Reversal checks resume for the
        # remainder of that leg and for the complete stage->pre leg, so a
        # sustained alternating solver remains fail-closed.
        settlement_end=max(2,count//2)
        # A reversal across the final settlement sample and the first sample
        # after it still belongs to the same bounded settling turn.  Mirror
        # the later quantized-waypoint audit, which keys this exemption to the
        # previous sample.  All reversals wholly after settlement remain
        # rejected, as do every jump/residual/limit/Jacobian gate above.
        previous_in_initial_settlement=(audit_index-1)<settlement_end
        if previous_ik_delta is not None and audit_index!=count and not previous_in_initial_settlement:
            for joint,(prior,current_delta) in enumerate(zip(previous_ik_delta,ik_delta)):
                if prior*current_delta<0 and min(abs(prior),abs(current_delta))>float(policy["maximum_joint_reversal_rad"]): raise DynamicPrepareBlocked("IK_BRANCH_DISCONTINUITY",f"IK sample reversal joint {joint}")
        previous_ik_delta=ik_delta
        audits.append(audit); seed=audit.q
    stage_index=count-1
    caps=tuple(float(x) for x in policy["per_command_caps_rad"])
    step=float(policy["encoder_step_rad"])
    effective_count=int(policy.get("effective_waypoint_count",0) or 0)
    effective_raw_waypoints=[]; effective_ratios=[]
    if effective_count:
        if not int(policy["minimum_waypoints"]) <= effective_count <= int(policy["maximum_waypoints"]):
            raise DynamicPrepareBlocked("WAYPOINT_CAP_REJECTED","effective waypoint count outside policy bounds")
        effective_raw_waypoints,waypoints,effective_ratios=_build_effective_joint_partition(
            session.q0_rad,
            audits[-1].q,
            count=effective_count,
            caps=caps,
            encoder_step_rad=step,
            minimum_useful_delta_rad=float(policy["minimum_useful_delta_rad"]),
        )
        waypoint_indices=[None]*effective_count
        waypoint_indices[-1]=len(audits)-1
    else:
        waypoint_indices=_select_bounded_waypoint_indices(
            audits,
            session.q0_rad,
            stage_index=stage_index,
            caps=caps,
            encoder_step_rad=step,
            minimum_useful_delta_rad=float(policy["minimum_useful_delta_rad"]),
            minimum_waypoints=int(policy["minimum_waypoints"]),
            maximum_waypoints=int(policy["maximum_waypoints"]),
            cartesian_sample_count=count,
            maximum_joint_reversal_rad=float(policy["maximum_joint_reversal_rad"]),
        )
        waypoints=[_quantize(audits[i].q,step) for i in waypoint_indices]
    if max(abs(a-b) for a,b in zip(waypoints[-1],session.q0_rad))<float(policy["minimum_useful_delta_rad"]): waypoints=[]
    if not waypoints: raise DynamicPrepareBlocked("DEADBAND_SHAPE_REJECTED")
    previous=session.q0_rad; previous_delta=None; quantized_audit=[]
    for index,q in enumerate(waypoints):
        model._validate_limits(q); delta=tuple(q[i]-previous[i] for i in range(6))
        if any(abs(delta[i])>caps[i]+1e-12 for i in range(6)): raise DynamicPrepareBlocked("WAYPOINT_CAP_REJECTED",str(index))
        if max(abs(x) for x in delta)<float(policy["minimum_useful_delta_rad"]): raise DynamicPrepareBlocked("DEADBAND_SHAPE_REJECTED",str(index))
        same_cartesian_leg = (not effective_count) and index > 0 and ((waypoint_indices[index-1] <= stage_index and waypoint_indices[index] <= stage_index) or (waypoint_indices[index-1] > stage_index and waypoint_indices[index] > stage_index))
        previous_in_initial_settlement = (waypoint_indices[index-1] < max(1,count//2)) if (not effective_count and index > 0) else False
        if previous_delta is not None and same_cartesian_leg and not previous_in_initial_settlement:
            for joint,(prior,current_delta) in enumerate(zip(previous_delta,delta)):
                if prior*current_delta<0 and min(abs(prior),abs(current_delta))>float(policy["maximum_joint_reversal_rad"]): raise DynamicPrepareBlocked("IK_BRANCH_DISCONTINUITY",f"reversal joint {joint}")
        state=model.fk(q)
        if effective_count:
            raw_state=model.fk(effective_raw_waypoints[index])
            residual=_norm(_sub(state.tcp,raw_state.tcp))
        else:
            target=cart[waypoint_indices[index]]
            residual=_norm(_sub(state.tcp,target))
        if residual>float(policy["position_residual_limit_m"]): raise DynamicPrepareBlocked("IK_RESIDUAL_EXCEEDED",f"post quantization waypoint {index}")
        axis_residual=_angle_deg(state.axis,desired)
        if axis_residual>float(policy["axis_residual_limit_deg"]): raise DynamicPrepareBlocked("ORIENTATION_WINDOW_MISS",f"post quantization waypoint {index}")
        condition,manipulability=model.jacobian_metrics(q)
        if not math.isfinite(condition) or condition>float(policy["jacobian_condition_limit"]) or not math.isfinite(manipulability) or manipulability<float(policy["manipulability_floor"]): raise DynamicPrepareBlocked("IK_NEAR_SINGULAR",f"post quantization waypoint {index}")
        quantized_audit.append({"cartesian_sample_index":waypoint_indices[index],"effective_partition_ratio":(effective_ratios[index] if effective_count else None),"audit_reference":("effective_joint_partition_unquantized_fk" if effective_count else "cartesian_ik_sample"),"position_residual_m":residual,"axis_residual_deg":axis_residual,"condition":condition,"manipulability":manipulability})
        previous_delta=delta
        previous=q
    final_state=model.fk(waypoints[-1]); final_residual=_norm(_sub(final_state.tcp,p_pre)); final_axis=_angle_deg(final_state.axis,desired)
    if final_residual>float(policy["position_residual_limit_m"]): raise DynamicPrepareBlocked("IK_RESIDUAL_EXCEEDED","post quantization")
    if final_axis>float(policy["axis_residual_limit_deg"]): raise DynamicPrepareBlocked("ORIENTATION_WINDOW_MISS","post quantization")
    constraint_counts={"wall":0,"self":0,"base":0,"bracket":0}; cursor=session.q0_rad
    for waypoint in waypoints:
        for sample_i in range(int(policy["segment_sample_count"])):
            ratio=sample_i/(int(policy["segment_sample_count"])-1); q=_add(cursor,_scale(_sub(waypoint,cursor),ratio)); state=model.fk(q)
            capsules={}
            for item in policy["capsules"]:
                a,b=state.points[int(item["joint_a"])],state.points[int(item["joint_b"])]; radius=float(item["radius_m"]); capsules[item["name"]]=(a,b,radius)
                if min(_dot(proof.n_out,_sub(a,session.wall_surface_point_base_m)),_dot(proof.n_out,_sub(b,session.wall_surface_point_base_m)))-radius<float(policy["wall_margin_m"]): raise DynamicPrepareBlocked("WALL_HALFSPACE_REJECTED",item["name"])
                constraint_counts["wall"]+=1
                for proxy_key,code,count_key in (("base_proxy","BASE_BRACKET_PROXY_REJECTED","base"),("camera_bracket_proxy","BASE_BRACKET_PROXY_REJECTED","bracket")):
                    configured = policy["base_check_capsules" if proxy_key == "base_proxy" else "bracket_check_capsules"]
                    if item["name"] not in configured:
                        continue
                    proxy=policy[proxy_key]
                    primitive = str(proxy.get("primitive") or "sphere")
                    if primitive == "sphere":
                        distance = _point_segment_distance(proxy["center_m"], a, b)
                    elif primitive == "capsule":
                        distance = _segment_distance(a, b, proxy["endpoint_a_m"], proxy["endpoint_b_m"])
                    else:
                        raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", f"unsupported {proxy_key} primitive")
                    if distance < radius+float(proxy["radius_m"]): raise DynamicPrepareBlocked(code,proxy_key)
                    constraint_counts[count_key]+=1
            for left,right in policy["self_pairs"]:
                a0,a1,ar=capsules[left]; b0,b1,br=capsules[right]
                if _segment_distance(a0,a1,b0,b1)<ar+br: raise DynamicPrepareBlocked("SELF_PROXY_REJECTED",f"{left}/{right}")
                constraint_counts["self"]+=1
        cursor=waypoint
    wrist_roll_locked=bool(policy.get("wrist_roll_lock_enabled",False))
    locked_wrist_roll_rad=_quantize((session.q0_rad[4],),float(policy["encoder_step_rad"]))[0]
    if wrist_roll_locked and any(waypoint[4] != locked_wrist_roll_rad for waypoint in waypoints):
        raise DynamicPrepareBlocked("WRIST_ROLL_LOCK_FAILED", "quantized waypoint drift")
    payload={"schema":SCHEMA,"executor_boot_epoch":model.boot_epoch,"live_boot_epoch":session.live_boot_epoch,"qc_boot_epoch":session.qc_boot_epoch,"cart_pose_epoch":session.cart_pose_epoch,"session":asdict(session),"model_hashes":model.hashes,"start_state_admission":{"mode":str(policy["start_state_admission_mode"]),"q0_hash":session.q0_hash,"q0_stamp_ns":session.q0_stamp_ns},"normal_proof":asdict(proof),"p_button_base_m":session.p_button_base_m,"p_stage_base_m":p_stage,"p_pre_base_m":p_pre,"d_stage_m":d_stage,"d_pre_m":d_pre,"ik_audit":[asdict(x) for x in audits],"wrist_roll_lock":{"enabled":wrist_roll_locked,"joint_index":4,"joint_name":"link4_to_link5","source":"fresh measured q0 quantized to firmware encoder step","measured_q0_rad":session.q0_rad[4],"commanded_value_rad":locked_wrist_roll_rad,"exact":wrist_roll_locked},"effective_waypoint_partition":{"enabled":bool(effective_count),"mode":("JOINT_LINEAR_EFFECTIVE_TO_EXACT_DYNAMIC_ENDPOINT" if effective_count else "IK_AUDIT_SUBSEQUENCE"),"count":len(waypoints),"exact_endpoint_retained":True,"stage_anchor_retained_in_ik_audit":True,"stage_commanded_exactly":not bool(effective_count)},"post_quantization":{"position_residual_m":final_residual,"axis_residual_deg":final_axis,"waypoint_audit":quantized_audit},"waypoints_rad":waypoints,"sampled_constraints":{"code":SUCCESS_CODE,"counts":constraint_counts},"core_zero_resource_invariant":{"publisher_count":0,"lease_active":False,"arm_client_count":0,"service_call_count":0,"action_call_count":0,"runtime_instrumentation":False},"no_contact":True,"press":False,"retract":False,"return":False,"automatic_retry":False,"fixed_v2_fallback":False,"ttl_s":float(policy["ticket_ttl_s"]),"created_monotonic":float(now_monotonic)}
    if initial_disarmed_state != "CONVERGED_DISARMED":
        payload["initial_disarmed_state"] = initial_disarmed_state
        payload["start_state_admission"]["mode"] = (
            "FRESH_STATIONARY_OPERATOR_DISARMED_LEGAL_Q0"
            if initial_disarmed_state == "DISARMED:operator"
            else "FRESH_STATIONARY_RECOVERED_DISARMED_LEGAL_Q0"
        )
    plan_hash=canonical_hash(payload); return PreparedPlan(payload,plan_hash,float(now_monotonic)+float(policy["ticket_ttl_s"]))


def verify_frozen_plan(ticket: PreparedPlan, *, model: FrozenModel, session: FrozenSession, current_q0: Sequence[float], now_monotonic: float, consume: bool = False) -> PreparedPlan:
    if ticket.consumed: raise DynamicPrepareBlocked("PLAN_INVALIDATED","replay")
    if now_monotonic>=ticket.expires_monotonic: raise DynamicPrepareBlocked("PLAN_EXPIRED")
    if canonical_hash(ticket.plan)!=ticket.plan_hash: raise DynamicPrepareBlocked("PLAN_INVALIDATED","hash")
    if ticket.plan["executor_boot_epoch"]!=model.boot_epoch: raise DynamicPrepareBlocked("PLAN_INVALIDATED","executor boot identity")
    if canonical_hash(ticket.plan.get("model_hashes"))!=canonical_hash(model.hashes): raise DynamicPrepareBlocked("PLAN_INVALIDATED","model identity")
    if canonical_hash(ticket.plan.get("session"))!=canonical_hash(asdict(session)): raise DynamicPrepareBlocked("PLAN_INVALIDATED","exact frozen session identity")
    if max(abs(float(a)-float(b)) for a,b in zip(current_q0,session.q0_rad))>float(model.policy["q0_drift_limit_rad"]): raise DynamicPrepareBlocked("PLAN_INVALIDATED","q0 drift")
    return PreparedPlan(ticket.plan,ticket.plan_hash,ticket.expires_monotonic,consume)


__all__=["DynamicPrepareBlocked","FKState","FrozenModel","FrozenSession","IKAudit","NormalProof","PreparedPlan","SCHEMA","STOPPED_MODE","SUCCESS_CODE","WRIST_XY_DISABLED","canonical_hash","orient_outward_normal","prepare_dynamic_plan","verify_frozen_plan"]
