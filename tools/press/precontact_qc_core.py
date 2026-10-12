"""Pure, fail-closed lease protocol for a future pre-contact UI bridge.

This module deliberately contains no ROS, planning, serial, or motion code.
It is only the protocol boundary between Button-1 Live's high-level intent and
Quick Control's future service/action clients.  In particular, it cannot turn
a diagnostic target into an executable target and it has no retry transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping

from press.button_target import (
    get_semantic_track_id, normalize_button_label, validate_target_selection,
)
MAX_INTENT_AGE_S = 2.0
MAX_TICKET_TTL_S = 60.0
MAX_DYNAMIC_TERMINAL_FEEDBACK_AGE_S = 0.5
MAX_DYNAMIC_ENDPOINT_ERROR_RAD = math.radians(0.5)
# This is the released Stage-A post-quantization Cartesian residual gate.  QC
# rechecks the value embedded in the already-hashed frozen plan before it is
# reduced into the Stage-B arrival receipt.
MAX_DYNAMIC_POSITION_RESIDUAL_M = 0.0025
REQUIRED_CONFIRMATION_TEXT = "MOVE_TO_PRECONTACT_ONLY"
TARGET_SCHEMA = "m0_demo_freeze_v1"


class LeasePhase(str, Enum):
    IDLE = "IDLE"
    CART_PARKED = "CART_PARKED"
    TARGET_FROZEN = "TARGET_FROZEN"
    PREPARE_IN_FLIGHT = "PREPARE_IN_FLIGHT"
    AWAIT_MOVE_CONFIRM = "AWAIT_MOVE_CONFIRM"
    ACTION_IN_FLIGHT = "ACTION_IN_FLIGHT"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class LiveIntent:
    """A high-level request.  It is not a service or action goal."""

    intent_id: str
    action: str
    issued_at_unix_s: float
    canonical_target_snapshot_json: str
    target_snapshot_sha256: str
    prepare_intent_id: str = ""
    cart_pose_epoch: str = ""
    cart_confirmed_stamp_ns: int = 0


@dataclass(frozen=True)
class PrepareReceipt:
    """Only the fields Quick Control must retain after a successful Prepare."""

    accepted: bool
    authorization_id: str
    ttl_ms: int
    execution_release_manifest_sha256: str
    route_artifact_sha256: str
    m0_frozen_artifact_sha256: str
    reason: str = ""
    session_id: str = ""
    executor_instance_nonce: str = ""
    template_code: str = ""
    plan_hash: str = ""
    cart_pose_epoch: str = ""
    kinematic_model_hash: str = ""
    collision_proxy_hash: str = ""
    tool_model_hash: str = ""
    template_hash: str = ""
    calibration_hash: str = ""
    publisher_count: int = -1
    lease_active: bool = True
    arm_client_count: int = -1


@dataclass(frozen=True)
class ActionReceipt:
    """The exact-plan terminal result returned by the future Action server."""

    success: bool
    terminal_code: str
    plan_hash: str
    reason: str = ""
    session_id: str = ""
    endpoint_max_joint_error_rad: float = math.inf
    terminal_joint_target_publishers: int = -1
    terminal_precontact_lease_active: bool = True
    terminal_armed: bool = True
    terminal_feedback_age_s: float = math.inf
    teardown_complete: bool = False


@dataclass(frozen=True)
class LeaseDecision:
    accepted: bool
    command: str
    reason: str
    phase: LeasePhase


class DynamicSessionAssemblyError(ValueError):
    """Stable fail-closed source-join error raised before any service call."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(self.code + (f": {self.detail}" if self.detail else ""))


def canonical_json(value: Any) -> str:
    """The single canonical representation accepted at the UI boundary."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"canonical JSON failed: {exc}") from exc


def _sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", name)
    return value


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", name)
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", name) from exc
    if not all(math.isfinite(item) for item in result):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", name)
    return result


def _rotation_xyzw(value: Any) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = _vector(value, 4, "TF rotation")
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not 0.999 <= norm <= 1.001:
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "TF quaternion")
    x, y, z, w = (item / norm for item in (x, y, z, w))
    return (
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)),
    )


def _rotate(rotation: tuple[tuple[float, float, float], ...], value: tuple[float, ...]) -> tuple[float, float, float]:
    return tuple(sum(rotation[row][column] * value[column] for column in range(3)) for row in range(3))


def validate_q0_after_target(
    feedback: Mapping[str, Any],
    *,
    feedback_age_s: float,
    target_stamp_ns: int,
    now_unix_ns: int,
) -> tuple[tuple[float, ...], int]:
    """Validate one fresh, stationary, healthy q0 newer than the target."""

    try:
        age_s = float(feedback_age_s)
    except (TypeError, ValueError):
        age_s = math.nan
    if not isinstance(feedback, Mapping) or not math.isfinite(age_s) or not 0.0 <= age_s <= 1.0:
        raise DynamicSessionAssemblyError("TARGET_STALE", "q0 feedback")
    if feedback.get("armed") is not False or str(feedback.get("fault") or ""):
        raise DynamicSessionAssemblyError("CART_STOP_PROOF_MISSING", "q0 feedback state")
    if str(feedback.get("state") or "") != "CONVERGED_DISARMED":
        raise DynamicSessionAssemblyError(
            "CART_STOP_PROOF_MISSING", "q0 not stationary CONVERGED_DISARMED"
        )
    if feedback.get("dry_run") is not False:
        raise DynamicSessionAssemblyError("CART_STOP_PROOF_MISSING", "q0 dry-run state")
    if feedback.get("torque_enabled") != [True] * 6:
        raise DynamicSessionAssemblyError("CART_STOP_PROOF_MISSING", "q0 torque state")
    try:
        voltage = float(feedback.get("voltage"))
    except (TypeError, ValueError):
        voltage = math.nan
    if not math.isfinite(voltage) or not 11.0 <= voltage <= 13.0:
        raise DynamicSessionAssemblyError("CART_STOP_PROOF_MISSING", "q0 voltage state")
    q0 = _vector(feedback.get("positions_rad"), 6, "q0")
    q0_stamp_ns = int(feedback.get("feedback_stamp_ns") or 0)
    if q0_stamp_ns <= 0 or q0_stamp_ns < int(target_stamp_ns):
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "q0 timestamp")
    if int(now_unix_ns) < q0_stamp_ns or int(now_unix_ns) - q0_stamp_ns > 1_000_000_000:
        raise DynamicSessionAssemblyError("TARGET_STALE", "q0 timestamp")
    return q0, q0_stamp_ns


def q0_error_is_waitable_pre_target(
    error: Exception,
    feedback: Mapping[str, Any],
    *,
    target_stamp_ns: int,
) -> bool:
    """True only for one otherwise-valid q0 whose positive stamp is early."""

    if not isinstance(error, DynamicSessionAssemblyError):
        return False
    if error.code != "RGBD_TF_SESSION_MISMATCH" or error.detail != "q0 timestamp":
        return False
    try:
        q0_stamp_ns = int(feedback.get("feedback_stamp_ns") or 0)
    except (AttributeError, TypeError, ValueError):
        return False
    return 0 < q0_stamp_ns < int(target_stamp_ns)


def assemble_dynamic_session(
    intent: LiveIntent,
    *,
    qc_boot_epoch: str,
    cart_pose_epoch: str,
    cart_confirmed_stamp_ns: int,
    executor_status: Mapping[str, Any],
    feedback: Mapping[str, Any],
    feedback_age_s: float,
    now_unix_ns: int,
) -> tuple[str, str]:
    """Join only attributable frozen sources into one canonical session.

    This function is ROS-agnostic and has no I/O.  Callers must return before
    constructing/calling a service request if it raises.
    """

    target = _target_from_intent(intent)
    frozen = _required_mapping(target.get("frozen_target"), "frozen_target")
    if intent.cart_pose_epoch != cart_pose_epoch or str(frozen.get("cart_pose_epoch") or "") != cart_pose_epoch:
        raise DynamicSessionAssemblyError("CART_EPOCH_MISMATCH")
    if int(cart_confirmed_stamp_ns) <= 0:
        raise DynamicSessionAssemblyError("CART_STOP_PROOF_MISSING", "cart timestamp")
    evidence = _required_mapping(frozen.get("dynamic_evidence"), "dynamic_evidence")
    surface = _required_mapping(frozen.get("surface_geometry"), "surface_geometry")
    camera_point = _required_mapping(frozen.get("camera_point_m"), "camera_point_m")
    wall_point = _required_mapping(surface.get("surface_point_m"), "surface_point_m")
    normal_value = _required_mapping(surface.get("panel_outward_normal"), "panel_outward_normal")
    point_camera = _vector(tuple(camera_point.get(key) for key in ("x", "y", "z")), 3, "camera point")
    wall_camera = _vector(tuple(wall_point.get(key) for key in ("x", "y", "z")), 3, "wall point")
    normal_camera = _vector(tuple(normal_value.get(key) for key in ("x", "y", "z")), 3, "surface normal")

    tf_snapshot = _required_mapping(evidence.get("tf_snapshot"), "tf_snapshot")
    camera_tf = _required_mapping(tf_snapshot.get("base_from_source"), "base_from_source TF")
    hand_tf = _required_mapping(tf_snapshot.get("base_from_hand_tcp"), "base_from_hand_tcp TF")
    trusted_camera_record = _required_mapping(
        camera_tf.get("trusted_camera_record"), "trusted camera record"
    )
    trusted_camera_record_hash = _sha256(
        str(camera_tf.get("trusted_camera_record_hash") or "")
    )
    if _digest(trusted_camera_record) != trusted_camera_record_hash:
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "trusted camera record self-hash"
        )
    target_stamp_ns = int(frozen.get("source_stamp_ns") or 0)
    source_frame = str(frozen.get("frame_id") or "")
    if target_stamp_ns <= 0 or not source_frame:
        raise DynamicSessionAssemblyError("TARGET_STALE", "target header")
    if camera_tf.get("parent_frame") != "roarm_base_link" or camera_tf.get("child_frame") != source_frame:
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "camera TF frames")
    if hand_tf.get("parent_frame") != "roarm_base_link" or hand_tf.get("child_frame") != "hand_tcp":
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "hand TF frames")
    if (
        camera_tf.get("source_kind") != "accepted_calibration_static_record"
        or camera_tf.get("ros_tf_observed") is not False
        or camera_tf.get("static_transform") is not True
    ):
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "camera transform provenance"
        )
    if (
        hand_tf.get("source_kind") != "ros_tf_exact_timestamp"
        or hand_tf.get("ros_tf_observed") is not True
        or hand_tf.get("static_transform") is not False
    ):
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "hand TF provenance"
        )
    if (
        int(camera_tf.get("query_stamp_ns") or 0) != target_stamp_ns
        or int(camera_tf.get("bound_target_stamp_ns") or 0) != target_stamp_ns
        or int(hand_tf.get("query_stamp_ns") or 0) != target_stamp_ns
        or int(hand_tf.get("message_stamp_ns") or 0) != target_stamp_ns
    ):
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "TF query stamp")
    if int(camera_tf.get("message_stamp_ns", -1)) != 0:
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "static calibration timestamp"
        )
    rotation = _rotation_xyzw(camera_tf.get("rotation_xyzw"))
    translation = _vector(camera_tf.get("translation_m"), 3, "camera TF translation")
    trusted_translation = _vector(
        trusted_camera_record.get("translation_m"), 3, "trusted camera translation"
    )
    trusted_rotation = _vector(
        trusted_camera_record.get("rotation_xyzw"), 4, "trusted camera rotation"
    )
    if translation != trusted_translation or _vector(
        camera_tf.get("rotation_xyzw"), 4, "camera TF rotation"
    ) != trusted_rotation:
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "camera transform differs from trusted record"
        )
    point_base = tuple(a+b for a, b in zip(_rotate(rotation, point_camera), translation))
    wall_base = tuple(a+b for a, b in zip(_rotate(rotation, wall_camera), translation))
    normal_base = _rotate(rotation, normal_camera)
    hand_base = _vector(hand_tf.get("translation_m"), 3, "hand TCP reference")

    camera_info = _required_mapping(evidence.get("camera_info"), "camera_info")
    d2c = _required_mapping(evidence.get("d2c"), "d2c")
    rgbd = _required_mapping(evidence.get("rgbd"), "rgbd")
    calibration = _required_mapping(evidence.get("calibration"), "calibration")
    camera_info_hash = _sha256(str(camera_info.get("sha256") or ""))
    if not str(camera_info.get("identity") or ""):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "CameraInfo identity")
    if camera_info_hash != str(surface.get("camera_info_sha256") or ""):
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "CameraInfo hash")
    camera_info_stamp_ns = int(camera_info.get("stamp_ns") or 0)
    if camera_info_stamp_ns != int(surface.get("camera_info_stamp_ns") or 0):
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "CameraInfo stamp")
    if str(d2c.get("status") or "") != "verified":
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "D2C not verified")
    if not str(d2c.get("identity") or ""):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "D2C identity")
    if int(d2c.get("bound_target_stamp_ns") or 0) != target_stamp_ns:
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "D2C target binding stamp")
    required_rgbd_fields = (
        "source_frame_id", "canonical_frame_id", "raw_to_canonical_alias_id",
        "raw_to_canonical_alias_status", "raw_to_canonical_adapter_sha256",
        "camera_optical_axis_status", "depth_scale_status",
    )
    if any(not str(rgbd.get(field) or "") for field in required_rgbd_fields):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "RGB-D provenance")
    try:
        depth_scale = float(rgbd.get("depth_scale_m_per_unit"))
    except (TypeError, ValueError) as exc:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "RGB-D depth scale") from exc
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "RGB-D depth scale")
    if str(calibration.get("identity") or "") != str(surface.get("section_b_calibration_id") or ""):
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "calibration identity")
    calibration_hash = _sha256(str(calibration.get("sha256") or ""))
    calibration_path = str(calibration.get("path") or "")
    calibration_status = str(calibration.get("status") or "")
    calibration_validation_identity = str(calibration.get("validation_identity") or "")
    calibration_acceptance_artifact = str(calibration.get("acceptance_artifact") or "")
    calibration_acceptance_sha256 = _sha256(str(calibration.get("acceptance_sha256") or ""))
    calibration_camera_record_hash = _sha256(
        str(calibration.get("accepted_camera_record_hash") or "")
    )
    if not calibration_path or not calibration_validation_identity or not calibration_acceptance_artifact:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "calibration path")
    if calibration_status != "b_section_accepted_relative_unpublished":
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "calibration acceptance status")
    camera_calibration_bindings = (
        (camera_tf.get("transform_key"), "T_roarm_base_link_camera_optical_frame"),
        (camera_tf.get("calibration_identity"), calibration.get("identity")),
        (camera_tf.get("calibration_sha256"), calibration_hash),
        (camera_tf.get("calibration_path"), calibration_path),
        (camera_tf.get("calibration_status"), calibration_status),
        (camera_tf.get("calibration_validation_identity"), calibration_validation_identity),
        (camera_tf.get("acceptance_artifact"), calibration_acceptance_artifact),
        (camera_tf.get("acceptance_sha256"), calibration_acceptance_sha256),
        (calibration_camera_record_hash, trusted_camera_record_hash),
        (trusted_camera_record.get("parent_frame"), "roarm_base_link"),
        (trusted_camera_record.get("child_frame"), source_frame),
        (trusted_camera_record.get("transform_key"), "T_roarm_base_link_camera_optical_frame"),
        (trusted_camera_record.get("calibration_identity"), calibration.get("identity")),
        (trusted_camera_record.get("calibration_sha256"), calibration_hash),
        (trusted_camera_record.get("calibration_source_path_absolute"), calibration_path),
        (trusted_camera_record.get("calibration_status"), calibration_status),
        (trusted_camera_record.get("calibration_validation_identity"), calibration_validation_identity),
        (trusted_camera_record.get("acceptance_artifact"), calibration_acceptance_artifact),
        (trusted_camera_record.get("acceptance_sha256"), calibration_acceptance_sha256),
    )
    if any(str(observed or "") != str(expected or "") for observed, expected in camera_calibration_bindings):
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "camera calibration provenance"
        )

    status = _required_mapping(executor_status, "executor status")
    if status.get("schema") != "t102_dynamic_executor_status_v1" or status.get("model_loaded") is not True:
        raise DynamicSessionAssemblyError("EXECUTOR_NOT_READY", "boot model status")
    executor_boot = str(status.get("executor_boot_epoch") or "")
    if not executor_boot or not str(qc_boot_epoch):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "boot identity")
    model_fields = (
        "kinematic_model_hash", "collision_proxy_hash", "tool_model_hash",
        "template_hash", "calibration_hash", "fn_artifact_hash",
        "accepted_camera_record_hash",
        "fn_acceptance_hash", "standby_acceptance_hash", "source_registry_hash",
    )
    model_hashes = {field: _sha256(str(status.get(field) or "")) for field in model_fields}
    if model_hashes["calibration_hash"] != calibration_hash:
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "executor calibration hash")
    if model_hashes["accepted_camera_record_hash"] != trusted_camera_record_hash:
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "executor trusted camera record hash"
        )
    source_binding_fields = (
        "calibration_source_path_relative", "calibration_source_path_absolute",
        "calibration_validation_identity", "fn_pointer_path_relative",
        "fn_pointer_path_absolute", "fn_artifact_path_relative",
        "fn_artifact_path_absolute", "fn_acceptance_path_relative",
        "fn_acceptance_path_absolute", "fn_candidate_sidecar_path_relative",
        "fn_candidate_sidecar_path_absolute", "fn_validation_identity",
        "collision_proxy_path_relative", "collision_proxy_path_absolute",
        "collision_proxy_validation_identity", "standby_acceptance_path_relative",
        "standby_acceptance_path_absolute", "standby_validation_identity",
    )
    source_bindings = {field: str(status.get(field) or "") for field in source_binding_fields}
    if any(not value for value in source_bindings.values()):
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "executor Fn/calibration source binding")
    if source_bindings["calibration_validation_identity"] != calibration_validation_identity:
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "calibration validation identity"
        )
    if source_bindings["calibration_source_path_absolute"] != calibration_path:
        raise DynamicSessionAssemblyError(
            "RGBD_TF_SESSION_MISMATCH", "calibration observation/executor path"
        )

    q0, q0_stamp_ns = validate_q0_after_target(
        feedback,
        feedback_age_s=feedback_age_s,
        target_stamp_ns=target_stamp_ns,
        now_unix_ns=now_unix_ns,
    )
    if int(cart_confirmed_stamp_ns) > target_stamp_ns:
        raise DynamicSessionAssemblyError("RGBD_TF_SESSION_MISMATCH", "cart/target chronology")

    tf_hash = _digest(tf_snapshot)
    d2c_hash = _digest(d2c)
    rgbd_identity = _digest(rgbd)
    unsigned = {
        "live_boot_epoch": str(frozen.get("boot_epoch") or ""),
        "qc_boot_epoch": str(qc_boot_epoch),
        "executor_boot_epoch": executor_boot,
        "cart_pose_epoch": cart_pose_epoch,
        "cart_confirmed_stamp_ns": int(cart_confirmed_stamp_ns),
        "stopped_proof_mode": "OPERATOR_CONFIRMED_CART_PARKED_PROTOTYPE",
        "target_label": str(frozen.get("target_label") or target.get("target_label") or ""),
        "detection_id": str(frozen.get("detection_id") or ""),
        "confidence": float(frozen.get("confidence")),
        "target_stamp_ns": target_stamp_ns,
        "source_frame": source_frame,
        "canonical_frame": "roarm_base_link",
        "p_button_base_m": point_base,
        "wall_surface_point_base_m": wall_base,
        "candidate_normal_base": normal_base,
        "camera_center_base_m": translation,
        "robot_reference_base_m": hand_base,
        "rgbd_identity": rgbd_identity,
        "rgbd_stamp_ns": target_stamp_ns,
        "camera_info_identity": str(camera_info.get("identity") or ""),
        "camera_info_hash": camera_info_hash,
        "camera_info_stamp_ns": camera_info_stamp_ns,
        "d2c_identity": str(d2c.get("identity") or ""),
        "d2c_hash": d2c_hash,
        "d2c_stamp_ns": target_stamp_ns,
        "tf_hash": tf_hash,
        "tf_stamp_ns": target_stamp_ns,
        "calibration_identity": str(calibration.get("identity") or ""),
        "calibration_hash": calibration_hash,
        "calibration_observation_path": calibration_path,
        "calibration_observation_status": calibration_status,
        "calibration_observation_validation_identity": calibration_validation_identity,
        "calibration_acceptance_artifact": calibration_acceptance_artifact,
        "calibration_acceptance_sha256": calibration_acceptance_sha256,
        **source_bindings,
        **model_hashes,
        "q0_rad": q0,
        "q0_stamp_ns": q0_stamp_ns,
        "q0_hash": _digest(list(q0)),
    }
    selection = _validated_target_selection(frozen, target_label=unsigned["target_label"])
    if selection is not None:
        unsigned["target_selection"] = selection
    for field in ("live_boot_epoch", "detection_id", "camera_info_identity", "d2c_identity", "calibration_identity"):
        if not str(unsigned.get(field) or ""):
            raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", field)
    session = {"session_id": _digest(unsigned), **unsigned}
    text = canonical_json(session)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def dynamic_plan_summary(plan: Mapping[str, Any], plan_hash: str) -> Mapping[str, Any]:
    """Validate and reduce an exact executor plan for QC/Live display."""

    if _digest(plan) != _sha256(str(plan_hash)):
        raise DynamicSessionAssemblyError("PLAN_INVALIDATED", "summary hash")
    normal = _required_mapping(plan.get("normal_proof"), "normal proof")
    session = _required_mapping(plan.get("session"), "plan session")
    sampled = _required_mapping(plan.get("sampled_constraints"), "sampled constraints")
    post_quantization = _required_mapping(plan.get("post_quantization"), "post-quantization audit")
    audits = post_quantization.get("waypoint_audit")
    if not isinstance(audits, list) or not audits:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "post-quantization waypoint audit")
    waypoints = plan.get("waypoints_rad")
    if not isinstance(waypoints, list) or not waypoints:
        raise DynamicSessionAssemblyError("SOURCE_INCOMPLETE", "planned terminal waypoint")
    planned_terminal_q = _vector(waypoints[-1], 6, "planned terminal waypoint")
    try:
        planned_terminal_residual = float(post_quantization.get("position_residual_m"))
    except (TypeError, ValueError) as exc:
        raise DynamicSessionAssemblyError(
            "SOURCE_INCOMPLETE", "planned terminal position residual"
        ) from exc
    if (
        not math.isfinite(planned_terminal_residual)
        or planned_terminal_residual < 0.0
        or planned_terminal_residual > MAX_DYNAMIC_POSITION_RESIDUAL_M
    ):
        raise DynamicSessionAssemblyError(
            "PLAN_INVALIDATED", "planned terminal position residual"
        )
    conditions = [float(item["condition"]) for item in audits]
    manip = [float(item["manipulability"]) for item in audits]
    return {
        "schema": "t102_dynamic_frozen_plan_summary_v1",
        "normal_out_base": _vector(normal.get("n_out"), 3, "normal summary"),
        "p_button_base_m": _vector(plan.get("p_button_base_m"), 3, "button summary"),
        "p_stage_base_m": _vector(plan.get("p_stage_base_m"), 3, "stage summary"),
        "p_pre_base_m": _vector(plan.get("p_pre_base_m"), 3, "pre summary"),
        "planned_terminal_q_rad": planned_terminal_q,
        "planned_terminal_position_residual_m": planned_terminal_residual,
        "ik_condition_max": max(conditions),
        "ik_manipulability_min": min(manip),
        "sampled_constraint_counts": dict(_required_mapping(sampled.get("counts"), "constraint counts")),
        "kinematic_model_hash": str(session.get("kinematic_model_hash") or ""),
        "collision_proxy_hash": str(session.get("collision_proxy_hash") or ""),
        "tool_model_hash": str(session.get("tool_model_hash") or ""),
        "template_hash": str(session.get("template_hash") or ""),
        "calibration_hash": str(session.get("calibration_hash") or ""),
        "plan_hash": str(plan_hash),
    }


def _validated_target_selection(
    frozen: Mapping[str, Any], *, target_label: str,
) -> Mapping[str, Any] | None:
    label = normalize_button_label(target_label)
    if not label:
        raise ValueError("target snapshot label is unknown")
    if frozen.get("pressable") is False:
        raise ValueError("target snapshot is not pressable")
    selection = frozen.get("target_selection")
    if selection is None:
        if label != "1":
            raise ValueError("selected target binding is required for non-Button-1 targets")
        return None
    if not isinstance(selection, Mapping):
        raise ValueError("selected target binding must be an object")
    bound = {**selection, "pressable": frozen.get("pressable")}
    validate_target_selection(
        bound, target_label=label, source_stamp_ns=int(frozen.get("source_stamp_ns") or 0),
    )
    if get_semantic_track_id(frozen) != str(bound["semantic_track_id"]):
        raise ValueError("selected target semantic track mismatch")
    return bound


def _target_from_intent(intent: LiveIntent) -> Mapping[str, Any]:
    if not isinstance(intent.canonical_target_snapshot_json, str):
        raise ValueError("target snapshot is missing")
    try:
        payload = json.loads(intent.canonical_target_snapshot_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("target snapshot is not JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("target snapshot must be an object")
    if canonical_json(payload) != intent.canonical_target_snapshot_json:
        raise ValueError("target snapshot is not canonical JSON")
    supplied = _sha256(intent.target_snapshot_sha256)
    actual = hashlib.sha256(intent.canonical_target_snapshot_json.encode("utf-8")).hexdigest()
    if actual != supplied:
        raise ValueError("target snapshot SHA-256 mismatch")
    if payload.get("schema") != TARGET_SCHEMA:
        raise ValueError("target snapshot is not an m0_demo_freeze_v1 package")
    frozen = payload.get("frozen_target")
    if not isinstance(frozen, Mapping):
        raise ValueError("target snapshot missing frozen_target")
    label = str(frozen.get("target_label") or payload.get("target_label") or "")
    if payload.get("target_label") and normalize_button_label(payload["target_label"]) != normalize_button_label(label):
        raise ValueError("target snapshot top-level label mismatch")
    _validated_target_selection(frozen, target_label=label)
    if not str(frozen.get("detection_id") or "").strip():
        raise ValueError("target snapshot missing detection_id")
    if int(frozen.get("source_stamp_ns") or 0) <= 0:
        raise ValueError("target snapshot missing source stamp")
    if not str(frozen.get("frame_id") or "").strip():
        raise ValueError("target snapshot missing source frame")
    surface = frozen.get("surface_geometry")
    if not isinstance(surface, Mapping) or surface.get("estimate_valid") is not True:
        raise ValueError("target snapshot surface is invalid")
    if surface.get("provenance_complete") is not True:
        raise ValueError("target snapshot provenance is incomplete")
    if frozen.get("motion_authorized") is not False or payload.get("motion_authorized") is not False:
        raise ValueError("target snapshot authority flags are malformed")
    if not str(frozen.get("cart_pose_epoch") or payload.get("cart_pose_epoch") or "").strip():
        raise ValueError("target snapshot missing cart_pose_epoch")
    return payload


def target_stamp_ns_from_intent(intent: LiveIntent) -> int:
    """Return the exact validated frozen target timestamp without I/O."""

    target = _target_from_intent(intent)
    return int(target["frozen_target"]["source_stamp_ns"])


class PrecontactLeaseCore:
    """One prepare/confirm/action lease with no automatic retry transition."""

    def __init__(self) -> None:
        self.phase = LeasePhase.IDLE
        self._seen_intent_ids: set[str] = set()
        self._target_snapshot_sha256 = ""
        self._canonical_target_snapshot_json = ""
        self._prepare_intent_id = ""
        self._authorization_id = ""
        self._outbound_release_manifest_sha256 = ""
        self._execution_release_manifest_sha256 = ""
        self._route_artifact_sha256 = ""
        self._m0_frozen_artifact_sha256 = ""
        self._expires_monotonic = 0.0
        self._terminal_reason = ""
        self._session_id = ""
        self._executor_instance_nonce = ""
        self._cart_pose_epoch = ""
        self._cart_confirmed_stamp_ns = 0
        # An adopted orphan has only enough authority for typed discard.  It
        # deliberately shares AWAIT_MOVE_CONFIRM for callback binding, but it
        # can never consume a cart/confirm/validation intent.
        self._discard_only_adoption = False

    @property
    def authorization_id(self) -> str:
        """Only Quick Control reads this; it must never be sent back to Live."""

        return self._authorization_id

    @property
    def target_snapshot_sha256(self) -> str:
        return self._target_snapshot_sha256

    @property
    def canonical_target_snapshot_json(self) -> str:
        return self._canonical_target_snapshot_json

    @property
    def prepare_intent_id(self) -> str:
        return self._prepare_intent_id

    @property
    def terminal_reason(self) -> str:
        return self._terminal_reason

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def executor_instance_nonce(self) -> str:
        return self._executor_instance_nonce

    @property
    def route_artifact_sha256(self) -> str:
        """The exact prepared-plan identity retained by QC."""

        return self._route_artifact_sha256

    @property
    def cart_pose_epoch(self) -> str:
        """The exact operator-confirmed cart epoch bound to target and ticket."""

        return self._cart_pose_epoch

    @property
    def cart_confirmed_stamp_ns(self) -> int:
        return self._cart_confirmed_stamp_ns

    @property
    def discard_only(self) -> bool:
        return self._discard_only_adoption

    def adopt_inert_ticket_for_clear(
        self,
        *,
        executor_instance_nonce: str,
        plan_hash: str,
    ) -> LeaseDecision:
        """Adopt one already-prepared, resource-free executor ticket for discard.

        This transition is intentionally narrower than Prepare: it is only
        available from a freshly started local IDLE core, a TARGET_FROZEN core
        that has not called Prepare, or a TERMINAL core already pinned to the
        same executor/plan identity.  It records no authorization, lease, or
        motion capability.  The caller must independently prove inert executor
        status and immediately call the typed discard service.
        """

        nonce = str(executor_instance_nonce or "").strip()
        plan = str(plan_hash or "").strip()
        if self.phase not in {
            LeasePhase.IDLE, LeasePhase.TERMINAL, LeasePhase.TARGET_FROZEN,
        }:
            return LeaseDecision(False, "NONE", f"lease is {self.phase.value}", self.phase)
        if not nonce or not plan:
            return LeaseDecision(False, "NONE", "inert ticket identity is missing", self.phase)
        if self.phase is LeasePhase.TERMINAL and (
            self._executor_instance_nonce != nonce
            or self._route_artifact_sha256 != plan
        ):
            return LeaseDecision(
                False, "NONE", "terminal inert ticket identity mismatch", self.phase
            )
        self._executor_instance_nonce = nonce
        self._route_artifact_sha256 = plan
        self._expires_monotonic = math.inf
        self._terminal_reason = ""
        self._discard_only_adoption = True
        self.phase = LeasePhase.AWAIT_MOVE_CONFIRM
        return LeaseDecision(
            True,
            "DISCARD_PREPARED_PRECONTACT",
            "exact inert executor ticket adopted for discard",
            self.phase,
        )

    def _terminal(self, reason: str) -> LeaseDecision:
        self.phase = LeasePhase.TERMINAL
        self._terminal_reason = str(reason)
        return LeaseDecision(False, "NONE", self._terminal_reason, self.phase)

    def _validate_new_intent(self, intent: LiveIntent, *, now_unix_s: float) -> tuple[bool, str]:
        if not isinstance(intent.intent_id, str) or not intent.intent_id:
            return False, "intent_id is required"
        if intent.intent_id in self._seen_intent_ids:
            return False, "duplicate/replayed intent"
        self._seen_intent_ids.add(intent.intent_id)
        try:
            issued = float(intent.issued_at_unix_s)
        except (TypeError, ValueError):
            return False, "intent timestamp is invalid"
        if not math.isfinite(issued) or abs(float(now_unix_s) - issued) > MAX_INTENT_AGE_S:
            return False, "stale intent"
        return True, ""

    def begin_prepare(self, intent: LiveIntent, *, now_unix_s: float) -> LeaseDecision:
        """Validate the first click and request that QC call Prepare once."""

        valid, reason = self._validate_new_intent(intent, now_unix_s=now_unix_s)
        if not valid:
            return LeaseDecision(False, "NONE", reason, self.phase)
        if self.phase is not LeasePhase.TARGET_FROZEN:
            return LeaseDecision(False, "NONE", f"lease is {self.phase.value}", self.phase)
        if intent.action != "PREPARE_DYNAMIC_IK":
            return LeaseDecision(False, "NONE", "unexpected precontact intent action", self.phase)
        try:
            target = _target_from_intent(intent)
        except ValueError as exc:
            return LeaseDecision(False, "NONE", str(exc), self.phase)
        if intent.cart_pose_epoch != self._cart_pose_epoch:
            return LeaseDecision(False, "NONE", "prepare cart epoch mismatch", self.phase)
        if str(target["frozen_target"].get("cart_pose_epoch") or "") != self._cart_pose_epoch:
            return LeaseDecision(False, "NONE", "prepare target cart epoch mismatch", self.phase)
        if intent.target_snapshot_sha256 != self._target_snapshot_sha256:
            return LeaseDecision(False, "NONE", "prepare target snapshot hash mismatch", self.phase)
        if intent.canonical_target_snapshot_json != self._canonical_target_snapshot_json:
            return LeaseDecision(False, "NONE", "prepare canonical target snapshot mismatch", self.phase)
        self._prepare_intent_id = intent.intent_id
        self._m0_frozen_artifact_sha256 = self._target_snapshot_sha256
        self.phase = LeasePhase.PREPARE_IN_FLIGHT
        return LeaseDecision(True, "CALL_PREPARE", "prepare request accepted", self.phase)

    def confirm_cart_parked(self, intent: LiveIntent, *, now_unix_s: float) -> LeaseDecision:
        if self._discard_only_adoption:
            return LeaseDecision(False,"NONE","ticket is adopted for discard only",self.phase)
        valid, reason = self._validate_new_intent(intent, now_unix_s=now_unix_s)
        if not valid: return LeaseDecision(False,"NONE",reason,self.phase)
        if self.phase is LeasePhase.ACTION_IN_FLIGHT:
            return LeaseDecision(False,"NONE","cart epoch cannot change during action",self.phase)
        if intent.action!="CONFIRM_CART_PARKED" or not intent.cart_pose_epoch or int(intent.cart_confirmed_stamp_ns)<=0:
            return LeaseDecision(False,"NONE","invalid cart confirmation",self.phase)
        self._cart_pose_epoch=intent.cart_pose_epoch
        self._cart_confirmed_stamp_ns=int(intent.cart_confirmed_stamp_ns)
        self._target_snapshot_sha256=""; self._canonical_target_snapshot_json=""; self._authorization_id=""
        self._prepare_intent_id=""; self._m0_frozen_artifact_sha256=""; self._session_id=""
        self._executor_instance_nonce=""; self._route_artifact_sha256=""; self._expires_monotonic=0.0
        self._terminal_reason=""
        self.phase=LeasePhase.CART_PARKED
        return LeaseDecision(True,"AWAIT_TARGET_FREEZE","cart epoch accepted",self.phase)

    def freeze_button1_target(self, intent: LiveIntent, *, now_unix_s: float) -> LeaseDecision:
        if self.phase is not LeasePhase.CART_PARKED:
            return LeaseDecision(False,"NONE",f"lease is {self.phase.value}",self.phase)
        valid,reason=self._validate_new_intent(intent,now_unix_s=now_unix_s)
        if not valid: return LeaseDecision(False,"NONE",reason,self.phase)
        if intent.action!="FREEZE_BUTTON1_TARGET": return LeaseDecision(False,"NONE","unexpected freeze action",self.phase)
        try: target=_target_from_intent(intent)
        except ValueError as exc: return LeaseDecision(False,"NONE",str(exc),self.phase)
        if intent.cart_pose_epoch != self._cart_pose_epoch:
            return LeaseDecision(False,"NONE","freeze intent cart epoch mismatch",self.phase)
        if str(target["frozen_target"].get("cart_pose_epoch") or "")!=self._cart_pose_epoch:
            return LeaseDecision(False,"NONE","freeze cart epoch mismatch",self.phase)
        self._target_snapshot_sha256=intent.target_snapshot_sha256; self._canonical_target_snapshot_json=intent.canonical_target_snapshot_json
        self.phase=LeasePhase.TARGET_FROZEN
        return LeaseDecision(True,"AWAIT_PREPARE","fresh target frozen",self.phase)

    def record_prepare(self, receipt: PrepareReceipt, *, now_monotonic: float) -> LeaseDecision:
        """Accept a route ticket only when it matches the frozen target binding."""

        if self.phase is not LeasePhase.PREPARE_IN_FLIGHT:
            return self._terminal("prepare receipt arrived outside PREPARE_IN_FLIGHT")
        if not receipt.accepted:
            return self._terminal(f"prepare rejected: {receipt.reason}")
        try:
            _sha256(receipt.execution_release_manifest_sha256)
            _sha256(receipt.route_artifact_sha256)
            _sha256(receipt.m0_frozen_artifact_sha256)
        except ValueError as exc:
            return self._terminal(str(exc))
        try:
            plan_hash = _sha256(receipt.plan_hash or receipt.route_artifact_sha256)
            _sha256(receipt.kinematic_model_hash)
            _sha256(receipt.collision_proxy_hash)
            _sha256(receipt.tool_model_hash)
            _sha256(receipt.template_hash)
            _sha256(receipt.calibration_hash)
        except ValueError as exc:
            return self._terminal(f"dynamic Prepare identity invalid: {exc}")
        if receipt.publisher_count != 0 or receipt.lease_active or receipt.arm_client_count != 0:
            return self._terminal("dynamic Prepare used motion resources")
        if not isinstance(receipt.authorization_id, str) or not receipt.authorization_id:
            return self._terminal("prepare receipt has no authorization id")
        if receipt.m0_frozen_artifact_sha256 != self._m0_frozen_artifact_sha256:
            return self._terminal("prepare M0 frozen artifact mismatch")
        if receipt.cart_pose_epoch != self._cart_pose_epoch:
            return self._terminal("prepare receipt cart epoch mismatch")
        if not receipt.session_id or len(receipt.session_id) != 64:
            return self._terminal("prepare session_id is invalid")
        if not receipt.executor_instance_nonce:
            return self._terminal("prepare executor instance nonce is missing")
        if receipt.template_code != "SAMPLED_LIGHTWEIGHT_CONSTRAINT_PASS":
            return self._terminal(receipt.template_code or "DYNAMIC_PREPARE_REJECTED")
        self._execution_release_manifest_sha256 = receipt.execution_release_manifest_sha256
        ttl_s = min(MAX_TICKET_TTL_S, float(receipt.ttl_ms) / 1000.0)
        if not math.isfinite(ttl_s) or ttl_s <= 0.0:
            return self._terminal("prepare ticket TTL is invalid")
        self._authorization_id = receipt.authorization_id
        self._session_id = receipt.session_id
        self._executor_instance_nonce = receipt.executor_instance_nonce
        self._route_artifact_sha256 = receipt.route_artifact_sha256
        self._expires_monotonic = float(now_monotonic) + ttl_s
        self._route_artifact_sha256 = plan_hash
        self.phase = LeasePhase.AWAIT_MOVE_CONFIRM
        return LeaseDecision(True, "AWAIT_MOVE_CONFIRM", "confirm exact frozen plan once before ticket expiry", self.phase)

    def begin_confirm(
        self,
        intent: LiveIntent,
        *,
        now_unix_s: float,
        now_monotonic: float,
    ) -> LeaseDecision:
        """Validate the second click; only QC may use the retained ticket."""

        if self.phase is not LeasePhase.AWAIT_MOVE_CONFIRM:
            return LeaseDecision(False, "NONE", f"lease is {self.phase.value}", self.phase)
        if self._discard_only_adoption:
            return LeaseDecision(
                False, "NONE", "ticket is adopted for discard only", self.phase
            )
        if float(now_monotonic) >= self._expires_monotonic:
            return self._terminal("authorization ticket expired")
        valid, reason = self._validate_new_intent(intent, now_unix_s=now_unix_s)
        if not valid:
            return self._terminal(reason)
        if intent.action != "CONFIRM_PRECONTACT":
            return self._terminal("unexpected precontact confirmation action")
        if intent.prepare_intent_id != self._prepare_intent_id:
            return self._terminal("confirmation does not bind the prepared intent")
        if intent.target_snapshot_sha256 != self._target_snapshot_sha256:
            return self._terminal("confirmation target snapshot hash mismatch")
        self.phase = LeasePhase.ACTION_IN_FLIGHT
        return LeaseDecision(True, "SEND_ACTION_GOAL", "action may be sent once", self.phase)

    def begin_handoff_validation(
        self,
        intent: LiveIntent,
        *,
        now_unix_s: float,
        now_monotonic: float,
    ) -> LeaseDecision:
        """Consume one exact ticket for an action-transport check with no motion."""

        if self.phase is not LeasePhase.AWAIT_MOVE_CONFIRM:
            return LeaseDecision(False, "NONE", f"lease is {self.phase.value}", self.phase)
        if self._discard_only_adoption:
            return LeaseDecision(
                False, "NONE", "ticket is adopted for discard only", self.phase
            )
        if float(now_monotonic) >= self._expires_monotonic:
            return self._terminal("authorization ticket expired")
        valid, reason = self._validate_new_intent(intent, now_unix_s=now_unix_s)
        if not valid:
            return self._terminal(reason)
        if intent.action != "VALIDATE_PRECONTACT_HANDOFF_NO_MOTION":
            return self._terminal("unexpected precontact validation action")
        if intent.prepare_intent_id != self._prepare_intent_id:
            return self._terminal("validation does not bind the prepared intent")
        if intent.target_snapshot_sha256 != self._target_snapshot_sha256:
            return self._terminal("validation target snapshot hash mismatch")
        self.phase = LeasePhase.ACTION_IN_FLIGHT
        return LeaseDecision(
            True,
            "SEND_NO_MOTION_VALIDATION_GOAL",
            "exact frozen-plan handoff may be validated once",
            self.phase,
        )

    def record_handoff_validation(self, receipt: ActionReceipt) -> LeaseDecision:
        """Accept only an exact, resource-free executor validation receipt."""

        if self.phase is not LeasePhase.ACTION_IN_FLIGHT:
            return self._terminal("validation result arrived outside ACTION_IN_FLIGHT")
        if receipt.success is not True:
            return self._terminal(
                f"handoff validation failed: {receipt.terminal_code or receipt.reason}"
            )
        if receipt.session_id != self._session_id:
            return self._terminal("validation result session mismatch")
        if receipt.plan_hash != self._route_artifact_sha256:
            return self._terminal("validation result frozen plan hash mismatch")
        if receipt.terminal_code != "PRECONTACT_HANDOFF_VALIDATED_NO_MOTION":
            return self._terminal("unexpected handoff validation terminal code")
        if (
            receipt.terminal_joint_target_publishers != 0
            or receipt.terminal_precontact_lease_active
            or receipt.terminal_armed
            or not receipt.teardown_complete
        ):
            return self._terminal("handoff validation used or retained motion resources")
        self.phase = LeasePhase.TERMINAL
        self._terminal_reason = "precontact handoff validated; no motion executed"
        return LeaseDecision(True, "NONE", self._terminal_reason, self.phase)

    def record_action_result(self, receipt: ActionReceipt) -> LeaseDecision:
        """Terminal result is accepted only if every pinned identity matches."""

        if self.phase is not LeasePhase.ACTION_IN_FLIGHT:
            return self._terminal("action result arrived outside ACTION_IN_FLIGHT")
        if receipt.success is not True:
            return self._terminal(f"action failed: {receipt.terminal_code or receipt.reason}")
        if receipt.session_id != self._session_id:
            return self._terminal("action result session mismatch")
        if receipt.plan_hash != self._route_artifact_sha256:
            return self._terminal("action result frozen plan hash mismatch")
        if receipt.terminal_code != "DYNAMIC_PRECONTACT_STOPPED_DISARMED":
            return self._terminal("action result is not an accepted exact-plan terminal")
        if (
            receipt.terminal_joint_target_publishers != 0
            or receipt.terminal_precontact_lease_active
            or receipt.terminal_armed
            or not receipt.teardown_complete
        ):
            return self._terminal("action result lacks terminal resource teardown proof")
        if not math.isfinite(receipt.terminal_feedback_age_s) or not 0.0 <= receipt.terminal_feedback_age_s <= MAX_DYNAMIC_TERMINAL_FEEDBACK_AGE_S:
            return self._terminal("action result terminal feedback is stale")
        if not math.isfinite(receipt.endpoint_max_joint_error_rad) or not 0.0 <= receipt.endpoint_max_joint_error_rad <= MAX_DYNAMIC_ENDPOINT_ERROR_RAD:
            return self._terminal("action endpoint error exceeds the frozen dynamic-plan terminal limit")
        self.phase = LeasePhase.TERMINAL
        self._terminal_reason = "exact frozen plan completed; no automatic retry"
        return LeaseDecision(True, "NONE", self._terminal_reason, self.phase)

    def expire(self, *, now_monotonic: float) -> LeaseDecision | None:
        if self.phase is LeasePhase.AWAIT_MOVE_CONFIRM and float(now_monotonic) >= self._expires_monotonic:
            return self._terminal("authorization ticket expired")
        return None

    def ticket_remaining_s(self, *, now_monotonic: float) -> float:
        if self.phase is not LeasePhase.AWAIT_MOVE_CONFIRM:
            return 0.0
        return max(0.0, self._expires_monotonic - float(now_monotonic))

    def fail_local(self, reason: str) -> LeaseDecision:
        """Latch an async client/server failure without offering a retry."""
        return self._terminal(str(reason))
