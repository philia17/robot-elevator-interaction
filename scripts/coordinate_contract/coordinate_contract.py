"""Pure-Python, ROS-independent coordinate-contract validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONTRACT_VERSION = "1.0.0"
EXPECTED_FRAMES = {
    "map": "body",
    "odom": "body",
    "cart_base": "body",
    "camera_link": "body",
    "camera_optical_frame": "optical",
    "roarm_base_link": "body",
    "base_link": "vendor_kinematic",
    "hand_tcp": "vendor_kinematic",
    "press_contact": "body",
    "panel_frame": "body",
    "button_target": "body",
}
EXPECTED_EDGES = {
    "map_to_odom": ("map", "odom", "dynamic"),
    "odom_to_cart_base": ("odom", "cart_base", "dynamic"),
    "cart_base_to_camera_link": ("cart_base", "camera_link", "static"),
    "camera_link_to_camera_optical_frame": (
        "camera_link", "camera_optical_frame", "static"
    ),
    "cart_base_to_roarm_base_link": (
        "cart_base", "roarm_base_link", "static"
    ),
    "roarm_base_link_to_vendor_base_link": (
        "roarm_base_link", "base_link", "static"
    ),
    "vendor_base_link_to_hand_tcp": (
        "base_link", "hand_tcp", "dynamic_kinematic_chain"
    ),
    "hand_tcp_to_press_contact": ("hand_tcp", "press_contact", "static"),
    "camera_optical_frame_to_panel_frame": (
        "camera_optical_frame", "panel_frame", "runtime_observation"
    ),
    "panel_frame_to_button_target": (
        "panel_frame", "button_target", "runtime_observation"
    ),
}
REQUIRED_STATIC_CALIBRATIONS = {
    "cart_base_to_camera_link",
    "cart_base_to_roarm_base_link",
    "hand_tcp_to_press_contact",
}
EXPECTED_COMPONENTS = {
    "hp60c_intrinsics",
    "hp60c_rgb_depth_registration",
    "hp60c_depth_scale",
    "hp60c_optical_axes",
    "eye_to_hand",
    "roarm_mount",
    "stock_contact_tcp",
}
EXPECTED_TARGET_FIELDS = [
    "schema_version",
    "coordinate_contract_version",
    "target_id",
    "detection_id",
    "pose.header.frame_id",
    "pose.header.stamp_ns",
    "pose.position_m",
    "pose.orientation_xyzw",
    "panel_outward_normal.frame_id",
    "panel_outward_normal.xyz",
    "source_timestamp_ns",
    "confidence",
    "calibration_id",
    "ttl_ms",
]


class ContractError(ValueError):
    """Raised when coordinate data is unsafe or ambiguous."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ContractError(f"{where}.{key} is required")
    return mapping[key]


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        raise ContractError(f"{where} must be a finite number")
    return result


def _vector(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{where} must be a {length}-element sequence")
    if len(value) != length:
        raise ContractError(f"{where} must contain exactly {length} values")
    return tuple(_finite(item, f"{where}[{index}]") for index, item in enumerate(value))


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def quaternion_to_matrix(value: Any, where: str = "quaternion") -> tuple[tuple[float, ...], ...]:
    x, y, z, w = _vector(value, 4, where)
    norm = _norm((x, y, z, w))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError(f"{where} must be unit length; got norm={norm}")
    matrix = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError(f"{where} is not a proper rotation; det={determinant}")
    return matrix


def load_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"invalid YAML: {exc}") from exc
    root = _mapping(loaded, "document")
    contract = dict(_mapping(_required(root, "coordinate_contract", "document"), "coordinate_contract"))
    validate_contract(contract)
    return contract


def _validate_axes(contract: Mapping[str, Any]) -> None:
    axes = _mapping(_required(contract, "axes", "coordinate_contract"), "axes")
    body = _mapping(_required(axes, "body", "axes"), "axes.body")
    optical = _mapping(_required(axes, "optical", "axes"), "axes.optical")
    if body != {"handedness": "right", "x": "forward", "y": "left", "z": "up"}:
        raise ContractError("axes.body must be REP-103 X-forward/Y-left/Z-up")
    if optical != {"handedness": "right", "x": "right", "y": "down", "z": "forward"}:
        raise ContractError("axes.optical must be REP-103 X-right/Y-down/Z-forward")


def _validate_units(contract: Mapping[str, Any]) -> None:
    units = _mapping(_required(contract, "units", "coordinate_contract"), "units")
    expected = {
        "external_length": "m",
        "external_angle": "rad",
        "orientation": "quaternion_xyzw",
        "image_coordinate": "px",
        "time": "ns",
        "confidence": "unit_interval",
        "raw_depth": "device_unit_requires_explicit_scale_to_m",
        "robodk_adapter_length": "mm",
    }
    for key, value in expected.items():
        if units.get(key) != value:
            raise ContractError(f"units.{key} must be {value!r}")


def _validate_frames_and_edges(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    frames = _mapping(_required(contract, "frames", "coordinate_contract"), "frames")
    if set(frames) != set(EXPECTED_FRAMES):
        raise ContractError("frames must exactly match the frozen v1 frame set")
    for name, frame_class in EXPECTED_FRAMES.items():
        frame = _mapping(frames[name], f"frames.{name}")
        if frame.get("class") != frame_class:
            raise ContractError(f"frames.{name}.class must be {frame_class!r}")
    if frames["press_contact"].get("fingertip_installed") is not False:
        raise ContractError("press_contact must represent the current no-fingertip setup")
    if frames["press_contact"].get("contact_surface") != "stock_end_effector":
        raise ContractError("press_contact must use the stock end-effector contact surface")

    raw_transforms = _required(contract, "transforms", "coordinate_contract")
    if not isinstance(raw_transforms, list):
        raise ContractError("transforms must be a list")
    transforms: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(raw_transforms):
        item = _mapping(value, f"transforms[{index}]")
        transform_id = str(_required(item, "id", f"transforms[{index}]"))
        if transform_id in transforms:
            raise ContractError(f"duplicate transform id {transform_id!r}")
        transforms[transform_id] = item
        _required(item, "child", transform_id)
        _required(item, "parent", transform_id)
    if set(transforms) != set(EXPECTED_EDGES):
        raise ContractError("transforms must exactly match the frozen v1 edge set")
    for transform_id, (parent, child, mode) in EXPECTED_EDGES.items():
        item = transforms[transform_id]
        actual = (item.get("parent"), item.get("child"), item.get("mode"))
        if actual != (parent, child, mode):
            raise ContractError(
                f"{transform_id} direction/mode must be {parent}->{child} ({mode})"
            )

    parents: dict[str, str] = {}
    for transform_id, item in transforms.items():
        child = str(item["child"])
        parent = str(item["parent"])
        if child in parents:
            raise ContractError(f"frame {child!r} has multiple parents")
        parents[child] = parent

    for frame in EXPECTED_FRAMES:
        visited: set[str] = set()
        cursor = frame
        while cursor in parents:
            if cursor in visited:
                raise ContractError(f"TF graph contains a cycle at {cursor!r}")
            visited.add(cursor)
            cursor = parents[cursor]

    optical = transforms["camera_link_to_camera_optical_frame"]
    if _vector(optical.get("translation_m"), 3, "optical.translation_m") != (0.0, 0.0, 0.0):
        raise ContractError("body-to-optical translation must be zero")
    expected_q = (-0.5, 0.5, -0.5, 0.5)
    if _vector(optical.get("rotation_xyzw"), 4, "optical.rotation_xyzw") != expected_q:
        raise ContractError("body-to-optical quaternion is not the frozen REP-103 rotation")
    matrix = quaternion_to_matrix(expected_q, "optical.rotation_xyzw")
    expected_matrix = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    for actual_row, expected_row in zip(matrix, expected_matrix):
        for actual_value, expected_value in zip(actual_row, expected_row):
            if not math.isclose(actual_value, expected_value, abs_tol=1e-9):
                raise ContractError("body-to-optical rotation does not map REP-103 axes")

    for transform_id, item in transforms.items():
        if item.get("mode") != "static":
            continue
        status = item.get("status")
        if status == "pending":
            if item.get("calibration_id") is not None:
                raise ContractError(f"pending {transform_id} calibration_id must be null")
            if item.get("translation_m") is not None or item.get("rotation_xyzw") is not None:
                raise ContractError(f"pending {transform_id} must use null, not placeholder values")
        elif status in {"calibrated", "conventional"}:
            _vector(item.get("translation_m"), 3, f"{transform_id}.translation_m")
            quaternion_to_matrix(item.get("rotation_xyzw"), f"{transform_id}.rotation_xyzw")
            if status == "calibrated" and not str(item.get("calibration_id") or "").strip():
                raise ContractError(f"calibrated {transform_id} needs calibration_id")
        else:
            raise ContractError(f"static {transform_id} has invalid status {status!r}")
    return transforms


def _validate_camera_adapter(contract: Mapping[str, Any]) -> None:
    adapters = _mapping(_required(contract, "sensor_adapters", "coordinate_contract"), "sensor_adapters")
    gemini = _mapping(_required(adapters, "gemini335", "sensor_adapters"), "sensor_adapters.gemini335")
    if gemini.get("observed_raw_frame_id") != "gemini335_color_optical_frame":
        raise ContractError("Gemini 335 observed raw frame changed; re-audit before updating")
    if gemini.get("canonical_frame_id") != "camera_optical_frame":
        raise ContractError("Gemini 335 normalized frame must be camera_optical_frame")
    expected_canonical_topics = {
        "rgb_image": "/lift_robot/camera/color/image_raw",
        "rgb_camera_info": "/lift_robot/camera/color/camera_info",
        "aligned_depth_image": "/lift_robot/camera/aligned_depth/image_raw",
        "aligned_depth_camera_info": "/lift_robot/camera/aligned_depth/camera_info",
    }
    if gemini.get("canonical_topics") != expected_canonical_topics:
        raise ContractError("Gemini 335 canonical topics do not match the adapter boundary")
    if gemini.get("vendor_tf_allowed_for_execution") is not False:
        raise ContractError("vendor Gemini 335 TF must be forbidden for execution")
    gemini_depth_status = gemini.get("depth_scale_status")
    gemini_depth_scale = gemini.get("depth_scale_m_per_unit")
    if gemini_depth_status == "pending":
        if gemini_depth_scale is not None:
            raise ContractError("pending Gemini 335 depth scale must be null")
    elif gemini_depth_status == "calibrated":
        if _finite(gemini_depth_scale, "gemini335.depth_scale_m_per_unit") <= 0:
            raise ContractError("Gemini 335 depth scale must be positive")
    else:
        raise ContractError("Gemini 335 depth_scale_status must be pending or calibrated")
    if gemini.get("optical_axis_validation_status") not in {"pending", "calibrated"}:
        raise ContractError("Gemini 335 optical_axis_validation_status is invalid")

    hp60c = _mapping(_required(adapters, "hp60c", "sensor_adapters"), "sensor_adapters.hp60c")
    if hp60c.get("observed_raw_frame_id") != "ascamera_hp60c_color_0":
        raise ContractError("HP60C observed raw frame changed; re-audit before updating")
    if hp60c.get("canonical_frame_id") != "camera_optical_frame":
        raise ContractError("HP60C normalized frame must be camera_optical_frame")
    if hp60c.get("vendor_tf_allowed_for_execution") is not False:
        raise ContractError("vendor HP60C TF must be forbidden for execution")
    depth_status = hp60c.get("depth_scale_status")
    depth_scale = hp60c.get("depth_scale_m_per_unit")
    if depth_status == "pending":
        if depth_scale is not None:
            raise ContractError("pending depth scale must be null")
    elif depth_status == "calibrated":
        if _finite(depth_scale, "depth_scale_m_per_unit") <= 0:
            raise ContractError("depth scale must be positive")
    else:
        raise ContractError("HP60C depth_scale_status must be pending or calibrated")
    if hp60c.get("optical_axis_validation_status") not in {"pending", "calibrated"}:
        raise ContractError("HP60C optical_axis_validation_status is invalid")


def validate_contract(contract: Mapping[str, Any]) -> bool:
    if contract.get("schema_version") != 1:
        raise ContractError("coordinate_contract.schema_version must be 1")
    if contract.get("version") != CONTRACT_VERSION:
        raise ContractError(f"coordinate_contract.version must be {CONTRACT_VERSION}")
    if contract.get("status") != "frozen":
        raise ContractError("coordinate_contract.status must be frozen")
    notation = _mapping(_required(contract, "notation", "coordinate_contract"), "notation")
    if notation.get("transform") != "T_parent_child":
        raise ContractError("transform notation must be T_parent_child")
    _validate_units(contract)
    _validate_axes(contract)
    transforms = _validate_frames_and_edges(contract)
    _validate_camera_adapter(contract)

    calibration = _mapping(_required(contract, "calibration", "coordinate_contract"), "calibration")
    if set(calibration.get("required_static_transform_ids", [])) != REQUIRED_STATIC_CALIBRATIONS:
        raise ContractError("required static calibration set does not match v1")
    components = _mapping(
        _required(calibration, "required_component_status", "calibration"),
        "calibration.required_component_status",
    )
    if set(components) != EXPECTED_COMPONENTS:
        raise ContractError("required calibration components do not match v1")
    if any(value not in {"pending", "calibrated"} for value in components.values()):
        raise ContractError("calibration component status must be pending or calibrated")

    interface = _mapping(_required(contract, "target_interface", "coordinate_contract"), "target_interface")
    if interface.get("allowed_pose_frames") != ["camera_optical_frame", "panel_frame"]:
        raise ContractError("target allowed_pose_frames must be camera_optical_frame/panel_frame")
    if interface.get("topic") != "/elevator/button_target":
        raise ContractError("target topic changed without a contract version change")
    if interface.get("required_fields") != EXPECTED_TARGET_FIELDS:
        raise ContractError("target required_fields changed without a contract version change")

    enabled = calibration.get("execution_enabled")
    if not isinstance(enabled, bool):
        raise ContractError("calibration.execution_enabled must be boolean")
    calibration_id = str(calibration.get("calibration_id") or "").strip()
    ready = enabled
    if enabled and not calibration_id:
        raise ContractError("execution-enabled calibration needs a calibration_id")
    for transform_id in REQUIRED_STATIC_CALIBRATIONS:
        item = transforms[transform_id]
        transform_ready = (
            item.get("status") == "calibrated"
            and str(item.get("calibration_id") or "").strip() == calibration_id
        )
        ready = ready and transform_ready
    ready = ready and all(value == "calibrated" for value in components.values())
    hp60c = contract["sensor_adapters"]["hp60c"]
    ready = ready and hp60c.get("depth_scale_status") == "calibrated"
    ready = ready and hp60c.get("optical_axis_validation_status") == "calibrated"
    if enabled and not ready:
        raise ContractError("execution_enabled=true but calibration bundle is incomplete or mismatched")
    return bool(ready)


def _reject_ambiguous_target_units(value: Any, where: str = "target") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered.endswith("_mm") or lowered.endswith("_deg") or lowered.endswith("_degrees"):
                raise ContractError(f"{where}.{key} uses a forbidden external unit")
            _reject_ambiguous_target_units(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_ambiguous_target_units(child, f"{where}[{index}]")


def validate_target(
    target: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    now_ns: int | None = None,
    require_execution_ready: bool = True,
) -> None:
    target = _mapping(target, "target")
    _reject_ambiguous_target_units(target)
    ready = validate_contract(contract)
    if require_execution_ready and not ready:
        raise ContractError("coordinate calibration is not execution-ready")
    if target.get("schema_version") != 1:
        raise ContractError("target.schema_version must be 1")
    if target.get("coordinate_contract_version") != CONTRACT_VERSION:
        raise ContractError("target coordinate_contract_version mismatch")
    for key in ("target_id", "detection_id"):
        if not str(target.get(key) or "").strip():
            raise ContractError(f"target.{key} cannot be empty")

    pose = _mapping(_required(target, "pose", "target"), "target.pose")
    header = _mapping(_required(pose, "header", "target.pose"), "target.pose.header")
    frame_id = str(_required(header, "frame_id", "target.pose.header"))
    interface = contract["target_interface"]
    if frame_id not in interface["allowed_pose_frames"]:
        raise ContractError(f"target pose frame {frame_id!r} is not allowed")
    stamp_ns = _required(header, "stamp_ns", "target.pose.header")
    if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns <= 0:
        raise ContractError("target pose stamp_ns must be a positive integer")
    position = _vector(_required(pose, "position_m", "target.pose"), 3, "target.pose.position_m")
    if any(abs(value) > 100.0 for value in position):
        raise ContractError("target position is implausible in metres")
    orientation_matrix = quaternion_to_matrix(
        _required(pose, "orientation_xyzw", "target.pose"),
        "target.pose.orientation_xyzw",
    )

    normal = _mapping(
        _required(target, "panel_outward_normal", "target"),
        "target.panel_outward_normal",
    )
    if normal.get("frame_id") != frame_id:
        raise ContractError("panel normal and target pose must use the same frame")
    normal_xyz = _vector(
        _required(normal, "xyz", "target.panel_outward_normal"),
        3,
        "target.panel_outward_normal.xyz",
    )
    if not math.isclose(_norm(normal_xyz), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ContractError("panel outward normal must be unit length")
    pose_x_axis = tuple(orientation_matrix[row][0] for row in range(3))
    if any(
        not math.isclose(axis, normal, rel_tol=0.0, abs_tol=1e-6)
        for axis, normal in zip(pose_x_axis, normal_xyz)
    ):
        raise ContractError("target pose +X axis must align with panel outward normal")

    source_timestamp_ns = target.get("source_timestamp_ns")
    if source_timestamp_ns != stamp_ns:
        raise ContractError("source_timestamp_ns must equal pose.header.stamp_ns")
    confidence = _finite(target.get("confidence"), "target.confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ContractError("target confidence must be in [0,1]")
    if confidence < float(interface["minimum_confidence"]):
        raise ContractError("target confidence is below the frozen minimum")

    target_calibration_id = str(target.get("calibration_id") or "").strip()
    active_calibration_id = str(contract["calibration"].get("calibration_id") or "").strip()
    if not target_calibration_id or target_calibration_id != active_calibration_id:
        raise ContractError("target calibration_id is empty, unknown, or stale")
    ttl_ms = target.get("ttl_ms")
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
        raise ContractError("target.ttl_ms must be a positive integer")
    if ttl_ms > int(interface["maximum_ttl_ms"]):
        raise ContractError("target.ttl_ms exceeds the frozen maximum")
    if now_ns is not None:
        if now_ns < stamp_ns:
            raise ContractError("target timestamp is in the future")
        if now_ns - stamp_ns > ttl_ms * 1_000_000:
            raise ContractError("target is stale")
