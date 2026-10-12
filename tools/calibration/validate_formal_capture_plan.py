#!/usr/bin/env python3
"""Static validator for the frozen B4 freeform capture design."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTED_IDS = [f"S{i:04d}" for i in range(12, 42)]
EXPECTED_HELD_OUT_ORDERS = [4, 8, 12, 16, 20, 24, 27, 30]
SOURCE_HASH_PATHS = {
    "calibration_core.py": (
        REPOSITORY
        / "ros2_arm/roarm_safe_driver/roarm_safe_driver/calibration_core.py"
    ),
    "calibration_recorder.py": (
        REPOSITORY
        / "ros2_arm/roarm_safe_driver/roarm_safe_driver/calibration_recorder.py"
    ),
    "safe_driver_core.py": (
        REPOSITORY / "ros2_arm/roarm_safe_driver/roarm_safe_driver/core.py"
    ),
    "safe_driver_node.py": (
        REPOSITORY / "ros2_arm/roarm_safe_driver/roarm_safe_driver/node.py"
    ),
    "safe_driver_transport.py": (
        REPOSITORY / "ros2_arm/roarm_safe_driver/roarm_safe_driver/transport.py"
    ),
    "torque_transaction.py": (
        REPOSITORY
        / "ros2_arm/roarm_safe_driver/roarm_safe_driver/torque_transaction.py"
    ),
    "tf_bridge.py": (
        REPOSITORY / "ros2_arm/roarm_safe_driver/roarm_safe_driver/tf_bridge.py"
    ),
    "phase4_tf_bridge.launch.py": (
        REPOSITORY / "ros2_arm/roarm_safe_driver/launch/phase4_tf_bridge.launch.py"
    ),
    "roarm_m3.xacro": (
        REPOSITORY
        / "RoArm-M3/ros2/roarm_ws/src/roarm_main/roarm_description/"
        "urdf/roarm_m3/roarm_m3.xacro"
    ),
    "apriltag_calibration_live_view.py": (
        REPOSITORY / "tools/apriltag_calibration_live_view.py"
    ),
    "live_capture_core.py": REPOSITORY / "tools/calibration/live_capture_core.py",
    "live_torque_core.py": REPOSITORY / "tools/calibration/live_torque_core.py",
    "validate_formal_capture_plan.py": (
        REPOSITORY / "tools/calibration/validate_formal_capture_plan.py"
    ),
    "apriltag_gemini335.yaml": (
        REPOSITORY / "config/calibration/apriltag_gemini335.yaml"
    ),
    "start_gemini335_apriltag.sh": (
        REPOSITORY / "tools/start_gemini335_apriltag.sh"
    ),
    "gemini335_canonical_adapter_manifest.yaml": (
        REPOSITORY
        / "ros2_perception/ros2_ws/src/lift_robot_camera_adapter/config/"
        "gemini335_canonical_adapter_manifest.yaml"
    ),
    "gemini335_canonical_cpp.yaml": (
        REPOSITORY
        / "ros2_perception/ros2_ws/src/lift_robot_camera_adapter_cpp/config/"
        "gemini335_canonical_cpp.yaml"
    ),
    "gemini335_canonical_adapter.cpp": (
        REPOSITORY
        / "ros2_perception/ros2_ws/src/lift_robot_camera_adapter_cpp/src/"
        "gemini335_canonical_adapter.cpp"
    ),
    "gemini335_canonical_components.launch.py": (
        REPOSITORY
        / "ros2_perception/ros2_ws/src/lift_robot_camera_adapter_cpp/launch/"
        "gemini335_canonical_components.launch.py"
    ),
    "coordinate_contract_v1.yaml": (
        REPOSITORY / "config/coordinate_contract_v1.yaml"
    ),
    "agx_unified_safe.yaml": (
        REPOSITORY / "ros2_arm/roarm_safe_driver/config/agx_unified_safe.yaml"
    ),
    "eye_to_hand_pilot_solver.py": (
        REPOSITORY / "tools/calibration/eye_to_hand_pilot_solver.py"
    ),
    "eye_to_hand_formal_solver.py": (
        REPOSITORY / "tools/calibration/eye_to_hand_formal_solver.py"
    ),
}
AGX_SOURCE_FALLBACKS = {
    "calibration_core.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/calibration_core.py"
    ),
    "calibration_recorder.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/calibration_recorder.py"
    ),
    "safe_driver_core.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/core.py"
    ),
    "safe_driver_node.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/node.py"
    ),
    "safe_driver_transport.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/transport.py"
    ),
    "torque_transaction.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/torque_transaction.py"
    ),
    "tf_bridge.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "roarm_safe_driver/tf_bridge.py"
    ),
    "phase4_tf_bridge.launch.py": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "launch/phase4_tf_bridge.launch.py"
    ),
    "roarm_m3.xacro": Path(
        "/home/robot/roarm_ws/src/roarm_main/roarm_description/"
        "urdf/roarm_m3/roarm_m3.xacro"
    ),
    "agx_unified_safe.yaml": Path(
        "/home/robot/roarm_ws/src/lift_robot/roarm_safe_driver/"
        "config/agx_unified_safe.yaml"
    ),
    "gemini335_canonical_cpp.yaml": Path(
        "/data/ros_ws/lift_robot_calibration_ws/src/"
        "lift_robot_camera_adapter_cpp/config/gemini335_canonical_cpp.yaml"
    ),
    "gemini335_canonical_adapter.cpp": Path(
        "/data/ros_ws/lift_robot_calibration_ws/src/"
        "lift_robot_camera_adapter_cpp/src/gemini335_canonical_adapter.cpp"
    ),
    "gemini335_canonical_components.launch.py": Path(
        "/data/ros_ws/lift_robot_calibration_ws/src/"
        "lift_robot_camera_adapter_cpp/launch/"
        "gemini335_canonical_components.launch.py"
    ),
}
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "purpose",
    "plan_status",
    "pose_mode",
    "dataset_id",
    "dataset_dir",
    "ros_domain_id",
    "status_source",
    "capture_enabled",
    "user_approved",
    "formal_capture_prerequisites_frozen",
    "motion_authorized",
    "execution_ready",
    "operator_contract",
    "split_policy",
    "measurement_contract",
    "b5_preregistration",
    "source_hashes_sha256",
    "capture_order",
}
PROHIBITED_TARGET_KEYS = {
    "image_target",
    "orientation_target",
    "joint_target",
    "joint_targets",
    "joint_target_deg",
    "joint_target_rad",
    "command",
    "command_deg",
    "command_rad",
}


def fail(message: str) -> None:
    raise ValueError(message)


def walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(walk_keys(child))
    return keys


def source_path_for_validation(source_name: str) -> Path:
    active_agx_path = AGX_SOURCE_FALLBACKS.get(source_name)
    if active_agx_path is not None and active_agx_path.is_file():
        return active_agx_path
    repository_path = SOURCE_HASH_PATHS[source_name]
    return repository_path


def validate(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        fail("capture plan root must be an object")
    missing_top = REQUIRED_TOP_LEVEL - document.keys()
    if missing_top:
        fail(f"missing top-level fields: {sorted(missing_top)}")
    if document["plan_status"] != "frozen":
        fail("formal capture plan must be frozen")
    if document["pose_mode"] != "freeform_diverse":
        fail("pose_mode must be freeform_diverse")
    if document["status_source"] != "safe_driver":
        fail("status_source must be safe_driver")
    if (
        document["capture_enabled"] is not True
        or document["user_approved"] is not True
        or document["formal_capture_prerequisites_frozen"] is not True
    ):
        fail("frozen capture plan must be enabled, approved, and prerequisite-frozen")
    if document["motion_authorized"] is not False:
        fail("freeform capture plan must not authorize automatic motion")
    if document["execution_ready"] is not False:
        fail("B4 capture must not set production execution_ready")
    purpose = document["purpose"]
    if (
        not isinstance(purpose, str)
        or "software-only clear" not in purpose
        or "does not authorize Torque" not in purpose
    ):
        fail("purpose must describe the exact guarded clear and no-Torque boundary")
    if document["operator_contract"].get("capture_timeout_s") != 15.0:
        fail("operator_contract.capture_timeout_s must equal 15.0")

    poses = document["capture_order"]
    if not isinstance(poses, list) or len(poses) != 30:
        fail("capture_order must contain exactly 30 entries")
    orders = [pose.get("capture_order") for pose in poses]
    ids = [pose.get("sample_id") for pose in poses]
    pose_indices = [pose.get("pose_index") for pose in poses]
    splits = Counter(pose.get("split") for pose in poses)
    if orders != list(range(1, 31)):
        fail("capture_order values must be ordered integers 1..30")
    if ids != EXPECTED_IDS:
        fail("sample IDs must be the exact ordered range S0012..S0041")
    if pose_indices != list(range(12, 42)):
        fail("pose_index values must be the exact ordered range 12..41")
    if splits != Counter({"train": 22, "held_out": 8}):
        fail(f"split counts must be train=22 and held_out=8, got {dict(splits)}")
    held_out_orders = [
        pose["capture_order"] for pose in poses if pose["split"] == "held_out"
    ]
    if held_out_orders != EXPECTED_HELD_OUT_ORDERS:
        fail(
            "held-out entries must be at capture orders "
            f"{EXPECTED_HELD_OUT_ORDERS}"
        )
    for pose in poses:
        if pose.get("pose_mode") != "freeform_diverse":
            fail(f"{pose.get('sample_id')}: pose_mode must be freeform_diverse")
        for key in ("pose_intent", "operator_prompt"):
            if not isinstance(pose.get(key), str) or not pose[key].strip():
                fail(f"{pose.get('sample_id')}: {key} must be non-empty text")

    prohibited = PROHIBITED_TARGET_KEYS.intersection(walk_keys(document))
    if prohibited:
        fail(f"exact target keys are prohibited: {sorted(prohibited)}")
    tag_window = document["measurement_contract"].get("tag_static_window")
    if not isinstance(tag_window, dict):
        fail("tag_static_window contract is missing")
    is_20260803_repeat = document.get("dataset_id") in {
        "gemini335_roarm_20260803_r4",
        "gemini335_roarm_20260803_r5",
        "gemini335_roarm_20260803_r6",
    }
    expected_tag_window = {
        "duration_s_minimum": 1.0,
        "sample_count_minimum": 10 if is_20260803_repeat else 15,
        "sample_gap_ms_maximum": 250.0,
        "translation_spread_m_maximum": 0.003,
        "rotation_spread_deg_maximum": 1.5,
    }
    for key, expected in expected_tag_window.items():
        if tag_window.get(key) != expected:
            fail(f"tag_static_window.{key} must equal {expected}")
    expected_measurement_basis = {
        "measured_on": "2026-08-03" if is_20260803_repeat else "2026-07-26",
        "canonical_image_publish_limit_hz": 30.0,
        "isolated_tag_detection_rate_hz": 12.72 if is_20260803_repeat else 17.7,
        "observed_tag_detection_gap_ms_maximum": 233.6 if is_20260803_repeat else 192.0,
    }
    measurement_basis = tag_window.get("measurement_basis")
    if not isinstance(measurement_basis, dict):
        fail("tag_static_window.measurement_basis is missing")
    for key, expected in expected_measurement_basis.items():
        if measurement_basis.get(key) != expected:
            fail(f"tag_static_window.measurement_basis.{key} must equal {expected}")

    joint_window = document["measurement_contract"].get("joint_and_safety_window")
    if not isinstance(joint_window, dict):
        fail("joint_and_safety_window contract is missing")
    expected_joint_window = {
        "duration_s_minimum": 1.0,
        "sample_count_minimum": 20,
        "sample_gap_ms_maximum": 100.0,
        "joint_span_encoder_steps_maximum": 2.0,
        "rgb_joint_skew_ms_maximum": 50.0,
        "all_six_torque_flags_required": True,
        "safe_driver_armed_required": False,
        "safe_driver_dry_run_required": False,
        "voltage_v_minimum": 11.0,
        "voltage_v_maximum": 13.0,
        "joint_target_publisher_count_required": 0,
    }
    for key, expected in expected_joint_window.items():
        if joint_window.get(key) != expected:
            fail(f"joint_and_safety_window.{key} must equal {expected}")

    expected_raw_provenance = {
        "capture_plan_sha256_required": True,
        "recorder_timeout_s": 15.0,
        "frozen_source_hashes_required": True,
        "driver_versions_required": True,
        "joint_window_statistics_required": True,
        "per_frame_safe_status_required": True,
        "recorder_plan_revalidation_required": True,
        "preappend_integrity_chain_required": True,
        "prefix_preserving_index_append_required": True,
    }
    raw_provenance = document["measurement_contract"].get(
        "raw_sample_provenance"
    )
    if raw_provenance != expected_raw_provenance:
        fail("measurement_contract.raw_sample_provenance is not frozen exactly")

    source_hashes = document["source_hashes_sha256"]
    if not isinstance(source_hashes, dict):
        fail("source_hashes_sha256 must be an object")
    if source_hashes.keys() != SOURCE_HASH_PATHS.keys():
        fail("source_hashes_sha256 keys do not match the frozen source set")
    for source_name in SOURCE_HASH_PATHS:
        source_path = source_path_for_validation(source_name)
        if not source_path.is_file():
            fail(f"frozen source file is missing: {source_name}")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_hashes[source_name] != digest:
            fail(f"source hash mismatch: {source_name}")

    split_policy = document["split_policy"]
    if (
        split_policy.get("train_count") != 22
        or split_policy.get("held_out_count") != 8
        or split_policy.get("held_out_capture_orders") != EXPECTED_HELD_OUT_ORDERS
    ):
        fail("split_policy conflicts with capture_order")
    b5 = document["b5_preregistration"]
    if (
        b5.get("formal_solver_entrypoint")
        != "tools/calibration/eye_to_hand_formal_solver.py"
        or "exact 22 preregistered train samples" not in b5.get(
            "train_isolation", ""
        )
        or "spends access with O_EXCL" not in b5.get("held_out_access", "")
    ):
        fail("formal solver split-isolation contract is not frozen")

    return {
        "path": str(path),
        "plan_status": document["plan_status"],
        "capture_enabled": document["capture_enabled"],
        "pose_mode": document["pose_mode"],
        "sample_ids": f"{ids[0]}..{ids[-1]}",
        "counts": dict(splits),
        "held_out_orders": held_out_orders,
        "motion_authorized": document["motion_authorized"],
        "execution_ready": document["execution_ready"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    result = validate(args.plan)
    print(json.dumps({"valid": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
