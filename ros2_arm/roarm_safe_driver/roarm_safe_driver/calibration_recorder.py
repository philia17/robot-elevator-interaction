"""One-shot, read-only eye-to-hand calibration sample recorder.

It consumes measured feedback, AprilTag observations and TF.  It never opens a
serial port, creates a motion interface, or publishes a command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import rclpy
from apriltag_msgs.msg import AprilTagDetectionArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from .calibration_core import (
    append_integrity_indexes,
    JointSample,
    MAX_JOINT_SAMPLE_GAP_NS,
    MAX_RGB_JOINT_SKEW_NS,
    MAX_SETTLE_DELTA_RAD,
    MAX_TAG_ROTATION_SPREAD_DEG,
    MAX_TAG_SAMPLE_GAP_NS,
    MAX_TAG_TRANSLATION_SPREAD_M,
    MIN_SETTLE_SAMPLES,
    MIN_TAG_SAMPLES,
    MONITOR_STATUS_SOURCE,
    SAFE_DRIVER_STATUS_SOURCE,
    SafetyError,
    SETTLE_WINDOW_NS,
    TAG_WINDOW_NS,
    joint_sample,
    nearest_joint_sample,
    raw_directory_lock,
    sha256_file,
    stamp_ns,
    tag_observation,
    validate_settled,
    validate_active_source_hashes,
    validate_preappend_integrity_chain,
    validate_safe_status_window,
    validate_status_alignment,
    validate_status_payload,
    validate_tag_window,
    validate_transform,
    validate_raw_integrity,
    write_sample_exclusive,
)
from .core import JOINT_NAMES

CAMERA_FRAME = "camera_optical_frame"
TAG_FRAME = "calibration_tag"
BASE_FRAME = "roarm_base_link"
TCP_FRAME = "hand_tcp"
TAG_EDGE_M = 0.064
STATUS_TOPICS = {
    MONITOR_STATUS_SOURCE: "/roarm_monitor/status",
    SAFE_DRIVER_STATUS_SOURCE: "/roarm_safe/feedback_status",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FROZEN_DRIVER_SOURCE_KEYS = (
    "calibration_core.py",
    "calibration_recorder.py",
    "safe_driver_core.py",
    "safe_driver_node.py",
    "safe_driver_transport.py",
    "torque_transaction.py",
    "agx_unified_safe.yaml",
    "tf_bridge.py",
    "phase4_tf_bridge.launch.py",
    "roarm_m3.xacro",
)


class CalibrationRecorder(Node):
    def __init__(self, parsed):
        super().__init__("roarm_calibration_recorder")
        self.parsed = parsed
        self.status_topic = STATUS_TOPICS[parsed.status_source]
        self.started_at = time.monotonic()
        self.failure = None
        self.completed = False
        self._joints: list[JointSample] = []
        self._camera_tag_tf = {}
        self._tag_detections = {}
        self._tag_observations = {}
        self._status = None
        self._safe_status_by_stamp = {}
        self._last_rejection = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.create_subscription(
            JointState, "/joint_states", self._joint_callback, 100
        )
        self.create_subscription(
            AprilTagDetectionArray,
            "/calibration/apriltag/detections",
            self._detection_callback,
            20,
        )
        self.create_subscription(TFMessage, "/tf", self._tf_callback, 100)
        status_qos = (
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            if parsed.status_source == MONITOR_STATUS_SOURCE
            else QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.create_subscription(
            String, self.status_topic, self._status_callback, status_qos
        )
        self.create_timer(0.02, self._attempt_capture)

    @staticmethod
    def _transform_values(transform):
        translation = transform.translation
        rotation = transform.rotation
        return validate_transform(
            (translation.x, translation.y, translation.z),
            (rotation.x, rotation.y, rotation.z, rotation.w),
            "TF",
        )

    def _reject(self, reason: str):
        if reason != self._last_rejection:
            self.get_logger().info(f"waiting: {reason}")
            self._last_rejection = reason

    def _joint_callback(self, message):
        try:
            sample = joint_sample(
                stamp_ns(message.header.stamp), message.name, message.position
            )
        except SafetyError as exc:
            self._reject(f"joint feedback rejected: {exc}")
            return
        self._joints.append(sample)
        self._joints = self._joints[-256:]

    def _status_callback(self, message):
        try:
            payload = json.loads(message.data)
            evidence = validate_status_payload(payload, self.parsed.status_source)
            status = {
                "received_ns": self.get_clock().now().nanoseconds,
                "payload": payload,
                "evidence": evidence,
            }
            if self.parsed.status_source == SAFE_DRIVER_STATUS_SOURCE:
                self._safe_status_by_stamp[evidence["feedback_stamp_ns"]] = status
                if len(self._safe_status_by_stamp) > 256:
                    for key in sorted(self._safe_status_by_stamp)[:-256]:
                        del self._safe_status_by_stamp[key]
            else:
                self._status = status
        except (TypeError, ValueError, json.JSONDecodeError, SafetyError) as exc:
            if self.parsed.status_source == MONITOR_STATUS_SOURCE:
                self._status = None
            self._reject(f"{self.parsed.status_source} status rejected: {exc}")

    def _validate_safe_driver_publishers(self):
        status_publishers = self.get_publishers_info_by_topic(self.status_topic)
        joint_publishers = self.get_publishers_info_by_topic("/joint_states")
        command_publishers = self.get_publishers_info_by_topic(
            "/roarm_safe/joint_target"
        )
        if command_publishers:
            raise SafetyError(
                "safe-driver capture requires zero joint-target publishers"
            )
        if len(status_publishers) != 1 or len(joint_publishers) != 1:
            raise SafetyError(
                "safe-driver mode requires one status and one joint-state publisher"
            )
        status_owner = (
            status_publishers[0].node_namespace,
            status_publishers[0].node_name,
        )
        joint_owner = (
            joint_publishers[0].node_namespace,
            joint_publishers[0].node_name,
        )
        if (
            status_owner != joint_owner
            or status_publishers[0].node_name != "roarm_safe_driver"
        ):
            raise SafetyError(
                "safe-driver status and joint states must share the unique roarm_safe_driver owner"
            )

    def _tf_callback(self, message):
        for transform in message.transforms:
            if (
                transform.header.frame_id == CAMERA_FRAME
                and transform.child_frame_id == TAG_FRAME
            ):
                try:
                    key = stamp_ns(transform.header.stamp)
                    self._camera_tag_tf[key] = transform
                    self._pair_tag_observation(key)
                except SafetyError:
                    continue
        if len(self._camera_tag_tf) > 256:
            for key in sorted(self._camera_tag_tf)[:-256]:
                del self._camera_tag_tf[key]

    def _pair_tag_observation(self, key):
        tag = self._tag_detections.get(key)
        camera_tf = self._camera_tag_tf.get(key)
        if tag is None or camera_tf is None:
            return
        translation, rotation = self._transform_values(camera_tf.transform)
        self._tag_observations[key] = tag_observation(
            key,
            translation,
            rotation,
            family=tag.family,
            tag_id=tag.id,
            hamming=tag.hamming,
            decision_margin=tag.decision_margin,
        )
        if len(self._tag_observations) > 256:
            for old_key in sorted(self._tag_observations)[:-256]:
                del self._tag_observations[old_key]

    def _detection_callback(self, message):
        if message.header.frame_id != CAMERA_FRAME:
            self._reject("AprilTag header frame is not camera_optical_frame")
            return
        matches = [
            item
            for item in message.detections
            if item.family == "tag36h11" and item.id == 0
        ]
        if len(matches) != 1:
            self._reject("expected exactly one tag36h11 ID 0 detection")
            return
        tag = matches[0]
        if (
            tag.hamming != 0
            or not math.isfinite(tag.decision_margin)
            or tag.decision_margin < 50.0
        ):
            self._reject("AprilTag quality is below the Phase 5 threshold")
            return
        try:
            key = stamp_ns(message.header.stamp)
        except SafetyError as exc:
            self._reject(str(exc))
            return
        self._tag_detections[key] = tag
        try:
            self._pair_tag_observation(key)
        except SafetyError as exc:
            self._reject(f"AprilTag pose rejected: {exc}")
            self._tag_detections.pop(key, None)
            return
        if len(self._tag_detections) > 256:
            for old_key in sorted(self._tag_detections)[:-256]:
                del self._tag_detections[old_key]

    def _attempt_capture(self):
        if self.completed or self.failure:
            return
        if time.monotonic() - self.started_at > self.parsed.timeout_s:
            self.failure = "calibration recorder timeout without an accepted sample"
            return
        if (
            self.parsed.status_source == MONITOR_STATUS_SOURCE
            and not self._status
        ):
            self._reject(f"healthy {self.parsed.status_source} status has not arrived")
            return
        if self.parsed.status_source == MONITOR_STATUS_SOURCE:
            age_ns = self.get_clock().now().nanoseconds - self._status["received_ns"]
            if age_ns < 0 or age_ns > 1_000_000_000:
                self._reject("monitor status is stale")
                return
        if not self._tag_observations:
            self._reject("timestamp-aligned AprilTag pose has not arrived")
            return
        rgb_stamp_ns = max(self._tag_observations)
        try:
            tag_window = validate_tag_window(
                tuple(self._tag_observations.values()), rgb_stamp_ns
            )
            tag = tag_window.observations[-1]
            joint = nearest_joint_sample(self._joints, rgb_stamp_ns)
            settle_window = validate_settled(
                self._joints, joint.stamp_ns
            )
            status = self._status
            safe_status_evidence_window = None
            if self.parsed.status_source == SAFE_DRIVER_STATUS_SOURCE:
                self._validate_safe_driver_publishers()
                status = self._safe_status_by_stamp.get(joint.stamp_ns)
                if status is None:
                    return
                safe_status_evidence_window = validate_safe_status_window(
                    settle_window,
                    {
                        stamp: item["payload"]
                        for stamp, item in self._safe_status_by_stamp.items()
                    },
                )
                validate_status_alignment(status["evidence"], joint)
            # Exact joint time only: Time(0) (latest) is deliberately never used.
            target_time = Time(nanoseconds=joint.stamp_ns)
            if not self.tf_buffer.can_transform(
                BASE_FRAME,
                TCP_FRAME,
                target_time,
                timeout=Duration(seconds=0.0),
            ):
                return
            base_tcp = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                TCP_FRAME,
                target_time,
                timeout=Duration(seconds=0.0),
            )
            base_stamp_ns = stamp_ns(base_tcp.header.stamp)
            if abs(base_stamp_ns - joint.stamp_ns) > 1_000_000:
                raise SafetyError(
                    "FK TF stamp does not match selected joint feedback"
                )
            base_translation, base_rotation = self._transform_values(
                base_tcp.transform
            )
            sample_path, manifest_path, sums_path = self._commit_sample(
                rgb_stamp_ns,
                joint,
                tag,
                base_translation,
                base_rotation,
                tag_window,
                status,
                settle_window,
                safe_status_evidence_window,
            )
            self.get_logger().info(
                "CALIBRATION_SAMPLE_COMPLETE "
                f"sample={sample_path} manifest={manifest_path} sums={sums_path}"
            )
            self.completed = True
            return
        except SafetyError as exc:
            self._reject(str(exc))
            return

    def _commit_sample(
        self,
        rgb_stamp_ns,
        joint,
        tag,
        base_translation,
        base_rotation,
        tag_window,
        status,
        settle_window,
        safe_status_evidence_window,
    ):
        with raw_directory_lock(self.parsed.dataset_dir):
            existing_raw_hashes = validate_raw_integrity(
                self.parsed.dataset_dir,
                dataset_id=self.parsed.dataset_id,
            )
            validate_preappend_integrity_chain(
                self.parsed.dataset_dir,
                dataset_id=self.parsed.dataset_id,
                existing_hashes=existing_raw_hashes,
            )
            expected_historical = [
                f"S{index:04d}.json" for index in range(1, 12)
            ]
            historical_existing = [
                name
                for name in existing_raw_hashes
                if int(name[1:5]) < 12
            ]
            # A newly frozen independent repeat may start directly at S0012.
            # If legacy samples are present, they must still be the complete
            # immutable S0001-S0011 prefix; partial prefixes remain rejected.
            if historical_existing not in ([], expected_historical):
                raise SafetyError(
                    "historical S0001-S0011 integrity prefix is incomplete"
                )
            formal_existing = sorted(
                name
                for name in existing_raw_hashes
                if int(name[1:5]) >= 12
            )
            expected_formal = [
                f"S{index:04d}.json"
                for index in range(12, 12 + len(formal_existing))
            ]
            if formal_existing != expected_formal:
                raise SafetyError("formal raw sample order is not contiguous")
            if self.parsed.sample_id != f"S{12 + len(formal_existing):04d}":
                raise SafetyError("requested sample is not the next formal sample ID")

            manifest_path = self.parsed.dataset_dir / "manifest.json"
            sums_path = self.parsed.dataset_dir / "SHA256SUMS"
            preappend_manifest_sha256 = (
                sha256_file(manifest_path) if manifest_path.is_file() else None
            )
            preappend_sums_sha256 = (
                sha256_file(sums_path) if sums_path.is_file() else None
            )
            document = self._document(
                rgb_stamp_ns,
                joint,
                tag,
                base_translation,
                base_rotation,
                tag_window,
                status,
                settle_window,
                safe_status_evidence_window,
                existing_raw_hashes,
                preappend_manifest_sha256,
                preappend_sums_sha256,
            )
            sample_path = write_sample_exclusive(
                self.parsed.dataset_dir,
                self.parsed.sample_id,
                document,
            )
            manifest_path, sums_path = append_integrity_indexes(
                self.parsed.dataset_dir,
                dataset_id=self.parsed.dataset_id,
                new_sample_path=sample_path,
                existing_hashes=existing_raw_hashes,
            )
        return sample_path, manifest_path, sums_path

    def _document(
        self,
        rgb_stamp_ns,
        joint,
        tag,
        base_translation,
        base_rotation,
        tag_window,
        status,
        settle_window,
        safe_status_evidence_window,
        existing_raw_hashes,
        preappend_manifest_sha256,
        preappend_sums_sha256,
    ):
        evidence = status["evidence"]
        joint_gaps_ns = [
            current.stamp_ns - previous.stamp_ns
            for previous, current in zip(settle_window, settle_window[1:])
        ]
        joint_spans_rad = {
            name: max(item.positions[index] for item in settle_window)
            - min(item.positions[index] for item in settle_window)
            for index, name in enumerate(JOINT_NAMES)
        }
        safe_observations = []
        if safe_status_evidence_window is not None:
            safe_observations = [
                {
                    "stamp_ns": joint_item.stamp_ns,
                    "positions_rad": list(joint_item.positions),
                    "voltage_v": status_evidence["voltage_v"],
                    "torque_enabled": status_evidence["torque_enabled"],
                    "torque_evidence": status_evidence["torque_evidence"],
                }
                for joint_item, status_evidence in zip(
                    settle_window, safe_status_evidence_window
                )
            ]
        driver_versions = {
            key: self.parsed.frozen_source_hashes_sha256[key]
            for key in FROZEN_DRIVER_SOURCE_KEYS
        }
        return {
            "sample_id": self.parsed.sample_id,
            "dataset_id": self.parsed.dataset_id,
            "capture_plan_sha256": self.parsed.capture_plan_sha256,
            "coordinate_contract_version": "1.0.0",
            "calibration_type": "eye_on_base",
            "timestamp_ns": rgb_stamp_ns,
            "rgb_header_stamp_ns": rgb_stamp_ns,
            "joint_states_stamp_ns": joint.stamp_ns,
            "settled": True,
            "settle_reason": (
                "1s measured window, >=20 samples, <=100ms gaps, <=2 encoder "
                "steps; every frame has aligned healthy safe-driver evidence"
                if self.parsed.status_source == SAFE_DRIVER_STATUS_SOURCE
                else
                "1s measured window, >=20 samples, <=100ms gaps, <=2 encoder steps"
            ),
            "joint_positions_rad": list(joint.positions),
            "joint_names": list(JOINT_NAMES),
            "voltage_v": evidence["voltage_v"],
            "torque_enabled": evidence["torque_enabled"],
            "torque_evidence": evidence["torque_evidence"],
            "T_roarm_base_link_hand_tcp": {
                "translation_m": base_translation,
                "rotation_xyzw": base_rotation,
            },
            "T_camera_optical_frame_calibration_tag": {
                "translation_m": list(tag_window.translation_m),
                "rotation_xyzw": list(tag_window.rotation_xyzw),
            },
            "tag_family": "tag36h11",
            "tag_id": 0,
            "tag_edge_m": TAG_EDGE_M,
            "tag_detection_quality": {
                "hamming": 0,
                "decision_margin": float(tag.decision_margin),
                "reprojection_error_px": None,
                "reprojection_error_available": False,
                "reprojection_error_reason": (
                    "apriltag_msgs detection does not provide this field"
                ),
            },
            "metadata": {
                "ros_domain_id": self.parsed.ros_domain_id,
                "capture_plan_sha256": self.parsed.capture_plan_sha256,
                "recorder_timeout_s": self.parsed.timeout_s,
                "frozen_source_hashes_sha256": (
                    self.parsed.frozen_source_hashes_sha256
                ),
                "driver_versions": driver_versions,
                "preappend_raw_integrity": {
                    "existing_sample_count": len(existing_raw_hashes),
                    "existing_sample_hashes_sha256": existing_raw_hashes,
                    "manifest_sha256": preappend_manifest_sha256,
                    "sha256sums_sha256": preappend_sums_sha256,
                },
                "pose_index": self.parsed.pose_index,
                "split": self.parsed.split,
                "operator_authorization_id": self.parsed.operator_authorization_id,
                "status_source": self.parsed.status_source,
                "status_topic": self.status_topic,
                "status_received_ns": status["received_ns"],
                "tag_static_window": {
                    "start_stamp_ns": tag_window.observations[0].stamp_ns,
                    "end_stamp_ns": tag_window.observations[-1].stamp_ns,
                    "duration_ns": tag_window.duration_ns,
                    "sample_count": len(tag_window.observations),
                    "maximum_gap_ns": max(
                        current.stamp_ns - previous.stamp_ns
                        for previous, current in zip(
                            tag_window.observations,
                            tag_window.observations[1:],
                        )
                    ),
                    "translation_spread_m": tag_window.translation_spread_m,
                    "rotation_spread_deg": tag_window.rotation_spread_deg,
                    "spread_definition": (
                        "maximum pairwise Euclidean translation distance and "
                        "maximum pairwise quaternion geodesic rotation distance"
                    ),
                    "aggregation_method": tag_window.aggregation_method,
                    "rejection_policy": {
                        "window_duration_ns_minimum": TAG_WINDOW_NS,
                        "sample_count_minimum": MIN_TAG_SAMPLES,
                        "sample_gap_ns_maximum": MAX_TAG_SAMPLE_GAP_NS,
                        "family": "tag36h11",
                        "tag_id": 0,
                        "hamming_maximum": 0,
                        "decision_margin_minimum": 50.0,
                        "translation_spread_m_maximum": (
                            MAX_TAG_TRANSLATION_SPREAD_M
                        ),
                        "rotation_spread_deg_maximum": (
                            MAX_TAG_ROTATION_SPREAD_DEG
                        ),
                    },
                    "observations": [
                        {
                            "stamp_ns": item.stamp_ns,
                            "translation_m": list(item.translation_m),
                            "rotation_xyzw": list(item.rotation_xyzw),
                            "family": item.family,
                            "tag_id": item.tag_id,
                            "hamming": item.hamming,
                            "decision_margin": item.decision_margin,
                        }
                        for item in tag_window.observations
                    ],
                },
                "settle_window_start_ns": settle_window[0].stamp_ns,
                "settle_window_end_ns": settle_window[-1].stamp_ns,
                "settle_window_sample_count": len(settle_window),
                "settle_window_safe_status": (
                    "all_frames_aligned_healthy"
                    if self.parsed.status_source == SAFE_DRIVER_STATUS_SOURCE
                    else "not_available_monitor_status"
                ),
                "joint_safe_window": {
                    "start_stamp_ns": settle_window[0].stamp_ns,
                    "end_stamp_ns": settle_window[-1].stamp_ns,
                    "duration_ns": (
                        settle_window[-1].stamp_ns - settle_window[0].stamp_ns
                    ),
                    "sample_count": len(settle_window),
                    "maximum_gap_ns": max(joint_gaps_ns),
                    "per_joint_span_rad": joint_spans_rad,
                    "maximum_joint_span_rad": max(joint_spans_rad.values()),
                    "safe_status_alignment": (
                        "all_frames_aligned_healthy"
                        if safe_status_evidence_window is not None
                        else "not_available_monitor_status"
                    ),
                    "rejection_policy": {
                        "window_duration_ns_minimum": SETTLE_WINDOW_NS,
                        "sample_count_minimum": MIN_SETTLE_SAMPLES,
                        "sample_gap_ns_maximum": MAX_JOINT_SAMPLE_GAP_NS,
                        "joint_span_rad_maximum": MAX_SETTLE_DELTA_RAD,
                        "rgb_joint_skew_ns_maximum": MAX_RGB_JOINT_SKEW_NS,
                        "all_six_torque_flags_required": True,
                        "safe_driver_armed_required": False,
                        "safe_driver_dry_run_required": False,
                        "voltage_v_minimum": 11.0,
                        "voltage_v_maximum": 13.0,
                        "joint_target_publisher_count_required": 0,
                    },
                    "observations": safe_observations,
                },
                "publisher_ownership_check": (
                    "ROS graph advertised one matching node name and namespace "
                    "for status and joint-state publishers, with zero "
                    "joint-target publishers"
                    if self.parsed.status_source == SAFE_DRIVER_STATUS_SOURCE
                    else "not_applicable"
                ),
            },
        }


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _arguments(args):
    parser = argparse.ArgumentParser(
        description="Read-only one-shot calibration recorder"
    )
    parser.add_argument("--capture-plan", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--operator-authorization-id", required=True)
    parser.add_argument("--capture-plan-sha256", required=True)
    parsed = parser.parse_args(args)

    parsed.capture_plan = parsed.capture_plan.expanduser().resolve()
    if not parsed.capture_plan.is_file() or parsed.capture_plan.suffix != ".json":
        parser.error("--capture-plan must be an existing JSON file")
    if not SHA256_PATTERN.fullmatch(parsed.capture_plan_sha256):
        parser.error("--capture-plan-sha256 must be a lowercase SHA-256 digest")
    plan_bytes = parsed.capture_plan.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != parsed.capture_plan_sha256:
        parser.error("--capture-plan bytes changed or do not match the supplied digest")
    try:
        plan = json.loads(
            plan_bytes,
            object_pairs_hook=_strict_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        parser.error("--capture-plan must contain strict JSON without duplicate keys")
    if (
        not isinstance(plan, dict)
        or plan.get("plan_status") != "frozen"
        or plan.get("capture_enabled") is not True
        or plan.get("user_approved") is not True
        or plan.get("formal_capture_prerequisites_frozen") is not True
        or plan.get("motion_authorized") is not False
        or plan.get("execution_ready") is not False
        or plan.get("status_source") != SAFE_DRIVER_STATUS_SOURCE
    ):
        parser.error("--capture-plan is not an enabled fail-closed formal plan")

    frozen_source_hashes = plan.get("source_hashes_sha256")
    if (
        not isinstance(frozen_source_hashes, dict)
        or not frozen_source_hashes
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not SHA256_PATTERN.fullmatch(value)
            for key, value in frozen_source_hashes.items()
        )
    ):
        parser.error(
            "--capture-plan source hashes must be lowercase SHA-256 digests"
        )
    missing_driver_keys = set(FROZEN_DRIVER_SOURCE_KEYS) - frozen_source_hashes.keys()
    if missing_driver_keys:
        parser.error(
            "--capture-plan is missing required driver provenance"
        )
    try:
        validate_active_source_hashes(
            parsed.capture_plan,
            frozen_source_hashes,
        )
    except SafetyError as exc:
        parser.error(f"--capture-plan active source validation failed: {exc}")

    dataset_dir = plan.get("dataset_dir")
    dataset_id = plan.get("dataset_id")
    ros_domain_id = plan.get("ros_domain_id")
    operator_contract = plan.get("operator_contract")
    capture_order = plan.get("capture_order")
    if (
        not isinstance(dataset_dir, str)
        or not isinstance(dataset_id, str)
        or not dataset_id
        or isinstance(ros_domain_id, bool)
        or not isinstance(ros_domain_id, int)
        or ros_domain_id < 0
        or not isinstance(operator_contract, dict)
        or operator_contract.get("capture_timeout_s") != 15.0
        or not isinstance(capture_order, list)
        or len(capture_order) != 30
        or any(not isinstance(item, dict) for item in capture_order)
    ):
        parser.error("--capture-plan runtime contract is invalid")
    expected_ids = [f"S{index:04d}" for index in range(12, 42)]
    if [item.get("sample_id") for item in capture_order] != expected_ids:
        parser.error("--capture-plan sample order is invalid")
    held_out_orders = [
        index
        for index, item in enumerate(capture_order, start=1)
        if item.get("split") == "held_out"
    ]
    if (
        sum(item.get("split") == "train" for item in capture_order) != 22
        or held_out_orders != [4, 8, 12, 16, 20, 24, 27, 30]
        or any(item.get("pose_mode") != "freeform_diverse" for item in capture_order)
    ):
        parser.error("--capture-plan split or pose-mode contract is invalid")
    entry = next(
        (item for item in capture_order if item.get("sample_id") == parsed.sample_id),
        None,
    )
    if (
        entry is None
        or isinstance(entry.get("pose_index"), bool)
        or not isinstance(entry.get("pose_index"), int)
        or entry.get("split") not in ("train", "held_out")
    ):
        parser.error("--sample-id is not a valid capture-plan entry")
    if not isinstance(
        parsed.operator_authorization_id, str
    ) or not parsed.operator_authorization_id.startswith(
        f"ui-click-{parsed.sample_id}-"
    ):
        parser.error("--operator-authorization-id is not bound to --sample-id")

    parsed.dataset_dir = Path(dataset_dir).expanduser().resolve()
    if parsed.dataset_dir.name != "raw":
        parser.error("--capture-plan dataset_dir must be the immutable raw directory")
    parsed.dataset_id = dataset_id
    parsed.pose_index = entry["pose_index"]
    parsed.split = entry["split"]
    parsed.status_source = SAFE_DRIVER_STATUS_SOURCE
    parsed.ros_domain_id = ros_domain_id
    parsed.timeout_s = 15.0
    parsed.frozen_source_hashes_sha256 = frozen_source_hashes
    return parsed


def main(args=None):
    parsed = _arguments(args)
    rclpy.init(args=None)
    node = CalibrationRecorder(parsed)
    try:
        while rclpy.ok() and not node.completed and not node.failure:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.failure:
            raise SystemExit(node.failure)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
