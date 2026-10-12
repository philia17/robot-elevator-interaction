"""Publish a timestamp-aligned, typed M0 target with no motion fields."""

import json
import math
from pathlib import Path

import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

from elevator_interfaces.msg import (
    ButtonObservation3D,
    ButtonObservation3DArray,
    ButtonSemanticArray,
    M0Target3D,
)

from .m0_target_core import M0TargetConfig, M0TargetTracker


def _stamp_ns(message):
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


class M0TargetObserverNode(Node):
    def __init__(self):
        super().__init__('m0_target_observer_node')
        self.declare_parameter(
            'observations_topic',
            '/elevator/button_observations_3d',
        )
        self.declare_parameter(
            'semantics_topic',
            '/elevator/button_semantics',
        )
        self.declare_parameter(
            'output_topic',
            '/elevator/m0/perception_target_3d',
        )
        self.declare_parameter('target_label', '1')
        self.declare_parameter('selected_target_topic', '/elevator/m0/selected_target')
        self.declare_parameter('required_consecutive_frames', 5)
        self.declare_parameter('maximum_semantic_age_sec', 2.0)
        self.declare_parameter('maximum_non_target_semantic_age_sec', 30.0)
        self.declare_parameter('maximum_frame_gap_sec', 0.50)
        self.declare_parameter('minimum_detector_confidence', 0.50)
        self.declare_parameter('minimum_depth_valid_ratio', 0.50)
        self.declare_parameter('minimum_bbox_iou', 0.65)
        self.declare_parameter('maximum_position_spread_m', 0.015)
        self.declare_parameter('target_semantic_grace_ttl_sec', 10.0)
        self.declare_parameter('require_complete_semantic_set', True)
        self.declare_parameter('evidence_jsonl_path', '')

        config = M0TargetConfig(
            target_label=str(self.get_parameter('target_label').value),
            required_consecutive_frames=int(
                self.get_parameter('required_consecutive_frames').value
            ),
            maximum_semantic_age_sec=float(
                self.get_parameter('maximum_semantic_age_sec').value
            ),
            maximum_non_target_semantic_age_sec=float(
                self.get_parameter(
                    'maximum_non_target_semantic_age_sec'
                ).value
            ),
            maximum_frame_gap_sec=float(
                self.get_parameter('maximum_frame_gap_sec').value
            ),
            minimum_detector_confidence=float(
                self.get_parameter('minimum_detector_confidence').value
            ),
            minimum_depth_valid_ratio=float(
                self.get_parameter('minimum_depth_valid_ratio').value
            ),
            minimum_bbox_iou=float(
                self.get_parameter('minimum_bbox_iou').value
            ),
            maximum_position_spread_m=float(
                self.get_parameter('maximum_position_spread_m').value
            ),
            target_semantic_grace_ttl_sec=float(
                self.get_parameter(
                    'target_semantic_grace_ttl_sec'
                ).value
            ),
            require_complete_semantic_set=bool(
                self.get_parameter('require_complete_semantic_set').value
            ),
        )
        self._tracker = M0TargetTracker(config)
        self._last_status = None
        evidence_path = str(self.get_parameter('evidence_jsonl_path').value)
        self._evidence_path = Path(evidence_path) if evidence_path else None
        if self._evidence_path is not None:
            self._evidence_path.parent.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self._observations_subscriber = message_filters.Subscriber(
            self,
            ButtonObservation3DArray,
            self.get_parameter('observations_topic').value,
            qos_profile=qos,
        )
        self._semantics_subscriber = message_filters.Subscriber(
            self,
            ButtonSemanticArray,
            self.get_parameter('semantics_topic').value,
            qos_profile=qos,
        )
        self._synchronizer = message_filters.TimeSynchronizer(
            [self._observations_subscriber, self._semantics_subscriber],
            queue_size=10,
        )
        self._synchronizer.registerCallback(self._on_frame)
        self._publisher = self.create_publisher(
            M0Target3D,
            self.get_parameter('output_topic').value,
            10,
        )
        selection_qos = QoSProfile(depth=1)
        selection_qos.reliability = ReliabilityPolicy.RELIABLE
        selection_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._selection_subscriber = self.create_subscription(
            String, self.get_parameter('selected_target_topic').value,
            self._on_selected_target, selection_qos,
        )
        self.get_logger().info(
            'M0 target observer active target_label={} mode=NO_MOTION '
            'required_frames={} output={}'.format(
                config.target_label,
                config.required_consecutive_frames,
                self.get_parameter('output_topic').value,
            )
        )

    @staticmethod
    def _semantic_dict(item):
        metadata = {}
        if item.raw_response:
            try:
                metadata = json.loads(item.raw_response)
            except (TypeError, ValueError):
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            'detection_id': item.detection_id,
            'label': item.label,
            'pressable': item.pressable,
            'track_id': metadata.get('track_id', ''),
            'semantic_status': metadata.get('semantic_status', ''),
            'semantic_age_sec': metadata.get('semantic_age_sec', math.nan),
            'association_valid': metadata.get('association_valid', False),
        }

    def _on_selected_target(self, message):
        """Only select perception evidence; this node has no motion client."""
        try:
            self._tracker.select_target(json.loads(message.data))
        except (TypeError, ValueError) as exc:
            # A malformed replacement cannot leave the old READY target live.
            self._tracker.select_target({'label': ''})
            self.get_logger().warning('m0_target_selection_rejected: ' + str(exc))
        empty = ButtonObservation3DArray()
        empty.header.stamp = self.get_clock().now().to_msg()
        decision = self._tracker.update([], [], _stamp_ns(empty), '')
        self._publish_decision(decision, empty)
        self._last_status = decision['status']

    @staticmethod
    def _observation_dict(item):
        surface = item.surface_geometry
        return {
            'detection_id': item.detection_id,
            'x_min': item.x_min,
            'y_min': item.y_min,
            'x_max': item.x_max,
            'y_max': item.y_max,
            'detector_confidence': item.detector_confidence,
            'depth_valid': item.depth_valid,
            'depth_valid_ratio': item.depth_valid_ratio,
            'x_m': item.x_m,
            'y_m': item.y_m,
            'z_m': item.z_m,
            'surface_geometry': {
                'estimate_valid': surface.estimate_valid,
                'provenance_complete': surface.provenance_complete,
                'failure_reason': surface.failure_reason,
                'surface_point_m': {
                    'x': surface.surface_point_x_m,
                    'y': surface.surface_point_y_m,
                    'z': surface.surface_point_z_m,
                },
                'surface_point_covariance_m2': list(
                    surface.surface_point_covariance_m2
                ),
                'panel_outward_normal': {
                    'x': surface.panel_outward_normal_x,
                    'y': surface.panel_outward_normal_y,
                    'z': surface.panel_outward_normal_z,
                },
                'normal_angular_uncertainty_rad': (
                    surface.normal_angular_uncertainty_rad
                ),
                'plane_rms_residual_m': surface.plane_rms_residual_m,
                'plane_sample_count': surface.plane_sample_count,
                'plane_inlier_count': surface.plane_inlier_count,
                'plane_inlier_ratio': surface.plane_inlier_ratio,
                'geometry_method_id': surface.geometry_method_id,
                'geometry_scope': surface.geometry_scope,
                'depth_sample_source': surface.depth_sample_source,
                'camera_info_stamp_ns': surface.camera_info_stamp_ns,
                'camera_info_sha256': surface.camera_info_sha256,
                'source_frame_id': surface.source_frame_id,
                'canonical_frame_id': surface.canonical_frame_id,
                'raw_to_canonical_alias_id': (
                    surface.raw_to_canonical_alias_id
                ),
                'raw_to_canonical_alias_status': (
                    surface.raw_to_canonical_alias_status
                ),
                'raw_to_canonical_adapter_sha256': (
                    surface.raw_to_canonical_adapter_sha256
                ),
                'section_b_calibration_id': (
                    surface.section_b_calibration_id
                ),
                'camera_optical_axis_status': (
                    surface.camera_optical_axis_status
                ),
                'rgb_depth_registration_id': (
                    surface.rgb_depth_registration_id
                ),
                'rgb_depth_registration_status': (
                    surface.rgb_depth_registration_status
                ),
                'depth_scale_m_per_unit': surface.depth_scale_m_per_unit,
                'depth_scale_status': surface.depth_scale_status,
            },
        }

    @staticmethod
    def _decision_json(decision):
        return json.dumps(
            decision,
            sort_keys=True,
            separators=(',', ':'),
        )

    @staticmethod
    def _find_observation(message, detection_id):
        for item in message.observations:
            if item.detection_id == detection_id:
                return item
        return ButtonObservation3D()

    def _publish_decision(self, decision, observations_message):
        output = M0Target3D()
        output.header = observations_message.header
        output.perception_ready = bool(decision['perception_ready'])
        output.target_label = decision['target_label']
        output.effective_label = decision['effective_label']
        output.matching_label_count = int(decision['matching_label_count'])
        selected = decision.get('selected') or {}
        output.detection_id = selected.get('detection_id', '')
        output.semantic_track_id = selected.get('track_id', '')
        output.semantic_status = selected.get('semantic_status', '')
        output.semantic_age_sec = float(
            selected.get('semantic_age_sec', math.nan)
        )
        output.confirmation_count = int(decision['confirmation_count'])
        output.required_confirmations = int(
            decision['required_confirmations']
        )
        output.position_spread_m = float(decision['position_spread_m'])
        output.observation = self._find_observation(
            observations_message,
            output.detection_id,
        )
        stabilized_point = selected.get('camera_point_m') or {}
        output.observation.x_m = float(
            stabilized_point.get('x', output.observation.x_m)
        )
        output.observation.y_m = float(
            stabilized_point.get('y', output.observation.y_m)
        )
        output.observation.z_m = float(
            stabilized_point.get('z', output.observation.z_m)
        )
        output.observation.surface_geometry.surface_point_x_m = (
            output.observation.x_m
        )
        output.observation.surface_geometry.surface_point_y_m = (
            output.observation.y_m
        )
        output.observation.surface_geometry.surface_point_z_m = (
            output.observation.z_m
        )
        output.status = decision['status']
        output.failure_reason = decision['failure_reason']
        output.motion_authorized = False
        self._publisher.publish(output)

    def _on_frame(self, observations_message, semantics_message):
        observation_stamp_ns = _stamp_ns(observations_message)
        semantic_stamp_ns = _stamp_ns(semantics_message)
        if (
            observation_stamp_ns != semantic_stamp_ns
            or observations_message.header.frame_id
            != semantics_message.header.frame_id
        ):
            self._tracker.reset()
            self.get_logger().warning(
                'm0_target_reject reason=source_header_mismatch '
                'observation_ns={} semantic_ns={} observation_frame={} '
                'semantic_frame={}'.format(
                    observation_stamp_ns,
                    semantic_stamp_ns,
                    observations_message.header.frame_id,
                    semantics_message.header.frame_id,
                )
            )
            return

        decision = self._tracker.update(
            [
                self._observation_dict(item)
                for item in observations_message.observations
            ],
            [
                self._semantic_dict(item)
                for item in semantics_message.semantics
            ],
            observation_stamp_ns,
            observations_message.header.frame_id,
        )
        self._publish_decision(decision, observations_message)
        payload = self._decision_json(decision)
        if self._evidence_path is not None:
            with self._evidence_path.open('a', encoding='utf-8') as stream:
                stream.write(payload + '\n')
                stream.flush()
        if decision['status'] != self._last_status:
            self.get_logger().info(
                'm0_target status={} ready={} confirmations={}/{} '
                'spread_m={:.6f} motion_authorized=false'.format(
                    decision['status'],
                    decision['perception_ready'],
                    decision['confirmation_count'],
                    decision['required_confirmations'],
                    decision['position_spread_m'],
                )
            )
            self._last_status = decision['status']


def main(args=None):
    rclpy.init(args=args)
    node = M0TargetObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
