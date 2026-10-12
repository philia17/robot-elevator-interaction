"""ROS 2 detection-semantic fusion node."""

import rclpy
from rclpy.node import Node

from elevator_interfaces.msg import (
    ButtonDetectionArray,
    ButtonObservation,
    ButtonObservationArray,
    ButtonSemanticArray,
)

from .fusion import FusionValidationError, fuse_detections_and_semantics


class ObservationFusionNode(Node):
    def __init__(self):
        super().__init__('observation_fusion_node')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter('semantics_topic', '/elevator/button_semantics')
        self.declare_parameter('observations_topic', '/elevator/button_observations')
        self.declare_parameter('detector_weight', 0.5)
        self.declare_parameter('semantic_weight', 0.5)
        self._latest_detections = None
        self._publisher = self.create_publisher(
            ButtonObservationArray,
            self.get_parameter('observations_topic').value,
            10,
        )
        self._detection_subscription = self.create_subscription(
            ButtonDetectionArray,
            self.get_parameter('detections_topic').value,
            self._detections_callback,
            10,
        )
        self._semantic_subscription = self.create_subscription(
            ButtonSemanticArray,
            self.get_parameter('semantics_topic').value,
            self._semantics_callback,
            10,
        )
        self.get_logger().info('Phase B detection-ID fusion is active')

    def _detections_callback(self, message):
        self._latest_detections = message

    def _semantics_callback(self, semantics):
        detections = self._latest_detections
        if detections is None:
            self.get_logger().warning('semantic result arrived before detections')
            return
        try:
            fused = fuse_detections_and_semantics(
                detections.detections,
                semantics.semantics,
                detections.image_width,
                detections.image_height,
                self.get_parameter('detector_weight').value,
                self.get_parameter('semantic_weight').value,
            )
        except FusionValidationError as exc:
            self.get_logger().error('fusion rejected input: {}'.format(exc))
            return

        output = ButtonObservationArray()
        output.header = detections.header
        output.image_width = detections.image_width
        output.image_height = detections.image_height
        for fused_item in fused:
            item = ButtonObservation()
            for field_name in (
                    'detection_id', 'label', 'button_type', 'x_min', 'y_min',
                    'x_max', 'y_max', 'center_x', 'center_y',
                    'detector_confidence', 'semantic_confidence',
                    'overall_confidence', 'pressable', 'valid',
                    'validation_reason'):
                setattr(item, field_name, getattr(fused_item, field_name))
            output.observations.append(item)
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = ObservationFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
