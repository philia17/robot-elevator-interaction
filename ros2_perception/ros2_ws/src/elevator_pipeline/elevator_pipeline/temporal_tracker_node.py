"""ROS 2 wrapper for reusable 2D temporal target stabilization."""

import rclpy
from rclpy.node import Node

from elevator_interfaces.msg import TargetButton2D, TargetCandidate2D

from .resolver import Candidate
from .temporal import TemporalTracker


class TemporalTrackerNode(Node):
    def __init__(self):
        super().__init__('temporal_tracker_node')
        self.declare_parameter('required_confirmations', 2)
        self.declare_parameter('history_size', 5)
        self.declare_parameter('minimum_iou', 0.5)
        self.declare_parameter('maximum_center_distance', 20.0)
        self.declare_parameter('minimum_candidate_confidence', 0.0)
        self.declare_parameter('candidate_topic', '/elevator/target_candidate')
        self.declare_parameter('target_topic', '/elevator/target_button')
        self._tracker = TemporalTracker(
            required_confirmations=self.get_parameter('required_confirmations').value,
            history_size=self.get_parameter('history_size').value,
            minimum_iou=self.get_parameter('minimum_iou').value,
            maximum_center_distance=self.get_parameter('maximum_center_distance').value,
            minimum_candidate_confidence=self.get_parameter(
                'minimum_candidate_confidence'
            ).value,
        )
        self._publisher = self.create_publisher(
            TargetButton2D,
            self.get_parameter('target_topic').value,
            10,
        )
        self._subscription = self.create_subscription(
            TargetCandidate2D,
            self.get_parameter('candidate_topic').value,
            self._candidate_callback,
            10,
        )
        self.get_logger().info('Phase B temporal tracker is active')

    def _candidate_callback(self, message):
        candidate = Candidate(
            found=message.found,
            request_id=message.request_id,
            building_id=message.building_id,
            requested_floor=message.requested_floor,
            effective_label=message.effective_label,
            button=message.button if message.found else None,
            candidate_confidence=message.candidate_confidence,
            reason=message.reason,
        )
        tracked = self._tracker.update(candidate)
        output = TargetButton2D()
        output.header = message.header
        output.valid = tracked.valid
        output.request_id = tracked.request_id
        output.building_id = tracked.building_id
        output.requested_floor = tracked.requested_floor
        output.effective_label = tracked.effective_label
        if tracked.button is not None:
            output.button = tracked.button
        output.temporal_confidence = tracked.temporal_confidence
        output.status = tracked.status
        output.failure_reason = tracked.failure_reason
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = TemporalTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
