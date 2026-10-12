"""ROS 2 target resolver using TaskContext and fused observations."""

import rclpy
from rclpy.node import Node

from elevator_interfaces.msg import (
    ButtonObservationArray,
    TargetCandidate2D,
    TaskContext,
)

from .floor_map import FloorMap, FloorMapError
from .resolver import resolve_target


class TargetResolverNode(Node):
    def __init__(self):
        super().__init__('target_resolver_node')
        self.declare_parameter('floor_map_path', '')
        self.declare_parameter('minimum_confidence', 0.5)
        self.declare_parameter('task_context_topic', '/elevator/task_context')
        self.declare_parameter('observations_topic', '/elevator/button_observations')
        self.declare_parameter('candidate_topic', '/elevator/target_candidate')
        self._floor_map = FloorMap.load(self.get_parameter('floor_map_path').value)
        self._latest_task = None
        self._publisher = self.create_publisher(
            TargetCandidate2D,
            self.get_parameter('candidate_topic').value,
            10,
        )
        self._task_subscription = self.create_subscription(
            TaskContext,
            self.get_parameter('task_context_topic').value,
            self._task_callback,
            10,
        )
        self._observation_subscription = self.create_subscription(
            ButtonObservationArray,
            self.get_parameter('observations_topic').value,
            self._observations_callback,
            10,
        )
        self.get_logger().info('Phase B deterministic target resolver is active')

    def _task_callback(self, message):
        self._latest_task = message

    def _observations_callback(self, message):
        if self._latest_task is None:
            self.get_logger().warning('observations arrived before TaskContext')
            return
        try:
            candidate = resolve_target(
                self._latest_task,
                message.observations,
                self._floor_map,
                self.get_parameter('minimum_confidence').value,
            )
        except FloorMapError as exc:
            self.get_logger().error('target resolution failed: {}'.format(exc))
            return
        output = TargetCandidate2D()
        output.header = message.header
        output.found = candidate.found
        output.request_id = candidate.request_id
        output.building_id = candidate.building_id
        output.requested_floor = candidate.requested_floor
        output.effective_label = candidate.effective_label
        if candidate.button is not None:
            output.button = candidate.button
        output.candidate_confidence = candidate.candidate_confidence
        output.reason = candidate.reason
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = TargetResolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
