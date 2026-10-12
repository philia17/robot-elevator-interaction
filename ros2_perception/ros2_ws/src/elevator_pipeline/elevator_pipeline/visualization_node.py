"""Headless ROS 2 debug-image visualization node."""

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from elevator_interfaces.msg import (
    ButtonDetectionArray,
    ButtonObservationArray,
    TargetButton2D,
    TargetCandidate2D,
)

from .visualization_core import build_overlay_plan, render_debug_image


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.declare_parameter('enabled', True)
        self.declare_parameter('input_image_topic', '/elevator/selected_frame')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter('observations_topic', '/elevator/button_observations')
        self.declare_parameter('candidate_topic', '/elevator/target_candidate')
        self.declare_parameter('target_topic', '/elevator/target_button')
        self.declare_parameter('task_state_topic', '/elevator/task_state')
        self.declare_parameter('debug_image_topic', '/elevator/debug_image')
        self._enabled = bool(self.get_parameter('enabled').value)
        self._bridge = CvBridge()
        self._detections = []
        self._observations = []
        self._candidate = None
        self._target = None
        self._state = 'IDLE'
        self._publisher = self.create_publisher(
            Image,
            self.get_parameter('debug_image_topic').value,
            qos_profile_sensor_data,
        )
        self._subscriptions = [
            self.create_subscription(
                Image,
                self.get_parameter('input_image_topic').value,
                self._image_callback,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                ButtonDetectionArray,
                self.get_parameter('detections_topic').value,
                self._detections_callback,
                10,
            ),
            self.create_subscription(
                ButtonObservationArray,
                self.get_parameter('observations_topic').value,
                self._observations_callback,
                10,
            ),
            self.create_subscription(
                TargetCandidate2D,
                self.get_parameter('candidate_topic').value,
                self._candidate_callback,
                10,
            ),
            self.create_subscription(
                TargetButton2D,
                self.get_parameter('target_topic').value,
                self._target_callback,
                10,
            ),
            self.create_subscription(
                String,
                self.get_parameter('task_state_topic').value,
                self._state_callback,
                10,
            ),
        ]
        self.get_logger().info(
            'Phase C headless visualization is active (enabled={})'.format(
                self._enabled
            )
        )

    def _detections_callback(self, message):
        self._detections = list(message.detections)

    def _observations_callback(self, message):
        self._observations = list(message.observations)

    def _candidate_callback(self, message):
        self._candidate = message

    def _target_callback(self, message):
        self._target = message

    def _state_callback(self, message):
        self._state = message.data or 'unknown'

    def _image_callback(self, message):
        if not self._enabled:
            return
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
        plan = build_overlay_plan(
            self._detections,
            self._observations,
            self._candidate,
            self._target,
        )
        debug = render_debug_image(image, plan, self._state)
        output = self._bridge.cv2_to_imgmsg(debug, encoding='rgb8')
        output.header = message.header
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
