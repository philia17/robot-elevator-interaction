"""ROS 2 frame gate with topic and optional file input modes."""

from pathlib import Path

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image

from elevator_interfaces.msg import FrameStatus

from .frame_metrics import FrameGate
from .qos import parse_image_qos


class FrameGateNode(Node):
    def __init__(self):
        super().__init__('frame_gate_node')
        self.declare_parameter('source_type', 'topic')
        self.declare_parameter(
            'image_topic',
            '/ascamera_hp60c/camera_publisher/rgb0/image',
        )
        self.declare_parameter('input_topic', '')
        self.declare_parameter('qos_reliability', 'best_effort')
        self.declare_parameter('qos_durability', 'volatile')
        self.declare_parameter('queue_depth', 10)
        self.declare_parameter('file_path', '')
        self.declare_parameter('file_publish_period_sec', 1.0)
        self.declare_parameter('selected_frame_topic', '/elevator/selected_frame')
        self.declare_parameter('full_resolution_frame_topic', '')
        self.declare_parameter('frame_status_topic', '/elevator/frame_status')
        self.declare_parameter('output_width', 640)
        self.declare_parameter('output_height', 480)
        self.declare_parameter('metric_width', 640)
        self.declare_parameter('metric_height', 480)
        self.declare_parameter('min_sharpness', 10.0)
        self.declare_parameter('min_brightness', 20.0)
        self.declare_parameter('max_brightness', 235.0)
        self.declare_parameter('max_motion', 50.0)
        self.declare_parameter('min_interval_sec', 0.1)

        self._bridge = CvBridge()
        self._gate = FrameGate(
            output_width=self.get_parameter('output_width').value,
            output_height=self.get_parameter('output_height').value,
            metric_width=self.get_parameter('metric_width').value,
            metric_height=self.get_parameter('metric_height').value,
            min_sharpness=self.get_parameter('min_sharpness').value,
            min_brightness=self.get_parameter('min_brightness').value,
            max_brightness=self.get_parameter('max_brightness').value,
            max_motion=self.get_parameter('max_motion').value,
            min_interval_sec=self.get_parameter('min_interval_sec').value,
        )
        self._selected_publisher = self.create_publisher(
            Image,
            self.get_parameter('selected_frame_topic').value,
            qos_profile_sensor_data,
        )
        full_resolution_topic = str(
            self.get_parameter('full_resolution_frame_topic').value or ''
        )
        self._full_resolution_publisher = (
            self.create_publisher(
                Image,
                full_resolution_topic,
                qos_profile_sensor_data,
            )
            if full_resolution_topic
            else None
        )
        self._status_publisher = self.create_publisher(
            FrameStatus,
            self.get_parameter('frame_status_topic').value,
            10,
        )

        qos_settings = parse_image_qos(
            self.get_parameter('qos_reliability').value,
            self.get_parameter('qos_durability').value,
            self.get_parameter('queue_depth').value,
        )
        image_qos = QoSProfile(depth=qos_settings.depth)
        image_qos.reliability = {
            'best_effort': ReliabilityPolicy.BEST_EFFORT,
            'reliable': ReliabilityPolicy.RELIABLE,
        }[qos_settings.reliability]
        image_qos.durability = {
            'volatile': DurabilityPolicy.VOLATILE,
            'transient_local': DurabilityPolicy.TRANSIENT_LOCAL,
        }[qos_settings.durability]

        source_type = self.get_parameter('source_type').value
        if source_type == 'topic':
            legacy_topic = self.get_parameter('input_topic').value
            image_topic = legacy_topic or self.get_parameter('image_topic').value
            self._subscription = self.create_subscription(
                Image,
                image_topic,
                self._image_callback,
                image_qos,
            )
        elif source_type == 'file':
            self._file_image = self._load_file_image(
                self.get_parameter('file_path').value
            )
            self._timer = self.create_timer(
                float(self.get_parameter('file_publish_period_sec').value),
                self._file_timer_callback,
            )
        else:
            raise ValueError("source_type must be 'topic' or 'file'")
        self.get_logger().info('Phase C configurable frame gate is active')

    def _load_file_image(self, file_path):
        path = Path(str(file_path or ''))
        if not path.is_file():
            raise ValueError('file input does not exist: {}'.format(path))
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('file mode requires the optional OpenCV package') from exc
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError('file input could not be decoded: {}'.format(path))
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _file_timer_callback(self):
        message = self._bridge.cv2_to_imgmsg(self._file_image, encoding='rgb8')
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'mock_file_image'
        self._image_callback(message)

    def _image_callback(self, message):
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
        now = self.get_clock().now()
        result = self._gate.evaluate(image, now.nanoseconds / 1.0e9)

        status = FrameStatus()
        status.header = message.header
        status.accepted = result.accepted
        status.sharpness = result.sharpness
        status.brightness = result.brightness
        status.motion_score = result.motion_score
        status.reason = result.reason
        self._status_publisher.publish(status)

        if result.accepted:
            if self._full_resolution_publisher is not None:
                full_resolution = self._bridge.cv2_to_imgmsg(
                    image,
                    encoding='rgb8',
                )
                full_resolution.header = message.header
                self._full_resolution_publisher.publish(full_resolution)
            selected = self._bridge.cv2_to_imgmsg(result.resized_image, encoding='rgb8')
            selected.header = message.header
            self._selected_publisher.publish(selected)


def main(args=None):
    rclpy.init(args=args)
    node = FrameGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
