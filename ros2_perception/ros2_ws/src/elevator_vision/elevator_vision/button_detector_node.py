"""ROS 2 wrapper for Phase B button detector backends and SoM generation."""

from time import monotonic

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image

from elevator_interfaces.msg import ButtonDetection, ButtonDetectionArray

from .detectors import MockDetector, YoloDetector
from .som import generate_som


class ButtonDetectorNode(Node):
    def __init__(self):
        super().__init__('button_detector_node')
        self.declare_parameter('backend', 'mock')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.30)
        self.declare_parameter('device', 0)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('iou_threshold', 0.70)
        self.declare_parameter('max_det', 300)
        self.declare_parameter('input_topic', '/elevator/selected_frame')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter('som_image_topic', '/elevator/som_image')

        backend = self.get_parameter('backend').value
        if backend == 'mock':
            self._detector = MockDetector()
        elif backend == 'yolo':
            self._detector = YoloDetector(
                self.get_parameter('model_path').value,
                self.get_parameter('confidence_threshold').value,
                self.get_parameter('device').value,
                self.get_parameter('imgsz').value,
                self.get_parameter('iou_threshold').value,
                self.get_parameter('max_det').value,
            )
        else:
            raise ValueError("detector backend must be 'mock' or 'yolo'")

        self._backend_name = backend
        self._bridge = CvBridge()
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._detections_publisher = self.create_publisher(
            ButtonDetectionArray,
            self.get_parameter('detections_topic').value,
            10,
        )
        self._som_publisher = self.create_publisher(
            Image,
            self.get_parameter('som_image_topic').value,
            qos_profile_sensor_data,
        )
        latest_frame_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(
            Image,
            self.get_parameter('input_topic').value,
            self._image_callback,
            latest_frame_qos,
        )
        self.get_logger().info(
            'button detector active backend={} device={} imgsz={} conf={} '
            'iou={} max_det={} input_qos_depth=1'.format(
                backend,
                self.get_parameter('device').value,
                self.get_parameter('imgsz').value,
                self.get_parameter('confidence_threshold').value,
                self.get_parameter('iou_threshold').value,
                self.get_parameter('max_det').value,
            )
        )

    def _image_callback(self, message):
        self._frames_received += 1
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
        except Exception as exc:
            self._frames_dropped += 1
            self.get_logger().error('RGB frame conversion failed: {}'.format(exc))
            return
        start = monotonic()
        detections = self._detector.detect(image)
        latency_ms = (monotonic() - start) * 1000.0
        self._frames_processed += 1
        stamp_ns = (
            int(message.header.stamp.sec) * 1000000000
            + int(message.header.stamp.nanosec)
        )
        frame_summary = (
            'frame_result received={} processed={} dropped={} stamp_ns={} '
            'frame_id={} detections={} latency_ms={:.3f}'
        ).format(
            self._frames_received,
            self._frames_processed,
            self._frames_dropped,
            stamp_ns,
            message.header.frame_id,
            len(detections),
            latency_ms,
        )
        if latency_ms > 100.0:
            self.get_logger().warning(frame_summary + ' latency_threshold_ms=100.0')
        else:
            self.get_logger().info(frame_summary)

        output = ButtonDetectionArray()
        output.header = message.header
        output.image_width = int(image.shape[1])
        output.image_height = int(image.shape[0])
        output.detector_latency_ms = latency_ms
        output.backend = self._backend_name
        for detection in detections:
            item = ButtonDetection()
            item.detection_id = detection.detection_id
            item.x_min = detection.x_min
            item.y_min = detection.y_min
            item.x_max = detection.x_max
            item.y_max = detection.y_max
            item.center_x = detection.center_x
            item.center_y = detection.center_y
            item.detector_confidence = detection.detector_confidence
            item.coarse_type = detection.coarse_type
            output.detections.append(item)
        self._detections_publisher.publish(output)

        som_result = generate_som(image, detections)
        som_message = self._bridge.cv2_to_imgmsg(
            som_result.annotated_image,
            encoding='rgb8',
        )
        som_message.header = message.header
        self._som_publisher.publish(som_message)


def main(args=None):
    rclpy.init(args=args)
    node = ButtonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            'detector_summary received={} processed={} dropped={}'.format(
                node._frames_received,
                node._frames_processed,
                node._frames_dropped,
            )
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
