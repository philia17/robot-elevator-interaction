"""ROS 2 health and DiagnosticArray publisher for the perception pipeline."""

from time import monotonic

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from elevator_interfaces.msg import (
    ButtonDetectionArray,
    ButtonObservationArray,
    ButtonSemanticArray,
    FrameStatus,
    PerceptionHealth,
    TargetButton2D,
    TaskContext,
)

from .diagnostics_core import DiagnosticsTracker


class DiagnosticsNode(Node):
    def __init__(self):
        super().__init__('diagnostics_node')
        self.declare_parameter('enabled', True)
        self.declare_parameter('publish_period_sec', 1.0)
        self.declare_parameter('history_size', 30)
        self.declare_parameter('stale_frame_sec', 2.5)
        self.declare_parameter('frame_status_topic', '/elevator/frame_status')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter('semantics_topic', '/elevator/button_semantics')
        self.declare_parameter('observations_topic', '/elevator/button_observations')
        self.declare_parameter('target_topic', '/elevator/target_button')
        self.declare_parameter('task_context_topic', '/elevator/task_context')
        self.declare_parameter('task_state_topic', '/elevator/task_state')
        self.declare_parameter('health_topic', '/elevator/perception_health')
        self.declare_parameter('diagnostics_topic', '/diagnostics')
        self._enabled = bool(self.get_parameter('enabled').value)
        self._tracker = DiagnosticsTracker(
            history_size=self.get_parameter('history_size').value,
            stale_frame_sec=self.get_parameter('stale_frame_sec').value,
        )
        self._health_publisher = self.create_publisher(
            PerceptionHealth,
            self.get_parameter('health_topic').value,
            10,
        )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            self.get_parameter('diagnostics_topic').value,
            10,
        )
        self._subscriptions = [
            self.create_subscription(
                FrameStatus,
                self.get_parameter('frame_status_topic').value,
                self._frame_callback,
                10,
            ),
            self.create_subscription(
                ButtonDetectionArray,
                self.get_parameter('detections_topic').value,
                self._detections_callback,
                10,
            ),
            self.create_subscription(
                ButtonSemanticArray,
                self.get_parameter('semantics_topic').value,
                self._semantics_callback,
                10,
            ),
            self.create_subscription(
                ButtonObservationArray,
                self.get_parameter('observations_topic').value,
                self._observations_callback,
                10,
            ),
            self.create_subscription(
                TargetButton2D,
                self.get_parameter('target_topic').value,
                self._target_callback,
                10,
            ),
            self.create_subscription(
                TaskContext,
                self.get_parameter('task_context_topic').value,
                self._task_callback,
                10,
            ),
            self.create_subscription(
                String,
                self.get_parameter('task_state_topic').value,
                self._state_callback,
                10,
            ),
        ]
        self._timer = self.create_timer(
            max(0.1, float(self.get_parameter('publish_period_sec').value)),
            self._publish,
        )
        self.get_logger().info(
            'Phase C diagnostics is active (enabled={})'.format(self._enabled)
        )

    def _frame_callback(self, message):
        self._tracker.record_frame(message.accepted, monotonic())

    def _detections_callback(self, message):
        self._tracker.record_detections(
            len(message.detections),
            message.detector_latency_ms,
        )

    def _semantics_callback(self, message):
        self._tracker.record_semantics(
            len(message.semantics),
            message.vlm_latency_ms,
        )

    def _observations_callback(self, message):
        self._tracker.record_observations(len(message.observations))

    def _target_callback(self, message):
        self._tracker.record_target(message.valid)

    def _task_callback(self, message):
        self._tracker.record_task(message.active)

    def _state_callback(self, message):
        self._tracker.record_state(message.data)

    @staticmethod
    def _key_value(key, value):
        item = KeyValue()
        item.key = str(key)
        item.value = str(value)
        return item

    def _publish(self):
        if not self._enabled:
            return
        snapshot = self._tracker.snapshot(monotonic())
        stamp = self.get_clock().now().to_msg()

        health = PerceptionHealth()
        health.header.stamp = stamp
        health.overall_status = snapshot.overall_status
        health.input_fps = snapshot.input_fps
        health.detector_latency_ms = snapshot.detector_latency_ms
        health.vlm_latency_ms = snapshot.vlm_latency_ms
        health.detected_count = snapshot.detected_count
        health.labeled_count = snapshot.labeled_count
        health.current_state = snapshot.current_state
        health.warnings = list(snapshot.warnings)
        self._health_publisher.publish(health)

        status = DiagnosticStatus()
        status.name = 'elevator_perception'
        status.hardware_id = 'orin_nano'
        status.message = snapshot.overall_status
        status.level = {
            'OK': DiagnosticStatus.OK,
            'WARN': DiagnosticStatus.WARN,
            'ERROR': DiagnosticStatus.ERROR,
        }.get(snapshot.overall_status, DiagnosticStatus.STALE)
        values = {
            'pipeline_state': snapshot.current_state,
            'last_frame_age_sec': snapshot.last_frame_age_sec,
            'last_frame_accepted': (
                snapshot.last_frame_accepted
                if snapshot.last_frame_accepted is not None
                else 'unknown'
            ),
            'detected_count': snapshot.detected_count,
            'semantic_count': snapshot.labeled_count,
            'observation_count': snapshot.observation_count,
            'target_valid': snapshot.target_valid,
            'current_task_active': snapshot.task_active,
            'detector_latency_ms': snapshot.detector_latency_ms,
            'vlm_latency_ms': snapshot.vlm_latency_ms,
            'fusion_latency_ms': snapshot.fusion_latency_ms,
            'total_latency_ms': snapshot.total_latency_ms,
            'timeout_count': snapshot.timeout_count,
            'failure_count': snapshot.failure_count,
            'last_error': snapshot.last_error or 'none',
            'warning_state': ','.join(snapshot.warnings) or 'none',
        }
        status.values = [
            self._key_value(key, value) for key, value in values.items()
        ]
        array = DiagnosticArray()
        array.header.stamp = stamp
        array.status = [status]
        self._diagnostics_publisher.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
