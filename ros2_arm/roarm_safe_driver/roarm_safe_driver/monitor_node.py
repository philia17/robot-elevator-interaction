"""Read-only RoArm feedback monitor for Web UI controlled calibration.

This executable intentionally exposes no command subscription and no ROS service.
It never writes to the serial port; the Web UI remains the sole motion authority.
"""

from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .core import JOINT_NAMES, SafetyError, parse_feedback_line
from .transport import ReadOnlySerialTransport


class RoArmFeedbackMonitor(Node):
    def __init__(self):
        super().__init__("roarm_feedback_monitor")
        self.declare_parameter("serial_port", "")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("feedback_stale_s", 1.0)

        self.transport = ReadOnlySerialTransport(
            str(self.get_parameter("serial_port").value),
            int(self.get_parameter("baud_rate").value),
        )
        self.transport.open()
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(String, "/roarm_monitor/status", status_qos)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_feedback = None
        self._sample_count = 0
        self._last_error = None
        self._reader = threading.Thread(
            target=self._reader_loop,
            name="roarm-monitor-feedback",
            daemon=False,
        )
        self._reader.start()
        self.create_timer(0.5, self._publish_status)
        self._publish_status()

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                line = self.transport.readline()
                if not line:
                    continue
                feedback = parse_feedback_line(line)
                if feedback is None:
                    continue
                with self._lock:
                    self._last_feedback = feedback
                    self._sample_count += 1
                    self._last_error = None
                message = JointState()
                message.header.stamp = self.get_clock().now().to_msg()
                message.name = list(JOINT_NAMES)
                message.position = list(feedback.positions)
                message.effort = [float(value) for value in feedback.loads]
                self.joint_pub.publish(message)
            except SafetyError as exc:
                with self._lock:
                    self._last_error = f"invalid feedback: {exc}"
            except Exception as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self._last_error = f"serial reader: {exc}"
                break

    def _publish_status(self):
        with self._lock:
            feedback = self._last_feedback
            sample_count = self._sample_count
            last_error = self._last_error
        age_s = None if feedback is None else time.monotonic() - feedback.received_at
        stale_limit = float(self.get_parameter("feedback_stale_s").value)
        if last_error:
            state = "MONITOR_FAULT"
        elif feedback is None:
            state = "MONITOR_WAITING_FOR_UNSOLICITED_FEEDBACK"
        elif age_s is not None and age_s > stale_limit:
            state = "MONITOR_FEEDBACK_STALE"
        else:
            state = "MONITOR_ONLY_NO_WRITES"
        payload = {
            "state": state,
            "sample_count": sample_count,
            "feedback_age_s": age_s,
            "positions_rad": None if feedback is None else list(feedback.positions),
            "loads": None if feedback is None else list(feedback.loads),
            "torque_enabled": None if feedback is None else list(feedback.torque_enabled),
            "voltage": None if feedback is None else feedback.voltage,
            "error": last_error,
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_pub.publish(message)

    def destroy_node(self):
        self._stop.set()
        self.transport.close()
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoArmFeedbackMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
