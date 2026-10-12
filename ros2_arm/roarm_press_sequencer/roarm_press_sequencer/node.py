"""ROS 2 wrapper for the offline-testable Phase 3 sequencer core."""

from __future__ import annotations

import json
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .core import (
    HighLevelState,
    JOINT_NAMES,
    PressMode,
    Sequencer,
    SequencerError,
    TeachCapture,
    load_artifact,
)


class PressSequencerNode(Node):
    def __init__(self) -> None:
        super().__init__("roarm_press_sequencer")
        self.declare_parameter("artifact_path", "")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("execution_enabled", False)
        self.declare_parameter("command_timeout_sec", 10.0)
        self.declare_parameter("audit_log_path", "")
        artifact_path = self.get_parameter("artifact_path").value
        if not artifact_path:
            raise RuntimeError("artifact_path is required; there is no hardware-ready default")
        self.core = Sequencer(
            load_artifact(artifact_path),
            dry_run=bool(self.get_parameter("dry_run").value),
            execution_enabled=bool(self.get_parameter("execution_enabled").value),
            timeout_sec=float(self.get_parameter("command_timeout_sec").value),
            audit_path=self.get_parameter("audit_log_path").value or None,
        )
        self.teach = TeachCapture(self.core.artifact.encoder_step_rad)
        self.measured_samples: deque[dict[str, float]] = deque(maxlen=100)
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.target_pub = self.create_publisher(JointState, "/roarm_safe/joint_target", 1)
        self.status_pub = self.create_publisher(String, "/roarm_press/status", status_qos)
        self.create_subscription(JointState, "/joint_states", self._joint_state, 10)
        self.create_subscription(String, "/roarm_safe/status", self._driver_status, status_qos)
        self.create_subscription(String, "/roarm_press/operator_command", self._operator_command, 10)
        self.arm_client = self.create_client(SetBool, "/roarm_safe/set_armed")
        self.timer = self.create_timer(0.1, self._poll)
        self._publish_status("started")

    def _joint_state(self, msg: JointState) -> None:
        try:
            values = dict(zip(msg.name, msg.position))
            ordered = [values[name] for name in JOINT_NAMES]
            self.core.update_measured(JOINT_NAMES, ordered)
            self.measured_samples.append(dict(zip(JOINT_NAMES, ordered)))
        except (KeyError, SequencerError) as exc:
            self._disarm()
            self.get_logger().error(str(exc))

    def _driver_status(self, msg: String) -> None:
        self.core.driver_status(msg.data, time.monotonic())
        if self.core.state is HighLevelState.ABORT_LATCHED_DISARMED:
            self._disarm()
        self._publish_status("driver_status")

    def _operator_command(self, msg: String) -> None:
        try:
            command = json.loads(msg.data)
            action = command.get("action")
            if action == "set_mode":
                self.core.set_mode(PressMode(command["mode"]))
            elif action == "request_transition":
                self.core.request_transition(HighLevelState(command["target_state"]))
            elif action == "confirm_microstep":
                now = time.monotonic()
                confirmation = self.core.expected_confirmation(
                    now, str(command["confirmation_id"]), str(command["confirmation_text"]),
                    float(command.get("ttl_sec", 30.0)),
                )
                target = self.core.approve_next(confirmation, now)
                if target is not None:
                    self._arm_then_publish(target)
            elif action == "complete_inspection":
                self.core.complete_inspection(str(command["confirmation_text"]))
            elif action == "capture_pose":
                captured = self.teach.capture(
                    str(command["pose"]), list(self.measured_samples)[-10:],
                    str(command["confirmation_id"]),
                )
                self.core.audit.record("teach_pose_captured", captured=captured)
            elif action == "simulate_driver_status" and self.core.dry_run:
                self.core.driver_status(str(command["status"]), time.monotonic())
            elif action == "record_cycle":
                self.core.record_cycle(bool(command["success"]),
                                       activation_evidence=str(command.get("activation_evidence", "")))
            elif action == "abort":
                self.core.abort(str(command.get("reason", "operator abort")))
                self._disarm()
            else:
                raise SequencerError("unknown or prohibited operator action")
            self._publish_status(action or "unknown")
        except (KeyError, ValueError, json.JSONDecodeError, SequencerError) as exc:
            self.get_logger().error(f"operator command rejected: {exc}")
            self._publish_status("command_rejected", error=str(exc))

    def _arm_then_publish(self, target) -> None:
        if not self.arm_client.service_is_ready():
            self.core.abort("safe-driver arm service unavailable")
            self._disarm()
            return
        request = SetBool.Request()
        request.data = True
        future = self.arm_client.call_async(request)

        def completed(result_future) -> None:
            try:
                response = result_future.result()
                if not response.success:
                    raise RuntimeError(response.message)
                message = JointState()
                message.header.stamp = self.get_clock().now().to_msg()
                message.name = list(target.names)
                message.position = list(target.positions)
                self.target_pub.publish(message)
                self.core.audit.record("single_joint_target_published", joint=target.changed_joint,
                                       target_rad=target.target_rad,
                                       confirmation_id=target.confirmation_id)
            except Exception as exc:  # ROS futures surface several exception types.
                self.core.abort(f"arming/publish failed: {exc}")
                self._disarm()
        future.add_done_callback(completed)

    def _disarm(self) -> None:
        if not self.arm_client.service_is_ready():
            return
        request = SetBool.Request()
        request.data = False
        self.arm_client.call_async(request)

    def _poll(self) -> None:
        prior = self.core.state
        self.core.poll(time.monotonic())
        if prior is not self.core.state and self.core.state is HighLevelState.ABORT_LATCHED_DISARMED:
            self._disarm()
            self._publish_status("timeout_abort")

    def _publish_status(self, event: str, **extra) -> None:
        pending = self.core.pending
        payload = {
            "schema_version": 1,
            "event": event,
            "artifact_id": self.core.artifact.artifact_id,
            "artifact_sha256": self.core.artifact.sha256,
            "mode": self.core.mode.value,
            "high_level_state": self.core.state.value,
            "control_state": self.core.control.value,
            "dry_run": self.core.dry_run,
            "execution_enabled": self.core.execution_enabled,
            "pending_transition": None if not pending else [pending.source.value, pending.target.value],
            "microstep_index": None if not pending else pending.next_step,
            "consecutive_press_successes": self.core.consecutive_press_successes,
            **extra,
        }
        out = String()
        out.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PressSequencerNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
