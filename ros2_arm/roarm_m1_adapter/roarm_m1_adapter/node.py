"""Inert-by-default M1 no-contact return adapter for RoArm-M3.

This node is deliberately NOT a general-purpose sequencer.  It starts with zero
motion publishers and zero arm-service clients.  Only a single explicit
authorization can create them, and any fault destroys them permanently — no
retry, no Torque, no clear_fault, no compensation, no auto-retract.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Resolve repo root for cross-package imports (tools/press is outside the
# ROS 2 workspace).  Search upward so both the Mac layout
# (repo/ros2_arm/roarm_m1_adapter/...) and the AGX layout
# (roarm_ws/src/lift_robot/roarm_m1_adapter/...) resolve correctly.
def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "tools" / "press" / "m1_continuous_return_task.py").is_file():
            return parent
    raise RuntimeError("could not locate repo root containing tools/press")


_REPO_ROOT = _find_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from roarm_safe_driver.calibration_core import (
    SAFE_DRIVER_STATUS_SOURCE,
    validate_status_alignment,
    validate_status_payload,
)
from roarm_safe_driver.core import JOINT_NAMES, SafetyError

from tools.press.m1_continuous_return_task import (
    ContinuousReturnTask,
    TaskAuthorization,
    TaskState,
    measured_digest,
)


class M1AdapterState:
    INERT = "INERT"
    AUTHORIZING = "AUTHORIZING"
    EXECUTING = "EXECUTING"
    COMPLETE = "COMPLETE"
    ABORT_LATCHED = "ABORT_LATCHED"


class M1ReturnAdapter(Node):
    """ROS 2 adapter for the 51-step pre_contact → press_standby no-contact return.

    Inert by default.  No /roarm_safe/joint_target publisher and no
    /roarm_safe/set_armed client are created until a single explicit
    authorization passes all preconditions.
    """

    def __init__(self) -> None:
        super().__init__("roarm_m1_return_adapter")

        # --- passive subscriptions (always active, even when inert) ---
        feedback_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10
        )
        self.create_subscription(
            String, "/roarm_safe/feedback_status", self._on_feedback_status, feedback_qos
        )
        self.create_subscription(
            String, "/roarm_safe/status", self._on_driver_status, feedback_qos
        )

        # --- authorization service (one-shot gate) ---
        self.authorize_srv = self.create_service(
            Trigger, "~/authorize_return", self._on_authorize
        )

        # --- mutable state ---
        self._state = M1AdapterState.INERT
        self._task: ContinuousReturnTask | None = None
        self._latest_joint_state: JointState | None = None
        self._latest_feedback_status: dict[str, Any] | None = None
        self._latest_driver_status: dict[str, Any] | None = None
        self._matched_pair: tuple[JointState, dict[str, Any]] | None = None
        self._target_pub = None
        self._arm_client = None
        self._execution_timer = None
        self._abort_reason: str = ""
        self._step_count: int = 0
        self._step_deadline: float | None = None
        self._feedback_source_node: str | None = None
        self._feedback_source_verified: bool = False

        # --- parameters ---
        self.declare_parameter("artifact_path", "")
        self.declare_parameter("step_timeout_sec", 20.0)
        self.declare_parameter("authorization_token_id", "")
        self.declare_parameter("authorization_artifact_sha256", "")
        self.declare_parameter("authorization_start_measured_digest", "")
        self.declare_parameter("authorization_expires_monotonic", 0.0)
        self.declare_parameter("authorization_confirmation_text", "")

        self.get_logger().info("M1 return adapter started — INERT (no publisher, no arm client)")

    # ------------------------------------------------------------------
    # passive subscription callbacks (always active)
    # ------------------------------------------------------------------

    def _on_joint_states(self, msg: JointState) -> None:
        self._latest_joint_state = msg
        self._try_match_pair()

    def _on_feedback_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        self._latest_feedback_status = payload
        self._try_match_pair()

    def _on_driver_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        self._latest_driver_status = payload

    def _try_match_pair(self) -> None:
        """Match /joint_states and /roarm_safe/feedback_status by exact stamp."""
        if self._latest_joint_state is None or self._latest_feedback_status is None:
            return
        js_stamp_ns = (
            self._latest_joint_state.header.stamp.sec * 1_000_000_000
            + self._latest_joint_state.header.stamp.nanosec
        )
        fb_stamp_ns = self._latest_feedback_status.get("feedback_stamp_ns")
        if js_stamp_ns != fb_stamp_ns:
            return
        self._matched_pair = (self._latest_joint_state, self._latest_feedback_status)

    # ------------------------------------------------------------------
    # one-shot authorization service
    # ------------------------------------------------------------------

    def _on_authorize(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._state is not M1AdapterState.INERT:
            response.success = False
            response.message = f"adapter is not INERT (current: {self._state})"
            return response

        self._state = M1AdapterState.AUTHORIZING
        try:
            self._validate_preconditions()
            self._create_execution_resources()
            self._start_execution()
            response.success = True
            response.message = "authorized — execution started"
        except Exception as exc:
            self._abort(f"authorization failed: {exc}")
            response.success = False
            response.message = self._abort_reason
        return response

    # ------------------------------------------------------------------
    # precondition validation
    # ------------------------------------------------------------------

    def _validate_preconditions(self) -> None:
        """Validate all preconditions before creating any execution resources.

        Raises on first failure — caller must abort without creating
        publisher or arm client.
        """
        # 1. task authorization fields
        artifact_path = self.get_parameter("artifact_path").value
        if not artifact_path:
            raise RuntimeError("artifact_path parameter is required")

        task = ContinuousReturnTask(
            artifact_path,
            timeout_sec=float(self.get_parameter("step_timeout_sec").value),
            expected_motion_publisher_count=0,
        )

        token = str(self.get_parameter("authorization_token_id").value).strip()
        artifact_sha = str(self.get_parameter("authorization_artifact_sha256").value).strip()
        start_digest = str(self.get_parameter("authorization_start_measured_digest").value).strip()
        expires = float(self.get_parameter("authorization_expires_monotonic").value)
        confirmation = str(self.get_parameter("authorization_confirmation_text").value).strip()

        if not token or not artifact_sha or not start_digest or not confirmation:
            raise RuntimeError("all authorization parameters are required")
        if expires <= 0.0:
            raise RuntimeError("authorization expiry must be positive")
        if artifact_sha != task.sha256:
            raise RuntimeError(
                f"artifact SHA mismatch: param={artifact_sha[:16]}… "
                f"file={task.sha256[:16]}…"
            )

        # 2. matched feedback pair
        if self._matched_pair is None:
            raise RuntimeError("no matched /joint_states + /roarm_safe/feedback_status pair")

        js_msg, fb_payload = self._matched_pair

        # 3. validate feedback via calibration_core
        try:
            status_evidence = validate_status_payload(fb_payload, SAFE_DRIVER_STATUS_SOURCE)
        except SafetyError as exc:
            raise RuntimeError(f"feedback_status invalid: {exc}") from exc

        # 4. validate alignment: exact timestamp + six positions match
        joint_positions = self._ordered_joint_positions(js_msg)
        class _JointSample:
            pass
        sample = _JointSample()
        sample.stamp_ns = (
            js_msg.header.stamp.sec * 1_000_000_000 + js_msg.header.stamp.nanosec
        )
        sample.positions = joint_positions
        try:
            validate_status_alignment(status_evidence, sample)
        except SafetyError as exc:
            raise RuntimeError(f"joint/feedback alignment failed: {exc}") from exc

        # 5. verify zero /roarm_safe/joint_target publishers
        target_pub_count = self.count_publishers("/roarm_safe/joint_target")
        if target_pub_count != 0:
            raise RuntimeError(
                f"pre-existing /roarm_safe/joint_target publishers: {target_pub_count}"
            )

        # 6. verify single legitimate safe-driver feedback source
        pub_info = self.get_publishers_info_by_topic("/roarm_safe/feedback_status")
        source_nodes = {
            info.node_name for info in pub_info
        }
        if len(source_nodes) != 1:
            raise RuntimeError(
                f"expected exactly one /roarm_safe/feedback_status publisher, "
                f"found {len(source_nodes)}: {source_nodes}"
            )
        self._feedback_source_node = source_nodes.pop()
        self._feedback_source_verified = True

        # 7. require all six direct torque flags true, voltage 11-13 V,
        #    disarmed terminal state
        torque = status_evidence.get("torque_enabled", [])
        if torque != [True] * 6:
            raise RuntimeError("not all six direct torque flags are true")
        voltage = status_evidence.get("voltage_v", 0.0)
        if not 11.0 <= voltage <= 13.0:
            raise RuntimeError(f"voltage {voltage:.2f} V outside 11–13 V")
        state = fb_payload.get("state", "").upper()
        if state not in {"FAULT_CLEARED_DISARMED", "CONVERGED_DISARMED", "DISARMED"}:
            raise RuntimeError(f"safe driver not in a disarmed terminal state: {state}")
        if fb_payload.get("armed") is not False:
            raise RuntimeError("safe driver is armed")
        if fb_payload.get("dry_run") is not False:
            raise RuntimeError("safe driver is in dry-run mode")

        # 8. require exact current pre-contact task axes
        auth = TaskAuthorization(
            token_id=token,
            artifact_sha256=artifact_sha,
            start_measured_digest=start_digest,
            expires_monotonic=expires,
            confirmation_text=confirmation,
        )
        now_mono = time.monotonic()
        task.authorize(auth, joint_positions, now_mono)

        self._task = task
        self.get_logger().info(
            f"preconditions passed: source={self._feedback_source_node}, "
            f"voltage={voltage:.2f}V, state={state}"
        )

    # ------------------------------------------------------------------
    # execution resource creation
    # ------------------------------------------------------------------

    def _create_execution_resources(self) -> None:
        """Create target publisher and arm client after preconditions pass."""
        # arm client
        from std_srvs.srv import SetBool
        self._arm_client = self.create_client(SetBool, "/roarm_safe/set_armed")
        if not self._arm_client.wait_for_service(timeout_sec=2.0):
            self._arm_client = None
            raise RuntimeError("/roarm_safe/set_armed service not available")

        # target publisher
        self._target_pub = self.create_publisher(
            JointState, "/roarm_safe/joint_target", 1
        )
        target_pub_count = self.count_publishers("/roarm_safe/joint_target")
        if target_pub_count != 1:
            self._destroy_publisher()
            raise RuntimeError(
                f"expected exactly 1 /roarm_safe/joint_target publisher "
                f"(this adapter), found {target_pub_count}"
            )
        self.get_logger().info("execution resources created: 1 publisher + arm client")

    # ------------------------------------------------------------------
    # step execution (timer-driven)
    # ------------------------------------------------------------------

    def _start_execution(self) -> None:
        self._state = M1AdapterState.EXECUTING
        self._step_count = 0
        self._execution_timer = self.create_timer(0.05, self._execute_loop)
        self.get_logger().info("execution loop started")

    def _execute_loop(self) -> None:
        if self._state is not M1AdapterState.EXECUTING:
            return
        if self._task is None:
            self._abort("task lost during execution")
            return

        assert self._task is not None

        if self._task.state is TaskState.COMPLETE_DISARMED:
            self._on_complete()
            return
        if self._task.state is TaskState.ABORT_LATCHED_DISARMED:
            self._abort(f"task aborted: {self._task.abort_reason}")
            return

        now = time.monotonic()

        if self._task.state is TaskState.WAIT_DRIVER:
            self._check_convergence(now)
            return

        if self._task.state is TaskState.READY_NEXT_STEP:
            self._issue_next_step(now)
            return

    def _issue_next_step(self, now: float) -> None:
        assert self._task is not None
        try:
            candidate = self._task.issue_next(now)
        except Exception as exc:
            self._abort(f"issue_next failed: {exc}")
            return

        if candidate is None:
            return

        # Verify feedback source is still legitimate
        self._verify_source_identity()

        # Arm immediately
        if not self._arm_client or not self._arm_client.service_is_ready():
            self._abort("arm service unavailable")
            return

        from std_srvs.srv import SetBool
        request = SetBool.Request()
        request.data = True
        future = self._arm_client.call_async(request)

        def _on_armed(arm_future):
            try:
                response = arm_future.result()
                if not response.success:
                    self._abort(f"arm request rejected: {response.message}")
                    return
                self._publish_target(candidate)
            except Exception as exc:
                self._abort(f"arm call failed: {exc}")

        future.add_done_callback(_on_armed)
        self._step_deadline = now + float(self.get_parameter("step_timeout_sec").value)

    def _publish_target(self, candidate) -> None:
        """Publish exactly one JointState to /roarm_safe/joint_target."""
        if self._target_pub is None:
            self._abort("target publisher destroyed before publish")
            return

        # Verify single publisher is still us
        pub_count = self.count_publishers("/roarm_safe/joint_target")
        if pub_count != 1:
            self._abort(f"joint_target publisher count changed: {pub_count}")
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = list(candidate.target_positions_rad)
        self._target_pub.publish(msg)
        self._step_count += 1
        self.get_logger().info(
            f"step {self._step_count}/51: {candidate.route_step_id} "
            f"joint={candidate.joint} target={candidate.target_rad:.6f}"
        )

    def _check_convergence(self, now: float) -> None:
        assert self._task is not None

        # Timeout check
        if self._step_deadline is not None and now > self._step_deadline:
            self._abort("step convergence timeout")
            return

        # Need a matched pair with NEW feedback
        if self._matched_pair is None:
            return

        _, fb_payload = self._matched_pair
        state = str(fb_payload.get("state", "")).upper()

        if state in {"COMMAND_ACTIVE", "DRY_RUN_ACTIVE"}:
            # Not yet converged; must be armed
            if fb_payload.get("armed") is not True:
                self._abort("active state without armed flag")
            return

        if state != "CONVERGED_DISARMED":
            return

        if fb_payload.get("armed") is not False:
            self._abort("CONVERGED_DISARMED with armed=true")
            return

        # Validate sample alignment for this convergence
        js_msg = self._matched_pair[0]
        joint_positions = self._ordered_joint_positions(js_msg)
        try:
            status_evidence = validate_status_payload(fb_payload, SAFE_DRIVER_STATUS_SOURCE)
        except SafetyError as exc:
            self._abort(f"convergence feedback invalid: {exc}")
            return

        class _JointSample:
            pass
        sample = _JointSample()
        sample.stamp_ns = (
            js_msg.header.stamp.sec * 1_000_000_000 + js_msg.header.stamp.nanosec
        )
        sample.positions = joint_positions
        try:
            validate_status_alignment(status_evidence, sample)
        except SafetyError as exc:
            self._abort(f"convergence alignment failed: {exc}")
            return

        # Verify source identity hasn't changed
        self._verify_source_identity()

        # Feed observed status to task state machine
        driver_status = {
            "state": state,
            "armed": fb_payload.get("armed", False),
            "fault": fb_payload.get("fault", ""),
            "torque_enabled": fb_payload.get("torque_enabled", []),
            "voltage": fb_payload.get("voltage", 0.0),
            "joint_target_publisher_count": 1,
        }
        try:
            self._task.observe(joint_positions, driver_status, now)
            self._step_deadline = None
        except Exception as exc:
            self._abort(f"observe failed: {exc}")

    # ------------------------------------------------------------------
    # completion
    # ------------------------------------------------------------------

    def _on_complete(self) -> None:
        self._state = M1AdapterState.COMPLETE
        self._stop_execution_timer()
        self._disarm()
        self._destroy_publisher()
        self.get_logger().info(
            f"M1 return complete — {self._step_count} steps converged, publisher destroyed"
        )

    # ------------------------------------------------------------------
    # abort (fail-closed, no retry)
    # ------------------------------------------------------------------

    def _abort(self, reason: str) -> None:
        if self._state is M1AdapterState.ABORT_LATCHED:
            return
        self._state = M1AdapterState.ABORT_LATCHED
        self._abort_reason = reason
        self._stop_execution_timer()
        self._disarm()
        self._destroy_publisher()
        if self._task is not None and self._task.state is not TaskState.ABORT_LATCHED_DISARMED:
            try:
                self._task.abort(reason)
            except Exception:
                pass
        self.get_logger().error(
            f"ABORT LATCHED: {reason} — publisher destroyed, will not retry"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _ordered_joint_positions(self, msg: JointState) -> tuple[float, ...]:
        if len(msg.name) < 6:
            raise RuntimeError(f"/joint_states has fewer than 6 names: {msg.name}")
        if len(msg.position) < 6:
            raise RuntimeError("/joint_states has fewer than 6 positions")
        by_name = dict(zip(msg.name, msg.position))
        result = []
        for name in JOINT_NAMES:
            if name not in by_name:
                raise RuntimeError(f"/joint_states missing {name}")
            value = by_name[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"/joint_states {name} is not numeric")
            result.append(float(value))
        return tuple(result)

    def _verify_source_identity(self) -> None:
        """Verify /roarm_safe/feedback_status publisher has not changed."""
        if not self._feedback_source_verified:
            return
        pub_info = self.get_publishers_info_by_topic("/roarm_safe/feedback_status")
        current = {info.node_name for info in pub_info}
        if len(current) != 1:
            self._abort(
                f"feedback_status publisher count changed: {len(current)} ({current})"
            )
            return
        if self._feedback_source_node not in current:
            self._abort(
                f"feedback_status source changed: was {self._feedback_source_node}, "
                f"now {current}"
            )

    def _destroy_publisher(self) -> None:
        if self._target_pub is not None:
            self.destroy_publisher(self._target_pub)
            self._target_pub = None
            self.get_logger().info("/roarm_safe/joint_target publisher destroyed")

    def _disarm(self) -> None:
        """Best-effort single disarm — fire and forget, never retry."""
        if self._arm_client is None:
            return
        if not self._arm_client.service_is_ready():
            return
        from std_srvs.srv import SetBool
        request = SetBool.Request()
        request.data = False
        try:
            self._arm_client.call_async(request)
        except Exception:
            pass

    def _stop_execution_timer(self) -> None:
        if self._execution_timer is not None:
            self.destroy_timer(self._execution_timer)
            self._execution_timer = None

    def destroy_node(self) -> None:
        self._stop_execution_timer()
        self._destroy_publisher()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = M1ReturnAdapter()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
