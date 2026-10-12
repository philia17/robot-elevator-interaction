"""Isolated ROS concurrency probe for the T102 executor.

The probe starts the real executor node and a fake safe-driver contract in the
same isolated ROS domain.  The fake service/topic implementation never opens
serial, never arms a real device, and never exposes a real command topic.  It
delays lease and arm responses and target convergence so the executor's
callback groups must make progress while the Action callback is waiting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import rclpy
from elevator_interfaces.action import PrecontactMove
from elevator_interfaces.srv import PrecontactSpeedLease, PreparePrecontact
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from roarm_t102_outbound_executor.node import T102OutboundExecutor


JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)


class FakeSafeDriver(Node):
    """A delayed, in-memory safe-driver protocol for the isolated domain."""

    def __init__(self, q0: list[float]) -> None:
        super().__init__("t102_fake_safe_driver")
        self._q = list(q0)
        self._target: list[float] | None = None
        self._target_ready_at = 0.0
        self.hold_target = False
        self.soft_converge_no_motion = False
        self.partial_undertravel_rad = 0.0
        self._telemetry_target = list(q0)
        self.lease_active = False
        self.lease_enable_count = 0
        self.lease_release_count = 0
        self.arm_history: list[bool] = []
        self.target_count = 0
        self.feedback_count = 0
        self.status_count = 0
        self._armed = False
        self._lease_group = ReentrantCallbackGroup()
        self._feedback_group = MutuallyExclusiveCallbackGroup()
        self._target_group = ReentrantCallbackGroup()
        self.create_service(
            PrecontactSpeedLease,
            "/roarm_safe/precontact_speed_lease",
            self._lease,
            callback_group=self._lease_group,
        )
        self.create_service(
            SetBool,
            "/roarm_safe/set_armed",
            self._arm,
            callback_group=self._lease_group,
        )
        self.create_subscription(
            JointState,
            "/roarm_safe/joint_target",
            self._target_received,
            10,
            callback_group=self._target_group,
        )
        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._feedback_pub = self.create_publisher(String, "/roarm_safe/feedback_status", 10)
        self._status_pub = self.create_publisher(
            String,
            "/roarm_safe/status",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.create_timer(0.02, self._publish_feedback, callback_group=self._feedback_group)

    def _lease(self, request, response):
        time.sleep(0.12)
        response.success = True
        response.message = (
            "PRECONTACT_SPEED_LEASE_ACTIVE"
            if request.enable
            else "PRECONTACT_SPEED_LEASE_RELEASED"
        )
        response.active = bool(request.enable)
        self.lease_active = bool(request.enable)
        if request.enable:
            self.lease_enable_count += 1
        else:
            self.lease_release_count += 1
        return response

    def _arm(self, request, response):
        time.sleep(0.12)
        self._armed = bool(request.data)
        self.arm_history.append(self._armed)
        response.success = True
        response.message = "FAKE_ARM_OK"
        return response

    def _target_received(self, message: JointState) -> None:
        self.target_count += 1
        self._target = [float(value) for value in message.position]
        self._telemetry_target = list(self._target)
        self._target_ready_at = time.monotonic() + 0.12

    def reset_to_start(self, q0: list[float]) -> None:
        """Reset only the fake state between independent action trials."""
        self._target = None
        self._q = list(q0)
        self._telemetry_target = list(q0)
        self._armed = False

    def _publish_feedback(self) -> None:
        if self._target is not None and not self.hold_target and time.monotonic() >= self._target_ready_at:
            if not self.soft_converge_no_motion:
                self._q = list(self._target)
                if self.partial_undertravel_rad:
                    self._q[0] += self.partial_undertravel_rad
            self._target = None
            self._armed = False
        stamp = self.get_clock().now().to_msg()
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = list(JOINT_NAMES)
        joint.position = list(self._q)
        self._joint_pub.publish(joint)
        feedback = {
            "state": "CONVERGED_DISARMED",
            "armed": bool(self._armed),
            "fault": "",
            "feedback_stamp_ns": stamp_ns,
            "positions_rad": list(self._q),
            "torque_enabled": [True] * 6,
            "dry_run": False,
            "voltage": 12.0,
        }
        message = String()
        message.data = json.dumps(feedback, sort_keys=True, separators=(",", ":"))
        self._feedback_pub.publish(message)
        self.feedback_count += 1
        status = {
            "state": "CONVERGED_DISARMED",
            "armed": bool(self._armed),
            "fault": "",
            "telemetry": {"target_positions": list(self._telemetry_target)},
            "joint_speed_steps_s": (
                [34] * 6 if self.lease_active else [34, 17, 17, 23, 34, 17]
            ),
            "precontact_speed_lease": {"active": self.lease_active},
        }
        status_message = String()
        status_message.data = json.dumps(status, sort_keys=True, separators=(",", ":"))
        self._status_pub.publish(status_message)
        self.status_count += 1


class ProbeClient(Node):
    def __init__(self) -> None:
        super().__init__("t102_runtime_probe_client")
        group = ReentrantCallbackGroup()
        self.prepare = self.create_client(
            PreparePrecontact,
            "/roarm_t102_outbound_executor/prepare_precontact",
            callback_group=group,
        )
        self.action = ActionClient(
            self,
            PrecontactMove,
            "/roarm_t102_outbound_executor/move_to_precontact",
            callback_group=group,
        )
        self.heartbeat = 0
        self.create_timer(0.02, self._tick, callback_group=group)

    def _tick(self) -> None:
        self.heartbeat += 1


def _wait_future(future, timeout: float = 8.0):
    event = threading.Event()
    future.add_done_callback(lambda _future: event.set())
    if not event.wait(timeout):
        raise TimeoutError("ROS future did not complete")
    return future.result()


def _prepare(probe: ProbeClient, route: dict) -> object:
    request = PreparePrecontact.Request()
    request.canonical_target_snapshot_json = json.dumps({"v4": True}, sort_keys=True)
    request.target_snapshot_sha256 = route["frozen_bindings"]["m0_target_sha256"]
    request.ui_request_id = "t102-runtime-probe-prepare"
    return _wait_future(probe.prepare.call_async(request))


def _goal(probe: ProbeClient, authorization_id: str, target_sha: str):
    goal = PrecontactMove.Goal()
    goal.authorization_id = authorization_id
    goal.target_snapshot_sha256 = target_sha
    goal.confirmation_text = "MOVE_TO_PRECONTACT_ONLY"
    return _wait_future(probe.action.send_goal_async(goal))


def run(route_path: Path, release_path: Path, case: str = "success_cancel") -> dict:
    route = json.loads(route_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    q0 = [float(value) for value in route["start_positions_rad"]]
    rclpy.init()
    fake = FakeSafeDriver(q0)
    executor_node = T102OutboundExecutor()
    executor_node.set_parameters(
        [
            rclpy.parameter.Parameter("execution_enabled", value=True),
            rclpy.parameter.Parameter("prepared_route_json", value=json.dumps(route, sort_keys=True)),
            rclpy.parameter.Parameter("prepared_route_sha256", value=route["route_sha256"]),
            rclpy.parameter.Parameter("supervised_release_json", value=json.dumps(release, sort_keys=True)),
            rclpy.parameter.Parameter("supervised_release_sha256", value=release["release_sha256"]),
            rclpy.parameter.Parameter("standby_joint_positions_rad", value=q0),
            rclpy.parameter.Parameter("standby_tolerance_rad", value=0.02),
            rclpy.parameter.Parameter("max_feedback_age_s", value=0.5),
            rclpy.parameter.Parameter("step_timeout_s", value=2.0),
            rclpy.parameter.Parameter("waypoint_tolerance_rad", value=0.005),
        ]
    )
    probe = ProbeClient()
    ros_executor = MultiThreadedExecutor(num_threads=8)
    for node in (fake, executor_node, probe):
        ros_executor.add_node(node)
    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()
    output: dict = {"route_sha256": route["route_sha256"]}
    try:
        if not probe.prepare.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("prepare service unavailable")
        if not probe.action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("action server unavailable")
        # Initial exact-stamp feedback/joint pair must be available before Prepare.
        deadline = time.monotonic() + 5.0
        while executor_node._matched_pair is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if executor_node._matched_pair is None:
            raise RuntimeError("fake feedback pair did not reach executor")

        if case in {"no_progress", "partial_undertravel"}:
            fake.soft_converge_no_motion = case == "no_progress"
            fake.partial_undertravel_rad = 0.006 if case == "partial_undertravel" else 0.0
            prepared_negative = _prepare(probe, route)
            if not prepared_negative.accepted:
                raise RuntimeError(f"{case} prepare rejected: {prepared_negative.reason}")
            negative_handle = _goal(
                probe,
                prepared_negative.authorization_id,
                prepared_negative.planned_target_snapshot_sha256,
            )
            if not negative_handle.accepted:
                raise RuntimeError(f"{case} goal rejected")
            negative_result = _wait_future(negative_handle.get_result_async()).result
            if negative_result.success or "undertravel" not in negative_result.terminal_reason:
                raise RuntimeError(f"{case} did not tight-gate abort: {negative_result.terminal_reason}")
            if fake.target_count != 1:
                raise RuntimeError(f"{case} emitted more than the first waypoint")
            if fake.lease_enable_count != 1 or fake.lease_release_count != 1 or fake.lease_active:
                raise RuntimeError(f"{case} did not restore the lease")
            if executor_node._target_pub is not None or executor_node._arm_client is not None:
                raise RuntimeError(f"{case} resources survived teardown")
            output.update(
                {
                    "case": case,
                    "action": "SAFE_FAILURE_FIRST_WAYPOINT_ONLY",
                    "terminal_reason": negative_result.terminal_reason,
                    "target_count": fake.target_count,
                    "lease_enable_count": fake.lease_enable_count,
                    "lease_release_count": fake.lease_release_count,
                    "resources_destroyed": True,
                    "lease_restored": True,
                }
            )
            return output

        prepared = _prepare(probe, route)
        if not prepared.accepted:
            raise RuntimeError(f"prepare rejected: {prepared.reason}")
        first = probe.action.send_goal_async(
            PrecontactMove.Goal(
                authorization_id=prepared.authorization_id,
                target_snapshot_sha256=prepared.planned_target_snapshot_sha256,
                confirmation_text="MOVE_TO_PRECONTACT_ONLY",
            )
        )
        duplicate = probe.action.send_goal_async(
            PrecontactMove.Goal(
                authorization_id=prepared.authorization_id,
                target_snapshot_sha256=prepared.planned_target_snapshot_sha256,
                confirmation_text="MOVE_TO_PRECONTACT_ONLY",
            )
        )
        first_handle, duplicate_handle = _wait_future(first), _wait_future(duplicate)
        accepted_handles = [handle for handle in (first_handle, duplicate_handle) if handle.accepted]
        if len(accepted_handles) != 1:
            raise RuntimeError(f"expected one accepted goal, got {len(accepted_handles)}")
        result = _wait_future(accepted_handles[0].get_result_async()).result
        if not result.success:
            raise RuntimeError(f"first action failed: {result.terminal_reason}")
        if fake.lease_enable_count != 1 or fake.lease_release_count != 1:
            raise RuntimeError("first action did not complete lease enable/release")
        first_target_count = fake.target_count

        # Second run is deliberately held at the target and cancelled.  This
        # exercises cancel observation, disarm teardown and the second release.
        fake.reset_to_start(q0)
        time.sleep(0.15)
        fake.hold_target = True
        prepared2 = _prepare(probe, route)
        if not prepared2.accepted:
            raise RuntimeError(f"cancel probe prepare rejected: {prepared2.reason}")
        second_handle = _goal(
            probe,
            prepared2.authorization_id,
            prepared2.planned_target_snapshot_sha256,
        )
        if not second_handle.accepted:
            raise RuntimeError("cancel probe goal rejected")
        time.sleep(0.25)
        _wait_future(second_handle.cancel_goal_async())
        cancelled_result = _wait_future(second_handle.get_result_async()).result
        if cancelled_result.success or cancelled_result.terminal_code != "SAFE_FAILURE":
            raise RuntimeError("cancel did not produce safe failure")
        if fake.lease_enable_count != 2 or fake.lease_release_count != 2:
            raise RuntimeError("cancel path did not release its lease")
        if fake.lease_active:
            raise RuntimeError("fake lease remained active")
        if executor_node._target_pub is not None or executor_node._arm_client is not None:
            raise RuntimeError("execution resources survived teardown")
        if executor_node._lease_client is not None or executor_node._lease_id:
            raise RuntimeError("lease client/identity survived teardown")

        output.update(
            {
                "first_action": "COMPLETE_NO_CONTACT",
                "duplicate_goal_rejected": True,
                "cancel_action": "SAFE_FAILURE",
                "lease_enable_count": fake.lease_enable_count,
                "lease_release_count": fake.lease_release_count,
                "arm_history": fake.arm_history,
                "target_count": fake.target_count,
                "first_route_target_count": first_target_count,
                "feedback_count": fake.feedback_count,
                "status_count": fake.status_count,
                "probe_heartbeat_count": probe.heartbeat,
                "resources_destroyed": True,
                "lease_restored": True,
            }
        )
        return output
    finally:
        ros_executor.shutdown(timeout_sec=3.0)
        for node in (probe, executor_node, fake):
            try:
                node.destroy_node()
            except Exception:
                pass
        spin_thread.join(timeout=3.0)
        if rclpy.ok():
            rclpy.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument(
        "--case",
        choices=("success_cancel", "no_progress", "partial_undertravel"),
        default="success_cancel",
    )
    args = parser.parse_args()
    print(json.dumps(run(Path(args.route).expanduser().resolve(), Path(args.release).expanduser().resolve(), args.case), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
