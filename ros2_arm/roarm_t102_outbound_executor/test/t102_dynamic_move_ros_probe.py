#!/usr/bin/env python3
"""Isolated ROS-graph proof for the Dynamic exact-plan Move path.

The probe uses the real generated Prepare service, Move Action, and executor
node with an in-memory fake safe driver on a caller-selected ROS domain.  It
never imports the serial transport and never exposes any hardware device.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import threading
import time

import rclpy
from elevator_interfaces.action import PrecontactMove
from elevator_interfaces.srv import PreparePrecontact, PrecontactSpeedLease
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from press.t102_dynamic_precontact_core import (
    FrozenSession,
    STOPPED_MODE,
    canonical_hash,
    canonical_json,
)
from roarm_t102_outbound_executor.execution import (
    DYNAMIC_LEASE_PURPOSE,
    MOVE_CONFIRMATION_TOKEN,
)
from roarm_t102_outbound_executor.node import DynamicExecutor


CASES = ("success", "ack_race", "timeout", "fault", "cancel", "lease_failure")


def _wait_future(future, timeout_s: float = 12.0):
    done = threading.Event()
    future.add_done_callback(lambda _future: done.set())
    if not done.wait(timeout_s):
        raise TimeoutError("ROS future did not complete")
    return future.result()


class FakeSafeDriver(Node):
    """Typed lease/arm/feedback contract with no serial or real driver."""

    def __init__(self, q0: tuple[float, ...], joint_names: tuple[str, ...], case: str):
        super().__init__("t102_dynamic_fake_safe_driver")
        self.case = case
        self.q = list(q0)
        self.joint_names = list(joint_names)
        self.state = "CONVERGED_DISARMED"
        self.armed = False
        self.fault = ""
        self.lease_active = False
        self.lease_purpose = ""
        self.lease_id = ""
        self.target: list[float] | None = None
        self.target_received_at = 0.0
        self.received_targets: list[list[float]] = []
        self.expected_waypoints = 4
        self.lease_enable_count = 0
        self.lease_release_count = 0
        self.arm_history: list[bool] = []
        group = ReentrantCallbackGroup()
        self.create_service(
            PrecontactSpeedLease,
            "/roarm_safe/precontact_speed_lease",
            self._lease,
            callback_group=group,
        )
        self.create_service(
            SetBool,
            "/roarm_safe/set_armed",
            self._arm,
            callback_group=group,
        )
        self.create_subscription(
            JointState,
            "/roarm_safe/joint_target",
            self._target_received,
            10,
            callback_group=group,
        )
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.feedback_pub = self.create_publisher(
            String, "/roarm_safe/feedback_status", 10
        )
        self.status_pub = self.create_publisher(
            String,
            "/roarm_safe/status",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.create_timer(0.02, self._tick, callback_group=group)

    def _lease(self, request, response):
        if request.enable:
            self.lease_enable_count += 1
            if self.case == "lease_failure":
                response.success = False
                response.active = False
                response.message = "INJECTED_LEASE_REJECTION"
                return response
            self.lease_active = True
            self.lease_purpose = str(request.purpose)
            self.lease_id = str(request.lease_id)
            response.success = True
            response.active = True
            response.message = "FAKE_DYNAMIC_LEASE_ACTIVE"
            return response
        self.lease_release_count += 1
        if (
            self.lease_active
            and request.purpose == self.lease_purpose
            and request.lease_id == self.lease_id
        ):
            self.lease_active = False
            self.lease_purpose = ""
            self.lease_id = ""
            response.success = True
            response.active = False
            response.message = "FAKE_DYNAMIC_LEASE_RELEASED"
            return response
        response.success = False
        response.active = self.lease_active
        response.message = "FAKE_LEASE_IDENTITY_MISMATCH"
        return response

    def _arm(self, request, response):
        enabled = bool(request.data)
        self.arm_history.append(enabled)
        if enabled and self.case == "ack_race":
            self.armed = True
            self.state = "ARMED"
        elif not enabled:
            self.armed = False
            self.state = "CONVERGED_DISARMED"
            self.fault = ""
            self.target = None
        response.success = True
        response.message = "FAKE_ARM_ACK"
        return response

    def _target_received(self, message: JointState) -> None:
        target = [float(value) for value in message.position]
        self.received_targets.append(target)
        self.target = target
        self.target_received_at = time.monotonic()
        if self.case != "ack_race":
            self.state = "COMMAND_ACTIVE"
            self.armed = True
        self.fault = ""

    def _tick(self) -> None:
        if self.target is not None:
            elapsed = time.monotonic() - self.target_received_at
            if self.case == "ack_race" and elapsed >= 0.16:
                self.q = list(self.target)
                self.target = None
                if len(self.received_targets) >= self.expected_waypoints:
                    self.state = "CONVERGED_DISARMED"
                    self.armed = False
                else:
                    self.state = "COMMAND_ACTIVE"
                    self.armed = True
            elif self.case == "ack_race" and elapsed >= 0.08:
                self.state = "COMMAND_ACTIVE"
                self.armed = True
            elif self.case == "success" and elapsed >= 0.08:
                self.q = list(self.target)
                self.target = None
                if len(self.received_targets) >= self.expected_waypoints:
                    self.state = "CONVERGED_DISARMED"
                    self.armed = False
                else:
                    self.state = "COMMAND_ACTIVE"
                    self.armed = True
            elif self.case == "fault" and elapsed >= 0.08:
                self.fault = "INJECTED_FAULT"
        stamp = self.get_clock().now().to_msg()
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = list(self.joint_names)
        joint.position = list(self.q)
        self.joint_pub.publish(joint)
        feedback = String()
        feedback.data = json.dumps(
            {
                "state": self.state,
                "armed": self.armed,
                "dry_run": False,
                "fault": self.fault,
                "feedback_stamp_ns": stamp_ns,
                "positions_rad": list(self.q),
                "torque_enabled": [True] * 6,
                "voltage": 12.0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.feedback_pub.publish(feedback)
        status = String()
        status.data = json.dumps(
            {
                "state": self.state,
                "armed": self.armed,
                "fault": self.fault,
                "precontact_speed_lease": {
                    "active": self.lease_active,
                    "purpose": self.lease_purpose,
                    "lease_id": self.lease_id,
                    "dynamic_exact_no_reissue": bool(
                        self.lease_active
                        and self.lease_purpose == DYNAMIC_LEASE_PURPOSE
                    ),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.status_pub.publish(status)


class ProbeClient(Node):
    def __init__(self) -> None:
        super().__init__("t102_dynamic_move_probe_client")
        group = ReentrantCallbackGroup()
        self.prepare = self.create_client(
            PreparePrecontact,
            "/roarm_t102_outbound_executor/prepare_precontact",
            callback_group=group,
        )
        self.move = ActionClient(
            self,
            PrecontactMove,
            "/roarm_t102_outbound_executor/move_to_precontact",
            callback_group=group,
        )


def _fresh_session(node: DynamicExecutor, q0: tuple[float, ...]) -> FrozenSession:
    model = node._model
    assert model is not None
    latest_joint = node._latest_joint
    assert latest_joint is not None
    now = int(node.get_clock().now().nanoseconds)
    q0_stamp_ns = (
        int(latest_joint.header.stamp.sec) * 1_000_000_000
        + int(latest_joint.header.stamp.nanosec)
    )
    fk = model.fk(q0)
    n_out = tuple(-value for value in fk.axis)
    p_button = tuple(fk.tcp[index] - 0.13 * n_out[index] for index in range(3))
    value = {
        "live_boot_epoch": "fake-live",
        "qc_boot_epoch": "fake-qc",
        "executor_boot_epoch": model.boot_epoch,
        "cart_pose_epoch": "fake-cart-parked",
        "cart_confirmed_stamp_ns": now - 100_000_000,
        "stopped_proof_mode": STOPPED_MODE,
        "target_label": "Button 1",
        "detection_id": "fake-button-1",
        "confidence": 0.99,
        "target_stamp_ns": now - 50_000_000,
        "source_frame": "fake_camera",
        "canonical_frame": "roarm_base_link",
        "p_button_base_m": p_button,
        "wall_surface_point_base_m": p_button,
        "candidate_normal_base": n_out,
        "camera_center_base_m": tuple(
            p_button[index] + 0.5 * n_out[index] for index in range(3)
        ),
        "robot_reference_base_m": tuple(
            p_button[index] + 0.2 * n_out[index] for index in range(3)
        ),
        "rgbd_identity": "fake-rgbd",
        "rgbd_stamp_ns": now - 50_000_000,
        "camera_info_identity": "fake-camera-info",
        "camera_info_hash": "e" * 64,
        "camera_info_stamp_ns": now - 50_000_000,
        "d2c_identity": "fake-d2c",
        "d2c_hash": "f" * 64,
        "d2c_stamp_ns": now - 50_000_000,
        "tf_hash": "1" * 64,
        "tf_stamp_ns": now - 50_000_000,
        "calibration_identity": model.policy["calibration_identity"],
        "calibration_hash": model.hashes["calibration_hash"],
        **model.source_bindings,
        "calibration_observation_path": model.source_bindings["calibration_source_path_absolute"],
        "calibration_observation_status": "b_section_accepted_relative_unpublished",
        "calibration_observation_validation_identity": model.source_bindings["calibration_validation_identity"],
        "calibration_acceptance_artifact": "config/calibration/accepted/probe_acceptance.json",
        "calibration_acceptance_sha256": "9" * 64,
        "accepted_camera_record_hash": model.hashes["accepted_camera_record_hash"],
        "fn_artifact_hash": model.hashes["fn_artifact_hash"],
        "fn_acceptance_hash": model.hashes["fn_acceptance_hash"],
        "standby_acceptance_hash": model.hashes["standby_acceptance_hash"],
        "source_registry_hash": model.hashes["source_registry_hash"],
        "kinematic_model_hash": model.hashes["kinematic_model_hash"],
        "collision_proxy_hash": model.hashes["collision_proxy_hash"],
        "tool_model_hash": model.hashes["tool_model_hash"],
        "template_hash": model.hashes["template_hash"],
        "q0_rad": q0,
        "q0_stamp_ns": q0_stamp_ns,
        "q0_hash": canonical_hash(list(q0)),
    }
    value["session_id"] = canonical_hash(value)
    return FrozenSession.from_mapping(value)


def run(case: str) -> dict:
    if case not in CASES:
        raise ValueError(case)
    os.environ["LIFT_ROBOT_DYNAMIC_MOVE_BUILD_TOKEN"] = (
        "ENABLE_T102_DYNAMIC_EXACT_PLAN_MOVE"
    )
    rclpy.init()
    executor_node = DynamicExecutor()
    if executor_node._model is None or executor_node._move_policy is None:
        raise RuntimeError(executor_node._failure or "executor model did not load")
    executor_node._model.policy["move_execution_approved"] = True
    executor_node.set_parameters(
        [rclpy.parameter.Parameter("move_runtime_enabled", value=True)]
    )
    if case in {"timeout", "cancel"}:
        executor_node._move_policy = replace(
            executor_node._move_policy, waypoint_timeout_s=0.35
        )
    q0 = tuple(float(value) for value in executor_node._model.policy["standard_standby_rad"])
    joint_names = tuple(executor_node._model.policy["joint_names"])
    fake = FakeSafeDriver(q0, joint_names, case)
    client = ProbeClient()
    ros_executor = MultiThreadedExecutor(num_threads=8)
    for node in (executor_node, fake, client):
        ros_executor.add_node(node)
    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()
    try:
        if not client.prepare.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Prepare service unavailable")
        if not client.move.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Move Action unavailable")
        deadline = time.monotonic() + 5.0
        while (
            executor_node._latest_joint is None
            or not executor_node._safe_feedback
            or not executor_node._safe_status
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        session = _fresh_session(executor_node, q0)
        request = PreparePrecontact.Request()
        request.dynamic_session_json = canonical_json(asdict(session))
        request.dynamic_session_sha256 = canonical_hash(asdict(session))
        request.ui_request_id = f"fake-prepare-{case}-{time.time_ns()}"
        prepared = _wait_future(client.prepare.call_async(request))
        if not prepared.accepted:
            raise RuntimeError(f"Prepare rejected: {prepared.result_code} {prepared.reason}")
        plan = json.loads(prepared.summary_json)
        fake.expected_waypoints = len(plan["waypoints_rad"])
        goal = PrecontactMove.Goal()
        goal.authorization_id = prepared.authorization_id
        goal.plan_hash = prepared.plan_hash
        goal.confirmation_text = MOVE_CONFIRMATION_TOKEN
        goal.session_id = prepared.session_id
        handle = _wait_future(client.move.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError("Move goal rejected")
        if case == "cancel":
            deadline = time.monotonic() + 5.0
            while not fake.received_targets and time.monotonic() < deadline:
                time.sleep(0.02)
            _wait_future(handle.cancel_goal_async())
        result = _wait_future(handle.get_result_async()).result
        exact_targets = fake.received_targets == plan["waypoints_rad"]
        output = {
            "case": case,
            "action_success": bool(result.success),
            "terminal_code": result.terminal_code,
            "terminal_reason": result.terminal_reason,
            "waypoint_count": len(plan["waypoints_rad"]),
            "target_publish_count": len(fake.received_targets),
            "exact_frozen_targets": exact_targets,
            "terminal_publishers": int(result.terminal_joint_target_publishers),
            "terminal_lease_active": bool(result.terminal_precontact_lease_active),
            "terminal_armed": bool(result.terminal_armed),
            "teardown_complete": bool(result.teardown_complete),
            "lease_enable_count": fake.lease_enable_count,
            "lease_release_count": fake.lease_release_count,
            "fake_lease_active": fake.lease_active,
            "fake_armed": fake.armed,
        }
        if case in {"success", "ack_race"}:
            assert result.success, output
            assert result.terminal_code == "DYNAMIC_PRECONTACT_STOPPED_DISARMED", output
            assert exact_targets, output
            assert len(fake.received_targets) == len(plan["waypoints_rad"]), output
        else:
            assert not result.success, output
            if case in {"timeout", "fault", "cancel"}:
                assert len(fake.received_targets) == 1, output
            if case == "lease_failure":
                assert len(fake.received_targets) == 0, output
        assert result.terminal_joint_target_publishers == 0, output
        assert not result.terminal_precontact_lease_active, output
        assert not result.terminal_armed, output
        assert result.teardown_complete, output
        assert not fake.lease_active, output
        assert not fake.armed, output
        if case == "success":
            # Model the operator's separate Go Standby action, then prove the
            # same long-running Live/QC/executor graph accepts a brand-new
            # session and plan without retaining the consumed frozen plan.
            fake.received_targets = []
            fake.q = list(q0)
            fake.state = "CONVERGED_DISARMED"
            fake.armed = False
            time.sleep(0.12)
            session2 = _fresh_session(executor_node, q0)
            request2 = PreparePrecontact.Request()
            request2.dynamic_session_json = canonical_json(asdict(session2))
            request2.dynamic_session_sha256 = canonical_hash(asdict(session2))
            request2.ui_request_id = f"fake-prepare-repeat-{time.time_ns()}"
            prepared2 = _wait_future(client.prepare.call_async(request2))
            if not prepared2.accepted:
                raise RuntimeError(
                    f"repeat Prepare rejected: {prepared2.result_code} {prepared2.reason}"
                )
            plan2 = json.loads(prepared2.summary_json)
            fake.expected_waypoints = len(plan2["waypoints_rad"])
            goal2 = PrecontactMove.Goal()
            goal2.authorization_id = prepared2.authorization_id
            goal2.plan_hash = prepared2.plan_hash
            goal2.confirmation_text = MOVE_CONFIRMATION_TOKEN
            goal2.session_id = prepared2.session_id
            handle2 = _wait_future(client.move.send_goal_async(goal2))
            if not handle2.accepted:
                raise RuntimeError("repeat Move goal rejected")
            result2 = _wait_future(handle2.get_result_async()).result
            repeat_exact = fake.received_targets == plan2["waypoints_rad"]
            assert result2.success and repeat_exact, output
            assert len(fake.received_targets) == len(plan2["waypoints_rad"]), output
            assert result2.terminal_joint_target_publishers == 0, output
            assert not result2.terminal_precontact_lease_active, output
            assert not result2.terminal_armed and result2.teardown_complete, output
            output["repeat_cycle_success"] = True
            output["repeat_target_publish_count"] = len(fake.received_targets)
            output["repeat_exact_frozen_targets"] = repeat_exact
        return output
    finally:
        ros_executor.shutdown(timeout_sec=2.0)
        for node in (client, fake, executor_node):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    args, _ros_args = parser.parse_known_args()
    print(json.dumps(run(args.case), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
