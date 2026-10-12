"""ROS shell for the fail-closed outbound-only pre-contact action.

The node starts as a read-only state observer plus Prepare service/Action
server.  It deliberately has no real motion executor: no joint-target
publisher and no ``/roarm_safe/set_armed`` client are created at startup or in
the current action path.  Until a separately reviewed executor is added, every
otherwise-admissible action ends as ``NOT_IMPLEMENTED_INERT``.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from typing import Any

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from elevator_interfaces.action import PrecontactMove
from elevator_interfaces.msg import M0Target3D
from elevator_interfaces.srv import PreparePrecontact
from .precontact_execution_core import (
    AdmissionGates,
    DEFAULT_RELEASE_MANIFEST,
    NotImplementedExecutor,
    PrecontactExecutionCore,
    PrecontactError,
    canonical_json,
)
from .precontact_planning_core import (
    DynamicPrepareTransaction,
    FrozenM0Cache,
    NotImplementedPlanner,
    PlanningBlocked,
    canonicalize_button1_m0_freeze_request,
)


class PrecontactAdapter(Node):
    """Read-only admission node with a deliberately unimplemented executor."""

    def __init__(self) -> None:
        super().__init__("roarm_precontact_adapter")
        self._lock = threading.Lock()
        self._core_lock = threading.Lock()
        self.declare_parameter("execution_enabled", False)
        self.declare_parameter("physical_authorized", False)
        self.declare_parameter("m0_target_topic", "/elevator/m0/perception_target_3d")
        self.declare_parameter("release_manifest_json", canonical_json(DEFAULT_RELEASE_MANIFEST))
        self._latest_joint_state: JointState | None = None
        self._latest_feedback: dict[str, Any] | None = None
        self._feedback_at: float | None = None
        self._configuration_error = ""
        # This value changes only on process restart; dashboards use it with
        # sequence checks to reject status replays/substitutions.
        self._status_instance_nonce = secrets.token_hex(16)
        self._status_sequence = 0
        manifest = self._load_manifest()
        try:
            self._core = PrecontactExecutionCore(
                manifest,
                executor=NotImplementedExecutor(),
            )
        except PrecontactError as exc:
            # A malformed release must degrade to the known inert release, not
            # accidentally make a future executor less restrictive.
            self._configuration_error = f"{self._configuration_error}; release rejected: {exc}".strip("; ")
            self._core = PrecontactExecutionCore(
                DEFAULT_RELEASE_MANIFEST,
                executor=NotImplementedExecutor(),
            )
        self._m0_cache = FrozenM0Cache()
        self._planner = NotImplementedPlanner()
        self._dynamic_prepare = DynamicPrepareTransaction(
            m0_cache=self._m0_cache,
            planner=self._planner,
        )

        # Read-only ROS entities only.  This status publisher carries state; it
        # is never a command channel.
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._status_pub = self.create_publisher(String, "/roarm_precontact/status", status_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(
            String,
            "/roarm_safe/feedback_status",
            self._on_feedback,
            status_qos,
        )
        self.create_subscription(
            M0Target3D,
            str(self.get_parameter("m0_target_topic").value),
            self._on_m0_target,
            status_qos,
        )
        self._prepare_service = self.create_service(
            PreparePrecontact,
            "~/prepare_precontact",
            self._on_prepare,
        )
        self._action_server = ActionServer(
            self,
            PrecontactMove,
            "~/move_to_precontact",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
        )
        # Status is intentionally periodic because the Live/QC dashboards are
        # late-joining observers.  It is still a read-only topic and does not
        # create a motion publisher or a safe-driver client.
        self._status_timer = self.create_timer(0.5, self._publish_periodic_status)
        self._publish_status("INERT", "real motion executor is NOT_IMPLEMENTED")

    def _load_manifest(self) -> dict[str, Any]:
        raw = str(self.get_parameter("release_manifest_json").value)
        try:
            manifest = json.loads(raw)
            if not isinstance(manifest, dict):
                raise ValueError("manifest is not an object")
            return manifest
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._configuration_error = f"release manifest invalid: {exc}"
            return dict(DEFAULT_RELEASE_MANIFEST)

    def _on_joint_state(self, message: JointState) -> None:
        with self._lock:
            self._latest_joint_state = message

    def _on_feedback(self, message: String) -> None:
        try:
            decoded = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(decoded, dict):
            with self._lock:
                self._latest_feedback = decoded
                self._feedback_at = time.monotonic()

    @staticmethod
    def _m0_stamp_ns(message: M0Target3D) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )

    def _on_m0_target(self, message: M0Target3D) -> None:
        """Cache trusted callback bytes; never publish or accept its token."""

        # A newer unusable callback invalidates any older private slot.  Never
        # let Prepare fall back to a previous target merely because the newest
        # observation cannot be canonicalized.
        self._dynamic_prepare.clear()
        observation = message.observation
        surface = observation.surface_geometry
        source_stamp_ns = self._m0_stamp_ns(message)
        camera_info_stamp_ns = int(surface.camera_info_stamp_ns)
        skew_ms = (
            abs(source_stamp_ns - camera_info_stamp_ns) / 1_000_000.0
            if camera_info_stamp_ns > 0
            else 1_000_000_000.0
        )
        fields = {
            "source_stamp_ns": source_stamp_ns,
            "source_frame_id": str(surface.source_frame_id or message.header.frame_id),
            "target_label": str(message.target_label),
            "effective_label": str(message.effective_label),
            "detection_id": str(message.detection_id),
            "semantic_track_id": str(message.semantic_track_id),
            "status": str(message.status),
            "perception_ready": bool(message.perception_ready),
            "motion_authorized": bool(message.motion_authorized),
            "point_camera_m": [
                float(surface.surface_point_x_m),
                float(surface.surface_point_y_m),
                float(surface.surface_point_z_m),
            ],
            "normal_camera": [
                float(surface.panel_outward_normal_x),
                float(surface.panel_outward_normal_y),
                float(surface.panel_outward_normal_z),
            ],
            "confidence": float(observation.detector_confidence),
            "rgb_depth_skew_ms": skew_ms,
            "depth_valid": bool(observation.depth_valid),
            "surface_estimate_valid": bool(surface.estimate_valid),
            "surface_provenance_complete": bool(surface.provenance_complete),
            "calibration_release_id": str(surface.section_b_calibration_id),
            # M0Target3D does not carry accepted Fn/P evidence, so this callback
            # must never promote its geometric normal to execution authority.
            "normal_direction_status": "UNVERIFIED",
            "input_normal_was_flipped": False,
            "dot_n_out_p_minus_a": 0.0,
            "camera_side_dot": 0.0,
        }
        try:
            canonical = canonicalize_button1_m0_freeze_request(fields)
            self._dynamic_prepare.observe(
                canonical,
                receive_monotonic=time.monotonic(),
            )
        except PlanningBlocked as exc:
            self._publish_status("M0_NOT_CACHED", str(exc))

    def _joint_target_publisher_count(self) -> int | None:
        try:
            return int(self.count_publishers("/roarm_safe/joint_target"))
        except Exception:  # noqa: BLE001 - an unreadable graph is unknown
            return None

    def _admission_gates(self) -> AdmissionGates:
        """Return runtime facts without creating a motion-capable entity.

        The present implementation intentionally keeps all physical gates
        closed.  A later executor must replace this method with exact-stamp
        feedback/standby/source checks, not merely change a parameter.
        """

        with self._lock:
            feedback = dict(self._latest_feedback or {})
            feedback_at = self._feedback_at
        age = None if feedback_at is None else time.monotonic() - feedback_at
        fault_clear = not bool(feedback.get("fault"))
        torque_all_enabled = feedback.get("torque_enabled") == [True] * 6
        voltage = feedback.get("voltage")
        voltage_safe = isinstance(voltage, (int, float)) and not isinstance(voltage, bool) and 11.0 <= float(voltage) <= 13.0
        driver_disarmed = feedback.get("armed") is False and str(feedback.get("state") or "").upper().startswith("DISARMED")
        publisher_count = self._joint_target_publisher_count()
        zero_motion_publishers = publisher_count == 0
        return AdmissionGates(
            execution_enabled=bool(self.get_parameter("execution_enabled").value),
            execution_ready=self._core.release.execution_ready,
            physical_authorized=bool(self.get_parameter("physical_authorized").value),
            feedback_fresh=age is not None and age <= 1.0,
            feedback_aligned=False,
            safe_driver_single_source=False,
            dry_run_false=feedback.get("dry_run") is False,
            fault_clear=fault_clear,
            torque_all_enabled=torque_all_enabled,
            voltage_safe=voltage_safe,
            driver_disarmed=driver_disarmed,
            at_configured_standby=False,
            zero_motion_publishers=zero_motion_publishers,
            auto_torque_recovery_disabled=False,
            workspace_keepout_passed=False,
            route_outbound_only=self._core.release.route is not None,
            route_t101_only=self._core.release.command_mode == "T101_SINGLE_JOINT_ONLY",
        )

    def _on_prepare(self, request, response):
        with self._core_lock:
            gates = self._admission_gates()
            outcome = self._dynamic_prepare.prepare(
                request.canonical_target_snapshot_json,
                request.target_snapshot_sha256,
                request.ui_request_id,
                gates,
                now_ns=time.time_ns(),
                now_monotonic=time.monotonic(),
            )
            self._core = outcome.core
            result = outcome.prepare_result
            response.accepted = outcome.accepted
            response.authorization_id = "" if result is None else result.authorization_id
            response.ttl_ms = 0 if result is None else result.ttl_ms
            response.release_manifest_sha256 = (
                "" if result is None else result.release_manifest_sha256
            )
            response.route_artifact_sha256 = (
                "" if result is None else result.route_artifact_sha256
            )
            response.m0_frozen_artifact_sha256 = (
                "" if result is None else result.m0_frozen_artifact_sha256
            )
            response.planned_target_snapshot_sha256 = outcome.target_snapshot_sha256
            response.reason = outcome.reason
            response.state = self._core.state.value
            event = "PREPARED_NO_MOTION" if outcome.accepted else "REJECTED_NO_MOTION"
        self._publish_status(event, outcome.reason)
        return response

    def _goal(self, goal_request):
        try:
            with self._core_lock:
                accepted, reason = self._core.reserve_goal(
                    goal_request.authorization_id,
                    goal_request.target_snapshot_sha256,
                    goal_request.confirmation_text,
                    now_monotonic=time.monotonic(),
                )
        except Exception as exc:  # noqa: BLE001 - malformed goals fail closed
            accepted, reason = False, f"goal preflight failed: {exc}"
        if not accepted:
            self._publish_status("GOAL_REJECTED_NO_MOTION", reason)
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel(self, _goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        request = goal_handle.request
        gates = self._admission_gates()
        with self._core_lock:
            result = self._core.admit(
                request.authorization_id,
                request.target_snapshot_sha256,
                request.confirmation_text,
                gates,
                now_monotonic=time.monotonic(),
            )
        feedback = PrecontactMove.Feedback()
        feedback.state = self._core.state.value
        feedback.gate = result.code.value
        feedback.step_index = 0
        feedback.step_count = 0
        feedback.safe_driver_state = "NOT_IMPLEMENTED"
        feedback.route_artifact_sha256 = (
            self._core.release.route.artifact_sha256
            if self._core.release.route is not None
            else ""
        )
        goal_handle.publish_feedback(feedback)
        if result.success:
            # This branch is unreachable with NotImplementedExecutor.  Keep the
            # explicit abort to prevent a future partial executor from claiming
            # a completed move without terminal feedback evidence.
            with self._core_lock:
                result = self._core.abort("executor returned without convergence evidence")
        action_result = PrecontactMove.Result()
        action_result.success = result.success
        action_result.terminal_code = result.code.value
        action_result.terminal_reason = result.reason
        action_result.state = self._core.state.value
        action_result.target_snapshot_sha256 = request.target_snapshot_sha256
        action_result.release_manifest_sha256 = self._core.release.manifest_hash
        action_result.route_artifact_sha256 = feedback.route_artifact_sha256
        action_result.m0_frozen_artifact_sha256 = (
            self._core.release.route.m0_frozen_artifact_sha256
            if self._core.release.route is not None
            else ""
        )
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        self._publish_status(action_result.terminal_code, action_result.terminal_reason)
        return action_result

    def _publish_status(self, event: str, reason: str) -> None:
        publisher_count = self._joint_target_publisher_count()
        self._status_sequence += 1
        message = String()
        message.data = json.dumps(
            {
                "schema": "roarm_precontact_status_v1",
                "instance_nonce": self._status_instance_nonce,
                "status_sequence": self._status_sequence,
                "published_at_unix_ns": time.time_ns(),
                "published_at_monotonic_ns": time.monotonic_ns(),
                "event": event,
                "state": self._core.state.value,
                "reason": reason,
                "execution_enabled": bool(self.get_parameter("execution_enabled").value),
                "execution_ready": self._core.release.execution_ready,
                # This installed executor has no publisher/client path.  A
                # consumer must require this false flag, rather than treating
                # a server's presence or a parameter alone as permission.
                "accepting_precontact": False,
                "executor": "NOT_IMPLEMENTED",
                "joint_target_publishers": (
                    publisher_count if publisher_count is not None else "unknown"
                ),
                "arm_clients": 0,
                "configuration_error": self._configuration_error,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self._status_pub.publish(message)

    def _publish_periodic_status(self) -> None:
        """Refresh inert status for late-joining dashboards; no command path."""

        self._publish_status("INERT", "real motion executor is NOT_IMPLEMENTED")

    def destroy_node(self) -> bool:
        self._action_server.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PrecontactAdapter()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
