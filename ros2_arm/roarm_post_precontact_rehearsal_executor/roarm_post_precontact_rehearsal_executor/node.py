#!/usr/bin/env python3
"""Gated owner for one frozen 70 mm positional contact approach and PRE return."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time


def _repo_root() -> Path:
    configured = os.environ.get("LIFT_ROBOT_ROOT", "").strip()
    candidates = ([Path(configured)] if configured else []) + list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "tools/press/stage_b_rehearsal_core.py").is_file():
            return candidate.resolve()
    raise RuntimeError("Lift_Robot root not found")


ROOT = _repo_root()
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from press.stage_b_rehearsal_core import (  # noqa: E402
    LEASE_PURPOSE,
    SAFE_IDLE_EXECUTION_STATES,
    APPROACH_DISTANCE_M,
    APPROACH_GEOMETRY_TOLERANCE_M,
    REQUIRED_CONFIRMATION_TOKEN,
    FrozenStageBPlan,
    PreTerminalReceipt,
    StageBBlocked,
    StageBExactExecution,
    StageBExecutionConfig,
    StageBGoal,
    StageBSafetySample,
    canonical_hash,
    canonical_json,
    prepare_stage_b_plan,
    verify_stage_b_plan,
)
from press.t102_dynamic_precontact_core import FrozenModel  # noqa: E402

import rclpy  # noqa: E402
from elevator_interfaces.action import StageBRehearsal  # noqa: E402
from elevator_interfaces.srv import PrepareStageBRehearsal, PrecontactSpeedLease  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.task import Future  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402


class StageBExecutor(Node):
    """Prepare owns zero command resources; Action lazily owns one episode."""

    LEASE_SERVICE = "/roarm_safe/precontact_speed_lease"
    ARM_SERVICE = "/roarm_safe/set_armed"
    JOINT_TARGET_TOPIC = "/roarm_safe/joint_target"
    STAGE_A_STATUS_TOPIC = "/roarm_t102_outbound_executor/status"
    QC_STATUS_TOPIC = "/roarm_precontact/quick_control_status"

    def __init__(self) -> None:
        super().__init__("roarm_post_precontact_rehearsal_executor")
        self.declare_parameter("runtime_enabled", False)
        self.declare_parameter("build_enabled", False)
        self.declare_parameter("contact_authorized", False)
        self.declare_parameter("press_authorized", False)
        self.declare_parameter("force_control", False)
        self.declare_parameter(
            "model_policy_path",
            str(ROOT / "config/press/releases/t102_dynamic_precontact_policy_release_20260818.json"),
        )
        self.declare_parameter(
            "stage_b_policy_path",
            str(ROOT / "config/press/releases/stage_b_contact_approach_out_back_release_v1.json"),
        )
        self._runtime_group = ReentrantCallbackGroup()
        self._prepare_group = MutuallyExclusiveCallbackGroup()
        self._lock = threading.RLock()
        self._sleep_waiters: list[tuple[float, Future]] = []
        self._instance_id = secrets.token_hex(16)
        self._state = "IDLE"
        self._failure = ""
        self._plan: FrozenStageBPlan | None = None
        self._receipt: PreTerminalReceipt | None = None
        self._authorization_id = ""
        self._prepared_qc_binding: dict = {}
        self._seen_request_ids: set[str] = set()
        self._seen_receipt_hashes: set[str] = set()
        self._consumed_plan_hashes: set[str] = set()
        self._goal_reserved = False
        self._machine: StageBExactExecution | None = None
        self._safe_status: dict = {}
        self._safe_status_at: float | None = None
        self._safe_feedback: dict = {}
        self._safe_feedback_at: float | None = None
        self._stage_a_status: dict = {}
        self._stage_a_status_at: float | None = None
        self._qc_status: dict = {}
        self._qc_status_at: float | None = None
        self._latest_joint: JointState | None = None
        self._lease_client = None
        self._arm_client = None
        self._target_pub = None
        self._lease_id = ""
        self._lease_active_local = False
        self._lease_uncertain = False
        self._lease_enable_definitive = False
        self._resources = {"publisher": 0, "lease_client": 0, "arm_client": 0, "action": 0}
        try:
            self._model = FrozenModel.load_once(
                ROOT, str(self.get_parameter("model_policy_path").value)
            )
            self._policy = self._load_stage_b_policy(
                str(self.get_parameter("stage_b_policy_path").value)
            )
            self._execution_config = StageBExecutionConfig(
                max_feedback_age_s=float(self._model.policy["move_feedback_max_age_s"]),
                endpoint_tolerance_rad=float(self._model.policy["move_waypoint_tolerance_rad"]),
                command_ack_timeout_s=0.75,
                waypoint_timeout_s=float(self._model.policy["move_waypoint_timeout_s"]),
                min_voltage_v=float(self._model.policy["move_min_voltage_v"]),
                max_voltage_v=float(self._model.policy["move_max_voltage_v"]),
                encoder_step_rad=float(self._model.policy["encoder_step_rad"]),
            )
            self._execution_config.validate()
        except Exception as exc:  # boot-frozen failure is terminal until restart
            self._model = None
            self._policy = {}
            self._execution_config = StageBExecutionConfig()
            self._failure = f"MODEL_OR_POLICY_LOAD_FAILED: {exc}"
        self._status_pub = self.create_publisher(String, "~/status", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint, 20, callback_group=self._runtime_group)
        self.create_subscription(String, "/roarm_safe/status", self._on_safe_status, 20, callback_group=self._runtime_group)
        self.create_subscription(String, "/roarm_safe/feedback_status", self._on_safe_feedback, 20, callback_group=self._runtime_group)
        self.create_subscription(String, self.STAGE_A_STATUS_TOPIC, self._on_stage_a_status, 20, callback_group=self._runtime_group)
        self.create_subscription(String, self.QC_STATUS_TOPIC, self._on_qc_status, 20, callback_group=self._runtime_group)
        self._prepare_service = self.create_service(
            PrepareStageBRehearsal, "~/prepare_stage_b", self._prepare,
            callback_group=self._prepare_group,
        )
        self._action_server = ActionServer(
            self, StageBRehearsal, "~/execute_stage_b",
            execute_callback=self._execute, goal_callback=self._goal,
            cancel_callback=self._cancel, callback_group=self._runtime_group,
        )
        self.create_timer(0.5, self._publish_status, callback_group=self._runtime_group)
        self.create_timer(0.005, self._wake_waiters, callback_group=self._runtime_group)
        self._publish_status()

    @staticmethod
    def _load_stage_b_policy(path: str) -> dict:
        value = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        required = {
            "schema": "t102_stage_b_contact_approach_out_back_release_v1",
            "no_contact": False,
            "press": False,
            "approach_geometry_tolerance_m": APPROACH_GEOMETRY_TOLERANCE_M,
            "return_policy": "EXACT_FROZEN_START_Q",
            "automatic_retry": False,
            "automatic_compensation": False,
            "forward_once_only": True,
            "zero_resource_prepare": True,
            "lease_purpose": LEASE_PURPOSE,
            "required_confirmation_token": REQUIRED_CONFIRMATION_TOKEN,
            "force_control": False,
            "zero_dwell": True,
            "button_activation_claimed": False,
            "positional_contact_capsules": ["tool"],
        }
        if not isinstance(value, dict):
            raise StageBBlocked("STAGE_B_POLICY_INVALID", "object")
        for key, expected in required.items():
            if value.get(key) != expected:
                raise StageBBlocked("STAGE_B_POLICY_INVALID", key)
        if any(type(value.get(key)) is not bool for key in (
            "runtime_enabled", "build_enabled", "contact_authorized", "press_authorized"
        )):
            raise StageBBlocked("STAGE_B_POLICY_INVALID", "enable gates")
        return value

    def _on_joint(self, message: JointState) -> None:
        with self._lock:
            self._latest_joint = message

    def _store_json(self, message: String, field: str, stamp_field: str) -> None:
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict):
                return
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        with self._lock:
            setattr(self, field, value)
            setattr(self, stamp_field, time.monotonic())

    def _on_safe_status(self, message: String) -> None:
        self._store_json(message, "_safe_status", "_safe_status_at")

    def _on_safe_feedback(self, message: String) -> None:
        self._store_json(message, "_safe_feedback", "_safe_feedback_at")

    def _on_stage_a_status(self, message: String) -> None:
        self._store_json(message, "_stage_a_status", "_stage_a_status_at")

    def _on_qc_status(self, message: String) -> None:
        self._store_json(message, "_qc_status", "_qc_status_at")

    def _runtime_gate(self) -> tuple[bool, str]:
        if self._model is None or not self._policy:
            return False, "MODEL_OR_POLICY_UNAVAILABLE"
        if not bool(self.get_parameter("build_enabled").value) or self._policy.get("build_enabled") is not True:
            return False, "BUILD_DISABLED"
        if not bool(self.get_parameter("runtime_enabled").value) or self._policy.get("runtime_enabled") is not True:
            return False, "RUNTIME_DISABLED"
        if not bool(self.get_parameter("contact_authorized").value) or self._policy.get("contact_authorized") is not True:
            return False, "CONTACT_AUTHORIZATION_DISABLED"
        if bool(self.get_parameter("press_authorized").value) is not bool(self._policy.get("press_authorized")):
            return False, "PRESS_AUTHORIZATION_MISMATCH"
        if self.get_parameter("force_control").value is not False or self._policy.get("force_control") is not False:
            return False, "FORCE_CONTROL_OUT_OF_SCOPE"
        return True, "STAGE_B_EXECUTION_READY"

    def _current_q(self) -> tuple[float, ...]:
        if self._model is None or self._latest_joint is None:
            raise StageBBlocked("CURRENT_Q_UNAVAILABLE")
        by_name = dict(zip(self._latest_joint.name, self._latest_joint.position))
        try:
            q = tuple(float(by_name[name]) for name in self._model.policy["joint_names"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StageBBlocked("CURRENT_Q_UNAVAILABLE", "joint order") from exc
        if len(q) != 6 or any(not math.isfinite(value) for value in q):
            raise StageBBlocked("CURRENT_Q_UNAVAILABLE", "six finite joints")
        return q

    @staticmethod
    def _strict_feedback(value: dict) -> tuple:
        positions = value.get("positions_rad")
        torque = value.get("torque_enabled")
        state = value.get("state")
        armed = value.get("armed")
        dry_run = value.get("dry_run")
        fault = value.get("fault")
        accepted = value.get("dynamic_exact_accepted_target_positions")
        accepted_sequence = value.get("dynamic_exact_accepted_command_sequence")
        advance_ready = value.get("dynamic_exact_advance_ready")
        fields_ok = (
            type(positions) is list and len(positions) == 6
            and type(torque) is list and len(torque) == 6 and all(type(item) is bool for item in torque)
            and type(value.get("voltage")) in {int, float}
            and type(value.get("feedback_stamp_ns")) is int
            and type(state) is str and bool(state)
            and type(armed) is bool
            and type(dry_run) is bool
            and type(fault) is str
            and type(accepted) is list and len(accepted) in {0, 6}
            and type(accepted_sequence) is int
            and type(advance_ready) is bool
        )
        numeric = list(positions or []) + [value.get("voltage")] + list(accepted or [])
        if not fields_ok or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in numeric):
            raise StageBBlocked("MOVE_FEEDBACK_MALFORMED")
        if value["feedback_stamp_ns"] <= 0 or accepted_sequence < 0:
            raise StageBBlocked("MOVE_FEEDBACK_MALFORMED")
        return (
            positions,
            torque,
            float(value["voltage"]),
            int(value["feedback_stamp_ns"]),
            state,
            armed,
            dry_run,
            fault,
            accepted,
            int(accepted_sequence),
            advance_ready,
        )

    @staticmethod
    def _strict_status(value: dict) -> tuple:
        lease = value.get("precontact_speed_lease")
        if (
            type(lease) is not dict
            or type(lease.get("active")) is not bool
            or type(lease.get("purpose")) is not str
            or type(lease.get("dynamic_exact_no_reissue")) is not bool
        ):
            raise StageBBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN")
        return (
            bool(lease["active"]),
            str(lease["purpose"]),
            bool(lease["dynamic_exact_no_reissue"]),
        )

    def _sample(self) -> StageBSafetySample:
        now = time.monotonic()
        with self._lock:
            feedback = dict(self._safe_feedback)
            status = dict(self._safe_status)
            feedback_at = self._safe_feedback_at
            status_at = self._safe_status_at
        (
            positions, torque, voltage, feedback_stamp_ns,
            state, armed, dry_run, fault,
            accepted, accepted_sequence, advance_ready,
        ) = self._strict_feedback(feedback)
        lease_active, lease_purpose, dynamic_exact_no_reissue = (
            self._strict_status(status)
        )
        source_age = (self.get_clock().now().nanoseconds - feedback_stamp_ns) / 1e9
        return StageBSafetySample(
            tuple(float(item) for item in positions),
            math.inf if feedback_at is None else now - feedback_at,
            str(state), bool(armed), bool(dry_run), str(fault), tuple(torque), voltage,
            int(self.count_publishers(self.JOINT_TARGET_TOPIC)), lease_active,
            lease_purpose, dynamic_exact_no_reissue,
            math.inf if status_at is None else now - status_at,
            float(feedback_at if feedback_at is not None else -math.inf), float(source_age),
            accepted_sequence, tuple(float(item) for item in accepted), bool(advance_ready),
            float(feedback_at if feedback_at is not None else -math.inf),
        )

    def _prepare_graph_gate(
        self, receipt: PreTerminalReceipt, *, expected_stage_b_phase: str = "PREPARE_IN_FLIGHT"
    ) -> dict:
        sample = self._sample()
        sample.validate_owner(self._execution_config, expected_publishers=0, expected_lease=False, require_disarmed=True)
        now = time.monotonic()
        with self._lock:
            stage_a, stage_a_at = dict(self._stage_a_status), self._stage_a_status_at
            qc, qc_at = dict(self._qc_status), self._qc_status_at
        if stage_a_at is None or now - stage_a_at > 1.5 or qc_at is None or now - qc_at > 1.5:
            raise StageBBlocked("RESOURCE_QC_STATUS_STALE")
        if (
            stage_a.get("schema") not in {"t102_dynamic_executor_status_v1", "t102_dynamic_executor_status_v2"}
            or str(stage_a.get("state") or "").upper() not in {"IDLE", "MOVE_COMPLETE_DISARMED", "FAILED_DISARMED"}
            or int(stage_a.get("joint_target_publishers", -1)) != 0
            or stage_a.get("precontact_lease_active") is not False
            or stage_a.get("terminal_armed") is not False
        ):
            raise StageBBlocked("CURRENT_RESOURCE_CLEANUP_UNPROVEN")
        resources = stage_a.get("resource_counts")
        if not isinstance(resources, dict) or not resources or any(type(value) is not int or value != 0 for value in resources.values()):
            raise StageBBlocked("CURRENT_RESOURCE_CLEANUP_UNPROVEN", "resources")
        if (
            qc.get("schema") != "roarm_precontact_qc_status_v3"
            or str(qc.get("stage_b_phase") or "").upper() != expected_stage_b_phase
            or qc.get("stage_b_controls_locked") is not True
            or str(qc.get("stage_b_panel_press_receipt_sha256") or "") != receipt.receipt_sha256
            or str(qc.get("stage_b_panel_geometry_sha256") or "") != receipt.panel_geometry_sha256
            or qc.get("precontact_owner") is not True
            or int(qc.get("qc_status_publisher_count", -1)) != 1
        ):
            raise StageBBlocked("QC_CURRENT_PANEL_BINDING_UNPROVEN")
        graph = {
            self.STAGE_A_STATUS_TOPIC: 1,
            self.QC_STATUS_TOPIC: 1,
            "/roarm_post_precontact_rehearsal_executor/status": 1,
            self.JOINT_TARGET_TOPIC: 0,
        }
        for topic, expected in graph.items():
            if int(self.count_publishers(topic)) != expected:
                raise StageBBlocked("OWNER_GRAPH_NOT_UNIQUE", topic)
        try:
            binding = {
                "owner_instance_nonce": str(qc["instance_nonce"]),
                "stage_b_instance_nonce": str(qc["stage_b_instance_nonce"]),
                "status_sequence": int(qc["status_sequence"]),
                "stage_b_status_sequence": int(qc["stage_b_status_sequence"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise StageBBlocked("QC_CURRENT_PANEL_BINDING_UNPROVEN", "owner binding") from exc
        if (
            not binding["owner_instance_nonce"]
            or not binding["stage_b_instance_nonce"]
            or binding["status_sequence"] < 0
            or binding["stage_b_status_sequence"] < 0
        ):
            raise StageBBlocked("QC_CURRENT_PANEL_BINDING_UNPROVEN", "owner binding")
        return binding

    def _action_graph_gate(self, receipt: PreTerminalReceipt, plan_hash: str) -> None:
        """Admit resources only from the exact fresh QC confirm transition."""

        self._prepare_graph_gate(receipt, expected_stage_b_phase="ACTION_IN_FLIGHT")
        with self._lock:
            qc = dict(self._qc_status)
            qc_at = self._qc_status_at
            prepared = dict(self._prepared_qc_binding)
        if qc_at is None or time.monotonic() - qc_at > 1.5 or not prepared:
            raise StageBBlocked("STAGE_B_ACTION_QC_STATUS_STALE")
        try:
            current_status_sequence = int(qc["status_sequence"])
            current_stage_b_sequence = int(qc["stage_b_status_sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StageBBlocked("STAGE_B_ACTION_QC_BINDING_INVALID") from exc
        if (
            str(qc.get("stage_b_phase") or "").upper() != "ACTION_IN_FLIGHT"
            or qc.get("stage_b_controls_locked") is not True
            or qc.get("stage_b_stop_requested") is not False
            or qc.get("precontact_owner") is not True
            or int(qc.get("qc_status_publisher_count", -1)) != 1
            or str(qc.get("instance_nonce") or "") != prepared["owner_instance_nonce"]
            or str(qc.get("stage_b_instance_nonce") or "")
            != prepared["stage_b_instance_nonce"]
            or current_status_sequence <= int(prepared["status_sequence"])
            or current_stage_b_sequence <= int(prepared["stage_b_status_sequence"])
            or str(qc.get("stage_b_plan_hash") or "") != plan_hash
            or str(qc.get("stage_b_panel_press_receipt_sha256") or "")
            != receipt.receipt_sha256
            or str(qc.get("stage_b_authorization_id") or "")
            != self._authorization_id
            or not str(qc.get("stage_b_confirm_intent_id") or "")
        ):
            raise StageBBlocked("STAGE_B_ACTION_QC_BINDING_INVALID")

    @staticmethod
    def _clear_prepare_response(response, code: str, reason: str, state: str) -> None:
        response.accepted = False
        response.authorization_id = ""
        response.plan_hash = ""
        response.stage_a_receipt_id = ""
        response.result_code = code
        response.summary_json = ""
        response.reason = reason
        response.state = state
        response.executor_instance_id = ""
        response.publisher_count = 0
        response.lease_active = False
        response.arm_client_count = 0

    def _prepare(self, request, response):
        before = dict(self._resources)
        try:
            if self._state not in {"IDLE", "FAILED_DISARMED"} or self._plan is not None:
                raise StageBBlocked("PREPARE_BUSY")
            if not str(request.request_id).strip() or request.request_id in self._seen_request_ids:
                raise StageBBlocked("PREPARE_REPLAY")
            raw = json.loads(request.canonical_stage_a_receipt_json)
            if canonical_json(raw) != request.canonical_stage_a_receipt_json:
                raise StageBBlocked("ARRIVAL_RECEIPT_INVALID", "non-canonical JSON")
            receipt = PreTerminalReceipt.from_mapping(raw, request.stage_a_receipt_sha256)
            if receipt.receipt_sha256 in self._seen_receipt_hashes:
                raise StageBBlocked("ARRIVAL_RECEIPT_REPLAY")
            if self._model is None:
                raise StageBBlocked("MODEL_UNAVAILABLE")
            prepared_qc_binding = self._prepare_graph_gate(receipt)
            plan = prepare_stage_b_plan(
                receipt, self._model, current_q_rad=self._current_q(),
                now_ns=self.get_clock().now().nanoseconds, now_monotonic=time.monotonic(),
                contact_authorized=bool(self._policy["contact_authorized"]),
                press_authorized=bool(self._policy["press_authorized"]),
                force_control=False, zero_dwell=True,
                require_wrist_tangent_correction=True,
            )
            if before != self._resources or any(self._resources.values()):
                raise StageBBlocked("ZERO_RESOURCE_PREPARE_FAILED")
            self._seen_request_ids.add(request.request_id)
            self._seen_receipt_hashes.add(receipt.receipt_sha256)
            self._plan, self._receipt = plan, receipt
            self._authorization_id = secrets.token_hex(16)
            self._prepared_qc_binding = prepared_qc_binding
            self._state = "AWAIT_EXECUTE_CONFIRM"
            summary = {
                "schema": plan.plan["schema"], "plan_hash": plan.plan_sha256,
                "panel_press_receipt_sha256": receipt.receipt_sha256,
                "panel_geometry_sha256": receipt.panel_geometry_sha256,
                "operation": "CONTACT_APPROACH_OUT_BACK",
                "approach_distance_m": plan.plan["approach_distance_m"],
                "command_count": plan.plan["command_count"],
                "forward_waypoint_count": len(plan.forward_waypoints_rad),
                "return_waypoint_count": len(plan.return_waypoints_rad),
                "gemini_p_button_base_m": plan.plan["gemini_p_button_base_m"],
                "wrist_tangent_delta_base_m": plan.plan[
                    "wrist_tangent_delta_base_m"
                ],
                "wrist_tangent_calibration_sha256": plan.plan[
                    "wrist_tangent_calibration_sha256"
                ],
                "wrist_image_error_px": plan.plan["wrist_image_error_px"],
                "wrist_correction_applied": plan.plan[
                    "wrist_correction_applied"
                ],
                "contact_authorized": True, "press_authorized": plan.plan["press_authorized"],
                "force_control": False, "zero_dwell": True,
                "expires_monotonic": plan.plan["expires_monotonic"],
                "model_hashes": plan.plan["model_hashes"],
            }
            response.accepted = True
            response.authorization_id = self._authorization_id
            response.plan_hash = plan.plan_sha256
            response.stage_a_receipt_id = receipt.receipt_id
            response.result_code = "STAGE_B_PLAN_FROZEN_NO_MOTION"
            response.summary_json = canonical_json(summary)
            response.reason = "PREPARED_ZERO_COMMAND_RESOURCES"
            response.state = self._state
            response.executor_instance_id = self._instance_id
            response.publisher_count = 0
            response.lease_active = False
            response.arm_client_count = 0
        except Exception as exc:
            code = getattr(exc, "code", "PREPARE_REJECTED")
            self._failure = str(exc)
            self._clear_prepare_response(response, code, str(exc), "FAILED_DISARMED")
            response.executor_instance_id = self._instance_id
        self._publish_status()
        return response

    def _goal(self, goal):
        with self._lock:
            if self._goal_reserved or self._plan is None or self._receipt is None:
                return GoalResponse.REJECT
            exact = StageBGoal(
                goal.plan_hash,
                goal.stage_a_receipt_sha256,
                goal.authorization_id,
                goal.confirmation_token,
            )
            if (
                exact.plan_sha256 != self._plan.plan_sha256
                or exact.stage_a_receipt_sha256 != self._receipt.receipt_sha256
                or exact.authorization_id != self._authorization_id
                or exact.confirmation_token != REQUIRED_CONFIRMATION_TOKEN
                or exact.plan_sha256 in self._consumed_plan_hashes
            ):
                return GoalResponse.REJECT
            ready, _reason = self._runtime_gate()
            if not ready:
                return GoalResponse.REJECT
            self._goal_reserved = True
            return GoalResponse.ACCEPT

    def _cancel(self, _goal):
        return CancelResponse.ACCEPT if self._goal_reserved else CancelResponse.REJECT

    def _wake_waiters(self) -> None:
        now = time.monotonic()
        with self._lock:
            ready = [item for item in self._sleep_waiters if item[0] <= now and not item[1].done()]
            self._sleep_waiters = [item for item in self._sleep_waiters if item not in ready and not item[1].done()]
        for _deadline, future in ready:
            future.set_result(None)

    async def _sleep(self, seconds: float) -> None:
        future = Future(executor=self.executor)
        item = (time.monotonic() + max(0.0, float(seconds)), future)
        with self._lock:
            self._sleep_waiters.append(item)
        try:
            await future
        finally:
            with self._lock:
                self._sleep_waiters = [value for value in self._sleep_waiters if value is not item]

    async def _await_response(self, future, *, timeout_s: float, goal_handle=None):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if future.done():
                return future.result()
            if goal_handle is not None and goal_handle.is_cancel_requested:
                future.cancel()
                raise StageBBlocked("MOVE_CANCELLED")
            await self._sleep(0.01)
        future.cancel()
        raise StageBBlocked("MOVE_SERVICE_TIMEOUT")

    async def _call_lease(self, enabled: bool, goal_handle=None) -> tuple[bool, str]:
        if self._lease_client is None:
            raise StageBBlocked("MOVE_RESOURCE_UNAVAILABLE", "lease client")
        request = PrecontactSpeedLease.Request()
        request.enable = enabled
        request.purpose = LEASE_PURPOSE
        request.lease_id = self._lease_id
        if enabled:
            self._lease_uncertain = True
            self._lease_enable_definitive = False
        response = await self._await_response(
            self._lease_client.call_async(request), timeout_s=0.75, goal_handle=goal_handle
        )
        if enabled:
            self._lease_enable_definitive = True
        return bool(response.success and response.active is enabled), str(response.message)

    async def _call_arm(self, enabled: bool, goal_handle=None) -> tuple[bool, str]:
        if self._arm_client is None:
            raise StageBBlocked("MOVE_RESOURCE_UNAVAILABLE", "arm client")
        request = SetBool.Request()
        request.data = enabled
        response = await self._await_response(
            self._arm_client.call_async(request), timeout_s=0.75, goal_handle=goal_handle
        )
        return bool(response.success), str(response.message)

    async def _wait_sample(self, predicate, timeout_s: float, *, goal_handle=None, min_status_at=None):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                raise StageBBlocked("MOVE_CANCELLED")
            sample = self._sample()
            if (min_status_at is None or sample.status_received_monotonic > min_status_at) and predicate(sample):
                return sample
            await self._sleep(0.02)
        raise StageBBlocked("MOVE_FEEDBACK_TIMEOUT")

    def _safe_driver_joint_target_subscribed(self) -> bool:
        try:
            endpoints = self.get_subscriptions_info_by_topic(self.JOINT_TARGET_TOPIC)
        except Exception as exc:
            raise StageBBlocked(
                "MOVE_RESOURCE_UNAVAILABLE", "safe-driver subscription graph"
            ) from exc
        return any(
            endpoint.node_name == "roarm_safe_driver"
            and endpoint.node_namespace == "/"
            and endpoint.topic_type == "sensor_msgs/msg/JointState"
            for endpoint in endpoints
        )

    async def _create_resources(self, goal_handle, machine: StageBExactExecution) -> None:
        if self._receipt is None or self._plan is None:
            raise StageBBlocked("PLAN_INVALIDATED")
        # Final zero-resource admission point.  STOP_REQUESTED or any stale /
        # foreign QC transition is rejected before even the lease client exists.
        self._action_graph_gate(self._receipt, self._plan.plan_sha256)
        self._lease_id = secrets.token_hex(16)
        self._lease_client = self.create_client(PrecontactSpeedLease, self.LEASE_SERVICE, callback_group=self._runtime_group)
        self._resources["lease_client"] = 1
        if not self._lease_client.wait_for_service(timeout_sec=0.0):
            raise StageBBlocked("MOVE_RESOURCE_UNAVAILABLE", "lease service")
        ok, reason = await self._call_lease(True, goal_handle)
        if not ok:
            raise StageBBlocked("MOVE_LEASE_REJECTED", reason)
        self._lease_active_local = True
        sample = await self._wait_sample(
            lambda value: (
                value.lease_active
                and value.lease_purpose == LEASE_PURPOSE
                and value.dynamic_exact_no_reissue
                and value.state in SAFE_IDLE_EXECUTION_STATES
                and not value.armed
            ),
            0.75, goal_handle=goal_handle,
        )
        self._lease_uncertain = False
        machine.lease_acquired(sample)
        self._arm_client = self.create_client(SetBool, self.ARM_SERVICE, callback_group=self._runtime_group)
        self._resources["arm_client"] = 1
        if not self._arm_client.wait_for_service(timeout_sec=0.0):
            raise StageBBlocked("MOVE_RESOURCE_UNAVAILABLE", "arm service")
        self._target_pub = self.create_publisher(JointState, self.JOINT_TARGET_TOPIC, 1)
        self._resources["publisher"] = 1
        try:
            sample = await self._wait_sample(
                lambda value: (
                    value.joint_target_publishers == 1
                    and self._target_pub is not None
                    and self._target_pub.get_subscription_count() >= 1
                    and self._safe_driver_joint_target_subscribed()
                ),
                0.75,
                goal_handle=goal_handle,
            )
        except StageBBlocked as exc:
            if exc.code == "MOVE_FEEDBACK_TIMEOUT":
                raise StageBBlocked(
                    "MOVE_RESOURCE_UNAVAILABLE",
                    "safe-driver subscription discovery",
                ) from exc
            raise
        machine.publisher_created(sample)

    def _publish_once(self, target: tuple[float, ...]) -> None:
        if self._target_pub is None or self._model is None:
            raise StageBBlocked("MOVE_RESOURCE_UNAVAILABLE", "publisher")
        if (
            self._target_pub.get_subscription_count() < 1
            or not self._safe_driver_joint_target_subscribed()
        ):
            raise StageBBlocked("MOVE_RESOURCE_UNAVAILABLE", "safe-driver subscription")
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self._model.policy["joint_names"])
        message.position = list(target)
        self._target_pub.publish(message)

    def _publish_machine_feedback(self, goal_handle, machine: StageBExactExecution) -> None:
        feedback = StageBRehearsal.Feedback()
        feedback.state = self._state
        feedback.phase = machine.phase
        feedback.waypoint_index = machine.command_index + 1
        feedback.waypoint_count = len(machine.commands)
        feedback.plan_hash = "" if self._plan is None else self._plan.plan_sha256
        feedback.stage_a_receipt_sha256 = (
            "" if self._receipt is None else self._receipt.receipt_sha256
        )
        goal_handle.publish_feedback(feedback)

    async def _publish_current_target(self, goal_handle, machine: StageBExactExecution, *, first_command: bool) -> None:
        if first_command:
            boundary = self._sample()
            if machine.state != "ARM_APPROACH" or machine.command_index != 0 or machine.publish_log:
                raise StageBBlocked("MOVE_PROTOCOL_ERROR", "first publish boundary")
            boundary.validate_owner(
                self._execution_config,
                expected_publishers=1,
                expected_lease=True,
                require_disarmed=True,
            )
            start_error = max(
                abs(value - expected)
                for value, expected in zip(boundary.positions_rad, machine.start_q_rad)
            )
            if start_error > machine.q0_drift_limit_rad:
                raise StageBBlocked("START_Q_DRIFT", f"publish boundary {start_error:.12g}")
            phase, target = machine.commands[machine.command_index]
            prior = boundary.accepted_command_sequence
            publish_boundary_positions = boundary.positions_rad
            ok, reason = await self._call_arm(True, goal_handle)
            machine.armed(ok, reason)
            if not ok:
                raise StageBBlocked("MOVE_ARM_REJECTED", reason)
        else:
            boundary = self._sample()
            machine.validate_publish_boundary(boundary, first_command=False)
            phase = machine.phase
            target = machine.target
            prior = boundary.accepted_command_sequence
            publish_boundary_positions = boundary.positions_rad
        published_at = time.monotonic()
        self._publish_once(target)
        machine.mark_published(
            phase,
            target,
            prior_sequence=prior,
            publish_boundary_positions=publish_boundary_positions,
            now_monotonic=published_at,
        )
        self._publish_machine_feedback(goal_handle, machine)
        try:
            accepted = await self._wait_sample(
                lambda value: value.state == "COMMAND_ACTIVE" and value.armed and value.accepted_command_sequence != prior,
                self._execution_config.command_ack_timeout_s,
                goal_handle=goal_handle, min_status_at=published_at,
            )
        except StageBBlocked as exc:
            raise StageBBlocked("MOVE_COMMAND_ACK_MISSING", str(exc)) from exc
        machine.accept_command(accepted, now_monotonic=time.monotonic())

    async def _reconcile_lease(self) -> bool:
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline:
            try:
                sample = self._sample()
                if not sample.lease_active:
                    if not self._lease_enable_definitive:
                        await self._sleep(0.02)
                        continue
                    self._lease_active_local = False
                    self._lease_uncertain = False
                    return True
                with self._lock:
                    lease = dict(self._safe_status.get("precontact_speed_lease") or {})
                if lease.get("purpose") != LEASE_PURPOSE or lease.get("lease_id") != self._lease_id:
                    raise StageBBlocked("MOVE_LEASE_RECONCILIATION_FOREIGN")
                ok, reason = await self._call_lease(False)
                if not ok:
                    raise StageBBlocked("MOVE_LEASE_RELEASE_FAILED", reason)
                self._lease_enable_definitive = True
            except StageBBlocked:
                raise
            await self._sleep(0.02)
        return False

    async def _teardown(self) -> tuple[bool, str]:
        reasons = []
        if self._arm_client is not None:
            try:
                ok, reason = await self._call_arm(False)
                if not ok:
                    reasons.append(f"disarm:{reason}")
            except Exception as exc:
                reasons.append(f"disarm:{exc}")
        if self._target_pub is not None:
            try:
                self.destroy_publisher(self._target_pub)
                self._target_pub = None
                self._resources["publisher"] = 0
            except Exception as exc:
                reasons.append(f"publisher:{exc}")
        if self._arm_client is not None:
            try:
                self.destroy_client(self._arm_client)
                self._arm_client = None
                self._resources["arm_client"] = 0
            except Exception as exc:
                reasons.append(f"arm-client:{exc}")
        lease_clean = not (self._lease_active_local or self._lease_uncertain)
        if not lease_clean:
            try:
                lease_clean = await self._reconcile_lease()
            except Exception as exc:
                reasons.append(f"lease-reconciliation:{exc}")
        if self._lease_client is not None:
            try:
                self.destroy_client(self._lease_client)
                self._lease_client = None
                self._resources["lease_client"] = 0
            except Exception as exc:
                reasons.append(f"lease-client:{exc}")
        if lease_clean:
            self._lease_id = ""
        try:
            sample = await self._wait_sample(
                lambda value: value.joint_target_publishers == 0 and not value.lease_active and not value.armed,
                0.75,
            )
            clean = sample.joint_target_publishers == 0 and not sample.lease_active and not sample.armed
        except Exception as exc:
            reasons.append(f"terminal-proof:{exc}")
            clean = False
        return bool(clean and lease_clean and not reasons), ";".join(reasons)

    def _fill_result(self, result, success: bool, code: str, reason: str, teardown: bool) -> None:
        result.success = success
        result.terminal_code = code
        result.terminal_reason = reason
        result.state = self._state
        result.plan_hash = "" if self._plan is None else self._plan.plan_sha256
        result.stage_a_receipt_sha256 = "" if self._receipt is None else self._receipt.receipt_sha256
        try:
            sample = self._sample()
            result.terminal_joint_target_publishers = sample.joint_target_publishers
            result.terminal_precontact_lease_active = sample.lease_active
            result.terminal_armed = sample.armed
            result.terminal_feedback_age_s = sample.feedback_age_s
            result.teardown_complete = bool(teardown)
        except Exception:
            result.terminal_joint_target_publishers = int(self.count_publishers(self.JOINT_TARGET_TOPIC))
            result.terminal_precontact_lease_active = True
            result.terminal_armed = True
            result.terminal_feedback_age_s = math.inf
            result.teardown_complete = False
        result.endpoint_max_joint_error_rad = math.inf if self._machine is None else self._machine.endpoint_error_rad

    async def _execute(self, goal_handle):
        result = StageBRehearsal.Result()
        success = False
        code = "MOVE_REJECTED"
        reason = ""
        teardown_ok = False
        try:
            if self._plan is None or self._receipt is None or self._model is None:
                raise StageBBlocked("PLAN_INVALIDATED")
            ready, gate_reason = self._runtime_gate()
            if not ready:
                raise StageBBlocked(gate_reason)
            request = goal_handle.request
            if (
                request.plan_hash != self._plan.plan_sha256
                or request.stage_a_receipt_sha256 != self._receipt.receipt_sha256
                or request.authorization_id != self._authorization_id
                or request.confirmation_token != REQUIRED_CONFIRMATION_TOKEN
            ):
                raise StageBBlocked("PLAN_IDENTITY_MISMATCH")
            self._action_graph_gate(self._receipt, self._plan.plan_sha256)
            verified = verify_stage_b_plan(
                canonical_json(self._plan.plan), self._plan.plan_sha256,
                expected_receipt=self._receipt, model=self._model,
                current_q_rad=self._sample().positions_rad,
                now_monotonic=time.monotonic(),
                expected_press_authorized=bool(self._policy["press_authorized"]),
            )
            self._consumed_plan_hashes.add(self._plan.plan_sha256)
            self._machine = StageBExactExecution(verified, self._execution_config, now_monotonic=time.monotonic())
            self._resources["action"] = 1
            self._machine.precheck(self._sample())
            await self._create_resources(goal_handle, self._machine)
            first_command = True
            while self._machine.state in {"ARM_APPROACH", "PUBLISH_WAYPOINT", "WAIT_WAYPOINT_CONVERGENCE"}:
                if goal_handle.is_cancel_requested:
                    self._machine.cancel()
                    raise StageBBlocked("MOVE_CANCELLED")
                if self._machine.state in {"ARM_APPROACH", "PUBLISH_WAYPOINT"}:
                    self._state = (
                        "EXECUTING_CONTACT_APPROACH"
                        if self._machine.phase == "FORWARD"
                        else "EXECUTING_EXACT_PRE_RETURN"
                    )
                    # No sleep/dwell is allowed here.  A convergence event
                    # that opens the next frozen waypoint falls through to
                    # this publication on the next tight loop iteration.
                    await self._publish_current_target(
                        goal_handle, self._machine, first_command=first_command
                    )
                    first_command = False
                    continue
                self._machine.observe(self._sample(), now_monotonic=time.monotonic())
                if self._machine.state == "WAIT_WAYPOINT_CONVERGENCE":
                    await self._sleep(0.02)
            success = self._machine.state == "TEARDOWN_SUCCESS"
            code, reason = self._machine.terminal_code, self._machine.terminal_reason
        except Exception as exc:
            code = getattr(exc, "code", "MOVE_RUNTIME_FAILURE")
            reason = str(exc)
            if self._machine is not None:
                if goal_handle.is_cancel_requested:
                    self._machine.cancel()
                else:
                    self._machine.fail(code, reason)
        finally:
            teardown_ok, teardown_reason = await self._teardown()
            if self._machine is not None and self._machine.state in {"TEARDOWN_SUCCESS", "TEARDOWN_FAILURE", "TEARDOWN_CANCEL"}:
                self._machine.teardown_complete(success=success and teardown_ok)
                success = self._machine.state == "SUCCEEDED"
                code, reason = self._machine.terminal_code, self._machine.terminal_reason
            if not teardown_ok:
                success = False
                code = "MOVE_TEARDOWN_FAILED"
                reason = ";".join(filter(None, (reason, teardown_reason)))
            self._resources["action"] = 0
            self._state = "CONTACT_APPROACH_OUT_BACK_COMPLETE" if success else "FAILED_DISARMED"
        self._fill_result(result, success, code, reason, teardown_ok)
        if success:
            goal_handle.succeed()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        self._plan = None
        self._receipt = None
        self._authorization_id = ""
        self._prepared_qc_binding = {}
        self._machine = None
        self._goal_reserved = False
        self._state = "IDLE"
        self._publish_status()
        return result

    def _publish_status(self) -> None:
        ready, gate = self._runtime_gate()
        message = String()
        message.data = canonical_json({
            "schema": "stage_b_contact_approach_executor_status_v1",
            "instance_id": self._instance_id,
            "state": self._state,
            "failure": self._failure,
            "runtime_enabled": bool(self.get_parameter("runtime_enabled").value),
            "build_enabled": bool(self.get_parameter("build_enabled").value),
            "contact_authorized": bool(self.get_parameter("contact_authorized").value),
            "press_authorized": bool(self.get_parameter("press_authorized").value),
            "force_control": bool(self.get_parameter("force_control").value),
            "execution_ready": ready,
            "execution_gate": gate,
            "plan_hash": "" if self._plan is None else self._plan.plan_sha256,
            "resource_counts": dict(self._resources),
            "joint_target_publishers": int(self.count_publishers(self.JOINT_TARGET_TOPIC)),
            "lease_purpose": LEASE_PURPOSE,
        })
        self._status_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = StageBExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
