#!/usr/bin/env python3
"""Dynamic-IK Prepare owner and gated exact frozen-plan Move executor.

Boot performs the only model/config reads. Prepare remains an in-memory,
zero-motion path. Motion resources are constructed lazily only after an exact
Move Action goal passes every identity, policy and runtime-enable gate.
"""
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
        if (candidate / "tools/press/t102_dynamic_precontact_core.py").is_file():
            return candidate.resolve()
    raise RuntimeError("Lift_Robot root not found")


ROOT = _repo_root()
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from press.t102_dynamic_precontact_core import (  # noqa: E402
    DynamicPrepareBlocked, FrozenModel, FrozenSession, PreparedPlan,
    canonical_hash, prepare_dynamic_plan, verify_frozen_plan,
)
from press.standby_route_core import (  # noqa: E402
    CONFIRMATION_TOKEN as STANDBY_CONFIRMATION_TOKEN,
    SOURCE_ROUTE_INITIAL_STATE,
    StandbyRouteBlocked, StandbyRoutePlan,
    canonical_hash as standby_canonical_hash,
    consume_standby_route, prepare_standby_route,
)
from .execution import (  # noqa: E402
    DYNAMIC_LEASE_PURPOSE, ExactPlanExecution, FAULT_CLEARED_DISARMED,
    MOVE_CONFIRMATION_TOKEN, MoveBlocked, MovePolicy, SafetySample,
    VALIDATE_CONFIRMATION_TOKEN,
    lease_reconciliation_decision, runtime_safety_failure_reason,
    validate_command_ack,
)

import rclpy  # noqa: E402
from elevator_interfaces.action import PrecontactMove, StandbyRoute  # noqa: E402
from elevator_interfaces.srv import (  # noqa: E402
    DiscardPreparedPrecontact, PreparePrecontact, PrepareStandbyRoute,
    PrecontactSpeedLease,
)
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import (  # noqa: E402
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.task import Future  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402


class State:
    IDLE = "IDLE"
    PREPARE_IN_FLIGHT = "PREPARE_IN_FLIGHT"
    AWAIT_MOVE_CONFIRM = "AWAIT_MOVE_CONFIRM"
    HANDOFF_VALIDATED_NO_MOTION = "HANDOFF_VALIDATED_NO_MOTION"
    MOVE_IN_FLIGHT = "MOVE_IN_FLIGHT"
    MOVE_COMPLETE_DISARMED = "MOVE_COMPLETE_DISARMED"
    FAILED_DISARMED = "FAILED_DISARMED"
    STANDBY_PREPARE_IN_FLIGHT = "STANDBY_PREPARE_IN_FLIGHT"
    AWAIT_STANDBY_CONFIRM = "AWAIT_STANDBY_CONFIRM"
    STANDBY_ACTION_IN_FLIGHT = "STANDBY_ACTION_IN_FLIGHT"
    STANDBY_COMPLETE_DISARMED = "STANDBY_COMPLETE_DISARMED"


class DynamicExecutor(Node):
    BUILD_ENABLE_VALUE = "ENABLE_T102_DYNAMIC_EXACT_PLAN_MOVE"
    LEASE_SERVICE = "/roarm_safe/precontact_speed_lease"
    LEASE_SERVICE_TYPE = "elevator_interfaces/srv/PrecontactSpeedLease"
    ARM_SERVICE = "/roarm_safe/set_armed"
    ARM_SERVICE_TYPE = "std_srvs/srv/SetBool"
    DISCARD_CONFIRMATION_TOKEN = "DISCARD_EXACT_PREPARED_PRECONTACT_NO_MOTION"
    def __init__(self) -> None:
        super().__init__("roarm_t102_outbound_executor")
        self.declare_parameter("planning_enabled", True)
        self.declare_parameter("move_runtime_enabled", False)
        self.declare_parameter(
            "policy_path",
            str(ROOT / "config/press/prototype/t102_dynamic_precontact_policy_20260816_v1.json"),
        )
        # The Action coroutine must yield while feedback subscriptions, service
        # replies, and bounded delay timers keep running.  Humble's rclpy uses
        # its own task scheduler rather than an asyncio event loop, so all
        # Move-time entities share an explicitly reentrant callback group.
        self._runtime_group = ReentrantCallbackGroup()
        self._prepare_group = MutuallyExclusiveCallbackGroup()
        self._lock = threading.RLock()
        self._sleep_waiters: list[tuple[float, Future]] = []
        self._instance_nonce = secrets.token_hex(16)
        self._sequence = 0
        self._state = State.IDLE
        self._plan: PreparedPlan | None = None
        self._last_completed_plan: PreparedPlan | None = None
        self._soft_standby_proof: dict = {}
        self._soft_precontact_plan_hash = ""
        self._standby_plan: StandbyRoutePlan | None = None
        self._standby_authorization_id = ""
        self._standby_seen: set[str] = set()
        self._authorization_id = ""
        self._seen: set[str] = set()
        self._latest_joint: JointState | None = None
        self._safe_status: dict = {}
        self._safe_status_received_monotonic: float | None = None
        self._safe_feedback: dict = {}
        self._safe_feedback_received_monotonic: float | None = None
        self._failure = ""
        self._prepare_calls = 0
        self._last_handoff_validation = ""
        self._active_machine: ExactPlanExecution | None = None
        self._goal_slot_reserved = False
        self._motion_publisher = None
        self._lease_client = None
        self._arm_client = None
        self._lease_id = ""
        self._lease_active_local = False
        self._lease_enable_uncertain = False
        self._lease_enable_response_definitive = False
        self._resource_counts = {
            "publisher": 0, "lease_client": 0, "arm_client": 0,
            "safe_driver_client": 0, "service_call": 0, "action_call": 0,
        }
        try:
            self._model = FrozenModel.load_once(
                ROOT, str(self.get_parameter("policy_path").value)
            )
            self._move_policy = MovePolicy.from_mapping(self._model.policy)
        except Exception as exc:
            self._model = None
            self._move_policy = None
            self._failure = f"MODEL_LOAD_FAILED: {exc}"
        self._status_pub = self.create_publisher(String, "~/status", 10)
        self.create_subscription(
            JointState, "/joint_states", self._on_joint, 20,
            callback_group=self._runtime_group,
        )
        self.create_subscription(
            String, "/roarm_safe/status", self._on_safe_status, 20,
            callback_group=self._runtime_group,
        )
        self.create_subscription(
            String, "/roarm_safe/feedback_status", self._on_safe_feedback, 20,
            callback_group=self._runtime_group,
        )
        self._prepare_service = self.create_service(
            PreparePrecontact, "~/prepare_precontact", self._prepare,
            callback_group=self._prepare_group,
        )
        self._discard_service = self.create_service(
            DiscardPreparedPrecontact, "~/discard_prepared_precontact",
            self._discard_prepared, callback_group=self._prepare_group,
        )
        self._action_server = ActionServer(
            self, PrecontactMove, "~/move_to_precontact",
            execute_callback=self._execute, goal_callback=self._goal,
            cancel_callback=self._cancel, callback_group=self._runtime_group,
        )
        self._standby_prepare_service = self.create_service(
            PrepareStandbyRoute, "~/prepare_standby_route", self._prepare_standby,
            callback_group=self._prepare_group,
        )
        self._standby_action_server = ActionServer(
            self, StandbyRoute, "~/execute_standby_route",
            execute_callback=self._execute_standby,
            goal_callback=self._standby_goal,
            cancel_callback=self._cancel,
            callback_group=self._runtime_group,
        )
        self.create_timer(
            0.5, self._publish_status, callback_group=self._runtime_group
        )
        self._sleep_timer = self.create_timer(
            0.005, self._wake_sleep_waiters,
            callback_group=self._runtime_group,
        )
        self._publish_status()

    def _on_joint(self, message: JointState) -> None:
        with self._lock:
            self._latest_joint = message

    def _on_safe_status(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            if isinstance(value, dict):
                with self._lock:
                    self._safe_status = value
                    self._safe_status_received_monotonic = time.monotonic()
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def _on_safe_feedback(self, message: String) -> None:
        try:
            value = json.loads(message.data)
            if isinstance(value, dict):
                with self._lock:
                    self._safe_feedback = value
                    self._safe_feedback_received_monotonic = time.monotonic()
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def _move_gate(self) -> tuple[bool, str]:
        model = self._model
        if model is None or self._move_policy is None:
            return False, "MOVE_MODEL_UNAVAILABLE"
        if model.policy.get("move_execution_approved") is not True:
            return False, "MOVE_POLICY_NOT_APPROVED"
        if not bool(self.get_parameter("move_runtime_enabled").value):
            return False, "MOVE_RUNTIME_DISABLED"
        if os.environ.get("LIFT_ROBOT_DYNAMIC_MOVE_BUILD_TOKEN", "") != self.BUILD_ENABLE_VALUE:
            return False, "MOVE_BUILD_TOKEN_MISSING"
        # Readiness is shown to Live before an Action goal exists.  Check the
        # ROS graph here without constructing either command-capable client;
        # otherwise the UI can advertise READY and only discover a missing
        # lease/arm service after the operator clicks GO PRE.
        try:
            services = {
                str(name): {str(type_name) for type_name in type_names}
                for name, type_names in self.get_service_names_and_types()
            }
        except Exception:  # noqa: BLE001 - graph uncertainty is fail-closed
            return False, "MOVE_RESOURCE_GRAPH_UNAVAILABLE"
        if self.LEASE_SERVICE_TYPE not in services.get(self.LEASE_SERVICE, set()):
            return False, "MOVE_RESOURCE_UNAVAILABLE: speed lease service"
        if self.ARM_SERVICE_TYPE not in services.get(self.ARM_SERVICE, set()):
            return False, "MOVE_RESOURCE_UNAVAILABLE: arm service"
        return True, "MOVE_EXECUTION_READY"

    def _status(self) -> dict:
        model = self._model
        move_ready, move_reason = self._move_gate()
        with self._lock:
            safe_status = dict(self._safe_status)
            safe_feedback = dict(self._safe_feedback)
            feedback_received = self._safe_feedback_received_monotonic
            status_received = self._safe_status_received_monotonic
        lease = dict(safe_status.get("precontact_speed_lease") or {})
        now = time.monotonic()
        feedback_age = (
            math.inf if feedback_received is None
            else now - feedback_received
        )
        lease_age = (
            math.inf if status_received is None
            else now - status_received
        )
        status_max_age = (
            0.5 if self._move_policy is None
            else self._move_policy.feedback_max_age_s
        )
        try:
            (
                _p, _s, feedback_armed, _d, _f, _t, _v, feedback_stamp,
                _accepted_sequence, _accepted_target, _advance_ready,
            ) = (
                self._strict_feedback_fields(safe_feedback)
            )
            feedback_stamp_age = (
                int(self.get_clock().now().nanoseconds) - feedback_stamp
            ) / 1_000_000_000.0
        except (MoveBlocked, AttributeError, TypeError, ValueError):
            feedback_armed = None
            feedback_stamp_age = math.inf
        terminal_armed = (
            feedback_armed
            if (
                0.0 <= feedback_age <= status_max_age
                and 0.0 <= feedback_stamp_age <= status_max_age
                and type(feedback_armed) is bool
            )
            else None
        )
        lease_active = (
            lease.get("active")
            if lease_age <= status_max_age and type(lease.get("active")) is bool
            else None
        )
        return {
            "schema": "t102_dynamic_executor_status_v1",
            "instance_nonce": self._instance_nonce,
            "executor_boot_epoch": "" if model is None else model.boot_epoch,
            "state": self._state,
            "planning_enabled": bool(self.get_parameter("planning_enabled").value),
            "model_loaded": bool(model and model.model_loaded),
            "model_failure_reason": self._failure if model is None else model.model_failure_reason,
            "execution_admission_ready": bool(model and model.execution_admission_ready),
            "admission_failure_reason": "" if model is None else model.admission_failure_reason,
            "move_execution_ready": move_ready,
            "move_execution_reason": move_reason,
            "move_policy_approved": bool(model and model.policy.get("move_execution_approved") is True),
            "move_runtime_enabled": bool(self.get_parameter("move_runtime_enabled").value),
            "move_build_token_present": os.environ.get("LIFT_ROBOT_DYNAMIC_MOVE_BUILD_TOKEN", "") == self.BUILD_ENABLE_VALUE,
            "kinematic_model_hash": "" if model is None else model.hashes["kinematic_model_hash"],
            "collision_proxy_hash": "" if model is None else model.hashes["collision_proxy_hash"],
            "tool_model_hash": "" if model is None else model.hashes["tool_model_hash"],
            "template_hash": "" if model is None else model.hashes["template_hash"],
            "calibration_hash": "" if model is None else model.hashes["calibration_hash"],
            "accepted_camera_record_hash": "" if model is None else model.hashes["accepted_camera_record_hash"],
            "fn_artifact_hash": "" if model is None else model.hashes["fn_artifact_hash"],
            "fn_acceptance_hash": "" if model is None else model.hashes["fn_acceptance_hash"],
            "standby_acceptance_hash": "" if model is None else model.hashes["standby_acceptance_hash"],
            "source_registry_hash": "" if model is None else model.hashes["source_registry_hash"],
            **({} if model is None else model.source_bindings),
            "startup_model_load_ms": None if model is None else model.startup_load_ms,
            "plan_hash": "" if self._plan is None else self._plan.plan_hash,
            "waypoint_count": 0 if self._plan is None else len(self._plan.plan["waypoints_rad"]),
            "standby_plan_hash": "" if self._standby_plan is None else self._standby_plan.plan_hash,
            "standby_waypoint_count": 0 if self._standby_plan is None else len(self._standby_plan.payload["waypoints_rad"]),
            "standby_already_at_target": bool(self._standby_plan and self._standby_plan.payload.get("already_at_standby")),
            "completed_precontact_plan_hash": (
                "" if self._last_completed_plan is None
                else self._last_completed_plan.plan_hash
            ),
            "completed_precontact_plan_consumed": bool(
                self._last_completed_plan is not None
                and self._last_completed_plan.consumed
            ),
            "resource_counts": dict(self._resource_counts),
            "joint_target_publishers": self.count_publishers("/roarm_safe/joint_target"),
            "precontact_lease_active": lease_active,
            "precontact_lease_active_known": lease_active is not None,
            "terminal_armed": terminal_armed,
            "terminal_armed_known": terminal_armed is not None,
            "dynamic_exact_no_reissue": bool(lease.get("dynamic_exact_no_reissue")),
            "handoff_validation_available": True,
            "last_handoff_validation": self._last_handoff_validation,
            "fixed_v2_fallback": False,
            "failure": self._failure,
            "published_at_unix_ns": time.time_ns(),
        }

    def _publish_status(self) -> None:
        self._sequence += 1
        value = self._status()
        value["sequence"] = self._sequence
        message = String()
        message.data = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self._status_pub.publish(message)

    @staticmethod
    def _clear_response(response, code, reason, state):
        response.accepted = False
        response.authorization_id = ""
        response.ttl_ms = 0
        response.plan_hash = ""
        response.session_id = ""
        response.result_code = str(code)
        response.summary_json = ""
        response.reason = str(reason)
        response.state = state
        response.executor_boot_epoch = ""
        response.kinematic_model_hash = ""
        response.collision_proxy_hash = ""
        response.tool_model_hash = ""
        response.template_hash = ""
        response.calibration_hash = ""
        response.publisher_count = 0
        response.lease_active = False
        response.arm_client_count = 0

    def _prepare(self, request, response):
        with self._lock:
            before = dict(self._resource_counts)
            self._prepare_calls += 1
            self._reap_expired_standby_ticket_locked()
            if (
                self._plan is not None or self._authorization_id
                or self._standby_plan is not None or self._standby_authorization_id
                or self._goal_slot_reserved or self._active_machine is not None
            ):
                self._clear_response(
                    response, "PLAN_INVALIDATED",
                    "existing route ticket/action owns executor", self._state,
                )
                response.publisher_count = int(self._resource_counts["publisher"])
                response.lease_active = bool(self._resource_counts["lease_client"])
                response.arm_client_count = int(self._resource_counts["arm_client"])
                return response
            self._state = State.PREPARE_IN_FLIGHT
            try:
                if not bool(self.get_parameter("planning_enabled").value):
                    raise DynamicPrepareBlocked("PLANNING_DISABLED")
                if self._model is None:
                    raise DynamicPrepareBlocked("MODEL_LOAD_FAILED", self._failure)
                if not self._model.execution_admission_ready:
                    raise DynamicPrepareBlocked("TOOL_MODEL_OR_COLLISION_PROXY_NOT_APPROVED", self._model.admission_failure_reason)
                if request.ui_request_id in self._seen:
                    raise DynamicPrepareBlocked("PLAN_INVALIDATED", "replayed Prepare")
                if self.count_publishers("/roarm_safe/joint_target") != 0:
                    raise DynamicPrepareBlocked("PLAN_INVALIDATED", "joint_target publisher exists")
                session_value = json.loads(request.dynamic_session_json)
                if canonical_hash(session_value) != request.dynamic_session_sha256:
                    raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "session hash")
                session = FrozenSession.from_mapping(session_value)
                if session.executor_boot_epoch != self._model.boot_epoch:
                    raise DynamicPrepareBlocked("PLAN_INVALIDATED", "executor boot mismatch")
                joint = self._latest_joint
                if joint is None:
                    raise DynamicPrepareBlocked("TARGET_STALE", "no fresh q0 feedback")
                stamp = int(joint.header.stamp.sec) * 1_000_000_000 + int(joint.header.stamp.nanosec)
                by_name = dict(zip(joint.name, joint.position))
                observed = tuple(float(by_name[name]) for name in self._model.policy["joint_names"])
                if stamp < session.q0_stamp_ns:
                    raise DynamicPrepareBlocked("RGBD_TF_SESSION_MISMATCH", "q0 observation chronology")
                if self.get_clock().now().nanoseconds - stamp > int(float(self._model.policy["maximum_q0_age_s"]) * 1e9):
                    raise DynamicPrepareBlocked("TARGET_STALE", "current q0 feedback")
                if max(abs(a-b) for a, b in zip(observed, session.q0_rad)) > float(self._model.policy["q0_drift_limit_rad"]):
                    raise DynamicPrepareBlocked("PLAN_INVALIDATED", "q0 drift")
                soft_sample = self._safety_sample()
                # QC has sampled a stationary q0. Bind the current healthy
                # disarmed state to this ticket, not an old Standby receipt:
                # Clear Fault / Torque ON may change the state without motion.
                soft_sample.validate(
                    self._move_policy, expected_publishers=0,
                    expected_lease=False,
                    allow_fault_cleared_disarmed=(
                        soft_sample.state == FAULT_CLEARED_DISARMED
                    ),
                    allow_torque_on_verified=(
                        soft_sample.state == "TORQUE_ON_VERIFIED"
                    ),
                    allow_operator_disarmed=(
                        soft_sample.state == SOURCE_ROUTE_INITIAL_STATE
                    ),
                )
                ticket = prepare_dynamic_plan(
                    self._model, session, now_ns=self.get_clock().now().nanoseconds,
                    now_monotonic=time.monotonic(),
                    initial_disarmed_state=soft_sample.state,
                )
                after = dict(self._resource_counts)
                if before != after or any(after.values()):
                    raise DynamicPrepareBlocked("ZERO_RESOURCE_INVARIANT_FAILED")
                self._seen.add(str(request.ui_request_id))
                self._last_completed_plan = None
                self._plan = ticket
                self._soft_precontact_plan_hash = ""
                self._authorization_id = secrets.token_hex(16)
                self._state = State.AWAIT_MOVE_CONFIRM
                self._failure = ""
                response.accepted = True
                response.authorization_id = self._authorization_id
                response.ttl_ms = int(self._model.policy["ticket_ttl_s"] * 1000)
                response.plan_hash = ticket.plan_hash
                response.session_id = session.session_id
                response.result_code = "SAMPLED_LIGHTWEIGHT_CONSTRAINT_PASS"
                response.summary_json = json.dumps(ticket.plan, sort_keys=True, separators=(",", ":"))
                response.reason = "PREPARED_DYNAMIC_IK_NO_MOTION"
                response.state = self._state
                response.executor_boot_epoch = self._model.boot_epoch
                for field in ("kinematic_model_hash", "collision_proxy_hash", "tool_model_hash", "template_hash", "calibration_hash"):
                    setattr(response, field, self._model.hashes[field])
                response.publisher_count = int(after["publisher"])
                response.lease_active = bool(after["lease_client"])
                response.arm_client_count = int(after["arm_client"])
            except Exception as exc:
                code = exc.code if isinstance(exc, DynamicPrepareBlocked) else "PREPARE_REJECTED"
                self._state = State.FAILED_DISARMED
                self._failure = str(exc)
                self._plan = None
                self._authorization_id = ""
                self._clear_response(response, code, str(exc), self._state)
                if self._model is not None:
                    response.executor_boot_epoch = self._model.boot_epoch
                    for field in ("kinematic_model_hash", "collision_proxy_hash", "tool_model_hash", "template_hash", "calibration_hash"):
                        setattr(response, field, self._model.hashes[field])
            self._publish_status()
            return response

    def _discard_prepared(self, request, response):
        """Drop one exact unconsumed ticket; never acquire or release resources."""

        with self._lock:
            status = self._status()
            plan_hash = "" if self._plan is None else str(self._plan.plan_hash)
            safe = bool(
                self._state == State.AWAIT_MOVE_CONFIRM
                and self._plan is not None
                and not self._plan.consumed
                and not self._goal_slot_reserved
                and self._active_machine is None
                and request.executor_instance_nonce == self._instance_nonce
                and request.plan_hash == plan_hash
                and request.confirmation_token == self.DISCARD_CONFIRMATION_TOKEN
                and not any(self._resource_counts.values())
                and int(status["joint_target_publishers"]) == 0
                and status["precontact_lease_active"] is False
                and status["terminal_armed"] is False
            )
            response.accepted = safe
            response.executor_instance_nonce = self._instance_nonce
            response.plan_hash = plan_hash
            response.publisher_count = int(self._resource_counts["publisher"])
            response.lease_active = bool(self._resource_counts["lease_client"])
            response.arm_client_count = int(self._resource_counts["arm_client"])
            if safe:
                self._plan = None
                self._authorization_id = ""
                self._failure = ""
                self._state = State.IDLE
                response.reason = "DISCARDED_PREPARED_PRECONTACT_NO_MOTION"
            else:
                response.reason = "DISCARD_PREPARED_PRECONTACT_BLOCKED"
            response.state = self._state
            self._publish_status()
            return response

    def _supersede_inert_ticket_for_standby_locked(self) -> bool:
        """Release one no-motion ticket so the same GO STA can replace it."""

        precontact_inert = bool(
            self._state == State.AWAIT_MOVE_CONFIRM
            and self._plan is not None
            and not self._plan.consumed
            and bool(self._authorization_id)
            and self._standby_plan is None
            and not self._standby_authorization_id
        )
        standby_inert = bool(
            self._state == State.AWAIT_STANDBY_CONFIRM
            and self._standby_plan is not None
            and not self._standby_plan.consumed
            and bool(self._standby_authorization_id)
            and self._plan is None
            and not self._authorization_id
        )
        if not (precontact_inert or standby_inert):
            return False
        before = dict(self._resource_counts)
        if (
            self._goal_slot_reserved
            or self._active_machine is not None
            or any(before.values())
            or self.count_publishers("/roarm_safe/joint_target") != 0
        ):
            return False
        try:
            sample = self._safety_sample()
            sample.validate(
                self._move_policy, expected_publishers=0,
                expected_lease=False,
                allow_fault_cleared_disarmed=(
                    sample.state == FAULT_CLEARED_DISARMED
                ),
                allow_torque_on_verified=(
                    sample.state == "TORQUE_ON_VERIFIED"
                ),
                allow_operator_disarmed=(
                    sample.state == SOURCE_ROUTE_INITIAL_STATE
                ),
            )
        except Exception:  # noqa: BLE001 - failed proof leaves ticket intact
            return False
        if before != self._resource_counts or any(self._resource_counts.values()):
            return False
        if precontact_inert:
            self._plan = None
            self._authorization_id = ""
            self._soft_precontact_plan_hash = ""
        else:
            self._standby_plan = None
            self._standby_authorization_id = ""
        self._state = State.IDLE
        self._failure = ""
        return True

    def _prepare_standby(self, request, response):
        """Freeze one no-retry reverse corridor; never create motion resources."""
        with self._lock:
            before = dict(self._resource_counts)
            self._reap_expired_standby_ticket_locked()
            if (
                self._plan is not None or self._authorization_id
                or self._standby_plan is not None or self._standby_authorization_id
                or self._goal_slot_reserved or self._active_machine is not None
            ) and not self._supersede_inert_ticket_for_standby_locked():
                response.accepted = False
                response.authorization_id = ""
                response.ttl_ms = 0
                response.plan_hash = ""
                response.session_id = ""
                response.result_code = "STANDBY_ROUTE_OWNER_BUSY"
                response.summary_json = ""
                response.reason = "existing route ticket/action owns executor"
                response.state = self._state
                response.waypoint_count = 0
                response.already_at_standby = False
                response.publisher_count = int(self._resource_counts["publisher"])
                response.lease_active = bool(self._resource_counts["lease_client"])
                response.arm_client_count = int(self._resource_counts["arm_client"])
                return response
            self._state = State.STANDBY_PREPARE_IN_FLIGHT
            self._soft_standby_proof = {}
            self._soft_precontact_plan_hash = ""
            try:
                if self._model is None or self._move_policy is None:
                    raise StandbyRouteBlocked("STANDBY_ROUTE_MODEL_UNAVAILABLE")
                if request.ui_request_id in self._standby_seen:
                    raise StandbyRouteBlocked("STANDBY_ROUTE_REPLAYED")
                move_ready, reason = self._move_gate()
                if not move_ready:
                    raise StandbyRouteBlocked(reason)
                sample = self._safety_sample()
                cold_bootstrap = bool(
                    self._last_completed_plan is None
                    and sample.state in {
                        FAULT_CLEARED_DISARMED, "CONVERGED_DISARMED",
                        "TORQUE_ON_VERIFIED",
                    }
                )
                source_return = bool(
                    self._last_completed_plan is not None
                    and sample.state in {
                        FAULT_CLEARED_DISARMED, "CONVERGED_DISARMED", "TORQUE_ON_VERIFIED",
                        SOURCE_ROUTE_INITIAL_STATE,
                    }
                )
                sample.validate(
                    self._move_policy, expected_publishers=0,
                    expected_lease=False,
                    allow_fault_cleared_disarmed=(
                        sample.state == FAULT_CLEARED_DISARMED
                    ),
                    allow_torque_on_verified=(
                        sample.state == "TORQUE_ON_VERIFIED"
                    ),
                    allow_operator_disarmed=(
                        source_return and sample.state == SOURCE_ROUTE_INITIAL_STATE
                    ),
                )
                ticket = prepare_standby_route(
                    self._model, self._last_completed_plan, sample.positions_rad,
                    ui_request_id=request.ui_request_id,
                    now_monotonic=time.monotonic(),
                    bootstrap_state=(
                        sample.state
                        if cold_bootstrap or source_return else ""
                    ),
                )
                after = dict(self._resource_counts)
                if before != after or any(after.values()):
                    raise StandbyRouteBlocked("ZERO_RESOURCE_INVARIANT_FAILED")
                self._standby_seen.add(str(request.ui_request_id))
                self._standby_plan = ticket
                self._standby_authorization_id = secrets.token_hex(16)
                self._state = State.AWAIT_STANDBY_CONFIRM
                self._failure = ""
                response.accepted = True
                response.authorization_id = self._standby_authorization_id
                response.ttl_ms = int(self._model.policy["ticket_ttl_s"] * 1000)
                response.plan_hash = ticket.plan_hash
                response.session_id = str(ticket.payload.get("session_id") or request.ui_request_id)
                response.result_code = "EXACT_SEGMENTED_STANDBY_ROUTE_PREPARED_NO_MOTION"
                response.summary_json = json.dumps(ticket.payload, sort_keys=True, separators=(",", ":"))
                response.reason = "STANDBY_PREPARED_NO_MOTION"
                response.state = self._state
                response.waypoint_count = len(ticket.payload["waypoints_rad"])
                response.already_at_standby = bool(ticket.payload["already_at_standby"])
                response.publisher_count = 0
                response.lease_active = False
                response.arm_client_count = 0
            except Exception as exc:
                self._standby_plan = None
                self._standby_authorization_id = ""
                self._state = State.FAILED_DISARMED
                self._failure = str(exc)
                response.accepted = False
                response.authorization_id = ""
                response.ttl_ms = 0
                response.plan_hash = ""
                response.session_id = ""
                response.result_code = getattr(exc, "code", "STANDBY_PREPARE_REJECTED")
                response.summary_json = ""
                response.reason = str(exc)
                response.state = self._state
                response.waypoint_count = 0
                response.already_at_standby = False
                response.publisher_count = int(self._resource_counts["publisher"])
                response.lease_active = bool(self._resource_counts["lease_client"])
                response.arm_client_count = int(self._resource_counts["arm_client"])
            self._publish_status()
            return response

    def _reap_expired_standby_ticket_locked(self) -> bool:
        """Drop only an expired, inert Standby Prepare ticket (no ROS writes)."""
        plan = self._standby_plan
        if not (
            plan is not None
            and time.monotonic() >= plan.expires_monotonic
            and not self._goal_slot_reserved
            and self._active_machine is None
            and not any(self._resource_counts.values())
        ):
            return False
        self._standby_plan = None
        self._standby_authorization_id = ""
        if self._state == State.AWAIT_STANDBY_CONFIRM:
            self._state = State.IDLE
        return True

    def _exact_standby_goal_locked(self, goal) -> bool:
        plan = self._standby_plan
        return bool(
            plan and not plan.consumed
            and goal.authorization_id == self._standby_authorization_id
            and goal.plan_hash == plan.plan_hash
            and goal.session_id == str(plan.payload.get("session_id") or plan.payload["ui_request_id"])
        )

    def _standby_goal(self, goal):
        with self._lock:
            if self._goal_slot_reserved or self._active_machine is not None:
                self._failure = "STANDBY_ACTION_ALREADY_RESERVED"
                return GoalResponse.REJECT
            if not self._exact_standby_goal_locked(goal):
                self._failure = "STANDBY_ROUTE_IDENTITY_MISMATCH"
                return GoalResponse.REJECT
            if goal.confirmation_text != STANDBY_CONFIRMATION_TOKEN:
                self._failure = "STANDBY_CONFIRMATION_TOKEN_REJECTED"
                return GoalResponse.REJECT
            ready, reason = self._move_gate()
            if not ready:
                self._failure = reason
                return GoalResponse.REJECT
            self._goal_slot_reserved = True
            return GoalResponse.ACCEPT

    def _exact_goal_locked(self, goal) -> bool:
        return bool(
            self._plan and not self._plan.consumed
            and goal.authorization_id == self._authorization_id
            and goal.plan_hash == self._plan.plan_hash
            and goal.session_id == self._plan.plan["session"]["session_id"]
        )

    def _goal(self, goal):
        with self._lock:
            if self._goal_slot_reserved:
                self._failure = "MOVE_ACTION_ALREADY_RESERVED"
                return GoalResponse.REJECT
            if not self._exact_goal_locked(goal):
                self._last_handoff_validation = "REJECTED_IDENTITY_MISMATCH"
                return GoalResponse.REJECT
            if goal.confirmation_text == VALIDATE_CONFIRMATION_TOKEN:
                self._last_handoff_validation = "ACCEPTED_EXACT_NO_MOTION_GOAL"
                self._goal_slot_reserved = True
                return GoalResponse.ACCEPT
            if goal.confirmation_text != MOVE_CONFIRMATION_TOKEN:
                self._failure = "MOVE_CONFIRMATION_TOKEN_REJECTED"
                return GoalResponse.REJECT
            move_ready, reason = self._move_gate()
            if not move_ready:
                self._failure = reason
                return GoalResponse.REJECT
            if self._active_machine is not None:
                self._failure = "MOVE_ACTION_ALREADY_ACTIVE"
                return GoalResponse.REJECT
            self._goal_slot_reserved = True
            return GoalResponse.ACCEPT

    def _cancel(self, _goal):
        with self._lock:
            return CancelResponse.ACCEPT if self._goal_slot_reserved else CancelResponse.REJECT

    def _current_positions(self) -> tuple[float, ...]:
        if self._model is None or self._latest_joint is None:
            raise MoveBlocked("MOVE_FEEDBACK_STALE", "joint state absent")
        by_name = dict(zip(self._latest_joint.name, self._latest_joint.position))
        try:
            return tuple(float(by_name[name]) for name in self._model.policy["joint_names"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MoveBlocked("MOVE_FEEDBACK_STALE", "joint order") from exc

    @staticmethod
    def _strict_feedback_fields(feedback: dict) -> tuple:
        """Return complete typed motion evidence; missing/None never coerces."""

        positions = feedback.get("positions_rad")
        if (
            type(positions) is not list
            or len(positions) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in positions
            )
        ):
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "positions_rad")
        armed = feedback.get("armed")
        dry_run = feedback.get("dry_run")
        if type(armed) is not bool:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "armed")
        if type(dry_run) is not bool:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "dry_run")
        fault = feedback.get("fault")
        state = feedback.get("state")
        if type(fault) is not str:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "fault")
        if type(state) is not str or not state:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "state")
        torque = feedback.get("torque_enabled")
        if type(torque) is not list or len(torque) != 6 or any(type(value) is not bool for value in torque):
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "torque_enabled")
        voltage = feedback.get("voltage")
        if isinstance(voltage, bool) or not isinstance(voltage, (int, float)) or not math.isfinite(float(voltage)):
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "voltage")
        stamp = feedback.get("feedback_stamp_ns")
        if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "feedback_stamp_ns")
        accepted_sequence = feedback.get("dynamic_exact_accepted_command_sequence")
        accepted_target = feedback.get("dynamic_exact_accepted_target_positions")
        advance_ready = feedback.get("dynamic_exact_advance_ready")
        if type(accepted_sequence) is not int or accepted_sequence < 0:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "accepted command sequence")
        if (
            type(accepted_target) is not list
            or len(accepted_target) not in {0, 6}
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in accepted_target
            )
        ):
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "accepted target")
        if type(advance_ready) is not bool:
            raise MoveBlocked("MOVE_FEEDBACK_MALFORMED", "advance ready")
        return (
            tuple(float(value) for value in positions), state, armed, dry_run,
            fault, tuple(torque), float(voltage), int(stamp),
            int(accepted_sequence),
            tuple(float(value) for value in accepted_target),
            bool(advance_ready),
        )

    def _safety_sample(self) -> SafetySample:
        with self._lock:
            status = dict(self._safe_status)
            feedback = dict(self._safe_feedback)
            received = self._safe_feedback_received_monotonic
            status_received = self._safe_status_received_monotonic
        age = math.inf if received is None else time.monotonic() - received
        lease = dict(status.get("precontact_speed_lease") or {})
        status_age = (
            math.inf
            if status_received is None
            else time.monotonic() - status_received
        )
        (
            positions, state, armed, dry_run, fault, torque, voltage, stamp,
            accepted_sequence, accepted_target, advance_ready,
        ) = self._strict_feedback_fields(feedback)
        stamp_age = (int(self.get_clock().now().nanoseconds) - stamp) / 1_000_000_000.0
        if type(lease.get("active")) is not bool:
            raise MoveBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN", "lease active missing")
        if type(lease.get("purpose")) is not str:
            raise MoveBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN", "lease purpose missing")
        if type(lease.get("dynamic_exact_no_reissue")) is not bool:
            raise MoveBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN", "lease mode missing")
        # Every motion-health field comes from the fresh event stream. The
        # transient-local status may be retained across a restart and is used
        # only for the lease manager's explicit ownership snapshot.
        return SafetySample(
            positions, float(age), state, armed, dry_run, fault,
            torque, voltage,
            int(self.count_publishers("/roarm_safe/joint_target")), lease["active"],
            lease["purpose"], lease["dynamic_exact_no_reissue"],
            float(status_age), float(received if received is not None else -math.inf),
            float(stamp_age),
            int(accepted_sequence), tuple(float(value) for value in accepted_target),
            bool(advance_ready),
            float(status_received if status_received is not None else -math.inf),
        )

    def _wake_sleep_waiters(self) -> None:
        now = time.monotonic()
        ready = []
        with self._lock:
            pending = []
            for waiter in self._sleep_waiters:
                deadline, future = waiter
                if future.done():
                    continue
                if deadline <= now:
                    ready.append(future)
                else:
                    pending.append(waiter)
            self._sleep_waiters = pending
        for future in ready:
            if not future.done():
                future.set_result(None)

    async def _ros_sleep(self, delay_s: float) -> None:
        """Yield through the node's persistent rclpy sleep scheduler."""

        future = Future(executor=self.executor)
        waiter = (time.monotonic() + max(float(delay_s), 0.0), future)
        with self._lock:
            self._sleep_waiters.append(waiter)
        try:
            await future
        finally:
            with self._lock:
                self._sleep_waiters = [
                    entry for entry in self._sleep_waiters
                    if entry is not waiter
                ]

    async def _await_service_response(self, future, *, timeout_s: float,
                                      cancel_requested=None):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if future.done():
                return future.result()
            if cancel_requested is not None and cancel_requested():
                future.cancel()
                raise MoveBlocked("MOVE_CANCELLED", "operator cancellation requested")
            await self._ros_sleep(0.01)
        future.cancel()
        raise MoveBlocked("MOVE_SERVICE_TIMEOUT", "bounded service acknowledgement timeout")

    async def _call_lease(self, enable: bool, *, cancel_requested=None) -> tuple[bool, str]:
        if self._lease_client is None:
            return False, "lease client absent"
        request = PrecontactSpeedLease.Request()
        request.enable = bool(enable)
        request.purpose = DYNAMIC_LEASE_PURPOSE
        request.lease_id = self._lease_id
        self._resource_counts["service_call"] += 1
        try:
            if enable:
                # From call_async onward the remote side may have applied the
                # lease even if the acknowledgement is lost or cancelled.
                self._lease_enable_uncertain = True
                self._lease_enable_response_definitive = False
            future = self._lease_client.call_async(request)
            response = await self._await_service_response(
                future, timeout_s=0.75, cancel_requested=cancel_requested
            )
            if enable:
                self._lease_enable_response_definitive = True
        finally:
            self._resource_counts["service_call"] -= 1
        return bool(response.success and response.active is bool(enable)), str(response.message)

    async def _call_arm(self, enabled: bool, *, cancel_requested=None) -> tuple[bool, str]:
        if self._arm_client is None:
            return False, "arm client absent"
        request = SetBool.Request()
        request.data = bool(enabled)
        self._resource_counts["service_call"] += 1
        try:
            future = self._arm_client.call_async(request)
            response = await self._await_service_response(
                future, timeout_s=0.75, cancel_requested=cancel_requested
            )
        finally:
            self._resource_counts["service_call"] -= 1
        return bool(response.success), str(response.message)

    async def _wait_sample(self, predicate, timeout_s: float, *, cancel_requested=None,
                           min_received_monotonic: float | None = None,
                           min_status_received_monotonic: float | None = None,
                           allow_fault_cleared_disarmed: bool = False,
                           allow_torque_on_verified: bool = False,
                           allow_operator_disarmed: bool = False) -> SafetySample:
        deadline = time.monotonic() + timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            if cancel_requested is not None and cancel_requested():
                raise MoveBlocked("MOVE_CANCELLED", "operator cancellation requested")
            try:
                sample = self._safety_sample()
                if (
                    min_received_monotonic is not None
                    and (
                        self._safe_feedback_received_monotonic is None
                        or self._safe_feedback_received_monotonic
                        <= min_received_monotonic
                    )
                ):
                    await self._ros_sleep(0.01)
                    continue
                if (
                    min_status_received_monotonic is not None
                    and sample.status_received_monotonic
                    <= min_status_received_monotonic
                ):
                    await self._ros_sleep(0.01)
                    continue
                if (
                    not math.isfinite(sample.feedback_age_s)
                    or sample.feedback_age_s < 0.0
                    or sample.feedback_age_s > self._move_policy.feedback_max_age_s
                ):
                    raise MoveBlocked("MOVE_FEEDBACK_STALE")
                safety_failure = runtime_safety_failure_reason(sample)
                if safety_failure:
                    raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", safety_failure)
                if (
                    not math.isfinite(sample.feedback_stamp_age_s)
                    or sample.feedback_stamp_age_s < 0.0
                    or sample.feedback_stamp_age_s > self._move_policy.feedback_max_age_s
                    or not math.isfinite(sample.lease_status_age_s)
                    or sample.lease_status_age_s < 0.0
                    or sample.lease_status_age_s > self._move_policy.feedback_max_age_s
                ):
                    raise MoveBlocked("MOVE_FEEDBACK_STALE")
                healthy_wait_state = (
                    sample.state == SOURCE_ROUTE_INITIAL_STATE and not sample.armed
                    if allow_operator_disarmed
                    else sample.state == FAULT_CLEARED_DISARMED and not sample.armed
                    if allow_fault_cleared_disarmed
                    else (sample.state == "TORQUE_ON_VERIFIED" and not sample.armed)
                    if allow_torque_on_verified else (
                        (sample.state == "ARMED" and sample.armed)
                        or (sample.state == "COMMAND_ACTIVE" and sample.armed)
                        or (sample.state == "CONVERGED_DISARMED" and not sample.armed)
                    )
                )
                if (
                    not math.isfinite(sample.voltage_v)
                    or not self._move_policy.min_voltage_v <= sample.voltage_v <= self._move_policy.max_voltage_v
                    or not healthy_wait_state
                ):
                    raise MoveBlocked("MOVE_SAFETY_GATE_FAILED", "fresh feedback event")
                # A post-publish ARMED sample is an expected service/topic
                # scheduling interleave: the arm acknowledgement can reach
                # feedback before the just-published target is consumed.  It
                # is healthy but is not command-acceptance evidence, so keep
                # waiting for a newer COMMAND_ACTIVE event within the same
                # bounded acknowledgement timeout.
                if predicate(sample):
                    return sample
            except MoveBlocked:
                raise
            except Exception as exc:
                last_error = str(exc)
            await self._ros_sleep(0.02)
        raise MoveBlocked("MOVE_FEEDBACK_STALE", last_error or "runtime state acknowledgement timeout")

    async def _create_execution_resources(
        self, goal_handle, machine: ExactPlanExecution
    ) -> None:
        # Ordering is safety-significant: obtain the isolated no-reissue lease
        # before constructing either command-capable resource.
        if self._lease_id:
            raise MoveBlocked("MOVE_PROTOCOL_ERROR", "lease id already assigned")
        self._lease_id = secrets.token_hex(16)
        self._lease_client = self.create_client(
            PrecontactSpeedLease, self.LEASE_SERVICE,
            callback_group=self._runtime_group,
        )
        self._resource_counts["lease_client"] = 1
        if not self._lease_client.wait_for_service(timeout_sec=0.0):
            raise MoveBlocked("MOVE_RESOURCE_UNAVAILABLE", "speed lease service")
        ok, reason = await self._call_lease(
            True, cancel_requested=lambda: goal_handle.is_cancel_requested
        )
        if not ok:
            raise MoveBlocked("MOVE_LEASE_REJECTED", reason)
        self._lease_active_local = True
        await self._wait_sample(
            lambda sample: sample.lease_active and sample.dynamic_exact_no_reissue,
            0.75, cancel_requested=lambda: goal_handle.is_cancel_requested,
            allow_fault_cleared_disarmed=(
                machine.bootstrap_admission_state == FAULT_CLEARED_DISARMED
                or machine.allow_initial_fault_cleared_disarmed
            ),
            allow_torque_on_verified=machine.allow_initial_torque_on_verified,
            allow_operator_disarmed=machine.allow_initial_operator_disarmed,
        )
        self._lease_enable_uncertain = False
        # Preserve the pure machine's safety-significant phase boundary:
        # lease proof is sampled while no command publisher exists.
        machine.lease_acquired(self._safety_sample())
        self._arm_client = self.create_client(
            SetBool, self.ARM_SERVICE,
            callback_group=self._runtime_group,
        )
        self._resource_counts["arm_client"] = 1
        self._resource_counts["safe_driver_client"] = 1
        if not self._arm_client.wait_for_service(timeout_sec=0.0):
            raise MoveBlocked("MOVE_RESOURCE_UNAVAILABLE", "arm service")
        self._motion_publisher = self.create_publisher(JointState, "/roarm_safe/joint_target", 1)
        self._resource_counts["publisher"] = 1
        await self._wait_sample(
            lambda sample: sample.joint_target_publishers == 1, 0.75,
            cancel_requested=lambda: goal_handle.is_cancel_requested,
            allow_fault_cleared_disarmed=(
                machine.bootstrap_admission_state == FAULT_CLEARED_DISARMED
                or machine.allow_initial_fault_cleared_disarmed
            ),
            allow_torque_on_verified=machine.allow_initial_torque_on_verified,
            allow_operator_disarmed=machine.allow_initial_operator_disarmed,
        )
        machine.publisher_created(self._safety_sample())

    def _publish_exact_waypoint_once(self, waypoint: tuple[float, ...]) -> None:
        if self._motion_publisher is None or self._model is None:
            raise MoveBlocked("MOVE_RESOURCE_UNAVAILABLE", "joint publisher")
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(self._model.policy["joint_names"])
        message.position = list(waypoint)
        self._motion_publisher.publish(message)

    def _lease_reconciliation_snapshot(self) -> tuple[str, dict]:
        with self._lock:
            status = dict(self._safe_status)
            received = self._safe_status_received_monotonic
        lease = status.get("precontact_speed_lease")
        if type(lease) is not dict:
            raise MoveBlocked("MOVE_LEASE_RECONCILIATION_UNKNOWN", "lease status missing")
        age = math.inf if received is None else time.monotonic() - received
        decision = lease_reconciliation_decision(
            status_age_s=age,
            maximum_age_s=self._move_policy.feedback_max_age_s,
            active=lease.get("active"),
            purpose=lease.get("purpose"),
            lease_id=lease.get("lease_id"),
            expected_lease_id=self._lease_id,
        )
        return decision, dict(lease)

    async def _reconcile_uncertain_lease(self) -> None:
        """Best-effort release without erasing an indeterminate enable outcome."""

        deadline = time.monotonic() + 0.75
        last_unknown = ""
        released = False
        observed_exact_active = False
        enable_was_definitive = self._lease_enable_response_definitive
        while time.monotonic() < deadline:
            try:
                decision, _lease = self._lease_reconciliation_snapshot()
            except MoveBlocked as exc:
                if exc.code == "MOVE_LEASE_RECONCILIATION_FOREIGN":
                    raise
                last_unknown = str(exc)
                await self._ros_sleep(0.02)
                continue
            if decision == "PROVEN_INACTIVE":
                if not enable_was_definitive:
                    # A timed-out/cancelled enable future can still execute on
                    # the server after this snapshot. Keep watching for an
                    # exact late-active lease so it can be released, but never
                    # promote this Action terminal to clean.
                    last_unknown = (
                        "enable acknowledgement indeterminate; fresh inactive "
                        "does not exclude a queued late apply"
                    )
                    await self._ros_sleep(0.02)
                    continue
                if observed_exact_active and not released:
                    raise MoveBlocked(
                        "MOVE_LEASE_RECONCILIATION_UNKNOWN",
                        "exact active lease lacked definitive release; operator intervention required",
                    )
                self._lease_active_local = False
                self._lease_enable_uncertain = False
                return
            if decision == "RELEASE_EXACT_ACTIVE" and not released:
                observed_exact_active = True
                if self._lease_client is None:
                    raise MoveBlocked(
                        "MOVE_LEASE_RECONCILIATION_UNKNOWN",
                        "exact active lease but release client unavailable; operator intervention required",
                    )
                ok, reason = await self._call_lease(False)
                if not ok:
                    raise MoveBlocked(
                        "MOVE_LEASE_RELEASE_FAILED",
                        f"{reason}; operator intervention required",
                    )
                released = True
                await self._ros_sleep(0.02)
                continue
            await self._ros_sleep(0.02)
        raise MoveBlocked(
            "MOVE_LEASE_RECONCILIATION_UNKNOWN",
            (last_unknown or "no fresh inactive lease proof")
            + "; operator intervention required",
        )

    def _terminal_resource_evidence(self) -> tuple[bool, bool, bool, float]:
        """Return (known, lease_active, armed, feedback_age) conservatively."""

        now = time.monotonic()
        with self._lock:
            feedback = dict(self._safe_feedback)
            status = dict(self._safe_status)
            feedback_received = self._safe_feedback_received_monotonic
            status_received = self._safe_status_received_monotonic
        feedback_age = math.inf if feedback_received is None else now - feedback_received
        lease_age = math.inf if status_received is None else now - status_received
        lease = status.get("precontact_speed_lease")
        try:
            (
                _positions, _state, armed, _dry, _fault, _torque, _voltage,
                stamp, _accepted_sequence, _accepted_target, _advance_ready,
            ) = (
                self._strict_feedback_fields(feedback)
            )
            stamp_age = (
                int(self.get_clock().now().nanoseconds) - stamp
            ) / 1_000_000_000.0
        except (MoveBlocked, AttributeError, TypeError, ValueError):
            armed = None
            stamp_age = math.inf
        known = bool(
            self._move_policy is not None
            and 0.0 <= feedback_age <= self._move_policy.feedback_max_age_s
            and 0.0 <= stamp_age <= self._move_policy.feedback_max_age_s
            and 0.0 <= lease_age <= self._move_policy.feedback_max_age_s
            and type(armed) is bool
            and type(lease) is dict
            and type(lease.get("active")) is bool
        )
        if not known:
            return False, True, True, feedback_age
        return True, bool(lease["active"]), bool(armed), feedback_age

    async def _teardown_execution_resources(self) -> tuple[bool, str]:
        reasons: list[str] = []
        if self._arm_client is not None:
            try:
                # The safe driver disarms itself after the final frozen
                # waypoint converges.  Do not overwrite that attributable
                # CONVERGED_DISARMED terminal with a redundant operator
                # disarm: the next Dynamic Prepare needs the fresh Standard
                # Standby terminal as its q0 provenance.  Every non-terminal,
                # stale, malformed, armed, faulted, or uncertain sample still
                # takes the existing explicit fail-closed disarm path.
                sample = self._safety_sample()
                already_converged_disarmed = bool(
                    sample.state == "CONVERGED_DISARMED"
                    and not sample.armed
                    and not sample.dry_run
                    and not sample.fault
                    and math.isfinite(sample.feedback_age_s)
                    and 0.0 <= sample.feedback_age_s
                    <= self._move_policy.feedback_max_age_s
                )
                if not already_converged_disarmed:
                    ok, reason = await self._call_arm(False)
                    if not ok:
                        reasons.append(f"disarm:{reason}")
            except Exception as exc:
                # If terminal sampling itself is uncertain, preserve the
                # original cleanup guarantee and attempt one explicit disarm.
                try:
                    ok, reason = await self._call_arm(False)
                    if not ok:
                        reasons.append(f"disarm:{reason}")
                except Exception as disarm_exc:
                    reasons.append(f"disarm:{exc};{disarm_exc}")
        if self._motion_publisher is not None:
            publisher_destroyed = False
            try:
                self.destroy_publisher(self._motion_publisher)
                publisher_destroyed = True
            except Exception as exc:
                reasons.append(f"publisher-destroy:{exc}")
            if publisher_destroyed:
                self._motion_publisher = None
                self._resource_counts["publisher"] = 0
        lease_proven_clean = not (
            self._lease_active_local or self._lease_enable_uncertain
        )
        if self._lease_active_local or self._lease_enable_uncertain:
            try:
                await self._reconcile_uncertain_lease()
                lease_proven_clean = True
            except Exception as exc:
                reasons.append(f"lease-reconciliation:{exc}")
        if self._arm_client is not None:
            arm_client_destroyed = False
            try:
                self.destroy_client(self._arm_client)
                arm_client_destroyed = True
            except Exception as exc:
                reasons.append(f"arm-client-destroy:{exc}")
            if arm_client_destroyed:
                self._arm_client = None
                self._resource_counts["arm_client"] = 0
                self._resource_counts["safe_driver_client"] = 0
        if self._lease_client is not None:
            lease_client_destroyed = False
            try:
                self.destroy_client(self._lease_client)
                lease_client_destroyed = True
            except Exception as exc:
                reasons.append(f"lease-client-destroy:{exc}")
            if lease_client_destroyed:
                self._lease_client = None
                self._resource_counts["lease_client"] = 0
        if lease_proven_clean:
            self._lease_id = ""
        deadline = time.monotonic() + 0.75
        clean = False
        while time.monotonic() < deadline:
            known, lease_active, armed, _feedback_age = self._terminal_resource_evidence()
            if (
                known and lease_proven_clean and not lease_active and not armed
                and self.count_publishers("/roarm_safe/joint_target") == 0
            ):
                clean = True
                break
            await self._ros_sleep(0.02)
        if not clean:
            reasons.append("terminal resources not proven fresh and inactive; operator intervention required")
        return clean and not reasons, ";".join(reasons)

    def _fill_result(self, result, *, success: bool, code: str, reason: str,
                     plan: PreparedPlan | None, teardown_complete: bool) -> None:
        result.success = bool(success)
        result.terminal_code = str(code)
        result.terminal_reason = str(reason)
        result.state = self._state
        result.session_id = "" if plan is None else str(plan.plan["session"]["session_id"])
        result.plan_hash = "" if plan is None else plan.plan_hash
        result.terminal_joint_target_publishers = int(self.count_publishers("/roarm_safe/joint_target"))
        evidence_known, lease_active, armed, feedback_age = (
            self._terminal_resource_evidence()
        )
        # The typed result has no tri-state fields. Unknown is therefore
        # represented conservatively as active/armed and teardown false.
        result.terminal_precontact_lease_active = lease_active
        result.terminal_armed = armed
        result.teardown_complete = bool(teardown_complete and evidence_known)
        machine = self._active_machine
        result.endpoint_max_joint_error_rad = math.inf if machine is None else float(machine.endpoint_error_rad)
        result.terminal_feedback_age_s = float(feedback_age)

    async def _execute_validation(self, goal_handle, plan: PreparedPlan | None):
        result = PrecontactMove.Result()
        with self._lock:
            exact = bool(plan is not None and self._exact_goal_locked(goal_handle.request))
        if not exact or plan is None:
            self._state = State.FAILED_DISARMED
            self._fill_result(result, success=False, code="PRECONTACT_HANDOFF_VALIDATION_IDENTITY_DRIFT", reason="exact plan identity changed after goal admission", plan=None, teardown_complete=True)
            goal_handle.abort()
            return result
        feedback = PrecontactMove.Feedback()
        feedback.state = "VALIDATING_NO_MOTION"
        feedback.gate = "EXACT_FROZEN_PLAN_HANDOFF"
        feedback.waypoint_index = 0
        feedback.waypoint_count = len(plan.plan["waypoints_rad"])
        feedback.plan_hash = plan.plan_hash
        feedback.session_id = str(plan.plan["session"]["session_id"])
        goal_handle.publish_feedback(feedback)
        self._state = State.HANDOFF_VALIDATED_NO_MOTION
        self._last_handoff_validation = "PASS_EXACT_PLAN_ACTION_ROUNDTRIP_NO_MOTION"
        self._fill_result(result, success=True, code="PRECONTACT_HANDOFF_VALIDATED_NO_MOTION", reason="exact frozen plan reached executor action boundary; no motion resources created", plan=plan, teardown_complete=True)
        goal_handle.succeed()
        self._publish_status()
        return result

    async def _execute(self, goal_handle):
        request = goal_handle.request
        with self._lock:
            plan = self._plan
        if request.confirmation_text == VALIDATE_CONFIRMATION_TOKEN:
            try:
                return await self._execute_validation(goal_handle, plan)
            finally:
                with self._lock:
                    self._goal_slot_reserved = False
        result = PrecontactMove.Result()
        machine: ExactPlanExecution | None = None
        terminal_code = "MOVE_REJECTED"
        terminal_reason = ""
        success = False
        teardown_complete = False
        soft_initial_state = ""
        try:
            preflight = self._safety_sample()
            with self._lock:
                if plan is None or self._model is None or self._move_policy is None or not self._exact_goal_locked(request):
                    raise MoveBlocked("PLAN_INVALIDATED", "identity drift after goal admission")
                move_ready, reason = self._move_gate()
                if not move_ready:
                    raise MoveBlocked(reason)
                session = FrozenSession.from_mapping(plan.plan["session"])
                soft_proof = dict(self._soft_standby_proof or {})
                soft_q = tuple(
                    float(value) for value in soft_proof.get("measured_q_rad", ())
                )
                proof_state = str(soft_proof.get("admission_state") or "")
                soft_proof_move = bool(
                    plan.plan_hash == self._soft_precontact_plan_hash
                    and soft_proof.get("executor_boot_epoch") == self._model.boot_epoch
                    and proof_state in {
                        FAULT_CLEARED_DISARMED, "TORQUE_ON_VERIFIED",
                    }
                    and preflight.state == proof_state
                    and len(soft_q) == 6
                    and max(abs(a - b) for a, b in zip(preflight.positions_rad, soft_q))
                    <= self._move_policy.encoder_step_rad + 1e-12
                )
                soft_initial_state = proof_state if soft_proof_move else ""
                consumed = verify_frozen_plan(
                    plan, model=self._model, session=session,
                    current_q0=preflight.positions_rad,
                    now_monotonic=time.monotonic(), consume=True,
                )
                self._plan = consumed
                plan = consumed
                machine = ExactPlanExecution(
                    plan.plan, self._move_policy, now=time.monotonic(),
                    initial_disarmed_state=(
                        str(plan.plan.get("initial_disarmed_state") or "")
                        or soft_initial_state or "CONVERGED_DISARMED"
                    ),
                )
                self._active_machine = machine
                self._resource_counts["action_call"] = 1
                self._state = State.MOVE_IN_FLIGHT
            machine.precheck(preflight)
            if goal_handle.is_cancel_requested:
                machine.cancel()
            else:
                await self._create_execution_resources(goal_handle, machine)
            if machine.state == "TEARDOWN_CANCEL":
                raise MoveBlocked("MOVE_CANCELLED", "operator cancellation requested")
            while machine.state in {"ARM_WAYPOINT", "PUBLISH_WAYPOINT"}:
                if goal_handle.is_cancel_requested:
                    machine.cancel()
                    break
                # Ticket TTL is an admission deadline, not a mid-motion stop.
                # After consume, only immutable plan/session/model identity is
                # rechecked; no IK or target is recomputed.
                if canonical_hash(plan.plan) != plan.plan_hash:
                    raise MoveBlocked("PLAN_INVALIDATED", "plan hash drift")
                if canonical_hash(plan.plan.get("model_hashes")) != canonical_hash(self._model.hashes):
                    raise MoveBlocked("PLAN_INVALIDATED", "model identity drift")
                if canonical_hash(plan.plan.get("session")) != canonical_hash(session.__dict__):
                    raise MoveBlocked("PLAN_INVALIDATED", "session identity drift")
                boundary = self._safety_sample()
                if machine.state == "ARM_WAYPOINT":
                    boundary.validate(
                        self._move_policy, expected_publishers=1,
                        expected_lease=True,
                        allow_operator_disarmed=(
                            machine.waypoint_index == 0
                            and machine.allow_initial_operator_disarmed
                        ),
                        allow_fault_cleared_disarmed=(
                            machine.waypoint_index == 0
                            and machine.allow_initial_fault_cleared_disarmed
                        ),
                        allow_torque_on_verified=(
                            machine.waypoint_index == 0
                            and machine.allow_initial_torque_on_verified
                        ),
                    )
                    ok, reason = await self._call_arm(
                        True, cancel_requested=lambda: goal_handle.is_cancel_requested
                    )
                    machine.armed(ok, reason)
                else:
                    # Continuous frozen-plan advance: the previous exact
                    # waypoint is already inside tolerance and the isolated
                    # lease/publisher/armed identity must remain intact.
                    if (
                        boundary.state != "COMMAND_ACTIVE"
                        or not boundary.armed
                        or boundary.joint_target_publishers != 1
                        or not boundary.lease_active
                        or boundary.lease_purpose != DYNAMIC_LEASE_PURPOSE
                        or not boundary.dynamic_exact_no_reissue
                        or not boundary.dynamic_exact_advance_ready
                    ):
                        raise MoveBlocked("MOVE_OWNER_DRIFT", "continuous waypoint boundary")
                if machine.state != "PUBLISH_WAYPOINT":
                    break
                target = machine.target
                if not machine.command_within_caps(target, boundary.positions_rad):
                    raise MoveBlocked("MOVE_COMMAND_CAP_EXCEEDED", "pre-publish measured pose")
                published_at = time.monotonic()
                prior_sequence = boundary.dynamic_exact_accepted_command_sequence
                self._publish_exact_waypoint_once(target)
                machine.mark_published(target, now=published_at)
                feedback = PrecontactMove.Feedback()
                feedback.state = machine.state
                feedback.gate = "EXACT_FROZEN_WAYPOINT_PUBLISHED_ONCE"
                feedback.waypoint_index = machine.waypoint_index + 1
                feedback.waypoint_count = len(machine.waypoints)
                feedback.plan_hash = plan.plan_hash
                feedback.session_id = str(plan.plan["session"]["session_id"])
                goal_handle.publish_feedback(feedback)
                # SafeDriver publishes controller state and command acceptance
                # atomically in one feedback event.  That fresh event is the
                # sole command-ACK authority; /roarm_safe/status remains the
                # independently delivered lease/owner evidence.
                try:
                    active_sample = await self._wait_sample(
                        lambda value: (
                            value.state == "COMMAND_ACTIVE"
                            and value.armed
                            and value.dynamic_exact_accepted_command_sequence
                            != prior_sequence
                        ),
                        0.75, cancel_requested=lambda: goal_handle.is_cancel_requested,
                        min_received_monotonic=published_at,
                    )
                except MoveBlocked as exc:
                    if exc.code == "MOVE_FEEDBACK_STALE":
                        raise MoveBlocked(
                            "MOVE_COMMAND_ACK_MISSING",
                            "SafeDriver did not accept the waypoint within the bounded wait",
                        ) from exc
                    raise
                validate_command_ack(
                    active_sample,
                    prior_sequence=prior_sequence,
                    expected_target=target,
                    pre_publish_positions=boundary.positions_rad,
                    encoder_step_rad=self._move_policy.encoder_step_rad,
                )
                machine.observe(active_sample, now=time.monotonic())
                while machine.state == "WAIT_CONVERGENCE":
                    if goal_handle.is_cancel_requested:
                        machine.cancel()
                        break
                    machine.observe(self._safety_sample(), now=time.monotonic())
                    if machine.state == "WAIT_CONVERGENCE":
                        await self._ros_sleep(0.02)
            success = machine.state == "TEARDOWN_SUCCESS"
            terminal_code = machine.terminal_code
            terminal_reason = machine.terminal_reason
        except (DynamicPrepareBlocked, MoveBlocked, Exception) as exc:
            if machine is not None:
                if goal_handle.is_cancel_requested:
                    machine.cancel()
                else:
                    machine.fail(getattr(exc, "code", "MOVE_RUNTIME_FAILURE"), str(exc))
                terminal_code = machine.terminal_code
                terminal_reason = machine.terminal_reason
            else:
                terminal_code = getattr(exc, "code", "MOVE_RUNTIME_FAILURE")
                terminal_reason = str(exc)
        finally:
            teardown_complete, teardown_reason = await self._teardown_execution_resources()
            if machine is not None and machine.state in {"TEARDOWN_SUCCESS", "TEARDOWN_FAILURE", "TEARDOWN_CANCEL"}:
                machine.teardown_complete(success=success and teardown_complete)
                success = machine.state == "SUCCEEDED"
                terminal_code = machine.terminal_code
                terminal_reason = machine.terminal_reason
            if not teardown_complete:
                success = False
                terminal_code = "MOVE_TEARDOWN_FAILED"
                terminal_reason = ";".join(filter(None, (terminal_reason, teardown_reason)))
            self._state = State.MOVE_COMPLETE_DISARMED if success else State.FAILED_DISARMED
            self._failure = "" if success else terminal_reason
            self._resource_counts["action_call"] = 0
        self._fill_result(result, success=success, code=terminal_code, reason=terminal_reason, plan=plan, teardown_complete=teardown_complete)
        if success:
            goal_handle.succeed()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        with self._lock:
            self._active_machine = None
            self._goal_slot_reserved = False
            # A terminal Action result consumes and releases the exact plan.
            # A later GO PRE cycle must bind a new q0/session/target and cannot
            # reuse the previous authorization or consumed plan.
            self._last_completed_plan = plan if success else None
            self._plan = None
            self._authorization_id = ""
            self._soft_precontact_plan_hash = ""
            self._soft_standby_proof = {}
            self._state = State.IDLE
        self._publish_status()
        return result

    def _fill_standby_result(
        self, result, *, success: bool, code: str, reason: str,
        plan: StandbyRoutePlan | None, teardown_complete: bool,
        no_motion: bool,
    ) -> None:
        result.success = bool(success)
        result.terminal_code = str(code)
        result.terminal_reason = str(reason)
        result.state = self._state
        result.session_id = "" if plan is None else str(
            plan.payload.get("session_id") or plan.payload.get("ui_request_id") or ""
        )
        result.plan_hash = "" if plan is None else plan.plan_hash
        result.terminal_joint_target_publishers = int(
            self.count_publishers("/roarm_safe/joint_target")
        )
        known, lease_active, armed, feedback_age = self._terminal_resource_evidence()
        result.terminal_precontact_lease_active = lease_active
        result.terminal_armed = armed
        result.teardown_complete = bool(teardown_complete and known)
        result.endpoint_max_joint_error_rad = (
            max(
                abs(a - b)
                for a, b in zip(
                    self._safety_sample().positions_rad,
                    plan.payload["standard_standby_rad"],
                )
            )
            if plan is not None and known else math.inf
        )
        result.terminal_feedback_age_s = float(feedback_age)
        result.waypoint_count = 0 if plan is None else len(plan.payload["waypoints_rad"])
        result.no_motion = bool(no_motion)

    async def _execute_standby(self, goal_handle):
        request = goal_handle.request
        with self._lock:
            plan = self._standby_plan
        result = StandbyRoute.Result()
        machine: ExactPlanExecution | None = None
        terminal_code = "STANDBY_ROUTE_REJECTED"
        terminal_reason = ""
        success = False
        teardown_complete = False
        no_motion = False
        try:
            preflight = self._safety_sample()
            with self._lock:
                if (
                    plan is None or self._model is None or self._move_policy is None
                    or not self._exact_standby_goal_locked(request)
                ):
                    raise StandbyRouteBlocked(
                        "STANDBY_ROUTE_IDENTITY_DRIFT", "after goal admission"
                    )
                ready, reason = self._move_gate()
                if not ready:
                    raise StandbyRouteBlocked(reason)
                consumed = consume_standby_route(
                    plan, model=self._model,
                    current_positions_rad=preflight.positions_rad,
                    now_monotonic=time.monotonic(),
                )
                self._standby_plan = consumed
                plan = consumed
                initial_disarmed_state = str(
                    plan.payload.get("initial_disarmed_state") or ""
                )
                if initial_disarmed_state and (
                    initial_disarmed_state not in {
                        FAULT_CLEARED_DISARMED, "TORQUE_ON_VERIFIED",
                        SOURCE_ROUTE_INITIAL_STATE,
                    }
                    or preflight.state != initial_disarmed_state
                ):
                    raise StandbyRouteBlocked(
                        "STANDBY_ROUTE_START_DRIFT",
                        "initial disarmed state identity drift",
                    )
                if plan.payload["already_at_standby"]:
                    soft_identity = plan.payload.get("soft_standby_identity")
                    soft_state = (
                        str(soft_identity.get("admission_state") or "")
                        if isinstance(soft_identity, dict) else ""
                    )
                    if soft_state and preflight.state != soft_state:
                        raise StandbyRouteBlocked(
                            "STANDBY_ROUTE_START_DRIFT", "soft Standby state drift"
                        )
                    preflight.validate(
                        self._move_policy, expected_publishers=0,
                        expected_lease=False,
                        allow_fault_cleared_disarmed=(
                            soft_state == FAULT_CLEARED_DISARMED
                        ),
                        allow_torque_on_verified=(
                            soft_state == "TORQUE_ON_VERIFIED"
                        ),
                    )
                    no_motion = True
                    success = True
                    teardown_complete = True
                    terminal_code = "EXACT_STANDBY_ALREADY_COMPLETE_NO_MOTION"
                    terminal_reason = "fresh measured q already matches exact Standard Standby"
                    self._state = State.STANDBY_COMPLETE_DISARMED
                else:
                    machine = ExactPlanExecution(
                        plan.payload, self._move_policy, now=time.monotonic(),
                        initial_disarmed_state=(
                            initial_disarmed_state or "CONVERGED_DISARMED"
                        ),
                    )
                    self._active_machine = machine
                    self._resource_counts["action_call"] = 1
                    self._state = State.STANDBY_ACTION_IN_FLIGHT
            if not no_motion:
                machine.precheck(preflight)
                if goal_handle.is_cancel_requested:
                    machine.cancel()
                else:
                    await self._create_execution_resources(goal_handle, machine)
                if machine.state == "TEARDOWN_CANCEL":
                    raise MoveBlocked("MOVE_CANCELLED", "operator cancellation requested")
                while machine.state in {"ARM_WAYPOINT", "PUBLISH_WAYPOINT"}:
                    if goal_handle.is_cancel_requested:
                        machine.cancel()
                        break
                    if standby_canonical_hash(plan.payload) != plan.plan_hash:
                        raise StandbyRouteBlocked("STANDBY_ROUTE_IDENTITY_DRIFT", "plan hash")
                    if standby_canonical_hash(plan.payload.get("model_hashes")) != standby_canonical_hash(self._model.hashes):
                        raise StandbyRouteBlocked("STANDBY_ROUTE_IDENTITY_DRIFT", "model hash")
                    boundary = self._safety_sample()
                    if machine.state == "ARM_WAYPOINT":
                        if (
                            machine.waypoint_index == 0
                            and initial_disarmed_state
                            and boundary.state != initial_disarmed_state
                        ):
                            raise MoveBlocked(
                                "MOVE_SAFETY_GATE_FAILED",
                                "source-route initial disarmed state drift",
                            )
                        boundary.validate(
                            self._move_policy, expected_publishers=1,
                            expected_lease=True,
                            allow_fault_cleared_disarmed=(
                                machine.waypoint_index == 0
                                and (
                                    machine.bootstrap_admission_state
                                    == FAULT_CLEARED_DISARMED
                                    or machine.allow_initial_fault_cleared_disarmed
                                )
                            ),
                            allow_torque_on_verified=(
                                machine.waypoint_index == 0
                                and machine.allow_initial_torque_on_verified
                            ),
                            allow_operator_disarmed=(
                                machine.waypoint_index == 0
                                and machine.allow_initial_operator_disarmed
                            ),
                        )
                        ok, reason = await self._call_arm(
                            True,
                            cancel_requested=lambda: goal_handle.is_cancel_requested,
                        )
                        machine.armed(ok, reason)
                    elif not (
                        boundary.state == "COMMAND_ACTIVE"
                        and boundary.armed
                        and boundary.joint_target_publishers == 1
                        and boundary.lease_active
                        and boundary.lease_purpose == DYNAMIC_LEASE_PURPOSE
                        and boundary.dynamic_exact_no_reissue
                        and boundary.dynamic_exact_advance_ready
                    ):
                        raise MoveBlocked("MOVE_OWNER_DRIFT", "Standby waypoint boundary")
                    if machine.state != "PUBLISH_WAYPOINT":
                        break
                    target = machine.target
                    if not machine.command_within_caps(target, boundary.positions_rad):
                        raise MoveBlocked("MOVE_COMMAND_CAP_EXCEEDED", "Standby pre-publish measured pose")
                    published_at = time.monotonic()
                    prior_sequence = boundary.dynamic_exact_accepted_command_sequence
                    self._publish_exact_waypoint_once(target)
                    machine.mark_published(target, now=published_at)
                    feedback = StandbyRoute.Feedback()
                    feedback.state = machine.state
                    feedback.gate = "EXACT_STANDBY_WAYPOINT_PUBLISHED_ONCE"
                    feedback.waypoint_index = machine.waypoint_index + 1
                    feedback.waypoint_count = len(machine.waypoints)
                    feedback.plan_hash = plan.plan_hash
                    feedback.session_id = str(
                        plan.payload.get("session_id") or plan.payload["ui_request_id"]
                    )
                    goal_handle.publish_feedback(feedback)
                    try:
                        active_sample = await self._wait_sample(
                            lambda value: (
                                value.state == "COMMAND_ACTIVE"
                                and value.armed
                                and value.dynamic_exact_accepted_command_sequence
                                != prior_sequence
                            ),
                            0.75,
                            cancel_requested=lambda: goal_handle.is_cancel_requested,
                            min_received_monotonic=published_at,
                        )
                    except MoveBlocked as exc:
                        if exc.code == "MOVE_FEEDBACK_STALE":
                            raise MoveBlocked(
                                "MOVE_COMMAND_ACK_MISSING",
                                "SafeDriver did not accept Standby waypoint",
                            ) from exc
                        raise
                    validate_command_ack(
                        active_sample,
                        prior_sequence=prior_sequence,
                        expected_target=target,
                        pre_publish_positions=boundary.positions_rad,
                        encoder_step_rad=self._move_policy.encoder_step_rad,
                    )
                    machine.observe(active_sample, now=time.monotonic())
                    while machine.state == "WAIT_CONVERGENCE":
                        if goal_handle.is_cancel_requested:
                            machine.cancel()
                            break
                        machine.observe(self._safety_sample(), now=time.monotonic())
                        if machine.state == "WAIT_CONVERGENCE":
                            await self._ros_sleep(0.02)
                success = machine.state == "TEARDOWN_SUCCESS"
                terminal_code = machine.terminal_code
                terminal_reason = machine.terminal_reason
        except Exception as exc:
            if machine is not None:
                if goal_handle.is_cancel_requested:
                    machine.cancel()
                else:
                    machine.fail(getattr(exc, "code", "STANDBY_RUNTIME_FAILURE"), str(exc))
                terminal_code = machine.terminal_code
                terminal_reason = machine.terminal_reason
            else:
                terminal_code = getattr(exc, "code", "STANDBY_RUNTIME_FAILURE")
                terminal_reason = str(exc)
        finally:
            if not no_motion:
                teardown_complete, teardown_reason = await self._teardown_execution_resources()
                if machine is not None and machine.state in {
                    "TEARDOWN_SUCCESS", "TEARDOWN_FAILURE", "TEARDOWN_CANCEL",
                }:
                    machine.teardown_complete(success=success and teardown_complete)
                    success = machine.state == "SUCCEEDED"
                    terminal_code = machine.terminal_code
                    terminal_reason = machine.terminal_reason
                if not teardown_complete:
                    success = False
                    terminal_code = "MOVE_TEARDOWN_FAILED"
                    terminal_reason = ";".join(filter(None, (terminal_reason, teardown_reason)))
            self._state = (
                State.STANDBY_COMPLETE_DISARMED if success
                else State.FAILED_DISARMED
            )
            self._failure = "" if success else terminal_reason
            self._resource_counts["action_call"] = 0
        self._fill_standby_result(
            result, success=success, code=terminal_code, reason=terminal_reason,
            plan=plan, teardown_complete=teardown_complete, no_motion=no_motion,
        )
        if success:
            goal_handle.succeed()
        elif goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        with self._lock:
            self._active_machine = None
            self._goal_slot_reserved = False
            self._standby_plan = None
            self._standby_authorization_id = ""
            if success:
                self._last_completed_plan = None
                soft_identity = (
                    plan.payload.get("soft_standby_identity")
                    if plan is not None and no_motion else None
                )
                self._soft_standby_proof = (
                    {
                        **dict(soft_identity),
                        "executor_boot_epoch": self._model.boot_epoch,
                        "standby_plan_hash": plan.plan_hash,
                    }
                    if isinstance(soft_identity, dict) else {}
                )
                self._soft_precontact_plan_hash = ""
            self._state = State.IDLE
        self._publish_status()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = DynamicExecutor()
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
