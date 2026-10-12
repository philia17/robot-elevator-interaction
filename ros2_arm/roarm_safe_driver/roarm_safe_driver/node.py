"""ROS 2 node exposing the fail-closed RoArm-M3 safe driver."""

from __future__ import annotations

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from .core import (
    DEFAULT_JOINT_ACC_REGISTERS,
    DEFAULT_JOINT_MAX_DELTA_RAD,
    DEFAULT_JOINT_SPEED_STEPS_S,
    EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES,
    DYNAMIC_PRECONTACT_PID_I,
    DYNAMIC_PRECONTACT_PID_JOINT_IDS,
    DYNAMIC_PRECONTACT_PID_P,
    ENCODER_STEP_RAD,
    FEEDBACK_REQUEST,
    IMPLAUSIBLE_JOINT_LIMITS_ERROR,
    JOINT_NAMES,
    PrecontactSpeedLease,
    SafeController,
    SafetyError,
    T101_COMMAND_LIMITS,
    TERMINAL_MIN_SETTLE_SAMPLES,
    count_moving_joints,
    build_joint_pid,
    build_reset_joint_pid,
    ordered_positions,
    parse_feedback_line,
    validate_command_age,
)
from .transport import (
    FakeSerialTransport,
    SerialTransport,
    TrackingFakeSerialTransport,
)
from .torque_transaction import (
    TORQUE_FEEDBACK_VERIFY_TIMEOUT_S,
    TorqueFeedbackTracker,
    has_recent_direct_feedback,
)

try:
    from elevator_interfaces.srv import (
        PrecontactSpeedLease as PrecontactSpeedLeaseSrv,
    )
except ImportError:
    # rosidl generation is unavailable in this checkout (no ROS toolchain);
    # the service is deliberately not registered so lease requests fail
    # closed.  AGX builds generate the real interface from
    # elevator_interfaces/srv/PrecontactSpeedLease.srv.
    PrecontactSpeedLeaseSrv = None


class _PrecontactSpeedLeaseContract:
    """Source-level service contract mirroring PrecontactSpeedLease.srv.

    Kept as documentation and for source tests only; it is never passed to
    rclpy when the generated interface is unavailable.
    """

    class Request:
        def __init__(self) -> None:
            self.enable = False
            self.purpose = ""
            self.lease_id = ""

    class Response:
        def __init__(self) -> None:
            self.success = False
            self.message = ""
            self.active = False


class SafeDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("roarm_safe_driver")
        self.declare_parameter("transport", "none")
        self.declare_parameter("serial_port", "")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("fake_feedback", False)
        # ROS-frame start pose for the tracking fake transport (isolated
        # rehearsals only; ignored by other transports).
        self.declare_parameter("fake_feedback_pose_rad", [0.0, 0.0, 1.6, 0.0, 0.0, 0.0])
        self.declare_parameter("request_feedback_once_on_start", False)
        # Some controller revisions do not maintain an unsolicited T:1051
        # stream.  A positive poll rate issues read-only T:105 requests so
        # torque faults can still be detected and evidenced promptly.
        self.declare_parameter("feedback_poll_hz", 0.0)
        self.declare_parameter("joint_speed_steps_s", list(DEFAULT_JOINT_SPEED_STEPS_S))
        self.declare_parameter("joint_acc_registers", list(DEFAULT_JOINT_ACC_REGISTERS))
        self.declare_parameter(
            "joint_max_delta_deg",
            [math.degrees(value) for value in DEFAULT_JOINT_MAX_DELTA_RAD],
        )
        # The real AGX T:1051 stream is about 0.63 s; keep the runtime default
        # above one observed inter-frame gap even if a profile is omitted.
        self.declare_parameter("feedback_timeout_s", 1.0)
        self.declare_parameter("command_timeout_s", 5.0)
        self.declare_parameter("tolerance_deg", 0.5)
        self.declare_parameter("convergence_samples", TERMINAL_MIN_SETTLE_SAMPLES)
        # Disabled-by-default endpoint binding for the narrow coarse T102
        # lease.  The ordinary lease and every other safe-driver user ignore
        # this value.
        self.declare_parameter("coarse_precontact_enabled", False)
        self.declare_parameter("coarse_precontact_endpoint_rad", [0.0] * 6)
        self.declare_parameter("startup_restore_enabled", False)
        self.declare_parameter("startup_restore_target_rad", [0.0] * 6)
        self.declare_parameter("startup_restore_initialize_rad", [0.0] * 6)
        self.declare_parameter("startup_restore_initialize_tolerance_deg", 8.0)
        # Explicit operator-requested return to the configured project
        # standby.  This is a service gate only: it is not called at startup
        # and it never substitutes the vendor initialize pose.
        self.declare_parameter("manual_standby_return_enabled", False)
        self.declare_parameter("manual_standby_speed_steps_s", 34)
        # Explicit operator-requested return to the configured vendor
        # initialize pose. This is a service gate only; it is never called at
        # startup and never substitutes standby during fault recovery.
        self.declare_parameter("manual_initialize_return_enabled", False)
        # A firmware torque-switch dropout terminates a motion in
        # ``SafeController``.  On the real robot we can then make exactly one
        # measured Torque-ON recovery attempt, but must never resume motion
        # automatically.  An operator's explicit Torque-OFF hand-teach
        # request remains authoritative until that operator turns Torque ON.
        self.declare_parameter("auto_torque_recovery_enabled", False)
        self.declare_parameter("auto_torque_recovery_delay_s", 1.0)
        self.declare_parameter("auto_torque_recovery_healthy_reset_s", 30.0)
        # Disabled in every repository default.  A supervised Dynamic exact
        # release may opt into the vendor's lowest non-zero integral setting
        # (P16/I8) to remove the measured load-induced steady-state error.
        self.declare_parameter("dynamic_exact_pid_boost_enabled", False)

        transport_mode = str(self.get_parameter("transport").value)
        dry_run = bool(self.get_parameter("dry_run").value)
        if transport_mode == "serial":
            self.transport = SerialTransport(
                str(self.get_parameter("serial_port").value),
                int(self.get_parameter("baud_rate").value),
            )
        elif transport_mode == "fake":
            self.transport = FakeSerialTransport()
        elif transport_mode == "tracking_fake":
            self.transport = TrackingFakeSerialTransport(
                [float(v) for v in self.get_parameter("fake_feedback_pose_rad").value]
            )
        elif transport_mode == "none" and dry_run:
            self.transport = FakeSerialTransport()
        else:
            raise RuntimeError(
                "transport must be fake/tracking_fake/none for hardware-isolated "
                "runs or serial for hardware"
            )

        self.controller = SafeController(
            self.transport.write,
            dry_run=dry_run,
            joint_speed_steps_s=tuple(self.get_parameter("joint_speed_steps_s").value),
            joint_acc_registers=tuple(self.get_parameter("joint_acc_registers").value),
            joint_max_delta_rad=tuple(
                math.radians(value)
                for value in self.get_parameter("joint_max_delta_deg").value
            ),
            feedback_timeout_s=float(self.get_parameter("feedback_timeout_s").value),
            command_timeout_s=float(self.get_parameter("command_timeout_s").value),
            tolerance_rad=math.radians(
                float(self.get_parameter("tolerance_deg").value)
            ),
            convergence_samples=int(self.get_parameter("convergence_samples").value),
        )
        if bool(self.get_parameter("coarse_precontact_enabled").value):
            raw_endpoint = tuple(
                float(value)
                for value in self.get_parameter(
                    "coarse_precontact_endpoint_rad"
                ).value
            )
            if len(raw_endpoint) != len(JOINT_NAMES):
                raise RuntimeError("coarse precontact endpoint must contain six joints")
            if any(not math.isfinite(value) for value in raw_endpoint):
                raise RuntimeError("coarse precontact endpoint must be finite")
            if any(
                not lower <= value <= upper
                for value, (lower, upper) in zip(
                    raw_endpoint, T101_COMMAND_LIMITS
                )
            ):
                raise RuntimeError("coarse precontact endpoint violates command limits")
            self.controller.coarse_progress_endpoint = raw_endpoint
        self._controller_lock = threading.RLock()
        self._precontact_lease = PrecontactSpeedLease()
        self._dynamic_exact_pid_boost_active = False
        self._torque_transaction_lock = threading.Lock()
        self._torque_feedback = TorqueFeedbackTracker()
        self._auto_torque_recovery_lock = threading.Lock()
        self._auto_torque_recovery_attempted = False
        self._auto_torque_recovery_healthy_since: float | None = None
        self._auto_torque_recovery_manual_off_in_progress = False
        self._auto_torque_recovery_operator_off_latched = False
        self.transport.open()
        if bool(self.get_parameter("request_feedback_once_on_start").value):
            if transport_mode != "serial" or dry_run:
                self.transport.close()
                raise RuntimeError(
                    "one-shot feedback request requires non-dry-run serial transport"
                )
            self.transport.write(FEEDBACK_REQUEST)
        self._stop = threading.Event()
        self._reader = threading.Thread(
            target=self._reader_loop,
            name="roarm-feedback",
            daemon=False,
        )

        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(
            String, "/roarm_safe/status", status_qos
        )
        feedback_status_qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.feedback_status_pub = self.create_publisher(
            String, "/roarm_safe/feedback_status", feedback_status_qos
        )
        self.command_sub = self.create_subscription(
            JointState, "/roarm_safe/joint_target", self._command_callback, 10
        )
        self.arm_srv = self.create_service(
            SetBool, "/roarm_safe/set_armed", self._arm_callback
        )
        self.torque_srv = self.create_service(
            SetBool,
            "/roarm_safe/set_torque",
            self._torque_callback,
        )
        self.clear_srv = self.create_service(
            Trigger,
            "/roarm_safe/clear_fault",
            self._clear_callback,
        )
        self.restore_standby_srv = self.create_service(
            Trigger,
            "/roarm_safe/restore_startup_standby",
            self._restore_startup_standby_callback,
        )
        self.manual_standby_srv = self.create_service(
            Trigger,
            "/roarm_safe/return_to_configured_standby",
            self._return_to_configured_standby_callback,
        )
        self.manual_initialize_srv = self.create_service(
            Trigger,
            "/roarm_safe/return_to_configured_initialize",
            self._return_to_configured_initialize_callback,
        )
        # Single-owner, non-replayable speed lease for the supervised
        # outbound-only T102 trial.  Registered only when the generated
        # interface exists; otherwise every lease call fails closed with an
        # unavailable service.
        self.precontact_lease_srv = None
        if PrecontactSpeedLeaseSrv is not None:
            self.precontact_lease_srv = self.create_service(
                PrecontactSpeedLeaseSrv,
                "/roarm_safe/precontact_speed_lease",
                self._precontact_lease_callback,
            )
        self._last_status_publish_monotonic = 0.0
        self.timer = self.create_timer(0.1, self._timer_callback)
        self._reader.start()
        self._fake_timer = None
        self._feedback_poll_timer = None
        if transport_mode in ("fake", "tracking_fake") and bool(
            self.get_parameter("fake_feedback").value
        ):
            self._fake_timer = self.create_timer(0.1, self._inject_fake_feedback)
        poll_hz = float(self.get_parameter("feedback_poll_hz").value)
        if transport_mode == "serial" and not dry_run and poll_hz > 0.0:
            if not math.isfinite(poll_hz) or poll_hz > 20.0:
                raise RuntimeError(
                    "feedback_poll_hz must be finite and no greater than 20"
                )
            self._feedback_poll_timer = self.create_timer(
                1.0 / poll_hz,
                self._request_feedback_poll,
            )
        self._publish_status("STARTED_DISARMED")

    def _inject_fake_feedback(self) -> None:
        if isinstance(self.transport, TrackingFakeSerialTransport):
            self.transport.lines.put(self.transport.sample_line())
            return
        sample = {
            "T": 1051,
            "b": 0.0,
            "s": 0.0,
            "e": 1.6,
            "t": 0.0,
            "r": 0.0,
            "g": 3.141592653589793,
            "tB": 0,
            "tS": 0,
            "tE": 0,
            "tT": 0,
            "tR": 0,
            "tG": 0,
            "torswitchB": 1,
            "torswitchS": 1,
            "torswitchE": 1,
            "torswitchT": 1,
            "torswitchR": 1,
            "torswitchG": 1,
            "v": 1210,
        }
        self.transport.lines.put((json.dumps(sample) + "\n").encode())

    def _request_feedback_poll(self) -> None:
        """Issue a read-only firmware state request; no torque or motion write."""

        try:
            self.transport.write(FEEDBACK_REQUEST)
        except Exception as exc:
            if not self._stop.is_set():
                with self._controller_lock:
                    self.controller.latch_fault(f"feedback poll: {exc}")
                self._publish_status(self.controller.last_result)

    def _reader_loop(self) -> None:
        seen_feedback = False
        while not self._stop.is_set():
            try:
                line = self.transport.readline()
                if not line:
                    continue
                feedback = parse_feedback_line(line)
                if feedback is None:
                    continue
                seen_feedback = True
                feedback_time = self.get_clock().now()
                # Record immediately after parsing, before any controller work.
                # A torque transaction still asks T:105 after T:210 and scans
                # every newer record until its bounded verification deadline.
                self._torque_feedback.record(feedback.torque_enabled)
                with self._controller_lock:
                    before = self.controller.last_result
                    was_active = self.controller.target is not None
                    self.controller.accept_feedback(feedback)
                    after = self.controller.last_result
                    feedback_status = {
                        "state": after,
                        "armed": self.controller.armed,
                        "dry_run": self.controller.dry_run,
                        "fault": self.controller.fault or "",
                        "feedback_stamp_ns": feedback_time.nanoseconds,
                        "positions_rad": list(feedback.positions),
                        "torque_enabled": list(feedback.torque_enabled),
                        "voltage": feedback.voltage,
                        "dynamic_exact_accepted_command_sequence": int(
                            self.controller.dynamic_exact_accepted_command_sequence
                        ),
                        "dynamic_exact_accepted_target_positions": list(
                            self.controller.dynamic_exact_accepted_target_positions
                        ),
                        "dynamic_exact_advance_ready": bool(
                            self.controller.dynamic_exact_advance_ready(
                                time.monotonic()
                            )
                        ),
                    }
                self._observe_auto_torque_recovery(feedback.torque_enabled)
                msg = JointState()
                msg.header.stamp = feedback_time.to_msg()
                msg.name = list(JOINT_NAMES)
                msg.position = list(feedback.positions)
                msg.effort = [float(value) for value in feedback.loads]
                self.joint_pub.publish(msg)
                feedback_status_msg = String()
                feedback_status_msg.data = json.dumps(
                    feedback_status, separators=(",", ":"), allow_nan=False
                )
                self.feedback_status_pub.publish(feedback_status_msg)
                if after != before or was_active:
                    self._publish_status(after)
            except SafetyError as exc:
                # Drop only these known single-frame feedback anomalies without
                # a sticky latch. A continuing stream of either anomaly does
                # not count as accepted feedback, so the stale watchdog still
                # latches the driver when the last accepted frame ages out.
                if str(exc) == IMPLAUSIBLE_JOINT_LIMITS_ERROR:
                    continue
                if str(exc) == "invalid torque status":
                    continue
                if seen_feedback:
                    with self._controller_lock:
                        self.controller.latch_fault(f"invalid feedback: {exc}")
                    self._publish_status(self.controller.last_result)
            except Exception as exc:  # serial disconnect and shutdown races fail closed
                if not self._stop.is_set():
                    with self._controller_lock:
                        self.controller.latch_fault(f"serial reader: {exc}")
                    self._publish_status(self.controller.last_result)
                break

    def _command_callback(self, msg: JointState) -> None:
        try:
            if self.count_publishers("/roarm_safe/joint_target") != 1:
                raise SafetyError("exactly one command publisher is required")
            stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            validate_command_age(
                stamp_s,
                self.get_clock().now().nanoseconds / 1e9,
            )
            target = ordered_positions(msg.name, msg.position)
            with self._controller_lock:
                # Dispatch by moving-joint count against the latest feedback:
                # exactly one moving joint keeps the strict single-joint T101
                # path (with its non-target-drift convergence model); two or
                # more moving joints use the multi-joint T102 path.  Without
                # feedback both paths fail closed on the health gate.
                # While a multi-joint command is already active, accept a
                # delta-limited replace so hold teleop can stream without
                # disarming between short tip segments.
                feedback = self.controller.feedback
                if feedback is not None:
                    moving = count_moving_joints(target, feedback.positions)
                else:
                    moving = 1
                if self.controller.target is not None:
                    if self.controller.exact_plan_no_reissue_active:
                        payload = self.controller.command_dynamic_exact_advance(
                            target, time.monotonic()
                        )
                    else:
                        payload = self.controller.command_multi_replace(
                            target, time.monotonic()
                        )
                elif self.controller.exact_plan_no_reissue_active:
                    # The isolated frozen-plan lease always selects one
                    # absolute T102, even when only one joint differs from the
                    # latest measured sample. It may never fall through to a
                    # T101 or replace-stream path.
                    payload = self.controller.command_multi(target, time.monotonic())
                elif moving > 1:
                    payload = self.controller.command_multi(target, time.monotonic())
                else:
                    payload = self.controller.command(target, time.monotonic())
            self._publish_status(self.controller.last_result, payload.decode().strip())
        except SafetyError as exc:
            self._publish_status(f"COMMAND_REJECTED:{exc}")

    def _arm_callback(self, request: SetBool.Request, response: SetBool.Response):
        try:
            if request.data:
                with self._controller_lock:
                    self.controller.arm(time.monotonic())
            else:
                with self._controller_lock:
                    self.controller.disarm("operator", now=time.monotonic())
            response.success = True
            response.message = self.controller.last_result
        except SafetyError as exc:
            response.success = False
            response.message = str(exc)
        self._publish_status(response.message)
        return response

    def _clear_callback(self, _request: Trigger.Request, response: Trigger.Response):
        with self._controller_lock:
            self.controller.clear_fault()
        response.success = True
        response.message = self.controller.last_result
        self._publish_status(response.message)
        return response

    def _restore_startup_standby_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ):
        """Send the configured startup standby only from verified initialize.

        This removes the ROS publisher/arm deadline race while retaining the
        driver's feedback health, single-command, convergence, and fault-stop
        mechanisms.  The exceptional large delta is scoped to this one named
        service call and is restored before the callback returns.
        """
        try:
            if not bool(self.get_parameter("startup_restore_enabled").value):
                raise SafetyError("startup standby restore is disabled")
            target = tuple(
                float(value)
                for value in self.get_parameter("startup_restore_target_rad").value
            )
            initialize = tuple(
                float(value)
                for value in self.get_parameter("startup_restore_initialize_rad").value
            )
            if len(target) != len(JOINT_NAMES) or len(initialize) != len(JOINT_NAMES):
                raise SafetyError("startup standby vectors must contain six joints")
            if not all(math.isfinite(value) for value in target + initialize):
                raise SafetyError("startup standby vectors must be finite")
            tolerance = math.radians(
                float(
                    self.get_parameter("startup_restore_initialize_tolerance_deg").value
                )
            )
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise SafetyError("startup initialize tolerance is invalid")
            with self._controller_lock:
                feedback = self.controller.feedback
                now = time.monotonic()
                if feedback is None or not self.controller.healthy(now):
                    raise SafetyError(
                        "startup standby requires healthy recent feedback"
                    )
                if self._precontact_lease.active:
                    raise SafetyError(
                        "startup standby is refused while the precontact speed "
                        "lease is active"
                    )
                if self.controller.armed or self.controller.target is not None:
                    raise SafetyError("startup standby requires disarmed idle driver")
                if any(
                    abs(current - expected) > tolerance
                    for current, expected in zip(feedback.positions, initialize)
                ):
                    raise SafetyError(
                        "current pose does not match configured initialize"
                    )
                original_limits = self.controller.joint_max_delta_rad
                widened_limits = tuple(
                    max(limit, abs(goal - current) + ENCODER_STEP_RAD + 1e-6)
                    for limit, goal, current in zip(
                        original_limits, target, feedback.positions
                    )
                )
                self.controller.arm(now)
                try:
                    self.controller.joint_max_delta_rad = widened_limits
                    payload = self.controller.command_multi(target, now)
                finally:
                    self.controller.joint_max_delta_rad = original_limits
            response.success = True
            response.message = "STARTUP_STANDBY_COMMAND_ACTIVE"
            self._publish_status(response.message, payload.decode().strip())
        except (SafetyError, TypeError, ValueError) as exc:
            # A service callback may have armed the one-shot gate before
            # target validation rejects the request.  Close that gate here so
            # the timer does not overwrite the useful rejection reason with a
            # later ARM_COMMAND_TIMEOUT_DISARMED status.
            with self._controller_lock:
                if self.controller.armed and self.controller.target is None:
                    self.controller.disarm("service_rejected")
            response.success = False
            response.message = str(exc)
            self._publish_status(f"STARTUP_STANDBY_REJECTED:{exc}")
        return response

    def _return_to_configured_standby_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ):
        """Send one explicit multi-joint command to the configured standby.

        Unlike the startup restore service, this deliberately has no vendor-
        initialize pose requirement.  It still requires current healthy direct
        feedback, an idle/disarmed driver, and uses the configured standby as
        its only target.  The temporary delta expansion is scoped to this
        service callback and restored before it returns.
        """

        try:
            if not bool(self.get_parameter("manual_standby_return_enabled").value):
                raise SafetyError("manual standby return is disabled")
            target = tuple(
                float(value)
                for value in self.get_parameter("startup_restore_target_rad").value
            )
            if len(target) != len(JOINT_NAMES) or not all(
                math.isfinite(value) for value in target
            ):
                raise SafetyError(
                    "configured standby target must contain six finite joints"
                )
            speed = int(self.get_parameter("manual_standby_speed_steps_s").value)
            if not 1 <= speed <= max(DEFAULT_JOINT_SPEED_STEPS_S):
                raise SafetyError(
                    "manual standby speed must be between 1 and "
                    f"{max(DEFAULT_JOINT_SPEED_STEPS_S)} steps/s"
                )
            with self._controller_lock:
                feedback = self.controller.feedback
                now = time.monotonic()
                if feedback is None or not self.controller.healthy(now):
                    raise SafetyError("manual standby requires healthy recent feedback")
                if self._precontact_lease.active:
                    raise SafetyError(
                        "manual standby is refused while the precontact speed "
                        "lease is active"
                    )
                if self.controller.armed or self.controller.target is not None:
                    raise SafetyError("manual standby requires disarmed idle driver")
                original_limits = self.controller.joint_max_delta_rad
                original_speeds = self.controller.joint_speed_steps_s
                widened_limits = tuple(
                    max(limit, abs(goal - current) + ENCODER_STEP_RAD + 1e-6)
                    for limit, goal, current in zip(
                        original_limits, target, feedback.positions
                    )
                )
                self.controller.arm(now)
                try:
                    self.controller.joint_max_delta_rad = widened_limits
                    self.controller.joint_speed_steps_s = (speed,) * len(JOINT_NAMES)
                    payload = self.controller.command_multi(target, now)
                finally:
                    self.controller.joint_max_delta_rad = original_limits
                    self.controller.joint_speed_steps_s = original_speeds
            response.success = True
            response.message = f"CONFIGURED_STANDBY_COMMAND_ACTIVE:spd={speed}"
            self._publish_status(response.message, payload.decode().strip())
        except (SafetyError, TypeError, ValueError) as exc:
            with self._controller_lock:
                if self.controller.armed and self.controller.target is None:
                    self.controller.disarm("service_rejected")
            response.success = False
            response.message = str(exc)
            self._publish_status(f"CONFIGURED_STANDBY_REJECTED:{exc}")
        return response

    def _return_to_configured_initialize_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ):
        """Send one explicit multi-joint command to vendor initialize.

        This is deliberately separate from standby recovery. It requires a
        healthy disarmed driver and an explicit service call, widens the
        per-command delta only for this one command, then restores the normal
        limits immediately. It never clears faults or changes Torque state.
        """

        try:
            if not bool(self.get_parameter("manual_initialize_return_enabled").value):
                raise SafetyError("configured initialize return is disabled")
            target = tuple(
                float(value)
                for value in self.get_parameter("startup_restore_initialize_rad").value
            )
            if len(target) != len(JOINT_NAMES) or not all(
                math.isfinite(value) for value in target
            ):
                raise SafetyError(
                    "configured initialize target must contain six finite joints"
                )
            speed = int(self.get_parameter("manual_standby_speed_steps_s").value)
            if not 1 <= speed <= max(DEFAULT_JOINT_SPEED_STEPS_S):
                raise SafetyError(
                    "initialize return speed must be between 1 and "
                    f"{max(DEFAULT_JOINT_SPEED_STEPS_S)} steps/s"
                )
            with self._controller_lock:
                feedback = self.controller.feedback
                now = time.monotonic()
                if feedback is None or not self.controller.healthy(now):
                    raise SafetyError(
                        "configured initialize return requires healthy recent feedback"
                    )
                if self._precontact_lease.active:
                    raise SafetyError(
                        "configured initialize return is refused while the "
                        "precontact speed lease is active"
                    )
                if self.controller.armed or self.controller.target is not None:
                    raise SafetyError(
                        "configured initialize return requires disarmed idle driver"
                    )
                original_limits = self.controller.joint_max_delta_rad
                original_speeds = self.controller.joint_speed_steps_s
                widened_limits = tuple(
                    max(limit, abs(goal - current) + ENCODER_STEP_RAD + 1e-6)
                    for limit, goal, current in zip(
                        original_limits, target, feedback.positions
                    )
                )
                self.controller.arm(now)
                try:
                    self.controller.joint_max_delta_rad = widened_limits
                    self.controller.joint_speed_steps_s = (speed,) * len(JOINT_NAMES)
                    payload = self.controller.command_multi(target, now)
                finally:
                    self.controller.joint_max_delta_rad = original_limits
                    self.controller.joint_speed_steps_s = original_speeds
            response.success = True
            response.message = f"CONFIGURED_INITIALIZE_COMMAND_ACTIVE:spd={speed}"
            self._publish_status(response.message, payload.decode().strip())
        except (SafetyError, TypeError, ValueError) as exc:
            with self._controller_lock:
                if self.controller.armed and self.controller.target is None:
                    self.controller.disarm("service_rejected")
            response.success = False
            response.message = str(exc)
            self._publish_status(f"CONFIGURED_INITIALIZE_REJECTED:{exc}")
        return response

    def _precontact_lease_callback(self, request, response):
        """Open/close the single-owner T102 speed lease, fail closed.

        The request carries no speed: enable applies the fixed
        ``PRECONTACT_LEASE_SPEED_STEPS_S`` profile after the node-side gates
        (healthy non-dry-run feedback, disarmed idle controller, torque flags,
        zero foreign command publishers) and the pure lease manager's own
        identity/replay checks pass.  Release restores the exact captured
        tuple only for the same purpose/lease identity while disarmed and
        idle.  A failed release keeps the lease active so the elevated speed
        is never silently dropped outside the owner's control.
        """
        with self._controller_lock:
            try:
                if request.enable:
                    ok, message = self._precontact_lease.enable(
                        self.controller,
                        purpose=str(request.purpose),
                        lease_id=str(request.lease_id),
                        foreign_publishers=(
                            self.count_publishers("/roarm_safe/joint_target") != 0
                        ),
                    )
                    if (
                        ok
                        and str(request.purpose)
                        in EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES
                    ):
                        if not bool(
                            self.get_parameter(
                                "dynamic_exact_pid_boost_enabled"
                            ).value
                        ):
                            self._precontact_lease.release(
                                self.controller,
                                purpose=str(request.purpose),
                                lease_id=str(request.lease_id),
                            )
                            ok = False
                            message = "Dynamic exact PID correction is disabled"
                        else:
                            try:
                                for joint_id in DYNAMIC_PRECONTACT_PID_JOINT_IDS:
                                    self.transport.write(
                                        build_joint_pid(
                                            joint_id,
                                            p=DYNAMIC_PRECONTACT_PID_P,
                                            i=DYNAMIC_PRECONTACT_PID_I,
                                        )
                                    )
                                self._dynamic_exact_pid_boost_active = True
                                message = (
                                    "POST_PRECONTACT_XY_EXACT_PID_P16_I8_LEASE_ACTIVE"
                                    if str(request.purpose).startswith(
                                        "T102_POST_PRECONTACT_XY_"
                                    )
                                    else "DYNAMIC_PRECONTACT_EXACT_PID_P16_I8_LEASE_ACTIVE"
                                )
                            except Exception as exc:
                                try:
                                    self.transport.write(build_reset_joint_pid())
                                except Exception:
                                    pass
                                self.controller.latch_fault(
                                    f"Dynamic exact PID apply failed: {exc}"
                                )
                                ok = False
                                message = f"Dynamic exact PID apply failed: {exc}"
                else:
                    ok, message = self._precontact_lease.release_precheck(
                        self.controller,
                        purpose=str(request.purpose),
                        lease_id=str(request.lease_id),
                    )
                    if ok:
                        if (
                            self._dynamic_exact_pid_boost_active
                            and str(request.purpose)
                            in EXACT_PLAN_NO_REISSUE_LEASE_PURPOSES
                        ):
                            self.transport.write(build_reset_joint_pid())
                            self._dynamic_exact_pid_boost_active = False
                        ok, message = self._precontact_lease.release(
                            self.controller,
                            purpose=str(request.purpose),
                            lease_id=str(request.lease_id),
                        )
            except Exception as exc:
                ok, message = False, f"lease service failure: {exc}"
            response.success = ok
            response.message = message
            response.active = self._precontact_lease.active
            state = message if ok else f"PRECONTACT_LEASE_REJECTED:{message}"
            self._publish_status(state)
        return response

    def _observe_auto_torque_recovery(self, torque_enabled: tuple[bool, ...]) -> None:
        """Schedule one verified, no-motion Torque-ON recovery per episode."""

        now = time.monotonic()
        if all(torque_enabled):
            with self._auto_torque_recovery_lock:
                if not self._auto_torque_recovery_attempted:
                    return
                if self._auto_torque_recovery_healthy_since is None:
                    self._auto_torque_recovery_healthy_since = now
                elif now - self._auto_torque_recovery_healthy_since >= float(
                    self.get_parameter("auto_torque_recovery_healthy_reset_s").value
                ):
                    # A new physical dropout after a sustained healthy,
                    # disarmed interval is a new episode.  It receives one
                    # new recovery attempt, never a retry loop.
                    self._auto_torque_recovery_attempted = False
                    self._auto_torque_recovery_healthy_since = None
            return

        if not bool(self.get_parameter("auto_torque_recovery_enabled").value):
            return
        with self._controller_lock:
            eligible = (
                not self.controller.dry_run
                and not self.controller.armed
                and self.controller.target is None
                and self.controller.fault == "one or more joint torque switches are off"
            )
        if not eligible:
            return
        with self._auto_torque_recovery_lock:
            if (
                self._auto_torque_recovery_attempted
                or self._auto_torque_recovery_manual_off_in_progress
                or self._auto_torque_recovery_operator_off_latched
            ):
                return
            self._auto_torque_recovery_attempted = True
            self._auto_torque_recovery_healthy_since = None
        threading.Thread(
            target=self._auto_torque_recovery_worker,
            name="roarm-auto-torque-recovery",
            daemon=True,
        ).start()

    def _auto_torque_recovery_worker(self) -> None:
        """Make one measured Torque-ON attempt after a torque fault.

        This intentionally does not arm the controller, clear a non-torque
        fault, or reissue a target.  A later healthy feedback frame clears the
        recoverable torque-switch latch while the driver remains disarmed.
        """

        delay_s = float(self.get_parameter("auto_torque_recovery_delay_s").value)
        if not math.isfinite(delay_s) or delay_s < 0.0:
            delay_s = 1.0
        if self._stop.wait(delay_s):
            return
        self._publish_status("AUTO_TORQUE_ON_ATTEMPT_DISARMED")
        success, message, payload = self._set_torque_state(
            True,
            require_no_command_publishers=False,
        )
        state = (
            "AUTO_TORQUE_ON_VERIFIED_DISARMED"
            if success
            else f"AUTO_TORQUE_ON_UNVERIFIED_DISARMED:{message}"
        )
        self._publish_status(state, payload)

    def _set_torque_state(
        self,
        requested: bool,
        *,
        require_no_command_publishers: bool,
    ) -> tuple[bool, str, str]:
        """Request Torque ON/OFF and verify it against a later T:1051 frame."""

        cmd = 1 if requested else 0
        payload = (
            json.dumps({"T": 210, "cmd": cmd}, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("ascii")
        label = "ON" if requested else "OFF"
        try:
            with self._torque_transaction_lock:
                if (
                    require_no_command_publishers
                    and self.count_publishers("/roarm_safe/joint_target") != 0
                ):
                    raise SafetyError(
                        "refuse torque change while joint_target publishers exist"
                    )
                feedback_sequence = None
                with self._controller_lock:
                    dry_run = self.controller.dry_run
                    if self.controller.target is not None:
                        raise SafetyError(
                            "refuse torque change while a command is active"
                        )
                    feedback = self.controller.feedback
                    if not dry_run and not has_recent_direct_feedback(
                        None if feedback is None else feedback.received_at,
                        now=time.monotonic(),
                        timeout_s=self.controller.feedback_timeout_s,
                    ):
                        raise SafetyError(
                            "refuse torque change without recent direct T:1051 feedback"
                        )
                    if self.controller.armed:
                        self.controller.disarm("torque_service")
                    if dry_run:
                        message = f"DRY_RUN_TORQUE_{label}"
                        return True, message, payload.decode("ascii").strip()
                    feedback_sequence = self._torque_feedback.snapshot()
                    self.transport.write(payload)
                    self.transport.write(FEEDBACK_REQUEST)
                assert feedback_sequence is not None
                verification = self._torque_feedback.wait_for_requested_state(
                    after_sequence=feedback_sequence,
                    requested=requested,
                )
                if verification.verified:
                    success, message = True, f"TORQUE_{label}_VERIFIED"
                elif verification.reason == "mismatch":
                    observed = list(verification.observed_flags or ())
                    success = False
                    message = (
                        f"TORQUE_{label}_UNVERIFIED: measured torque_enabled={observed}"
                    )
                else:
                    success = False
                    message = (
                        f"TORQUE_{label}_UNVERIFIED: no fresh T:1051 feedback within "
                        f"{TORQUE_FEEDBACK_VERIFY_TIMEOUT_S:.1f}s"
                    )
                with self._controller_lock:
                    self.controller.last_result = message
                return success, message, payload.decode("ascii").strip()
        except Exception as exc:
            return False, str(exc), payload.decode("ascii").strip()

    def _torque_callback(self, request: SetBool.Request, response: SetBool.Response):
        """Hand-teach torque lock via firmware T:210 (does not arm motion).

        request.data True  -> torque ON  ({"T":210,"cmd":1})
        request.data False -> torque OFF ({"T":210,"cmd":0})
        """
        if not request.data:
            with self._auto_torque_recovery_lock:
                self._auto_torque_recovery_manual_off_in_progress = True
        response.success, response.message, outgoing = self._set_torque_state(
            bool(request.data),
            require_no_command_publishers=True,
        )
        with self._auto_torque_recovery_lock:
            if request.data:
                if response.success:
                    self._auto_torque_recovery_operator_off_latched = False
            else:
                self._auto_torque_recovery_manual_off_in_progress = False
                if response.success:
                    self._auto_torque_recovery_operator_off_latched = True
        state = (
            response.message
            if response.success
            else f"TORQUE_REJECTED:{response.message}"
        )
        self._publish_status(state, outgoing)
        return response

    def _timer_callback(self) -> None:
        now = time.monotonic()
        with self._controller_lock:
            before = self.controller.last_result
            self.controller.tick(now)
            after = self.controller.last_result
        # Lease ownership is safety evidence, not merely an event log.  Keep
        # the transient-local snapshot fresh enough for the executor's
        # independent 0.5 s receive-age gate without flooding the graph with
        # the retained telemetry payload.
        if after != before or now - self._last_status_publish_monotonic >= 0.2:
            self._publish_status(after)

    def _publish_status(self, state: str, outgoing: str = "") -> None:
        self._last_status_publish_monotonic = time.monotonic()
        with self._controller_lock:
            feedback = self.controller.feedback
            payload = self.controller.status_snapshot()
            payload["state"] = state
            payload["status_time_monotonic_s"] = time.monotonic()
            payload["status_time_ros_s"] = self.get_clock().now().nanoseconds / 1e9
            payload["feedback_age_s"] = (
                None if feedback is None else time.monotonic() - feedback.received_at
            )
            payload["voltage"] = None if feedback is None else feedback.voltage
            payload["precontact_speed_lease"] = self._precontact_lease.snapshot()
            payload["dynamic_exact_pid_boost"] = {
                "enabled": bool(
                    self.get_parameter("dynamic_exact_pid_boost_enabled").value
                ),
                "active": self._dynamic_exact_pid_boost_active,
                "p": DYNAMIC_PRECONTACT_PID_P,
                "i": DYNAMIC_PRECONTACT_PID_I,
                "joint_ids": list(DYNAMIC_PRECONTACT_PID_JOINT_IDS),
            }
            if outgoing:
                payload["outgoing"] = outgoing
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_pub.publish(msg)

    def destroy_node(self):
        with self._controller_lock:
            self.controller.disarm("shutdown")
            if self._dynamic_exact_pid_boost_active:
                try:
                    self.transport.write(build_reset_joint_pid())
                    self._dynamic_exact_pid_boost_active = False
                except Exception as exc:
                    self.get_logger().error(
                        f"failed to reset Dynamic exact PID during shutdown: {exc}"
                    )
        self._stop.set()
        self.transport.close()
        if self._reader.is_alive():
            self._reader.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafeDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
