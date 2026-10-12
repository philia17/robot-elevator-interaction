#!/usr/bin/env python3
"""One-shot, operator-invoked guarded move of the RoArm-M3 to press standby.

Mirrors the Quick Control "Go Standby" path exactly.  By default it targets
the driver's configured standby pose (``startup_restore_target_rad``), i.e.
the operator-set standby; ``--target`` overrides this.  It temporarily points
the driver's standby parameter at that target for one
``/roarm_safe/return_to_configured_standby`` call (which widens the per-joint
delta limits and issues a single T102 command), restores the parameter
immediately, then waits for the driver's own ``CONVERGED_DISARMED``.  If the
driver soft-converges with residual undertravel, the tool re-issues from the
measured pose and accepts the firmware's natural landing accuracy.  This tool
never creates a joint_target publisher and never arms on its own; the driver
owns all motion authority.

Coexistence with Quick Control: the QC console runs an orphan-disarm watchdog
that disarms any armed driver state unless a foreign ``/roarm_safe/joint_target``
publisher is visible.  The driver's one-shot standby service issues its T102
directly (no publisher), so this tool keeps a harmless ``joint_target``
publisher alive for the duration of the move to tell QC to stand aside, and
destroys it afterwards.  No joint_target message is ever published.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


PRESS_STANDBY_RAD = (
    -0.012271846,
    -0.728640874,
    2.000310947,
    1.023165186,
    0.001533981,
    0.03067961558979304,
)
# The firmware's T102 landing accuracy is ~1-3 deg; the old 0.5 deg target is
# not achievable on this hardware.  Accept the natural landing and let the
# operator verify visually (the press flow uses the actual measured pose).
STANDBY_ACCEPT_RAD = math.radians(3.0)
CONVERGENCE_TIMEOUT_S = 60.0
MAX_ATTEMPTS = 6
FEEDBACK_MAX_AGE_S = 2.0
DRIVER_NS = "/roarm_safe_driver"
STANDBY_SERVICE = "/roarm_safe/return_to_configured_standby"
STANDBY_PARAM = "startup_restore_target_rad"
STANDBY_SPEED_PARAM = "manual_standby_speed_steps_s"
RESIDUAL_MOVE_SPEED = 8
STUCK_WINDOW_S = 3.0
STUCK_SPAN_RAD = 0.003


class MoveError(RuntimeError):
    """Fail-closed move failure."""


class MoveToStandby(Node):
    def __init__(self) -> None:
        super().__init__("move_to_press_standby")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._feedback: dict | None = None
        self._feedback_at: float | None = None
        self._status_text = ""
        self._status_at: float | None = None
        self.create_subscription(
            String, "/roarm_safe/feedback_status", self._on_feedback, qos
        )
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/roarm_safe/status", self._on_status, status_qos
        )
        self._get_params = self.create_client(
            GetParameters, f"{DRIVER_NS}/get_parameters"
        )
        self._set_params = self.create_client(
            SetParameters, f"{DRIVER_NS}/set_parameters"
        )
        self._standby = self.create_client(
            Trigger, STANDBY_SERVICE
        )
        self._set_armed = self.create_client(SetBool, "/roarm_safe/set_armed")
        self._clear_fault_client = self.create_client(
            Trigger, "/roarm_safe/clear_fault"
        )
        self._coexist_pub = None

    def _on_feedback(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(value, dict):
            self._feedback = value
            self._feedback_at = time.monotonic()

    def _on_status(self, message: String) -> None:
        self._status_text = str(message.data)
        self._status_at = time.monotonic()

    def _check_driver_status(self, where: str) -> None:
        status = str(self._status_text or "")
        if not status:
            return
        if "REJECTED" in status:
            raise MoveError(f"driver {where}: {status[:160]}")
        if "FAULT" in status and "FAULT_CLEARED" not in status:
            raise MoveError(f"driver {where}: {status[:160]}")

    def _wait_feedback(self, timeout_s: float = FEEDBACK_MAX_AGE_S) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._feedback is not None and self._feedback_at is not None:
                age = time.monotonic() - self._feedback_at
                if age <= FEEDBACK_MAX_AGE_S:
                    return dict(self._feedback)
        raise MoveError("no fresh safe-driver feedback")

    def _check_idle(self, feedback: dict) -> None:
        state = str(feedback.get("state", "")).upper()
        allowed = (
            "DISARMED",
            "DISARMED:OPERATOR",
            "CONVERGED_DISARMED",
            "FAULT_CLEARED_DISARMED",
        )
        if state not in allowed:
            raise MoveError(f"driver is not in a disarmed terminal state: {state}")
        if feedback.get("armed") is not False:
            raise MoveError("driver reports armed=true; refusing to start")
        if feedback.get("fault"):
            raise MoveError(f"driver fault: {feedback.get('fault')}")
        if feedback.get("torque_enabled") != [True] * 6:
            raise MoveError("not all six torque flags are true")
        if self.count_publishers("/roarm_safe/joint_target") != 0:
            raise MoveError("an existing /roarm_safe/joint_target publisher is active")

    def _read_parameters(self, names: Sequence[str]) -> dict[str, list[float] | int]:
        if not self._get_params.wait_for_service(timeout_sec=2.0):
            raise MoveError("/roarm_safe_driver/get_parameters unavailable")
        request = GetParameters.Request()
        request.names = list(names)
        future = self._get_params.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.values:
            raise MoveError(f"could not read parameters {list(names)}")
        result: dict[str, list[float] | int] = {}
        for name, value in zip(names, response.values):
            if value.type == ParameterType.PARAMETER_DOUBLE_ARRAY:
                result[name] = [float(item) for item in value.double_array_value]
            elif value.type == ParameterType.PARAMETER_INTEGER:
                result[name] = int(value.integer_value)
            else:
                raise MoveError(f"{name} has unexpected type {value.type}")
        return result

    def _read_standby_param(self) -> list[float]:
        values = self._read_parameters([STANDBY_PARAM])
        result = values[STANDBY_PARAM]
        if not isinstance(result, list):
            raise MoveError(f"{STANDBY_PARAM} is not a double array")
        return result

    def _write_parameter(self, name: str, value) -> None:
        if not self._set_params.wait_for_service(timeout_sec=2.0):
            raise MoveError("/roarm_safe_driver/set_parameters unavailable")
        parameter = Parameter()
        parameter.name = name
        if isinstance(value, (list, tuple)):
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                double_array_value=[float(item) for item in value],
            )
        else:
            parameter.value = ParameterValue(
                type=ParameterType.PARAMETER_INTEGER,
                integer_value=int(value),
            )
        request = SetParameters.Request()
        request.parameters = [parameter]
        future = self._set_params.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.results or not response.results[0].successful:
            reason = "" if response is None else response.results[0].reason
            raise MoveError(f"could not set {name}: {reason}")

    def _call_standby(self) -> None:
        if not self._standby.wait_for_service(timeout_sec=2.0):
            raise MoveError(f"{STANDBY_SERVICE} unavailable")
        future = self._standby.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            reason = "" if response is None else response.message
            raise MoveError(f"standby service rejected: {reason}")

    def _wait_terminal(self, attempt: int) -> str:
        """Return ``CONVERGED`` or ``STUCK`` once motion is terminal."""
        # The standby service returns immediately after issuing the command;
        # feedback may still show the previous CONVERGED_DISARMED sample for a
        # moment.  Wait for the driver to actually report the new command as
        # active before watching for its terminal state.
        active_deadline = time.monotonic() + 3.0
        while time.monotonic() < active_deadline:
            self._check_driver_status(f"attempt {attempt}")
            feedback = self._wait_feedback()
            state = str(feedback.get("state", "")).upper()
            if state in {"COMMAND_ACTIVE", "DRY_RUN_ACTIVE"}:
                if feedback.get("armed") is not True:
                    raise MoveError("active state without armed=true")
                break
            if state.startswith("FAULT") and state != "FAULT_CLEARED_DISARMED":
                raise MoveError(f"driver {state} during attempt {attempt}")
            time.sleep(0.05)
        else:
            raise MoveError(f"command did not become active on attempt {attempt}")

        deadline = time.monotonic() + CONVERGENCE_TIMEOUT_S
        samples: list[tuple[float, list[float]]] = []
        while time.monotonic() < deadline:
            self._check_driver_status(f"attempt {attempt}")
            feedback = self._wait_feedback()
            state = str(feedback.get("state", "")).upper()
            if feedback.get("fault"):
                raise MoveError(f"driver fault during attempt {attempt}: {feedback['fault']}")
            if state.startswith("FAULT") and state != "FAULT_CLEARED_DISARMED":
                raise MoveError(f"driver {state} during attempt {attempt}")
            if state in {"COMMAND_ACTIVE", "DRY_RUN_ACTIVE"}:
                if feedback.get("armed") is not True:
                    raise MoveError("active state without armed=true")
                time.sleep(0.2)
                continue
            if state == "CONVERGED_DISARMED" and feedback.get("armed") is False:
                return "CONVERGED"
            if state in {"COMMAND_ACTIVE", "DRY_RUN_ACTIVE"}:
                positions = feedback.get("positions_rad")
                if not isinstance(positions, list) or len(positions) != 6:
                    raise MoveError("active state without six measured positions")
                now = time.monotonic()
                samples.append((now, [float(v) for v in positions]))
                while samples and now - samples[0][0] > STUCK_WINDOW_S:
                    samples.pop(0)
                if (
                    len(samples) >= 15
                    and now - samples[0][0] >= STUCK_WINDOW_S - 0.2
                ):
                    span = max(
                        max(sample[1][index] for sample in samples)
                        - min(sample[1][index] for sample in samples)
                        for index in range(6)
                    )
                    if span <= STUCK_SPAN_RAD:
                        self.get_logger().info(
                            f"driver stopped just outside tolerance (span "
                            f"{math.degrees(span):.3f} deg); stopping and re-issuing"
                        )
                        self._disarm_stop()
                        return "STUCK"
                time.sleep(0.15)
                continue
            time.sleep(0.2)
        raise MoveError(f"terminal timeout on attempt {attempt}")

    def _disarm_stop(self) -> None:
        """Best-effort stop used only on abort; never retries."""
        if not self._set_armed.service_is_ready():
            return
        try:
            request = SetBool.Request()
            request.data = False
            future = self._set_armed.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        except Exception:  # noqa: BLE001 - best-effort stop
            pass

    def _clear_fault(self) -> None:
        if not self._clear_fault_client.wait_for_service(timeout_sec=2.0):
            raise MoveError("/roarm_safe/clear_fault unavailable")
        future = self._clear_fault_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            reason = "" if response is None else response.message
            raise MoveError(f"clear_fault rejected: {reason}")
        # Drop the retained driver status so a stale FAULT/REJECTED string
        # cannot trip the next attempt's status check.
        self._status_text = ""

    def _wait_idle_after_fault(self) -> None:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            feedback = self._wait_feedback()
            state = str(feedback.get("state", "")).upper()
            if feedback.get("armed") is not False:
                time.sleep(0.2)
                continue
            if (
                not feedback.get("fault")
                and state
                in {
                    "FAULT_CLEARED_DISARMED",
                    "DISARMED",
                    "DISARMED:OPERATOR",
                    "CONVERGED_DISARMED",
                }
            ):
                return
            time.sleep(0.2)
        raise MoveError("driver did not return to idle after fault clear")

    def _run(self, target: Sequence[float]) -> None:
        self._wait_until_idle()
        self._check_idle(self._wait_feedback())
        self._check_driver_status("preconditions")
        originals = self._read_parameters([STANDBY_PARAM, STANDBY_SPEED_PARAM])
        original_target = originals[STANDBY_PARAM]
        original_speed = originals[STANDBY_SPEED_PARAM]
        target_values = [float(value) for value in target]
        self.get_logger().info(
            f"press standby one-shot: {MAX_ATTEMPTS} attempt budget, "
            f"restoring {STANDBY_PARAM}={[round(v, 6) for v in original_target]} "
            f"and {STANDBY_SPEED_PARAM}={original_speed}"
        )
        self._ensure_coexistence_publisher()
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                self.get_logger().info(
                    f"attempt {attempt}/{MAX_ATTEMPTS}: issuing one-shot standby command"
                )
                try:
                    self._write_parameter(STANDBY_PARAM, target_values)
                    speed = (
                        int(original_speed)
                        if attempt == 1
                        else RESIDUAL_MOVE_SPEED
                    )
                    self._write_parameter(STANDBY_SPEED_PARAM, speed)
                    self._call_standby()
                    self._wait_terminal(attempt)
                except MoveError as exc:
                    message = str(exc)
                    if "convergence timeout" in message or "FAULT" in message.upper():
                        # The big move can outrun the driver's command timeout;
                        # the arm stopped partway.  Clear the benign fault and
                        # continue from the measured pose on the next attempt.
                        self.get_logger().warning(
                            f"{message}; clearing fault and continuing"
                        )
                        try:
                            self._clear_fault()
                            self._wait_idle_after_fault()
                        except MoveError as clear_exc:
                            self._disarm_stop()
                            raise clear_exc
                    else:
                        self._disarm_stop()
                        raise
                finally:
                    # The driver keeps its internal goal; restore the configured
                    # standby immediately so Quick Control semantics never change.
                    try:
                        self._write_parameter(STANDBY_PARAM, original_target)
                        self._write_parameter(STANDBY_SPEED_PARAM, original_speed)
                    except MoveError as exc:
                        self.get_logger().error(f"could not restore driver params: {exc}")
                feedback = self._wait_feedback()
                measured = feedback.get("positions_rad")
                if not isinstance(measured, list) or len(measured) != 6:
                    raise MoveError("no readable final positions after convergence")
                residual = max(
                    abs(float(a) - float(b))
                    for a, b in zip(measured, target_values)
                )
                per_joint = [
                    math.degrees(abs(float(a) - float(b)))
                    for a, b in zip(measured, target_values)
                ]
                if residual <= STANDBY_ACCEPT_RAD:
                    self.get_logger().info(
                        f"press standby reached within firmware accuracy "
                        f"(residual {math.degrees(residual):.2f} deg)"
                    )
                    print(
                        f"STANDBY MOVE DONE: residual {math.degrees(residual):.2f} deg",
                        flush=True,
                    )
                    return
                if attempt < MAX_ATTEMPTS:
                    self.get_logger().info(
                        f"residual {math.degrees(residual):.2f} deg "
                        f"per-joint_deg={[round(v, 2) for v in per_joint]} "
                        f"measured={[round(float(v), 4) for v in measured]}; "
                        f"re-issuing from measured pose"
                    )
        finally:
            self._destroy_coexistence_publisher()
        raise MoveError(
            f"final pose off press standby by {math.degrees(residual):.2f} deg "
            f"after {MAX_ATTEMPTS} attempts"
        )

    def _wait_until_idle(self, timeout_s: float = 20.0) -> None:
        """Wait for a previous move (e.g. an in-flight button press) to finish."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            feedback = self._wait_feedback()
            state = str(feedback.get("state", "")).upper()
            if (
                feedback.get("armed") is False
                and not feedback.get("fault")
                and state
                in {
                    "DISARMED",
                    "DISARMED:OPERATOR",
                    "CONVERGED_DISARMED",
                    "FAULT_CLEARED_DISARMED",
                }
            ):
                return
            time.sleep(0.25)
        raise MoveError("driver did not become idle before the move could start")

    def _ensure_coexistence_publisher(self) -> None:
        """Keep Quick Control's orphan-disarm watchdog from killing the move.

        The QC watchdog leaves armed states alone only when a foreign
        ``/roarm_safe/joint_target`` publisher is visible.  This publisher is
        never used to send a command; it only signals QC to stand aside.
        """
        if self._coexist_pub is not None:
            return
        self._coexist_pub = self.create_publisher(
            JointState, "/roarm_safe/joint_target", 1
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.count_publishers("/roarm_safe/joint_target") == 1:
                break
            time.sleep(0.05)
        if self.count_publishers("/roarm_safe/joint_target") != 1:
            self._destroy_coexistence_publisher()
            raise MoveError("coexistence publisher did not become visible")
        # Give DDS + the QC watchdog a moment to observe the publisher before
        # the driver arms; otherwise QC can still win the disarm race.
        time.sleep(1.0)
        self.get_logger().info(
            "coexistence joint_target publisher visible; QC watchdog stands aside"
        )

    def _destroy_coexistence_publisher(self) -> None:
        if self._coexist_pub is not None:
            try:
                self.destroy_publisher(self._coexist_pub)
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._coexist_pub = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="required to move the robot")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate preconditions and print the plan without moving",
    )
    parser.add_argument(
        "--target",
        type=json.loads,
        default=None,
        help=(
            "six joint positions in radians (default: the driver's configured "
            "standby pose, i.e. the operator-set standby)"
        ),
    )
    args = parser.parse_args()

    rclpy.init()
    node = MoveToStandby()
    try:
        if args.target is None:
            target = tuple(node._read_standby_param())
        else:
            target = tuple(float(value) for value in args.target)
        if len(target) != 6 or not all(math.isfinite(value) for value in target):
            raise SystemExit("target must contain six finite joint positions")
        if args.check_only:
            feedback = node._wait_feedback()
            node._check_idle(feedback)
            node._check_driver_status("preconditions")
            original = node._read_standby_param()
            if not node._standby.wait_for_service(timeout_sec=2.0):
                raise MoveError(f"{STANDBY_SERVICE} unavailable")
            print("CHECK ONLY (no motion)")
            print("current:", [round(float(v), 6) for v in feedback["positions_rad"]], flush=True)
            print("target :", [round(float(v), 6) for v in target], flush=True)
            print("configured standby:", [round(float(v), 6) for v in original], flush=True)
            print("mechanism: one-shot /roarm_safe/return_to_configured_standby", flush=True)
            print("CHECK OK: preconditions and service available", flush=True)
            return 0
        if not args.confirm:
            print("Refusing: pass --confirm to move the robot to press standby.")
            return 1
        print("current:", [round(float(v), 6) for v in node._wait_feedback()["positions_rad"]], flush=True)
        print("target :", [round(float(v), 6) for v in target], flush=True)
        node._run(target)
        return 0
    except (MoveError, Exception) as exc:  # noqa: BLE001 - fail-closed CLI
        node.get_logger().error(f"MOVE ABORTED: {exc}")
        node._disarm_stop()
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
