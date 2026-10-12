#!/usr/bin/env python3
"""One-shot, safe-driver-owned T101 actuation probe.

This utility is intentionally narrower than the T102 outbound executor.  It
does not open a serial device and it has no route/retry/return logic.  The
normal ``roarm_safe_driver`` owns serial output, feedback health, arm deadline,
fault handling, and T101/T102 dispatch.  The script only creates one temporary
ROS command owner after an explicit ``--confirm`` flag and measures the fresh
feedback response to exactly one target.

The pure helpers are importable without ROS so that the safety predicates can
be tested on the development host.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import time
from typing import Any, Iterable, Sequence


JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)
DEFAULT_JOINT_INDEX = 4
DEFAULT_DELTA_RAD = 0.040
MIN_OBSERVED_MOTION_RAD = 0.015
NO_MOTION_RAD = 0.005
MAX_NON_TARGET_DRIFT_RAD = 0.020
MAX_OBSERVATION_S = 5.0
MAX_FEEDBACK_AGE_S = 1.0


class ProbeError(ValueError):
    """A preflight, feedback, or probe-result predicate failed."""


@dataclass(frozen=True)
class FeedbackSnapshot:
    positions_rad: tuple[float, ...]
    state: str
    armed: bool
    dry_run: bool
    fault: str
    torque_enabled: tuple[bool, ...]
    voltage: float
    feedback_stamp_ns: int
    received_monotonic: float


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProbeError(f"{field} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise ProbeError(f"{field} must be finite")
    return result


def parse_feedback_payload(payload: str | bytes, *, received_monotonic: float | None = None) -> FeedbackSnapshot:
    """Parse one safe-driver feedback-status JSON payload."""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        raw = json.loads(payload)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("feedback payload is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ProbeError("feedback payload must be an object")
    positions = raw.get("positions_rad")
    if not isinstance(positions, list) or len(positions) != 6:
        raise ProbeError("feedback positions_rad must contain six values")
    torque = raw.get("torque_enabled")
    if not isinstance(torque, list) or len(torque) != 6 or any(not isinstance(v, bool) for v in torque):
        raise ProbeError("feedback torque_enabled must contain six booleans")
    return FeedbackSnapshot(
        positions_rad=tuple(_finite(v, f"positions_rad[{i}]") for i, v in enumerate(positions)),
        state=str(raw.get("state", "")),
        armed=raw.get("armed") is True,
        dry_run=raw.get("dry_run") is True,
        fault=str(raw.get("fault", "")),
        torque_enabled=tuple(torque),
        voltage=_finite(raw.get("voltage"), "voltage"),
        feedback_stamp_ns=int(raw.get("feedback_stamp_ns", 0)),
        received_monotonic=time.monotonic() if received_monotonic is None else received_monotonic,
    )


def validate_preflight(snapshot: FeedbackSnapshot, *, now: float | None = None) -> None:
    """Require a fresh, healthy, disarmed, torque-complete safe-driver state."""

    current = time.monotonic() if now is None else now
    if current - snapshot.received_monotonic > MAX_FEEDBACK_AGE_S:
        raise ProbeError("feedback is stale")
    if snapshot.armed or snapshot.dry_run:
        raise ProbeError("probe requires armed=false and dry_run=false")
    if snapshot.fault:
        raise ProbeError("safe-driver fault is present")
    if len(snapshot.torque_enabled) != 6 or not all(snapshot.torque_enabled):
        raise ProbeError("all six torque switches must be enabled")
    if snapshot.voltage <= 0.0:
        raise ProbeError("feedback voltage is not positive")
    if snapshot.state not in {"CONVERGED_DISARMED", "DISARMED:operator"}:
        raise ProbeError(f"probe requires a disarmed idle state, got {snapshot.state!r}")


def build_probe_target(
    measured_positions_rad: Sequence[float],
    *,
    joint_index: int = DEFAULT_JOINT_INDEX,
    delta_rad: float = DEFAULT_DELTA_RAD,
) -> tuple[float, ...]:
    """Build one full six-joint target with exactly one changed joint."""

    if len(measured_positions_rad) != 6:
        raise ProbeError("measured positions must contain six values")
    if not isinstance(joint_index, int) or not 0 <= joint_index < 5:
        raise ProbeError("probe joint must be one of the five non-gripper joints")
    delta = _finite(delta_rad, "delta_rad")
    if not 0.035 <= abs(delta) <= 0.050:
        raise ProbeError("delta_rad must be within 0.035..0.050 rad")
    target = [_finite(value, f"measured_positions_rad[{i}]") for i, value in enumerate(measured_positions_rad)]
    target[joint_index] += delta
    if not -math.pi <= target[0] <= math.pi:
        raise ProbeError("base probe target is outside its command limit")
    if not -math.pi / 2.0 <= target[1] <= math.pi / 2.0:
        raise ProbeError("shoulder probe target is outside its command limit")
    if not 0.0 <= target[2] <= 2.95:
        raise ProbeError("elbow probe target is outside its command limit")
    if not -math.pi / 2.0 <= target[3] <= math.pi / 2.0:
        raise ProbeError("wrist probe target is outside its command limit")
    if not -math.pi <= target[4] <= math.pi:
        raise ProbeError("roll probe target is outside its command limit")
    return tuple(target)


def evaluate_probe_samples(
    baseline_positions_rad: Sequence[float],
    samples: Iterable[Sequence[float]],
    *,
    joint_index: int = DEFAULT_JOINT_INDEX,
    commanded_delta_rad: float = DEFAULT_DELTA_RAD,
) -> dict[str, Any]:
    """Classify fresh samples without ever treating zero movement as success."""

    baseline = tuple(_finite(value, f"baseline[{i}]") for i, value in enumerate(baseline_positions_rad))
    if len(baseline) != 6:
        raise ProbeError("baseline must contain six values")
    sign = 1.0 if commanded_delta_rad > 0.0 else -1.0
    best_signed = 0.0
    max_non_target_drift = 0.0
    sample_count = 0
    for raw in samples:
        values = tuple(_finite(value, f"sample[{i}]") for i, value in enumerate(raw))
        if len(values) != 6:
            raise ProbeError("each sample must contain six values")
        sample_count += 1
        selected_delta = values[joint_index] - baseline[joint_index]
        best_signed = max(best_signed, sign * selected_delta)
        max_non_target_drift = max(
            max_non_target_drift,
            max(abs(values[i] - baseline[i]) for i in range(6) if i != joint_index),
        )
        if sign * selected_delta < -NO_MOTION_RAD:
            return {
                "success": False,
                "reason": "wrong_direction",
                "sample_count": sample_count,
                "best_signed_motion_rad": best_signed,
                "max_non_target_drift_rad": max_non_target_drift,
            }
        if max_non_target_drift > MAX_NON_TARGET_DRIFT_RAD:
            return {
                "success": False,
                "reason": "unexpected_non_target_drift",
                "sample_count": sample_count,
                "best_signed_motion_rad": best_signed,
                "max_non_target_drift_rad": max_non_target_drift,
            }
    if best_signed >= MIN_OBSERVED_MOTION_RAD:
        return {
            "success": True,
            "reason": "measured_correct_direction_motion",
            "sample_count": sample_count,
            "best_signed_motion_rad": best_signed,
            "max_non_target_drift_rad": max_non_target_drift,
        }
    return {
        "success": False,
        "reason": "no_measurable_motion",
        "sample_count": sample_count,
        "best_signed_motion_rad": best_signed,
        "max_non_target_drift_rad": max_non_target_drift,
    }


def _wait_for_future(future: Any, node: Any, *, timeout_s: float) -> Any:
    import rclpy

    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not future.done():
        raise ProbeError("ROS service call timed out")
    result = future.result()
    if result is None:
        raise ProbeError("ROS service call returned no result")
    return result


class _ProbeNode:
    def __init__(self, rclpy_module: Any, node_cls: Any, joint_index: int, delta_rad: float) -> None:
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from std_srvs.srv import SetBool

        self._rclpy = rclpy_module
        self.node = node_cls("t101_actuation_probe")
        self.joint_index = joint_index
        self.delta_rad = delta_rad
        self.latest: FeedbackSnapshot | None = None
        self._feedback_count = 0
        self._target_pub = None
        self._arm_client = self.node.create_client(SetBool, "/roarm_safe/set_armed")
        self._feedback_sub = self.node.create_subscription(
            String, "/roarm_safe/feedback_status", self._on_feedback, 10
        )
        self._JointState = JointState
        self.node.get_logger().info("T101 probe is staged; no target has been published")

    def _on_feedback(self, message: Any) -> None:
        try:
            self.latest = parse_feedback_payload(message.data)
            self._feedback_count += 1
        except (ProbeError, ValueError) as exc:
            self.node.get_logger().error(f"invalid feedback: {exc}")

    def spin_until_feedback(self, *, timeout_s: float) -> FeedbackSnapshot:
        deadline = time.monotonic() + timeout_s
        while self.latest is None and time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.latest is None:
            raise ProbeError("no safe-driver feedback received")
        return self.latest

    def call_arm(self, enabled: bool) -> None:
        from std_srvs.srv import SetBool

        request = SetBool.Request()
        request.data = enabled
        result = _wait_for_future(self._arm_client.call_async(request), self.node, timeout_s=2.0)
        if not result.success:
            raise ProbeError(f"set_armed({enabled}) rejected: {result.message}")

    def execute_once(self) -> dict[str, Any]:
        from rclpy.qos import QoSProfile

        baseline = self.spin_until_feedback(timeout_s=3.0)
        validate_preflight(baseline)
        if self.node.count_publishers("/roarm_safe/joint_target") != 0:
            raise ProbeError("foreign /roarm_safe/joint_target publisher present")
        if not self._arm_client.wait_for_service(timeout_sec=2.0):
            raise ProbeError("/roarm_safe/set_armed is unavailable")
        target = build_probe_target(
            baseline.positions_rad, joint_index=self.joint_index, delta_rad=self.delta_rad
        )
        self._target_pub = self.node.create_publisher(self._JointState, "/roarm_safe/joint_target", QoSProfile(depth=1))
        deadline = time.monotonic() + 2.0
        while self.node.count_publishers("/roarm_safe/joint_target") != 1 and time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
        if self.node.count_publishers("/roarm_safe/joint_target") != 1:
            raise ProbeError("could not establish exclusive target ownership")

        before_count = self._feedback_count
        self.call_arm(True)
        message = self._JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(target)
        self._target_pub.publish(message)

        samples: list[tuple[float, ...]] = []
        deadline = time.monotonic() + MAX_OBSERVATION_S
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            current = self.latest
            if current is None or self._feedback_count <= before_count:
                continue
            if current.fault:
                break
            samples.append(current.positions_rad)
            result = evaluate_probe_samples(
                baseline.positions_rad,
                samples,
                joint_index=self.joint_index,
                commanded_delta_rad=self.delta_rad,
            )
            if result["success"] or result["reason"] != "no_measurable_motion":
                break
        result = evaluate_probe_samples(
            baseline.positions_rad,
            samples,
            joint_index=self.joint_index,
            commanded_delta_rad=self.delta_rad,
        )
        self.call_arm(False)
        result.update(
            {
                "baseline_positions_rad": list(baseline.positions_rad),
                "target_positions_rad": list(target),
                "joint_index": self.joint_index,
                "joint_name": JOINT_NAMES[self.joint_index],
                "commanded_delta_rad": self.delta_rad,
                "feedback_samples": len(samples),
            }
        )
        return result

    def close(self) -> None:
        if self._target_pub is not None:
            self.node.destroy_publisher(self._target_pub)
            self._target_pub = None
        self.node.destroy_node()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="required fresh operator confirmation")
    parser.add_argument("--joint-index", type=int, default=DEFAULT_JOINT_INDEX)
    parser.add_argument("--delta-rad", type=float, default=DEFAULT_DELTA_RAD)
    args = parser.parse_args(argv)
    if not args.confirm:
        print("REFUSING: pass --confirm only after the exact onsite confirmation")
        return 2
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        print(json.dumps({"success": False, "reason": "ROS_IMPORT_UNAVAILABLE", "detail": str(exc)}))
        return 3

    rclpy.init()
    probe = _ProbeNode(rclpy, Node, args.joint_index, args.delta_rad)
    try:
        result = probe.execute_once()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0 if result.get("success") else 1
    except Exception as exc:  # noqa: BLE001 - terminal result is fail-closed
        print(json.dumps({"success": False, "reason": type(exc).__name__, "detail": str(exc)}, sort_keys=True))
        return 1
    finally:
        probe.close()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
