#!/usr/bin/env python3
"""Inert-by-default, at-most-three-command coarse T102 precontact runner.

The runtime never opens the serial port.  It uses the safe driver's arm,
lease and joint-target ROS interfaces, recalculates each remaining chord from
measured feedback, and reruns the existing conservative clearance checker
before every publication.
"""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from press.t102_outbound_route_builder import (
    build_t102_outbound_diagnostic_route,
    canonical_sha256,
)
import press.t102_prototype_continuous_clearance as clearance_checker
from press.t102_prototype_continuous_clearance import (
    _all_fk,
    _parse_urdf_joints,
    build_clearance_result,
    load_profile,
)

# The conservative mesh model is immutable for one action.  Building and
# hashing it repeatedly on AGX dominates runtime, so cache the verified model
# and parsed joints within this one process; every segment still runs the full
# adaptive interval clearance proof against its exact actual->candidate chord.
clearance_checker._build_model = lru_cache(maxsize=2)(
    clearance_checker._build_model
)
clearance_checker._parse_urdf_joints = lru_cache(maxsize=2)(
    clearance_checker._parse_urdf_joints
)

CONFIRMATION = (
    "确认立即执行一次Button1粗略precontact三大步T102单程运动；"
    "每步约3.5至7.5度并按实际位置修正剩余量；允许欠行程和约50mm粗定位误差；"
    "无运动、反向或异常立即停；不按压、不重试、不回程。"
)
COARSE_PURPOSE = "T102_OUTBOUND_COARSE_PRECONTACT_ONLY"
JOINT_NAMES = (
    "base_link_to_link1",
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_gripper_link",
)
CAPS_RAD = tuple(math.radians(value) for value in (5, 10, 10, 15, 30, 15))
MIN_USEFUL_RAD = 0.035
MAX_WRONG_RAD = 0.0087
ENCODER_STEP_RAD = 2.0 * math.pi / 4096.0
PASS = "PROVISIONAL_CONTINUOUS_CLEARANCE_PASS"


class CoarseBlocked(RuntimeError):
    pass


def _vec6(value: Sequence[Any], name: str) -> tuple[float, ...]:
    if len(value) != 6:
        raise CoarseBlocked(f"{name} must contain six joints")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise CoarseBlocked(f"{name} must be finite")
    return result


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def next_candidate(
    actual: Sequence[float], endpoint: Sequence[float], slots_left: int
) -> tuple[float, ...]:
    actual = _vec6(actual, "actual")
    endpoint = _vec6(endpoint, "endpoint")
    if slots_left not in (1, 2, 3):
        raise CoarseBlocked("slots_left must be 1, 2, or 3")
    candidate = tuple(
        current + (goal - current) / slots_left
        for current, goal in zip(actual, endpoint)
    )
    deltas = tuple(abs(after - before) for before, after in zip(actual, candidate))
    if any(delta > cap + 1e-12 for delta, cap in zip(deltas, CAPS_RAD)):
        raise CoarseBlocked("equal-share candidate exceeds per-command cap")
    residual = tuple(abs(goal - point) for goal, point in zip(endpoint, candidate))
    if any(value > cap * (slots_left - 1) + 1e-12 for value, cap in zip(residual, CAPS_RAD)):
        raise CoarseBlocked("endpoint cannot be reached inside remaining slots and caps")
    if max(deltas) <= ENCODER_STEP_RAD:
        raise CoarseBlocked("candidate is not materially different from actual")
    return candidate


def classify_progress(
    before: Sequence[float],
    target: Sequence[float],
    after: Sequence[float],
    endpoint: Sequence[float],
) -> dict[str, Any]:
    before = _vec6(before, "before")
    target = _vec6(target, "target")
    after = _vec6(after, "after")
    endpoint = _vec6(endpoint, "endpoint")
    signed = []
    for q0, q1, measured in zip(before, target, after):
        delta = q1 - q0
        signed.append(0.0 if abs(delta) <= ENCODER_STEP_RAD else (measured - q0) * (1 if delta > 0 else -1))
    max_useful = max(signed)
    max_wrong = max(-value for value in signed)
    initial_distance = distance(before, endpoint)
    final_distance = distance(after, endpoint)
    result = {
        "signed_progress_rad": signed,
        "max_useful_motion_rad": max_useful,
        "max_wrong_direction_rad": max_wrong,
        "initial_endpoint_distance_rad": initial_distance,
        "final_endpoint_distance_rad": final_distance,
        "endpoint_distance_reduction_rad": initial_distance - final_distance,
        "accepted": (
            max_useful >= MIN_USEFUL_RAD - 1e-12
            and max_wrong <= MAX_WRONG_RAD + 1e-12
            and final_distance < initial_distance - 1e-12
        ),
    }
    return result


def make_route(
    start: Sequence[float],
    endpoint: Sequence[float],
    frozen_bindings: dict[str, str],
    driver_profile: Path,
) -> dict[str, Any]:
    return build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=start,
        verified_static_ik_candidate_rad=endpoint,
        frozen_bindings=frozen_bindings,
        profile_path=driver_profile,
        max_profile_delta_fraction=1.0,
    )


def static_route(template: dict[str, Any], q: Sequence[float]) -> dict[str, Any]:
    route = copy.deepcopy(template)
    point = list(_vec6(q, "static q"))
    route["start_positions_rad"] = point
    route["static_ik_candidate_positions_rad"] = point
    route["setpoints_rad"] = [point]
    route.pop("route_sha256", None)
    route["route_sha256"] = canonical_sha256(route)
    return route


def clearance(
    profile: dict[str, Any], route: dict[str, Any], panel: dict[str, Any]
) -> dict[str, Any]:
    result = build_clearance_result(profile, route, panel)
    if result.get("result") != PASS:
        raise CoarseBlocked(f"continuous clearance is not PASS: {result.get('result')}")
    return result


def fingertip_base(profile: dict[str, Any], q: Sequence[float]) -> tuple[float, ...]:
    joints = _parse_urdf_joints(profile["robot_model"]["urdf_or_xacro_path"])
    fk = _all_fk(joints, q)["hand_tcp"]
    endpoint = profile["unified_tool_envelope"]["endpoint_b_hand_tcp_m"]
    point = fk[:3, 3] + fk[:3, :3] @ endpoint
    return tuple(float(value) for value in point)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RosRuntime:
    def __init__(self) -> None:
        import rclpy
        from elevator_interfaces.srv import PrecontactSpeedLease
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from std_srvs.srv import SetBool

        self.rclpy = rclpy
        self.JointState = JointState
        self.PrecontactSpeedLease = PrecontactSpeedLease
        self.SetBool = SetBool
        rclpy.init()
        self.node = rclpy.create_node("t102_three_large_step_coarse")
        self.joint_samples: list[tuple[float, tuple[float, ...]]] = []
        self.feedback: dict[str, Any] | None = None
        self.status: dict[str, Any] | None = None
        self.feedback_seen = 0.0
        self.status_seen = 0.0
        self.node.create_subscription(JointState, "/joint_states", self._joint, 20)
        self.node.create_subscription(String, "/roarm_safe/feedback_status", self._feedback, 20)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(String, "/roarm_safe/status", self._status, qos)
        self.lease = self.node.create_client(PrecontactSpeedLease, "/roarm_safe/precontact_speed_lease")
        self.arm = self.node.create_client(SetBool, "/roarm_safe/set_armed")
        self.publisher = None

    def _joint(self, message) -> None:
        try:
            by_name = dict(zip(message.name, message.position))
            q = tuple(float(by_name[name]) for name in JOINT_NAMES)
        except (KeyError, TypeError, ValueError):
            return
        now = time.monotonic()
        self.joint_samples.append((now, q))
        self.joint_samples = self.joint_samples[-256:]

    def _feedback(self, message) -> None:
        try:
            value = json.loads(message.data)
        except Exception:
            return
        if isinstance(value, dict):
            self.feedback, self.feedback_seen = value, time.monotonic()

    def _status(self, message) -> None:
        try:
            value = json.loads(message.data)
        except Exception:
            return
        if isinstance(value, dict):
            self.status, self.status_seen = value, time.monotonic()

    def spin(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=min(0.05, deadline - time.monotonic()))

    def stable_q(self, timeout_s: float = 5.0) -> tuple[float, ...]:
        deadline = time.monotonic() + timeout_s
        last_spans: list[float] | None = None
        while time.monotonic() < deadline:
            self.spin(0.1)
            if not self.joint_samples:
                continue
            end = self.joint_samples[-1][0]
            window = [item for stamp, item in self.joint_samples if stamp >= end - 1.0]
            stamps = [stamp for stamp, _ in self.joint_samples if stamp >= end - 1.0]
            if len(window) < 5 or stamps[-1] - stamps[0] < 0.95:
                continue
            spans = [max(q[i] for q in window) - min(q[i] for q in window) for i in range(6)]
            if max(spans) > 2.0 * ENCODER_STEP_RAD + 1e-12:
                last_spans = spans
                continue
            return window[-1]
        if last_spans is not None:
            raise CoarseBlocked(f"joint feedback is not stable: spans={last_spans}")
        raise CoarseBlocked("no fresh one-second stable joint window")

    def live_gate(self, *, expect_lease: bool, expected_publishers: int) -> dict[str, Any]:
        self.spin(0.3)
        now = time.monotonic()
        # /roarm_safe/status is TRANSIENT_LOCAL and changes only when the
        # controller state changes.  A settled robot can therefore retain a
        # valid snapshot for longer than one second; live freshness comes from
        # the continuously sampled feedback_status below.
        if self.status is None:
            raise CoarseBlocked("safe-driver status is missing")
        if self.feedback is None or now - self.feedback_seen > 1.0:
            raise CoarseBlocked("safe feedback is missing or stale")
        if self.status.get("dry_run") is not False or self.status.get("armed") is not False:
            raise CoarseBlocked("driver must be real and disarmed")
        if self.status.get("fault") not in ("", None):
            raise CoarseBlocked(f"safe-driver fault: {self.status.get('fault')}")
        if self.feedback.get("torque_enabled") != [True] * 6:
            raise CoarseBlocked("torque is not enabled on all six joints")
        lease = self.status.get("precontact_speed_lease") or {}
        if bool(lease.get("active")) != expect_lease:
            raise CoarseBlocked("lease state differs from expected state")
        count = self.node.count_publishers("/roarm_safe/joint_target")
        if count != expected_publishers:
            raise CoarseBlocked(f"joint-target publisher count is {count}, expected {expected_publishers}")
        return {"status": self.status, "feedback": self.feedback, "publisher_count": count}

    def call(self, client, request, timeout_s: float = 3.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise CoarseBlocked("required safe-driver service is unavailable")
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise CoarseBlocked("safe-driver service call timed out")
        return future.result()

    def enable_lease(self, lease_id: str) -> None:
        request = self.PrecontactSpeedLease.Request()
        request.enable = True
        request.purpose = COARSE_PURPOSE
        request.lease_id = lease_id
        response = self.call(self.lease, request)
        if not response.success or not response.active:
            raise CoarseBlocked(f"coarse lease rejected: {response.message}")

    def release_lease(self, lease_id: str) -> None:
        request = self.PrecontactSpeedLease.Request()
        request.enable = False
        request.purpose = COARSE_PURPOSE
        request.lease_id = lease_id
        response = self.call(self.lease, request)
        if not response.success or response.active:
            raise CoarseBlocked(f"coarse lease release failed: {response.message}")

    def publish_once(self, target: Sequence[float]) -> tuple[str, dict[str, Any]]:
        request = self.SetBool.Request()
        request.data = True
        armed = self.call(self.arm, request)
        if not armed.success:
            raise CoarseBlocked(f"arm rejected: {armed.message}")
        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(target)
        emitted_at = time.monotonic()
        self.publisher.publish(message)
        deadline = emitted_at + 20.0
        terminal = None
        while time.monotonic() < deadline:
            self.spin(0.1)
            if self.status_seen <= emitted_at or self.status is None:
                continue
            state = str(self.status.get("state", ""))
            if state in ("COARSE_PROGRESS_DISARMED", "CONVERGED_DISARMED"):
                terminal = state
                break
            if state.startswith("FAULT") or state.startswith("DISARMED:"):
                raise CoarseBlocked(f"safe-driver stopped command: {state}")
        if terminal is None:
            raise CoarseBlocked("command did not reach a safe terminal state")
        return terminal, copy.deepcopy(self.status)

    def close(self) -> None:
        if self.publisher is not None:
            self.node.destroy_publisher(self.publisher)
            self.publisher = None
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def run(args) -> dict[str, Any]:
    if args.confirmation_text != CONFIRMATION:
        raise CoarseBlocked("exact fresh confirmation text does not match")
    endpoint_route = json.loads(args.endpoint_route.read_text(encoding="utf-8"))
    endpoint = _vec6(endpoint_route["static_ik_candidate_positions_rad"], "endpoint")
    bindings = dict(endpoint_route["frozen_bindings"])
    profile, profile_file_hash = load_profile(args.profile)
    old_clearance = json.loads(args.clearance_template.read_text(encoding="utf-8"))
    panel = dict(old_clearance["panel"])
    runtime = RosRuntime()
    lease_id = f"t102-coarse-{secrets.token_hex(16)}"
    evidence: dict[str, Any] = {
        "started_at_unix_s": time.time(),
        "confirmation_text": args.confirmation_text,
        "endpoint_rad": list(endpoint),
        "commands": [],
        "inputs": {
            "endpoint_route_sha256": _sha(args.endpoint_route),
            "profile_sha256": profile_file_hash,
            "clearance_template_sha256": _sha(args.clearance_template),
        },
    }
    lease_active = False
    try:
        q_start = runtime.stable_q()
        evidence["pre_gate"] = runtime.live_gate(expect_lease=False, expected_publishers=0)
        evidence["q_start_rad"] = list(q_start)
        planned = make_route(q_start, endpoint, bindings, args.driver_profile)
        if len(planned["setpoints_rad"]) != 3:
            raise CoarseBlocked(f"planned route has {len(planned['setpoints_rad'])} segments, expected 3")
        planned_clearance = clearance(profile, planned, panel)
        _write_json(args.evidence_dir / "planned_three_step_route.json", planned)
        _write_json(args.evidence_dir / "planned_three_step_clearance.json", planned_clearance)
        runtime.enable_lease(lease_id)
        lease_active = True
        from sensor_msgs.msg import JointState
        runtime.publisher = runtime.node.create_publisher(JointState, "/roarm_safe/joint_target", 10)
        runtime.live_gate(expect_lease=True, expected_publishers=1)
        actual = q_start
        for index in range(3):
            slots_left = 3 - index
            candidate = next_candidate(actual, endpoint, slots_left)
            segment_route = make_route(actual, candidate, bindings, args.driver_profile)
            if len(segment_route["setpoints_rad"]) != 1:
                raise CoarseBlocked("exact next segment unexpectedly requires subdivision")
            segment_clearance = clearance(profile, segment_route, panel)
            _write_json(args.evidence_dir / f"segment_{index + 1}_route.json", segment_route)
            _write_json(args.evidence_dir / f"segment_{index + 1}_clearance.json", segment_clearance)
            runtime.live_gate(expect_lease=True, expected_publishers=1)
            terminal, terminal_status = runtime.publish_once(candidate)
            measured = runtime.stable_q()
            progress = classify_progress(actual, candidate, measured, endpoint)
            telemetry = (terminal_status.get("telemetry") or {})
            if int(telemetry.get("t102_reissue_count", 0) or 0) != 0:
                raise CoarseBlocked("safe driver reported a forbidden T102 reissue")
            if terminal != "CONVERGED_DISARMED" and not progress["accepted"]:
                raise CoarseBlocked(f"command {index + 1} did not make acceptable coarse progress")
            evidence["commands"].append(
                {
                    "index": index + 1,
                    "slots_left": slots_left,
                    "before_rad": list(actual),
                    "candidate_rad": list(candidate),
                    "measured_after_rad": list(measured),
                    "commanded_delta_deg": [math.degrees(b - a) for a, b in zip(actual, candidate)],
                    "terminal": terminal,
                    "progress": progress,
                    "outgoing": telemetry.get("outgoing"),
                    "safe_driver_telemetry": telemetry,
                    "clearance_sha256": segment_clearance["canonical_result_sha256"],
                }
            )
            actual = measured
        final_route = static_route(planned, actual)
        final_clearance = clearance(profile, final_route, panel)
        fingertip = fingertip_base(profile, actual)
        pre = _vec6((*panel["P_pre_base_m"], 0, 0, 0), "panel pre")[:3]
        a = panel["A_base_m"]
        n = panel["N_out_base"]
        fk_error = math.sqrt(sum((x - y) ** 2 for x, y in zip(fingertip, pre)))
        panel_sep = sum((x - y) * normal for x, y, normal in zip(fingertip, a, n))
        fixed_margin = float(final_clearance["interval_results"]["route_minimum_bracket_lower_bound_m"])
        reached = fk_error <= 0.050 and panel_sep >= 0.070 and fixed_margin >= 0.020
        evidence["final"] = {
            "q_rad": list(actual),
            "fingertip_base_m": list(fingertip),
            "fk_translation_error_m": fk_error,
            "fingertip_panel_separation_m": panel_sep,
            "fixed_obstacle_lower_bound_m": fixed_margin,
            "clearance_result": final_clearance["result"],
            "classification": (
                "COARSE_PRECONTACT_REACHED"
                if reached
                else "THREE_LARGE_STEPS_COMPLETE_COARSE_TARGET_NOT_REACHED"
            ),
        }
        _write_json(args.evidence_dir / "final_static_clearance.json", final_clearance)
        return evidence
    finally:
        if runtime.publisher is not None:
            runtime.node.destroy_publisher(runtime.publisher)
            runtime.publisher = None
            runtime.spin(0.2)
        if lease_active:
            try:
                runtime.release_lease(lease_id)
                evidence["lease_released"] = True
            except Exception as exc:
                evidence["lease_release_error"] = str(exc)
        try:
            evidence["post_gate"] = runtime.live_gate(expect_lease=False, expected_publishers=0)
        except Exception as exc:
            evidence["post_gate_error"] = str(exc)
        runtime.close()
        evidence["ended_at_unix_s"] = time.time()
        _write_json(args.output, evidence)


def plan_only(args) -> dict[str, Any]:
    endpoint_route = json.loads(args.endpoint_route.read_text(encoding="utf-8"))
    endpoint = _vec6(endpoint_route["static_ik_candidate_positions_rad"], "endpoint")
    start = _vec6(args.plan_start_rad, "plan start")
    profile, _ = load_profile(args.profile)
    panel = json.loads(args.clearance_template.read_text(encoding="utf-8"))["panel"]
    route = make_route(start, endpoint, endpoint_route["frozen_bindings"], args.driver_profile)
    if len(route["setpoints_rad"]) != 3:
        raise CoarseBlocked(f"planned route has {len(route['setpoints_rad'])} segments, expected 3")
    result = clearance(profile, route, panel)
    route_path = args.evidence_dir / "planned_three_step_route.json"
    result_path = args.evidence_dir / "planned_three_step_clearance.json"
    _write_json(route_path, route)
    _write_json(result_path, result)
    return {
        "state": "PLANNED_INERT_NO_ROS_RESOURCES",
        "start_rad": list(start),
        "endpoint_rad": list(endpoint),
        "setpoints_rad": route["setpoints_rad"],
        "route_sha256": route["route_sha256"],
        "clearance_result": result["result"],
        "clearance_sha256": result["canonical_result_sha256"],
        "route_path": str(route_path),
        "clearance_path": str(result_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-start-rad", nargs=6, type=float)
    parser.add_argument("--confirmation-text", default="")
    parser.add_argument("--endpoint-route", type=Path, default=REPO_ROOT / "config/press/prototype/t102_supervised_executable_route_20260815_v5.json")
    parser.add_argument("--profile", type=Path, default=REPO_ROOT / "config/press/prototype/t102_prototype_geometry_profile_20260815_v5.json")
    parser.add_argument("--clearance-template", type=Path, default=REPO_ROOT / "config/press/prototype/t102_prototype_clearance_result_20260815_v5.json")
    parser.add_argument("--driver-profile", type=Path, default=REPO_ROOT / "ros2_arm/roarm_safe_driver/config/agx_t102_coarse_precontact.yaml")
    parser.add_argument("--evidence-dir", type=Path, default=REPO_ROOT / "temp/t102_three_large_step_coarse_precontact_20260815")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "temp/t102_three_large_step_coarse_precontact_20260815/physical_result.json")
    args = parser.parse_args(argv)
    if args.plan_start_rad is not None:
        try:
            print(json.dumps(plan_only(args), sort_keys=True))
        except Exception as exc:
            print(json.dumps({"state": "PLAN_BLOCKED", "reason": str(exc)}, sort_keys=True))
            return 2
        return 0
    if not args.execute:
        print(json.dumps({"state": "INERT", "execution_enabled": False, "publisher_created": False, "lease_requested": False}, sort_keys=True))
        return 0
    try:
        result = run(args)
    except Exception as exc:
        print(json.dumps({"state": "ABORTED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"state": result["final"]["classification"], "command_count": len(result["commands"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
