#!/usr/bin/env python3
"""Fast v2 four-target T102 coarse-precontact runner.

The default path loads one persistent route package and completes its cheap
standard-library preflight before importing ROS or constructing resources.
It never plans a route, loads robot geometry, launches a subprocess, checks
continuous clearance, presses, contacts, returns, retries, or clears faults.

The package and this script remain non-authoritative offline artifacts.  A
future physical invocation still requires a separately reviewed deployment,
fresh live safety gates, and explicit on-site operator confirmation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from press.t102_fast_runtime_envelope import (  # noqa: E402
    ENCODER_STEP_RAD,
    EXPECTED_PACKAGE_SHA256,
    JOINT_NAMES,
    STANDARD_STANDBY_RAD,
    EnvelopeBlocked,
    assess_measured_handoff,
    load_and_validate_package,
    vector,
)
from press.t102_v2_execution_core import (  # noqa: E402
    DEFAULT_V2_PACKAGE,
    V2ExecutionBlocked,
    V2ExecutionMachine,
    load_v2_package,
    validate_v2_package,
)


CONFIRMATION = (
    "确认仅在完成部署复核、全部现场安全门和当次现场确认后执行一次 Standard Standby -> "
    "Button 1 70 mm 无接触粗预定位；四个大目标、三次实测反馈衔接；异常立即停；"
    "不按压、不接触、不重试、不回程。"
)
COARSE_PURPOSE = "T102_OUTBOUND_COARSE_PRECONTACT_ONLY"


class SmoothBlocked(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def preflight_v2(package_path: str | Path = DEFAULT_V2_PACKAGE) -> dict[str, Any]:
    """Load and validate the complete package before any ROS import/resource."""

    try:
        return load_and_validate_package(
            package_path, expected_package_sha256=EXPECTED_PACKAGE_SHA256
        )
    except (EnvelopeBlocked, V2ExecutionBlocked) as exc:
        raise SmoothBlocked(str(exc)) from exc


def measured_handoff_match(
    package: Mapping[str, Any],
    transition_index: int,
    segment_start_rad: Sequence[float],
    actual_rad: Sequence[float],
    *,
    feedback_age_s: float,
) -> dict[str, Any] | None:
    """Return one admitted asynchronous large-step handoff, or ``None``.

    The current and next commands are derived from the sealed package by
    transition index.  ``segment_start_rad`` is the live start of the active
    command (the reviewed package start for transition 0, then the prior
    accepted measured handoff).  There is no projected-progress proof box.
    """

    if transition_index == 0:
        reviewed_start = vector(package["start"]["positions_rad"], "package start")
        supplied_start = vector(segment_start_rad, "segment start")
        if any(abs(a - b) > ENCODER_STEP_RAD + 1e-12 for a, b in zip(reviewed_start, supplied_start)):
            raise SmoothBlocked("first handoff segment start is not Standard Standby")
    # Compatibility seam retained for focused runner tests.  The production
    # episode below uses V2ExecutionMachine directly.
    try:
        assessment = assess_measured_handoff(
            package,
            transition_index,
            segment_start_rad,
            actual_rad,
            feedback_age_s=feedback_age_s,
        )
    except EnvelopeBlocked as exc:
        raise SmoothBlocked(str(exc)) from exc
    return assessment if assessment["accepted"] else None


class SmoothRosRuntime:
    """ROS adapter whose imports and resources are delayed until construction."""

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
        self.node = rclpy.create_node("t102_smooth_precontact_fast_v2")
        self.joint_samples: list[tuple[float, tuple[float, ...]]] = []
        self.feedback: dict[str, Any] | None = None
        self.status: dict[str, Any] | None = None
        self.feedback_seen = 0.0
        self.status_seen = 0.0
        self.node.create_subscription(JointState, "/joint_states", self._joint, 20)
        self.node.create_subscription(
            String, "/roarm_safe/feedback_status", self._feedback, 20
        )
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.node.create_subscription(
            String, "/roarm_safe/status", self._status, qos
        )
        self.lease = self.node.create_client(
            PrecontactSpeedLease, "/roarm_safe/precontact_speed_lease"
        )
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
            self.feedback = value
            self.feedback_seen = time.monotonic()

    def _status(self, message) -> None:
        try:
            value = json.loads(message.data)
        except Exception:
            return
        if isinstance(value, dict):
            self.status = value
            self.status_seen = time.monotonic()

    def spin(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.rclpy.spin_once(
                self.node, timeout_sec=min(0.05, deadline - time.monotonic())
            )

    def stable_q(self, timeout_s: float = 5.0) -> tuple[float, ...]:
        deadline = time.monotonic() + timeout_s
        last_spans: list[float] | None = None
        while time.monotonic() < deadline:
            self.spin(0.1)
            if not self.joint_samples:
                continue
            end = self.joint_samples[-1][0]
            samples = [
                (stamp, q)
                for stamp, q in self.joint_samples
                if stamp >= end - 1.0
            ]
            if len(samples) < 5 or samples[-1][0] - samples[0][0] < 0.95:
                continue
            spans = [
                max(q[index] for _stamp, q in samples)
                - min(q[index] for _stamp, q in samples)
                for index in range(6)
            ]
            if max(spans) > 2.0 * ENCODER_STEP_RAD + 1e-12:
                last_spans = spans
                continue
            return samples[-1][1]
        if last_spans is not None:
            raise SmoothBlocked(f"joint feedback is not stable: spans={last_spans}")
        raise SmoothBlocked("no fresh one-second stable joint window")

    def call(self, client, request, timeout_s: float = 3.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise SmoothBlocked("required safe-driver service is unavailable")
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=timeout_s
        )
        if not future.done() or future.result() is None:
            raise SmoothBlocked("safe-driver service call timed out")
        return future.result()

    def gate(
        self, *, expect_armed: bool, expect_lease: bool, expected_publishers: int
    ) -> dict[str, Any]:
        self.spin(0.15)
        now = time.monotonic()
        if self.status is None:
            raise SmoothBlocked("safe-driver status is missing")
        if self.feedback is None or now - self.feedback_seen > 1.0:
            raise SmoothBlocked("safe-driver feedback is missing or stale")
        if self.status.get("dry_run") is not False:
            raise SmoothBlocked("driver is not in real mode")
        if bool(self.status.get("armed")) != expect_armed:
            raise SmoothBlocked("armed state differs from expected state")
        if self.status.get("fault") not in ("", None):
            raise SmoothBlocked(f"safe-driver fault: {self.status.get('fault')}")
        if self.feedback.get("torque_enabled") != [True] * 6:
            raise SmoothBlocked("torque is not enabled on all six joints")
        lease = self.status.get("precontact_speed_lease") or {}
        if bool(lease.get("active")) != expect_lease:
            raise SmoothBlocked("lease state differs from expected state")
        count = self.node.count_publishers("/roarm_safe/joint_target")
        if count != expected_publishers:
            raise SmoothBlocked(
                f"joint-target publisher count is {count}, expected {expected_publishers}"
            )
        return {
            "status": copy.deepcopy(self.status),
            "feedback": copy.deepcopy(self.feedback),
            "publisher_count": count,
        }

    def enable_lease(self, lease_id: str) -> None:
        request = self.PrecontactSpeedLease.Request()
        request.enable = True
        request.purpose = COARSE_PURPOSE
        request.lease_id = lease_id
        response = self.call(self.lease, request)
        if not response.success or not response.active:
            raise SmoothBlocked(f"coarse lease rejected: {response.message}")

    def release_lease(self, lease_id: str) -> None:
        request = self.PrecontactSpeedLease.Request()
        request.enable = False
        request.purpose = COARSE_PURPOSE
        request.lease_id = lease_id
        response = self.call(self.lease, request)
        if not response.success or response.active:
            raise SmoothBlocked(f"coarse lease release failed: {response.message}")

    def arm_once(self) -> None:
        request = self.SetBool.Request()
        request.data = True
        response = self.call(self.arm, request)
        if not response.success:
            raise SmoothBlocked(f"arm rejected: {response.message}")

    def disarm_once(self) -> None:
        request = self.SetBool.Request()
        request.data = False
        response = self.call(self.arm, request)
        if not response.success:
            raise SmoothBlocked(f"disarm rejected: {response.message}")

    def publish_target(self, target: Sequence[float]) -> float:
        message = self.JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(target)
        emitted = time.monotonic()
        self.publisher.publish(message)
        self.spin(0.08)
        return emitted

    def close(self) -> None:
        if self.publisher is not None:
            self.node.destroy_publisher(self.publisher)
            self.publisher = None
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


RuntimeFactory = Callable[[], SmoothRosRuntime]


def execute(
    args,
    package: Mapping[str, Any] | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> dict[str, Any]:
    """Run only after v2 package preflight; never generate or finalize a route."""

    if args.confirmation_text != CONFIRMATION:
        raise SmoothBlocked("exact fresh confirmation text does not match")
    if package is None:
        package = preflight_v2(args.package)
    else:
        try:
            package = validate_v2_package(package)
        except V2ExecutionBlocked as exc:
            raise SmoothBlocked(str(exc)) from exc

    # This is the first point at which the default v2 path may construct the
    # adapter that imports ROS.  All package/envelope checks above have passed.
    runtime = (runtime_factory or SmoothRosRuntime)()
    targets = [tuple(item) for item in package["outbound_targets_rad"]]
    endpoint = tuple(package["endpoint"]["joint_positions_rad"])
    lease_id = f"t102-smooth-v2-{secrets.token_hex(16)}"
    evidence: dict[str, Any] = {
        "started_at_unix_s": time.time(),
        "scenario": package["scenario"],
        "package_sha256": package["package_sha256"],
        "publications": [],
        "replace_events": [],
        "arm_call_count": 0,
        "lease_enable_count": 0,
        "lease_release_count": 0,
        "command_contract": copy.deepcopy(package["command_contract"]),
    }
    lease_active = False
    armed = False
    try:
        q_start = runtime.stable_q()
        # The shared machine owns the exact one-step start and all handoffs.
        evidence["pre_gate"] = runtime.gate(
            expect_armed=False, expect_lease=False, expected_publishers=0
        )
        runtime.enable_lease(lease_id)
        lease_active = True
        evidence["lease_enable_count"] = 1
        runtime.publisher = runtime.node.create_publisher(
            runtime.JointState, "/roarm_safe/joint_target", 10
        )
        runtime.gate(
            expect_armed=False, expect_lease=True, expected_publishers=1
        )
        runtime.arm_once()
        armed = True
        evidence["arm_call_count"] = 1
        runtime.gate(expect_armed=True, expect_lease=True, expected_publishers=1)

        published_at = runtime.publish_target(targets[0])
        machine = V2ExecutionMachine(
            package, q_start, first_published_monotonic=published_at
        )
        evidence["start_admission"] = machine.start_admission
        evidence["publications"].append(
            {
                "target_index": 0,
                "target_rad": list(targets[0]),
                "published_monotonic_s": published_at,
                "replacement": False,
            }
        )
        last_publication = published_at
        minimum_interval = float(package["handoff_policy"]["minimum_replace_interval_s"])
        for transition in range(3):
            deadline = time.monotonic() + 15.0
            matched = None
            while time.monotonic() < deadline:
                runtime.spin(0.03)
                if runtime.status is not None:
                    state = str(runtime.status.get("state", ""))
                    if (
                        state.startswith("FAULT")
                        or state.startswith("DISARMED")
                        or state.endswith("_DISARMED")
                    ):
                        if state.startswith("DISARMED") or state.endswith("_DISARMED"):
                            armed = False
                        raise SmoothBlocked(
                            f"unexpected intermediate terminal state: {state}"
                        )
                runtime.gate(
                    expect_armed=True, expect_lease=True, expected_publishers=1
                )
                if runtime.joint_samples:
                    stamp, actual = runtime.joint_samples[-1]
                    feedback_age = time.monotonic() - stamp
                    release = machine.consider_feedback(
                        actual,
                        feedback_age_s=feedback_age,
                        now_monotonic=time.monotonic(),
                    )
                    matched = None if release is None else dict(release.assessment)
                if (
                    matched is not None
                    and time.monotonic() - last_publication >= minimum_interval
                ):
                    break
            if matched is None:
                raise SmoothBlocked(
                    f"handoff {transition + 1} missed the v2 measured-feedback rule"
                )
            next_target = targets[transition + 1]
            emitted = runtime.publish_target(next_target)
            evidence["replace_events"].append(
                {
                    "transition": transition + 1,
                    "actual_rad": matched["positions_rad"],
                    "assessment": matched,
                    "published_monotonic_s": emitted,
                }
            )
            evidence["publications"].append(
                {
                    "target_index": transition + 1,
                    "target_rad": list(next_target),
                    "published_monotonic_s": emitted,
                    "replacement": True,
                }
            )
            last_publication = emitted

        final_deadline = time.monotonic() + 20.0
        terminal = None
        while time.monotonic() < final_deadline:
            runtime.spin(0.05)
            if runtime.status is None:
                continue
            state = str(runtime.status.get("state", ""))
            if state in ("COARSE_PROGRESS_DISARMED", "CONVERGED_DISARMED"):
                terminal = state
                armed = False
                break
            if state.startswith("FAULT") or state.startswith("DISARMED:"):
                raise SmoothBlocked(f"safe-driver stopped final target: {state}")
        if terminal is None:
            raise SmoothBlocked("final target did not reach a safe terminal state")
        final_q = runtime.stable_q()
        try:
            evidence.update(machine.validate_terminal(terminal, final_q))
        except V2ExecutionBlocked as exc:
            raise SmoothBlocked(str(exc)) from exc
        return evidence
    finally:
        if armed:
            try:
                runtime.disarm_once()
                evidence["abnormal_disarm_called"] = True
            except Exception as exc:
                evidence["abnormal_disarm_error"] = str(exc)
        if runtime.publisher is not None:
            runtime.node.destroy_publisher(runtime.publisher)
            runtime.publisher = None
            runtime.spin(0.15)
        if lease_active:
            try:
                runtime.release_lease(lease_id)
                evidence["lease_release_count"] = 1
            except Exception as exc:
                evidence["lease_release_error"] = str(exc)
        try:
            evidence["post_gate"] = runtime.gate(
                expect_armed=False, expect_lease=False, expected_publishers=0
            )
        except Exception as exc:
            evidence["post_gate_error"] = str(exc)
        runtime.close()
        evidence["ended_at_unix_s"] = time.time()
        _write_json(args.output, evidence)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--package", type=Path, default=DEFAULT_V2_PACKAGE)
    parser.add_argument("--current-start-rad", nargs=6, type=float)
    parser.add_argument("--confirmation-text", default="")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "temp/t102_fast_runtime_no_urdf_clearance_20260816/physical_result.json",
    )
    args = parser.parse_args(argv)
    if not args.preflight and not args.execute:
        print(
            json.dumps(
                {
                    "state": "INERT",
                    "execution_enabled": False,
                    "publisher_created": False,
                    "lease_requested": False,
                    "arm_called": False,
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        package = preflight_v2(args.package)
        if args.preflight:
            if args.current_start_rad is None:
                raise SmoothBlocked("current start is required for preflight")
            start = validate_start_posture(package, args.current_start_rad)
            print(
                json.dumps(
                    {
                        "state": "V2_FAST_PREFLIGHT_PASS",
                        "package_sha256": package["package_sha256"],
                        "target_count": len(package["outbound_targets_rad"]),
                        "start": start,
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = execute(args, package)
    except Exception as exc:
        print(
            json.dumps(
                {"state": "V2_FAST_RUNTIME_ABORTED", "reason": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "state": result["classification"],
                "target_count": len(result["publications"]),
                "replace_count": len(result["replace_events"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
