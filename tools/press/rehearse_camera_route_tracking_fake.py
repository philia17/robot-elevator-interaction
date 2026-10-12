#!/usr/bin/env python3
"""Hardware-isolated full press-cycle rehearsal (no ROS, no serial).

Drives MultiJointPressTask against SafeController + TrackingFakeSerialTransport
for a camera-route (or taught M2) artifact. Use this before asking for a
one-shot real-motion authorization. Never opens /dev/serial.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_DRIVER_PACKAGE = REPO_ROOT / "ros2_arm" / "roarm_safe_driver"
if str(SAFE_DRIVER_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SAFE_DRIVER_PACKAGE))

from roarm_safe_driver.core import SafeController  # noqa: E402
from roarm_safe_driver.transport import TrackingFakeSerialTransport  # noqa: E402
from tools.press.m1_continuous_return_task import (  # noqa: E402
    TaskAuthorization,
    measured_digest,
)
from tools.press.m2_press_task import (  # noqa: E402
    CONFIRMATION_REQUIRED_BEFORE,
    MultiJointPressTask,
    POST_VERIFY_REQUIRED_OUTCOME,
    PressTaskState,
)


DEFAULT_ARTIFACT = (
    REPO_ROOT / "config" / "press" / "pilots" / "m3_camera_demo_route_v1.json"
)


def _parse_feedback(line: bytes) -> list[float]:
    frame = json.loads(line.decode())
    import math

    return [
        float(frame["b"]),
        float(frame["s"]),
        float(frame["e"]),
        float(frame["t"]),
        float(frame["r"]),
        math.pi - float(frame["g"]),
    ]


def rehearse(artifact_path: Path) -> dict:
    task = MultiJointPressTask(artifact_path)
    start = tuple(task.artifact["endpoints"]["press_standby_start_rad"])
    transport = TrackingFakeSerialTransport(start)
    writes: list[bytes] = []

    def write(data: bytes) -> None:
        transport.write(data)
        writes.append(data)

    controller = SafeController(
        write,
        dry_run=False,
        joint_max_delta_rad=tuple(
            task.artifact["joint_contract"]["default_joint_max_delta_rad"]
        ),
    )

    def feed(positions, now):
        from roarm_safe_driver.core import Feedback

        controller.accept_feedback(
            Feedback(
                positions=tuple(float(v) for v in positions),
                loads=(0, 0, 0, 0, 0, 0),
                torque_enabled=(True, True, True, True, True, True),
                voltage=12.1,
                received_at=now,
                raw={"source": "tracking_fake_rehearsal"},
            )
        )

    now = time.monotonic()
    feed(start, now)
    task.authorize(
        TaskAuthorization(
            token_id="rehearsal-token",
            artifact_sha256=task.sha256,
            start_measured_digest=measured_digest(start),
            expires_monotonic=now + 3600.0,
            confirmation_text="isolated tracking-fake rehearsal",
        ),
        start,
        now,
    )

    confirmations: list[str] = []
    steps = 0
    while task.state in (
        PressTaskState.READY_NEXT_STEP,
        PressTaskState.AWAIT_PHASE_CONFIRMATION,
        PressTaskState.WAIT_DRIVER,
    ):
        if task.state is PressTaskState.AWAIT_PHASE_CONFIRMATION:
            phase = task.next_phase
            assert phase in CONFIRMATION_REQUIRED_BEFORE
            task.confirm_phase(phase, f"rehearsal confirms {phase}")
            confirmations.append(phase)
            continue
        candidate = task.issue_next(now)
        if candidate is None:
            continue
        measured = _parse_feedback(transport.sample_line())
        feed(measured, now)
        controller.arm(now)
        payload = controller.command_multi(candidate.target_positions_rad, now)
        decoded = json.loads(payload)
        if decoded.get("T") != 102:
            raise RuntimeError(f"expected T102, got {decoded.get('T')}")
        # Instant settle via tracking fake + synthetic convergence samples.
        target = _parse_feedback(transport.sample_line())
        settle = now + 0.1
        for sample_idx in range(25):
            feed(target, settle + sample_idx * 0.05)
        if controller.last_result != "CONVERGED_DISARMED":
            raise RuntimeError(f"driver did not converge: {controller.last_result}")
        now = settle + 2.0
        task.observe(
            target,
            {
                "state": "CONVERGED_DISARMED",
                "armed": False,
                "fault": "",
                "torque_enabled": [True] * 6,
                "voltage": 12.12,
                "joint_target_publisher_count": 0,
            },
            now,
        )
        steps += 1

    if task.state is not PressTaskState.AWAIT_POST_VERIFY:
        raise RuntimeError(f"unexpected state after motion: {task.state}")
    task.record_post_verify(
        {
            "outcome": POST_VERIFY_REQUIRED_OUTCOME,
            "trial_id": "tracking_fake_rehearsal",
            "target_label": "1",
        }
    )
    if task.state is not PressTaskState.COMPLETE_VERIFIED:
        raise RuntimeError(f"post-verify failed: {task.state}")
    return {
        "passed": True,
        "artifact": str(artifact_path),
        "artifact_sha256": task.sha256,
        "waypoint_steps": steps,
        "t102_writes": len(writes),
        "confirmations": confirmations,
        "final_state": task.state.value,
        "serial_used": False,
        "ros_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    args = parser.parse_args()
    report = rehearse(Path(args.artifact).expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
