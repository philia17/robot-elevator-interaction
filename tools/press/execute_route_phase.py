#!/usr/bin/env python3
"""Drive one phase of an m3 press-route artifact with one-shot chunk moves.

The RoArm firmware has a ~1-2 deg T102 positioning deadband and ignores
repeated absolute targets, so the M2 adapter's strict per-step convergence is
not usable on this hardware.  This tool instead re-plans the phase from the
current measured pose into a few large chunks (per-joint delta capped at 10
deg so every chunk passes the driver's ceiling) and issues each chunk as a
fresh one-shot command through the driver's standby service.  Large chunks
execute almost fully; the reported residual reflects the firmware's natural
landing accuracy and the operator verifies visually before any contact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import rclpy


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
for _candidate in (
    REPO_ROOT / "ros2_arm" / "roarm_safe_driver",
    REPO_ROOT / "roarm_safe_driver",
):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from press.move_to_press_standby import (  # noqa: E402
    MoveError,
    MoveToStandby,
    STANDBY_PARAM,
    STANDBY_SPEED_PARAM,
)

from tools.press.m2_multi_joint_route import (  # noqa: E402
    _interpolate_waypoints,
    _required_waypoints,
)


# Per-joint chunk ceiling: below the driver's elbow/wrist hard caps, big
# enough to execute well.  Conservative for every joint.
CHUNK_MAX_DELTA_RAD = tuple(math.radians(10.0) for _ in range(6))
PHASE_ENDPOINT_KEY = {
    "outbound": "pre_contact_rad",
    "press": "press_target_rad",
    "retract": "retract_target_rad",
    "return": "return_target_rad",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, help="m2_multi_joint_route_v1 artifact")
    parser.add_argument(
        "--phase",
        required=True,
        choices=tuple(PHASE_ENDPOINT_KEY),
    )
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        print("Refusing: pass --confirm to move the robot.")
        return 1

    artifact = json.loads(Path(args.route).read_text())
    endpoint_key = PHASE_ENDPOINT_KEY[args.phase]
    endpoint = [float(v) for v in artifact["endpoints"][endpoint_key]]

    rclpy.init()
    node = MoveToStandby()
    try:
        originals = node._read_parameters([STANDBY_PARAM, STANDBY_SPEED_PARAM])
        node._check_driver_status("start")
        node._wait_until_idle()
        feedback = node._wait_feedback()
        current = [float(v) for v in feedback["positions_rad"]]
        chunks = _interpolate_waypoints(
            tuple(current),
            tuple(endpoint),
            _required_waypoints(
                tuple(current),
                tuple(endpoint),
                max_delta_rad=CHUNK_MAX_DELTA_RAD,
            ),
        )
        print(
            f"PHASE {args.phase}: {len(chunks)} chunk(s) to "
            f"{[round(v, 4) for v in endpoint]}",
            flush=True,
        )
        node._ensure_coexistence_publisher()
        for index, chunk in enumerate(chunks, start=1):
            target = [float(v) for v in chunk]
            print(
                f"[{args.phase} {index}/{len(chunks)}] "
                f"target={[round(v, 4) for v in target]}",
                flush=True,
            )
            node._write_parameter(STANDBY_PARAM, target)
            node._write_parameter(
                STANDBY_SPEED_PARAM, int(originals[STANDBY_SPEED_PARAM])
            )
            try:
                node._call_standby()
                node._wait_terminal(index)
            except MoveError as exc:
                message = str(exc)
                if "convergence timeout" in message or "FAULT" in message.upper():
                    # Partial move: clear the benign fault and let the next
                    # absolute chunk carry the residual forward.
                    node.get_logger().warning(f"{message}; clearing and continuing")
                    node._clear_fault()
                    node._wait_idle_after_fault()
                else:
                    node._disarm_stop()
                    raise
            finally:
                try:
                    node._write_parameter(STANDBY_PARAM, originals[STANDBY_PARAM])
                    node._write_parameter(
                        STANDBY_SPEED_PARAM, originals[STANDBY_SPEED_PARAM]
                    )
                except MoveError as exc:
                    node.get_logger().error(f"could not restore driver params: {exc}")
            feedback = node._wait_feedback()
            measured = [float(v) for v in feedback["positions_rad"]]
            residual = max(
                abs(float(a) - float(b)) for a, b in zip(measured, target)
            )
            print(
                f"[{args.phase} {index}/{len(chunks)}] done "
                f"residual={math.degrees(residual):.2f} deg",
                flush=True,
            )
        print(f"PHASE {args.phase} COMPLETE", flush=True)
        return 0
    finally:
        node._destroy_coexistence_publisher()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
