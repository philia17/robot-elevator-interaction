"""One-shot Prepare + Move client for the released T102 outbound executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from elevator_interfaces.action import PrecontactMove
from elevator_interfaces.srv import PreparePrecontact
from rclpy.action import ActionClient
from rclpy.node import Node


class OneShotClient(Node):
    def __init__(self) -> None:
        super().__init__("t102_supervised_oneshot_client")
        self.prepare = self.create_client(
            PreparePrecontact,
            "/roarm_t102_outbound_executor/prepare_precontact",
        )
        self.move = ActionClient(
            self,
            PrecontactMove,
            "/roarm_t102_outbound_executor/move_to_precontact",
        )
        self.feedback: list[dict] = []

    def _feedback(self, message) -> None:
        value = message.feedback
        self.feedback.append(
            {
                "observed_at_unix_s": time.time(),
                "state": value.state,
                "gate": value.gate,
                "step_index": int(value.step_index),
                "step_count": int(value.step_count),
                "safe_driver_state": value.safe_driver_state,
                "route_artifact_sha256": value.route_artifact_sha256,
            }
        )

    def run(self, target_json: str, target_sha256: str) -> dict:
        if not self.prepare.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Prepare service unavailable")
        if not self.move.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Move Action unavailable")
        request = PreparePrecontact.Request()
        request.canonical_target_snapshot_json = target_json
        request.target_snapshot_sha256 = target_sha256
        request.ui_request_id = f"t102-supervised-oneshot-{time.time_ns()}"
        prepared_future = self.prepare.call_async(request)
        rclpy.spin_until_future_complete(self, prepared_future, timeout_sec=1.0)
        if not prepared_future.done():
            raise RuntimeError("Prepare did not complete inside the ticket window")
        prepared = prepared_future.result()
        prepared_evidence = {
            "accepted": bool(prepared.accepted),
            "reason": prepared.reason,
            "state": prepared.state,
            "ttl_ms": int(prepared.ttl_ms),
            "release_manifest_sha256": prepared.release_manifest_sha256,
            "route_artifact_sha256": prepared.route_artifact_sha256,
            "planned_target_snapshot_sha256": prepared.planned_target_snapshot_sha256,
        }
        if not prepared.accepted:
            return {"prepare": prepared_evidence, "move": None, "feedback": []}
        goal = PrecontactMove.Goal()
        goal.authorization_id = prepared.authorization_id
        goal.target_snapshot_sha256 = prepared.planned_target_snapshot_sha256
        goal.confirmation_text = "MOVE_TO_PRECONTACT_ONLY"
        goal_future = self.move.send_goal_async(goal, feedback_callback=self._feedback)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=1.0)
        if not goal_future.done() or not goal_future.result().accepted:
            raise RuntimeError("Move goal was not accepted inside the ticket window")
        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=180.0)
        if not result_future.done():
            goal_future.result().cancel_goal_async()
            raise RuntimeError("Move Action exceeded the bounded client timeout")
        wrapper = result_future.result()
        result = wrapper.result
        return {
            "prepare": prepared_evidence,
            "move": {
                "action_status": int(wrapper.status),
                "success": bool(result.success),
                "terminal_code": result.terminal_code,
                "terminal_reason": result.terminal_reason,
                "state": result.state,
                "target_snapshot_sha256": result.target_snapshot_sha256,
                "release_manifest_sha256": result.release_manifest_sha256,
                "route_artifact_sha256": result.route_artifact_sha256,
                "m0_frozen_artifact_sha256": result.m0_frozen_artifact_sha256,
            },
            "feedback": self.feedback,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target_path = Path(args.target_json).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    target_json = target_path.read_text(encoding="utf-8")
    rclpy.init()
    node = OneShotClient()
    try:
        evidence = node.run(target_json, args.target_sha256)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, sort_keys=True))
    move = evidence.get("move")
    return 0 if isinstance(move, dict) and move.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
