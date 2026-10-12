#!/usr/bin/env python3
"""T5 observe-pose audit-only preflight.

Legacy T5 80-100 mm observation authorization is retired. This module never
creates, copies, mutates, persists, or returns motion authority. The CLI
rejects --confirm/--auth-id/--write-plan before reading the plan artifact or
writing anything.

Reports are unambiguous audit records:

- report_semantics = legacy_t5_audit_only
- effective_motion_authority = false
- authority_creation_supported = false
- historical artifact facts are exposed only under stored_* names
- artifact bytes are read once, hashed once, and JSON-decoded from those same
  bytes; plan_sha256 is retained only as an exact-bytes compatibility alias of
  artifact_sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    REPO_ROOT
    / "config"
    / "press"
    / "pilots"
    / "t5_visual_observe_auth_plan_20260805.json"
)
HISTORICAL_ARCHIVE = (
    REPO_ROOT
    / "config"
    / "press"
    / "pilots"
    / "archive"
    / "t5_visual_observe_authorization_20260807.json"
)
KNOWN_ARTIFACT_PATHS = (DEFAULT_PLAN, HISTORICAL_ARCHIVE)

LEGACY_T5_AUTHORITY_RETIRED = "LEGACY_T5_AUTHORITY_RETIRED"
LEGACY_T5_UNRECOGNIZED_ARTIFACT = "LEGACY_T5_UNRECOGNIZED_ARTIFACT"

_DEFAULT_PLAN_RESOLVED = DEFAULT_PLAN.resolve()
_HISTORICAL_ARCHIVE_RESOLVED = HISTORICAL_ARCHIVE.resolve()
_REQUIRED_PLAN_KEYS = (
    "plan_id",
    "required_confirmation_text",
    "motion_authorized",
    "execution_ready",
    "press_authorized",
    "observation_gap_m",
)
_HARD_LOCK_KEYS = (
    "execution_ready",
    "press_authorized",
    "contact_authorized",
    "vendor_initialize_authorized",
    "publish_tf",
)


class PreflightError(RuntimeError):
    pass


def _decode_plan_document(data: bytes) -> dict[str, Any]:
    document = json.loads(data.decode("utf-8"))
    missing = [key for key in _REQUIRED_PLAN_KEYS if key not in document]
    if missing:
        raise PreflightError(f"plan missing {', '.join(missing)}")
    gap = document["observation_gap_m"]
    try:
        gap_min = float(gap["min"])
        gap_max = float(gap["max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError("observation_gap_m must contain numeric min/max") from exc
    if not (0.079 <= gap_min < gap_max <= 0.101):
        raise PreflightError("observation_gap_m out of T5 80-100 mm contract")
    return document


def load_plan(path: Path) -> dict[str, Any]:
    return _decode_plan_document(path.read_bytes())


def check_static_gates(plan: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in _HARD_LOCK_KEYS:
        if plan.get(key) is not False:
            failures.append(f"{key} must be false")
    return failures


def check_t4_tests_present() -> list[str]:
    failures: list[str] = []
    module = REPO_ROOT / "tools" / "press" / "visual_closure_t4_fake.py"
    tests = REPO_ROOT / "tools" / "press" / "test_visual_closure_t4_fake.py"
    if not module.is_file():
        failures.append(f"missing T4 module: {module}")
    if not tests.is_file():
        failures.append(f"missing T4 tests: {tests}")
    return failures


def authorize_plan(
    plan: dict[str, Any], confirmation: str, auth_id: str
) -> dict[str, Any]:
    """Always reject legacy T5 authorization creation.

    Kept import-compatible for callers, but no input is inspected, copied,
    mutated, persisted, or returned. Every call raises deterministically.
    """
    raise PreflightError(
        f"{LEGACY_T5_AUTHORITY_RETIRED}: legacy T5 80-100 mm observe "
        "authorization is retired; no input can create or return authority"
    )


def ros_preflight() -> dict[str, Any]:
    """Read-only ROS checks. Never publishes joint_target."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    report: dict[str, Any] = {
        "ros_used": True,
        "serial_used": False,
        "motion_commanded": False,
    }
    rclpy.init()
    node = Node("t5_observe_preflight")
    try:
        joint_target = "/roarm_safe/joint_target"
        color = "/lift_robot/camera/color/image_raw"
        depth = "/lift_robot/camera/depth/image_raw"
        report["joint_target_publisher_count"] = int(
            node.count_publishers(joint_target)
        )
        report["color_publisher_count"] = int(node.count_publishers(color))
        report["depth_publisher_count"] = int(node.count_publishers(depth))
        # Best-effort one spin for discovery.
        rclpy.spin_once(node, timeout_sec=0.5)
        report["joint_target_publisher_count"] = int(
            node.count_publishers(joint_target)
        )
        report["color_publisher_count"] = int(node.count_publishers(color))
        report["depth_publisher_count"] = int(node.count_publishers(depth))
        failures = []
        if report["joint_target_publisher_count"] != 0:
            failures.append(
                f"{joint_target} publisher count "
                f"{report['joint_target_publisher_count']} != 0"
            )
        if report["color_publisher_count"] < 1:
            failures.append(f"{color} has no publishers")
        report["failures"] = failures
        report["passed"] = len(failures) == 0
        # qos import kept for future subscription probes
        _ = qos_profile_sensor_data
        return report
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _artifact_role(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return "unrecognized_artifact"
    if resolved == _DEFAULT_PLAN_RESOLVED:
        return "inert_default_template"
    if resolved == _HISTORICAL_ARCHIVE_RESOLVED:
        return "historical_legacy_authorization"
    return "unrecognized_artifact"


def build_report(
    plan: dict[str, Any],
    *,
    plan_path: Path,
    ros_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build an audit-only report.

    `plan` is accepted for signature compatibility, but the bytes at
    `plan_path` are authoritative. The artifact is read once, hashed once, and
    JSON-decoded from those same bytes to avoid TOCTOU. `plan_sha256` is an
    exact-bytes compatibility alias of `artifact_sha256`.
    """
    data = plan_path.read_bytes()
    document = _decode_plan_document(data)
    artifact_role = _artifact_role(plan_path)
    static_failures = check_static_gates(document)
    t4_failures = check_t4_tests_present()
    audit_failures = list(static_failures) + list(t4_failures)
    if artifact_role == "unrecognized_artifact":
        audit_failures.append(
            "artifact_path is not the known default template or historical archive"
        )
    audit_failures.extend((ros_report or {}).get("failures") or [])
    audit_passed = not audit_failures
    artifact_sha256 = hashlib.sha256(data).hexdigest()
    report: dict[str, Any] = {
        "kind": "t5_observe_preflight",
        "schema_version": "1.0.0",
        "report_semantics": "legacy_t5_audit_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "artifact_role": artifact_role,
        "artifact_sha256": artifact_sha256,
        # Exact-bytes compatibility alias: same bytes are hashed and decoded.
        "plan_sha256": artifact_sha256,
        "stored_motion_authorized": bool(document.get("motion_authorized")),
        "stored_user_approved": bool(document.get("user_approved")),
        "stored_capture_enabled": bool(document.get("capture_enabled")),
        "effective_motion_authority": False,
        "authority_creation_supported": False,
        "execution_ready": False,
        "serial_used": False,
        "motion_commanded": False,
        "static_failures": static_failures,
        "t4_failures": t4_failures,
        "audit_failures": audit_failures,
        "failures": audit_failures,
        "audit_passed": audit_passed,
        "passed": audit_passed,
        "next_operator_step": (
            "Audit only. Legacy T5 observe authority is retired; no operator "
            "step can create effective motion authority."
        ),
        "ros": ros_report,
    }
    if "authorization_id" in document:
        report["stored_authorization_id"] = document["authorization_id"]
    if "authorized_at_utc" in document:
        report["stored_authorized_at_utc"] = document["authorized_at_utc"]
    if "plan_id" in document:
        report["stored_plan_id"] = document["plan_id"]
    if "plan_status" in document:
        report["stored_plan_status"] = document["plan_status"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN,
        help="Known default template or historical archive (resolved exactly)",
    )
    parser.add_argument(
        "--confirm",
        default=None,
        help="Deprecated: rejected; legacy T5 authorization creation is retired",
    )
    parser.add_argument(
        "--auth-id",
        default=None,
        help="Deprecated: rejected; legacy T5 authorization creation is retired",
    )
    parser.add_argument(
        "--ros",
        action="store_true",
        help="Run read-only ROS publisher checks (AGX). Never commands motion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write audit-only JSON report",
    )
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="Deprecated: rejected; legacy T5 authorization persistence is retired",
    )
    args = parser.parse_args(argv)

    if args.confirm is not None or args.auth_id is not None or args.write_plan:
        raise SystemExit(
            f"{LEGACY_T5_AUTHORITY_RETIRED}: legacy T5 observe authority is "
            "retired; --confirm/--auth-id/--write-plan are rejected before any "
            "artifact read or disk write"
        )

    try:
        resolved_plan = args.plan.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(
            f"{LEGACY_T5_UNRECOGNIZED_ARTIFACT}: cannot resolve --plan: {exc}"
        ) from exc
    if resolved_plan not in (_DEFAULT_PLAN_RESOLVED, _HISTORICAL_ARCHIVE_RESOLVED):
        raise SystemExit(
            f"{LEGACY_T5_UNRECOGNIZED_ARTIFACT}: --plan must be exactly the "
            "known default template or known historical archive; refusing "
            "unrecognized artifact"
        )

    ros_report = None
    if args.ros:
        ros_report = ros_preflight()

    report = build_report({}, plan_path=args.plan, ros_report=ros_report)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(text)
    return 0 if report["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
