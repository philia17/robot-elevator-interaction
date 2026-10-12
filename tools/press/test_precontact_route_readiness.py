"""Focused tests for the non-executable static route-readiness report."""

from __future__ import annotations

import copy

import pytest

from press.precontact_route_readiness import (
    RouteReadinessBlocked,
    build_route_readiness_report,
    canonical_sha256,
    validate_route_readiness_report,
)


def _preview():
    value = {
        "schema": "precontact_preview_v1",
        "target_identity": "Button 1",
        "precontact_fingertip_base_m": [0.1, 0.2, 0.3],
        "precontact_distance_m": 0.05,
        "execution_ready": False,
        "motion_authorized": False,
        "trajectory_generated": False,
        "ik_run": False,
    }
    value["canonical_payload_sha256"] = canonical_sha256(value)
    value["preview_hash"] = value["canonical_payload_sha256"]
    value["preview_id"] = "precontact-preview-test"
    return value


def _releases(preview):
    return {
        name: {
            "schema": schema,
            "release_id": f"{name}-static-review",
            "artifact_sha256": "a" * 64,
            "preview_hash": preview["preview_hash"],
            "release_status": "released_static",
            "result": "pass",
        }
        for name, schema in (
            ("ik_fk", "ik_fk_static_release_v1"),
            ("joint_limits", "joint_limits_static_release_v1"),
            ("rear_continuous_keepout", "rear_continuous_keepout_static_release_v1"),
        )
    }


def test_current_diagnostic_preview_generates_explicit_blocked_report():
    preview = _preview()
    report = build_route_readiness_report(preview)

    assert report["preview_hash"] == preview["preview_hash"]
    assert report["execution_ready"] is False
    assert report["motion_authorized"] is False
    assert report["trajectory_generated"] is False
    assert report["command_generated"] is False
    assert report["adapter_consumable"] is False
    assert report["contact_authorized"] is False
    assert report["automatic_retry"] is False
    assert report["route_readiness"] == "BLOCKED_DIAGNOSTIC_OR_UNRELEASED"
    assert "FN_REFERENCE_DIAGNOSTIC_ONLY" in report["missing_release_reasons"]
    assert "IK_FK_STATIC_RELEASE_MISSING" in report["missing_release_reasons"]
    assert "JOINT_LIMITS_STATIC_RELEASE_MISSING" in report["missing_release_reasons"]
    assert "REAR_CONTINUOUS_KEEPOUT_STATIC_RELEASE_MISSING" in report["missing_release_reasons"]
    validate_route_readiness_report(report, preview=preview)


def test_released_static_receipts_still_cannot_authorize_motion():
    preview = _preview()
    report = build_route_readiness_report(preview, static_releases=_releases(preview))

    assert all(item["accepted"] is True for item in report["static_release_evidence"].values())
    assert report["execution_ready"] is False
    assert report["motion_authorized"] is False
    assert report["trajectory_generated"] is False
    assert "FN_REFERENCE_DIAGNOSTIC_ONLY" in report["missing_release_reasons"]


def test_unbound_or_unreleased_receipt_is_a_clear_blocker():
    preview = _preview()
    releases = _releases(preview)
    releases["ik_fk"]["preview_hash"] = "b" * 64
    releases["joint_limits"]["release_status"] = "diagnostic"
    report = build_route_readiness_report(preview, static_releases=releases)

    assert "IK_FK_STATIC_RELEASE_UNACCEPTED" in report["missing_release_reasons"]
    assert "JOINT_LIMITS_STATIC_RELEASE_UNACCEPTED" in report["missing_release_reasons"]


def test_preview_or_report_tampering_fails_closed():
    preview = _preview()
    preview["precontact_distance_m"] = 0.051
    with pytest.raises(RouteReadinessBlocked, match="canonical payload hash mismatch"):
        build_route_readiness_report(preview)

    clean = _preview()
    report = build_route_readiness_report(clean)
    tampered = copy.deepcopy(report)
    tampered["motion_authorized"] = True
    with pytest.raises(RouteReadinessBlocked, match="motion_authorized"):
        validate_route_readiness_report(tampered, preview=clean)


def test_module_has_no_robot_or_execution_imports():
    from pathlib import Path

    source = Path(__file__).with_name("precontact_route_readiness.py").read_text(encoding="utf-8")
    for forbidden in ("import rclpy", "create_publisher", "create_client", "ActionClient", "serial.Serial", "subprocess"):
        assert forbidden not in source
