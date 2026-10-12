from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from press.t102_session_template_core import TemplateBlocked, bind_session, load_policy


ROOT = Path(__file__).resolve().parents[2]


def _snapshot_from_base_artifact(path: Path, *, detection="button_004", stamp=1_000_000_000, nonce="capture-1"):
    value = json.loads(path.read_text())
    return {
        "schema": "m0_demo_freeze_v1",
        "frozen_target": {
            "target_label": "BUTTON 1",
            "detection_id": detection,
            "source_stamp_ns": stamp,
            "confidence": 0.99,
            "status": "READY",
            "capture_nonce": nonce,
            "snapshot_id": nonce,
            "boot_epoch": "boot-1",
            "cart_pose_epoch": "cart-1",
            "camera_point_m": dict(zip("xyz", value["camera_point_m"])),
            "surface_geometry": {
                "estimate_valid": True,
                "provenance_complete": True,
                "section_b_calibration_id": value["calibration_id"],
                "panel_outward_normal": dict(zip("xyz", value["panel_outward_normal_camera"])),
            },
        },
    }


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("relative", [
    "temp/t102_immediate_120mm_motion_20260815/base_target_120mm.json",
    "temp/t102_immediate_120mm_motion_20260815/base_target_120mm_1510.json",
    "temp/t102_speed_lease_deadlock_fix_and_motion_20260815/base_target_v4.json",
    "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json",
])
def test_same_placement_artifacts_fit_exact_derived_window(relative):
    policy = load_policy()
    snapshot = _snapshot_from_base_artifact(ROOT / relative)
    session = bind_session(snapshot, _digest(snapshot), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    assert session.template_code == "TEMPLATE_MATCH"
    assert len(session.session_id) == 64
    assert session.route_sha256 == policy["template"]["route_package_sha256"]


def test_target_window_miss_reports_observed_base_offset():
    policy = load_policy()
    snapshot = _snapshot_from_base_artifact(ROOT / "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json")
    snapshot["frozen_target"]["camera_point_m"]["x"] += 0.01
    with pytest.raises(TemplateBlocked) as caught:
        bind_session(snapshot, _digest(snapshot), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    assert caught.value.code == "TEMPLATE_MISS"
    assert len(caught.value.offset_m) == 3


def test_new_freeze_start_or_route_change_changes_or_invalidates_session():
    policy = load_policy()
    source = ROOT / "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json"
    first = _snapshot_from_base_artifact(source, stamp=1_000_000_000, nonce="capture-a")
    second = _snapshot_from_base_artifact(source, stamp=1_000_000_000, nonce="capture-b")
    a = bind_session(first, _digest(first), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    b = bind_session(second, _digest(second), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    assert a.session_id != b.session_id
    drifted = list(policy["template"]["start_positions_rad"])
    drifted[0] += 2 * policy["template"]["start_tolerance_rad"]
    with pytest.raises(TemplateBlocked, match="Standard Standby") as caught:
        bind_session(first, _digest(first), drifted, policy=policy, now_unix_ns=1_500_000_000)
    assert caught.value.code == "START_DRIFT"
    changed = json.loads(json.dumps(policy))
    changed["template"]["route_package_sha256"] = "f" * 64
    c = bind_session(first, _digest(first), policy["template"]["start_positions_rad"], policy=changed, now_unix_ns=1_500_000_000)
    assert c.session_id != a.session_id


def test_calibration_identity_is_bound_and_prepare_is_pure_zero_motion_source():
    policy = load_policy()
    snapshot = _snapshot_from_base_artifact(ROOT / "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json")
    snapshot["frozen_target"]["surface_geometry"]["section_b_calibration_id"] = "other"
    with pytest.raises(TemplateBlocked) as caught:
        bind_session(snapshot, _digest(snapshot), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    assert caught.value.code == "SESSION_INVALID"
    source = (ROOT / "tools/press/t102_session_template_core.py").read_text()
    import ast
    tree = ast.parse(source)
    imports = {alias.name.lower() for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {str(node.module).lower() for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imports.intersection({"rclpy", "subprocess", "urdf", "trimesh"})
    for forbidden_call in ("create_publisher(", "create_client("):
        assert forbidden_call not in source


@pytest.mark.parametrize("mutation,code", [
    (lambda s: s["frozen_target"].update(status="STABILIZING"), "TARGET_NOT_READY"),
    (lambda s: s["frozen_target"].update(confidence=0.0), "TARGET_CONFIDENCE_INVALID"),
    (lambda s: s["frozen_target"].update(capture_nonce=""), "SESSION_INVALID"),
])
def test_status_confidence_and_capture_generation_fail_closed(mutation, code):
    policy = load_policy()
    snapshot = _snapshot_from_base_artifact(ROOT / "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json")
    mutation(snapshot)
    with pytest.raises(TemplateBlocked) as caught:
        bind_session(snapshot, _digest(snapshot), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=1_500_000_000)
    assert caught.value.code == code


def test_target_freshness_is_explicit_and_fail_closed():
    policy = load_policy()
    snapshot = _snapshot_from_base_artifact(ROOT / "temp/t102_build_execution_release_deploy_and_move_20260815/base_target_v5.json")
    with pytest.raises(TemplateBlocked) as caught:
        bind_session(snapshot, _digest(snapshot), policy["template"]["start_positions_rad"], policy=policy, now_unix_ns=4_000_000_001)
    assert caught.value.code == "TARGET_STALE"
