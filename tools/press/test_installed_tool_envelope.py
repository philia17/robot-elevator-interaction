from __future__ import annotations

from press.installed_tool_envelope import seal_record, validate_installed_tool_envelope


def _sha(char: str) -> str:
    return char * 64


def _pose():
    return {"translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}


def _part(kind: str, char: str):
    return {
        "body_id": f"press_{kind}", "part_kind": kind, "primitive": "capsule", "frame_id": "hand_tcp",
        "attachment_pose_hand_tcp": _pose(), "conservative_radius_m": 0.01,
        "clearance_margin_m": 0.003, "source_sha256": _sha(char),
        "geometry": {"endpoint_a_hand_tcp_m": [0, 0, 0], "endpoint_b_hand_tcp_m": [0, 0, 0.1]},
    }


def _record(**changes):
    value = {
        "schema": "installed_tool_envelope_record_v1", "mode": "OFFLINE_STATIC_GEOMETRY",
        "installation": {"installation_id": "fixture-install", "tool_id": "fixture-tool", "parent_frame_id": "hand_tcp", "hand_tcp_attachment": _pose(), "installation_source_sha256": _sha("a")},
        "tool_parts": [_part("shaft", "b"), _part("soft_tip", "c"), _part("cable", "d")],
        "rear_obstacles_base": [{"obstacle_id": "rear-panel", "primitive": "box", "frame_id": "roarm_base_link", "pose_base": _pose(), "conservative_margin_m": 0.004, "source_sha256": _sha("e"), "geometry": {"size_m": [0.1, 0.1, 0.1]}}],
        "review": {"release_status": "released_static", "review_record_sha256": _sha("f"), "conservative_geometry_confirmed": True, "installation_identity_confirmed": True, "rear_obstacles_confirmed": True},
    }
    value.update(changes)
    return seal_record(value)


def test_complete_offline_record_is_static_only_pass():
    receipt = validate_installed_tool_envelope(_record())
    assert receipt["result"] == "pass"
    assert receipt["tool_geometry_sha256"] == receipt["artifact_sha256"]
    assert receipt["execution_ready"] is False
    assert receipt["motion_authorized"] is False


def test_blocked_template_review_fails_closed():
    value = _record()
    value["review"]["release_status"] = "blocked_static"
    receipt = validate_installed_tool_envelope(seal_record(value))
    assert receipt["result"] == "blocked"
    assert "unreleased_static" in receipt["missing_release_reasons"][0]


def test_missing_cable_fails_closed():
    value = _record(tool_parts=[_part("shaft", "b"), _part("soft_tip", "c")])
    receipt = validate_installed_tool_envelope(value)
    assert receipt["result"] == "blocked"
    assert "shaft, soft_tip, and cable" in receipt["missing_release_reasons"][0]


def test_rear_obstacle_not_base_frame_fails_closed():
    value = _record()
    value["rear_obstacles_base"][0]["frame_id"] = "camera_optical_frame"
    receipt = validate_installed_tool_envelope(seal_record(value))
    assert receipt["result"] == "blocked"
    assert "roarm_base_link" in receipt["missing_release_reasons"][0]


def test_tampered_record_hash_fails_closed():
    value = _record()
    value["installation"]["tool_id"] = "other-tool"
    receipt = validate_installed_tool_envelope(value)
    assert receipt["result"] == "blocked"
    assert "record hash mismatch" in receipt["missing_release_reasons"]
