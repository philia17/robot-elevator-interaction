from __future__ import annotations

from press.precontact_rear_continuous_keepout import (
    METHOD, build_rear_continuous_keepout_release, seal_request,
)


def _request(**changes):
    value = {
        "schema": "rear_continuous_keepout_request_v1",
        "mode": "OFFLINE_STATIC_GEOMETRY",
        "preview_hash": "a" * 64,
        "route_hash": "b" * 64,
        "panel_plane": {"frame_id": "roarm_base_link", "point_base_m": [0, 0, 0.5], "outward_normal_base": [0, 0, 1]},
        "minimum_front_clearance_m": 0.002,
        "model": {"urdf_sha256": "c" * 64, "collision_geometry_sha256": "d" * 64, "tool_geometry_sha256": "e" * 64, "model_status": "released_static"},
        "body_ids": ["link1", "link2", "tool"],
        "segments": [
            {"start_waypoint_index": 0, "end_waypoint_index": 1, "method": METHOD, "covered_body_ids": ["link1", "link2", "tool"], "certified_minimum_signed_distance_m": 0.013, "certificate_sha256": "f" * 64},
            {"start_waypoint_index": 1, "end_waypoint_index": 2, "method": METHOD, "covered_body_ids": ["link1", "link2", "tool"], "certified_minimum_signed_distance_m": 0.009, "certificate_sha256": "1" * 64},
        ],
    }
    value.update(changes)
    return seal_request(value)


def test_complete_interval_certificates_create_static_only_pass():
    report = build_rear_continuous_keepout_release(_request())
    assert report["release_status"] == "released_static"
    assert report["result"] == "pass"
    assert report["checked_segment_count"] == 2
    assert report["minimum_certified_clearance_m"] == 0.009
    assert report["execution_ready"] is False
    assert report["motion_authorized"] is False


def test_waypoint_only_or_missing_segment_fails_closed():
    report = build_rear_continuous_keepout_release(_request(segments=[]))
    assert report["release_status"] == "blocked_static"
    assert "continuous segment certificates are required" in report["missing_release_reasons"]


def test_partial_body_coverage_fails_closed():
    request = _request()
    request["segments"][1]["covered_body_ids"] = ["link1", "tool"]
    request = seal_request(request)
    report = build_rear_continuous_keepout_release(request)
    assert report["release_status"] == "blocked_static"
    assert "exact full robot/tool body set" in report["missing_release_reasons"][0]


def test_insufficient_rear_margin_fails_closed():
    request = _request()
    request["segments"][0]["certified_minimum_signed_distance_m"] = 0.001
    request = seal_request(request)
    report = build_rear_continuous_keepout_release(request)
    assert report["release_status"] == "blocked_static"
    assert "below required margin" in report["missing_release_reasons"][0]


def test_tampered_request_hash_is_blocked():
    request = _request()
    request["route_hash"] = "9" * 64
    report = build_rear_continuous_keepout_release(request)
    assert report["release_status"] == "blocked_static"
    assert "request hash mismatch" in report["missing_release_reasons"]
