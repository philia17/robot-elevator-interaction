"""Focused tests for the SIM_ONLY keep-out/corridor preview."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from press.precontact_keepout_preview import (
    MODE,
    STATUS,
    KeepoutPreviewBlocked,
    build_keepout_preview_report,
    canonical_report_sha256,
    seal_preview_request,
    validate_keepout_preview_report,
)


def _request(*, observed_points=None, proxy_points=None):
    return seal_preview_request(
        {
            "schema": "precontact_keepout_preview_request_v1",
            "mode": MODE,
            "status": STATUS,
            "geometry": {
                "frame_id": "roarm_base_link",
                "A_base_m": [0.0, 0.0, 0.500],
                "N_out_base": [0.0, 0.0, 1.0],
                "P_pre_base_m": [0.0, 0.0, 0.550],
                "Fn_base_m": [0.200, 0.0, 0.600],
            },
            "rear_keepout": {
                "model": "synthetic_panel_rear_halfspace_v1",
                "synthetic_only": True,
                "minimum_front_signed_distance_m": 0.010,
                "proxy_points_base_m": [] if proxy_points is None else proxy_points,
            },
            "front_corridor": {
                "model": "observed_cartesian_capsule_v1",
                "field_observed_only": True,
                "corridor_radius_m": 0.020,
                "observed_points_base_m": (
                    [[0.100, 0.100, 0.575]]
                    if observed_points is None
                    else observed_points
                ),
            },
        }
    )


def _replace(request, path, value):
    candidate = copy.deepcopy(request)
    target = candidate
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return seal_preview_request(candidate)


def test_clear_observed_samples_remain_pending_and_never_executable():
    request = _request()
    report = build_keepout_preview_report(request)
    assert report["mode"] == "SIM_ONLY"
    assert report["status"] == "offline_synthetic_pending"
    assert report["preview_verdict"] == "PENDING_FIELD_VALIDATION"
    assert report["executable"] is False
    assert report["execution_ready"] is False
    assert report["adapter_consumable"] is False
    assert report["field_observed_only"] is True
    assert report["geometry_binding"]["P_pre_base_m"] == [0.0, 0.0, 0.55]
    assert report["geometry_binding"]["exact_precontact_distance_m"] == 0.050
    assert report["rear_keepout_evidence"]["verdict"] == "SYNTHETIC_CLEAR_PENDING"
    assert report["front_corridor_evidence"]["verdict"] == "OBSERVED_CLEAR_NOT_COMPLETE"
    assert report["request_sha256"] == request["request_sha256"]
    assert report["canonical_report_sha256"] == canonical_report_sha256(report)
    validate_keepout_preview_report(report, request=request)


def test_synthetic_rear_proxy_behind_panel_is_blocked():
    report = build_keepout_preview_report(
        _request(proxy_points=[[0.0, 0.0, 0.490]])
    )
    assert report["preview_verdict"] == "BLOCKED_SYNTHETIC_OR_OBSERVED"
    assert report["rear_keepout_evidence"]["violation_count"] == 1
    assert report["rear_keepout_evidence"]["verdict"] == "SYNTHETIC_VIOLATION"
    assert report["execution_ready"] is False


def test_observed_front_point_inside_fn_to_p_pre_capsule_is_blocked():
    report = build_keepout_preview_report(
        _request(observed_points=[[0.100, 0.0, 0.575]])
    )
    assert report["preview_verdict"] == "BLOCKED_SYNTHETIC_OR_OBSERVED"
    assert report["front_corridor_evidence"]["observed_violation_count"] == 1
    assert report["front_corridor_evidence"]["verdict"] == "OBSERVED_BLOCKED"


def test_no_front_observations_is_explicitly_unobserved_pending():
    report = build_keepout_preview_report(_request(observed_points=[]))
    assert report["preview_verdict"] == "PENDING_FIELD_VALIDATION"
    assert report["front_corridor_evidence"]["verdict"] == "UNOBSERVED_PENDING"
    assert "NO_UNOBSERVED_FIELD_CLEARANCE_CLAIM" in report["limitations"]


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("geometry", "P_pre_base_m"), [0.0, 0.0, 0.551], "P_pre"),
        (("geometry", "N_out_base"), [0.0, 0.0, 2.0], "unit vector"),
        (("front_corridor", "field_observed_only"), False, "field_observed_only"),
        (("rear_keepout", "synthetic_only"), False, "synthetic_only"),
    ],
)
def test_geometry_and_evidence_boundaries_fail_closed(path, value, reason):
    with pytest.raises(KeepoutPreviewBlocked, match=reason):
        build_keepout_preview_report(_replace(_request(), path, value))


def test_request_hash_mismatch_rejected():
    request = _request()
    request["geometry"]["Fn_base_m"][0] = 0.201
    with pytest.raises(KeepoutPreviewBlocked, match="request hash mismatch"):
        build_keepout_preview_report(request)


def test_report_fn_tamper_rejected_even_after_report_rehash():
    request = _request()
    report = build_keepout_preview_report(request)
    report["geometry_binding"]["Fn_base_m"] = [0.201, 0.0, 0.600]
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    with pytest.raises(KeepoutPreviewBlocked, match="Fn does not bind"):
        validate_keepout_preview_report(report, request=request)


def test_report_has_no_execution_payload_keys_or_forbidden_phases():
    report = build_keepout_preview_report(_request())
    forbidden_keys = {
        "joint_positions",
        "joint_trajectory",
        "trajectory",
        "setpoints",
        "commands",
        "action_goal",
        "authorization_id",
        "route",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert forbidden_keys.isdisjoint(set(keys(report)))
    serialized = json.dumps(report)
    assert '"PRESS"' not in serialized
    assert '"RETRACT"' not in serialized
    assert '"RETURN"' not in serialized

    report["commands"] = []
    report["canonical_report_sha256"] = canonical_report_sha256(report)
    with pytest.raises(KeepoutPreviewBlocked, match="unknown or missing fields"):
        validate_keepout_preview_report(report, request=_request())


def test_preview_schema_file_is_valid_json_and_names_both_artifacts():
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "config/press/preview/precontact_keepout_preview_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    assert schema["$defs"]["request"]["properties"]["schema"]["const"] == (
        "precontact_keepout_preview_request_v1"
    )
    assert schema["$defs"]["report"]["properties"]["schema"]["const"] == (
        "precontact_keepout_preview_report_v1"
    )
