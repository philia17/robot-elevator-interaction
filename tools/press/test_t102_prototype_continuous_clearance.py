"""Offline regression tests for the narrow T102 prototype clearance checker."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / "tools/press/t102_prototype_continuous_clearance.py"
PROFILE_PATH = ROOT / "config/press/prototype/t102_prototype_geometry_profile_20260814_v1.json"
ROUTE_PATH = ROOT / "config/press/prototype/t102_prototype_route_candidate_20260814_v1.json"
RESULT_PATH = ROOT / "config/press/prototype/t102_prototype_clearance_result_20260814_v1.json"


spec = importlib.util.spec_from_file_location("t102_prototype_clearance", CHECKER_PATH)
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _route() -> dict:
    return json.loads(ROUTE_PATH.read_text(encoding="utf-8"))


def test_historical_candidate_is_five_intervals_and_result_is_provisional() -> None:
    profile, _ = checker.load_profile(PROFILE_PATH)
    route, _ = checker.load_route(ROUTE_PATH)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    checker.validate_result(result)
    assert len(route["setpoints_rad"]) == 5
    assert result["route"]["interval_count"] == 5
    assert result["route"]["route_sha256"] == route["route_sha256"]
    # The 2026-08-14 receipt is historical evidence.  Its profile hash is
    # intentionally checked against its own sealed contract, not the current
    # checker profile, which later gained an explicit excluded pair.
    assert result["created_at"] == "2026-08-14"
    assert result["profile"]["canonical_profile_sha256"] == (
        "031d26090e67e75790ccf1ef832ea006934199ddf43459e7a253fff467eb84a5"
    )
    assert result["profile"]["canonical_profile_sha256"] != profile[
        "canonical_profile_sha256"
    ]
    assert result["result"] == "UNPROVEN_OR_COLLISION_RISK"
    assert result["provisional"] is True
    assert result["panel_provisional"] is True
    assert all(value is False for value in result["authority"].values())


def test_continuous_mid_segment_collision_is_not_endpoint_sampling() -> None:
    start = np.array([-1.0, 0.0, 0.0])
    end = np.array([1.0, 0.0, 0.0])
    obstacle_center = np.zeros(3)
    obstacle_half = np.array([0.05, 0.05, 0.05])
    bound = checker._segment_aabb_lower_bound(
        start, end, 0.0, obstacle_center, obstacle_half
    )
    assert bound < 0.0


def test_safe_synthetic_interval_has_positive_lower_bound() -> None:
    bound = checker._segment_aabb_lower_bound(
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        0.01,
        np.zeros(3),
        np.array([0.05, 0.05, 0.05]),
    )
    assert bound > checker.MIN_SELF_NON_OVERLAP_TOLERANCE_M


@pytest.mark.parametrize(
    ("path", "nested", "field", "value"),
    [
        ("profile", "unified_tool_envelope", "radius_m", 0.029),
        ("profile", "wrist_cable_envelope", "radius_m", 0.119),
        ("profile", "rear_camera_bracket", "size_m", [0.139, 0.430, 0.140]),
        ("profile", "clearance_requirements", "fixed_obstacle_clearance_margin_m", 0.019),
    ],
)
def test_profile_never_accepts_shrunk_prototype_floors(path, nested, field, value) -> None:
    assert path == "profile"
    profile = _profile()
    profile[nested][field] = value
    with pytest.raises(checker.PrototypeClearanceBlocked):
        checker.validate_profile(profile)


def test_omitted_body_is_rejected() -> None:
    profile = _profile()
    profile["robot_model"]["body_ids"] = profile["robot_model"]["body_ids"][:-1]
    with pytest.raises(checker.PrototypeClearanceBlocked):
        checker.validate_profile(profile)


def test_current_explicit_exclusion_policy_is_separate_from_historical_receipt() -> None:
    profile = _profile()
    assert len(checker._non_adjacent_pairs()) == 24
    assert ("base_link", "link2") in checker._non_adjacent_pairs()
    assert sorted(map(sorted, profile["excluded_adjacent_pairs"])) == sorted(
        map(sorted, checker.ADJACENT_PAIRS)
    )

    historical = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert historical["coverage"]["non_adjacent_pair_count"] == 25
    assert len(historical["coverage"]["excluded_adjacent_pairs"]) == 11
    assert historical["excluded_pair_overlap_verification"][
        "all_excluded_pairs_overlap_by_construction"
    ] is True


def test_route_hash_drift_is_rejected(tmp_path: Path) -> None:
    route = _route()
    route["route_sha256"] = "0" * 64
    path = tmp_path / "route.json"
    path.write_text(json.dumps(route), encoding="utf-8")
    with pytest.raises(checker.PrototypeClearanceBlocked):
        checker.load_route(path)


def test_unproven_cannot_be_promoted_by_dropping_body_or_shrinking_envelope() -> None:
    profile = _profile()
    profile["robot_model"]["body_ids"] = profile["robot_model"]["body_ids"][:-1]
    with pytest.raises(checker.PrototypeClearanceBlocked):
        checker.validate_profile(profile)

    profile = _profile()
    profile["unified_tool_envelope"]["radius_m"] = 0.01
    with pytest.raises(checker.PrototypeClearanceBlocked):
        checker.validate_profile(profile)


def test_model_profile_hashes_and_route_authority_are_bound() -> None:
    profile = _profile()
    route = _route()
    assert profile["robot_model"]["collision_mesh_hashes"]
    assert profile["robot_model"]["urdf_or_xacro_sha256"]
    assert route["route_mode"] == "OFFLINE_DIAGNOSTIC_ONLY"
    assert route["command_interface"] == "NOT_IMPLEMENTED"
    for field in (
        "execution_ready",
        "motion_authorized",
        "press",
        "retract",
        "return",
        "automatic_retry",
        "command_generated",
        "trajectory_generated",
        "adapter_consumable",
    ):
        assert route[field] is False


def test_result_has_all_interval_evidence_and_no_authority() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert len(result["segments"]) == 5
    assert all(
        segment["method"]
        == "adaptive_interval_fk_conservative_primitive_bounds_with_recursive_subdivision"
        for segment in result["segments"]
    )
    assert all(segment["status"] == "UNPROVEN" for segment in result["segments"])
    # This is the historical receipt's own body/pair contract.  The current
    # checker policy is asserted separately above and intentionally has 24
    # non-adjacent pairs.
    assert result["coverage"]["non_adjacent_pair_count"] == 25
    assert result["excluded_pair_overlap_verification"][
        "all_excluded_pairs_overlap_by_construction"
    ] is True
    assert all(value is False for value in result["authority"].values())
