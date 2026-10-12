"""Tests for the T102 multi-joint press route generator."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.press.m2_multi_joint_route import (
    REPO_ROOT,
    RouteError,
    _interpolate_waypoints,
    _required_waypoints,
    build_route,
)
from tools.press.m1_no_contact_route import (
    JOINT_NAMES,
)


STANDBY = (
    REPO_ROOT
    / "config"
    / "press"
    / "poses"
    / "press_standby_manual_20260804_preflight.yaml"
)
PRE_CONTACT = REPO_ROOT / "config" / "press" / "poses" / "pre_contact_candidate_v1.yaml"


class TestWaypointInterpolation:
    def test_single_waypoint_returns_endpoint(self):
        start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        end = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        wps = _interpolate_waypoints(start, end, 1)
        assert len(wps) == 1
        assert wps[0] == pytest.approx(end)

    def test_three_waypoints_evenly_spaced(self):
        start = (0.0,) * 6
        end = (1.0,) * 6
        wps = _interpolate_waypoints(start, end, 3)
        assert len(wps) == 3
        assert wps[0] == pytest.approx((1 / 3,) * 6)
        assert wps[1] == pytest.approx((2 / 3,) * 6)
        assert wps[2] == pytest.approx((1.0,) * 6)

    def test_zero_count_raises(self):
        with pytest.raises(RouteError):
            _interpolate_waypoints((0.0,) * 6, (1.0,) * 6, 0)


class TestRequiredWaypoints:
    def test_no_movement_returns_one(self):
        pos = (0.0,) * 6
        assert _required_waypoints(pos, pos) == 1

    def test_small_movement_within_limits(self):
        start = (0.0,) * 6
        end = list(start)
        end[0] = 0.05  # base delta well within the 5° profile limit
        assert _required_waypoints(start, tuple(end)) == 1

    def test_large_shoulder_delta_dominates(self):
        """shoulder (39.7° at the 2° limit minus encoder headroom) dominates."""
        from tools.press.m1_no_contact_route import load_recorded_pose

        standby = load_recorded_pose(STANDBY, "press_standby")
        pre = load_recorded_pose(PRE_CONTACT, "pre_contact")
        count = _required_waypoints(standby["positions"], pre["positions"])
        # ceil(39.72 / (2 deg - 0.0879 deg encoder step)) = 21
        assert count == 21


class TestBuildRoute:
    def test_artifact_generated_with_valid_schema(self):
        artifact = build_route(STANDBY, PRE_CONTACT)
        assert artifact["schema"] == "m2_multi_joint_route_v1"
        assert artifact["status"] == "offline_validated_candidate"

    def test_all_phases_present(self):
        artifact = build_route(STANDBY, PRE_CONTACT)
        totals = artifact["totals"]
        assert totals["outbound_count"] >= 1
        assert totals["press_count"] == 1
        assert totals["retract_count"] == 1
        assert totals["return_count"] >= 1

    def test_fake_driver_validation_passes(self):
        artifact = build_route(STANDBY, PRE_CONTACT)
        sim = artifact["fake_driver_validation"]
        assert sim["passed"]
        assert sim["t102_only"]
        assert sim["t101_absent"]
        assert sim["waypoint_count"] == artifact["totals"]["waypoint_count"]

    def test_all_execution_flags_false(self):
        artifact = build_route(STANDBY, PRE_CONTACT)
        for key in ("execution_ready", "motion_authorized",
                     "contact_authorized", "press_authorized"):
            assert artifact[key] is False

    def test_endpoints_reach_targets(self):
        artifact = build_route(STANDBY, PRE_CONTACT)
        endpoints = artifact["endpoints"]
        # Return should land exactly at press_standby
        ret = endpoints["return_target_rad"]
        start = endpoints["press_standby_start_rad"]
        for i in range(5):
            assert ret[i] == pytest.approx(start[i])

    def test_gripper_held_throughout(self):
        artifact = build_route(STANDBY, PRE_CONTACT)
        gripper = artifact["endpoints"]["press_standby_start_rad"][5]
        for wp in artifact["waypoints"]:
            assert wp["positions_rad"][5] == pytest.approx(gripper)

    def test_press_step_moves_tcp_forward(self):
        artifact = build_route(STANDBY, PRE_CONTACT, press_distance_mm=5.0)
        geo = artifact["geometry_audit"]
        press_disp = geo["tcp_displacement_mm"]["press"]
        # Press should move TCP by approximately 5 mm
        assert 3.0 < press_disp < 8.0

    def test_press_ik_converges(self):
        artifact = build_route(STANDBY, PRE_CONTACT, press_distance_mm=5.0)
        ik = artifact["press_step"]["ik_audit"]
        assert ik["iterations"] <= 20
        assert ik["final_error_mm"] is not None
        assert ik["final_error_mm"] < 1.0  # sub-mm precision

    def test_custom_press_distance(self):
        artifact = build_route(STANDBY, PRE_CONTACT, press_distance_mm=10.0)
        assert artifact["press_step"]["press_distance_mm"] == 10.0
        assert artifact["fake_driver_validation"]["passed"]

    def test_waypoint_count_reduces_from_t101(self):
        """T102 waypoint count should be much less than 102 T101 steps."""
        artifact = build_route(STANDBY, PRE_CONTACT)
        total = artifact["totals"]["waypoint_count"]
        # Under the deployed AGX profile limits the wrist needs 30 outbound
        # and 30 return waypoints; all task joints still move simultaneously,
        # so the cycle stays well below the 102 sequential T101 steps.
        assert total < 70

    def test_waypoint_steps_respect_deployed_profile_limits(self):
        """Every waypoint-to-waypoint delta must fit the AGX profile."""
        from roarm_safe_driver.core import DEFAULT_JOINT_MAX_DELTA_RAD

        artifact = build_route(STANDBY, PRE_CONTACT)
        previous = artifact["endpoints"]["press_standby_start_rad"]
        for wp in artifact["waypoints"]:
            for index in range(6):
                delta = abs(wp["positions_rad"][index] - previous[index])
                assert delta <= DEFAULT_JOINT_MAX_DELTA_RAD[index] + 1e-9
            previous = wp["positions_rad"]

    def test_canonical_sha256_stable(self):
        a1 = build_route(STANDBY, PRE_CONTACT, press_distance_mm=5.0)
        a2 = build_route(STANDBY, PRE_CONTACT, press_distance_mm=5.0)
        assert a1["canonical_payload_sha256"] == a2["canonical_payload_sha256"]
