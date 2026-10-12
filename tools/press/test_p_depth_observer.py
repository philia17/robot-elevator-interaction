#!/usr/bin/env python3

import unittest

import numpy as np

from tools.press.p_depth_observer import (
    observe_fingertip_from_depth,
    temporal_stability,
)


class PDepthObserverTests(unittest.TestCase):
    def setUp(self):
        self.fx = self.fy = 200.0
        self.cx = self.cy = 100.0
        self.panel_point = (0.0, 0.0, 1.0)
        self.normal = (0.0, 0.0, -1.0)
        self.predicted = (0.0, 0.0, 0.90)

    def observe(self, depth):
        return observe_fingertip_from_depth(
            depth,
            "32FC1",
            panel_point_camera_m=self.panel_point,
            panel_outward_normal_camera=self.normal,
            predicted_tip_camera_m=self.predicted,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            roi_radius_px=40,
            minimum_component_pixels=10,
            maximum_anchor_distance_px=20.0,
        )

    def test_observes_panel_front_component_not_model_point(self):
        depth = np.full((200, 200), 1.0, dtype=np.float32)
        depth[94:107, 96:105] = 0.92
        result = self.observe(depth)
        self.assertTrue(result.valid, result.failure_reason)
        self.assertAlmostEqual(result.point_camera_m[2], 0.92, places=5)
        self.assertAlmostEqual(result.signed_gap_m, 0.08, places=5)

    def test_rejects_background_only(self):
        result = self.observe(np.full((200, 200), 1.0, dtype=np.float32))
        self.assertFalse(result.valid)
        self.assertIn("no panel-front", result.failure_reason)

    def test_rejects_component_far_from_prediction(self):
        depth = np.full((200, 200), 1.0, dtype=np.float32)
        depth[125:140, 125:140] = 0.92
        result = self.observe(depth)
        self.assertFalse(result.valid)

    def test_depth_holes_do_not_become_foreground(self):
        depth = np.full((200, 200), 1.0, dtype=np.float32)
        depth[90:110, 90:110] = np.nan
        result = self.observe(depth)
        self.assertFalse(result.valid)

    def test_temporal_gate_requires_five_tight_frames(self):
        points = [(0.0, 0.0, 0.9)] * 4
        stable, _center, spread = temporal_stability(points)
        self.assertFalse(stable)
        self.assertEqual(spread, 0.0)
        points.append((0.001, 0.0, 0.9))
        stable, center, spread = temporal_stability(points)
        self.assertTrue(stable)
        self.assertTrue(np.all(np.isfinite(center)))
        self.assertLessEqual(spread, 0.003)


if __name__ == "__main__":
    unittest.main()
