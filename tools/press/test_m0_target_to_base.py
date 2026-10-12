"""Unit tests for M0 frozen-target → roarm_base transform."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.press.m0_target_to_base import (
    DEFAULT_BUNDLE,
    EXPECTED_CALIBRATION_ID,
    TransformError,
    apply_transform,
    load_relative_bundle,
    quat_xyzw_to_rotation,
    rotate,
    transform_frozen_target,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample_frozen(**overrides):
    target = {
        "target_label": "1",
        "frame_id": "camera_optical_frame",
        "camera_point_m": {"x": 0.1, "y": 0.0, "z": 0.5},
        "surface_geometry": {
            "estimate_valid": True,
            "provenance_complete": False,
            "panel_outward_normal": {"x": 0.0, "y": 0.0, "z": -1.0},
        },
    }
    target.update(overrides)
    return target


def test_bundle_binds_expected_calibration_id():
    bundle = load_relative_bundle(DEFAULT_BUNDLE)
    assert bundle["calibration_id"] == EXPECTED_CALIBRATION_ID
    assert len(bundle["translation_m"]) == 3
    assert len(bundle["rotation_xyzw"]) == 4


def test_wrong_calibration_id_rejected(tmp_path):
    raw = json.loads(DEFAULT_BUNDLE.read_text())
    raw["calibration_id"] = "wrong_id"
    path = tmp_path / "bad_bundle.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(TransformError, match="calibration_id mismatch"):
        load_relative_bundle(path)


def test_execution_ready_bundle_rejected(tmp_path):
    raw = json.loads(DEFAULT_BUNDLE.read_text())
    raw["execution_ready"] = True
    path = tmp_path / "exec_ready.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(TransformError, match="execution_ready"):
        load_relative_bundle(path)


def test_transform_identity_math():
    # Identity quaternion XYZW = (0,0,0,1)
    point = apply_transform((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.1, 0.2, 0.3))
    assert point == pytest.approx((1.1, 2.2, 3.3))


def test_diagnostic_incomplete_provenance_allowed():
    artifact = transform_frozen_target(_sample_frozen())
    assert artifact["quality"] == "diagnostic"
    assert artifact["provenance_complete"] is False
    assert artifact["calibration_id"] == EXPECTED_CALIBRATION_ID
    assert artifact["execution_ready"] is False
    assert artifact["tf_static_published"] is False
    normal = artifact["panel_outward_normal_roarm_base"]
    assert math.isclose(math.sqrt(sum(v * v for v in normal)), 1.0, abs_tol=1e-9)


def test_require_complete_provenance_rejects():
    with pytest.raises(TransformError, match="provenance_complete"):
        transform_frozen_target(
            _sample_frozen(), allow_incomplete_provenance=False
        )


def test_missing_normal_rejected():
    bad = _sample_frozen()
    del bad["surface_geometry"]["panel_outward_normal"]
    with pytest.raises(TransformError, match="panel_outward_normal"):
        transform_frozen_target(bad)


def test_round_trip_matches_bundle_forward():
    bundle = load_relative_bundle(DEFAULT_BUNDLE)
    p_cam = (0.05, -0.02, 0.4)
    n_cam = (0.0, 0.0, -1.0)
    expected_p = apply_transform(
        bundle["translation_m"], bundle["rotation_xyzw"], p_cam
    )
    matrix = quat_xyzw_to_rotation(bundle["rotation_xyzw"])
    expected_n = rotate(matrix, n_cam)
    nn = math.sqrt(sum(v * v for v in expected_n))
    expected_n = tuple(v / nn for v in expected_n)
    artifact = transform_frozen_target(
        {
            "target_label": "1",
            "camera_point_m": {"x": p_cam[0], "y": p_cam[1], "z": p_cam[2]},
            "surface_geometry": {
                "estimate_valid": True,
                "provenance_complete": True,
                "panel_outward_normal": {
                    "x": n_cam[0],
                    "y": n_cam[1],
                    "z": n_cam[2],
                },
            },
        }
    )
    assert artifact["quality"] == "accepted"
    assert artifact["point_roarm_base_m"] == pytest.approx(expected_p)
    assert artifact["panel_outward_normal_roarm_base"] == pytest.approx(expected_n)
