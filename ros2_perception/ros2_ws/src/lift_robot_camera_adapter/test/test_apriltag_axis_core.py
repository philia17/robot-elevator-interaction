import pytest

from lift_robot_camera_adapter.apriltag_axis_core import evaluate_axis_captures


def _capture(label, u, v, z):
    samples = []
    for index in range(30):
        samples.append(
            {
                "stamp_ns": 1_000_000_000 + index * 100_000_000,
                "camera_frame": "camera_optical_frame",
                "tag_frame": "calibration_tag",
                "translation_m": [u * z / 100000.0, v * z / 100000.0, z],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "centre_px": [u, v],
                "corners_px": [
                    [u - 40.0, v - 40.0],
                    [u + 40.0, v - 40.0],
                    [u + 40.0, v + 40.0],
                    [u - 40.0, v + 40.0],
                ],
                "decision_margin": 100.0,
                "hamming": 0,
            }
        )
    return {
        "schema_version": 2,
        "capture_status": "complete",
        "label": label,
        "target": {"family": "tag36h11", "id": 0, "edge_m": 0.064},
        "samples": samples,
    }


def _passing_documents():
    return {
        "center": _capture("center", 900.0, 500.0, 0.40),
        "right": _capture("right", 1020.0, 500.0, 0.40),
        "down_baseline": _capture("down_baseline", 900.0, 500.0, 0.40),
        "down": _capture("down", 900.0, 620.0, 0.40),
        "forward_baseline": _capture("forward_baseline", 900.0, 500.0, 0.40),
        "forward": _capture("forward", 950.0, 450.0, 0.36),
    }


def test_axis_evaluation_passes_all_three_signs():
    result = evaluate_axis_captures(_passing_documents())
    assert result["status"] == "pass"
    assert all(check["pass"] for check in result["checks"].values())


def test_axis_evaluation_rejects_wrong_forward_sign():
    documents = _passing_documents()
    documents["forward"] = _capture("forward", 950.0, 450.0, 0.42)
    result = evaluate_axis_captures(documents)
    assert result["status"] == "fail"
    assert result["checks"]["toward_camera_z_decreases"]["pass"] is False


def test_axis_evaluation_rejects_low_margin_sample():
    documents = _passing_documents()
    documents["center"]["samples"][0]["decision_margin"] = 20.0
    with pytest.raises(ValueError, match="margin"):
        evaluate_axis_captures(documents)


def test_axis_evaluation_rejects_noncanonical_frame():
    documents = _passing_documents()
    documents["center"]["samples"][0]["camera_frame"] = "vendor_frame"
    with pytest.raises(ValueError, match="canonical"):
        evaluate_axis_captures(documents)


def test_axis_evaluation_requires_every_capture():
    documents = _passing_documents()
    del documents["forward"]
    with pytest.raises(ValueError, match="missing"):
        evaluate_axis_captures(documents)
