import pytest

from press.alignment_guidance import fixed_pose_alignment_guidance


def target():
    return {
        "camera_xyz": (0.0, 0.0, 1.0),
        "normal": (0.0, 0.0, -1.0),
        "depth_valid": True,
        "normal_valid": True,
    }


def test_positive_camera_vertical_says_raise_and_positive_outward_normal_says_advance():
    result = fixed_pose_alignment_guidance(
        (0.0, 0.010, 0.980), target(), p_available=True, p_stable=True, depth_live=True
    )
    assert result["ready"] is True
    assert result["vertical_m"] == pytest.approx(0.010)
    assert result["normal_m"] == pytest.approx(0.020)
    assert result["height"].startswith("RAISE")
    assert result["standoff"].startswith("ADVANCE")
    assert result["horizontal_policy"] == "OBSERVE ONLY (no lateral correction)"


def test_negative_camera_vertical_says_lower_and_too_close_says_retract():
    result = fixed_pose_alignment_guidance(
        (0.0, -0.010, 1.020), target(), p_available=True, p_stable=True, depth_live=True
    )
    assert result["height"].startswith("LOWER")
    assert result["standoff"].startswith("RETRACT")


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"p_available": False, "p_stable": False, "depth_live": False}, "P unavailable"),
        ({"p_available": True, "p_stable": True, "depth_live": False}, "P/depth stale"),
        ({"p_available": True, "p_stable": False, "depth_live": True}, "P not stable"),
    ],
)
def test_unavailable_or_stale_p_never_suggests_a_correction(kwargs, reason):
    result = fixed_pose_alignment_guidance((0.0, 0.0, 1.0), target(), **kwargs)
    assert result == {"ready": False, "reason": reason, "height": "WAIT", "standoff": "WAIT"}
