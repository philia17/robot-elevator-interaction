from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from press.wrist_local_tangent_core import (
    WristTangentBlocked,
    WristTangentCalibration,
    build_calibration_candidate,
    solve_wrist_tangent_correction,
)


def calibration(*, accepted=True):
    return build_calibration_candidate(
        wrist_lock_identity_suffix=("wrist-source", "wrist-cal", "tip", "model", "640x480", "rotate=0"),
        tool_apex_px=(320.0, 240.0),
        tangent_u_base=(0.0, 1.0, 0.0),
        tangent_v_base=(0.0, 0.0, 1.0),
        reference_normal_out_base=(1.0, 0.0, 0.0),
        q0_button_centers_px=((320, 240), (320, 240), (320, 240)),
        plus_u_button_centers_px=((300, 240), (300, 240), (300, 240)),
        plus_v_button_centers_px=((320, 260), (320, 260), (320, 260)),
        probe_u_m=0.010,
        probe_v_m=0.010,
        accepted_for_press=accepted,
    )


def identity():
    return ("panel", "acquire", "wrist-source", "wrist-cal", "tip", "model", "640x480", "rotate=0")


def test_maps_image_error_to_expected_tangent_and_zero_normal():
    parsed = WristTangentCalibration.from_mapping(calibration())
    correction = solve_wrist_tangent_correction(
        parsed,
        wrist_lock_identity=identity(),
        observed_button_center_px=(310.0, 250.0),
        normal_out_base=(1.0, 0.0, 0.0),
    )
    assert correction.tangent_delta_m == pytest.approx((-0.005, -0.005))
    assert correction.delta_base_m == pytest.approx((0.0, -0.005, -0.005))
    assert sum(a * b for a, b in zip(correction.delta_base_m, (1, 0, 0))) == pytest.approx(0.0)
    assert correction.capped is False


def test_explicit_contact_reference_drives_signed_tangent_correction():
    correction = solve_wrist_tangent_correction(
        calibration(),
        wrist_lock_identity=identity(),
        observed_button_center_px=(310.0, 250.0),
        contact_reference_px=(300.0, 260.0),
        normal_out_base=(1.0, 0.0, 0.0),
    )
    assert correction.image_error_px == pytest.approx((10.0, -10.0))
    assert correction.tangent_delta_m == pytest.approx((0.005, 0.005))
    assert correction.delta_base_m == pytest.approx((0.0, 0.005, 0.005))


def test_checked_in_analytic_calibration_loads_with_exact_identity_and_cap():
    path = (
        Path(__file__).resolve().parents[2]
        / "config/press/wrist_local_tangent_calibration.json"
    )
    parsed = WristTangentCalibration.from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert parsed.calibration_sha256 == (
        "ca13b72bd3077b4fed25c1d562d51bb63e7e428fa8aa4e8e265d895cbd25bd77"
    )
    assert parsed.wrist_lock_identity_suffix == (
        "/dev/v4l/by-id/usb-lihappe8_Corp._USB_2.0_Camera-video-index0",
        "70902e11e849",
        "580e21d842c831787bc138dcabe5453b34d571fd2b1e852c4be319aaa4762e44",
        "5babcb0d9b627fcfe789cb7bff3e1e04db987fe251e9c272b2585156b31d73f4",
        "640x480",
        "rotate=0",
    )
    assert parsed.maximum_condition_number == pytest.approx(10.0)
    assert parsed.maximum_correction_m == pytest.approx(0.035)


def test_press_acquisition_prefix_preserves_checked_in_calibration_identity():
    path = (
        Path(__file__).resolve().parents[2]
        / "config/press/wrist_local_tangent_calibration.json"
    )
    parsed = WristTangentCalibration.from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )
    completed_pre = ("cart", "session", "plan", "target", "cal", "executor")
    acquisition_id = "press-acquisition-current"
    old_appended_identity = (
        *completed_pre,
        *parsed.wrist_lock_identity_suffix,
        acquisition_id,
    )
    with pytest.raises(WristTangentBlocked, match="WRIST_IDENTITY_MISMATCH"):
        solve_wrist_tangent_correction(
            parsed,
            wrist_lock_identity=old_appended_identity,
            observed_button_center_px=parsed.tool_apex_px,
            normal_out_base=parsed.reference_normal_out_base,
        )

    press_identity = (
        acquisition_id,
        *completed_pre,
        *parsed.wrist_lock_identity_suffix,
    )
    correction = solve_wrist_tangent_correction(
        parsed,
        wrist_lock_identity=press_identity,
        observed_button_center_px=parsed.tool_apex_px,
        normal_out_base=parsed.reference_normal_out_base,
    )
    assert correction.applied_norm_m == pytest.approx(0.0)
    assert press_identity[-len(parsed.wrist_lock_identity_suffix):] == (
        parsed.wrist_lock_identity_suffix
    )

    changed_identity = (*press_identity[:-1], "rotate=changed")
    with pytest.raises(WristTangentBlocked, match="WRIST_IDENTITY_MISMATCH"):
        solve_wrist_tangent_correction(
            parsed,
            wrist_lock_identity=changed_identity,
            observed_button_center_px=parsed.tool_apex_px,
            normal_out_base=parsed.reference_normal_out_base,
        )


def test_demonstrated_thirty_one_mm_request_is_not_capped():
    requested_m = 0.0309382
    correction = solve_wrist_tangent_correction(
        calibration(),
        wrist_lock_identity=identity(),
        observed_button_center_px=(320.0 + 2000.0 * requested_m, 240.0),
        normal_out_base=(1.0, 0.0, 0.0),
    )
    assert correction.requested_norm_m == pytest.approx(requested_m)
    assert correction.applied_norm_m == pytest.approx(requested_m)
    assert correction.capped is False


def test_correction_over_thirty_five_mm_is_capped_at_limit():
    correction = solve_wrist_tangent_correction(
        calibration(),
        wrist_lock_identity=identity(),
        observed_button_center_px=(400.0, 240.0),
        normal_out_base=(1.0, 0.0, 0.0),
    )
    assert correction.requested_norm_m == pytest.approx(0.040)
    assert correction.applied_norm_m == pytest.approx(0.035)
    assert correction.capped is True


def test_small_fresh_normal_tilt_reprojects_tangent_frame():
    tilt = math.radians(3.0)
    normal = (math.cos(tilt), math.sin(tilt), 0.0)
    correction = solve_wrist_tangent_correction(
        calibration(),
        wrist_lock_identity=identity(),
        observed_button_center_px=(310.0, 250.0),
        normal_out_base=normal,
    )
    assert sum(
        value * axis for value, axis in zip(correction.delta_base_m, normal)
    ) == pytest.approx(0.0, abs=1e-12)


def test_fresh_normal_tilt_over_five_degrees_still_blocks():
    tilt = math.radians(6.0)
    with pytest.raises(WristTangentBlocked, match="PANEL_NORMAL_MISMATCH"):
        solve_wrist_tangent_correction(
            calibration(),
            wrist_lock_identity=identity(),
            observed_button_center_px=(310.0, 250.0),
            normal_out_base=(math.cos(tilt), math.sin(tilt), 0.0),
        )


@pytest.mark.parametrize("normal", [
    (-.999184717061947, .03254935459536937, -.023883900545392356),
    (-math.cos(math.radians(15)), math.sin(math.radians(15)), 0.),
    (-math.cos(math.radians(30)), math.sin(math.radians(30)), 0.),
])
@pytest.mark.parametrize("offset_m", [0., .005])
def test_current_plane_solver_does_not_apply_reference_jacobian_angle_gate(normal, offset_m):
    pytest.importorskip("cv2")
    from press.wrist_local_tangent_core import solve_projected_target_correction

    path = Path(__file__).resolve().parents[2] / "config/press/wrist_local_tangent_calibration.json"
    parsed = WristTangentCalibration.from_mapping(json.loads(path.read_text()))
    device = ("current-press", *parsed.wrist_lock_identity_suffix)
    # First normal is the actual 2026-08-28 13:59 rejection (5.762 degrees).
    # Synthetic projection makes the expected physical correction independent
    # of a measured image, while exercising the deployed calibration unchanged.
    context = {
        "camera_from_base": [[0., 1., 0., -.01], [0., 0., 1., 0.],
                             [1., 0., 0., .3], [0., 0., 0., 1.]],
        "camera_matrix": [[500., 0., 320.], [0., 510., 240.], [0., 0., 1.]],
        "dist_coeffs": [0.] * 5,
    }
    delta = (-normal[1] * offset_m / normal[0], offset_m, 0.)
    observed = (320. + 500. * offset_m / (.33 + delta[0]), 240.)
    correction = solve_projected_target_correction(
        parsed, wrist_lock_identity=device, observed_button_center_px=observed,
        projected_target_px=(320., 240.), p_button_base_m=(.03, .01, 0.),
        normal_out_base=normal, projection_context=context, suppress_upward_correction=True,
    )
    assert correction.delta_base_m == pytest.approx(delta, abs=1e-8)
    assert abs(sum(x * y for x, y in zip(normal, correction.delta_base_m))) < 1e-9
    assert correction.applied_norm_m <= .035
    assert correction.suppress_upward_correction
    # The historical fixed-Jacobian solver still requires its reference pose.
    with pytest.raises(WristTangentBlocked, match="PANEL_NORMAL_MISMATCH"):
        solve_wrist_tangent_correction(
            parsed, wrist_lock_identity=device, observed_button_center_px=observed,
            normal_out_base=normal,
        )


def test_identity_mismatch_blocks():
    with pytest.raises(WristTangentBlocked, match="WRIST_IDENTITY_MISMATCH"):
        solve_wrist_tangent_correction(
            calibration(),
            wrist_lock_identity=("foreign",),
            observed_button_center_px=(320.0, 240.0),
            normal_out_base=(1.0, 0.0, 0.0),
        )


def test_unaccepted_candidate_does_not_enter_press_path():
    with pytest.raises(WristTangentBlocked, match="CALIBRATION_NOT_ACCEPTED"):
        WristTangentCalibration.from_mapping(calibration(accepted=False))


def test_singular_probe_blocks():
    raw = build_calibration_candidate(
        wrist_lock_identity_suffix=("identity",),
        tool_apex_px=(0, 0),
        tangent_u_base=(0, 1, 0),
        tangent_v_base=(0, 0, 1),
        reference_normal_out_base=(1, 0, 0),
        q0_button_centers_px=((0, 0),) * 3,
        plus_u_button_centers_px=((10, 0),) * 3,
        plus_v_button_centers_px=((20, 0),) * 3,
        probe_u_m=0.01,
        probe_v_m=0.01,
        accepted_for_press=True,
    )
    with pytest.raises(WristTangentBlocked, match="CALIBRATION_SINGULAR"):
        WristTangentCalibration.from_mapping(raw)
