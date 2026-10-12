"""Departure-pose projection correction and absolute PRESS endpoint regressions."""
from copy import deepcopy
import ast
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from press.test_wrist_local_tangent_core import calibration, identity
from press.test_stage_b_rehearsal_core import TangentModel, make_wrist_corrected_receipt
from press.wrist_local_tangent_core import (
    WristTangentBlocked, WristTangentCalibration, solve_projected_target_correction,
)
from press.stage_b_rehearsal_core import (
    PRESS_OVERTRAVEL_M, PreTerminalReceipt, canonical_hash, canonical_json,
    prepare_stage_b_plan, verify_stage_b_plan,
)
from press.display_projection import project_camera_point


def projection_fixture(distance=0.3, roll=0.0, tilt=0.0):
    rotation, _ = cv2.Rodrigues(np.array([tilt, 0.0, roll]))
    rotation = rotation @ np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ np.array([-distance, 0.0, 0.0])
    context = {
        "camera_from_base": transform.tolist(),
        "camera_matrix": [[500., 0., 320.], [0., 510., 240.], [0., 0., 1.]],
        "dist_coeffs": [-0.12, 0.01, 0.001, -0.002, 0.0],
        "feedback_positions_rad": [0.0] * 6,
    }

    def project(point):
        camera = rotation @ np.array(point) + transform[:3, 3]
        result, _ = cv2.projectPoints(camera.reshape(1, 3), np.zeros(3), np.zeros(3),
                                      np.array(context["camera_matrix"]),
                                      np.array(context["dist_coeffs"]))
        return result.reshape(2).tolist()

    return context, project


def solve(context, projected, observed, button=(0.03, 0.01, 0.0), **extra):
    args = dict(wrist_lock_identity=identity(), observed_button_center_px=observed,
                projected_target_px=projected, p_button_base_m=button,
                normal_out_base=(1.0, 0.0, 0.0), projection_context=context)
    args.update(extra)
    return solve_projected_target_correction(WristTangentCalibration.from_mapping(calibration()), **args)


@pytest.mark.parametrize("distance", [0.15, 0.30, 0.50])
@pytest.mark.parametrize("roll,tilt", [(0., 0.), (0.4, 0.1), (-0.25, -0.15)])
def test_current_pose_changes_pixel_scale_but_not_corrected_physical_target(distance, roll, tilt):
    context, project = projection_fixture(distance, roll, tilt)
    button = (0.03, 0.01, 0.0)
    visual = (0.03, 0.015, -0.008)
    correction = solve(context, project(button), project(visual))
    assert correction.delta_base_m == pytest.approx((0., 0.005, -0.008), abs=1e-8)
    assert correction.capped is False


def test_vis_equal_projected_target_needs_zero_correction_not_touch_offset():
    context, project = projection_fixture()
    pixel = project((0.03, 0.01, 0.0))
    correction = solve(context, pixel, pixel)
    assert correction.delta_base_m == pytest.approx((0., 0., 0.), abs=1e-10)


@pytest.mark.parametrize("yaw_deg", [6., 15., 30.])
def test_current_plane_not_reference_jacobian_controls_cart_yaw(yaw_deg):
    context, project = projection_fixture()
    yaw = math.radians(yaw_deg)
    normal = np.array([math.cos(yaw), math.sin(yaw), 0.])
    button = np.array([.03, .01, 0.])
    delta = np.array([-math.sin(yaw) * .005, math.cos(yaw) * .005, -.008])
    correction = solve(context, project(button), project(button + delta),
                       normal_out_base=normal.tolist(), suppress_upward_correction=True)
    assert correction.delta_base_m == pytest.approx(delta, abs=1e-8)
    assert abs(float(normal @ correction.delta_base_m)) < 1e-9
    assert not correction.capped


@pytest.mark.parametrize("offset_m", [0., .005])
def test_aug28_rejected_panel_normal_with_existing_calibration(offset_m):
    path = Path(__file__).resolve().parents[2] / "config/press/wrist_local_tangent_calibration.json"
    parsed = WristTangentCalibration.from_mapping(json.loads(path.read_text()))
    # Actual 13:59 failure: 5.762 degrees from the unchanged calibration.
    # Camera/pixels here are synthetic so the expected correction is exact.
    normal = np.array([-.999184717061947, .03254935459536937, -.023883900545392356])
    angle = math.degrees(math.acos(float(normal @ parsed.reference_normal_out_base)))
    assert angle == pytest.approx(5.76205614709667)
    context, project = projection_fixture()
    button = np.array([.03, .01, 0.])
    delta = offset_m * np.array([-normal[1] / normal[0], 1., 0.])
    correction = solve_projected_target_correction(
        parsed, wrist_lock_identity=("current-click", *parsed.wrist_lock_identity_suffix),
        observed_button_center_px=project(button + delta), projected_target_px=project(button),
        p_button_base_m=button.tolist(), normal_out_base=normal.tolist(), projection_context=context,
        suppress_upward_correction=True,
    )
    assert correction.delta_base_m == pytest.approx(delta, abs=1e-8)
    assert correction.calibration_sha256 == parsed.calibration_sha256
    assert correction.applied_norm_m <= .035


def test_current_plane_still_rejects_parallel_ray_and_target_behind_camera():
    context, project = projection_fixture()
    button = (.03, 0., 0.)
    with pytest.raises(WristTangentBlocked, match="PROJECTION_GEOMETRY_INVALID"):
        solve(context, project(button), project(button), button=button,
              normal_out_base=(0., 1., 0.))
    behind = (-.4, 0., 0.)
    with pytest.raises(WristTangentBlocked, match="PROJECTION_GEOMETRY_INVALID"):
        solve(context, project(behind), project(behind), button=behind)


def test_rounding_projected_marker_does_not_shift_ray_panel_result():
    context, project = projection_fixture()
    correction = solve(context, np.round(project((0.03, 0.01, 0.0))).tolist(),
                       project((0.03, 0.015, -0.008)))
    assert correction.delta_base_m == pytest.approx((0., .005, -.008), abs=1e-8)


def test_existing_tangent_cap_is_preserved():
    context, project = projection_fixture()
    correction = solve(context, project((0.03, 0.01, 0.0)), project((0.03, 0.06, 0.0)))
    assert correction.requested_norm_m == pytest.approx(.05, abs=1e-8)
    assert correction.applied_norm_m == pytest.approx(.035)
    assert correction.delta_base_m == pytest.approx((0., .035, 0.), abs=1e-8)
    assert correction.capped


@pytest.mark.parametrize("z", [-.008, 0., .027])
@pytest.mark.parametrize("tilted", [False, True])
def test_ab_removes_only_upward_component_without_changing_panel_gap(z, tilted):
    context, project = projection_fixture()
    normal = np.array([1., .016, .026] if tilted else [1., 0., 0.])
    normal /= np.linalg.norm(normal)
    button = np.array([.03, .01, 0.])
    delta = np.array([0., .012, z])
    delta[0] = -float(normal[1:] @ delta[1:]) / normal[0]
    baseline = solve(context, project(button), project(button + delta),
                     normal_out_base=normal.tolist())
    actual = solve(context, project(button), project(button + delta),
                   normal_out_base=normal.tolist(), suppress_upward_correction=True)
    assert actual.suppress_upward_correction is True
    assert abs(float(normal @ actual.delta_base_m)) < 1e-9
    assert actual.delta_base_m[2] == pytest.approx(min(z, 0.), abs=1e-8)
    horizontal = np.cross(normal, [0., 0., 1.])
    assert float(horizontal @ actual.delta_base_m) == pytest.approx(
        float(horizontal @ baseline.delta_base_m), abs=1e-8)
    assert actual.applied_norm_m <= baseline.applied_norm_m + 1e-9
    assert actual.suppressed_upward_base_z_m == pytest.approx(max(z, 0.), abs=1e-8)


def test_ab_replays_observed_27mm_upward_correction_and_retains_cap():
    context, project = projection_fixture()
    # 17:28 trial: same panel plane with normal sign reversed for this fixture.
    normal = [.9995429107887478, .01603547756590442, .02562875243141243]
    delta = np.array([-.0008924637878561792, .012216853797462728, .02716295961557927])
    button = np.array([.03, .01, 0.])
    actual = solve(context, project(button), project(button + delta),
                   normal_out_base=normal, suppress_upward_correction=True)
    assert actual.delta_base_m[2] == pytest.approx(0., abs=1e-8)
    assert actual.suppressed_upward_base_z_m == pytest.approx(delta[2], abs=1e-8)
    for scale in [1., 3.]:
        capped = solve(context, project(button), project(button + scale * delta),
                       normal_out_base=normal, suppress_upward_correction=True)
        assert capped.applied_norm_m <= .035
        assert abs(float(np.array(normal) @ capped.delta_base_m)) < 1e-9


def test_mixed_projection_pose_and_wrong_identity_are_rejected():
    context, project = projection_fixture()
    pixel = project((.03, .01, 0.0))
    with pytest.raises(WristTangentBlocked, match="PROJECTION_TARGET_MISMATCH"):
        solve(context, [pixel[0] + 5, pixel[1]], pixel)
    with pytest.raises(WristTangentBlocked, match="WRIST_IDENTITY_MISMATCH"):
        solve(context, pixel, pixel, wrist_lock_identity=("wrong-camera",))
    broken = deepcopy(context)
    broken["camera_from_base"][0][0] = float("nan")
    with pytest.raises(WristTangentBlocked, match="PROJECTION_GEOMETRY_INVALID"):
        solve(broken, pixel, pixel)


@pytest.mark.parametrize("pre_y", [-0.012, 0.01, 0.021])
def test_press_aims_at_absolute_corrected_target_not_start_plus_correction(pre_y):
    unsigned = dict(make_wrist_corrected_receipt().unsigned_mapping())
    unsigned["terminal_q_rad"] = list(unsigned["terminal_q_rad"])
    unsigned["terminal_q_rad"][1] = pre_y
    # The correction remains Gemini-relative. It must not be applied to PRE again.
    unsigned.update(wrist_correction_reference="VIS_LOCK_MINUS_PROJECTED_GEMINI_TARGET",
                    wrist_correction_method="CURRENT_POSE_RAY_PANEL_INTERSECTION",
                    wrist_projected_target_px=[320.0, 240.0])
    digest = canonical_hash(unsigned)
    receipt = PreTerminalReceipt.from_mapping(dict(unsigned, receipt_id=digest, receipt_sha256=digest))
    model = TangentModel()
    plan = prepare_stage_b_plan(receipt, model, now_ns=1_100_000_000,
                               contact_authorized=True, press_authorized=True,
                               force_control=False, zero_dwell=True,
                               require_wrist_tangent_correction=True)
    expected = (.03 - PRESS_OVERTRAVEL_M, .005, 0.)
    assert plan.plan["forward_target_tcp_base_m"] == pytest.approx(expected)
    assert plan.plan["forward_sample_tcp_base_m"][-1] == pytest.approx(expected, abs=.0025)
    verified = verify_stage_b_plan(canonical_json(plan.plan), plan.plan_sha256,
                                  expected_receipt=receipt, model=model,
                                  current_q_rad=receipt.terminal_q_rad,
                                  now_ns=1_100_000_000, now_monotonic=.1,
                                  expected_press_authorized=True)
    assert verified["forward_target_tcp_base_m"] == pytest.approx(expected)


@pytest.mark.parametrize("visual_z", [0., .02])
@pytest.mark.parametrize("normal_yaw_deg", [0., 5.76205614709667])
def test_real_qc_correction_receipt_and_executor_share_one_target_reference(visual_z, normal_yaw_deg):
    """Run the deployed QC method bodies without ROS clients or motion."""
    source = Path(__file__).resolve().parents[1] / "roarm_quick_control_panel.py"
    tree = ast.parse(source.read_text())
    names = {"_stage_b_wrist_correction_locked", "_stage_b_build_receipt_locked"}
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"math": math}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    q = (.1, .012, 0., 0., .2, .3)
    context, project = projection_fixture()
    context["feedback_positions_rad"] = list(q)
    model = TangentModel()
    yaw = math.radians(normal_yaw_deg)
    normal = np.array([math.cos(yaw), math.sin(yaw), 0.])
    visual = np.array([.03 - math.tan(yaw) * .005, .005, visual_z])
    geometry = dict(model.hashes, panel_geometry_sha256="e" * 64,
                    p_button_base_m=[.03, 0., 0.], normal_out_base=normal.tolist(),
                    wall_surface_point_base_m=[.03, 0., 0.])
    wrist = dict(wrist_projection_context=context,
                 wrist_projected_target_px=project((.03, 0., 0.)),
                 wrist_lock_center_px=project(visual),
                 wrist_lock_identity=identity(), wrist_acquisition_id="press-click-1",
                 wrist_lock_stage="PRE-LOCK", wrist_lock_observation_count=3)
    feedback = dict(positions_rad=q, feedback_stamp_ns=1_000_000_000)
    node = SimpleNamespace(
        feedback=feedback, _wrist_tangent_calibration=WristTangentCalibration.from_mapping(calibration()),
        _stage_b_q_stability_locked=lambda: (True, "", q, 3, 0.),
        _stage_b_canonical=lambda value: (canonical_json(value), canonical_hash(value)),
    )
    correction = namespace["_stage_b_wrist_correction_locked"](node, geometry, wrist)
    expected_z = min(visual_z, 0.)
    corrected = np.array([visual[0], .005, expected_z])
    assert correction.delta_base_m == pytest.approx(corrected - [.03, 0., 0.], abs=1e-8)
    text, _ = namespace["_stage_b_build_receipt_locked"](
        node, geometry, feedback, {"resource_counts": {"publisher": 0, "lease": 0, "armed": 0}},
        wrist, correction,
    )
    receipt = PreTerminalReceipt.from_mapping(json.loads(text))
    assert json.loads(text)["wrist_suppress_upward_correction"] is True
    assert json.loads(text)["wrist_suppressed_upward_base_z_m"] == pytest.approx(max(visual_z, 0.), abs=1e-8)
    assert receipt.p_button_base_m == pytest.approx(corrected, abs=1e-8)
    plan = prepare_stage_b_plan(receipt, model, now_ns=1_100_000_000,
                               contact_authorized=True, press_authorized=True,
                               force_control=False, zero_dwell=True,
                               require_wrist_tangent_correction=True)
    verified = verify_stage_b_plan(canonical_json(plan.plan), plan.plan_sha256,
                                  expected_receipt=receipt, model=model, current_q_rad=q,
                                  now_ns=1_100_000_000, now_monotonic=.1,
                                  expected_press_authorized=True)
    assert verified["forward_target_tcp_base_m"] == pytest.approx(
        corrected - PRESS_OVERTRAVEL_M * normal, abs=1e-8)
    assert plan.return_waypoints_rad[-1] == q
    node.feedback = dict(feedback, positions_rad=[.11, .012, 0., 0., .2, .3])
    with pytest.raises(ValueError, match="WRIST_PROJECTION_POSE_CHANGED"):
        namespace["_stage_b_wrist_correction_locked"](node, geometry, wrist)


def test_moving_overlay_never_paints_departure_pixel_on_button2_or_redetects():
    source = Path(__file__).resolve().parents[1] / "button1_precontact_live_view.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "wrist_overlay")
    markers = []
    namespace = dict(np=np, cv2=cv2, math=math, time=time,
                     project_with_reason=lambda point, k, frame_shape=None, dist_coeffs=None:
                         project_camera_point(point, k, dist_coeffs=dist_coeffs, frame_shape=frame_shape),
                     draw_marker=lambda frame, pixel, name, color: markers.append(name) or pixel is not None)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    lock = SimpleNamespace(center_px=(320., 100.), acquired_projection_px=(330., 120.),
                           target_base_m=(0., 0., .3), stage="PRE-LOCK", identity=("click",))
    tracker = SimpleNamespace(locked=lock)
    # Deliberately provide a fresh other-button observation: it must not be consumed.
    observation = SimpleNamespace(frame_size_wh=(640, 480), received_monotonic_s=time.monotonic(),
                                  error="", candidates=("button2",))
    notes = namespace["wrist_overlay"](
        np.zeros((480, 640, 3), np.uint8), np.eye(4), np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]]),
        np.zeros(5), np.array([0., .02, .2]), np.eye(4), {},
        SimpleNamespace(available=True, target_base_m=(0., 0., .3)), observation, tracker,
        {"enabled": False, "reason": "PRESS_ACTIVE", "phase": "ACTION_IN_FLIGHT", "normal_out_base": [0., 0., -1.]},
        ("click",), 123,
    )
    assert "VIS-LOCK" not in markers
    assert any("HIDDEN DURING MOTION" in note for note in notes)
    assert any("SUSPENDED_AFTER_PRESS_LOCK" in note for note in notes)
    assert tracker.locked is lock


def test_current_panel_projection_is_not_overwritten_by_old_stage_a_plan():
    source = Path(__file__).resolve().parents[1] / "button1_precontact_live_view.py"
    tree = ast.parse(source.read_text())
    render = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render")
    assignment = next(n for n in ast.walk(render) if isinstance(n, ast.If)
                      and any(isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Name)
                              and stmt.value.id == "frozen_overlay" for stmt in n.body))
    expression = compile(ast.Expression(assignment.test), str(source), "eval")
    assert eval(expression, dict(frozen_overlay=SimpleNamespace(available=True),
                                qc_current={"press_panel_geometry_ready": True})) is False
    assert eval(expression, dict(frozen_overlay=SimpleNamespace(available=True), qc_current={})) is True
