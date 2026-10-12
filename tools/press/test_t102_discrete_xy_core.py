from __future__ import annotations

import copy

from press.t102_discrete_xy_core import DiscreteXYCore, XYState
from press.t102_session_template_core import load_policy


def _policy():
    value = copy.deepcopy(load_policy()["xy"])
    value["hard_joint_limits_rad"] = [[-3.14, 3.14], [-1.57, 1.57], [0.0, 2.95], [-1.57, 1.57], [-3.14, 3.14], [0.0, 1.5]]
    return value


Q = (0.1, -0.8, 2.0, 0.0, -1.4, 0.1)


def _measure(core, error=(30.0, 0.0), frame=100):
    return core.measure(session_id="s", frame_stamp_ns=frame, frame_age_s=0.01,
                        button_available=True, tool_center_available=True,
                        error_px=error, feedback_age_s=0.01, current_q=Q)


def test_bands_aligned_and_real_runtime_is_precisely_disabled():
    core = DiscreteXYCore("s", _policy())
    assert _measure(core, (6.0, -2.0)).code == "NO_CONTACT_XY_ALIGNED"
    assert core.state is XYState.ALIGNED
    core = DiscreteXYCore("s", _policy())
    assert _measure(core, (20.0, 0.0)).band == "MEDIUM"
    assert core.apply_one(session_id="s", candidate_target_rad=(0.16, *Q[1:]), predicted_cross_axis_drift_px=0, predicted_normal_drift_m=0).code == "TOOL_CENTER_UNAVAILABLE"


def test_unavailable_stale_and_new_frame_gates():
    core = DiscreteXYCore("s", _policy())
    assert core.measure(session_id="s", frame_stamp_ns=1, frame_age_s=0, button_available=False, tool_center_available=True, error_px=(1, 1), feedback_age_s=0, current_q=Q).code == "BUTTON_ROI_UNAVAILABLE"
    assert core.measure(session_id="s", frame_stamp_ns=1, frame_age_s=0, button_available=True, tool_center_available=False, error_px=(1, 1), feedback_age_s=0, current_q=Q).code == "TOOL_CENTER_UNAVAILABLE"
    assert core.measure(session_id="s", frame_stamp_ns=1, frame_age_s=9, button_available=True, tool_center_available=True, error_px=(1, 1), feedback_age_s=0, current_q=Q).code == "WRIST_FRAME_STALE"
    assert _measure(core, frame=10).accepted
    core.state = XYState.NEED_MEASURE
    assert _measure(core, frame=10).code == "NEW_FRAME_REQUIRED"


def test_envelope_deadband_cross_axis_normal_and_limits():
    for change, code in (
        ({"target": (0.11, *Q[1:])}, "DEADBAND_UNRESOLVED"),
        ({"target": (0.3, *Q[1:])}, "CORRECTION_ENVELOPE_REJECTED"),
        ({"target": (0.16, *Q[1:]), "cross": 4.0}, "CORRECTION_ENVELOPE_REJECTED"),
        ({"target": (0.16, *Q[1:]), "normal": 0.002}, "CORRECTION_ENVELOPE_REJECTED"),
        ({"target": (9.0, *Q[1:])}, "CORRECTION_ENVELOPE_REJECTED"),
    ):
        core = DiscreteXYCore("s", _policy(), runtime_enabled=True)
        assert _measure(core).accepted
        result = core.apply_one(session_id="s", candidate_target_rad=change["target"], predicted_cross_axis_drift_px=change.get("cross", 0), predicted_normal_drift_m=change.get("normal", 0))
        assert result.code == code


def test_exactly_one_target_per_click_new_frame_direction_progress_and_max_three():
    core = DiscreteXYCore("s", _policy(), runtime_enabled=True)
    assert _measure(core, frame=100).accepted
    target = (0.16, *Q[1:])
    assert core.apply_one(session_id="s", candidate_target_rad=target, predicted_cross_axis_drift_px=0, predicted_normal_drift_m=0).accepted
    assert core.apply_one(session_id="s", candidate_target_rad=target, predicted_cross_axis_drift_px=0, predicted_normal_drift_m=0).code == "MEASURE_REQUIRED"
    assert core.record_terminal(measured_q=target, new_frame_stamp_ns=100, new_error_px=(20, 0)).code == "NEW_FRAME_REQUIRED"

    core = DiscreteXYCore("s", _policy(), runtime_enabled=True)
    assert _measure(core, frame=100).accepted
    assert core.apply_one(session_id="s", candidate_target_rad=target, predicted_cross_axis_drift_px=0, predicted_normal_drift_m=0).accepted
    wrong = (0.04, *Q[1:])
    assert core.record_terminal(measured_q=wrong, new_frame_stamp_ns=101, new_error_px=(20, 0)).code == "WRONG_DIRECTION"

    core = DiscreteXYCore("s", _policy(), runtime_enabled=True)
    for index in range(3):
        assert _measure(core, frame=100 + index * 2).accepted
        target = (Q[0] + 0.06, *Q[1:])
        assert core.apply_one(session_id="s", candidate_target_rad=target, predicted_cross_axis_drift_px=0, predicted_normal_drift_m=0).accepted
        assert core.record_terminal(measured_q=target, new_frame_stamp_ns=101 + index * 2, new_error_px=(20-index, 0)).accepted
    assert _measure(core, frame=200).code == "CORRECTION_LIMIT_REACHED"
