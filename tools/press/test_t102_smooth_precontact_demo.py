import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from press.t102_fast_runtime_envelope import (
    ENCODER_STEP_RAD,
    STANDARD_STANDBY_RAD,
    canonical_sha256,
    load_and_validate_package,
)
import press.t102_smooth_precontact_demo as smooth_demo
from press.t102_smooth_precontact_demo import (
    CONFIRMATION,
    DEFAULT_V2_PACKAGE,
    SmoothBlocked,
    measured_handoff_match,
)


SPEED_STEPS_S = (34, 17, 17, 23, 34, 17)


def package_copy():
    return copy.deepcopy(load_and_validate_package(DEFAULT_V2_PACKAGE))


def speed_model_state(start, target, elapsed_s):
    result = []
    for before, goal, speed in zip(start, target, SPEED_STEPS_S):
        total_steps = round(abs(goal - before) / ENCODER_STEP_RAD)
        moved_steps = min(total_steps, math.floor(speed * elapsed_s))
        direction = 1.0 if goal >= before else -1.0
        result.append(before + direction * moved_steps * ENCODER_STEP_RAD)
    return tuple(result)


def projected_progress(start, target, actual):
    delta = [goal - before for before, goal in zip(start, target)]
    return sum(
        (measured - before) * direction
        for before, measured, direction in zip(start, actual, delta)
    ) / sum(direction * direction for direction in delta)


def test_realistic_asynchronous_handoff_accepts_old_out_of_band_state():
    package = package_copy()
    start = tuple(package["start"]["positions_rad"])
    current = tuple(package["outbound_targets_rad"][0])
    actual = speed_model_state(start, current, 4.5)
    assert projected_progress(start, current, actual) > 0.85
    matched = measured_handoff_match(
        package, 0, start, actual, feedback_age_s=0.1
    )
    assert matched is not None
    assert matched["source"] == "measured_large_step_progress_v2"
    assert matched["next_command"]["within_delta_caps"] is True
    assert matched["endpoint_distance_reduction_rad"] > ENCODER_STEP_RAD


def test_handoff_rejects_stale_wrong_direction_small_progress_and_over_cap():
    package = package_copy()
    start = tuple(package["start"]["positions_rad"])
    current = tuple(package["outbound_targets_rad"][0])
    accepted_actual = speed_model_state(start, current, 4.5)
    with pytest.raises(SmoothBlocked, match="stale"):
        measured_handoff_match(
            package, 0, start, accepted_actual, feedback_age_s=1.0 + 1e-6
        )

    wrong = list(accepted_actual)
    wrong[0] = start[0] - math.radians(1.0)
    assert measured_handoff_match(
        package, 0, start, wrong, feedback_age_s=0.1
    ) is None

    tiny = list(start)
    tiny[1] += ENCODER_STEP_RAD
    assert measured_handoff_match(
        package, 0, start, tiny, feedback_age_s=0.1
    ) is None

    early = speed_model_state(start, current, 2.5)
    assessment = smooth_demo.assess_measured_handoff(
        package, 0, start, early, feedback_age_s=0.1
    )
    assert assessment["next_command"]["within_delta_caps"] is False
    assert measured_handoff_match(
        package, 0, start, early, feedback_age_s=0.1
    ) is None


def test_late_sample_cannot_replace_with_a_reversing_next_target():
    package = package_copy()
    start = tuple(package["start"]["positions_rad"])
    next_target = tuple(package["outbound_targets_rad"][1])
    endpoint = tuple(package["endpoint"]["joint_positions_rad"])
    # The measured state is already beyond the sealed next anchor toward the
    # endpoint. Republishing that anchor would reverse every moving joint.
    actual = tuple(
        before + 0.25 * (goal - before)
        for before, goal in zip(next_target, endpoint)
    )
    assert measured_handoff_match(
        package, 0, start, actual, feedback_age_s=0.1
    ) is None


def test_first_handoff_start_is_bound_to_standard_standby():
    package = package_copy()
    fake_start = list(package["start"]["positions_rad"])
    fake_start[0] += 2.0 * ENCODER_STEP_RAD
    actual = speed_model_state(
        fake_start, package["outbound_targets_rad"][0], 4.5
    )
    with pytest.raises(SmoothBlocked, match="not Standard Standby"):
        measured_handoff_match(
            package, 0, fake_start, actual, feedback_age_s=0.1
        )


def build_fake_runtime(package, *, final_offset_rad=0.0, events=None):
    targets = [tuple(item) for item in package["outbound_targets_rad"]]
    starts = [tuple(package["start"]["positions_rad"])]
    handoffs = []
    for index, elapsed in enumerate((4.3, 5.6, 5.5)):
        actual = speed_model_state(starts[-1], targets[index], elapsed)
        handoffs.append(actual)
        starts.append(actual)

    class FakeNode:
        def create_publisher(self, *_args):
            return object()

        def destroy_publisher(self, _publisher):
            return None

    class FakeRuntime:
        JointState = object

        def __init__(self):
            if events is not None:
                events.append("runtime_ctor")
            self.node = FakeNode()
            self.publisher = None
            self.status = {"state": "CONVERGED_DISARMED"}
            self.joint_samples = []
            self.publish_count = 0
            self.arm_count = 0
            self.disarm_count = 0
            self.lease_enable_count = 0
            self.lease_release_count = 0

        def stable_q(self):
            if self.publish_count == 0:
                return tuple(package["start"]["positions_rad"])
            final = list(package["endpoint"]["joint_positions_rad"])
            final[0] += final_offset_rad
            return tuple(final)

        def gate(self, **_kwargs):
            return {
                "status": dict(self.status),
                "feedback": {"torque_enabled": [True] * 6},
                "publisher_count": 0 if self.publisher is None else 1,
            }

        def enable_lease(self, _lease_id):
            self.lease_enable_count += 1

        def release_lease(self, _lease_id):
            self.lease_release_count += 1

        def arm_once(self):
            self.arm_count += 1
            self.status = {"state": "ARMED"}

        def disarm_once(self):
            self.disarm_count += 1
            self.status = {"state": "DISARMED:operator"}

        def publish_target(self, _target):
            self.publish_count += 1
            self.status = {"state": "COMMAND_ACTIVE"}
            return time.monotonic() - 1.0

        def spin(self, _seconds):
            if 1 <= self.publish_count <= 3:
                actual = handoffs[self.publish_count - 1]
                if not self.joint_samples or self.joint_samples[-1][1] != actual:
                    self.joint_samples.append((time.monotonic(), actual))
            elif self.publish_count == 4:
                endpoint = tuple(package["endpoint"]["joint_positions_rad"])
                if not self.joint_samples or self.joint_samples[-1][1] != endpoint:
                    self.joint_samples.append((time.monotonic(), endpoint))
                self.status = {"state": "COARSE_PROGRESS_DISARMED"}

        def close(self):
            return None

    return FakeRuntime


def test_fake_runtime_uses_four_persistent_targets_three_measured_replacements(tmp_path):
    package = package_copy()
    events = []
    runtime_class = build_fake_runtime(package, events=events)
    args = SimpleNamespace(
        confirmation_text=CONFIRMATION,
        package=DEFAULT_V2_PACKAGE,
        output=tmp_path / "physical_result.json",
    )
    result = smooth_demo.execute(
        args, package, runtime_factory=runtime_class
    )
    assert events == ["runtime_ctor"]
    assert [item["target_rad"] for item in result["publications"]] == package[
        "outbound_targets_rad"
    ]
    assert len(result["replace_events"]) == 3
    assert all(
        event["assessment"]["source"] == "measured_large_step_progress_v2"
        for event in result["replace_events"]
    )
    assert result["command_contract"]["press_included"] is False
    assert result["command_contract"]["return_included"] is False
    assert result["classification"] == "COARSE_PRECONTACT_STOPPED_DISARMED"
    assert result["arm_call_count"] == 1
    assert result["lease_enable_count"] == 1
    assert result["lease_release_count"] == 1


def test_invalid_package_stops_before_runtime_construction(tmp_path):
    package = package_copy()
    package["press"] = {"enabled": False}
    package.pop("package_sha256")
    package["package_sha256"] = canonical_sha256(package)
    invalid_path = tmp_path / "invalid_v2.json"
    invalid_path.write_text(json.dumps(package), encoding="utf-8")
    calls = []

    def forbidden_runtime():
        calls.append("runtime_ctor")
        raise AssertionError("runtime constructed before preflight")

    args = SimpleNamespace(
        confirmation_text=CONFIRMATION,
        package=invalid_path,
        output=tmp_path / "should_not_exist.json",
    )
    with pytest.raises(SmoothBlocked, match="keys mismatch"):
        smooth_demo.execute(args, runtime_factory=forbidden_runtime)
    assert calls == []
    assert not args.output.exists()


def test_default_order_is_load_validate_then_runtime_constructor(monkeypatch, tmp_path):
    package = package_copy()
    events = []
    original = smooth_demo.load_and_validate_package

    def recording_load(*args, **kwargs):
        events.append("load")
        result = original(*args, **kwargs)
        events.append("validated")
        return result

    monkeypatch.setattr(smooth_demo, "load_and_validate_package", recording_load)
    runtime_class = build_fake_runtime(package, events=events)
    args = SimpleNamespace(
        confirmation_text=CONFIRMATION,
        package=DEFAULT_V2_PACKAGE,
        output=tmp_path / "ordered.json",
    )
    smooth_demo.execute(args, runtime_factory=runtime_class)
    assert events[:3] == ["load", "validated", "runtime_ctor"]


def test_safe_terminal_for_wrong_endpoint_is_rejected(tmp_path):
    package = package_copy()
    runtime_class = build_fake_runtime(package, final_offset_rad=0.0201)
    args = SimpleNamespace(
        confirmation_text=CONFIRMATION,
        package=DEFAULT_V2_PACKAGE,
        output=tmp_path / "wrong_endpoint.json",
    )
    with pytest.raises(SmoothBlocked, match="does not match the sealed package endpoint"):
        smooth_demo.execute(args, package, runtime_factory=runtime_class)


def test_stale_confirmation_cannot_reach_runtime(tmp_path):
    package = package_copy()
    calls = []

    def forbidden_runtime():
        calls.append("runtime_ctor")
        raise AssertionError("unexpected runtime")

    args = SimpleNamespace(
        confirmation_text="old confirmation",
        package=DEFAULT_V2_PACKAGE,
        output=tmp_path / "never.json",
    )
    with pytest.raises(SmoothBlocked, match="exact fresh confirmation"):
        smooth_demo.execute(args, package, runtime_factory=forbidden_runtime)
    assert calls == []
