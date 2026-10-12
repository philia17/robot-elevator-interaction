import math

import pytest

from press.t102_three_large_step_coarse import (
    CoarseBlocked,
    classify_progress,
    next_candidate,
)


START = (0.0, -1.2, 2.3, 0.48, -1.5, 0.0)
END = (0.18, -0.84, 2.06, 0.09, -1.49, 0.0)


def test_three_receding_horizon_targets_end_exactly_without_overshoot():
    actual = START
    targets = []
    for slots in (3, 2, 1):
        target = next_candidate(actual, END, slots)
        targets.append(target)
        actual = target
    assert actual == pytest.approx(END)
    assert all(target[1] <= END[1] for target in targets)


def test_candidate_refuses_impossible_remaining_cap():
    with pytest.raises(CoarseBlocked, match="cap"):
        next_candidate(START, (1.0, *END[1:]), 3)


def test_progress_accepts_useful_undertravel_toward_endpoint():
    target = next_candidate(START, END, 3)
    after = tuple(a + 0.6 * (b - a) for a, b in zip(START, target))
    result = classify_progress(START, target, after, END)
    assert result["accepted"] is True
    assert result["max_useful_motion_rad"] >= 0.035
    assert result["endpoint_distance_reduction_rad"] > 0.0


@pytest.mark.parametrize("scale", [0.0, 0.1])
def test_zero_and_small_progress_rejected(scale):
    target = next_candidate(START, END, 3)
    after = tuple(a + scale * (b - a) for a, b in zip(START, target))
    assert classify_progress(START, target, after, END)["accepted"] is False


def test_wrong_direction_rejected():
    target = next_candidate(START, END, 3)
    after = list(START)
    after[1] -= math.radians(0.6)
    result = classify_progress(START, target, after, END)
    assert result["accepted"] is False
    assert result["max_wrong_direction_rad"] > math.radians(0.5)
