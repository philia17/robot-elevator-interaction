from __future__ import annotations

from press.button_light_observer import CLEAR_NO_LIGHT, LIGHT_ON, UNOBSERVABLE
from press.supervised_button_cycle_core import SupervisedButtonCycle


def reach_press(cycle, target="target-1", wrist="wrist-1", *, first=True):
    cycle.observable(target)
    if first:
        cycle.go_sta_complete()
        cycle.go_pre_complete()
    cycle.wrist_ready(wrist)
    cycle.press_started()


def test_light_on_waits_for_back_then_goes_to_final_standby():
    cycle = SupervisedButtonCycle(max_attempts=2)
    cycle.start()
    reach_press(cycle)
    view = cycle.press_terminal(back_proven=True, outcome=LIGHT_ON)
    assert view.state == "GO_STA_FINAL"
    assert view.next_action == "GO_STA"
    assert cycle.final_go_sta_complete().state == "STANDBY"


def test_back_unproven_never_classifies_or_returns_standby():
    cycle = SupervisedButtonCycle()
    cycle.start()
    reach_press(cycle)
    view = cycle.press_terminal(back_proven=False, outcome=LIGHT_ON)
    assert view.state == "HUMAN_REQUIRED"
    assert view.outcome == UNOBSERVABLE
    assert view.next_action == "STOP_FOR_HUMAN"


def test_clear_requires_fresh_target_and_wrist_for_second_attempt():
    cycle = SupervisedButtonCycle(max_attempts=2)
    cycle.start()
    reach_press(cycle)
    view = cycle.press_terminal(back_proven=True, outcome=CLEAR_NO_LIGHT)
    assert (view.state, view.attempt) == ("WAIT_OBSERVABLE", 2)
    try:
        cycle.observable("target-1")
    except ValueError as exc:
        assert str(exc) == "FRESH_TARGET_REQUIRED"
    else:
        raise AssertionError("stale target accepted")
    next_view = cycle.observable("target-2")
    assert next_view.state == "WRIST_ACQUIRE"
    assert next_view.next_action == "ACQUIRE_FRESH_WRIST"
    try:
        cycle.wrist_ready("wrist-1")
    except ValueError as exc:
        assert str(exc) == "FRESH_WRIST_REQUIRED"
    else:
        raise AssertionError("stale wrist accepted")


def test_attempt_budget_is_finite():
    cycle = SupervisedButtonCycle(max_attempts=2)
    cycle.start()
    reach_press(cycle)
    cycle.press_terminal(back_proven=True, outcome=CLEAR_NO_LIGHT)
    reach_press(cycle, "target-2", "wrist-2", first=False)
    view = cycle.press_terminal(back_proven=True, outcome=CLEAR_NO_LIGHT)
    assert view.attempt == 2
    assert view.state == "HUMAN_REQUIRED"


def test_unobservable_stops_for_human():
    cycle = SupervisedButtonCycle()
    cycle.start()
    reach_press(cycle)
    view = cycle.press_terminal(back_proven=True, outcome=UNOBSERVABLE)
    assert view.state == "HUMAN_REQUIRED"


def test_external_failure_stops_active_cycle_for_human():
    cycle = SupervisedButtonCycle()
    cycle.start()
    view = cycle.fail_unobservable()
    assert view.state == "HUMAN_REQUIRED"
    assert view.outcome == UNOBSERVABLE
    assert view.next_action == "STOP_FOR_HUMAN"
