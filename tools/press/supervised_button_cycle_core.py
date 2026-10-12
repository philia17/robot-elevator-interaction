"""Pure BACK-first state machine for the final supervised button cycle."""

from __future__ import annotations

from dataclasses import dataclass

from press.button_light_observer import CLEAR_NO_LIGHT, LIGHT_ON, UNOBSERVABLE


@dataclass(frozen=True)
class CycleView:
    state: str
    attempt: int
    max_attempts: int
    outcome: str
    next_action: str


class SupervisedButtonCycle:
    def __init__(self, max_attempts: int = 2, *, single_press: bool = False) -> None:
        if max_attempts < 1:
            raise ValueError("ATTEMPT_BUDGET_INVALID")
        self.single_press = bool(single_press)
        self.max_attempts = 1 if self.single_press else int(max_attempts)
        self.state = "IDLE"
        self.attempt = 0
        self.outcome = ""
        self._target_epoch = ""
        self._wrist_epoch = ""
        self._press_started = False

    def _view(self, next_action: str) -> CycleView:
        return CycleView(self.state, self.attempt, self.max_attempts, self.outcome, next_action)

    def start(self) -> CycleView:
        if self.state != "IDLE":
            raise ValueError("CYCLE_BUSY")
        self.attempt = 1
        self.state = "WAIT_OBSERVABLE"
        return self._view("ACQUIRE_FRESH_TARGET")

    def observable(self, target_epoch: str) -> CycleView:
        if self.state != "WAIT_OBSERVABLE" or not str(target_epoch) or str(target_epoch) == self._target_epoch:
            raise ValueError("FRESH_TARGET_REQUIRED")
        self._target_epoch = str(target_epoch)
        if self.single_press:
            # The accepted click owns a current-q0 approach, not a compulsory
            # detour through Standby.  Its one BACK is followed by GO STA.
            self.state = "GO_PRE"
            return self._view("GO_PRE")
        if self.attempt == 1:
            self.state = "GO_STA"
            return self._view("GO_STA")
        # Stage-B already proved exact BACK to PRE.  A CLEAR retry refreshes
        # target and wrist evidence in place; it must not leave PRE and run a
        # second GO STA -> GO PRE pair.
        self.state = "WRIST_ACQUIRE"
        return self._view("ACQUIRE_FRESH_WRIST")

    def go_sta_complete(self) -> CycleView:
        if self.state != "GO_STA":
            raise ValueError("GO_STA_OUT_OF_ORDER")
        self.state = "GO_PRE"
        return self._view("GO_PRE")

    def go_pre_complete(self) -> CycleView:
        if self.state != "GO_PRE":
            raise ValueError("GO_PRE_OUT_OF_ORDER")
        self.state = "WRIST_ACQUIRE"
        return self._view("ACQUIRE_FRESH_WRIST")

    def wrist_ready(self, wrist_epoch: str) -> CycleView:
        if self.state != "WRIST_ACQUIRE" or not str(wrist_epoch) or str(wrist_epoch) == self._wrist_epoch:
            raise ValueError("FRESH_WRIST_REQUIRED")
        self._wrist_epoch = str(wrist_epoch)
        self.state = "PRESS_BACK"
        self._press_started = False
        return self._view("PRESS" if self.single_press else "START_LIGHT_BASELINE_THEN_PRESS")

    def press_started(self) -> CycleView:
        if self.state != "PRESS_BACK" or self._press_started:
            raise ValueError("PRESS_OUT_OF_ORDER")
        self._press_started = True
        return self._view("WAIT_BACK" if self.single_press else "WAIT_BACK_AND_LIGHT")

    def press_terminal(self, *, back_proven: bool, outcome: str) -> CycleView:
        if self.state != "PRESS_BACK" or not self._press_started:
            raise ValueError("PRESS_OUT_OF_ORDER")
        if back_proven is not True:
            self.state = "HUMAN_REQUIRED"
            self.outcome = UNOBSERVABLE
            return self._view("STOP_FOR_HUMAN")
        if self.single_press:
            # This mode is one operator-requested stroke, not a light-based
            # retry policy.  Missing/unlit light never prevents safe return.
            self.outcome = outcome if outcome in {LIGHT_ON, CLEAR_NO_LIGHT} else UNOBSERVABLE
            self.state = "GO_STA_FINAL"
            return self._view("GO_STA")
        if outcome == LIGHT_ON:
            self.outcome = LIGHT_ON
            self.state = "GO_STA_FINAL"
            return self._view("GO_STA")
        if outcome == CLEAR_NO_LIGHT and self.attempt < self.max_attempts:
            self.outcome = CLEAR_NO_LIGHT
            self.attempt += 1
            self.state = "WAIT_OBSERVABLE"
            return self._view("ACQUIRE_FRESH_TARGET")
        if outcome == CLEAR_NO_LIGHT:
            self.outcome = CLEAR_NO_LIGHT
        else:
            self.outcome = UNOBSERVABLE
        self.state = "HUMAN_REQUIRED"
        return self._view("STOP_FOR_HUMAN")

    def final_go_sta_complete(self) -> CycleView:
        if self.state != "GO_STA_FINAL":
            raise ValueError("FINAL_GO_STA_OUT_OF_ORDER")
        self.state = "STANDBY"
        return self._view("COMPLETE")

    def fail_unobservable(self) -> CycleView:
        if self.state in {"IDLE", "STANDBY", "HUMAN_REQUIRED"}:
            raise ValueError("CYCLE_NOT_ACTIVE")
        self.state = "HUMAN_REQUIRED"
        self.outcome = UNOBSERVABLE
        return self._view("STOP_FOR_HUMAN")
