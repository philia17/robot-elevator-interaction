"""Pure, safety-oriented state machine with no hardware side effects."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from voice_control.audit import AuditLogger
from voice_control.config import VoiceControlConfig
from voice_control.gateway import DryRunGateway
from voice_control.intents import Intent, ParseResult
from voice_control.parser import parse_utterance


class State(str, Enum):
    BOOT = "BOOT"
    DEVICE_CHECK = "DEVICE_CHECK"
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PARSING = "PARSING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    SUBMITTING = "SUBMITTING"
    EXECUTING = "EXECUTING"
    STOP_LATCHED = "STOP_LATCHED"


@dataclass(frozen=True)
class PendingCommand:
    command_id: str
    floor: int
    expires_monotonic: float


@dataclass(frozen=True)
class ProcessOutcome:
    state: State
    parse: ParseResult
    safety_decision: str
    command_id: str | None = None
    generation: int = 0


@dataclass
class VoiceStateMachine:
    config: VoiceControlConfig
    gateway: DryRunGateway
    audit: AuditLogger
    clock: Callable[[], float] = time.monotonic
    state: State = State.BOOT
    generation: int = 0
    pending: PendingCommand | None = None
    last_command_id: str | None = None
    _bootstrapped: bool = field(default=False, repr=False)

    def bootstrap(self) -> None:
        if self._bootstrapped:
            return
        self._transition(State.DEVICE_CHECK, reason="boot_complete")
        self._transition(State.IDLE, reason="device_check_ok_dry_run")
        self._bootstrapped = True

    def process_text(self, text: str, *, now: float | None = None) -> ProcessOutcome:
        self.bootstrap()
        current_time = self.clock() if now is None else now
        started = self.clock()
        previous_state = self.state
        parse = parse_utterance(
            text,
            floor_min=self.config.floor_min,
            floor_max=self.config.floor_max,
            allow_english_no_floor_correction=(
                previous_state == State.AWAITING_CONFIRMATION
                and self.pending is not None
            ),
        )
        self.audit.log("parse", source_text=text, intent=parse.intent, floor=parse.floor,
                       reject_reason=parse.reject_reason,
                       latency_ms=round((self.clock() - started) * 1000, 3))

        # STOP is checked before every state-specific branch, including EXECUTING
        # and STOP_LATCHED.
        if parse.intent == Intent.STOP:
            return self._stop(parse, text)
        if previous_state == State.STOP_LATCHED:
            return self._outcome(parse, "reject_stop_latched")
        if previous_state == State.EXECUTING:
            return self._outcome(parse, "reject_busy_executing")

        self._transition(State.LISTENING, reason="utterance_received")
        self._transition(State.PARSING, reason="parse_begin")

        if parse.intent == Intent.PRESS_FLOOR:
            return self._press_floor(parse, current_time)
        if parse.intent == Intent.CONFIRM:
            if self.pending is None:
                self._transition(State.IDLE, reason="confirm_without_pending")
                return self._outcome(parse, "reject_confirm_no_pending")
            # A bare confirmation binds to the pending floor at the instant it is
            # processed. Explicit floor text must match it.
            floor = self.pending.floor if parse.floor is None else parse.floor
            return self.confirm(self.pending.command_id, floor, now=current_time,
                                parse=parse)
        if parse.intent == Intent.CANCEL:
            return self._cancel(parse)
        if parse.intent == Intent.STATUS:
            self._transition(State.IDLE, reason="status_reported")
            return self._outcome(parse, "status_reported", self.last_command_id)
        if parse.intent == Intent.WAKE:
            self._transition(State.LISTENING, reason="wake_ack")
            return self._outcome(parse, "wake_ack")

        self._transition(State.IDLE, reason="unknown_intent")
        return self._outcome(parse, "reject_unknown")

    def confirm(self, command_id: str, floor: int, *, now: float | None = None,
                parse: ParseResult | None = None) -> ProcessOutcome:
        """Confirm only an exact command_id + floor binding within its TTL."""
        parsed = parse or ParseResult(Intent.CONFIRM, floor=floor, raw_text="confirm_api")
        current_time = self.clock() if now is None else now
        if self.pending is None:
            return self._outcome(parsed, "reject_confirm_no_pending")
        pending = self.pending
        if current_time > pending.expires_monotonic:
            self.pending = None
            self._transition(State.IDLE, reason="confirm_expired", command_id=pending.command_id)
            return self._outcome(parsed, "reject_confirm_expired", pending.command_id)
        if command_id != pending.command_id:
            self._transition(State.AWAITING_CONFIRMATION, reason="command_id_mismatch")
            return self._outcome(parsed, "reject_confirm_command_id_mismatch", pending.command_id)
        if floor != pending.floor:
            self._transition(State.AWAITING_CONFIRMATION, reason="floor_mismatch")
            return self._outcome(parsed, "reject_confirm_floor_mismatch", pending.command_id)

        self.pending = None
        self.last_command_id = pending.command_id
        generation = self.generation
        self._transition(State.SUBMITTING, reason="confirm_accepted",
                         command_id=pending.command_id, floor=pending.floor)
        self.audit.log("safety_decision", decision="confirm_accepted",
                       command_id=pending.command_id, floor=pending.floor)
        self._transition(State.EXECUTING, reason="gateway_submit",
                         command_id=pending.command_id)
        self.gateway.submit(pending.command_id, pending.floor, generation,
                            self._on_execution_complete)
        return self._outcome(parsed, "submitted", pending.command_id)

    def _press_floor(self, parse: ParseResult, now: float) -> ProcessOutcome:
        if parse.floor is None or not parse.accepted:
            self._transition(State.IDLE, reason="invalid_floor")
            return self._outcome(parse, "reject_press_floor")
        if self.pending is not None:
            old_pending = self.pending
            if now > old_pending.expires_monotonic:
                self.audit.log(
                    "pending_command_expired",
                    command_id=old_pending.command_id,
                    floor=old_pending.floor,
                    replacement_floor=parse.floor,
                )
                self.pending = None
            elif parse.floor == old_pending.floor:
                self._transition(
                    State.AWAITING_CONFIRMATION,
                    reason="pending_floor_repeated",
                    command_id=old_pending.command_id,
                    floor=old_pending.floor,
                )
                return self._outcome(
                    parse, "repeat_pending_floor", old_pending.command_id)
            else:
                command_id = uuid.uuid4().hex[:12]
                self.pending = PendingCommand(
                    command_id,
                    parse.floor,
                    now + self.config.confirmation_ttl_seconds,
                )
                self.last_command_id = command_id
                self.audit.log(
                    "pending_command_replaced",
                    old_command_id=old_pending.command_id,
                    old_floor=old_pending.floor,
                    new_command_id=command_id,
                    new_floor=parse.floor,
                    ttl_seconds=self.config.confirmation_ttl_seconds,
                )
                self._transition(
                    State.AWAITING_CONFIRMATION,
                    reason="pending_replaced",
                    old_command_id=old_pending.command_id,
                    command_id=command_id,
                    floor=parse.floor,
                )
                return self._outcome(parse, "pending_replaced", command_id)
        command_id = uuid.uuid4().hex[:12]
        self.pending = PendingCommand(command_id, parse.floor,
                                      now + self.config.confirmation_ttl_seconds)
        self.last_command_id = command_id
        self.audit.log("pending_command", command_id=command_id, floor=parse.floor,
                       ttl_seconds=self.config.confirmation_ttl_seconds)
        self._transition(State.AWAITING_CONFIRMATION, reason="await_confirmation")
        return self._outcome(parse, "await_confirmation", command_id)

    def _cancel(self, parse: ParseResult) -> ProcessOutcome:
        command_id = self.pending.command_id if self.pending else None
        self.pending = None
        self._transition(State.IDLE, reason="cancelled" if command_id else "cancel_no_pending")
        return self._outcome(parse, "cancelled_pending" if command_id else "cancel_no_pending",
                             command_id)

    def _stop(self, parse: ParseResult, source_text: str) -> ProcessOutcome:
        self.generation += 1
        aborted = self.gateway.abort()
        command_id = self.pending.command_id if self.pending else self.last_command_id
        self.pending = None
        self._transition(State.STOP_LATCHED, reason="stop", generation=self.generation)
        self.audit.log("safety_decision", decision="stop", source_text=source_text,
                       command_id=command_id, abort_effective=aborted,
                       generation=self.generation)
        return self._outcome(parse, "stop_latched", command_id)

    def _on_execution_complete(self, generation: int, outcome: str) -> None:
        if generation != self.generation:
            self.audit.log("execution_outcome", outcome=outcome, applied=False,
                           reason="stale_generation", callback_generation=generation,
                           current_generation=self.generation)
            return
        if self.state != State.EXECUTING:
            self.audit.log("execution_outcome", outcome=outcome, applied=False,
                           reason="unexpected_state", state=self.state)
            return
        self.audit.log("execution_outcome", outcome=outcome, applied=True,
                       generation=generation)
        self._transition(State.IDLE, reason="execution_complete", outcome=outcome)

    def _outcome(self, parse: ParseResult, decision: str,
                 command_id: str | None = None) -> ProcessOutcome:
        self.audit.log("safety_decision", decision=decision, command_id=command_id,
                       state=self.state, generation=self.generation)
        return ProcessOutcome(self.state, parse, decision, command_id, self.generation)

    def _transition(self, new_state: State, **details: Any) -> None:
        old = self.state
        self.state = new_state
        self.audit.log("state_transition", from_state=old, to_state=new_state, **details)

    def status_dict(self) -> dict[str, Any]:
        return {"state": self.state.value, "generation": self.generation,
                "pending_command_id": self.pending.command_id if self.pending else None,
                "pending_floor": self.pending.floor if self.pending else None,
                "dry_run": self.config.dry_run}
