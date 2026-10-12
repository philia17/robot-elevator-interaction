"""Focused no-hardware tests for the isolated T102 outbound-only core."""

from __future__ import annotations

import math

import pytest

from press.t102_outbound_route_builder import OBSERVED_FIELD_FAILURE_RESIDUAL_RAD
from press.t102_outbound_route_builder import build_t102_outbound_diagnostic_route
from press.t102_outbound_executor_core import (
    ENCODER_STEP_RAD,
    StatusTargetAck,
    ExecutionTicket,
    ExecutorConfig,
    FakeController,
    Feedback,
    FrozenBindings,
    SealedOutboundRoute,
    T102ExecutorError,
    T102OutboundExecutorCore,
    T102State,
    canonical_sha256,
    setpoint_sha256,
)


# Deterministic six-joint vectors and 64-hex bindings for the real diagnostic
# route builder output.  These mirror the reviewed offline candidate only as
# finite numeric fixtures; they are not an execution release.
DIAGNOSTIC_START_RAD = [
    0.046019424,
    -1.188835111,
    2.328582836,
    0.46786414,
    -1.515573018,
    0.0,
]
DIAGNOSTIC_CANDIDATE_RAD = [
    0.17195432546434475,
    -0.7592116272033396,
    2.0769242265081016,
    0.03485803655203858,
    -1.497108414621783,
    0.0,
]
DIAGNOSTIC_BINDINGS = {
    "m0_target_sha256": "a" * 64,
    "fn_sha256": "b" * 64,
    "r6_sha256": "c" * 64,
    "provenance_sha256": "d" * 64,
}


def _bindings() -> FrozenBindings:
    return FrozenBindings(*("a" * 64, "b" * 64, "c" * 64, "d" * 64))


def _route_mapping() -> dict:
    b = _bindings()
    return {
        "schema": "t102_outbound_only_route_v1",
        "phase": "OUTBOUND_TO_PRECONTACT",
        "no_contact": True,
        "press": False,
        "retract": False,
        "return": False,
        "automatic_retry": False,
        "frozen_bindings": vars(b),
        "setpoints_rad": [[0.0] * 6, [0.1, -0.1, 0.0, 0.0, 0.0, 0.0]],
    }


def _route() -> SealedOutboundRoute:
    raw = _route_mapping()
    return SealedOutboundRoute.from_mapping(raw, canonical_sha256(raw))


def _certificate(route: SealedOutboundRoute) -> dict:
    raw_route = _route_mapping()
    return {
        "schema": "t102_prepared_route_certificate_v1",
        "status": "OFFLINE_ROUTE_RELEASED",
        "execution_ready": False,
        "motion_authorized": False,
        "no_hardware_transition": True,
        "sealed_outbound_route": raw_route,
        "sealed_outbound_route_sha256": canonical_sha256(raw_route),
        "frozen_bindings": vars(route.bindings),
    }


def _diagnostic_route_raw(**changes) -> dict:
    """Real ``build_t102_outbound_diagnostic_route`` output, optionally mutated.

    Any mutation is re-sealed so the embedded route self-hash always covers
    the exact unsigned fields the executor route parser will verify.
    """

    raw = build_t102_outbound_diagnostic_route(
        verified_start_positions_rad=DIAGNOSTIC_START_RAD,
        verified_static_ik_candidate_rad=DIAGNOSTIC_CANDIDATE_RAD,
        frozen_bindings=DIAGNOSTIC_BINDINGS,
    )
    raw.update(changes)
    unsigned = {key: value for key, value in raw.items() if key != "route_sha256"}
    raw["route_sha256"] = canonical_sha256(unsigned)
    return raw


def _sealed(raw: dict) -> SealedOutboundRoute:
    # Current artifacts embed a self-hash over every field except route_sha256;
    # that embedded value is the hash the executor parser must be supplied.
    return SealedOutboundRoute.from_mapping(raw, raw["route_sha256"])


def _certificate_for(raw_route: dict) -> dict:
    return {
        "schema": "t102_prepared_route_certificate_v1",
        "status": "OFFLINE_ROUTE_RELEASED",
        "execution_ready": False,
        "motion_authorized": False,
        "no_hardware_transition": True,
        "sealed_outbound_route": raw_route,
        "sealed_outbound_route_sha256": raw_route["route_sha256"],
        "frozen_bindings": dict(raw_route["frozen_bindings"]),
    }


def _ticket(route: SealedOutboundRoute, **changes) -> ExecutionTicket:
    values = dict(
        route_sha256=route.route_sha256,
        bindings=route.bindings,
        issued_monotonic=10.0,
        expires_monotonic=11.0,
        operator_confirmation="MOVE_TO_PRECONTACT_ONLY",
    )
    values.update(changes)
    return ExecutionTicket(**values)


def _post_feedback(route: SealedOutboundRoute) -> list[Feedback]:
    feedback = []
    for index, point in enumerate(route.setpoints_rad):
        final = index == len(route.setpoints_rad) - 1
        feedback.append(
            Feedback(
                10.10 + (index * 0.05),
                "CONVERGED_DISARMED" if final else "TRACKING_OUTBOUND",
                acknowledged_setpoint_sha256=setpoint_sha256(point),
                measured_positions_rad=point if final else None,
            )
        )
    return feedback


def _status_acks(route: SealedOutboundRoute) -> list[StatusTargetAck]:
    """Fresh, encoder-quantized status telemetry targets, one per setpoint.

    The safe driver rounds moving joints onto the 4096-step grid before
    publishing its T102 payload target, so the replay mirrors that
    quantization instead of emitting float-identical setpoints.
    """

    acks = []
    for index, point in enumerate(route.setpoints_rad):
        quantized = tuple(
            round(value / ENCODER_STEP_RAD) * ENCODER_STEP_RAD for value in point
        )
        acks.append(StatusTargetAck(10.08 + (index * 0.05), quantized))
    return acks


def _controller(**changes) -> FakeController:
    values = {
        "latest_feedback": Feedback(10.05, "CONVERGED_DISARMED"),
        "latest_status_ack": StatusTargetAck(10.05),
    }
    values.update(changes)
    return FakeController(**values)


def _core() -> T102OutboundExecutorCore:
    return T102OutboundExecutorCore(ExecutorConfig(enabled=True))


def test_success_emits_only_sealed_outbound_setpoints_and_tears_down_owner():
    route = _route()
    controller = _controller(
        post_emit_feedback=_post_feedback(route),
        post_emit_status_acks=_status_acks(route),
    )
    result = _core().execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert result is T102State.COMPLETE
    assert [point for _, point in controller.emitted] == list(route.setpoints_rad)
    assert controller.acquired == controller.released == ["t102-outbound-only-core"]
    # Every emitted point consumed its own fresh status acknowledgement.
    assert len(controller.status_acks_read) >= 2 * len(route.setpoints_rad)


@pytest.mark.parametrize(
    "controller",
    [
        _controller(latest_feedback=Feedback(9.0, "CONVERGED_DISARMED")),
        _controller(foreign_owner_ids=("quick-control",)),
        _controller(latest_feedback=Feedback(10.05, "CONVERGED_DISARMED", "fault")),
        _controller(cancel_requested=True),
    ],
)
def test_gate_failures_emit_nothing_and_teardown(controller):
    route, core = _route(), _core()
    with pytest.raises(T102ExecutorError):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert core.state is T102State.ABORTED
    assert controller.emitted == []
    assert controller.released == ["t102-outbound-only-core"]


def test_expired_ticket_never_acquires_owner():
    route, controller = _route(), _controller()
    with pytest.raises(T102ExecutorError, match="not current"):
        _core().execute(
            route,
            _ticket(route, expires_monotonic=10.05),
            controller,
            now_monotonic=10.1,
        )
    assert controller.acquired == controller.released == []


def test_ticket_timeout_above_hard_limit_never_acquires_owner():
    route, controller = _route(), _controller()
    with pytest.raises(T102ExecutorError, match="exceeds maximum"):
        _core().execute(
            route,
            _ticket(route, expires_monotonic=13.0),
            controller,
            now_monotonic=10.1,
        )
    assert controller.acquired == controller.released == []


def test_cancellation_between_setpoints_stops_without_retry_and_tears_down():
    route = _route()
    controller = _controller(
        cancel_after_emits=1,
        post_emit_feedback=_post_feedback(route),
        post_emit_status_acks=_status_acks(route),
    )
    core = _core()
    with pytest.raises(T102ExecutorError, match="cancellation"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 1
    assert controller.released == ["t102-outbound-only-core"]


def test_default_configuration_is_inert():
    route, controller = _route(), _controller()
    with pytest.raises(T102ExecutorError, match="disabled"):
        T102OutboundExecutorCore().execute(
            route, _ticket(route), controller, now_monotonic=10.1
        )
    assert controller.emitted == []


def test_unexpected_hook_failure_aborts_and_tears_down_owner():
    class ExplodingController(FakeController):
        def emit_t102_setpoint(self, owner_id, positions_rad):
            raise RuntimeError("transport failed")

    route, core = _route(), _core()
    controller = ExplodingController(
        latest_feedback=Feedback(10.05, "CONVERGED_DISARMED"),
        latest_status_ack=StatusTargetAck(10.05),
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert core.state is T102State.ABORTED
    assert controller.emitted == []
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_prepared_certificate_is_rejected_by_default_without_staging_or_motion():
    route, core = _route(), T102OutboundExecutorCore()
    certificate = _certificate(route)
    result = core.accept_prepared_route_certificate(
        certificate, canonical_sha256(certificate)
    )
    assert not result.accepted_for_inert_staging
    assert core.staged_route is None
    assert core.state is T102State.INERT


def test_explicit_inert_certificate_staging_binds_route_and_keeps_execution_disabled():
    route = _route()
    certificate = _certificate(route)
    core = T102OutboundExecutorCore(ExecutorConfig(accept_prepared_certificates=True))
    result = core.accept_prepared_route_certificate(
        certificate, canonical_sha256(certificate)
    )
    assert result.accepted_for_inert_staging
    assert result.route == route == core.staged_route
    assert core.state is T102State.INERT
    controller = _controller(post_emit_feedback=_post_feedback(route))
    with pytest.raises(T102ExecutorError, match="disabled"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert controller.emitted == []


def test_certificate_hash_or_binding_tamper_is_rejected_without_replacing_previous_stage():
    route = _route()
    good = _certificate(route)
    core = T102OutboundExecutorCore(ExecutorConfig(accept_prepared_certificates=True))
    assert core.accept_prepared_route_certificate(
        good, canonical_sha256(good)
    ).accepted_for_inert_staging
    bad = _certificate(route)
    bad["frozen_bindings"] = dict(bad["frozen_bindings"], fn_sha256="e" * 64)
    result = core.accept_prepared_route_certificate(bad, canonical_sha256(bad))
    assert not result.accepted_for_inert_staging
    assert core.staged_route == route


def test_certificate_rejects_dynamically_constructed_invalid_status_value():
    route = _route()
    certificate = _certificate(route)
    # Keep the invalid value non-interned so this guards the policy check's
    # value semantics rather than relying on Python string identity behavior.
    certificate["status"] = "".join(("OFFLINE_ROUTE_RELEASED", "_TAMPERED"))
    core = T102OutboundExecutorCore(ExecutorConfig(accept_prepared_certificates=True))
    result = core.accept_prepared_route_certificate(
        certificate, canonical_sha256(certificate)
    )
    assert not result.accepted_for_inert_staging
    assert "status" in result.reason
    assert core.staged_route is None


def test_certificate_accepts_noninterned_policy_values_by_value():
    route = _route()
    certificate = _certificate(route)
    certificate["schema"] = "".join(("t102_prepared_", "route_certificate_v1"))
    certificate["status"] = "".join(("OFFLINE_ROUTE_", "RELEASED"))
    core = T102OutboundExecutorCore(ExecutorConfig(accept_prepared_certificates=True))
    result = core.accept_prepared_route_certificate(
        certificate, canonical_sha256(certificate)
    )
    assert result.accepted_for_inert_staging
    assert core.staged_route == route


def test_certificate_cannot_claim_execution_or_motion_authority():
    route = _route()
    certificate = _certificate(route)
    certificate["execution_ready"] = True
    core = T102OutboundExecutorCore(ExecutorConfig(accept_prepared_certificates=True))
    result = core.accept_prepared_route_certificate(
        certificate, canonical_sha256(certificate)
    )
    assert not result.accepted_for_inert_staging
    assert core.staged_route is None


def test_real_diagnostic_route_is_rejected_before_owner_acquisition_or_emission():
    """The builder's real artifact cannot execute, even fully enabled.

    ExecutorConfig(enabled=True) and a valid current ticket are deliberately
    supplied; the offline-diagnostic markers must still fail closed at the
    admission boundary, before ``acquire_owner`` and before any setpoint.
    """

    raw = _diagnostic_route_raw()
    assert raw["route_mode"] == "OFFLINE_DIAGNOSTIC_ONLY"
    assert raw["command_interface"] == "NOT_IMPLEMENTED"
    assert raw["diagnostic_only"] is True
    route = _sealed(raw)
    core = _core()
    controller = _controller()
    with pytest.raises(T102ExecutorError, match="OFFLINE_DIAGNOSTIC_ONLY"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert core.state is T102State.INERT
    assert controller.acquired == controller.released == []
    assert controller.emitted == []


@pytest.mark.parametrize(
    "changes, error",
    [
        (
            {"command_interface": "IMPLEMENTED", "diagnostic_only": False},
            "OFFLINE_DIAGNOSTIC_ONLY",
        ),
        (
            {"route_mode": "OUTBOUND_EXECUTION", "diagnostic_only": False},
            "command interface is NOT_IMPLEMENTED",
        ),
        (
            {"route_mode": "OUTBOUND_EXECUTION", "command_interface": "IMPLEMENTED"},
            "diagnostic_only",
        ),
    ],
)
def test_each_diagnostic_marker_alone_blocks_execution(changes, error):
    """Every diagnostic marker is independently fail-closed at admission."""

    route = _sealed(_diagnostic_route_raw(**changes))
    controller = _controller()
    with pytest.raises(T102ExecutorError, match=error):
        _core().execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert controller.acquired == controller.released == []
    assert controller.emitted == []


def test_inert_certificate_staging_still_parses_diagnostic_route_but_execute_stays_blocked():
    """Staging behavior is unchanged; the marker rejection only lives at execute."""

    raw = _diagnostic_route_raw()
    route = _sealed(raw)
    core = T102OutboundExecutorCore(ExecutorConfig(accept_prepared_certificates=True))
    certificate = _certificate_for(raw)
    result = core.accept_prepared_route_certificate(
        certificate, canonical_sha256(certificate)
    )
    assert result.accepted_for_inert_staging
    assert core.staged_route == route
    assert core.state is T102State.INERT
    controller = _controller()
    with pytest.raises(T102ExecutorError, match="OFFLINE_DIAGNOSTIC_ONLY"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert controller.acquired == controller.released == []
    assert controller.emitted == []


@pytest.mark.parametrize(
    "feedback, error",
    [
        (
            Feedback(10.10, "TRACKING_OUTBOUND", acknowledged_setpoint_sha256="0" * 64),
            "does not acknowledge",
        ),
        (
            Feedback(9.0, "TRACKING_OUTBOUND", acknowledged_setpoint_sha256="0" * 64),
            "stale",
        ),
    ],
)
def test_each_setpoint_requires_fresh_command_bound_feedback(feedback, error):
    route, core = _route(), _core()
    # Deliberately malformed acknowledgement is replaced with the correct hash
    # for the stale case, so that staleness is the observed failure.
    if feedback.observed_monotonic == 9.0:
        feedback = Feedback(
            feedback.observed_monotonic,
            feedback.state,
            acknowledged_setpoint_sha256=setpoint_sha256(route.setpoints_rad[0]),
        )
    controller = _controller(
        post_emit_feedback=[feedback],
        post_emit_status_acks=[_status_acks(route)[0]],
    )
    with pytest.raises(T102ExecutorError, match=error):
        core.execute(route, _ticket(route), controller, now_monotonic=10.1)
    assert len(controller.emitted) == 1


def test_final_success_requires_disarmed_and_measured_endpoint_within_tolerance():
    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    sequence[-1] = Feedback(
        10.15,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=(final[0] + 0.03, *final[1:]),
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="outside tolerance"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == len(route.setpoints_rad)


def test_field_failure_elbow_residual_2_11deg_aborts_with_teardown():
    """Replay the observed first-field-waypoint failure deterministically.

    The elbow stopped 0.036815538 rad (2.11 deg) short of the target and the
    safe driver faulted after its single allowed reissue.  The executor must
    abort, emit nothing further, and tear down ownership.
    """

    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    elbow = final[2] + OBSERVED_FIELD_FAILURE_RESIDUAL_RAD
    sequence[-1] = Feedback(
        10.15,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=(final[0], final[1], elbow, *final[3:]),
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    assert math.degrees(OBSERVED_FIELD_FAILURE_RESIDUAL_RAD) > 2.0
    with pytest.raises(T102ExecutorError, match="outside tolerance"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert core.state is T102State.ABORTED
    assert len(controller.emitted) == len(route.setpoints_rad)
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_stable_residual_within_soft_band_succeeds_only_when_driver_terminals():
    """A <=2 deg stable residual succeeds only under the existing driver policy.

    The executor accepts the terminal only when the replayed safe driver
    declares CONVERGED_DISARMED (its soft-converge policy) and the measured
    endpoint also satisfies the executor's own 0.02 rad final joint gate.  The
    executor never invents convergence from a small residual alone.
    """

    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    residual = (
        0.01  # 0.57 deg: inside the driver's 2 deg soft band and the executor gate.
    )
    sequence[-1] = Feedback(
        10.15,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=(final[0] + residual, *final[1:]),
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    assert math.degrees(residual) <= 2.0
    assert (
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
        is T102State.COMPLETE
    )
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_same_stable_residual_without_driver_terminal_does_not_succeed():
    """Without the driver's CONVERGED_DISARMED terminal, even a tiny residual aborts."""

    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    sequence[-1] = Feedback(
        10.15,
        "TRACKING_OUTBOUND",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=None,
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="not converged and disarmed"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert core.state is T102State.ABORTED


def test_driver_soft_band_alone_is_not_sufficient_executor_gate_never_relaxed():
    """A driver soft-converged 1.9 deg residual still fails the executor's gate.

    This documents the bounded fix: the executor does not widen its final
    joint gate to the driver's 2x-tolerance soft band.
    """

    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    residual = math.radians(1.9)
    sequence[-1] = Feedback(
        10.15,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=(final[0] + residual, *final[1:]),
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="outside tolerance"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_each_emitted_point_requires_fresh_status_target_ack():
    """A later waypoint's status acknowledgement must be strictly newer."""

    route, core = _route(), _core()
    acks = _status_acks(route)
    acks[1] = StatusTargetAck(acks[0].observed_monotonic, acks[1].target_positions_rad)
    controller = _controller(
        post_emit_feedback=_post_feedback(route),
        post_emit_status_acks=acks,
    )
    with pytest.raises(
        T102ExecutorError, match="status target acknowledgement did not transition"
    ):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 2
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_each_emitted_point_requires_status_target_bound_within_encoder_step():
    """A status target two encoder steps away belongs to a different command."""

    route, core = _route(), _core()
    acks = _status_acks(route)
    target = list(acks[1].target_positions_rad)
    target[1] -= 2.0 * ENCODER_STEP_RAD
    acks[1] = StatusTargetAck(acks[1].observed_monotonic, tuple(target))
    controller = _controller(
        post_emit_feedback=_post_feedback(route),
        post_emit_status_acks=acks,
    )
    with pytest.raises(
        T102ExecutorError, match="does not acknowledge emitted setpoint"
    ):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 2


def test_each_emitted_point_requires_fresh_feedback_terminal():
    """A later waypoint's feedback terminal must be strictly newer and bound."""

    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    sequence[1] = Feedback(
        sequence[0].observed_monotonic,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=final,
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="did not transition"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 2


def test_later_waypoint_requires_feedback_ack_of_that_exact_setpoint():
    route, core = _route(), _core()
    sequence = _post_feedback(route)
    first = route.setpoints_rad[0]
    sequence[1] = Feedback(
        10.15,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(first),
        measured_positions_rad=route.setpoints_rad[1],
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(
        T102ExecutorError, match="does not acknowledge emitted setpoint"
    ):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 2


def test_stale_feedback_after_later_setpoint_aborts_with_teardown():
    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    sequence[1] = Feedback(
        9.9,
        "CONVERGED_DISARMED",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=final,
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="stale"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 2
    assert core.state is T102State.ABORTED
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_fault_after_later_setpoint_aborts_with_teardown():
    route, core = _route(), _core()
    sequence = _post_feedback(route)
    final = route.setpoints_rad[-1]
    sequence[1] = Feedback(
        10.15,
        "CONVERGED_DISARMED",
        "safe-driver fault after reissue",
        acknowledged_setpoint_sha256=setpoint_sha256(final),
        measured_positions_rad=final,
    )
    controller = _controller(
        post_emit_feedback=sequence,
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="fault present after setpoint"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 2
    assert core.state is T102State.ABORTED
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_foreign_publisher_mid_route_aborts_with_teardown():
    route, core = _route(), _core()
    controller = _controller(
        activate_foreign_after_emits=1,
        foreign_owner_ids_after_emits=("quick-control",),
        post_emit_feedback=_post_feedback(route),
        post_emit_status_acks=_status_acks(route),
    )
    with pytest.raises(T102ExecutorError, match="foreign command publisher present"):
        core.execute(route, _ticket(route), controller, now_monotonic=10.2)
    assert len(controller.emitted) == 1
    assert core.state is T102State.ABORTED
    assert controller.released == controller.acquired == ["t102-outbound-only-core"]


def test_forbidden_route_phase_or_return_is_rejected_before_execution():
    raw = _route_mapping()
    raw["return"] = True
    with pytest.raises(T102ExecutorError, match="outbound-only"):
        SealedOutboundRoute.from_mapping(raw, canonical_sha256(raw))
