"""Pure shared state machine for the fixed T102 fast-v2 outbound episode.

The module has no ROS, serial, subprocess, kinematics, URDF, mesh, receipt, or
planner dependency.  Both the CLI runner and the ROS executor use it to load
the same reviewed package and to decide the three measured-feedback handoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from press.t102_fast_runtime_envelope import (
    EXPECTED_PACKAGE_SHA256,
    EnvelopeBlocked,
    assess_measured_handoff,
    load_and_validate_package,
    validate_package_document,
    validate_start_posture,
    vector,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_PACKAGE = (
    REPO_ROOT
    / "config/press/prototype/t102_button1_standard_standby_70mm_fast_route_20260816_v2.json"
)
FINAL_DRIVER_STATES = frozenset({"COARSE_PROGRESS_DISARMED", "CONVERGED_DISARMED"})


class V2ExecutionBlocked(RuntimeError):
    """A package, handoff, or terminal fact failed closed."""


def load_v2_package(path: str | Path = DEFAULT_V2_PACKAGE) -> dict[str, Any]:
    try:
        return load_and_validate_package(
            path, expected_package_sha256=EXPECTED_PACKAGE_SHA256
        )
    except EnvelopeBlocked as exc:
        raise V2ExecutionBlocked(str(exc)) from exc


def validate_v2_package(package: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_package_document(
            package, expected_package_sha256=EXPECTED_PACKAGE_SHA256
        )
    except EnvelopeBlocked as exc:
        raise V2ExecutionBlocked(str(exc)) from exc


@dataclass(frozen=True)
class HandoffRelease:
    transition_index: int
    target_index: int
    target_rad: tuple[float, ...]
    measured_rad: tuple[float, ...]
    assessment: Mapping[str, Any]


class V2ExecutionMachine:
    """Exactly four publications with three asynchronous measured handoffs."""

    def __init__(
        self,
        package: Mapping[str, Any],
        start_rad: Sequence[float],
        *,
        first_published_monotonic: float,
    ) -> None:
        self.package = validate_v2_package(package)
        try:
            self.start_admission = validate_start_posture(self.package, start_rad)
        except EnvelopeBlocked as exc:
            raise V2ExecutionBlocked(str(exc)) from exc
        self.targets = tuple(
            vector(item, f"outbound_targets_rad[{index}]")
            for index, item in enumerate(self.package["outbound_targets_rad"])
        )
        self.endpoint = vector(
            self.package["endpoint"]["joint_positions_rad"], "endpoint"
        )
        self.transition_index = 0
        self.segment_start = vector(start_rad, "start_rad")
        self.last_publication_monotonic = float(first_published_monotonic)

    @property
    def first_target(self) -> tuple[float, ...]:
        return self.targets[0]

    @property
    def complete_handoffs(self) -> bool:
        return self.transition_index == 3

    def consider_feedback(
        self,
        actual_rad: Sequence[float],
        *,
        feedback_age_s: float,
        now_monotonic: float,
    ) -> HandoffRelease | None:
        if self.complete_handoffs:
            return None
        minimum_interval = float(
            self.package["handoff_policy"]["minimum_replace_interval_s"]
        )
        if float(now_monotonic) - self.last_publication_monotonic < minimum_interval:
            return None
        try:
            assessment = assess_measured_handoff(
                self.package,
                self.transition_index,
                self.segment_start,
                actual_rad,
                feedback_age_s=float(feedback_age_s),
            )
        except EnvelopeBlocked as exc:
            raise V2ExecutionBlocked(str(exc)) from exc
        if assessment["accepted"] is not True:
            return None
        measured = vector(assessment["positions_rad"], "measured handoff")
        release = HandoffRelease(
            transition_index=self.transition_index,
            target_index=self.transition_index + 1,
            target_rad=self.targets[self.transition_index + 1],
            measured_rad=measured,
            assessment=assessment,
        )
        self.segment_start = measured
        self.transition_index += 1
        self.last_publication_monotonic = float(now_monotonic)
        return release

    def validate_terminal(
        self, state: str, final_rad: Sequence[float]
    ) -> dict[str, Any]:
        if not self.complete_handoffs:
            raise V2ExecutionBlocked("final state arrived before all three handoffs")
        if str(state) not in FINAL_DRIVER_STATES:
            raise V2ExecutionBlocked("driver terminal is not an accepted disarmed state")
        final = vector(final_rad, "final_rad")
        maximum_error = max(abs(a - b) for a, b in zip(final, self.endpoint))
        limit = float(self.package["endpoint"]["terminal_max_joint_error_rad"])
        if maximum_error > limit + 1e-12:
            raise V2ExecutionBlocked(
                "safe terminal state does not match the sealed package endpoint"
            )
        return {
            "terminal": str(state),
            "final_q_rad": list(final),
            "maximum_final_joint_error_rad": maximum_error,
            "classification": "COARSE_PRECONTACT_STOPPED_DISARMED",
        }
