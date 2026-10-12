#!/usr/bin/env python3
"""T4 fake visual-closure state machine (no serial, no ROS, no motion).

Rehearses:
  OBSERVE → ALIGN_PANEL → APPROACH_NORMAL → VERIFY_PRECONTACT → DONE

Abort paths (fail-closed, no auto-retry, no vendor initialize):
  lost target, invalid normal/depth, stale feedback, non-convergence,
  max iterations exceeded.

This module never imports SafeController, never emits T101/T102, and never
writes serial bytes. Intended Cartesian corrections are recorded as fake
commands only for offline evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from press.alignment_guidance import panel_tangent_residual


class Phase(str, Enum):
    OBSERVE = "OBSERVE"
    ALIGN_PANEL = "ALIGN_PANEL"
    APPROACH_NORMAL = "APPROACH_NORMAL"
    VERIFY_PRECONTACT = "VERIFY_PRECONTACT"
    DONE = "DONE"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class VisualObservation:
    """One matched observation tick for the fake closed-loop rehearsal."""

    point_camera_m: tuple[float, float, float] | None
    target_camera_xyz_m: tuple[float, float, float] | None
    panel_normal_camera: tuple[float, float, float] | None
    p_available: bool
    p_stable: bool
    depth_live: bool
    target_depth_valid: bool
    normal_valid: bool
    feedback_fresh: bool
    joints_converged: bool = True
    tick: int = 0
    # The default preserves the original P-depth diagnostic rehearsal.  The
    # current visual-closure route selects ``fn_overlay`` explicitly.
    tip_source: Literal["fn_overlay", "p_depth"] = "p_depth"
    fn_camera_m: tuple[float, float, float] | None = None
    active_candidate_id: str | None = None
    candidate_status: str | None = None


@dataclass(frozen=True)
class FakeMotionCommand:
    """Recorded intent only — never a serial payload."""

    phase: str
    kind: str  # "tangent" | "normal" | "hold"
    delta_camera_m: tuple[float, float, float]
    reason: str
    serial_write: bool = False
    t101: bool = False
    t102: bool = False

    def __post_init__(self) -> None:
        if self.serial_write or self.t101 or self.t102:
            raise ValueError("T4 fake commands must not claim serial/T101/T102 writes")


@dataclass
class T4Config:
    tangent_deadband_m: float = 0.003
    tangent_step_m: float = 0.003
    precontact_gap_m: float = 0.050
    precontact_tolerance_m: float = 0.003
    normal_step_m: float = 0.010
    max_align_steps: int = 40
    max_approach_steps: int = 40
    max_observe_ticks: int = 30
    max_verify_ticks: int = 10
    gain: float = 1.0
    expected_fn_candidate_id: str = (
        "section_c_fingertip_c0_operator_click_best_20260806_v1"
    )
    expected_fn_candidate_status: str = "diagnostic_overlay_default"


@dataclass
class T4Result:
    passed: bool
    final_phase: str
    abort_reason: str | None
    history: list[dict[str, Any]] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)
    serial_write_count: int = 0
    notes: list[str] = field(default_factory=list)
    tip_source: str | None = None
    active_candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "visual_closure_t4_fake_rehearsal",
            "schema_version": "1.0.0",
            "execution_ready": False,
            "publish_tf": False,
            "serial_used": False,
            "ros_used": False,
            "passed": self.passed,
            "final_phase": self.final_phase,
            "abort_reason": self.abort_reason,
            "serial_write_count": self.serial_write_count,
            "command_count": len(self.commands),
            "commands": self.commands,
            "history": self.history,
            "notes": self.notes,
            "tip_source": self.tip_source,
            "active_candidate_id": self.active_candidate_id,
        }


def outward_normal(
    panel_normal: Sequence[float],
    point_camera: Sequence[float],
    target_camera: Sequence[float],
) -> np.ndarray:
    """Return n_out from panel toward the tool (mandatory sign check)."""
    normal = np.asarray(panel_normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9 or not np.all(np.isfinite(normal)):
        raise ValueError("panel normal must be finite and non-zero")
    normal = normal / norm
    tip_minus_target = np.asarray(point_camera, dtype=float) - np.asarray(
        target_camera, dtype=float
    )
    if float(np.dot(normal, tip_minus_target)) < 0.0:
        normal = -normal
    return normal


def panel_axes(n_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal panel tangent axes u, v for a given outward normal."""
    normal = np.asarray(n_out, dtype=float)
    normal = normal / float(np.linalg.norm(normal))
    camera_x = np.array([1.0, 0.0, 0.0], dtype=float)
    u = camera_x - float(np.dot(camera_x, normal)) * normal
    if float(np.linalg.norm(u)) < 1e-6:
        u = np.array([0.0, 1.0, 0.0], dtype=float)
        u = u - float(np.dot(u, normal)) * normal
    u = u / float(np.linalg.norm(u))
    v = np.cross(normal, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def panel_frame_errors(
    point_camera: Sequence[float],
    target_camera: Sequence[float],
    panel_normal: Sequence[float],
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (e_u, e_v, e_n, u, v, n_out) with e_* = (A-P)·axis.

    Applying ``delta = e_u*u + e_v*v`` moves P toward A in the panel tangent
    plane. ``e_n = (A-P)·n_out`` is negative when P is in front of the panel
    (standoff along outward normal).
    """
    n_out = outward_normal(panel_normal, point_camera, target_camera)
    u, v = panel_axes(n_out)
    a_minus_p = np.asarray(target_camera, dtype=float) - np.asarray(
        point_camera, dtype=float
    )
    e_u = float(np.dot(a_minus_p, u))
    e_v = float(np.dot(a_minus_p, v))
    e_n = float(np.dot(a_minus_p, n_out))
    return e_u, e_v, e_n, u, v, n_out


def clamp_step(delta: np.ndarray, max_step_m: float) -> np.ndarray:
    length = float(np.linalg.norm(delta))
    if length <= max_step_m or length < 1e-12:
        return delta
    return delta * (max_step_m / length)


def _selected_tip_point(obs: VisualObservation) -> tuple[float, float, float] | None:
    """Return the selected diagnostic tip estimate without changing authority."""
    if obs.tip_source == "fn_overlay":
        return obs.fn_camera_m
    if obs.tip_source == "p_depth":
        return obs.point_camera_m
    return None


def _abort_gates(obs: VisualObservation, config: T4Config) -> str | None:
    if not obs.feedback_fresh:
        return "stale feedback"
    if not obs.joints_converged:
        return "joints did not converge"
    if not obs.target_depth_valid or obs.target_camera_xyz_m is None:
        return "lost target / depth invalid"
    if not obs.normal_valid or obs.panel_normal_camera is None:
        return "invalid panel normal"
    if obs.tip_source == "fn_overlay":
        if obs.active_candidate_id != config.expected_fn_candidate_id:
            return "Fn candidate ID mismatch"
        if obs.candidate_status != config.expected_fn_candidate_status:
            return "Fn candidate is not diagnostic overlay"
        if obs.fn_camera_m is None:
            return "lost fingertip Fn"
        # P-depth is optional diagnostic evidence for the F-first route.  Do
        # not silently fall back to it or reject a valid frozen Fn when it is
        # stale/unavailable.
        return None
    if obs.tip_source == "p_depth":
        if not obs.depth_live:
            return "invalid depth / P stale"
        if not obs.p_available or obs.point_camera_m is None:
            return "lost fingertip P"
        if not obs.p_stable:
            return "P not stable"
        return None
    return "unknown tip source"


def _target_dict(obs: VisualObservation) -> dict[str, Any]:
    assert obs.target_camera_xyz_m is not None
    assert obs.panel_normal_camera is not None
    return {
        "camera_xyz": obs.target_camera_xyz_m,
        "normal": obs.panel_normal_camera,
        "depth_valid": obs.target_depth_valid,
        "normal_valid": obs.normal_valid,
    }


class VisualClosureT4Fake:
    """Pure offline state machine for T4 rehearsal."""

    def __init__(self, config: T4Config | None = None) -> None:
        self.config = config or T4Config()
        self.phase = Phase.OBSERVE
        self.abort_reason: str | None = None
        self.commands: list[FakeMotionCommand] = []
        self.history: list[dict[str, Any]] = []
        self._align_steps = 0
        self._approach_steps = 0
        self._observe_ticks = 0
        self._verify_ticks = 0
        self._tip_source: str | None = None
        self._active_candidate_id: str | None = None

    @property
    def serial_write_count(self) -> int:
        return sum(1 for command in self.commands if command.serial_write)

    def step(self, obs: VisualObservation) -> FakeMotionCommand | None:
        if self.phase in (Phase.DONE, Phase.ABORTED):
            return None

        record: dict[str, Any] = {
            "tick": obs.tick,
            "phase_in": self.phase.value,
            "tip_source": obs.tip_source,
            "active_candidate_id": obs.active_candidate_id,
            "p_depth_diagnostic": {
                "available": obs.p_available,
                "stable": obs.p_stable,
                "live": obs.depth_live,
            },
        }

        if self.phase != Phase.OBSERVE:
            reason = _abort_gates(obs, self.config)
            if reason is not None:
                return self._abort(reason, obs, record)

        command: FakeMotionCommand | None = None
        if self.phase == Phase.OBSERVE:
            command = self._step_observe(obs, record)
        elif self.phase == Phase.ALIGN_PANEL:
            command = self._step_align(obs, record)
        elif self.phase == Phase.APPROACH_NORMAL:
            command = self._step_approach(obs, record)
        elif self.phase == Phase.VERIFY_PRECONTACT:
            command = self._step_verify(obs, record)

        record["phase_out"] = self.phase.value
        record["abort_reason"] = self.abort_reason
        if command is not None:
            record["command"] = asdict(command)
            self.commands.append(command)
        self.history.append(record)
        return command

    def _abort(
        self, reason: str, obs: VisualObservation, record: dict[str, Any]
    ) -> None:
        self.phase = Phase.ABORTED
        self.abort_reason = reason
        record["phase_out"] = self.phase.value
        record["abort_reason"] = reason
        record["tick"] = obs.tick
        self.history.append(record)
        return None

    def _step_observe(
        self, obs: VisualObservation, record: dict[str, Any]
    ) -> FakeMotionCommand | None:
        self._observe_ticks += 1
        record["observe_ticks"] = self._observe_ticks
        reason = _abort_gates(obs, self.config)
        if reason is None:
            self._tip_source = obs.tip_source
            self._active_candidate_id = obs.active_candidate_id
            self.phase = Phase.ALIGN_PANEL
            return FakeMotionCommand(
                phase=Phase.OBSERVE.value,
                kind="hold",
                delta_camera_m=(0.0, 0.0, 0.0),
                reason="gates open; enter ALIGN_PANEL",
            )
        if self._observe_ticks > self.config.max_observe_ticks:
            self.phase = Phase.ABORTED
            self.abort_reason = f"observe timeout: {reason}"
            return None
        return FakeMotionCommand(
            phase=Phase.OBSERVE.value,
            kind="hold",
            delta_camera_m=(0.0, 0.0, 0.0),
            reason=f"waiting: {reason}",
        )

    def _step_align(
        self, obs: VisualObservation, record: dict[str, Any]
    ) -> FakeMotionCommand | None:
        point_camera_m = _selected_tip_point(obs)
        assert point_camera_m is not None
        assert obs.target_camera_xyz_m is not None
        assert obs.panel_normal_camera is not None
        e_u, e_v, e_n, u, v, _n_out = panel_frame_errors(
            point_camera_m, obs.target_camera_xyz_m, obs.panel_normal_camera
        )
        # Diagnostic residual in alignment_guidance convention (P-A).
        h, vert, n_diag = panel_tangent_residual(
            point_camera_m, _target_dict(obs)
        )
        tangent_norm = math.hypot(e_u, e_v)
        record["e_u_m"] = e_u
        record["e_v_m"] = e_v
        record["e_n_m"] = e_n
        record["tangent_residual_m"] = tangent_norm
        record["diag_tip_minus_a_m"] = {"horizontal": h, "vertical": vert, "normal": n_diag}
        if tangent_norm <= self.config.tangent_deadband_m:
            self.phase = Phase.APPROACH_NORMAL
            return FakeMotionCommand(
                phase=Phase.ALIGN_PANEL.value,
                kind="hold",
                delta_camera_m=(0.0, 0.0, 0.0),
                reason="tangent within deadband; enter APPROACH_NORMAL",
            )
        self._align_steps += 1
        if self._align_steps > self.config.max_align_steps:
            self.phase = Phase.ABORTED
            self.abort_reason = "non-convergence in ALIGN_PANEL"
            return None
        delta = self.config.gain * (e_u * u + e_v * v)
        delta = clamp_step(delta, self.config.tangent_step_m)
        return FakeMotionCommand(
            phase=Phase.ALIGN_PANEL.value,
            kind="tangent",
            delta_camera_m=(float(delta[0]), float(delta[1]), float(delta[2])),
            reason=f"tangent step |e|={tangent_norm * 1000.0:.1f} mm",
        )

    def _step_approach(
        self, obs: VisualObservation, record: dict[str, Any]
    ) -> FakeMotionCommand | None:
        point_camera_m = _selected_tip_point(obs)
        assert point_camera_m is not None
        assert obs.target_camera_xyz_m is not None
        assert obs.panel_normal_camera is not None
        n_out = outward_normal(
            obs.panel_normal_camera, point_camera_m, obs.target_camera_xyz_m
        )
        a = np.asarray(obs.target_camera_xyz_m, dtype=float)
        p = np.asarray(point_camera_m, dtype=float)
        gap = float(np.dot(p - a, n_out))
        record["normal_gap_m"] = gap
        target_gap = self.config.precontact_gap_m
        if abs(gap - target_gap) <= self.config.precontact_tolerance_m:
            self.phase = Phase.VERIFY_PRECONTACT
            return FakeMotionCommand(
                phase=Phase.APPROACH_NORMAL.value,
                kind="hold",
                delta_camera_m=(0.0, 0.0, 0.0),
                reason="gap at pre-contact; enter VERIFY_PRECONTACT",
            )
        self._approach_steps += 1
        if self._approach_steps > self.config.max_approach_steps:
            self.phase = Phase.ABORTED
            self.abort_reason = "non-convergence in APPROACH_NORMAL"
            return None
        # Move P along -n_out when gap too large (advance toward panel).
        error = gap - target_gap
        delta = -error * n_out * self.config.gain
        delta = clamp_step(delta, self.config.normal_step_m)
        return FakeMotionCommand(
            phase=Phase.APPROACH_NORMAL.value,
            kind="normal",
            delta_camera_m=(float(delta[0]), float(delta[1]), float(delta[2])),
            reason=f"normal step gap={gap * 1000.0:.1f} mm -> {target_gap * 1000.0:.1f} mm",
        )

    def _step_verify(
        self, obs: VisualObservation, record: dict[str, Any]
    ) -> FakeMotionCommand | None:
        point_camera_m = _selected_tip_point(obs)
        assert point_camera_m is not None
        assert obs.target_camera_xyz_m is not None
        assert obs.panel_normal_camera is not None
        self._verify_ticks += 1
        e_u, e_v, _e_n, _u, _v, n_out = panel_frame_errors(
            point_camera_m, obs.target_camera_xyz_m, obs.panel_normal_camera
        )
        tangent_norm = math.hypot(e_u, e_v)
        gap = float(
            np.dot(
                np.asarray(point_camera_m) - np.asarray(obs.target_camera_xyz_m),
                n_out,
            )
        )
        record["tangent_residual_m"] = tangent_norm
        record["normal_gap_m"] = gap
        gap_ok = abs(gap - self.config.precontact_gap_m) <= self.config.precontact_tolerance_m
        tangent_ok = tangent_norm <= self.config.tangent_deadband_m
        if gap_ok and tangent_ok:
            self.phase = Phase.DONE
            return FakeMotionCommand(
                phase=Phase.VERIFY_PRECONTACT.value,
                kind="hold",
                delta_camera_m=(0.0, 0.0, 0.0),
                reason="VERIFY_PRECONTACT passed; stop before PRESS",
            )
        if self._verify_ticks > self.config.max_verify_ticks:
            self.phase = Phase.ABORTED
            self.abort_reason = "VERIFY_PRECONTACT failed gates"
            return None
        return FakeMotionCommand(
            phase=Phase.VERIFY_PRECONTACT.value,
            kind="hold",
            delta_camera_m=(0.0, 0.0, 0.0),
            reason="verify hold; gates not yet green",
        )

    def result(self) -> T4Result:
        return T4Result(
            passed=self.phase == Phase.DONE and self.serial_write_count == 0,
            final_phase=self.phase.value,
            abort_reason=self.abort_reason,
            history=list(self.history),
            commands=[asdict(command) for command in self.commands],
            serial_write_count=self.serial_write_count,
            notes=[
                "T4 fake rehearsal only; ends at VERIFY_PRECONTACT.",
                "No serial writes, no T101/T102, no press, no vendor initialize.",
                "execution_ready remains false.",
                "Fn is a diagnostic overlay, never published TF or contact geometry.",
            ],
            tip_source=self._tip_source,
            active_candidate_id=self._active_candidate_id,
        )


def apply_fake_delta(
    point_camera_m: Sequence[float], delta_camera_m: Sequence[float]
) -> tuple[float, float, float]:
    point = np.asarray(point_camera_m, dtype=float) + np.asarray(delta_camera_m, dtype=float)
    return (float(point[0]), float(point[1]), float(point[2]))


def run_synthetic_rehearsal(
    *,
    target_xyz: Sequence[float] = (0.0, 0.0, 1.0),
    panel_normal: Sequence[float] = (0.0, 0.0, -1.0),
    p0: Sequence[float] = (0.012, -0.008, 0.88),
    config: T4Config | None = None,
    settle_ticks: int = 2,
    tip_source: Literal["fn_overlay", "p_depth"] = "fn_overlay",
) -> T4Result:
    """Drive the SM with a perfect plant and an explicit diagnostic tip source."""
    machine = VisualClosureT4Fake(config=config)
    point = tuple(float(v) for v in p0)
    target = tuple(float(v) for v in target_xyz)
    normal = tuple(float(v) for v in panel_normal)

    # Warm-up OBSERVE with unstable then stable P.
    for tick in range(settle_ticks):
        obs = VisualObservation(
            point_camera_m=point,
            target_camera_xyz_m=target,
            panel_normal_camera=normal,
            p_available=True,
            p_stable=False,
            depth_live=True,
            target_depth_valid=True,
            normal_valid=True,
            feedback_fresh=True,
            joints_converged=True,
            tick=tick,
            tip_source=tip_source,
            fn_camera_m=point if tip_source == "fn_overlay" else None,
            active_candidate_id=(
                machine.config.expected_fn_candidate_id if tip_source == "fn_overlay" else None
            ),
            candidate_status=(
                machine.config.expected_fn_candidate_status if tip_source == "fn_overlay" else None
            ),
        )
        machine.step(obs)

    tick = settle_ticks
    while machine.phase not in (Phase.DONE, Phase.ABORTED) and tick < 200:
        obs = VisualObservation(
            point_camera_m=point,
            target_camera_xyz_m=target,
            panel_normal_camera=normal,
            p_available=True,
            p_stable=True,
            depth_live=True,
            target_depth_valid=True,
            normal_valid=True,
            feedback_fresh=True,
            joints_converged=True,
            tick=tick,
            tip_source=tip_source,
            fn_camera_m=point if tip_source == "fn_overlay" else None,
            active_candidate_id=(
                machine.config.expected_fn_candidate_id if tip_source == "fn_overlay" else None
            ),
            candidate_status=(
                machine.config.expected_fn_candidate_status if tip_source == "fn_overlay" else None
            ),
        )
        command = machine.step(obs)
        if command is not None and command.kind in ("tangent", "normal"):
            point = apply_fake_delta(point, command.delta_camera_m)
        tick += 1
    return machine.result()


def write_rehearsal_report(result: T4Result, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temp/visual_closure_t4_fake_rehearsal.json"),
    )
    args = parser.parse_args(argv)
    result = run_synthetic_rehearsal()
    path = write_rehearsal_report(result, args.output)
    print(f"passed={result.passed} final={result.final_phase} wrote={path}")
    if result.abort_reason:
        print(f"abort_reason={result.abort_reason}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
