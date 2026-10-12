#!/usr/bin/env python3
"""Fast v2 package preflight and deterministic local benchmark.

The historical v1 cache remains on disk as offline evidence.  This default
path reads only the persistent v2 route package and invokes the standard-
library command envelope.  It performs no route generation, checker work,
subprocess launch, geometry-file access, or ROS resource creation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from press.t102_fast_runtime_envelope import (  # noqa: E402
    EXPECTED_PACKAGE_SHA256,
    STANDARD_STANDBY_RAD,
    EnvelopeBlocked,
    load_and_validate_package,
    validate_start_posture,
)


DEFAULT_V2_PACKAGE = (
    REPO_ROOT
    / "config/press/prototype/t102_button1_standard_standby_70mm_fast_route_20260816_v2.json"
)


class CacheBlocked(EnvelopeBlocked):
    """Compatibility name for callers that treat a failed preflight as a miss."""


def verify_v2_preflight(
    package_path: str | Path = DEFAULT_V2_PACKAGE,
    current_start_rad: Sequence[float] = STANDARD_STANDBY_RAD,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    try:
        package = load_and_validate_package(
            package_path, expected_package_sha256=EXPECTED_PACKAGE_SHA256
        )
        start_result = validate_start_posture(package, current_start_rad)
    except EnvelopeBlocked as exc:
        raise CacheBlocked(str(exc)) from exc
    elapsed_ns = time.perf_counter_ns() - started_ns
    return {
        "state": "V2_FAST_PREFLIGHT_PASS",
        "schema": package["schema"],
        "scenario": package["scenario"],
        "package_sha256": package["package_sha256"],
        "outbound_target_count": len(package["outbound_targets_rad"]),
        "start": start_result,
        "duration_ns": elapsed_ns,
        "duration_ms": elapsed_ns / 1_000_000.0,
    }


def benchmark_v2_preflight(
    package_path: str | Path = DEFAULT_V2_PACKAGE,
    current_start_rad: Sequence[float] = STANDARD_STANDBY_RAD,
    *,
    iterations: int = 1000,
) -> dict[str, Any]:
    if isinstance(iterations, bool) or iterations < 1000:
        raise CacheBlocked("benchmark requires at least 1000 iterations")
    durations_ns: list[int] = []
    benchmark_started_ns = time.perf_counter_ns()
    for _ in range(iterations):
        call_started_ns = time.perf_counter_ns()
        verify_v2_preflight(package_path, current_start_rad)
        durations_ns.append(time.perf_counter_ns() - call_started_ns)
    total_ns = time.perf_counter_ns() - benchmark_started_ns
    ordered = sorted(durations_ns)
    p95_index = max(0, math.ceil(0.95 * iterations) - 1)
    return {
        "state": "V2_FAST_PREFLIGHT_BENCHMARK_PASS",
        "package": str(Path(package_path).resolve()),
        "package_sha256": EXPECTED_PACKAGE_SHA256,
        "iterations": iterations,
        "statistics_rule": "median and nearest-rank p95 over full file-load plus validation calls",
        "median_ms": statistics.median(durations_ns) / 1_000_000.0,
        "p95_ms": ordered[p95_index] / 1_000_000.0,
        "maximum_ms": max(durations_ns) / 1_000_000.0,
        "minimum_ms": min(durations_ns) / 1_000_000.0,
        "total_ms": total_ns / 1_000_000.0,
        "target_p95_ms": 10.0,
        "target_met": ordered[p95_index] <= 10_000_000,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--package", type=Path, default=DEFAULT_V2_PACKAGE)
    parser.add_argument("--current-start-rad", nargs=6, type=float)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.preflight and not args.benchmark:
        print(
            json.dumps(
                {
                    "state": "INERT",
                    "execution_ready": False,
                    "motion_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.current_start_rad is None:
        print(
            json.dumps(
                {
                    "state": "V2_FAST_PREFLIGHT_BLOCKED",
                    "reason": "current start is required",
                },
                sort_keys=True,
            )
        )
        return 2
    try:
        result = (
            benchmark_v2_preflight(
                args.package,
                args.current_start_rad,
                iterations=args.iterations,
            )
            if args.benchmark
            else verify_v2_preflight(args.package, args.current_start_rad)
        )
    except Exception as exc:
        print(
            json.dumps(
                {"state": "V2_FAST_PREFLIGHT_BLOCKED", "reason": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    if args.output is not None:
        _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("target_met", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
