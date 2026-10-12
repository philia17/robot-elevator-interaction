"""Promote complete AprilTag axis captures to formal evidence only on PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .apriltag_axis_core import CAPTURE_FILES, evaluate_axis_captures
from .apriltag_pose_sampler import _write_json_atomic


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-id", required=True)
    parsed = parser.parse_args(args)
    input_dir = parsed.input_dir.expanduser().resolve()
    output = parsed.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing evidence: {output}")

    documents = {}
    sources = {}
    missing = []
    for label, filename in CAPTURE_FILES.items():
        path = input_dir / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        documents[label] = json.loads(path.read_text(encoding="utf-8"))
        sources[label] = {"path": str(path), "sha256": _sha256(path)}
    if missing:
        raise SystemExit(f"formal evidence not created; missing inputs: {missing}")

    try:
        result = evaluate_axis_captures(documents)
    except ValueError as exc:
        raise SystemExit(f"formal evidence not created; validation error: {exc}") from exc
    if result["status"] != "pass":
        print(json.dumps(result["checks"], indent=2, sort_keys=True))
        raise SystemExit("formal evidence not created; one or more axis checks failed")

    formal = {
        "schema_version": 1,
        "calibration_id": parsed.calibration_id,
        "method": "apriltag_only_paired_static_observations",
        "target": {"family": "tag36h11", "id": 0, "edge_m": 0.064},
        "sources": sources,
        "result": result,
    }
    _write_json_atomic(output, formal)
    print(f"PHASE2_APRILTAG_AXIS PASS output={output}")


if __name__ == "__main__":
    main()
