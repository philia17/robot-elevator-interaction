#!/usr/bin/env python3
"""Solve and validate a fixed-contact hand_tcp -> fingertip pivot session.

This is offline only.  It never publishes TF or commands a robot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def rotation_xyzw(quaternion) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def load_samples(session_dir: Path) -> list[dict]:
    samples = []
    for path in sorted((session_dir / "contact_samples").glob("C*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if not document.get("operator_contact_confirmed") or not document.get("base_fixed_confirmed"):
            continue
        transform = document["T_roarm_base_link_hand_tcp"]
        samples.append(
            {
                "id": document["sample_id"],
                "split": document["split"],
                "R": rotation_xyzw(transform["rotation_xyzw"]),
                "t": np.asarray(transform["translation_m"], dtype=float),
            }
        )
    return samples


def solve(train: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(train) < 3:
        raise ValueError("need at least three train contact samples")
    blocks, rhs = [], []
    for sample in train:
        blocks.append(np.hstack((sample["R"], -np.eye(3))))
        rhs.append(-sample["t"])
    matrix = np.vstack(blocks)
    vector = np.hstack(rhs)
    solution, _residuals, rank, singular = np.linalg.lstsq(matrix, vector, rcond=None)
    diff = np.vstack([sample["R"] - train[0]["R"] for sample in train[1:]])
    diff_singular = np.linalg.svd(diff, compute_uv=False)
    condition = math.inf if diff_singular[-1] <= 1e-12 else float(diff_singular[0] / diff_singular[-1])
    diagnostics = {
        "stack_rank": int(rank),
        "stack_singular_values": singular.tolist(),
        "rotation_difference_rank": int(np.linalg.matrix_rank(diff)),
        "rotation_difference_singular_values": diff_singular.tolist(),
        "rotation_difference_condition": condition,
    }
    return solution[:3], solution[3:], diagnostics


def residuals(samples: list[dict], hand_tip: np.ndarray, base_dot: np.ndarray) -> list[dict]:
    output = []
    for sample in samples:
        vector = sample["R"] @ hand_tip + sample["t"] - base_dot
        output.append({"sample_id": sample["id"], "vector_m": vector.tolist(), "norm_mm": float(np.linalg.norm(vector) * 1000.0)})
    return output


def summary(values: list[dict]) -> dict:
    norms = np.asarray([item["norm_mm"] for item in values], dtype=float)
    if not len(norms):
        return {"count": 0}
    return {
        "count": int(len(norms)), "mean_mm": float(np.mean(norms)),
        "rms_mm": float(np.sqrt(np.mean(norms ** 2))), "median_mm": float(np.median(norms)),
        "p95_mm": float(np.percentile(norms, 95)), "max_mm": float(np.max(norms)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads((args.session_dir / "session_manifest.json").read_text(encoding="utf-8"))
    samples = load_samples(args.session_dir)
    train = [sample for sample in samples if sample["split"] == "train"]
    held_out = [sample for sample in samples if sample["split"] == "held_out"]
    hand_tip, base_dot, diagnostics = solve(train)
    train_residuals = residuals(train, hand_tip, base_dot)
    held_out_residuals = residuals(held_out, hand_tip, base_dot)
    result = {
        "schema": "fixed_contact_pivot_result_v1",
        "purpose": "offline diagnostic hand_tcp to fingertip pivot solve; does not publish TF",
        "session_manifest_sha256": __import__("hashlib").sha256((args.session_dir / "session_manifest.json").read_bytes()).hexdigest(),
        "calibration_id": manifest.get("calibration_id"),
        "T_hand_tcp_fingertip_translation_m": hand_tip.tolist(),
        "fixed_black_dot_base_m": base_dot.tolist(),
        "observability": diagnostics,
        "train": {"residuals": train_residuals, "summary": summary(train_residuals)},
        "held_out": {"residuals": held_out_residuals, "summary": summary(held_out_residuals)},
        "acceptance": {
            "status": "diagnostic_only",
            "reason": "thresholds must be frozen from repeatability data; no TF publication or execution authorization",
        },
    }
    output = args.output or args.session_dir / "pivot_result.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "train": result["train"]["summary"], "held_out": result["held_out"]["summary"], "observability": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
