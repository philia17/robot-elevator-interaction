#!/usr/bin/env python3
"""Greedy IoU matching between ground-truth boxes and detector contract outputs.

Pairs GT annotations with predictions, classifies TP/FP/FN, and auto-tags
likely failure reasons before human review.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_SPRINT3 = _SCRIPT_DIR.parent
if str(_SPRINT3) not in sys.path:
    sys.path.insert(0, str(_SPRINT3))

from detector_contract import COARSE_TYPES  # noqa: E402

# GT annotations use detector coarse_type values plus legacy labels from annotate_gt v1.
GT_COARSE_TYPES = frozenset(COARSE_TYPES) | {"alarm", "other"}

IOU_THRESHOLD = 0.5
CENTER_DIST_SHIFTED = 30  # px
AREA_RATIO_TOO_BIG = 1.5
AREA_RATIO_TOO_SMALL = 0.5
NO_MATCH_RADIUS = 60  # px, FP with no GT within this radius → needs_review
FN_NO_PRED_RADIUS = 40  # px, FN with no pred within this radius → needs_review


@dataclass
class Box:
    bbox_xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-match GT boxes to contract predictions")
    p.add_argument("--gt-dir", type=Path, required=True, help="Directory containing *_gt.json")
    p.add_argument("--run-dir", type=Path, required=True, help="Review bundle or benchmark run directory")
    p.add_argument("--output", type=Path, required=True, help="Output auto_match_report.json path")
    return p.parse_args()


def box_area(xyxy: tuple[float, float, float, float]) -> float:
    return Box(xyxy).area


def center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax, ay = Box(a).center
    bx, by = Box(b).center
    return math.hypot(ax - bx, ay - by)


def compute_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def round_float(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def bbox_list(xyxy: tuple[float, float, float, float]) -> list[float]:
    return [round_float(v, 1) for v in xyxy]


def load_gt_index(gt_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(gt_dir.glob("*_gt.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stem = payload.get("image_stem") or path.name.replace("_gt.json", "")
        index[stem] = payload
    return index


def load_contract_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*_contract.json")):
        stem = path.name.replace("_contract.json", "")
        index[stem] = json.loads(path.read_text(encoding="utf-8"))
    return index


def resolve_run_id(run_dir: Path) -> str:
    run_id_path = run_dir / "run_id.txt"
    if run_id_path.is_file():
        text = run_id_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return run_dir.name


def greedy_match(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
) -> tuple[list[tuple[int, int, float, float]], list[int], list[int]]:
    """Return (tp_pairs, unmatched_gt_idxs, unmatched_pred_idxs).

    Each tp_pair is (gt_idx, pred_idx, iou, center_dist).
    """
    candidates: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = compute_iou(gt, pred)
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp_pairs: list[tuple[int, int, float, float]] = []

    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        tp_pairs.append((gi, pi, iou, center_distance(gt_boxes[gi], pred_boxes[pi])))

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    return tp_pairs, unmatched_gt, unmatched_pred


def classify_fp(
    pred_box: tuple[float, float, float, float],
    unmatched_gt_boxes: list[tuple[int, tuple[float, float, float, float]]],
) -> dict[str, Any]:
    if not unmatched_gt_boxes:
        return {"reason": "needs_review"}

    nearest_idx, nearest_box, dist = min(
        ((gi, box, center_distance(pred_box, box)) for gi, box in unmatched_gt_boxes),
        key=lambda item: item[2],
    )
    if dist > NO_MATCH_RADIUS:
        return {"reason": "needs_review"}

    area_ratio = box_area(pred_box) / max(box_area(nearest_box), 1e-6)
    out: dict[str, Any] = {
        "gt_idx": nearest_idx,
        "area_ratio": round_float(area_ratio),
    }
    if dist > CENTER_DIST_SHIFTED:
        out["reason"] = "shifted"
    elif area_ratio > AREA_RATIO_TOO_BIG:
        out["reason"] = "too_big"
    elif area_ratio < AREA_RATIO_TOO_SMALL:
        out["reason"] = "too_small"
    else:
        out["reason"] = "needs_review"
    return out


def classify_fn(
    gt_box: tuple[float, float, float, float],
    pred_boxes: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    if not pred_boxes:
        return {"reason": "needs_review"}

    nearest_pred_idx, nearest_pred, dist = min(
        ((pi, box, center_distance(gt_box, box)) for pi, box in enumerate(pred_boxes)),
        key=lambda item: item[2],
    )
    if dist >= FN_NO_PRED_RADIUS:
        return {"reason": "needs_review"}

    iou = compute_iou(gt_box, nearest_pred)
    if iou < IOU_THRESHOLD:
        return {"reason": "shifted"}
    return {"reason": "needs_review"}


def match_image(
    gt_payload: dict[str, Any],
    contract_payload: dict[str, Any],
) -> dict[str, Any]:
    gt_entries = gt_payload.get("ground_truth", [])
    gt_boxes = [tuple(float(v) for v in entry["bbox_xyxy"]) for entry in gt_entries]

    pred_entries = contract_payload.get("detections", [])
    pred_boxes = [tuple(float(v) for v in det["bbox_xyxy"]) for det in pred_entries]
    pred_ids = [str(det["id"]) for det in pred_entries]

    tp_pairs, unmatched_gt, unmatched_pred = greedy_match(gt_boxes, pred_boxes)

    unmatched_gt_lookup = [(gi, gt_boxes[gi]) for gi in unmatched_gt]

    tp = [
        {
            "gt_idx": gi,
            "pred_id": pred_ids[pi],
            "iou": round_float(iou),
            "center_dist": round_float(dist, 1),
            "gt_bbox": bbox_list(gt_boxes[gi]),
            "prediction_bbox": bbox_list(pred_boxes[pi]),
        }
        for gi, pi, iou, dist in tp_pairs
    ]

    fp: list[dict[str, Any]] = []
    for pi in unmatched_pred:
        entry: dict[str, Any] = {
            "pred_id": pred_ids[pi],
            "prediction_bbox": bbox_list(pred_boxes[pi]),
        }
        entry.update(classify_fp(pred_boxes[pi], unmatched_gt_lookup))
        fp.append(entry)

    fn: list[dict[str, Any]] = []
    for gi in unmatched_gt:
        entry: dict[str, Any] = {
            "gt_idx": gi,
            "gt_bbox": bbox_list(gt_boxes[gi]),
        }
        entry.update(classify_fn(gt_boxes[gi], pred_boxes))
        fn.append(entry)

    return {"tp": tp, "fp": fp, "fn": fn}


def build_summary(per_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_gt = total_pred = tp = fp = fn = 0
    fp_by_reason: dict[str, int] = {"too_small": 0, "too_big": 0, "shifted": 0, "needs_review": 0}
    fn_by_reason: dict[str, int] = {"dark": 0, "too_small": 0, "shifted": 0, "needs_review": 0}

    for result in per_image.values():
        total_gt += len(result["tp"]) + len(result["fn"])
        total_pred += len(result["tp"]) + len(result["fp"])
        tp += len(result["tp"])
        fp += len(result["fp"])
        fn += len(result["fn"])
        for item in result["fp"]:
            reason = item.get("reason", "needs_review")
            fp_by_reason[reason] = fp_by_reason.get(reason, 0) + 1
        for item in result["fn"]:
            reason = item.get("reason", "needs_review")
            fn_by_reason[reason] = fn_by_reason.get(reason, 0) + 1

    precision = round(tp / total_pred, 3) if total_pred else 0.0
    recall = round(tp / total_gt, 3) if total_gt else 0.0
    return {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_needs_review": fp_by_reason.get("needs_review", 0),
        "fn_needs_review": fn_by_reason.get("needs_review", 0),
        "precision": precision,
        "recall": recall,
    }, fp_by_reason, fn_by_reason


def main() -> int:
    args = parse_args()
    gt_dir = args.gt_dir.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()

    if not gt_dir.is_dir():
        print(f"error: --gt-dir not found: {gt_dir}", file=sys.stderr)
        return 1
    if not run_dir.is_dir():
        print(f"error: --run-dir not found: {run_dir}", file=sys.stderr)
        return 1

    gt_index = load_gt_index(gt_dir)
    contract_index = load_contract_index(run_dir)
    common_stems = sorted(set(gt_index) & set(contract_index))
    if not common_stems:
        print("error: no overlapping image_stem between GT and contract files", file=sys.stderr)
        return 1

    per_image: dict[str, dict[str, Any]] = {}
    for stem in common_stems:
        per_image[stem] = match_image(gt_index[stem], contract_index[stem])

    summary, fp_by_reason, fn_by_reason = build_summary(per_image)
    report = {
        "run_id": resolve_run_id(run_dir),
        "matched_at": date.today().isoformat(),
        "per_image": per_image,
        "summary": summary,
        "fp_by_reason": fp_by_reason,
        "fn_by_reason": fn_by_reason,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(
        f"images={len(per_image)} tp={summary['tp']} fp={summary['fp']} fn={summary['fn']} "
        f"precision={summary['precision']} recall={summary['recall']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
