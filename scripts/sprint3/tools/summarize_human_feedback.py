#!/usr/bin/env python3
"""Summarize auto_match_report.json with image-level human label overrides."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEM_RE = re.compile(r"^(?P<building>[a-z]+)_(?P<location>.+?)_\d+_rgb$", re.I)
SCORING_LABELS = {"tp", "fp", "fn"}
EXCLUDED_LABELS = {"ambiguous", "ignore"}
VALID_HUMAN_LABELS = SCORING_LABELS | EXCLUDED_LABELS


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ambiguous: int = 0
    ignore: int = 0


@dataclass
class ImageFeedback:
    path: Path
    image_stem: str
    run_id: str
    status: str
    building: str
    location: str
    auto: MetricCounts = field(default_factory=MetricCounts)
    corrected: MetricCounts = field(default_factory=MetricCounts)
    reviewed_overrides: int = 0
    changed_overrides: int = 0
    unreviewed_needs_review: int = 0
    fp_reasons: list[str] = field(default_factory=list)
    fn_reasons: list[str] = field(default_factory=list)
    human_reasons: list[str] = field(default_factory=list)
    transitions: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize auto_match_report.json into markdown")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Review bundle root containing auto_match_report.json",
    )
    p.add_argument("--output", type=Path, required=True, help="Output markdown path")
    return p.parse_args()


def parse_stem_tags(stem: str) -> tuple[str, str]:
    match = STEM_RE.match(stem)
    if match:
        return match.group("building").lower(), match.group("location").lower()
    return "unknown", "unknown"


def load_metadata_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    meta_dir = run_dir / "metadata"
    if not meta_dir.is_dir():
        return index

    for meta_path in sorted(meta_dir.glob("*.jsonl")):
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = str(record.get("filename") or "")
            stem = Path(filename).stem if filename else ""
            if stem:
                index[stem] = record
    return index


def count_contract_images(run_dir: Path) -> int:
    return len(list(run_dir.glob("*_contract.json")))


def pct(count: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{100.0 * count / total:.1f}%"


def precision(tp: int, fp: int) -> str:
    denom = tp + fp
    if denom <= 0:
        return "-"
    return f"{100.0 * tp / denom:.1f}%"


def recall(tp: int, fn: int) -> str:
    denom = tp + fn
    if denom <= 0:
        return "-"
    return f"{100.0 * tp / denom:.1f}%"


def human_label(entry: dict[str, Any], auto_label: str) -> str:
    raw = entry.get("human_label")
    if raw is None:
        return auto_label
    label = str(raw).strip().lower()
    if label in VALID_HUMAN_LABELS:
        return label
    return auto_label


def add_label_count(counts: MetricCounts, label: str) -> None:
    if label == "tp":
        counts.tp += 1
    elif label == "fp":
        counts.fp += 1
    elif label == "fn":
        counts.fn += 1
    elif label == "ambiguous":
        counts.ambiguous += 1
    elif label == "ignore":
        counts.ignore += 1


def collect_image_counts(result: dict[str, Any]) -> tuple[
    MetricCounts,
    MetricCounts,
    Counter[str],
    int,
    int,
    int,
    list[str],
]:
    auto = MetricCounts(
        tp=len(result.get("tp", [])),
        fp=len(result.get("fp", [])),
        fn=len(result.get("fn", [])),
    )
    corrected = MetricCounts()
    transitions: Counter[str] = Counter()
    reviewed = 0
    changed = 0
    unreviewed_needs_review = 0
    human_reasons: list[str] = []

    for auto_label_name in ("tp", "fp", "fn"):
        for entry in result.get(auto_label_name, []):
            final_label = human_label(entry, auto_label_name)
            add_label_count(corrected, final_label)

            if entry.get("human_label") is not None:
                reviewed += 1
                transitions[f"{auto_label_name}->{final_label}"] += 1
                if final_label != auto_label_name:
                    changed += 1
                reason = str(entry.get("human_reason") or "").strip()
                if reason:
                    human_reasons.append(reason)
            elif auto_label_name in {"fp", "fn"} and entry.get("reason") == "needs_review":
                unreviewed_needs_review += 1

    return auto, corrected, transitions, reviewed, changed, unreviewed_needs_review, human_reasons


def load_match_report(
    run_dir: Path,
    report: dict[str, Any],
    metadata_index: dict[str, dict[str, Any]],
) -> list[ImageFeedback]:
    run_id = str(report.get("run_id") or run_dir.name)
    per_image = report.get("per_image") or {}
    items: list[ImageFeedback] = []

    for stem in sorted(per_image):
        result = per_image[stem]
        building, location = parse_stem_tags(stem)
        metadata = metadata_index.get(stem, {})
        if building == "unknown" and metadata.get("building"):
            building = str(metadata["building"]).lower()
        if location == "unknown":
            location = str(metadata.get("location") or metadata.get("view") or "unknown").lower()

        auto, corrected, transitions, reviewed, changed, unreviewed, human_reasons = (
            collect_image_counts(result)
        )
        fp_reasons = [str(entry.get("reason") or "needs_review") for entry in result.get("fp", [])]
        fn_reasons = [str(entry.get("reason") or "needs_review") for entry in result.get("fn", [])]
        status = "has_overrides" if reviewed else "auto_only"

        items.append(
            ImageFeedback(
                path=run_dir / "auto_match_report.json",
                image_stem=stem,
                run_id=run_id,
                status=status,
                building=building,
                location=location,
                auto=auto,
                corrected=corrected,
                reviewed_overrides=reviewed,
                changed_overrides=changed,
                unreviewed_needs_review=unreviewed,
                fp_reasons=fp_reasons,
                fn_reasons=fn_reasons,
                human_reasons=human_reasons,
                transitions=transitions,
            )
        )

    return items


def sum_counts(items: list[ImageFeedback], attr: str) -> MetricCounts:
    total = MetricCounts()
    for item in items:
        counts: MetricCounts = getattr(item, attr)
        total.tp += counts.tp
        total.fp += counts.fp
        total.fn += counts.fn
        total.ambiguous += counts.ambiguous
        total.ignore += counts.ignore
    return total


def top_images_by_count(
    items: list[ImageFeedback],
    *,
    kind: str,
    n: int = 5,
) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for item in items:
        count = getattr(item.corrected, kind)
        if count > 0:
            rows.append((item.image_stem, count))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:n]


def render_metric_line(label: str, counts: MetricCounts) -> str:
    extra = ""
    if counts.ambiguous or counts.ignore:
        extra = f" | ambiguous={counts.ambiguous} | ignore={counts.ignore}"
    return (
        f"- **{label}**: TP={counts.tp} | FP={counts.fp} | FN={counts.fn}"
        f"{extra} | Precision={precision(counts.tp, counts.fp)} | Recall={recall(counts.tp, counts.fn)}"
    )


def render_reason_table(title: str, reason_counts: Counter[str]) -> list[str]:
    lines = [f"## {title}", ""]
    total = sum(reason_counts.values())
    if total <= 0:
        lines.extend(["_No data_", ""])
        return lines

    lines.extend(["| reason | count | pct |", "|--------|------:|----:|"])
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} | {pct(count, total)} |")
    lines.append("")
    return lines


def build_report(run_dir: Path, items: list[ImageFeedback], total_images: int) -> str:
    run_dir = run_dir.resolve()
    run_id = items[0].run_id if items else run_dir.name
    auto_total = sum_counts(items, "auto")
    corrected_total = sum_counts(items, "corrected")
    images_with_overrides = sum(1 for item in items if item.reviewed_overrides)
    reviewed_overrides = sum(item.reviewed_overrides for item in items)
    changed_overrides = sum(item.changed_overrides for item in items)
    unreviewed_needs_review = sum(item.unreviewed_needs_review for item in items)

    transitions: Counter[str] = Counter()
    fp_reason_total: Counter[str] = Counter()
    fn_reason_total: Counter[str] = Counter()
    human_reason_total: Counter[str] = Counter()
    for item in items:
        transitions.update(item.transitions)
        fp_reason_total.update(item.fp_reasons)
        fn_reason_total.update(item.fn_reasons)
        human_reason_total.update(item.human_reasons)

    by_building_location: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "images": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ambiguous": 0,
            "ignore": 0,
        }
    )
    for item in items:
        key = (item.building, item.location)
        bucket = by_building_location[key]
        bucket["images"] += 1
        bucket["tp"] += item.corrected.tp
        bucket["fp"] += item.corrected.fp
        bucket["fn"] += item.corrected.fn
        bucket["ambiguous"] += item.corrected.ambiguous
        bucket["ignore"] += item.corrected.ignore

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Human Review Summary",
        "",
        f"- **Bundle**: `{run_dir}`",
        f"- **Run ID**: `{run_id}`",
        f"- **Generated**: {now}",
        f"- **Images in report**: {len(items)}",
        f"- **Contract images**: {total_images}",
        "",
        "## Overall Metrics",
        "",
        render_metric_line("Auto", auto_total),
        render_metric_line("Corrected", corrected_total),
        f"- **Images with overrides**: {images_with_overrides} / {len(items)}",
        f"- **Reviewed overrides**: {reviewed_overrides}",
        f"- **Label-changing overrides**: {changed_overrides}",
        f"- **Unreviewed needs_review entries**: {unreviewed_needs_review}",
        "",
        "## Correction Breakdown",
        "",
    ]

    changed_transitions = {
        transition: count
        for transition, count in transitions.items()
        if transition.split("->", 1)[0] != transition.split("->", 1)[1]
    }
    if changed_transitions:
        lines.extend(["| auto_label -> human_label | count |", "|--------------------------|------:|"])
        for transition, count in sorted(changed_transitions.items()):
            lines.append(f"| {transition} | {count} |")
    else:
        lines.append("_No overrides changed labels._")
    lines.append("")

    if transitions:
        lines.extend(["## All Reviewed Overrides", "", "| auto_label -> human_label | count |", "|--------------------------|------:|"])
        for transition, count in sorted(transitions.items()):
            lines.append(f"| {transition} | {count} |")
        lines.append("")

    lines.extend(
        [
            "## Corrected By Building And Location",
            "",
            "| building | location | images | TP | FP | FN | ambiguous | ignore | Precision | Recall |",
            "|----------|----------|-------:|---:|---:|---:|----------:|-------:|----------:|-------:|",
        ]
    )
    for building, location in sorted(by_building_location):
        bucket = by_building_location[(building, location)]
        lines.append(
            f"| {building} | {location} | {bucket['images']} | {bucket['tp']} | "
            f"{bucket['fp']} | {bucket['fn']} | {bucket['ambiguous']} | {bucket['ignore']} | "
            f"{precision(bucket['tp'], bucket['fp'])} | {recall(bucket['tp'], bucket['fn'])} |"
        )
    lines.append("")

    lines.extend(render_reason_table("Human Reason Breakdown", human_reason_total))
    lines.extend(render_reason_table("Auto FP Reason Breakdown", fp_reason_total))
    lines.extend(render_reason_table("Auto FN Reason Breakdown", fn_reason_total))

    lines.extend(["## Top 5 Corrected FP Images", ""])
    top_fp = top_images_by_count(items, kind="fp")
    if top_fp:
        for stem, count in top_fp:
            lines.append(f"- `{stem}` - FP={count}")
    else:
        lines.append("_No corrected FP entries_")
    lines.append("")

    lines.extend(["## Top 5 Corrected FN Images", ""])
    top_fn = top_images_by_count(items, kind="fn")
    if top_fn:
        for stem, count in top_fn:
            lines.append(f"- `{stem}` - FN={count}")
    else:
        lines.append("_No corrected FN entries_")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report_path = run_dir / "auto_match_report.json"

    if not report_path.is_file():
        raise SystemExit(f"auto_match_report.json not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata_index = load_metadata_index(run_dir)
    items = load_match_report(run_dir, report, metadata_index)
    if not items:
        raise SystemExit(f"no per_image entries in {report_path}")

    total_images = count_contract_images(run_dir) or len(items)
    markdown = build_report(run_dir, items, total_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    auto_total = sum_counts(items, "auto")
    corrected_total = sum_counts(items, "corrected")
    print(f"Wrote {args.output} ({len(items)} image(s), {total_images} total contract images)")
    print(
        "Auto: "
        f"tp={auto_total.tp} fp={auto_total.fp} fn={auto_total.fn} "
        f"precision={precision(auto_total.tp, auto_total.fp)} "
        f"recall={recall(auto_total.tp, auto_total.fn)}"
    )
    print(
        "Corrected: "
        f"tp={corrected_total.tp} fp={corrected_total.fp} fn={corrected_total.fn} "
        f"ambiguous={corrected_total.ambiguous} ignore={corrected_total.ignore} "
        f"precision={precision(corrected_total.tp, corrected_total.fp)} "
        f"recall={recall(corrected_total.tp, corrected_total.fn)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
