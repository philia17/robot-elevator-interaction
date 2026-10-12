#!/usr/bin/env python3
"""Benchmark YOLO-World and Grounding DINO on raw_capture images with contract JSON validation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detector_contract import (  # noqa: E402
    BACKEND_GROUNDING_DINO,
    BACKEND_YOLO_WORLD,
    Detection,
    DetectionArray,
    validate_detection_array,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STEM_RE = re.compile(r"^(?P<building>[a-z]+)_(?P<location>[a-z]+)_\d+_rgb$", re.I)

DEFAULT_YOLO_PYTHON = "/data/venvs/yolo_zeroshot/bin/python3"
DEFAULT_YOLO_DETECTOR = Path.home() / "scripts/sprint3/detect_panel_then_buttons.py"
DEFAULT_GDINO_DETECTOR = Path.home() / "scripts/sprint2/gdino_detect.py"
DEFAULT_GDINO_PYTHON = "python3"

BACKEND_ALIASES = {
    "yolo_world": BACKEND_YOLO_WORLD,
    "grounding_dino": BACKEND_GROUNDING_DINO,
    "gdino": BACKEND_GROUNDING_DINO,
}


@dataclass
class ImageRecord:
    path: Path
    stem: str
    building: str
    location: str
    session_dir: Path
    metadata: dict = field(default_factory=dict)


@dataclass
class ImageResult:
    record: ImageRecord
    ok: bool
    detection_count: int = 0
    avg_score: float = 0.0
    detector_latency_ms: float = 0.0
    contract_path: Path | None = None
    overlay_path: Path | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark detector contract outputs on raw_capture")
    p.add_argument("--backend", required=True, choices=sorted(BACKEND_ALIASES),
                   help="Detector backend: yolo_world or grounding_dino")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Root containing *_session/images/*.jpg")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Parent directory for run_YYYYMMDD_HHMM_{backend}/ outputs")
    p.add_argument("--building", default=None,
                   help="Only include images whose filename building prefix matches (e.g. pmb)")
    p.add_argument("--session", default=None,
                   help="Only include one session directory name (e.g. pmb_20260705_1517_session)")
    p.add_argument("--yolo-python", default=DEFAULT_YOLO_PYTHON)
    p.add_argument("--yolo-detector", type=Path, default=DEFAULT_YOLO_DETECTOR)
    p.add_argument("--gdino-python", default=DEFAULT_GDINO_PYTHON)
    p.add_argument("--gdino-detector", type=Path, default=DEFAULT_GDINO_DETECTOR)
    return p.parse_args()


def normalize_backend(name: str) -> str:
    return BACKEND_ALIASES[name]


def parse_stem_tags(stem: str) -> tuple[str, str]:
    m = STEM_RE.match(stem)
    if not m:
        return "unknown", "unknown"
    return m.group("building").lower(), m.group("location").lower()


def load_metadata_index(data_root: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for session_dir in sorted(data_root.glob("*_session")):
        meta_path = session_dir / "metadata.jsonl"
        if not meta_path.is_file():
            continue
        for line_no, line in enumerate(meta_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN: skip {meta_path}:{line_no}: {exc}", file=sys.stderr)
                continue
            filename = rec.get("filename", "")
            stem = Path(filename).stem if filename else ""
            if stem:
                index[stem] = rec
    return index


def discover_images(
    data_root: Path,
    *,
    building: str | None = None,
    session: str | None = None,
    metadata_index: dict[str, dict] | None = None,
) -> list[ImageRecord]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data-root not found: {data_root}")

    metadata_index = metadata_index or {}
    records: list[ImageRecord] = []

    session_dirs = sorted(data_root.glob("*_session"))
    if session:
        session_dirs = [data_root / session]

    for session_dir in session_dirs:
        if not session_dir.is_dir():
            if session:
                raise FileNotFoundError(f"session not found: {session_dir}")
            continue
        images_dir = session_dir / "images"
        if not images_dir.is_dir():
            continue
        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = img_path.stem
            bld, loc = parse_stem_tags(stem)
            if building and bld != building.lower():
                continue
            meta = dict(metadata_index.get(stem, {}))
            if bld == "unknown" and meta.get("building"):
                bld = str(meta["building"]).lower()
            if loc == "unknown" and meta.get("location"):
                loc = str(meta["location"]).lower()
            records.append(ImageRecord(
                path=img_path,
                stem=stem,
                building=bld,
                location=loc,
                session_dir=session_dir,
                metadata=meta,
            ))
    return records


def allocate_run_dir(output_dir: Path, backend: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"run_{stamp}_{backend}"
    run_dir = output_dir / base
    suffix = 1
    while run_dir.exists():
        run_dir = output_dir / f"{base}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def detection_from_dict(d: dict) -> Detection:
    return Detection(
        detection_id=d["id"],
        bbox_xyxy=tuple(d["bbox_xyxy"]),
        center_xy=tuple(d["center_xy"]),
        score=float(d["score"]),
        coarse_type=d["coarse_type"],
        detector_class=d.get("detector_class", ""),
        raw=dict(d.get("raw", {})),
    )


def detection_array_from_dict(d: dict) -> DetectionArray:
    return DetectionArray(
        schema_version=d["schema_version"],
        source_detector=d["source_detector"],
        image_path=d["image"]["path"],
        image_width=int(d["image"]["width"]),
        image_height=int(d["image"]["height"]),
        detector_latency_ms=float(d["detector_latency_ms"]),
        detections=[detection_from_dict(det) for det in d["detections"]],
    )


def load_and_validate_contract(path: Path) -> tuple[DetectionArray, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    det_array = detection_array_from_dict(payload)
    return det_array, validate_detection_array(det_array)


def run_yolo_world(
    python: str,
    detector: Path,
    image: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    cmd = [
        python, str(detector),
        "--image", str(image),
        "--run-dir", str(run_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}")

    contract_path = run_dir / f"{image.stem}_contract.json"
    overlay_path = run_dir / f"{image.stem}_overlay.jpg"
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing contract: {contract_path}")
    return contract_path, overlay_path


def run_grounding_dino(
    python: str,
    detector: Path,
    image: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    with tempfile.TemporaryDirectory(prefix="gdino_one_") as tmp:
        tmp_dir = Path(tmp)
        shutil.copy2(image, tmp_dir / image.name)
        cmd = [python, str(detector), "--image-dir", str(tmp_dir), "--output", str(run_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}")

    contract_path = run_dir / f"{image.stem}_contract.json"
    annotated = run_dir / f"{image.stem}_annotated.png"
    overlay_path = run_dir / f"{image.stem}_overlay.jpg"
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing contract: {contract_path}")
    if annotated.is_file():
        shutil.copy2(annotated, overlay_path)
    return contract_path, overlay_path


def process_image(
    backend: str,
    args: argparse.Namespace,
    record: ImageRecord,
    run_dir: Path,
) -> ImageResult:
    try:
        if backend == BACKEND_YOLO_WORLD:
            contract_path, overlay_path = run_yolo_world(
                args.yolo_python, args.yolo_detector, record.path, run_dir,
            )
        else:
            contract_path, overlay_path = run_grounding_dino(
                args.gdino_python, args.gdino_detector, record.path, run_dir,
            )

        det_array, errors = load_and_validate_contract(contract_path)
        if errors:
            raise RuntimeError("contract validation failed: " + "; ".join(errors))

        scores = [float(d.score) for d in det_array.detections if float(d.score) >= 0.0]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return ImageResult(
            record=record,
            ok=True,
            detection_count=len(det_array.detections),
            avg_score=avg_score,
            detector_latency_ms=float(det_array.detector_latency_ms),
            contract_path=contract_path,
            overlay_path=overlay_path if overlay_path.is_file() else None,
        )
    except Exception as exc:  # noqa: BLE001 — collect per-image failures
        return ImageResult(record=record, ok=False, error=str(exc))


def bucket_detection_count(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 10:
        return "6-10"
    return "11+"


def bucket_score(score: float) -> str:
    if score < 0.1:
        return "0-0.1"
    if score < 0.3:
        return "0.1-0.3"
    if score < 0.5:
        return "0.3-0.5"
    return "0.5+"


def bucket_distance_cm(value: object) -> str:
    if value is None:
        return "unknown"
    try:
        cm = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if cm < 50:
        return "<50"
    if cm <= 100:
        return "50-100"
    return "100+"


def metadata_value(meta: dict, key: str) -> str:
    val = meta.get(key)
    if val is None or val == "":
        return "unknown"
    return str(val)


def write_benchmark_summary(
    run_dir: Path,
    backend: str,
    results: list[ImageResult],
    *,
    wall_s: float,
    data_root: Path,
) -> Path:
    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    total_detections = sum(r.detection_count for r in ok_results)

    group_stats: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "images": 0,
        "total_detections": 0,
        "score_sum": 0.0,
        "score_count": 0,
        "latency_sum": 0.0,
    })

    det_count_dist: dict[str, int] = defaultdict(int)
    score_dist: dict[str, int] = defaultdict(int)
    distance_dist: dict[str, int] = defaultdict(int)
    button_lit_dist: dict[str, int] = defaultdict(int)

    all_scores: list[tuple[str, float]] = []
    det_count_rank: list[tuple[str, int]] = []

    for r in ok_results:
        key = (r.record.building, r.record.location)
        g = group_stats[key]
        g["images"] += 1
        g["total_detections"] += r.detection_count
        g["latency_sum"] += r.detector_latency_ms
        det_count_dist[bucket_detection_count(r.detection_count)] += 1
        det_count_rank.append((r.record.stem, r.detection_count))

        if r.detection_count > 0:
            g["score_sum"] += r.avg_score * r.detection_count
            g["score_count"] += r.detection_count

        contract_path = r.contract_path
        if contract_path and contract_path.is_file():
            det_array, _ = load_and_validate_contract(contract_path)
            for det in det_array.detections:
                if float(det.score) >= 0.0:
                    score_dist[bucket_score(float(det.score))] += 1
                    all_scores.append((r.record.stem, float(det.score)))

        dist_bucket = bucket_distance_cm(r.record.metadata.get("distance_cm"))
        distance_dist[dist_bucket] += 1
        lit_bucket = metadata_value(r.record.metadata, "button_lit")
        button_lit_dist[lit_bucket] += 1

    lines = [
        f"# Detector contract benchmark — {backend}",
        "",
        f"- **Backend**: `{backend}`",
        f"- **Data root**: `{data_root}`",
        f"- **Run directory**: `{run_dir}`",
        f"- **Total images**: {len(results)}",
        f"- **Succeeded**: {len(ok_results)}",
        f"- **Failed**: {len(fail_results)}",
        f"- **Wall time**: {wall_s:.1f}s",
        f"- **Total detections**: {total_detections}",
        "",
        "## By building + location",
        "",
        "| building | location | images | total_detections | avg_per_image | avg_score | avg_latency_ms |",
        "|----------|----------|--------|------------------|---------------|-----------|----------------|",
    ]

    for (building, location) in sorted(group_stats):
        g = group_stats[(building, location)]
        avg_per_image = g["total_detections"] / g["images"] if g["images"] else 0.0
        avg_score = g["score_sum"] / g["score_count"] if g["score_count"] else 0.0
        avg_latency = g["latency_sum"] / g["images"] if g["images"] else 0.0
        lines.append(
            f"| {building} | {location} | {g['images']} | {g['total_detections']} "
            f"| {avg_per_image:.2f} | {avg_score:.3f} | {avg_latency:.1f} |"
        )

    lines.extend(["", "## Detection count distribution (images)", ""])
    for label in ("0", "1-5", "6-10", "11+"):
        lines.append(f"- **{label}**: {det_count_dist.get(label, 0)}")

    lines.extend(["", "## Score distribution (detections)", ""])
    for label in ("0-0.1", "0.1-0.3", "0.3-0.5", "0.5+"):
        lines.append(f"- **{label}**: {score_dist.get(label, 0)}")

    lines.extend(["", "## Metadata-linked stats", ""])
    lines.append("### By distance_cm")
    for label in ("<50", "50-100", "100+", "unknown"):
        lines.append(f"- **{label}**: {distance_dist.get(label, 0)}")
    lines.append("")
    lines.append("### By button_lit")
    for label in sorted(button_lit_dist):
        lines.append(f"- **{label}**: {button_lit_dist[label]}")

    lines.extend(["", "## Notable images", ""])
    if det_count_rank:
        det_count_rank.sort(key=lambda t: t[1])
        low_stem, low_n = det_count_rank[0]
        high_stem, high_n = det_count_rank[-1]
        lines.append(f"- **Lowest detection count**: `{low_stem}` ({low_n})")
        lines.append(f"- **Highest detection count**: `{high_stem}` ({high_n})")
    if all_scores:
        all_scores.sort(key=lambda t: t[1])
        low_s_stem, low_s = all_scores[0]
        high_s_stem, high_s = all_scores[-1]
        lines.append(f"- **Lowest score**: `{low_s_stem}` ({low_s:.4f})")
        lines.append(f"- **Highest score**: `{high_s_stem}` ({high_s:.4f})")

    if fail_results:
        lines.extend(["", "## Failures", ""])
        for r in fail_results:
            lines.append(f"- `{r.record.path.name}`: {r.error}")

    out = run_dir / "benchmark_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_run_meta(
    run_dir: Path,
    backend: str,
    data_root: Path,
    results: list[ImageResult],
    *,
    wall_s: float,
    args: argparse.Namespace,
) -> Path:
    ok_results = [r for r in results if r.ok]
    meta = {
        "backend": backend,
        "data_root": str(data_root),
        "output_dir": str(run_dir.parent),
        "run_dir": str(run_dir),
        "created_at": datetime.now().astimezone().isoformat(),
        "wall_time_s": round(wall_s, 3),
        "image_count": len(results),
        "success_count": len(ok_results),
        "fail_count": len(results) - len(ok_results),
        "total_detections": sum(r.detection_count for r in ok_results),
        "filters": {
            "building": args.building,
            "session": args.session,
        },
        "commands": {
            "yolo_python": args.yolo_python,
            "yolo_detector": str(args.yolo_detector),
            "gdino_python": args.gdino_python,
            "gdino_detector": str(args.gdino_detector),
        },
        "failures": [
            {"image": str(r.record.path), "error": r.error}
            for r in results if not r.ok
        ],
    }
    out = run_dir / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    backend = normalize_backend(args.backend)
    metadata_index = load_metadata_index(args.data_root)
    images = discover_images(
        args.data_root,
        building=args.building,
        session=args.session,
        metadata_index=metadata_index,
    )
    if not images:
        raise SystemExit("No images matched the scan filters.")

    run_dir = allocate_run_dir(args.output_dir, backend)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Images to process: {len(images)}", flush=True)

    t0 = time.time()
    results: list[ImageResult] = []
    for idx, record in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {record.path.name} ...", flush=True)
        result = process_image(backend, args, record, run_dir)
        results.append(result)
        if result.ok:
            print(
                f"  ok: detections={result.detection_count} "
                f"avg_score={result.avg_score:.3f} latency_ms={result.detector_latency_ms:.1f}",
                flush=True,
            )
        else:
            print(f"  FAIL: {result.error}", flush=True)

    wall_s = time.time() - t0
    summary_path = write_benchmark_summary(run_dir, backend, results, wall_s=wall_s, data_root=args.data_root)
    meta_path = write_run_meta(run_dir, backend, args.data_root, results, wall_s=wall_s, args=args)

    ok_count = sum(1 for r in results if r.ok)
    contract_count = sum(1 for r in results if r.ok and r.contract_path and r.contract_path.is_file())
    print(json.dumps({
        "backend": backend,
        "run_dir": str(run_dir),
        "images": len(images),
        "success": ok_count,
        "fail_count": len(images) - ok_count,
        "contract_json_count": contract_count,
        "summary": str(summary_path),
        "run_meta": str(meta_path),
        "wall_time_s": round(wall_s, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
