"""Paths and numbered run directories for PMB zero-shot benchmark on AGX."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

IMAGES_DIR = Path("/home/robot/datasets/pmb/zero_shot_benchmark_images")
OUTPUTS_ROOT = Path("/home/robot/datasets/pmb/zero_shot_benchmark_outputs")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def model_slug(model: str) -> str:
    return Path(model).stem.replace("yolov8", "yolo-")


def discover_images(images_dir: Path | None = None) -> list[Path]:
    root = images_dir or IMAGES_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark images dir not found: {root}")
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _existing_run_ids(outputs_root: Path) -> list[int]:
    ids: list[int] = []
    if not outputs_root.is_dir():
        return ids
    for p in outputs_root.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        parts = p.name.split("_")
        if len(parts) < 2:
            continue
        try:
            ids.append(int(parts[1]))
        except ValueError:
            continue
    return ids


def run_dir_name(run_id: int, pipeline: str) -> str:
    return f"run_{run_id:04d}_{pipeline}"


def allocate_run_dir(
    outputs_root: Path,
    pipeline: str,
    *,
    images_dir: Path | None = None,
    extra_meta: dict | None = None,
) -> tuple[int, Path]:
    """Create the next numbered run directory and write run_meta.json."""
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_id = max(_existing_run_ids(outputs_root), default=0) + 1
    run_dir = outputs_root / run_dir_name(run_id, pipeline)
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "run_id": run_id,
        "run_name": run_dir.name,
        "pipeline": pipeline,
        "images_dir": str(images_dir or IMAGES_DIR),
        "outputs_root": str(outputs_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_id, run_dir


def artifact_paths(
    run_dir: Path,
    image_stem: str,
    pipeline: str,
    model: str | None = None,
) -> tuple[Path, Path]:
    """Return (overlay_path, result_path) with rule-based names inside run_dir."""
    parts = [image_stem]
    if model:
        parts.append(model_slug(model))
    parts.append(pipeline)
    base = "_".join(parts)
    return (
        run_dir / f"{base}_overlay.jpg",
        run_dir / f"{base}_result.json",
    )
