from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path("D:/yolo/project/Lift_Robot_Project")
RAW_DIR = Path("D:/yolo/dataset/raw")
ANNOTATIONS_DIR = Path("D:/yolo/dataset/annotations")
OUTPUT_DIR = Path("D:/yolo/dataset/elevator_buttons")
CLASS_NAMES = {
    0: "floor_button",
    1: "door_open",
    2: "door_close",
    3: "alarm",
    4: "call_up",
    5: "call_down",
    6: "emergency",
    7: "other_button",
}


@dataclass(frozen=True)
class CaptureRecord:
    rgb_path: Path
    depth_path: Path
    meta_path: Path
    label_path: Path
    building: str
    elevator: str
    sequence: str
    group_key: str
    split: str
    image_sha256: str | None
    issues: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare elevator button captures for YOLO segmentation.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "manifest.csv")
    parser.add_argument("--apply", action="store_true", help="Copy RGB images and labels into YOLO layout.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--group-by", choices=("building", "elevator"), default="building")
    return parser.parse_args()


def read_meta(meta_path: Path) -> dict[str, object]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def is_image_readable(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def infer_from_path(rgb_path: Path, raw_dir: Path) -> tuple[str, str, str]:
    relative_parts = rgb_path.relative_to(raw_dir).parts
    building = relative_parts[0] if len(relative_parts) >= 2 else "unknown_building"
    elevator = relative_parts[1] if len(relative_parts) >= 3 else building
    sequence = rgb_path.stem.removesuffix("_rgb")
    return building, elevator, sequence


def label_for(rgb_path: Path, raw_dir: Path, annotations_dir: Path) -> Path:
    rel = rgb_path.relative_to(raw_dir)
    label_rel = rel.with_name(rgb_path.name.replace("_rgb.jpg", ".txt"))
    return annotations_dir / label_rel


def split_groups(group_keys: Iterable[str], train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, str]:
    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train, val, and test ratios must sum to 1.0")
    keys = sorted(set(group_keys))
    total = len(keys)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if total and train_end == 0:
        train_end = 1
    mapping: dict[str, str] = {}
    for index, key in enumerate(keys):
        if index < train_end:
            mapping[key] = "train"
        elif index < val_end:
            mapping[key] = "val"
        else:
            mapping[key] = "test"
    return mapping


def discover_records(args: argparse.Namespace) -> list[CaptureRecord]:
    rgb_paths = sorted(args.raw_dir.rglob("*_rgb.jpg"))
    partial: list[dict[str, object]] = []
    hashes: dict[str, list[Path]] = {}

    for rgb_path in rgb_paths:
        depth_path = rgb_path.with_name(rgb_path.name.replace("_rgb.jpg", "_depth.png"))
        meta_path = rgb_path.with_name(rgb_path.name.replace("_rgb.jpg", "_meta.json"))
        label_path = label_for(rgb_path, args.raw_dir, args.annotations_dir)
        meta = read_meta(meta_path)
        path_building, path_elevator, path_sequence = infer_from_path(rgb_path, args.raw_dir)
        building = str(meta.get("building") or path_building)
        elevator = str(meta.get("elevator") or meta.get("elevator_id") or path_elevator)
        sequence = str(meta.get("sequence") or meta.get("capture_sequence") or path_sequence)
        group_value = building if args.group_by == "building" else f"{building}/{elevator}"
        image_hash = file_sha256(rgb_path)
        if image_hash:
            hashes.setdefault(image_hash, []).append(rgb_path)
        partial.append(
            {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "meta_path": meta_path,
                "label_path": label_path,
                "building": building,
                "elevator": elevator,
                "sequence": sequence,
                "group_key": group_value,
                "image_sha256": image_hash,
            }
        )

    split_by_group = split_groups((str(item["group_key"]) for item in partial), args.train_ratio, args.val_ratio, args.test_ratio)
    records: list[CaptureRecord] = []
    duplicate_hashes = {digest for digest, paths in hashes.items() if len(paths) > 1}
    for item in partial:
        issues: list[str] = []
        if not Path(item["depth_path"]).exists():
            issues.append("missing_depth")
        if not Path(item["meta_path"]).exists():
            issues.append("missing_meta")
        if not Path(item["label_path"]).exists():
            issues.append("missing_label")
        if item["image_sha256"] in duplicate_hashes:
            issues.append("duplicate_image")
        if not is_image_readable(Path(item["rgb_path"])):
            issues.append("corrupt_rgb")
        group_key = str(item["group_key"])
        records.append(
            CaptureRecord(
                rgb_path=Path(item["rgb_path"]),
                depth_path=Path(item["depth_path"]),
                meta_path=Path(item["meta_path"]),
                label_path=Path(item["label_path"]),
                building=str(item["building"]),
                elevator=str(item["elevator"]),
                sequence=str(item["sequence"]),
                group_key=group_key,
                split=split_by_group.get(group_key, "train"),
                image_sha256=str(item["image_sha256"]) if item["image_sha256"] else None,
                issues=tuple(issues),
            )
        )
    return records


def write_manifest(records: list[CaptureRecord], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "group_key",
                "building",
                "elevator",
                "sequence",
                "rgb_path",
                "depth_path",
                "meta_path",
                "label_path",
                "image_sha256",
                "issues",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record.split,
                    "group_key": record.group_key,
                    "building": record.building,
                    "elevator": record.elevator,
                    "sequence": record.sequence,
                    "rgb_path": record.rgb_path.as_posix(),
                    "depth_path": record.depth_path.as_posix(),
                    "meta_path": record.meta_path.as_posix(),
                    "label_path": record.label_path.as_posix(),
                    "image_sha256": record.image_sha256 or "",
                    "issues": ";".join(record.issues),
                }
            )


def copy_yolo_files(records: list[CaptureRecord], output_dir: Path) -> None:
    for record in records:
        if record.issues:
            continue
        image_dir = output_dir / "images" / record.split
        label_dir = output_dir / "labels" / record.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target_stem = record.rgb_path.stem.removesuffix("_rgb")
        shutil.copy2(record.rgb_path, image_dir / f"{target_stem}.jpg")
        shutil.copy2(record.label_path, label_dir / f"{target_stem}.txt")


def write_dataset_yaml(output_dir: Path) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in CLASS_NAMES.items())
    (output_dir / "elevator_buttons_seg.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = discover_records(args)
    write_manifest(records, args.manifest)
    if args.apply:
        copy_yolo_files(records, args.output_dir)
        write_dataset_yaml(args.output_dir)
    issue_count = sum(1 for record in records if record.issues)
    print(f"records={len(records)}")
    print(f"records_with_issues={issue_count}")
    print(f"manifest={args.manifest}")
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    return 1 if issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
