from __future__ import annotations

import argparse
from pathlib import Path

import yaml


DEFAULT_DATA = Path("D:/yolo/dataset/elevator_buttons/elevator_buttons_seg.yaml")
EXPECTED_NAMES = {
    0: "floor_button",
    1: "door_open",
    2: "door_close",
    3: "alarm",
    4: "call_up",
    5: "call_down",
    6: "emergency",
    7: "other_button",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a prepared YOLO segmentation dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_dataset(data_path: Path) -> list[str]:
    issues: list[str] = []
    if not data_path.exists():
        return [f"missing data yaml: {data_path}"]
    data = load_yaml(data_path)
    root = Path(data.get("path", data_path.parent))
    names = {int(key): value for key, value in dict(data.get("names", {})).items()}
    if names != EXPECTED_NAMES:
        issues.append("class names do not match expected elevator button classes")
    for split in ("train", "val", "test"):
        image_dir = root / str(data.get(split, f"images/{split}"))
        label_dir = root / "labels" / split
        if not image_dir.exists():
            issues.append(f"missing image directory: {image_dir}")
            continue
        if not label_dir.exists():
            issues.append(f"missing label directory: {label_dir}")
            continue
        for image_path in image_dir.glob("*.jpg"):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                issues.append(f"missing label for {image_path}")
    return issues


def main() -> int:
    args = parse_args()
    issues = validate_dataset(args.data)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1
    print(f"Dataset OK: {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
