#!/usr/bin/env python3
"""Build a portable, verified YOLO Detect handoff from a reviewed legacy export.

The source workspace and legacy export are treated as read-only. The resulting
ZIP contains both the canonical legacy JSON/image pairs and a coarse-class
Ultralytics YOLO Detect dataset with building-disjoint splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import yaml

try:
    from .preannotate import COARSE_CLASS_NAMES
except ImportError:
    from preannotate import COARSE_CLASS_NAMES


DEFAULT_SPLIT_BUILDINGS = {
    "train": ("iamet", "ieb", "lib", "pmb"),
    "val": ("db",),
    "test": ("pb",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def image_source(workspace: Path, session: str, gt: dict[str, Any]) -> Path:
    source = gt.get("capture_source", {}).get("image")
    if not source:
        source = gt.get("capture_source", {}).get("raw_metadata", {}).get("filename")
    if not isinstance(source, str) or not source:
        source = f"images/{gt['image_stem']}.jpg"
    candidate = workspace / session / source
    if not candidate.is_file():
        raise FileNotFoundError(f"RGB image missing for {session}/{gt['image_stem']}: {candidate}")
    return candidate


def split_lookup(split_buildings: dict[str, Iterable[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split, buildings in split_buildings.items():
        for building in buildings:
            normalized = building.lower()
            if normalized in result:
                raise ValueError(f"Building appears in multiple splits: {normalized}")
            result[normalized] = split
    return result


def yolo_line(
    box: dict[str, Any],
    width: int,
    height: int,
    class_to_id: dict[str, int],
) -> tuple[str, str]:
    class_name = str(box.get("coarse_type") or "button_candidate")
    if class_name not in class_to_id:
        raise ValueError(f"Unsupported coarse class: {class_name!r}")
    coords = box.get("bbox_xyxy")
    if not isinstance(coords, list) or len(coords) != 4:
        raise ValueError(f"Invalid bbox_xyxy: {coords!r}")
    x1, y1, x2, y2 = (float(value) for value in coords)
    x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
    y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox after clipping: {coords!r}")
    values = (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    if any(value <= 0.0 or value > 1.0 for value in values[2:]):
        raise ValueError(f"Invalid normalized bbox size: {values!r}")
    if any(value < 0.0 or value > 1.0 for value in values[:2]):
        raise ValueError(f"Invalid normalized bbox center: {values!r}")
    line = f"{class_to_id[class_name]} " + " ".join(f"{value:.8f}" for value in values)
    return line, class_name


def collect_records(
    legacy_export: Path,
    workspace: Path,
    building_to_split: dict[str, str],
) -> list[dict[str, Any]]:
    gt_paths = sorted(legacy_export.glob("*/*_gt.json"))
    if not gt_paths:
        raise RuntimeError(f"No legacy GT files found under {legacy_export}")
    records: list[dict[str, Any]] = []
    seen_stems: set[str] = set()
    for gt_path in gt_paths:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        session = gt_path.parent.name
        stem = str(gt.get("image_stem") or gt_path.stem.removesuffix("_gt"))
        if stem in seen_stems:
            raise RuntimeError(f"Duplicate image stem cannot be flattened safely: {stem}")
        seen_stems.add(stem)
        building = str(gt.get("image_meta", {}).get("building") or session.split("_", 1)[0]).lower()
        if building not in building_to_split:
            raise RuntimeError(f"No split mapping for building {building!r} ({gt_path})")
        rgb = image_source(workspace, session, gt)
        image = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode RGB image: {rgb}")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid image dimensions for {rgb}: {width}x{height}")
        records.append(
            {
                "stem": stem,
                "session": session,
                "building": building,
                "split": building_to_split[building],
                "gt_path": gt_path,
                "gt": gt,
                "rgb_path": rgb,
                "width": width,
                "height": height,
            }
        )
    return records


def build_tree(
    root: Path,
    records: list[dict[str, Any]],
    legacy_export: Path,
    split_buildings: dict[str, Iterable[str]],
) -> dict[str, Any]:
    class_names = tuple(COARSE_CLASS_NAMES)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    excluded_source = legacy_export / "excluded_manifest.json"
    if not excluded_source.is_file():
        raise FileNotFoundError(f"Missing excluded_manifest.json: {excluded_source}")
    excluded_payload = json.loads(excluded_source.read_text(encoding="utf-8"))

    total_classes: Counter[str] = Counter()
    split_stats: dict[str, dict[str, Any]] = {
        split: {
            "buildings": list(buildings),
            "images": 0,
            "boxes": 0,
            "class_counts": Counter(),
        }
        for split, buildings in split_buildings.items()
    }
    manifest_records: list[dict[str, Any]] = []

    for split in split_buildings:
        (root / "yolo_detect_coarse" / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "yolo_detect_coarse" / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "legacy_gt" / "images").mkdir(parents=True, exist_ok=True)

    for record in records:
        stem = record["stem"]
        session = record["session"]
        split = record["split"]
        rgb_path = record["rgb_path"]
        gt_path = record["gt_path"]
        width = record["width"]
        height = record["height"]
        suffix = rgb_path.suffix.lower()
        if suffix not in {".jpg", ".jpeg"}:
            raise RuntimeError(f"Expected RGB JPEG for handoff, got: {rgb_path}")

        legacy_image_rel = Path("legacy_gt") / "images" / f"{stem}{suffix}"
        legacy_gt_rel = Path("legacy_gt") / "annotations" / session / gt_path.name
        yolo_image_rel = Path("yolo_detect_coarse") / "images" / split / f"{stem}{suffix}"
        yolo_label_rel = Path("yolo_detect_coarse") / "labels" / split / f"{stem}.txt"
        for relative, source in (
            (legacy_image_rel, rgb_path),
            (legacy_gt_rel, gt_path),
            (yolo_image_rel, rgb_path),
        ):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=True)
            if destination.is_symlink():
                raise RuntimeError(f"Generated handoff must not contain symlinks: {destination}")

        label_lines: list[str] = []
        record_classes: Counter[str] = Counter()
        for box in record["gt"].get("ground_truth", []):
            line, class_name = yolo_line(box, width, height, class_to_id)
            label_lines.append(line)
            record_classes[class_name] += 1
        (root / yolo_label_rel).write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        total_classes.update(record_classes)
        split_stats[split]["images"] += 1
        split_stats[split]["boxes"] += len(label_lines)
        split_stats[split]["class_counts"].update(record_classes)
        manifest_records.append(
            {
                "image_stem": stem,
                "session": session,
                "building": record["building"],
                "split": split,
                "width": width,
                "height": height,
                "boxes": len(label_lines),
                "class_counts": {name: record_classes.get(name, 0) for name in class_names},
                "rgb_sha256": sha256_file(rgb_path),
                "legacy_image": legacy_image_rel.as_posix(),
                "legacy_gt": legacy_gt_rel.as_posix(),
                "yolo_image": yolo_image_rel.as_posix(),
                "yolo_label": yolo_label_rel.as_posix(),
            }
        )

    shutil.copy2(excluded_source, root / "legacy_gt" / "excluded_manifest.json")
    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (root / "yolo_detect_coarse" / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )

    normalized_splits: dict[str, Any] = {}
    for split, stats in split_stats.items():
        normalized_splits[split] = {
            "buildings": stats["buildings"],
            "images": stats["images"],
            "boxes": stats["boxes"],
            "class_counts": {name: stats["class_counts"].get(name, 0) for name in class_names},
        }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "Lift Robot reviewed RGB button annotations",
        "task": "Ultralytics YOLO11n Detect bounding-box detection",
        "class_mode": "coarse",
        "source_export": legacy_export.name,
        "counts": {
            "images": len(records),
            "boxes": sum(total_classes.values()),
            "excluded_images": len(excluded_payload.get("excluded", [])),
            "classes_configured": len(class_names),
            "classes_present": sum(total_classes.get(name, 0) > 0 for name in class_names),
        },
        "classes": [
            {"id": class_to_id[name], "name": name, "boxes": total_classes.get(name, 0)}
            for name in class_names
        ],
        "splits": normalized_splits,
        "records": manifest_records,
    }
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "yolo_detect_coarse" / "manifest.json",
        {
            "schema_version": 1,
            "task": manifest["task"],
            "class_mode": "coarse",
            "counts": manifest["counts"],
            "classes": manifest["classes"],
            "splits": normalized_splits,
        },
    )
    write_readme(root, manifest)
    write_checksums(root)
    return manifest


def write_readme(root: Path, manifest: dict[str, Any]) -> None:
    split_lines = []
    for split in ("train", "val", "test"):
        stats = manifest["splits"][split]
        split_lines.append(
            f"| {split} | {', '.join(name.upper() for name in stats['buildings'])} "
            f"| {stats['images']} | {stats['boxes']} |"
        )
    class_lines = [
        f"| {entry['id']} | `{entry['name']}` | {entry['boxes']} |"
        for entry in manifest["classes"]
    ]
    readme = f"""# Lift Robot YOLO Detect Handoff

This package contains {manifest['counts']['images']} human-reviewed RGB images and
{manifest['counts']['boxes']} bounding boxes. It is intended for **Ultralytics
YOLO11n Detect** (`yolo11n.pt`), not a segmentation model.

## Contents

- `legacy_gt/images/`: copied RGB JPEG files (real files, not symlinks).
- `legacy_gt/annotations/<session>/`: canonical legacy `*_gt.json` annotations.
- `legacy_gt/excluded_manifest.json`: traceability record for excluded captures.
- `yolo_detect_coarse/`: ready-to-train YOLO Detect images and five-column labels.
- `manifest.json`: image-level provenance, hashes, dimensions, classes, and splits.
- `SHA256SUMS`: integrity hashes for every other file in this package.

Depth images are intentionally omitted because standard RGB YOLO Detect training
does not use them.

## Building-disjoint split

The split is fixed by building so images from the same building cannot leak
between training and evaluation:

| Split | Buildings | Images | Boxes |
|---|---|---:|---:|
{chr(10).join(split_lines)}

The split was chosen to keep roughly 75/13/12 percent of images in
train/validation/test while preserving every class that is present in the data
in all three splits.

## Coarse classes

Class IDs exactly match `COARSE_CLASS_NAMES` in
`scripts/annotation_web/preannotate.py`:

| ID | Class | Boxes |
|---:|---|---:|
{chr(10).join(class_lines)}

Classes with zero boxes remain configured for compatibility with the annotation
pipeline. Fine semantic labels such as floor numbers and `open`/`close` remain
available in the canonical legacy JSON but are not the target of this coarse
YOLO dataset.

## Train on Windows

1. Extract the ZIP completely.
2. Install the agreed Ultralytics version:

   ```powershell
   py -m pip install ultralytics==8.4.82
   ```

3. Run training from any directory, passing the extracted YAML path:

   ```powershell
   yolo detect train data="C:\\path\\to\\{root.name}\\yolo_detect_coarse\\dataset.yaml" model=yolo11n.pt epochs=80 imgsz=960 batch=4 device=0 seed=22
   ```

The YAML deliberately omits an absolute `path:` field. Ultralytics therefore
resolves the relative `images/...` paths from the YAML location, so the dataset
remains portable after extraction on Windows, macOS, or Linux.

## Evaluate

Use the held-out PB test building only for final evaluation:

```powershell
yolo detect val data="C:\\path\\to\\{root.name}\\yolo_detect_coarse\\dataset.yaml" model="path\\to\\best.pt" split=test imgsz=960 device=0
```

## Verify the handoff

On macOS/Linux:

```bash
shasum -a 256 -c SHA256SUMS
```

On Windows PowerShell, compare `Get-FileHash -Algorithm SHA256 <file>` against
the corresponding entry in `SHA256SUMS`.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_tree(root: Path, manifest: dict[str, Any]) -> dict[str, int]:
    expected_classes = {entry["id"]: entry["name"] for entry in manifest["classes"]}
    image_count = label_count = box_count = 0
    for split in ("train", "val", "test"):
        image_dir = root / "yolo_detect_coarse" / "images" / split
        label_dir = root / "yolo_detect_coarse" / "labels" / split
        images = sorted(image_dir.glob("*"))
        labels = sorted(label_dir.glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise RuntimeError(f"Image/label mismatch in {split}")
        for image_path in images:
            if image_path.is_symlink():
                raise RuntimeError(f"Symlink found in generated data: {image_path}")
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise RuntimeError(f"Generated image cannot be decoded: {image_path}")
            image_count += 1
        for label_path in labels:
            label_count += 1
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    raise RuntimeError(f"{label_path}:{line_number} is not five-column YOLO")
                class_id = int(fields[0])
                values = [float(field) for field in fields[1:]]
                if class_id not in expected_classes:
                    raise RuntimeError(f"{label_path}:{line_number} unknown class ID {class_id}")
                if any(value < 0.0 or value > 1.0 for value in values):
                    raise RuntimeError(f"{label_path}:{line_number} coordinate outside [0, 1]")
                if values[2] <= 0.0 or values[3] <= 0.0:
                    raise RuntimeError(f"{label_path}:{line_number} non-positive box size")
                box_count += 1
    if image_count != manifest["counts"]["images"] or label_count != image_count:
        raise RuntimeError("Generated image/label counts do not match manifest")
    if box_count != manifest["counts"]["boxes"]:
        raise RuntimeError("Generated box count does not match manifest")

    checksum_lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected_checksum_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    listed_checksum_files: set[str] = set()
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Checksum verification failed: {relative}")
        listed_checksum_files.add(relative)
    if listed_checksum_files != expected_checksum_files:
        raise RuntimeError("SHA256SUMS file list does not match generated package")
    return {"images": image_count, "labels": label_count, "boxes": box_count}


def create_zip(root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())


def verify_zip(zip_path: Path, root_name: str, expected_file_count: int) -> dict[str, int]:
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC verification failed: {bad}")
        files = [info for info in archive.infolist() if not info.is_dir()]
        if len(files) != expected_file_count:
            raise RuntimeError(f"ZIP file count mismatch: {len(files)} != {expected_file_count}")
        for info in files:
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or pure.parts[0] != root_name:
                raise RuntimeError(f"Unsafe or non-portable ZIP path: {info.filename}")
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise RuntimeError(f"ZIP contains a symlink: {info.filename}")
    return {"files": expected_file_count}


def build_handoff(
    workspace: Path,
    legacy_export: Path,
    output_dir: Path,
    package_name: str,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    legacy_export = legacy_export.resolve()
    output_dir = output_dir.resolve()
    if not workspace.is_dir() or not legacy_export.is_dir():
        raise FileNotFoundError("Workspace and legacy export must both exist")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{package_name}.zip"
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing handoff: {zip_path}")

    building_to_split = split_lookup(DEFAULT_SPLIT_BUILDINGS)
    records = collect_records(legacy_export, workspace, building_to_split)
    with tempfile.TemporaryDirectory(prefix=f".{package_name}-", dir=output_dir) as temporary:
        root = Path(temporary) / package_name
        root.mkdir()
        manifest = build_tree(root, records, legacy_export, DEFAULT_SPLIT_BUILDINGS)
        tree_validation = verify_tree(root, manifest)
        file_count = sum(path.is_file() for path in root.rglob("*"))
        create_zip(root, zip_path)
        zip_validation = verify_zip(zip_path, package_name, file_count)
    return {
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "counts": manifest["counts"],
        "splits": manifest["splits"],
        "classes": manifest["classes"],
        "validation": {**tree_validation, **zip_validation},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--legacy-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package-name",
        default=f"lift_robot_yolo11n_detect_handoff_{datetime.now():%Y%m%d}",
    )
    args = parser.parse_args()
    result = build_handoff(
        workspace=args.workspace,
        legacy_export=args.legacy_export,
        output_dir=args.output_dir,
        package_name=args.package_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
