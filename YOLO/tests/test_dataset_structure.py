from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jpeg(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (8, 8), color=(10, 20, 30))
    image.save(path, format="JPEG")


def test_prepare_dataset_dry_run_preserves_rgb_depth_meta_pairing(tmp_path: Path) -> None:
    module = load_module()
    raw_dir = tmp_path / "raw"
    annotations_dir = tmp_path / "annotations"
    output_dir = tmp_path / "prepared"
    capture_dir = raw_dir / "building_a" / "elevator_1"
    label_dir = annotations_dir / "building_a" / "elevator_1"
    capture_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    rgb_path = capture_dir / "seq001_rgb.jpg"
    depth_path = capture_dir / "seq001_depth.png"
    meta_path = capture_dir / "seq001_meta.json"
    label_path = label_dir / "seq001.txt"
    write_jpeg(rgb_path)
    depth_path.write_bytes(b"depth-placeholder")
    meta_path.write_text(json.dumps({"building": "building_a", "elevator": "elevator_1", "sequence": "seq001"}), encoding="utf-8")
    label_path.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")

    args = type(
        "Args",
        (),
        {
            "raw_dir": raw_dir,
            "annotations_dir": annotations_dir,
            "output_dir": output_dir,
            "manifest": output_dir / "manifest.csv",
            "apply": False,
            "train_ratio": 1.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "group_by": "building",
        },
    )()
    records = module.discover_records(args)
    module.write_manifest(records, args.manifest)

    assert len(records) == 1
    assert records[0].rgb_path == rgb_path
    assert records[0].depth_path == depth_path
    assert records[0].meta_path == meta_path
    assert records[0].issues == ()
    assert not (output_dir / "images").exists()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["building"] == "building_a"
    assert rows[0]["split"] == "train"


def test_missing_pair_and_label_are_reported(tmp_path: Path) -> None:
    module = load_module()
    raw_dir = tmp_path / "raw"
    annotations_dir = tmp_path / "annotations"
    capture_dir = raw_dir / "building_a"
    capture_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    write_jpeg(capture_dir / "seq002_rgb.jpg")

    args = type(
        "Args",
        (),
        {
            "raw_dir": raw_dir,
            "annotations_dir": annotations_dir,
            "output_dir": tmp_path / "prepared",
            "manifest": tmp_path / "prepared" / "manifest.csv",
            "apply": False,
            "train_ratio": 1.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "group_by": "building",
        },
    )()
    records = module.discover_records(args)
    assert set(records[0].issues) == {"missing_depth", "missing_meta", "missing_label"}
