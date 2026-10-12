import json
from pathlib import Path

import cv2
import numpy as np

from scripts.annotation_web.app import Workspace
from scripts.annotation_web.export_annotations import export_annotations
from scripts.annotation_web.preannotate import coarse_type_for_label, prepare_training_data


def write_old_gt(root: Path, count: int = 10) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    for index in range(count):
        stem = f"pmb_inside_{index:03d}_rgb"
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images / f"{stem}.jpg"), image)
        payload = {
            "image_stem": stem,
            "ground_truth": [
                {
                    "bbox_xyxy": [20, 10, 60, 50],
                    "coarse_type": "floor_button_candidate",
                    "label": "1",
                }
            ],
        }
        (root / f"{stem}_gt.json").write_text(json.dumps(payload), encoding="utf-8")


def write_capture_workspace(root: Path) -> str:
    session_name = "pmb_20260722_1605_session"
    session = root / session_name
    (session / "images").mkdir(parents=True)
    (session / "depth").mkdir()
    stem = "capture_20260722_160550_043"
    assert cv2.imwrite(str(session / "images" / f"{stem}.jpg"), np.zeros((40, 60, 3), np.uint8))
    assert cv2.imwrite(str(session / "depth" / f"{stem}_depth.png"), np.full((40, 60), 500, np.uint16))
    record = {
        "filename": f"images/{stem}.jpg",
        "depth_filename": f"depth/{stem}_depth.png",
        "timestamp": "2026-07-22T16:05:50+08:00",
        "camera": "Orbbec-Gemini-335",
        "width": 60,
        "height": 40,
        "depth_scale_m_per_unit": 0.001,
    }
    (session / "metadata.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return f"{session_name}::{stem}"


def test_prepare_training_data_has_deterministic_train_and_val(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    write_old_gt(gt, 20)
    output = tmp_path / "yolo"
    counts = prepare_training_data(gt, output, validation_percent=30)
    assert counts["train"] > 0
    assert counts["val"] > 0
    assert counts["boxes"] == 20
    assert (output / "dataset.yaml").is_file()
    label = next((output / "labels" / "train").glob("*.txt"))
    fields = label.read_text().split()
    assert fields[0] == "0"
    assert all(0 <= float(value) <= 1 for value in fields[1:])


def test_prepare_semantic_training_data_uses_button_labels(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    write_old_gt(gt, 20)
    output = tmp_path / "semantic_yolo"
    counts = prepare_training_data(gt, output, validation_percent=30, class_mode="label")
    assert counts["boxes"] == 20
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["class_mode"] == "label"
    label = next((output / "labels" / "train").glob("*.txt"))
    assert label.read_text().split()[0] == "1"  # semantic class "1"


def test_keyhole_is_exported_as_a_distinct_training_class(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    write_old_gt(gt, 10)
    for payload_path in gt.glob("*_gt.json"):
        payload = json.loads(payload_path.read_text())
        payload["ground_truth"][0].update({"label": "keyhole", "coarse_type": "keyhole"})
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "keyhole_yolo"
    counts = prepare_training_data(gt, output, class_mode="label")
    assert counts["boxes"] == 10
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["classes"][-1] == "keyhole"
    label = next((output / "labels" / "train").glob("*.txt"), None)
    if label is None:
        label = next((output / "labels" / "val").glob("*.txt"))
    assert label.read_text().split()[0] == "9"
    assert coarse_type_for_label("keyhole") == "keyhole"


def test_export_only_reviewed_and_writes_excluded_manifest(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    item_id = write_capture_workspace(workspace_root)
    workspace = Workspace(workspace_root)
    item = workspace.get_item(item_id)
    annotation = workspace.load_annotation(item)
    annotation["ground_truth"] = [
        {
            "bbox_xyxy": [1, 2, 10, 12],
            "coarse_type": "floor_button_candidate",
            "label": "1",
        }
    ]
    annotation["review_status"] = "reviewed"
    workspace.save_annotation(item, annotation)

    output = tmp_path / "export"
    counts = export_annotations(workspace_root, output)
    assert counts == {"exported": 1, "excluded": 0, "not_reviewed": 0}
    target = output / "pmb_20260722_1605_session" / "capture_20260722_160550_043_gt.json"
    assert json.loads(target.read_text())["ground_truth"][0]["label"] == "1"

    annotation = workspace.load_annotation(item)
    annotation["excluded"] = True
    workspace.save_annotation(item, annotation)
    second = tmp_path / "export2"
    counts = export_annotations(workspace_root, second)
    assert counts["excluded"] == 1
    manifest = json.loads((second / "excluded_manifest.json").read_text())
    assert manifest["excluded"][0]["image_id"] == item_id
