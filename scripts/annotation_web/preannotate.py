#!/usr/bin/env python3
"""Prepare local YOLO training data and create AI-draft annotations on Mac.

No AGX inference service is used. The command runs Ultralytics locally and
writes predictions through the same annotation schema as the web app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import yaml

try:
    from .app import Workspace
except ImportError:
    from app import Workspace


COARSE_CLASS_NAMES = (
    "floor_button_candidate",
    "door_button_candidate",
    "call_button_candidate",
    "alarm",
    "button_candidate",
    "keyhole",
)
SEMANTIC_CLASS_NAMES = (
    "B1", "1", "2", "3", "4", "open", "close", "up", "down", "keyhole"
)


def _ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for local pre-annotation. Install with:\n"
            "  python3 -m pip install 'ultralytics==8.4.82'"
        ) from exc
    return YOLO


def _image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _split_for(stem: str, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "val" if bucket < validation_percent else "train"


def prepare_training_data(
    gt_root: Path,
    output: Path,
    validation_percent: int = 20,
    class_mode: str = "coarse",
) -> dict[str, int]:
    """Convert existing GT JSON files into a deterministic YOLO dataset."""
    if class_mode == "coarse":
        class_names = COARSE_CLASS_NAMES
    elif class_mode == "label":
        class_names = SEMANTIC_CLASS_NAMES
    else:
        raise ValueError(f"Unknown class mode: {class_mode}")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    gt_root = gt_root.resolve()
    images_dir = gt_root / "images"
    output = output.resolve()
    if output.exists():
        manifest = output / "manifest.json"
        if any(output.iterdir()) and not manifest.is_file():
            raise RuntimeError(
                f"Refusing to replace non-generated directory without manifest.json: {output}"
            )
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "boxes": 0, "skipped": 0}
    for gt_path in sorted(gt_root.glob("*_gt.json")):
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        stem = str(payload.get("image_stem") or gt_path.stem.removesuffix("_gt"))
        image_path = _image_for_stem(images_dir, stem)
        if image_path is None:
            counts["skipped"] += 1
            continue
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            counts["skipped"] += 1
            continue
        height, width = image.shape[:2]
        yolo_lines: list[str] = []
        for box in payload.get("ground_truth", []):
            class_name = (
                str(box.get("coarse_type") or "button_candidate")
                if class_mode == "coarse"
                else str(box.get("label") or "").strip()
            )
            if class_name not in class_to_id:
                continue
            coords = box.get("bbox_xyxy")
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in coords)
            x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
            y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
            if x2 <= x1 or y2 <= y1:
                continue
            xc = (x1 + x2) / (2 * width)
            yc = (y1 + y2) / (2 * height)
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            yolo_lines.append(
                f"{class_to_id[class_name]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
            )
        split = _split_for(stem, validation_percent)
        destination_image = output / "images" / split / image_path.name
        try:
            destination_image.symlink_to(image_path)
        except OSError:
            shutil.copy2(image_path, destination_image)
        (output / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )
        counts[split] += 1
        counts["boxes"] += len(yolo_lines)

    if counts["train"] == 0 or counts["val"] == 0:
        raise RuntimeError(f"Training split is empty: {counts}")
    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {"counts": counts, "classes": class_names, "class_mode": class_mode}, indent=2
        ) + "\n",
        encoding="utf-8",
    )
    return counts


def local_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_model(
    dataset_yaml: Path,
    output: Path,
    base_model: str,
    epochs: int,
    image_size: int,
    batch: int,
) -> Path:
    YOLO = _ultralytics()
    model = YOLO(base_model)
    result = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=local_device(),
        project=str(output.resolve()),
        name="button_preannotation",
        exist_ok=True,
        patience=20,
        seed=22,
        deterministic=True,
        plots=True,
    )
    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training completed without best.pt: {save_dir}")
    return best


def predict_workspace(
    workspace_root: Path,
    weights: str,
    confidence: float,
    iou: float,
    force: bool,
    label_weights: str | None = None,
) -> dict[str, int]:
    YOLO = _ultralytics()
    workspace = Workspace(workspace_root)
    model = YOLO(weights)
    label_model = YOLO(label_weights) if label_weights else None
    counts = {"images": 0, "boxes": 0, "skipped_existing": 0, "errors": 0}
    model_names = model.names

    for item in workspace._items.values():
        annotation = workspace.load_annotation(item)
        existing = annotation.get("ground_truth", [])
        if existing and not force:
            counts["skipped_existing"] += 1
            continue
        try:
            results = model.predict(
                source=str(item["image_path"]),
                conf=confidence,
                iou=iou,
                imgsz=960,
                device=local_device(),
                verbose=False,
            )
            result = results[0]
            boxes: list[dict[str, Any]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                for index, (coords, class_id, score) in enumerate(zip(xyxy, classes, scores)):
                    raw_name = model_names.get(int(class_id), "button_candidate")
                    semantic_label = raw_name if raw_name in SEMANTIC_CLASS_NAMES else ""
                    coarse_type = (
                        raw_name
                        if raw_name in COARSE_CLASS_NAMES
                        else coarse_type_for_label(semantic_label)
                    )
                    boxes.append(
                        {
                            "bbox_xyxy": [round(float(v), 2) for v in coords],
                            "coarse_type": coarse_type,
                            "label": semantic_label,
                            "lit": False,
                            "source": "ai",
                            "confidence": round(float(score), 5),
                            "reviewed": False,
                            "model_id": Path(weights).name,
                            "prediction_id": f"{item['stem']}:{index}",
                        }
                    )
            if label_model is not None and boxes:
                label_results = label_model.predict(
                    source=str(item["image_path"]),
                    conf=max(0.20, confidence),
                    iou=iou,
                    imgsz=960,
                    device=local_device(),
                    verbose=False,
                )
                apply_semantic_labels(
                    boxes,
                    label_results[0],
                    label_model.names,
                    Path(label_weights).name,
                )
            annotation["ground_truth"] = boxes
            annotation["review_status"] = "ai_draft"
            workspace.save_annotation(item, annotation)
            workspace.compute_distance_for(item)
            counts["images"] += 1
            counts["boxes"] += len(boxes)
        except Exception as exc:
            counts["errors"] += 1
            print(f"ERROR {item['id']}: {exc}")
    return counts


def coarse_type_for_label(label: str) -> str:
    lowered = label.lower()
    if lowered in {"open", "close"}:
        return "door_button_candidate"
    if lowered in {"up", "down"}:
        return "call_button_candidate"
    if lowered == "alarm":
        return "alarm"
    if lowered == "keyhole":
        return "keyhole"
    if label:
        return "floor_button_candidate"
    return "button_candidate"


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def apply_semantic_labels(
    boxes: list[dict[str, Any]],
    result: Any,
    model_names: dict[int, str],
    model_id: str,
    minimum_iou: float = 0.35,
) -> None:
    """Fill draft labels only when a semantic detection overlaps a proposal."""
    if result.boxes is None:
        return
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    semantic = []
    for coords, class_id, score in zip(xyxy, classes, scores):
        label = str(model_names.get(int(class_id), ""))
        if label in SEMANTIC_CLASS_NAMES:
            semantic.append(([float(v) for v in coords], label, float(score)))
    for box in boxes:
        matches = [
            (box_iou(box["bbox_xyxy"], coords), label, score)
            for coords, label, score in semantic
        ]
        if not matches:
            continue
        overlap, label, score = max(matches, key=lambda row: (row[0], row[2]))
        if overlap < minimum_iou:
            continue
        box["label"] = label
        box["coarse_type"] = coarse_type_for_label(label)
        box["label_confidence"] = round(score, 5)
        box["label_model_id"] = model_id


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="convert existing GT into YOLO data")
    prepare.add_argument("--gt-root", type=Path, default=Path("datasets/ground_truth"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--validation-percent", type=int, default=20)
    prepare.add_argument("--class-mode", choices=("coarse", "label"), default="coarse")

    train = sub.add_parser("train", help="fine-tune a local detector")
    train.add_argument("--dataset-yaml", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--base-model", default="yolo11n.pt")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--image-size", type=int, default=960)
    train.add_argument("--batch", type=int, default=4)

    predict = sub.add_parser("predict", help="write AI drafts into a workspace")
    predict.add_argument("--workspace", type=Path, required=True)
    predict.add_argument("--weights", required=True)
    predict.add_argument("--confidence", type=float, default=0.18)
    predict.add_argument("--iou", type=float, default=0.5)
    predict.add_argument("--force", action="store_true")
    predict.add_argument(
        "--label-weights",
        help="Optional local semantic YOLO weights used to prefill labels on overlapping boxes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        if not 1 <= args.validation_percent <= 50:
            raise SystemExit("--validation-percent must be in 1..50")
        print(
            json.dumps(
                prepare_training_data(
                    args.gt_root, args.output, args.validation_percent, args.class_mode
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        best = train_model(
            args.dataset_yaml, args.output, args.base_model, args.epochs, args.image_size, args.batch
        )
        print(best)
    elif args.command == "predict":
        print(
            json.dumps(
                predict_workspace(
                    args.workspace,
                    args.weights,
                    args.confidence,
                    args.iou,
                    args.force,
                    args.label_weights,
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
