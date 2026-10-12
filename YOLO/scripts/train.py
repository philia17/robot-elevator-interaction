from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_MODEL = Path("D:/yolo/trainingruns/pretrained/yolo11n-seg.pt")
DEFAULT_PROJECT = Path("D:/yolo/trainingruns/elevator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11n-seg for elevator button segmentation.")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML path.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="yolo11n_seg_elevator")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        task="segment",
        seed=42,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
