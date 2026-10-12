from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_PROJECT = Path("D:/yolo/trainingruns/elevator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO segmentation prediction.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="predict")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        task="segment",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
