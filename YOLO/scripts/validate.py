from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_PROJECT = Path("D:/yolo/trainingruns/elevator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO segmentation model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default="val")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        task="segment",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
