from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO segmentation model to ONNX.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    model.export(format="onnx", imgsz=args.imgsz, device=args.device, opset=args.opset, dynamic=args.dynamic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
