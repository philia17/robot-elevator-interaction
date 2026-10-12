#!/usr/bin/env python3
"""Traditional YOLO inference runner — Ultralytics predict → contract JSON + overlay.

Short-term: YOLO-World weights (e.g. yolov8m-worldv2.pt) with set_classes(["button"])
for zero-shot button detection on AGX.

Long-term: replace with trained YOLO11n-seg weights from YOLO/ (8 elevator classes).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapter_traditional_yolo import adapt_traditional_yolo  # noqa: E402
from detector_contract import (  # noqa: E402
    draw_contract_overlay,
    validate_detection_array,
    write_contract_json,
)

PIPELINE = "traditional_yolo"
DEFAULT_CLASSES = ["button"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traditional YOLO detector → contract JSON + overlay")
    p.add_argument("--image", required=True, help="Input RGB image path")
    p.add_argument("--model", default="yolov8m-worldv2.pt", help="Ultralytics model weights")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for contract JSON and overlay")
    p.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   help="YOLO-World class prompts (default: button)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    p.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold")
    p.add_argument("--device", default=0, help="Ultralytics device (e.g. 0 or cpu)")
    return p.parse_args()


def artifact_paths(output_dir: Path, image_stem: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{image_stem}_overlay.jpg",
        output_dir / f"{image_stem}_contract.json",
    )


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: cannot read image: {image_path}", file=sys.stderr)
        return 1

    img_h, img_w = img.shape[:2]

    t0 = time.perf_counter()
    model = YOLO(args.model)
    model.set_classes(list(args.classes))
    load_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    results = model.predict(
        img,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        max_det=300,
    )
    infer_ms = (time.perf_counter() - t1) * 1000.0

    result = results[0]
    det_array = adapt_traditional_yolo(
        result.boxes,
        model.names,
        image_path=str(image_path.resolve()),
        image_width=img_w,
        image_height=img_h,
        detector_latency_ms=infer_ms,
        model_path=str(args.model),
    )

    errors = validate_detection_array(det_array)
    if errors:
        print("WARN: contract validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)

    overlay_path, contract_path = artifact_paths(output_dir, image_path.stem)
    write_contract_json(contract_path, det_array)
    title = (
        f"{PIPELINE} det={len(det_array.detections)} "
        f"conf={args.conf} infer={infer_ms:.0f}ms"
    )
    overlay = draw_contract_overlay(img, det_array, title)
    oh, ow = overlay.shape[:2]
    if max(oh, ow) > 1400:
        scale = 1400 / max(oh, ow)
        overlay = cv2.resize(overlay, (int(ow * scale), int(oh * scale)))
    cv2.imwrite(str(overlay_path), overlay)

    summary = {
        "pipeline": PIPELINE,
        "image": str(image_path),
        "model": args.model,
        "classes": list(args.classes),
        "conf": args.conf,
        "image_size": [img_w, img_h],
        "load_ms": round(load_ms, 1),
        "infer_ms": round(infer_ms, 1),
        "raw_detections": 0 if result.boxes is None else len(result.boxes),
        "detections_kept": len(det_array.detections),
        "contract_json": str(contract_path),
        "overlay": str(overlay_path),
    }
    print(json.dumps(summary, indent=2))
    for det in det_array.detections:
        print(
            f"  {det.detector_class!r} score={det.score:.3f} "
            f"center=({det.center_xy[0]:.1f}, {det.center_xy[1]:.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
