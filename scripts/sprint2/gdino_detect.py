#!/usr/bin/env python3
"""
Grounding DINO detector — outputs detector contract JSON (v1.0).
See docs/detector_interface_contract.md for schema.

Usage:
  python3 gdino_detect.py --image-dir /data/pmb/zero_shot_benchmark_images \
                          --output /data/pmb/zero_shot_benchmark_outputs/run_000X_gdino

Environment: nvcr.io/nvidia/pytorch:25.06-py3-igpu (Docker, GPU)
"""

import json, os, time, argparse
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "button . floor button . door open . door close"
THRESHOLD = 0.15
MAX_AREA_PCT = 10          # filter boxes larger than 10% of image
IOU_NMS_THRESHOLD = 0.5    # NMS dedup threshold


def iou(box_a, box_b):
    """Intersection over Union of two xyxy boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def nms(detections, threshold=IOU_NMS_THRESHOLD):
    """Simple NMS on xyxy boxes, keep higher score."""
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best["bbox_xyxy"], d["bbox_xyxy"]) < threshold]
    return keep


def clip_box(x1, y1, x2, y2, w, h):
    """Clip box to image bounds."""
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(x1, min(x2, w))
    y2 = max(y1, min(y2, h))
    return x1, y1, x2, y2


def run(image_dir, output_dir):
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to("cuda")

    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))

    for img_path in image_files:
        name = img_path.stem
        print("Processing %s ..." % name)

        # Load image
        pil_img = Image.open(img_path).convert("RGB")
        w, h = pil_img.size

        # Inference
        inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=THRESHOLD
        )[0]
        latency_ms = (time.time() - t0) * 1000

        boxes = results["boxes"]
        labels = results["text_labels"]
        scores = results["scores"]

        # Convert to pixel coordinates + filter by area
        raw_dets = []
        for box, label, score in zip(boxes, labels, scores):
            x1 = box[0].item() * w
            y1 = box[1].item() * h
            x2 = box[2].item() * w
            y2 = box[3].item() * h
            area_pct = (x2 - x1) * (y2 - y1) / (w * h) * 100
            if area_pct >= MAX_AREA_PCT:
                continue  # skip large garbage boxes

            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue  # skip zero-area after clipping

            raw_dets.append({
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "score": round(score.item(), 4),
                "detector_class": label,
            })

        # NMS dedup
        kept = nms(raw_dets)

        # Build contract detections
        detections = []
        draw = ImageDraw.Draw(pil_img)
        for i, det in enumerate(kept):
            b = det["bbox_xyxy"]
            center_x = (b[0] + b[2]) / 2.0
            center_y = (b[1] + b[3]) / 2.0
            detection = {
                "id": "grounding_dino_%03d" % i,
                "bbox_xyxy": b,
                "center_xy": [center_x, center_y],
                "score": det["score"],
                "coarse_type": "button_candidate",
                "detector_class": det["detector_class"],
                "raw": {},
            }
            detections.append(detection)

            # Draw on image
            color = "lime"
            draw.rectangle(b, outline=color, width=3)
            draw.text(
                (b[0], b[1] - 18),
                "[%d,%d %d,%d]" % (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                fill="cyan",
            )
            draw.text(
                (b[0], b[1] - 6),
                "%s %.2f" % (det["detector_class"], det["score"]),
                fill=color,
            )

        # Build contract
        contract = {
            "schema_version": "1.0",
            "source_detector": "grounding_dino",
            "image": {
                "path": str(img_path),
                "width": w,
                "height": h,
            },
            "detector_latency_ms": round(latency_ms, 1),
            "detections": detections,
        }

        # Save
        contract_path = output_dir / ("%s_contract.json" % name)
        with open(contract_path, "w") as f:
            json.dump(contract, f, indent=2)

        annot_path = output_dir / ("%s_annotated.png" % name)
        pil_img.save(annot_path)

        print("  -> %d detections (raw=%d), %.0f ms" % (len(detections), len(raw_dets), latency_ms))

    # Summary
    summary = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "threshold": THRESHOLD,
        "max_area_pct": MAX_AREA_PCT,
        "nms_threshold": IOU_NMS_THRESHOLD,
        "images": len(image_files),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done. %d images -> %s" % (len(image_files), output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.image_dir, args.output)
