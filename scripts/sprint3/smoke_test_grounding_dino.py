#!/usr/bin/env python3
"""
Smoke test for Grounding DINO on AGX after install_grounding_dino.sh.

Runs zero-shot detection with prompt "button" on one image and prints
detection count + wall-clock time (Expt 5 sanity check).
"""

import argparse
import sys
import time
from pathlib import Path

DEFAULT_IMAGE = Path("/data/demo_images/ieb_inside_panel_full_01.jpg")
DEFAULT_CONFIG = Path(
    "/data/tools/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DEFAULT_CHECKPOINT = Path(
    "/data/models/groundingdino/groundingdino_swint_ogc.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO AGX smoke test")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="button")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image.is_file():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        print("Run: bash ~/scripts/sprint3/install_grounding_dino.sh", file=sys.stderr)
        return 1
    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    try:
        import torch
        from groundingdino.util.inference import load_model, predict
        from groundingdino.util.inference import load_image
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Activate venv: source /data/venvs/groundingdino/bin/activate",
              file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Image:  {args.image}")
    print(f"Prompt: {args.prompt!r}")

    t0 = time.perf_counter()
    model = load_model(str(args.config), str(args.checkpoint))
    model = model.to(device)
    load_s = time.perf_counter() - t0

    image_source, image_tensor = load_image(str(args.image))
    t1 = time.perf_counter()
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )
    infer_s = time.perf_counter() - t1

    print(f"\nDetections: {len(boxes)}")
    for idx, (phrase, score) in enumerate(zip(phrases, logits), start=1):
        print(f"  {idx}. {phrase!r}  conf={float(score):.3f}")
    print(f"\nModel load: {load_s:.2f}s")
    print(f"Inference:  {infer_s:.2f}s  (Expt 5 target: <=5s per frame)")
    print("PASS" if infer_s <= 5.0 else "WARN: inference slower than 5s target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
