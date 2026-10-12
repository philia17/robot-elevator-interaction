#!/usr/bin/env python3
"""Label detector-contract button crops and detect red button illumination.

The semantic stage uses a HuggingFace/PyTorch Qwen3-VL-2B model. The
illumination stage uses local red contrast between each floor-button box and
its surrounding panel, which suppresses broad warm reflections.
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from transformers import AutoProcessor


LABELS = ("alarm", "b1", "4", "3", "2", "1", "open", "close", "none")
FLOOR_LABELS = {"b1", "4", "3", "2", "1"}
ILLUMINATION_LABELS = set(FLOOR_LABELS)
ILLUMINATION_METHOD = "red_local_contrast_v1"

PROMPT = """\
You are classifying ONE cropped elevator-panel button candidate.

Choose exactly one label from this closed list:
alarm, b1, 4, 3, 2, 1, open, close, none

Definitions:
- alarm: emergency/alarm/bell button.
- b1: basement-floor button printed B1.
- 4, 3, 2, 1: floor-number buttons showing that exact digit.
- open: door-open icon. Look for two opposing triangles/arrows pointing away from each other, like < >. The pointed tips face outward, away from the center, and the flat/wide backs are nearer the center. It may also resemble two door panels separating.
- close: door-close icon. Look for two opposing triangles/arrows pointing toward each other, like > <. The pointed tips face inward toward the center. It may also resemble two door panels coming together.
- none: use this if the crop is not one of the allowed labels, is unreadable, is a reflection/display/background, or you are unsure.

Return ONLY one lowercase label from the closed list. No sentence. No punctuation.
If the door icon direction is unclear, return none.
"""

DOOR_PAIR_PROMPT = """\
You are looking at a local crop from an INSIDE elevator panel.
The two LOWER circular buttons in the crop are the target candidates.
Ignore any floor button or number above them.

Decide the label for the LEFT candidate button and the RIGHT candidate button.

Allowed labels for each candidate:
open, close, none

Definitions:
- open: door-open button. The icon suggests elevator doors moving apart, or two arrows/triangles pointing away from each other. It may look like two separated vertical strokes or outward-facing wedges.
- close: door-close button. The icon suggests elevator doors moving together, or two arrows/triangles pointing toward each other. It may look like inward-facing wedges or a pinched hourglass shape.
- none: not a readable door-control button, too dark, reflection, background, or unclear.

Use the spatial context:
- Door open and door close often appear as a left-right pair below the floor buttons.
- On many inside panels, the left button is door-open and the right button is door-close.
- Use this left/right context only when the two lower circular button icons are visible.
- Do not classify the upper floor button as left or right.

Return ONLY valid JSON. For a normal readable pair, use:
{"left":"open","right":"close"}

If neither candidate is readable, use:
{"left":"none","right":"none"}
"""


def load_model(model_path):
    import transformers

    kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    class_names = (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    )
    last_error = None
    for class_name in class_names:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            return model_cls.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Could not load VLM model from %s: %s" % (model_path, last_error))


def clip_box(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
    return x1, y1, x2, y2


def pad_box(box, width, height, pad_ratio):
    x1, y1, x2, y2 = clip_box(box, width, height)
    bw = x2 - x1
    bh = y2 - y1
    pad = max(bw, bh) * pad_ratio
    return clip_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def crop_candidate(image, box, pad_ratio, upscale):
    width, height = image.size
    x1, y1, x2, y2 = pad_box(box, width, height, pad_ratio)
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    if upscale > 1.0:
        new_size = (
            max(1, int(round(crop.width * upscale))),
            max(1, int(round(crop.height * upscale))),
        )
        crop = crop.resize(new_size, Image.Resampling.BICUBIC)
    return crop


def make_messages(prompt):
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]


def build_inputs(processor, image, device, prompt):
    messages = make_messages(prompt)
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def normalize_answer(text):
    text = (text or "").strip().lower()
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for label in LABELS:
        if re.search(r"\b%s\b" % re.escape(label), text):
            return label
    return "none"


def classify_crop(model, processor, image, max_new_tokens):
    device = next(model.parameters()).device
    inputs = build_inputs(processor, image, device, PROMPT)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if input_len:
        output_ids = output_ids[:, input_len:]
    raw = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return normalize_answer(raw), raw


def classify_crops_batched(model, processor, images, max_new_tokens, batch_size):
    if not images:
        return []

    device = next(model.parameters()).device
    messages = make_messages(PROMPT)
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = PROMPT

    results = []
    batch_size = max(1, int(batch_size or 1))
    for start in range(0, len(images), batch_size):
        batch_images = images[start:start + batch_size]
        inputs = processor(
            text=[text] * len(batch_images),
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if input_len:
            output_ids = output_ids[:, input_len:]
        raws = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results.extend((normalize_answer(raw), raw) for raw in raws)
    return results


def generate_raw(model, processor, image, prompt, max_new_tokens):
    device = next(model.parameters()).device
    inputs = build_inputs(processor, image, device, prompt)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if input_len:
        output_ids = output_ids[:, input_len:]
    return processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def normalize_pair_label(value):
    value = str(value or "").strip().lower()
    if value in ("open", "door_open"):
        return "open"
    if value in ("close", "door_close"):
        return "close"
    return "none"


def parse_pair_response(raw):
    text = (raw or "").strip()
    text = re.sub(r"<\|[^>]+?\|>", " ", text).strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "left": normalize_pair_label(data.get("left")),
                "right": normalize_pair_label(data.get("right")),
            }
        except json.JSONDecodeError:
            pass

    cleaned = text.lower()
    labels = {"left": "none", "right": "none"}
    for side in ("left", "right"):
        side_match = re.search(
            r"\b%s\b[^a-z0-9_]+(open|close|none|door_open|door_close)\b" % side,
            cleaned,
        )
        if side_match:
            labels[side] = normalize_pair_label(side_match.group(1))
    return labels


def classify_door_pair(model, processor, image, max_new_tokens):
    raw = generate_raw(
        model,
        processor,
        image,
        DOOR_PAIR_PROMPT,
        max_new_tokens,
    )
    return parse_pair_response(raw), raw


def load_font(size=14):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_text_box(draw, xy, text, fill, font):
    x, y = xy
    text_box = draw.textbbox((0, 0), text, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    pad_x = 5
    pad_y = 3
    draw.rectangle(
        (x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y),
        fill=fill,
    )
    draw.text((x + pad_x, y + pad_y), text, fill="black", font=font)
    return tw + 2 * pad_x, th + 2 * pad_y


def rects_overlap(first, second, gap=2):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return not (
        ax2 + gap <= bx1
        or bx2 + gap <= ax1
        or ay2 + gap <= by1
        or by2 + gap <= ay1
    )


def choose_label_position(box, label_size, image_size, occupied, avoid_boxes):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    label_w, label_h = label_size
    image_w, image_h = image_size
    candidates = [
        (x2 + 4, y1),
        (x1 - label_w - 4, y1),
        (x1, y1 - label_h - 3),
        (x1, y2 + 3),
    ]
    for step in range(1, 8):
        offset = step * (label_h + 3)
        candidates.extend([
            (x2 + 4, y1 - offset),
            (x2 + 4, y1 + offset),
            (x1 - label_w - 4, y1 - offset),
            (x1 - label_w - 4, y1 + offset),
            (x1, y1 - label_h - 3 - offset),
            (x1, y2 + 3 + offset),
        ])

    seen = set()
    fallback = None
    fallback_conflicts = None
    for raw_x, raw_y in candidates:
        tx = min(max(0, raw_x), max(0, image_w - label_w))
        ty = min(max(0, raw_y), max(0, image_h - label_h))
        if (tx, ty) in seen:
            continue
        seen.add((tx, ty))
        rect = (tx, ty, tx + label_w, ty + label_h)
        conflicts = sum(rects_overlap(rect, other) for other in occupied)
        conflicts += sum(rects_overlap(rect, other) for other in avoid_boxes)
        if conflicts == 0:
            return tx, ty, rect
        if fallback is None or conflicts < fallback_conflicts:
            fallback = (tx, ty, rect)
            fallback_conflicts = conflicts
    return fallback


def format_label_text(label, score=None, illumination=None):
    display_label = "B1" if label == "b1" else label
    text = display_label if score is None else "%s %.2f" % (display_label, float(score))
    if illumination and illumination.get("eligible"):
        text += " | %s %.2f" % (
            illumination["state"],
            float(illumination["confidence"]),
        )
    return text


def draw_label(
    draw,
    box,
    label,
    score=None,
    illumination=None,
    font=None,
    image_size=None,
    occupied=None,
    avoid_boxes=None,
):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if illumination and illumination.get("state") == "lit":
        color = "orange"
    elif illumination and illumination.get("state") == "uncertain":
        color = "yellow"
    else:
        color = "lime" if label != "none" else "red"
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text = format_label_text(label, score, illumination)
    font = font or load_font()
    text_box = draw.textbbox((0, 0), text, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    label_w = tw + 10
    label_h = th + 6
    image_size = image_size or (max(x2, x1 + label_w), max(y2, y1 + label_h))
    occupied = occupied if occupied is not None else []
    avoid_boxes = avoid_boxes or []
    other_boxes = [
        other for other in avoid_boxes
        if tuple(int(round(v)) for v in other) != (x1, y1, x2, y2)
    ]
    tx, ty, label_rect = choose_label_position(
        (x1, y1, x2, y2),
        (label_w, label_h),
        image_size,
        occupied,
        other_boxes,
    )
    draw_text_box(draw, (tx, ty), text, color, font)
    occupied.append(label_rect)


def draw_footer(draw, image_size, elapsed_s, illumination_s=None, font=None):
    font = font or load_font(size=14)
    text = "image time  %.2fs" % elapsed_s
    if illumination_s is not None:
        text += " | light  %.3fs" % illumination_s
    _, height = image_size
    text_box = draw.textbbox((0, 0), text, font=font)
    th = text_box[3] - text_box[1]
    x = 8
    y = max(8, height - th - 12)
    draw_text_box(draw, (x, y), text, "white", font)


def box_center(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def box_size(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def label_text(det):
    return det.get("label", {}).get("text") or "none"


def red_features(image, box, context_ratio=0.45):
    image_array = np.asarray(image, dtype=np.float32)
    image_h, image_w = image_array.shape[:2]
    x1, y1, x2, y2 = clip_box(box, image_w, image_h)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    ix1 = max(0, int(round(x1)))
    iy1 = max(0, int(round(y1)))
    ix2 = min(image_w, max(ix1 + 1, int(round(x2))))
    iy2 = min(image_h, max(iy1 + 1, int(round(y2))))
    ex1 = max(0, int(round(x1 - context_ratio * box_w)))
    ey1 = max(0, int(round(y1 - context_ratio * box_h)))
    ex2 = min(image_w, int(round(x2 + context_ratio * box_w)))
    ey2 = min(image_h, int(round(y2 + context_ratio * box_h)))

    inside = image_array[iy1:iy2, ix1:ix2]
    expanded = image_array[ey1:ey2, ex1:ex2]
    if inside.size == 0 or expanded.size == 0:
        raise ValueError("Empty illumination crop for box %r" % (box,))

    red = inside[..., 0]
    green = inside[..., 1]
    blue = inside[..., 2]
    red_excess = red - 0.5 * (green + blue)
    red_mask = (
        (red > 45.0)
        & (red > 1.16 * green)
        & (red > 1.10 * blue)
    )

    outer_red = expanded[..., 0]
    outer_green = expanded[..., 1]
    outer_blue = expanded[..., 2]
    outer_excess = outer_red - 0.5 * (outer_green + outer_blue)
    outer_mask = np.ones(expanded.shape[:2], dtype=bool)
    outer_mask[
        max(0, iy1 - ey1):min(expanded.shape[0], iy2 - ey1),
        max(0, ix1 - ex1):min(expanded.shape[1], ix2 - ex1),
    ] = False
    context_values = outer_excess[outer_mask]
    if context_values.size == 0:
        context_values = outer_excess.reshape(-1)

    inside_p90 = float(np.quantile(red_excess, 0.90))
    context_p90 = float(np.quantile(context_values, 0.90))
    red_fraction = float(np.mean(red_mask))
    return {
        "inside_red_excess_p90": round(inside_p90, 4),
        "context_red_excess_p90": round(context_p90, 4),
        "red_delta_p90": round(inside_p90 - context_p90, 4),
        "red_fraction": round(red_fraction, 6),
        "context_ratio": context_ratio,
    }


def illumination_probability(red_delta, red_fraction):
    signal = (
        (float(red_delta) - 10.0) / 3.5
        + (float(red_fraction) - 0.045) / 0.04
    )
    signal = max(-20.0, min(20.0, signal))
    return 1.0 / (1.0 + math.exp(-signal))


def classify_illumination(image, det, args):
    label = label_text(det)
    if label not in ILLUMINATION_LABELS:
        return {
            "eligible": False,
            "state": "not_applicable",
            "confidence": 1.0,
            "method": ILLUMINATION_METHOD,
            "reason": "semantic label is not a floor button",
        }

    features = red_features(image, det["bbox_xyxy"], args.lit_context_ratio)
    red_delta = features["red_delta_p90"]
    red_fraction = features["red_fraction"]
    probability = illumination_probability(red_delta, red_fraction)

    if (
        red_delta >= args.lit_red_delta_threshold
        and red_fraction >= args.lit_red_fraction_threshold
    ):
        state = "lit"
        confidence = max(0.70, probability)
    elif (
        red_delta >= args.lit_uncertain_delta_threshold
        and red_fraction >= args.lit_uncertain_fraction_threshold
    ):
        state = "uncertain"
        confidence = 0.50 + 0.20 * abs(probability - 0.50)
    else:
        state = "unlit"
        confidence = max(0.55, 1.0 - probability)

    return {
        "eligible": True,
        "state": state,
        "confidence": round(min(0.999, confidence), 4),
        "lit_probability": round(probability, 4),
        "method": ILLUMINATION_METHOD,
        "features": features,
    }


def find_door_pair(detections, allowed_labels=None):
    allowed_labels = allowed_labels or {"none"}
    floor_dets = [
        det for det in detections
        if label_text(det) in FLOOR_LABELS
    ]
    if not floor_dets:
        return None

    floor_centers = [box_center(det["bbox_xyxy"]) for det in floor_dets]
    floor_bottom = max(float(det["bbox_xyxy"][3]) for det in floor_dets)
    floor_x_min = min(cx for cx, _ in floor_centers)
    floor_x_max = max(cx for cx, _ in floor_centers)
    floor_x_center = 0.5 * (floor_x_min + floor_x_max)

    candidate_dets = [
        det for det in detections
        if label_text(det) in allowed_labels
        and box_center(det["bbox_xyxy"])[1] > floor_bottom + 8
    ]
    best = None
    best_score = None
    for i, left in enumerate(candidate_dets):
        for right in candidate_dets[i + 1:]:
            lx, ly = box_center(left["bbox_xyxy"])
            rx, ry = box_center(right["bbox_xyxy"])
            if lx > rx:
                left, right = right, left
                lx, ly, rx, ry = rx, ry, lx, ly

            lw, lh = box_size(left["bbox_xyxy"])
            rw, rh = box_size(right["bbox_xyxy"])
            avg_w = max(1.0, 0.5 * (lw + rw))
            avg_h = max(1.0, 0.5 * (lh + rh))
            y_diff = abs(ly - ry)
            center_gap = rx - lx
            size_ratio = max(lw, rw, lh, rh) / max(1.0, min(lw, rw, lh, rh))
            pair_center_x = 0.5 * (lx + rx)

            if y_diff > 0.55 * avg_h:
                continue
            if center_gap < 0.8 * avg_w or center_gap > 3.2 * avg_w:
                continue
            if size_ratio > 1.45:
                continue
            if abs(pair_center_x - floor_x_center) > 2.2 * avg_w:
                continue

            score = (
                y_diff / avg_h
                + abs(pair_center_x - floor_x_center) / avg_w
                + abs(center_gap - 1.75 * avg_w) / avg_w
                - 0.01 * max(ly, ry)
            )
            if best is None or score < best_score:
                best = (left, right)
                best_score = score
    return best


def apply_geometry_door_completion(detections):
    pair = find_door_pair(detections, allowed_labels={"none", "open", "close"})
    if not pair:
        return {"status": "no_candidate_pair"}

    left_det, right_det = pair
    left_label = label_text(left_det)
    right_label = label_text(right_det)
    inferred = []

    if left_label == "none" and right_label == "close":
        apply_inferred_label(left_det, "open", right_det, "left")
        inferred.append({"detection_id": left_det.get("id", ""), "label": "open"})
    elif left_label == "open" and right_label == "none":
        apply_inferred_label(right_det, "close", left_det, "right")
        inferred.append({"detection_id": right_det.get("id", ""), "label": "close"})

    return {
        "status": "applied" if inferred else "no_completion_needed",
        "left_detection_id": left_det.get("id", ""),
        "right_detection_id": right_det.get("id", ""),
        "left_before": left_label,
        "right_before": right_label,
        "inferred": inferred,
        "rule": "complete adjacent door pair only in the common left=open, right=close layout",
    }


def crop_pair_region(image, left_det, right_det, pad_ratio, upscale):
    width, height = image.size
    lbox = left_det["bbox_xyxy"]
    rbox = right_det["bbox_xyxy"]
    union = (
        min(float(lbox[0]), float(rbox[0])),
        min(float(lbox[1]), float(rbox[1])),
        max(float(lbox[2]), float(rbox[2])),
        max(float(lbox[3]), float(rbox[3])),
    )
    x1, y1, x2, y2 = pad_box(union, width, height, pad_ratio)
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    if upscale > 1.0:
        crop = crop.resize(
            (
                max(1, int(round(crop.width * upscale))),
                max(1, int(round(crop.height * upscale))),
            ),
            Image.Resampling.BICUBIC,
        )
    return crop


def apply_inferred_label(target_det, label, neighbor_det, side):
    target_det["label"] = {
        "text": label,
        "type": "button_label",
        "valid": True,
        "allowed_labels": list(LABELS),
        "raw_response": "geometry_pair_completion",
        "source": "geometry_door_pair_completion",
        "geometry_pair_completion": {
            "side": side,
            "neighbor_detection_id": neighbor_det.get("id", ""),
            "neighbor_label": label_text(neighbor_det),
            "result": label,
        },
    }


def enhance_pair_crop(crop):
    crop = ImageOps.autocontrast(crop, cutoff=1)
    crop = ImageEnhance.Brightness(crop).enhance(1.35)
    crop = ImageEnhance.Contrast(crop).enhance(1.55)
    crop = ImageEnhance.Sharpness(crop).enhance(1.25)
    return crop


def apply_pair_label(det, label, raw, side):
    if label == "none":
        det.setdefault("label", {})
        det["label"]["door_pair_review"] = {
            "side": side,
            "raw_response": raw,
            "result": label,
            "source": "pytorch_hf_2b_vlm_pair_review",
        }
        return
    det["label"] = {
        "text": label,
        "type": "button_label",
        "valid": True,
        "allowed_labels": list(LABELS),
        "raw_response": raw,
        "source": "pytorch_hf_2b_vlm_pair_review",
        "door_pair_review": {
            "side": side,
            "result": label,
            "source": "pytorch_hf_2b_vlm_pair_review",
        },
    }


def resolve_image_path(contract, image_dir):
    image_path = contract.get("image", {}).get("path", "")
    if not os.path.isabs(image_path):
        image_path = os.path.join(image_dir, image_path)
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir, os.path.basename(image_path))
    return image_path


def process_contract(
    contract_path,
    image_dir,
    output_json,
    output_image,
    model,
    processor,
    args,
    model_load_time_s,
):
    image_start = time.perf_counter()
    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    image_path = resolve_image_path(contract, image_dir)
    image = Image.open(image_path).convert("RGB")

    detections = contract.get("detections", [])
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    label_font = load_font(size=args.font_size)
    footer_font = load_font(size=args.font_size)

    crops = [
        crop_candidate(image, det["bbox_xyxy"], args.pad_ratio, args.upscale)
        for det in detections
    ]
    crop_labels = classify_crops_batched(
        model,
        processor,
        crops,
        args.max_new_tokens,
        args.vlm_batch_size,
    )

    for index, (det, result) in enumerate(zip(detections, crop_labels)):
        label, raw = result
        det["label"] = {
            "text": "" if label == "none" else label,
            "type": "rejected" if label == "none" else "button_label",
            "valid": label != "none",
            "allowed_labels": list(LABELS),
            "raw_response": raw,
            "source": "pytorch_hf_2b_vlm",
        }
        print("[%d/%d] %s -> %s raw=%r" % (
            index + 1,
            len(detections),
            det.get("id", "det"),
            label,
            raw,
        ))

    geometry_completion = None
    if not args.disable_geometry_door_completion:
        geometry_completion = apply_geometry_door_completion(detections)
        print("[door-geometry] %s" % geometry_completion.get("status", "unknown"))

    pair_review = None
    if args.enable_door_pair_review and not args.disable_door_pair_review:
        pair = find_door_pair(detections, allowed_labels={"none"})
        if pair:
            left_det, right_det = pair
            pair_crop = crop_pair_region(
                image,
                left_det,
                right_det,
                args.pair_pad_ratio,
                args.pair_upscale,
            )
            if not args.disable_pair_enhance:
                pair_crop = enhance_pair_crop(pair_crop)
            if args.debug_crop_dir:
                debug_dir = Path(args.debug_crop_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_name = "%s_%s_door_pair.png" % (
                    left_det.get("id", "left"),
                    right_det.get("id", "right"),
                )
                pair_crop.save(debug_dir / debug_name)
            pair_labels, pair_raw = classify_door_pair(
                model,
                processor,
                pair_crop,
                args.pair_max_new_tokens,
            )
            apply_pair_label(left_det, pair_labels["left"], pair_raw, "left")
            apply_pair_label(right_det, pair_labels["right"], pair_raw, "right")
            pair_review = {
                "left_detection_id": left_det.get("id", ""),
                "right_detection_id": right_det.get("id", ""),
                "left_result": pair_labels["left"],
                "right_result": pair_labels["right"],
                "raw_response": pair_raw,
                "prompt": DOOR_PAIR_PROMPT,
            }
            print(
                "[door-pair] %s/%s -> left=%s right=%s raw=%r" % (
                    left_det.get("id", "left"),
                    right_det.get("id", "right"),
                    pair_labels["left"],
                    pair_labels["right"],
                    pair_raw,
                )
            )
        else:
            pair_review = {"status": "no_candidate_pair"}
            print("[door-pair] no candidate pair")

    illumination_start = time.perf_counter()
    for det in detections:
        det["illumination"] = classify_illumination(image, det, args)
        if det["illumination"].get("eligible"):
            print(
                "[light] %s label=%s state=%s confidence=%.2f delta=%.2f fraction=%.3f"
                % (
                    det.get("id", "det"),
                    label_text(det),
                    det["illumination"]["state"],
                    det["illumination"]["confidence"],
                    det["illumination"]["features"]["red_delta_p90"],
                    det["illumination"]["features"]["red_fraction"],
                )
            )
    illumination_time_s = time.perf_counter() - illumination_start

    occupied_labels = []
    detection_boxes = [det["bbox_xyxy"] for det in detections]
    for det in sorted(
        detections,
        key=lambda item: (box_center(item["bbox_xyxy"])[1], box_center(item["bbox_xyxy"])[0]),
    ):
        label = label_text(det)
        draw_label(
            draw,
            det["bbox_xyxy"],
            label,
            det.get("score"),
            det.get("illumination"),
            font=label_font,
            image_size=annotated.size,
            occupied=occupied_labels,
            avoid_boxes=detection_boxes,
        )

    image_time_s = time.perf_counter() - image_start
    contract["label_source"] = {
        "backend": "pytorch_hf_vlm",
        "model": args.model,
        "allowed_labels": list(LABELS),
        "prompt": PROMPT,
        "geometry_door_completion_enabled": not args.disable_geometry_door_completion,
        "geometry_door_completion": geometry_completion,
        "door_pair_review_enabled": args.enable_door_pair_review and not args.disable_door_pair_review,
        "door_pair_prompt": DOOR_PAIR_PROMPT,
        "door_pair_review": pair_review,
        "model_load_time_s": round(model_load_time_s, 3),
        "image_time_s": round(image_time_s, 3),
    }
    contract["illumination_source"] = {
        "method": ILLUMINATION_METHOD,
        "eligible_labels": sorted(ILLUMINATION_LABELS),
        "thresholds": {
            "lit_red_delta_p90": args.lit_red_delta_threshold,
            "lit_red_fraction": args.lit_red_fraction_threshold,
            "uncertain_red_delta_p90": args.lit_uncertain_delta_threshold,
            "uncertain_red_fraction": args.lit_uncertain_fraction_threshold,
            "context_ratio": args.lit_context_ratio,
        },
        "illumination_time_s": round(illumination_time_s, 6),
    }
    draw_footer(
        draw,
        annotated.size,
        image_time_s,
        illumination_time_s,
        footer_font,
    )

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    annotated.save(output_image)
    print("Saved JSON: " + output_json)
    print("Saved image: " + output_image)
    return {
        "contract": str(contract_path),
        "image": image_path,
        "output_json": output_json,
        "output_image": output_image,
        "detections": len(detections),
        "valid_labels": sum(1 for det in detections if det.get("label", {}).get("valid")),
        "lit_detections": sum(
            1 for det in detections
            if det.get("illumination", {}).get("state") == "lit"
        ),
        "unlit_detections": sum(
            1 for det in detections
            if det.get("illumination", {}).get("state") == "unlit"
        ),
        "uncertain_detections": sum(
            1 for det in detections
            if det.get("illumination", {}).get("state") == "uncertain"
        ),
        "illumination_time_s": round(illumination_time_s, 6),
        "image_time_s": round(image_time_s, 3),
    }


def build_jobs(args):
    if args.contract_dir:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --contract-dir")
        contract_paths = sorted(Path(args.contract_dir).glob("*_contract.json"))
        if not contract_paths:
            raise ValueError("No *_contract.json files found in %s" % args.contract_dir)
        jobs = []
        for contract_path in contract_paths:
            base = contract_path.name.replace("_contract.json", "")
            jobs.append((
                contract_path,
                str(Path(args.output_dir) / ("%s_vlm2b_lit_v1_labeled.json" % base)),
                str(Path(args.output_dir) / ("%s_vlm2b_lit_v1_labeled.png" % base)),
            ))
        return jobs

    if not args.contract:
        raise ValueError("Either --contract or --contract-dir is required")
    if not args.output_json or not args.output_image:
        raise ValueError("--output-json and --output-image are required with --contract")
    return [(Path(args.contract), args.output_json, args.output_image)]


def run(args):
    jobs = build_jobs(args)
    batch_start = time.perf_counter()
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = load_model(args.model)
    model.eval()
    model_load_time_s = time.perf_counter() - load_start
    print("Model load time: %.2fs" % model_load_time_s)

    results = []
    for job_index, (contract_path, output_json, output_image) in enumerate(jobs, start=1):
        print("=== Image %d/%d: %s ===" % (job_index, len(jobs), contract_path.name))
        results.append(process_contract(
            contract_path,
            args.image_dir,
            output_json,
            output_image,
            model,
            processor,
            args,
            model_load_time_s,
        ))

    batch_time_s = time.perf_counter() - batch_start
    if args.output_dir:
        summary = {
            "model": args.model,
            "model_load_time_s": round(model_load_time_s, 3),
            "batch_total_time_s": round(batch_time_s, 3),
            "images": len(results),
            "avg_image_time_s": round(
                sum(item["image_time_s"] for item in results) / max(1, len(results)),
                3,
            ),
            "illumination_method": ILLUMINATION_METHOD,
            "illumination_time_total_s": round(
                sum(item["illumination_time_s"] for item in results),
                6,
            ),
            "avg_illumination_time_s": round(
                sum(item["illumination_time_s"] for item in results)
                / max(1, len(results)),
                6,
            ),
            "lit_detections": sum(item["lit_detections"] for item in results),
            "unlit_detections": sum(item["unlit_detections"] for item in results),
            "uncertain_detections": sum(
                item["uncertain_detections"] for item in results
            ),
            "results": results,
        }
        summary_path = Path(args.output_dir) / "batch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("Saved batch summary: %s" % summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", help="GDINO contract JSON")
    parser.add_argument("--contract-dir", help="Directory containing *_contract.json files")
    parser.add_argument("--image-dir", required=True, help="Directory containing original image")
    parser.add_argument("--model", required=True, help="HF/PyTorch 2B model path or model id")
    parser.add_argument("--output-json")
    parser.add_argument("--output-image")
    parser.add_argument("--output-dir", help="Batch output directory")
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--upscale", type=float, default=3.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--vlm-batch-size", type=int, default=8)
    parser.add_argument("--enable-door-pair-review", action="store_true")
    parser.add_argument("--disable-door-pair-review", action="store_true")
    parser.add_argument("--disable-geometry-door-completion", action="store_true")
    parser.add_argument("--pair-pad-ratio", type=float, default=0.8)
    parser.add_argument("--pair-upscale", type=float, default=4.0)
    parser.add_argument("--pair-max-new-tokens", type=int, default=80)
    parser.add_argument("--disable-pair-enhance", action="store_true")
    parser.add_argument("--debug-crop-dir", default="")
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--lit-context-ratio", type=float, default=0.45)
    parser.add_argument("--lit-red-delta-threshold", type=float, default=12.0)
    parser.add_argument("--lit-red-fraction-threshold", type=float, default=0.055)
    parser.add_argument("--lit-uncertain-delta-threshold", type=float, default=8.5)
    parser.add_argument("--lit-uncertain-fraction-threshold", type=float, default=0.04)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
