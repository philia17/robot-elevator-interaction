#!/usr/bin/env python3
"""Two-stage zero-shot localization: elevator panel, then buttons inside the panel crop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detector_contract import (  # noqa: E402
    BACKEND_YOLO_WORLD,
    COARSE_BUTTON_CANDIDATE,
    COARSE_DOOR_BUTTON_CANDIDATE,
    COARSE_FLOOR_BUTTON_CANDIDATE,
    SCHEMA_VERSION,
    Detection,
    DetectionArray,
    clip_xyxy,
    compute_center,
    is_valid_box,
    validate_detection_array,
    write_contract_json,
    xywh_to_xyxy,
)
from zero_shot_benchmark_paths import (  # noqa: E402
    OUTPUTS_ROOT,
    allocate_run_dir,
)

PIPELINE = "panel_then_buttons"

PANEL_CLASSES = [
    "metal elevator control panel with round or rectangular buttons",
    "rectangular metal elevator control panel with square buttons",
    "silver elevator button panel",
    "elevator passenger control panel",
    "elevator car button panel with floor buttons",
    "metal elevator control panel",
    "elevator interior control panel",
    "",
]

BUTTON_CLASSES = [
    "square elevator button with white number",
    "rectangular metal elevator button",
    "square elevator floor button",
    "elevator floor button",
    "elevator floor button with number",
    "door open button with arrow",
    "door close button with arrow",
    "elevator door open button with triangle symbol",
    "elevator door close button with triangle symbol",
    "",
]

DOOR_OPEN_LABEL = "door open button with arrow"
DOOR_CLOSE_LABEL = "door close button with arrow"

# Button filter limits relative to panel crop (not full image).
BUTTON_MAX_REL_AREA = 0.12
BUTTON_MIN_WH = 20
BUTTON_MAX_WH = 120
PANEL_MIN_REL_AREA = 0.03
PANEL_MAX_REL_AREA = 0.85
DEFAULT_BOTTOM_DOOR_FRAC = 0.28


def dedupe_buttons(buttons: list[dict], min_dist_px: float = 18.0) -> list[dict]:
    """Drop physical-button duplicates and lower-confidence mirror reflections."""
    kept: list[dict] = []
    for d in sorted(buttons, key=lambda x: x["confidence"], reverse=True):
        cx, cy = d["center"]
        if any(
            (cx - k["center"][0]) ** 2 + (cy - k["center"][1]) ** 2 < min_dist_px ** 2
            or (abs(cy - k["center"][1]) < 20.0 and abs(cx - k["center"][0]) > 60.0)
            for k in kept
        ):
            continue
        kept.append(d)
    kept.sort(key=lambda d: d["center"][1])
    return kept


def drop_panel_top_strip(buttons: list[dict], panel_bbox: list[float], top_frac: float) -> list[dict]:
    """Ignore detections in the top strip of the panel (common alarm / display zone)."""
    if top_frac <= 0:
        return buttons
    _, py, _, ph = panel_bbox
    y_min = py + ph * top_frac
    return [b for b in buttons if b["center"][1] >= y_min]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Panel-then-buttons YOLO-World detector")
    p.add_argument("--image", required=True)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Numbered run output directory (default: new run under output-root)")
    p.add_argument("--output-root", type=Path, default=OUTPUTS_ROOT,
                   help="Root for auto-allocated run_NNNN_* directories")
    p.add_argument("--model", default="yolov8m-worldv2.pt")
    p.add_argument("--panel-conf", type=float, default=0.05)
    p.add_argument("--button-conf", type=float, default=0.01)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--device", default=0)
    p.add_argument("--panel-padding", type=float, default=0.08,
                   help="Fractional padding around panel bbox before button stage")
    p.add_argument("--main-column-frac", type=float, default=0.35,
                   help="Keep floor buttons whose x is within this fraction of panel width from main column")
    p.add_argument("--bottom-door-frac", type=float, default=DEFAULT_BOTTOM_DOOR_FRAC,
                   help="Bottom fraction of panel treated as door-button row (skips column filter)")
    p.add_argument("--panel-bottom-pad-frac", type=float, default=0.12,
                   help="Extra downward padding on panel crop for door row")
    p.add_argument("--panel-top-drop-frac", type=float, default=0.12,
                   help="Drop button candidates in top fraction of panel bbox (alarm/display strip)")
    p.add_argument("--dedupe-dist", type=float, default=20.0,
                   help="Min center distance (px) between kept buttons")
    p.add_argument("--runs", type=int, default=1, help="Warm repeat count for button stage timing")
    return p.parse_args()


def boxes_to_dicts(boxes, class_names: list[str], offset_x: float = 0, offset_y: float = 0,
                    area_ref: int | None = None, min_wh: int = 0, max_wh: int = 10_000,
                    max_rel_area: float = 1.0) -> list[dict]:
    if boxes is None or len(boxes) == 0:
        return []
    area_ref = area_ref or 1
    out = []
    for b in boxes:
        xywh = b.xywh[0].cpu().numpy()
        x = float(xywh[0] - xywh[2] / 2)
        y = float(xywh[1] - xywh[3] / 2)
        w, h = float(xywh[2]), float(xywh[3])
        rel = (w * h) / area_ref
        if rel > max_rel_area or w < min_wh or h < min_wh or w > max_wh or h > max_wh:
            continue
        cid = int(b.cls[0])
        out.append({
            "class_name": class_names[cid] if cid < len(class_names) else str(cid),
            "confidence": round(float(b.conf[0]), 4),
            "bbox": [round(x + offset_x, 1), round(y + offset_y, 1), round(w, 1), round(h, 1)],
            "center": [round(float(xywh[0] + offset_x), 1), round(float(xywh[1] + offset_y), 1)],
            "rel_area": round(rel, 4),
        })
    return out


def merge_vertical_panels(candidates: list[dict], img_w: int, img_h: int,
                          column_frac: float = 0.08) -> list[dict]:
    """Merge panel boxes in the same vertical column (close-up shots often split into strips)."""
    if len(candidates) <= 1:
        return candidates
    img_area = img_w * img_h
    max_dx = img_w * column_frac
    groups: list[list[dict]] = []
    for c in sorted(candidates, key=lambda x: x["center"][0]):
        for g in groups:
            gx = float(np.median([x["center"][0] for x in g]))
            if abs(c["center"][0] - gx) <= max_dx:
                g.append(c)
                break
        else:
            groups.append([c])

    merged: list[dict] = []
    for g in groups:
        if len(g) == 1:
            merged.append(g[0])
            continue
        xs = [d["bbox"][0] for d in g]
        ys = [d["bbox"][1] for d in g]
        x2 = max(d["bbox"][0] + d["bbox"][2] for d in g)
        y2 = max(d["bbox"][1] + d["bbox"][3] for d in g)
        x1, y1 = min(xs), min(ys)
        w, h = x2 - x1, y2 - y1
        merged.append({
            "class_name": g[0]["class_name"],
            "confidence": max(d["confidence"] for d in g),
            "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
            "center": [round(x1 + w / 2, 1), round(y1 + h / 2, 1)],
            "rel_area": round((w * h) / img_area, 4),
            "merged_from": len(g),
        })
    return merged


def select_panel(candidates: list[dict], img_w: int, img_h: int) -> dict | None:
    """Pick one passenger control panel bbox (general heuristics, not per-image constants)."""
    if not candidates:
        return None
    img_area = img_w * img_h
    scored = []
    for c in candidates:
        x, y, w, h = c["bbox"]
        area = w * h
        rel = area / img_area
        if rel < PANEL_MIN_REL_AREA or rel > PANEL_MAX_REL_AREA:
            continue
        aspect = h / max(w, 1e-6)
        # Prefer tall-ish panels; penalize extremely flat boxes.
        aspect_score = 1.0 if 0.8 <= aspect <= 6.0 else 0.5
        # Prefer moderate size (not tiny, not nearly full frame).
        size_score = 1.0 - abs(rel - 0.25)
        score = c["confidence"] * aspect_score * (0.5 + size_score)
        scored.append((score, c))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def expand_bbox(bbox: list[float], img_w: int, img_h: int, padding_frac: float,
                bottom_extra_frac: float = 0.0) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad_x = w * padding_frac
    pad_top = h * padding_frac
    pad_bottom = h * (padding_frac + bottom_extra_frac)
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_top))
    x2 = min(img_w, int(x + w + pad_x))
    y2 = min(img_h, int(y + h + pad_bottom))
    return x1, y1, x2, y2


def is_door_class_name(name: str) -> bool:
    n = name.lower()
    return "door open" in n or "door close" in n


def keep_main_button_column(buttons: list[dict], panel_bbox: list[float],
                            column_frac: float, bottom_door_frac: float) -> list[dict]:
    """Keep floor column in the upper/mid panel; preserve full width in bottom door strip."""
    if len(buttons) <= 1:
        return buttons
    px, py, pw, ph = panel_bbox
    y_door = py + ph * (1.0 - bottom_door_frac)
    floor_btns = [d for d in buttons if d["center"][1] < y_door]
    door_strip = [d for d in buttons if d["center"][1] >= y_door]
    if len(floor_btns) <= 1:
        floor_kept = floor_btns
    else:
        col_x = float(np.median([d["center"][0] for d in floor_btns]))
        max_dx = max(pw * column_frac, 20.0)
        floor_kept = [d for d in floor_btns if abs(d["center"][0] - col_x) <= max_dx]
        if not floor_kept:
            floor_kept = floor_btns
    return floor_kept + door_strip


def assign_door_pair_labels(buttons: list[dict], panel_bbox: list[float],
                            bottom_frac: float) -> list[dict]:
    """Label bottom-row left/right pair as door open / door close (PMB-style layout)."""
    px, py, pw, ph = panel_bbox
    y_cut = py + ph * (1.0 - bottom_frac)
    upper = [b for b in buttons if b["center"][1] < y_cut]
    bottom = [b for b in buttons if b["center"][1] >= y_cut]
    if not bottom:
        return buttons

    row_tol = max(ph * 0.08, 35.0)
    bottom.sort(key=lambda b: b["center"][1], reverse=True)
    anchor_y = bottom[0]["center"][1]
    row = [b for b in bottom if abs(b["center"][1] - anchor_y) <= row_tol]
    extras = [b for b in bottom if b not in row]

    if len(row) == 1:
        b = dict(row[0])
        if is_door_class_name(b["class_name"]):
            b["class_name"] = DOOR_OPEN_LABEL if "open" in b["class_name"].lower() else DOOR_CLOSE_LABEL
        elif b["center"][0] < px + pw * 0.5:
            b["class_name"] = DOOR_OPEN_LABEL
        else:
            b["class_name"] = DOOR_CLOSE_LABEL
        return sorted(upper + extras + [b], key=lambda d: d["center"][1])

    row.sort(key=lambda b: b["center"][0])
    left, right = dict(row[0]), dict(row[-1])
    left["class_name"] = DOOR_OPEN_LABEL
    right["class_name"] = DOOR_CLOSE_LABEL
    return sorted(upper + extras + [left, right], key=lambda d: d["center"][1])


def predict(model: YOLO, image, conf: float, imgsz: int, iou: float, device):
    return model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        max_det=300,
    )[0]


def coarse_type_from_class_name(class_name: str) -> str:
    lowered = class_name.lower()
    if "door open" in lowered or "door close" in lowered:
        return COARSE_DOOR_BUTTON_CANDIDATE
    if "floor" in lowered:
        return COARSE_FLOOR_BUTTON_CANDIDATE
    return COARSE_BUTTON_CANDIDATE


def buttons_to_detection_array(
    buttons: list[dict],
    image_path: str,
    image_width: int,
    image_height: int,
    detector_latency_ms: float,
) -> DetectionArray:
    detections: list[Detection] = []
    next_id = 0

    for button in buttons:
        native_bbox = button.get("bbox")
        if not native_bbox or len(native_bbox) < 4:
            continue

        xyxy = clip_xyxy(xywh_to_xyxy(native_bbox[:4]), image_width, image_height)
        if not is_valid_box(xyxy, image_width, image_height):
            continue

        class_name = str(button.get("class_name") or "")
        confidence = float(button.get("confidence", -1.0))
        rel_area = button.get("rel_area")
        if rel_area is None:
            _, _, w, h = native_bbox[:4]
            rel_area = (float(w) * float(h)) / float(max(image_width * image_height, 1))

        detections.append(Detection(
            detection_id=f"yolo_world_{next_id:03d}",
            bbox_xyxy=xyxy,
            center_xy=compute_center(xyxy),
            score=confidence,
            coarse_type=coarse_type_from_class_name(class_name),
            detector_class=class_name,
            raw={
                "native_class": class_name,
                "rel_area": float(rel_area),
                "native_bbox": [float(v) for v in native_bbox[:4]],
            },
        ))
        next_id += 1

    return DetectionArray(
        schema_version=SCHEMA_VERSION,
        source_detector=BACKEND_YOLO_WORLD,
        image_path=image_path,
        image_width=int(image_width),
        image_height=int(image_height),
        detector_latency_ms=float(detector_latency_ms),
        detections=detections,
    )


_OVERLAY_LIME = (0, 255, 0)
_OVERLAY_CYAN = (255, 255, 0)


def draw_contract_overlay_gdino_style(img: np.ndarray, det_array: DetectionArray, title: str) -> np.ndarray:
    """Overlay with lime boxes and cyan coordinate text (matches gdino_detect.py style)."""
    out = img.copy()
    for det in det_array.detections:
        x_min, y_min, x_max, y_max = (int(round(v)) for v in det.bbox_xyxy)
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), _OVERLAY_LIME, 2)
        cx, cy = (int(round(v)) for v in det.center_xy)
        cv2.circle(out, (cx, cy), 3, _OVERLAY_LIME, -1)
        coord_text = f"[{x_min},{y_min} {x_max},{y_max}] c=({cx},{cy})"
        cv2.putText(
            out, coord_text, (x_min, max(y_min - 18, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _OVERLAY_CYAN, 1, cv2.LINE_AA,
        )
        label = f"{det.coarse_type} {det.score:.2f}"
        cv2.putText(
            out, label, (x_min, max(y_min - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, _OVERLAY_LIME, 1, cv2.LINE_AA,
        )
    if not det_array.detections:
        cv2.putText(
            out, "NO DETECTIONS", (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, _OVERLAY_LIME, 2, cv2.LINE_AA,
        )
    cv2.putText(
        out, title, (10, out.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _OVERLAY_LIME, 1, cv2.LINE_AA,
    )
    return out


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        _, run_dir = allocate_run_dir(args.output_root, PIPELINE)

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {image_path}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge([l, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img_h, img_w = img.shape[:2]
    img_area = img_w * img_h

    t0 = time.time()
    panel_model = YOLO(args.model)
    button_model = YOLO(args.model)
    load_s = time.time() - t0

    # Stage A: panel (separate model instance avoids CLIP device issues on re-set_classes)
    panel_model.set_classes(PANEL_CLASSES)
    t1 = time.time()
    panel_result = predict(panel_model, img, args.panel_conf, args.imgsz, args.iou, args.device)
    panel_infer_s = time.time() - t1

    raw_panels = panel_result.boxes
    raw_panel_n = 0 if raw_panels is None else len(raw_panels)
    panel_candidates = boxes_to_dicts(
        raw_panels, PANEL_CLASSES,
        area_ref=img_area, min_wh=20, max_wh=max(img_w, img_h),
        max_rel_area=PANEL_MAX_REL_AREA,
    )
    panel_candidates = merge_vertical_panels(panel_candidates, img_w, img_h)
    panel = select_panel(panel_candidates, img_w, img_h)
    used_full_frame = False
    if panel is None:
        # Panel not found — fallback to right-half ROI (panel is usually on the right side of elevator images)
        x1 = int(img_w * 0.45)
        panel_bbox = [float(x1), 0.0, float(img_w - x1), float(img_h)]
        crop = (x1, 0, img_w, img_h)
    else:
        panel_bbox = panel["bbox"]
        crop = expand_bbox(panel_bbox, img_w, img_h, args.panel_padding, args.panel_bottom_pad_frac)

    x1, y1, x2, y2 = crop
    crop_img = img[y1:y2, x1:x2]
    crop_w, crop_h = x2 - x1, y2 - y1
    crop_area = crop_w * crop_h

    # Stage B: buttons inside panel crop
    button_model.set_classes(BUTTON_CLASSES)
    t2 = time.time()
    button_result = predict(button_model, crop_img, args.button_conf, args.imgsz, args.iou, args.device)
    first_button_s = time.time() - t2

    warm = []
    for _ in range(max(0, args.runs - 1)):
        t = time.time()
        predict(button_model, crop_img, args.button_conf, args.imgsz, args.iou, args.device)
        warm.append(time.time() - t)

    raw_buttons = button_result.boxes
    raw_button_n = 0 if raw_buttons is None else len(raw_buttons)
    buttons = boxes_to_dicts(
        raw_buttons, BUTTON_CLASSES,
        offset_x=x1, offset_y=y1,
        area_ref=crop_area,
        min_wh=BUTTON_MIN_WH, max_wh=BUTTON_MAX_WH,
        max_rel_area=BUTTON_MAX_REL_AREA,
    )
    # Outside panels (small area, only 2 side-by-side buttons) should not be column-filtered.
    panel_area_frac = (panel_bbox[2] * panel_bbox[3]) / img_area
    if panel_area_frac >= 0.10:
        buttons = keep_main_button_column(
            buttons, panel_bbox, args.main_column_frac, args.bottom_door_frac,
        )
    buttons = drop_panel_top_strip(buttons, panel_bbox, args.panel_top_drop_frac)
    buttons = dedupe_buttons(buttons, args.dedupe_dist)
    buttons = assign_door_pair_labels(buttons, panel_bbox, args.bottom_door_frac)

    stem = image_path.stem
    contract_path = run_dir / f"{stem}_contract.json"
    overlay_path = run_dir / f"{stem}_overlay.jpg"

    detector_latency_ms = round((panel_infer_s + first_button_s) * 1000.0, 1)
    det_array = buttons_to_detection_array(
        buttons, str(image_path), img_w, img_h, detector_latency_ms,
    )
    validation_errors = validate_detection_array(det_array)
    if validation_errors:
        raise SystemExit("Contract validation failed:\n  " + "\n  ".join(validation_errors))

    write_contract_json(contract_path, det_array)

    title = (f"yolo_world buttons={len(det_array.detections)} "
             f"panel={panel_infer_s:.2f}s btn={first_button_s:.2f}s")
    overlay = draw_contract_overlay_gdino_style(img, det_array, title)
    oh, ow = overlay.shape[:2]
    if max(oh, ow) > 1400:
        scale = 1400 / max(oh, ow)
        overlay = cv2.resize(overlay, (int(ow * scale), int(oh * scale)))
    cv2.imwrite(str(overlay_path), overlay)

    print(json.dumps({
        "run_dir": str(run_dir),
        "image": image_path.name,
        "panel_fallback": used_full_frame,
        "raw_panels": raw_panel_n,
        "raw_buttons": raw_button_n,
        "buttons_kept": len(buttons),
        "contract_detections": len(det_array.detections),
        "panel_infer_s": round(panel_infer_s, 3),
        "button_infer_s": round(first_button_s, 3),
        "load_time_s": round(load_s, 3),
        "detector_latency_ms": detector_latency_ms,
        "contract": str(contract_path),
        "overlay": str(overlay_path),
    }, indent=2))
    for det in det_array.detections:
        print(
            f"  {det.detector_class!r} {det.coarse_type} "
            f"score={det.score} center={list(det.center_xy)}"
        )


if __name__ == "__main__":
    main()
