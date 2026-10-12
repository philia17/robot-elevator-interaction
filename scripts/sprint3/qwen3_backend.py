#!/usr/bin/env python3
"""Qwen3-VL inference for live bbox + per-button labels (llama-mtmd-cli subprocess)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import cv2

LLAMA_BIN = Path(os.environ.get(
    "LLAMA_BIN",
    "/data/tools/llama.cpp/build-no-vmm/bin/llama-mtmd-cli",
)).expanduser()

MODEL_PROFILES = {
    "qwen3-2b": {
        "label": "Qwen3-VL-2B",
        "model": "/data/models/qwen3-vl-2b/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj": "/data/models/qwen3-vl-2b/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
        "timeout": 90,
        "default_interval": 6.0,
    },
    "qwen3-8b": {
        "label": "Qwen3-VL-8B",
        "model": "/data/models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj": "/data/models/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
        "timeout": 180,
        "default_interval": 20.0,
    },
}

BBOX_DETECT_PROMPT = """\
You are inspecting a camera image. The operator is facing an elevator or hall-call control panel.

Find every visible pressable button (floor numbers, door open/close, hall call up/down).

Output ONLY a JSON array. Each item:
{{"bbox_2d":[x1,y1,x2,y2],"label":"visible text","description":"short phrase"}}

Rules:
- bbox_2d: integer pixel coordinates in THIS image, origin top-left, x right, y down.
- label: text on the button (e.g. "3", "G", "door_open").
- description: max 8 English words.

If you truly see zero pressable buttons, output [].
Do not use markdown. No text outside the JSON array.
"""

BBOX_DETECT_PROMPT_ARMED = """\
The operator confirms this image shows an elevator or hall-call button panel.

Detect ALL visible pressable buttons. Be inclusive: floor buttons, door open/close, \
hall call, alarm, and labeled square keys.

Output ONLY a JSON array. Each item:
{{"bbox_2d":[x1,y1,x2,y2],"label":"visible text","description":"short phrase"}}

bbox_2d must be integer pixel coordinates in this image (width={img_w}, height={img_h}).
Never return [] if any button-like control is visible.

No markdown. No text outside the JSON array.
"""

BBOX_DETECT_PROMPT_FALLBACK = """\
List every button-like control in this elevator panel image.
Output ONLY a JSON array with bbox_2d [x1,y1,x2,y2], label, and description per item.
Image size: {img_w}x{img_h} pixels. Use pixel coordinates.
"""


def build_detect_prompt(img_w: int, img_h: int, panel_armed: bool = False) -> str:
    if panel_armed:
        return BBOX_DETECT_PROMPT_ARMED.format(img_w=img_w, img_h=img_h)
    return BBOX_DETECT_PROMPT


def list_model_keys():
    return list(MODEL_PROFILES.keys())


def profile(model_key: str) -> dict:
    if model_key not in MODEL_PROFILES:
        raise KeyError(f"Unknown model {model_key!r}; choose {list_model_keys()}")
    return MODEL_PROFILES[model_key]


def _clean_response(stdout: str, stderr: str) -> str:
    text = (stdout or "").strip()
    if not text and stderr:
        text = stderr.strip()
    if "assistant" in text.lower():
        parts = re.split(r"assistant\s*", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()
    return text


def _salvage_json_objects(text: str):
    """Extract complete JSON objects from a truncated model array."""
    start = text.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    items = []
    idx = start + 1
    n = len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or text[idx] == "]":
            break
        if text[idx] != "{":
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            items.append(obj)
        idx = end
    return items


def _extract_json_array(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            for key in ("buttons", "detections", "items", "results"):
                if isinstance(data.get(key), list):
                    return data[key], None
        if isinstance(data, list):
            return data, None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*", cleaned)
    if not match:
        return None, "no JSON array in model output"
    fragment = match.group()
    try:
        return json.loads(fragment), None
    except json.JSONDecodeError:
        salvaged = _salvage_json_objects(fragment)
        if salvaged:
            return salvaged, None
        return None, "JSON parse failed on model output"


def bbox_from_item(item: dict, img_w: int, img_h: int):
    if "bbox_2d" in item and isinstance(item["bbox_2d"], (list, tuple)):
        coords = [float(v) for v in item["bbox_2d"][:4]]
        x1, y1, x2, y2 = coords
    elif all(k in item for k in ("x_min", "y_min", "x_max", "y_max")):
        x1 = float(item["x_min"])
        y1 = float(item["y_min"])
        x2 = float(item["x_max"])
        y2 = float(item["y_max"])
    else:
        return None

    # Qwen3-VL may emit 0–1000 normalized coords; convert if values fit that range.
    if max(x1, y1, x2, y2) <= 1000.0 and max(x1, y1, x2, y2) > max(img_w, img_h):
        x1 = x1 * img_w / 1000.0
        x2 = x2 * img_w / 1000.0
        y1 = y1 * img_h / 1000.0
        y2 = y2 * img_h / 1000.0
    elif max(x1, y1, x2, y2) <= 1000.0 and img_w <= 1000 and img_h <= 1000:
        # Small images: treat as normalized when coords stay in 0–1000 grid.
        if x2 <= 1000 and y2 <= 1000 and (x2 > img_w or y2 > img_h):
            x1 = x1 * img_w / 1000.0
            x2 = x2 * img_w / 1000.0
            y1 = y1 * img_h / 1000.0
            y2 = y2 * img_h / 1000.0

    x1, y1, x2, y2 = [int(round(v)) for v in (x1, y1, x2, y2)]

    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w - 1, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def parse_detection_response(text: str, img_w: int, img_h: int):
    """Return (detections, status_message, parse_error)."""
    items, err = _extract_json_array(text)
    if err:
        return [], "VLM output could not be parsed", err

    detections = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        bbox = bbox_from_item(item, img_w, img_h)
        if bbox is None:
            continue
        label = str(item.get("label") or item.get("name") or "?").strip()
        description = str(item.get("description") or label).strip()
        detections.append({
            "bbox": bbox,
            "label": label,
            "description": description,
        })

    if not detections:
        return [], "No elevator buttons detected", None
    n = len(detections)
    return detections, f"Detected {n} button{'s' if n != 1 else ''}", None


def prepare_frame_for_vlm(bgr_image, upscale: float = 2.0):
    """Upscale frame so small live-camera buttons are easier for the VLM."""
    if upscale <= 1.0:
        return bgr_image, 1.0
    h, w = bgr_image.shape[:2]
    new_w, new_h = int(w * upscale), int(h * upscale)
    enlarged = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return enlarged, upscale


def scale_detections_to_source(detections, scale: float):
    if scale <= 1.0 or not detections:
        return detections
    out = []
    inv = 1.0 / scale
    for det in detections:
        det = dict(det)
        x, y, w, h = det["bbox"]
        det["bbox"] = (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
        out.append(det)
    return out


def _write_debug_response(response: str, suffix: str = "last"):
    log_dir = Path(os.environ.get("VLM_DEBUG_DIR", "/data/datasets/pmb/snapshots"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"vlm_{suffix}_response.txt"
        path.write_text(response or "(empty)", encoding="utf-8")
    except OSError:
        pass


def run_vlm_detect(
    bgr_image,
    model_key: str,
    prompt: str | None = None,
    panel_armed: bool = False,
    upscale: float = 2.0,
):
    """Run one VLM pass on a BGR image. Returns result dict."""
    prof = profile(model_key)
    model_path = Path(prof["model"]).expanduser()
    mmproj_path = Path(prof["mmproj"]).expanduser()
    llama_bin = LLAMA_BIN

    if not llama_bin.is_file():
        raise FileNotFoundError(f"llama-mtmd-cli not found: {llama_bin}")
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not mmproj_path.is_file():
        raise FileNotFoundError(f"mmproj not found: {mmproj_path}")

    src_h, src_w = bgr_image.shape[:2]
    infer_image, scale = prepare_frame_for_vlm(bgr_image, upscale=upscale)
    h, w = infer_image.shape[:2]
    if prompt is None:
        prompt = build_detect_prompt(w, h, panel_armed=panel_armed)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, infer_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cmd = [
            str(llama_bin),
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "--image", tmp_path,
            "-p", prompt,
            "-n", "1024",
            "--no-perf",
            "--no-warmup",
            "-ngl", "0",
        ]
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=prof["timeout"],
            env={**os.environ, "PATH": f"{Path.home() / '.local/bin'}:{os.environ.get('PATH', '')}"},
        )
        elapsed = time.perf_counter() - t0
        response = _clean_response(result.stdout, result.stderr)
        _write_debug_response(response)
        detections, status, parse_error = parse_detection_response(response, w, h)

        # Retry once with a simpler prompt when armed but still empty.
        if panel_armed and not detections and not parse_error:
            fallback = BBOX_DETECT_PROMPT_FALLBACK.format(img_w=w, img_h=h)
            cmd_fb = [
                str(llama_bin), "-m", str(model_path), "--mmproj", str(mmproj_path),
                "--image", tmp_path, "-p", fallback, "-n", "1024",
                "--no-perf", "--no-warmup", "-ngl", "0",
            ]
            result2 = subprocess.run(
                cmd_fb, capture_output=True, text=True,
                timeout=max(30, prof["timeout"] - int(elapsed)),
                env={**os.environ, "PATH": f"{Path.home() / '.local/bin'}:{os.environ.get('PATH', '')}"},
            )
            elapsed = time.perf_counter() - t0
            response = _clean_response(result2.stdout, result2.stderr)
            _write_debug_response(response, "fallback")
            detections, status, parse_error = parse_detection_response(response, w, h)

        detections = scale_detections_to_source(detections, scale)
        if detections and src_w != w:
            # Re-clip to source image bounds after scale-back.
            fixed = []
            for det in detections:
                bbox = bbox_from_item(
                    {"bbox_2d": [
                        det["bbox"][0], det["bbox"][1],
                        det["bbox"][0] + det["bbox"][2],
                        det["bbox"][1] + det["bbox"][3],
                    ]},
                    src_w, src_h,
                )
                if bbox:
                    det = dict(det)
                    det["bbox"] = bbox
                    fixed.append(det)
            detections = fixed
            if fixed:
                status = f"Detected {len(fixed)} button{'s' if len(fixed) != 1 else ''}"

        return {
            "model_key": model_key,
            "model_label": prof["label"],
            "elapsed": elapsed,
            "returncode": result.returncode,
            "response": response,
            "detections": detections,
            "status_message": status,
            "parse_error": parse_error,
            "error": None if result.returncode == 0 else f"llama exit {result.returncode}",
        }
    except subprocess.TimeoutExpired:
        return {
            "model_key": model_key,
            "model_label": prof["label"],
            "elapsed": prof["timeout"],
            "returncode": 124,
            "response": "",
            "detections": [],
            "status_message": "VLM inference timed out",
            "parse_error": "timeout",
            "error": "timeout",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
