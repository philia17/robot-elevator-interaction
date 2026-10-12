#!/usr/bin/env python3
"""
Sprint 2 - Elevator Panel Recognition Demo (Person A).

Phone photo -> llama.cpp VLM -> annotated image / JSON result.

Examples:
  python3 scripts/sprint2/elevator_demo_qwen3.py --mode elevator --image demo_images/golden/ieb_inside_panel_full_01.jpg
  python3 scripts/sprint2/elevator_demo_qwen3.py --mode elevator --image-folder demo_images/
  python3 scripts/sprint2/elevator_demo_qwen3.py --mode elevator \
    --image-folder /data/datasets/library/ --max-dim 768
  # Qwen2.5-VL-3B baseline (separate script):
  python3 scripts/sprint2/elevator_demo.py --mode elevator \
    --image-folder /data/datasets/library/ --max-dim 768
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import cv2


LLAMA_BIN = Path(os.environ.get(
    "LLAMA_BIN",
    "/data/tools/llama.cpp/build/bin/llama-mtmd-cli",
)).expanduser()
MODEL_PATH = Path(os.environ.get(
    "VLM_MODEL",
    "/data/models/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q8_0.gguf",
)).expanduser()
MMPROJ_PATH = Path(os.environ.get(
    "VLM_MMPROJ",
    "/data/models/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
)).expanduser()

SUPPORTED_BUTTON_TYPES = {
    "floor_button",
    "door_open",
    "door_close",
    "alarm",
    "hall_call_up",
    "hall_call_down",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXCLUDED_IMAGE_DIR_NAMES = {"output", "outputs"}
DEFAULT_IMAGE_MIN_TOKENS = 0

# Keep this tiny for Sprint 2. Sprint 3 should use the pipeline config.
BUILDING_MAPS = {
    "building_a": {"4": "3B"},
    "ieb": {},
    "library": {},
    "pb": {},
    "db": {},
    "trent": {},
}

FREE_TEXT_PROMPT = """\
You are inspecting ONE elevator or hall-call control panel in a photo.

List ONLY physical pressable buttons that a passenger would use.

Include:
- floor buttons with labels that are actually visible on large elevator buttons
- door open and door close buttons
- alarm or emergency bell buttons only when the panel type hint allows alarm
- hall-call up/down buttons only when the panel type hint says this is a hall-call panel

Do NOT list:
- floor labels that are not visibly printed on a button; do not complete a common sequence
- floor/status display numbers or arrows
- indicator lights, lit rings, logos, brand names, stickers, warning signs
- key switches, service switches, fireman's switches, Lgt/FAN/Att controls
- service buttons labeled start up, by pass, start down, VIP, FAN, Att, Lgt
- intercom or phone keypad numbers, telephone units, or emergency instruction labels
- emergency call signs, emergency instruction stickers, or telephone/intercom panels
- hall_call_up or hall_call_down when a main inside floor-button group is visible

Scan the panel systematically:
1. First find the main floor-button group and list every visible floor label exactly as printed.
2. Then find door open, door close, alarm, and hall-call up/down buttons.
3. Ignore any lower maintenance/service-control row even if it looks pressable.
4. If the image is an inside elevator panel with floor buttons, hall-call buttons should usually be absent.

Output in this exact readable format:
Buttons found:
- label: "<visible label>"; type: floor_button; position: <rough position>
- label: "door_open"; type: door_open; position: <rough position>

Allowed type values are only:
floor_button, door_open, door_close, alarm, hall_call_up, hall_call_down.
Use alarm only for a standalone physical alarm/bell button.

If a button is ambiguous, include it only if it is clearly a physical button.
"""

JSON_PROMPT_TEMPLATE = """\
You are inspecting ONE elevator or hall-call control panel in a photo.

Output ONLY valid JSON. Do not include markdown fences, explanations, comments,
or extra text before or after the JSON.

Detect ONLY physical pressable buttons that a passenger would use.

Include:
- floor buttons with labels that are actually visible on large elevator buttons
- door open and door close buttons
- alarm or emergency bell buttons only when the panel type hint allows alarm
- hall-call up/down buttons only when the panel type hint says this is a hall-call panel

Do NOT include:
- floor labels that are not visibly printed on a button; do not complete a common sequence
- floor/status display numbers or arrows
- indicator lights, lit rings, logos, brand names, stickers, warning signs
- key switches, service switches, fireman's switches, Lgt/FAN/Att controls
- service buttons labeled start up, by pass, start down, VIP, FAN, Att, Lgt
- intercom or phone keypad numbers, telephone units, or emergency instruction labels
- emergency call signs, emergency instruction stickers, or telephone/intercom panels
- hall_call_up or hall_call_down when a main inside floor-button group is visible

Scan the panel systematically:
1. First find the main floor-button group and list every visible floor label exactly as printed.
2. Then find door open, door close, alarm, and hall-call up/down buttons.
3. Ignore any lower maintenance/service-control row even if it looks pressable.
4. If the image is an inside elevator panel with floor buttons, hall-call buttons should usually be absent.

Use this exact schema (include bounding box coordinates in pixels):
{{
  "interface_type": "elevator_panel",
  "image_width": {image_width},
  "image_height": {image_height},
  "buttons": [
    {{
      "id": "button_1",
      "button_type": "floor_button",
      "label": "3",
      "pressable": true
    }}
  ]
}}

Allowed button_type values:
floor_button, door_open, door_close, alarm, hall_call_up, hall_call_down.

Reminder: include ALL visible passenger buttons. A complete inside panel has 5-8 buttons.
"""


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_LOG_RE = re.compile(r"^\d+(?:\.\d+){2,}\s")
CHAT_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")

NOISE_PREFIXES = (
    "ggml_",
    "llama_",
    "mtmd",
    "common_",
    "load_",
    "print_info",
    "sampler",
    "encoding mtmd",
    "decoding",
    "encode",
    "decode",
    "main:",
    "system_info",
    "build:",
    "warning:",
    "warn:",
    "error:",
    "cublas",
    "for normal use",
)

ALLOWED_TYPES_BY_PANEL = {
    "inside": {"floor_button", "door_open", "door_close"},
    "accessible": {"floor_button", "door_open", "door_close", "alarm"},
    "hall": {"hall_call_up", "hall_call_down"},
}

EXCLUDED_LABELS = {
    "start_up",
    "start up",
    "start_down",
    "start down",
    "by_pass",
    "by pass",
    "vip",
    "fan",
    "att",
    "lgt",
    "light",
    "1000",
    "emergency",
}


def infer_panel_type(image_path, requested):
    if requested and requested != "auto":
        return requested

    name = Path(image_path).name.lower()
    if "hall_call" in name:
        return "hall"
    if "accessible" in name:
        return "accessible"
    return "inside"


def panel_type_instruction(panel_type):
    if panel_type == "hall":
        return (
            "Panel type hint: this is an OUTSIDE hall-call panel. "
            "Allowed button types for this image: hall_call_up, hall_call_down. "
            "Do not output floor_button, door_open, door_close, or alarm. "
            "Do not treat floor/status displays as buttons."
        )
    if panel_type == "accessible":
        return (
            "Panel type hint: this is an INSIDE accessible elevator panel. "
            "Allowed button types for this image: floor_button, door_open, door_close, alarm. "
            "Never output hall_call_up or hall_call_down."
        )
    return (
        "Panel type hint: this is an INSIDE elevator panel. "
        "Allowed button types for this image: floor_button, door_open, door_close. "
        "Never output hall_call_up, hall_call_down, or alarm."
    )


def build_prompt(mode, image_width, image_height, panel_type="inside",
                 building=None, target_floor=None):
    if mode == "elevator-json":
        prompt = JSON_PROMPT_TEMPLATE.format(
            image_width=image_width,
            image_height=image_height,
        )
    else:
        prompt = FREE_TEXT_PROMPT

    prompt += "\n" + panel_type_instruction(panel_type) + "\n"

    if target_floor and building:
        building_key = str(building).lower()
        floor_key = str(target_floor)
        effective = BUILDING_MAPS.get(building_key, {}).get(floor_key, floor_key)
        prompt += (
            "\nTarget selection note for downstream code: "
            f"requested floor={floor_key}, building={building_key}, "
            f"effective visible label={effective}. "
            "Do not guess the target; still list all visible passenger buttons."
        )
    return prompt


def prepare_inference_image(image_path, max_dim):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    original_h, original_w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(original_w, original_h)))
    resized_w = max(1, int(round(original_w * scale)))
    resized_h = max(1, int(round(original_h * scale)))

    if (resized_w, resized_h) == (original_w, original_h):
        resized = img
    else:
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    cv2.imwrite(str(tmp_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 88])

    meta = {
        "original_width": original_w,
        "original_height": original_h,
        "model_width": resized_w,
        "model_height": resized_h,
        "scale_to_original_x": original_w / resized_w,
        "scale_to_original_y": original_h / resized_h,
    }
    return img, tmp_path, meta


def strip_chat_template(text):
    text = text.replace("\r\n", "\n")

    marker = "<|im_start|>assistant"
    if marker in text:
        text = text.split(marker)[-1]
        text = text.split("<|im_end|>")[0]

    text = CHAT_TOKEN_RE.sub("", text)
    return text.strip()


def is_noise_line(line):
    s = ANSI_RE.sub("", line).strip()
    if not s:
        return True

    lower = s.lower()
    if TIMESTAMP_LOG_RE.match(lower):
        return True
    if any(lower.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True
    if "llama.cpp" in lower or "llama_model_loader" in lower:
        return True
    if "prompt eval time" in lower or "eval time" in lower or "total time" in lower:
        return True
    if lower.startswith(("model loaded", "server is listening", "sampling params")):
        return True
    return False


def clean_model_response(stdout, stderr=""):
    # llama-mtmd-cli normally writes the answer to stdout and logs to stderr.
    # Prefer stdout. Only fall back to stderr if stdout has no usable content.
    candidates = [stdout or "", (stdout or "") + "\n" + (stderr or ""), stderr or ""]

    for candidate in candidates:
        candidate = strip_chat_template(candidate)
        kept = []
        for line in candidate.splitlines():
            if not is_noise_line(line):
                kept.append(ANSI_RE.sub("", line).rstrip())
        cleaned = "\n".join(kept).strip()
        if cleaned:
            if "Buttons found:" in cleaned:
                cleaned = cleaned[cleaned.index("Buttons found:") :]
            return cleaned

    return ""


def parse_free_text_line(line):
    label_match = re.search(r'label:\s*"([^"]+)"', line)
    type_match = re.search(r'type:\s*([a-zA-Z0-9_]+)', line)
    label = label_match.group(1).strip() if label_match else ""
    button_type = type_match.group(1).strip() if type_match else ""
    return label, button_type


def filter_free_text_response(text, panel_type):
    allowed = ALLOWED_TYPES_BY_PANEL.get(panel_type, ALLOWED_TYPES_BY_PANEL["inside"])
    kept = []
    removed = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            kept.append(line)
            continue

        label, button_type = parse_free_text_line(stripped)
        label_key = label.lower().replace("-", "_")
        if button_type not in allowed or label_key in EXCLUDED_LABELS:
            removed.append(line)
            continue
        kept.append(line)

    filtered = "\n".join(kept).strip()
    return filtered or text, removed


def run_llama(prepared_image_path, prompt, timeout, image_min_tokens=DEFAULT_IMAGE_MIN_TOKENS,
              verbose=False):
    cmd = [
        str(LLAMA_BIN),
        "-m",
        str(MODEL_PATH),
        "--mmproj",
        str(MMPROJ_PATH),
        "--image",
        str(prepared_image_path),
        "-p",
        prompt,
        "-n",
        "1024",
        "--mmproj-offload",
        "--no-perf",
        "--no-warmup",
    ]
    if image_min_tokens:
        cmd.extend(["--image-min-tokens", str(image_min_tokens)])

    if verbose:
        print("[cmd] " + " ".join(cmd))

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "PATH": f"{Path.home() / '.local/bin'}:{os.environ.get('PATH', '')}",
            },
        )
        elapsed = time.time() - t0
    except FileNotFoundError:
        elapsed = time.time() - t0
        return {
            "response": "",
            "stdout": "",
            "stderr": f"llama binary not found: {LLAMA_BIN}",
            "returncode": 127,
            "elapsed": elapsed,
            "error": f"llama binary not found: {LLAMA_BIN}",
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return {
            "response": clean_model_response(stdout, stderr),
            "stdout": stdout,
            "stderr": stderr,
            "returncode": 124,
            "elapsed": elapsed,
            "error": f"inference timed out after {timeout}s",
        }

    response = clean_model_response(result.stdout, result.stderr)
    return {
        "response": response,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "error": None if result.returncode == 0 else f"llama exited with {result.returncode}",
    }


def extract_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    # Try clean parse first
    try:
        data = json.loads(cleaned)
        return data, None
    except json.JSONDecodeError:
        pass

    # Try fixing truncation: close unclosed braces/brackets (handle trailing commas)
    truncated_suffixes = [
        "}]}",       # button } + array ] + object }
        "]}",        # array ] + object }
        "]",         # just close array
        "}",         # just close object
        "]}]}",      # extra close button + array + object
        "}]",        # close button + array
    ]
    for suffix in truncated_suffixes:
        candidate = cleaned.rstrip().rstrip(",") + suffix
        try:
            data = json.loads(candidate)
            return data, None
        except json.JSONDecodeError:
            continue

    # Last resort: raw_decode from first '{' looking for the outermost object
    decoder = json.JSONDecoder()
    last_error = None
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(cleaned[idx:])
            # Must be the root object (has buttons array), not a nested button
            if isinstance(data, dict) and "buttons" in data:
                return data, None
            if isinstance(data, dict) and "interface_type" in data:
                return data, None
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error:
        return None, str(last_error)
    return None, "no JSON object found"


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_buttons(data):
    errors = []
    if not isinstance(data, dict):
        return False, ["root is not an object"]

    for field in ("interface_type", "image_width", "image_height", "buttons"):
        if field not in data:
            errors.append(f"missing top-level field: {field}")

    image_width = data.get("image_width")
    image_height = data.get("image_height")
    if not is_number(image_width) or image_width <= 0:
        errors.append("image_width must be a positive number")
    if not is_number(image_height) or image_height <= 0:
        errors.append("image_height must be a positive number")

    buttons = data.get("buttons")
    if not isinstance(buttons, list):
        errors.append("buttons must be a list")
        return False, errors

    ids = set()
    # Labels-only or full grounding: adapt required fields
    has_bbox = all(f in buttons[0] for f in ("x_min", "y_min", "x_max", "y_max")) if buttons else False
    required = ["id", "button_type", "label", "pressable"]
    if has_bbox:
        required += ["x_min", "y_min", "x_max", "y_max", "center_x", "center_y",
                     "occluded", "disabled", "score"]

    for idx, button in enumerate(buttons):
        if not isinstance(button, dict):
            errors.append(f"buttons[{idx}] is not an object")
            continue

        for field in required:
            if field not in button:
                errors.append(f"buttons[{idx}] missing field: {field}")

        button_id = button.get("id", f"buttons[{idx}]")
        if not isinstance(button_id, str) or not button_id.strip():
            errors.append(f"buttons[{idx}] id must be a non-empty string")
        elif button_id in ids:
            errors.append(f"duplicate button id: {button_id}")
        ids.add(button_id)

        button_type = button.get("button_type")
        if button_type not in SUPPORTED_BUTTON_TYPES:
            errors.append(f"{button_id}: unsupported button_type {button_type!r}")

        label = button.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"{button_id}: label must be a non-empty string")

        for field in ("pressable",):
            if not isinstance(button.get(field), bool):
                errors.append(f"{button_id}: {field} must be boolean")

        # Coordinate validation: only if bbox fields are present
        if has_bbox:
            x1 = button.get("x_min")
            y1 = button.get("y_min")
            x2 = button.get("x_max")
            y2 = button.get("y_max")
            cx = button.get("center_x")
            cy = button.get("center_y")
            coords = (x1, y1, x2, y2, cx, cy)
            if not all(is_number(v) for v in coords):
                errors.append(f"{button_id}: bbox and center fields must be numbers")
                continue
            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                errors.append(f"{button_id}: bbox out of image bounds")
            if x1 >= x2 or y1 >= y2:
                errors.append(f"{button_id}: bbox has invalid ordering")
            if cx < x1 - 50 or cx > x2 + 50 or cy < y1 - 50 or cy > y2 + 50:
                errors.append(f"{button_id}: center far outside bbox")
            for field in ("occluded", "disabled"):
                if not isinstance(button.get(field), bool):
                    errors.append(f"{button_id}: {field} must be boolean")
            score = button.get("score")
            if not is_number(score) or score < 0 or score > 1:
                errors.append(f"{button_id}: score must be a number in [0, 1]")

    return len(errors) == 0, errors


def wrap_overlay_lines(text, width=88, max_lines=9):
    lines = []
    for raw in text.strip().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        wrapped = textwrap.wrap(raw, width=width) or [raw]
        lines.extend(wrapped)
        if len(lines) >= max_lines:
            break
    return lines[:max_lines]


def draw_status(img, status, color):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, max(0, h - 34)), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    cv2.putText(
        img,
        status[:110],
        (10, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return img


def parse_free_text_to_json(free_text, image_width, image_height):
    """Convert free-text button list to labels-only JSON. Returns dict or None."""
    import re
    buttons = []
    seen_ids = set()
    # Match: "- label: "3"; type: floor_button; position: center"
    pattern = re.compile(
        r'-\s*label:\s*"([^"]+)"\s*;\s*type:\s*(\S+)')
    for match in pattern.finditer(free_text):
        label = match.group(1).strip()
        btype = match.group(2).strip()
        btype = btype.rstrip(";,")
        if btype not in SUPPORTED_BUTTON_TYPES:
            continue
        if label in seen_ids:
            continue
        seen_ids.add(label)
        buttons.append({
            "id": f"button_{len(buttons)+1}",
            "button_type": btype,
            "label": label,
            "pressable": True,
        })
    if not buttons:
        return None
    return {
        "interface_type": "elevator_panel",
        "image_width": image_width,
        "image_height": image_height,
        "buttons": buttons,
    }


def draw_free_text(img, text, elapsed, error=None):
    overlay = img.copy()
    h, w = img.shape[:2]
    box_h = min(max(150, h // 4), h - 20)
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.68, img, 0.32, 0)

    if error and not text:
        lines = [f"ERROR: {error}"]
    else:
        lines = wrap_overlay_lines(text or "(empty VLM response)")

    y = h - box_h + 24
    for line in lines:
        cv2.putText(
            img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18

    timing = f"Inference: {elapsed:.1f}s"
    if error:
        timing += f" | {error}"
    cv2.putText(
        img,
        timing[:120],
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return img


def scaled_bbox(button, data, original_shape):
    original_h, original_w = original_shape[:2]
    data_w = data.get("image_width") or original_w
    data_h = data.get("image_height") or original_h
    sx = float(original_w) / float(data_w)
    sy = float(original_h) / float(data_h)

    x1 = int(round(float(button.get("x_min", 0)) * sx))
    y1 = int(round(float(button.get("y_min", 0)) * sy))
    x2 = int(round(float(button.get("x_max", 0)) * sx))
    y2 = int(round(float(button.get("y_max", 0)) * sy))

    x1 = max(0, min(original_w - 1, x1))
    x2 = max(0, min(original_w - 1, x2))
    y1 = max(0, min(original_h - 1, y1))
    y2 = max(0, min(original_h - 1, y2))
    return x1, y1, x2, y2


def draw_json(img, data, is_valid, errors):
    color = (0, 220, 0) if is_valid else (0, 0, 255)
    has_bbox = any("x_min" in b for b in data.get("buttons", []) if isinstance(b, dict))

    if isinstance(data, dict) and has_bbox:
        for button in data.get("buttons", []):
            if not isinstance(button, dict):
                continue
            try:
                x1, y1, x2, y2 = scaled_bbox(button, data, img.shape)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(button.get("label", "?")) + " " + str(button.get("button_type", "?"))[:36],
                        (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    elif isinstance(data, dict):
        # Labels-only: show text list on the image
        labels = [f'{b.get("label","?")} ({b.get("button_type","?")})'
                  for b in data.get("buttons", []) if isinstance(b, dict)]
        text = "Buttons: " + ", ".join(labels)
        img = draw_free_text(img, text, 0)

    status = "VALID" if is_valid else "INVALID: " + "; ".join(errors[:2])
    return draw_status(img, status, color)


def save_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_result_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_image(image_path, mode, output_dir, building=None, target_floor=None,
                  panel_type="auto",
                  max_dim=640, timeout=120,
                  image_min_tokens=DEFAULT_IMAGE_MIN_TOKENS, verbose=False):
    image_path = Path(image_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    raw_dir = output_dir / "raw_responses"
    basename = image_path.stem

    try:
        original_img, prepared_path, meta = prepare_inference_image(image_path, max_dim)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return False

    inferred_panel_type = infer_panel_type(image_path, panel_type)
    prompt = build_prompt(
        mode,
        meta["model_width"],
        meta["model_height"],
        panel_type=inferred_panel_type,
        building=building,
        target_floor=target_floor,
    )

    try:
        print(
            f"Running {mode} on {image_path.name} "
            f"({meta['original_width']}x{meta['original_height']} -> "
            f"{meta['model_width']}x{meta['model_height']})..."
        )
        inference = run_llama(
            prepared_path,
            prompt,
            timeout=timeout,
            image_min_tokens=image_min_tokens,
            verbose=verbose,
        )
    finally:
        prepared_path.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    save_text(raw_dir / f"{basename}_response.txt", inference["response"])
    save_text(raw_dir / f"{basename}_stdout.txt", inference["stdout"])
    save_text(raw_dir / f"{basename}_stderr.txt", inference["stderr"])

    common_payload = {
        "image": str(image_path),
        "mode": mode,
        "panel_type": inferred_panel_type,
        "model_image": {
            "width": meta["model_width"],
            "height": meta["model_height"],
        },
        "original_image": {
            "width": meta["original_width"],
            "height": meta["original_height"],
        },
        "elapsed_seconds": inference["elapsed"],
        "returncode": inference["returncode"],
        "error": inference["error"],
        "image_min_tokens": image_min_tokens,
        "raw_model_response": inference["response"],
    }

    if mode == "elevator-json":
        parsed, parse_error = extract_json(inference["response"])
        if parsed is None:
            is_valid = False
            errors = [f"JSON parse failed: {parse_error}"]
        else:
            is_valid, errors = validate_buttons(parsed)

        result_payload = {
            **common_payload,
            "response": inference["response"],
            "parsed": parsed,
            "validation": {
                "valid": is_valid,
                "errors": errors,
            },
        }
        save_result_json(output_dir / f"{basename}_result.json", result_payload)
        annotated = draw_json(original_img.copy(), parsed or {}, is_valid, errors)
        status = "VALID" if is_valid else "INVALID"
        print(f"  {basename}: {status} ({inference['elapsed']:.1f}s)")
    else:
        filtered_response, removed_lines = filter_free_text_response(
            inference["response"],
            inferred_panel_type,
        )
        save_text(raw_dir / f"{basename}_filtered_response.txt", filtered_response)

        # Parse free-text into labels-only JSON
        parsed = parse_free_text_to_json(
            filtered_response,
            meta["model_width"],
            meta["model_height"],
        )
        if parsed is not None:
            is_valid, val_errors = validate_buttons(parsed)
            validation = {"valid": is_valid, "errors": val_errors}
        else:
            is_valid, val_errors = False, ["parser returned no buttons"]
            validation = {"valid": False, "errors": val_errors}

        save_result_json(output_dir / f"{basename}_result.json", {
            **common_payload,
            "response": filtered_response,
            "postprocess": {
                "removed_lines": removed_lines,
            },
            "parsed": parsed,
            "validation": validation,
        })
        annotated = draw_free_text(
            original_img.copy(),
            filtered_response,
            inference["elapsed"],
            error=inference["error"],
        )
        preview = filtered_response[:140].replace("\n", " | ")
        if inference["error"]:
            preview = f"ERROR: {inference['error']}"
        print(f"  {basename}: {preview} ({inference['elapsed']:.1f}s)")

    out_path = output_dir / f"{basename}_annotated.png"
    cv2.imwrite(str(out_path), annotated)
    return inference["returncode"] == 0 and not inference["error"]


def image_files_in_folder(folder):
    """Return image paths for batch inference.

    RGB-D dataset folders use *_rgb.jpg pairs; skip *_depth.png automatically.
    Falls back to all images (excluding *_depth*) for demo/reference folders.
    """
    folder = Path(folder).expanduser()
    candidates = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and EXCLUDED_IMAGE_DIR_NAMES.isdisjoint(path.relative_to(folder).parts[:-1])
    )
    rgb_captures = [p for p in candidates if p.name.endswith("_rgb.jpg")]
    if rgb_captures:
        return rgb_captures
    return [p for p in candidates if "_depth" not in p.stem]


def default_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    if args.image_folder:
        return Path(args.image_folder).expanduser() / "output"
    if args.image:
        return Path(args.image).expanduser().parent / "output"
    return Path("~/demo_images/output").expanduser()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Sprint 2 Elevator Panel Demo")
    parser.add_argument("--mode", choices=["elevator", "elevator-json"], default="elevator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Single image path")
    group.add_argument("--image-folder", type=str, help="Process all images in folder")
    parser.add_argument("--building", type=str, help="Building ID for floor mapping")
    parser.add_argument("--target-floor", type=str, help="Target floor number")
    parser.add_argument("--panel-type", choices=["auto", "inside", "hall", "accessible"],
                        default="auto", help="Panel type hint for prompt; auto uses filename")
    parser.add_argument("--output-dir", type=str, help="Output directory; default is <image-folder>/output")
    parser.add_argument("--max-dim", type=int, default=640, help="Max image side sent to VLM")
    parser.add_argument("--timeout", type=int, default=120, help="Inference timeout in seconds")
    parser.add_argument("--image-min-tokens", type=int, default=DEFAULT_IMAGE_MIN_TOKENS,
                        help="llama.cpp --image-min-tokens value; use 0 to omit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print llama.cpp command")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    output_dir = default_output_dir(args)

    if args.image:
        ok = process_image(
            args.image,
            args.mode,
            output_dir,
            building=args.building,
            target_floor=args.target_floor,
            panel_type=args.panel_type,
            max_dim=args.max_dim,
            timeout=args.timeout,
            image_min_tokens=args.image_min_tokens,
            verbose=args.verbose,
        )
        return 0 if ok else 1

    images = image_files_in_folder(args.image_folder)
    if not images:
        print(f"No images found in {args.image_folder}")
        return 1

    print(f"Processing {len(images)} images from {Path(args.image_folder).expanduser()}...")
    failures = 0
    for image_path in images:
        ok = process_image(
            image_path,
            args.mode,
            output_dir,
            building=args.building,
            target_floor=args.target_floor,
            panel_type=args.panel_type,
            max_dim=args.max_dim,
            timeout=args.timeout,
            image_min_tokens=args.image_min_tokens,
            verbose=args.verbose,
        )
        failures += 0 if ok else 1

    print(f"\nDone. Outputs in {output_dir}/")
    if failures:
        print(f"{failures} image(s) had inference errors.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
