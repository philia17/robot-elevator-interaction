"""Small, ROS-free helpers for an operator-selected button identity."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


def normalize_button_label(value: Any) -> str:
    """Canonical floor/control content; unknown text is never a target."""
    label = str(value or "").strip().lower()
    label = re.sub(r"^(?:button|floor)[ _]+", "", label).strip()
    aliases = {
        "one": "1", "door_open": "open", "open door": "open",
        "door_close": "close", "close door": "close", "开门": "open",
        "关门": "close", "上": "up", "上行": "up", "下": "down", "下行": "down",
    }
    label = aliases.get(label, label)
    if label in {"", "?", "unknown", "none", "null", "n/a", "nan", "unrecognized"}:
        return ""
    if label in {"open", "close", "up", "down", "alarm", "help", "stop"}:
        return label
    return label if re.fullmatch(r"(?:-?\d+[a-z]?|[a-z]{1,3}\d*)", label) else ""


def get_semantic_track_id(item: Mapping[str, Any]) -> str:
    direct = str(item.get("semantic_track_id") or item.get("track_id") or "").strip()
    if direct:
        return direct
    try:
        metadata = json.loads(item.get("raw_response") or "{}")
    except (TypeError, ValueError):
        return ""
    return str(metadata.get("track_id") or "").strip() if isinstance(metadata, Mapping) else ""


semantic_track_id = get_semantic_track_id


def target_identity_matches(
    item: Mapping[str, Any], *, target_label: str = "1",
    detection_id: str | None = None, semantic_track_id: str | None = None,
) -> bool:
    """Track IDs span frames; detection IDs only join a single source frame."""
    expected = normalize_button_label(target_label)
    if not expected or normalize_button_label(item.get("label") or item.get("target_label")) != expected:
        return False
    if semantic_track_id:
        return get_semantic_track_id(item) == str(semantic_track_id)
    return not detection_id or str(item.get("detection_id") or "") == str(detection_id)


def validate_target_selection(
    selection: Mapping[str, Any], *, target_label: str, source_stamp_ns: int,
) -> None:
    """Validate the operator binding carried by a selected dynamic target."""
    label = normalize_button_label(target_label)
    if not label or normalize_button_label(selection.get("label")) != label:
        raise ValueError("selected target label mismatch")
    for key in ("selection_id", "detection_id", "semantic_track_id"):
        if not str(selection.get(key) or "").strip():
            raise ValueError(f"selected target missing {key}")
    after = int(selection.get("after_stamp_ns") or 0)
    if after <= 0 or int(source_stamp_ns) <= after:
        raise ValueError("selected target requires a new source frame")
    if selection.get("pressable") is not True:
        raise ValueError("selected target is not confirmed pressable")
