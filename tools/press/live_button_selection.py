"""Pure, bounded Gemini recognition/selection state.  Never issues robot intents."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

from press.button_target import normalize_button_label, semantic_track_id


@dataclass(frozen=True)
class RecognizedButton:
    label: str
    detection_id: str
    semantic_track_id: str
    bbox: tuple[float, float, float, float]
    pressable: bool
    generation: int = 0

    @property
    def selectable(self) -> bool:
        # Detection IDs are frame-local.  A label alone is not an entity ID.
        return bool(self.pressable and self.semantic_track_id and normalize_button_label(self.label))


def recognized_buttons(observations: list[dict], semantics: list[dict], *, generation: int = 0) -> tuple[RecognizedButton, ...]:
    """Join only an already same-header pair by its frame-local detection ID."""
    semantic_rows: dict[str, list[dict]] = {}
    for item in semantics:
        semantic_rows.setdefault(str(item.get("detection_id") or ""), []).append(item)
    result = []
    for observation in observations:
        detection_id = str(observation.get("detection_id") or "")
        try:
            bbox = tuple(float(observation[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
        except (KeyError, TypeError, ValueError):
            continue
        if not detection_id or not all(math.isfinite(value) for value in bbox):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        matches = semantic_rows.get(detection_id, [])
        semantic = matches[0] if len(matches) == 1 else {}
        label = normalize_button_label(semantic.get("label"))
        result.append(RecognizedButton(
            label=label or str(semantic.get("label") or "?"),
            detection_id=detection_id,
            semantic_track_id=semantic_track_id(semantic),
            bbox=bbox,
            pressable=bool(semantic.get("pressable") is True and label),
            generation=generation,
        ))
    return tuple(result)


def button_at_pixel(buttons: tuple[RecognizedButton, ...], point: tuple[float, float]) -> RecognizedButton | None:
    """An overlapping/ambiguous click never silently selects a neighbor."""
    x, y = point
    matches = [item for item in buttons if item.bbox[0] <= x <= item.bbox[2] and item.bbox[1] <= y <= item.bbox[3]]
    return matches[0] if len(matches) == 1 else None


def fitted_image_rect(panel_rect: tuple[int, int, int, int], image_shape: tuple[int, ...], *, header_h: int = 42, info_h: int = 48) -> tuple[int, int, int, int]:
    """Use the same integer resize and letterbox placement as panel_image."""
    x0, y0, x1, y1 = panel_rect
    panel_w, panel_h = x1 - x0, y1 - y0
    height, width = image_shape[:2]
    image_h, image_w = max(1, panel_h - header_h - info_h), max(1, panel_w - 8)
    scale = min(image_w / float(width), image_h / float(height))
    fitted_w, fitted_h = max(1, round(width * scale)), max(1, round(height * scale))
    left = x0 + (panel_w - fitted_w) // 2
    top = y0 + header_h + (image_h - fitted_h) // 2
    return left, top, left + fitted_w, top + fitted_h


def panel_to_image(point: tuple[float, float], image_rect: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[float, float] | None:
    x, y = point
    left, top, right, bottom = image_rect
    if not (left <= x < right and top <= y < bottom):
        return None
    height, width = image_shape[:2]
    return (x - left) * width / (right - left), (y - top) * height / (bottom - top)


class LiveButtonSelection:
    """Recognize waits for a post-click aligned pair; selection uses stable track."""

    def __init__(self) -> None:
        self.enabled = False
        self.after_stamp_ns = 0
        self.source_stamp_ns = 0
        self.received_at = 0.0
        self.buttons: tuple[RecognizedButton, ...] = ()
        self.selected: dict[str, object] | None = None
        self.status = "Click RECOGNIZE"
        self.generation = 0
        self._pairs: OrderedDict[tuple[str, int], dict] = OrderedDict()

    def recognize(self, *, after_stamp_ns: int) -> None:
        self.generation += 1
        self.enabled = True
        self.after_stamp_ns = int(after_stamp_ns)
        self.source_stamp_ns = 0
        self.received_at = 0.0
        self.buttons = ()
        self.selected = None
        self._pairs.clear()
        self.status = "RECOGNIZING - waiting fresh YOLO + VLM"

    def ingest(self, kind: str, rows: list[dict], *, frame_id: str, stamp_ns: int, received_at: float) -> bool:
        if not self.enabled or not frame_id or stamp_ns <= max(self.after_stamp_ns, self.source_stamp_ns):
            return False
        key = (str(frame_id), int(stamp_ns))
        pair = self._pairs.setdefault(key, {})
        pair[kind] = rows
        self._pairs.move_to_end(key)
        while len(self._pairs) > 8:
            self._pairs.popitem(last=False)
        if "observations" not in pair or "semantics" not in pair:
            return False
        self.buttons = recognized_buttons(pair["observations"], pair["semantics"], generation=self.generation)
        self.source_stamp_ns = int(stamp_ns)
        self.received_at = float(received_at)
        self.status = f"{len(self.buttons)} buttons - click a box" if self.buttons else "NO BUTTONS - keep panel visible"
        return True

    def select(self, clicked: RecognizedButton, *, selection_id: str, after_stamp_ns: int, now: float) -> dict[str, object]:
        if not self.enabled or now - self.received_at > 3.0:
            raise ValueError("RECOGNITION_STALE: click RECOGNIZE")
        if clicked.generation != self.generation:
            raise ValueError("RECOGNITION_CHANGED: click the current box")
        matches = [item for item in self.buttons if item.label == clicked.label and item.semantic_track_id == clicked.semantic_track_id]
        if not clicked.selectable or len(matches) != 1 or not matches[0].selectable:
            raise ValueError("BUTTON_IDENTITY_NOT_READY")
        current = matches[0]
        self.selected = {
            "selection_id": str(selection_id),
            "label": current.label,
            "detection_id": current.detection_id,
            "semantic_track_id": current.semantic_track_id,
            "after_stamp_ns": int(after_stamp_ns),
        }
        self.status = "SELECTED: " + current.label
        return dict(self.selected)

    def is_selected(self, button: RecognizedButton) -> bool:
        return bool(self.selected and button.label == self.selected["label"] and button.semantic_track_id == self.selected["semantic_track_id"])
