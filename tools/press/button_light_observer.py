"""Pure fixed-camera light observer for one PRESS/OUT/BACK attempt."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


LIGHT_ON = "LIGHT_ON"
CLEAR_NO_LIGHT = "CLEAR_NO_LIGHT"
UNOBSERVABLE = "UNOBSERVABLE"


@dataclass(frozen=True)
class LightObserverConfig:
    baseline_frames: int = 5
    latch_window: int = 5
    latch_required: int = 3
    post_back_frames: int = 3
    minimum_relative_luma_rise: float = 0.12
    minimum_changed_fraction: float = 0.15
    changed_pixel_relative_luma: float = 0.10
    maximum_global_luma_shift: float = 0.10


class ButtonLightObserver:
    """Observe one frozen Gemini ROI; classification is locked behind BACK."""

    def __init__(self, roi_xyxy: Sequence[int], target_identity: str, config: LightObserverConfig | None = None) -> None:
        if len(roi_xyxy) != 4 or not str(target_identity):
            raise ValueError("LIGHT_OBSERVER_INVALID")
        self.roi = tuple(int(value) for value in roi_xyxy)
        self.target_identity = str(target_identity)
        self.config = config or LightObserverConfig()
        if (
            self.config.baseline_frames < 5
            or self.config.latch_window < self.config.latch_required
            or self.config.latch_required < 1
            or self.config.post_back_frames < self.config.latch_required
        ):
            raise ValueError("LIGHT_OBSERVER_CONFIG_INVALID")
        self._baseline_rois: list[np.ndarray] = []
        self._baseline_global: list[float] = []
        self._decisions: deque[bool] = deque(maxlen=self.config.latch_window)
        self._latched = False
        self._back_proven = False
        self._post_back_count = 0
        self._failure = ""
        self._last_stamp_ns = 0
        self.metrics: list[dict[str, float | bool | int | str]] = []

    @staticmethod
    def _luma(frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("FRAME_INVALID")
        image = frame[..., :3].astype(np.float32)
        # Input is BGR in the OpenCV/ROS Live pipeline.
        return (0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]) / 255.0

    def _extract(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        luma = self._luma(frame)
        x0, y0, x1, y1 = self.roi
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > luma.shape[1] or y1 > luma.shape[0]:
            raise ValueError("ROI_OUT_OF_FRAME")
        roi = luma[y0:y1, x0:x1]
        if roi.size < 16:
            raise ValueError("ROI_TOO_SMALL")
        return roi, float(np.mean(luma))

    def observe(
        self,
        frame: np.ndarray,
        *,
        stamp_ns: int,
        target_identity: str,
        phase: str,
        roi_visible: bool = True,
        occluded: bool = False,
    ) -> None:
        if self._failure:
            return
        if str(target_identity) != self.target_identity:
            self._failure = "TARGET_IDENTITY_MISMATCH"
            return
        if stamp_ns <= self._last_stamp_ns:
            self._failure = "FRAME_NOT_FRESH"
            return
        self._last_stamp_ns = int(stamp_ns)
        if not roi_visible or occluded:
            self._failure = "ROI_UNOBSERVABLE"
            return
        try:
            roi, global_luma = self._extract(frame)
        except ValueError as exc:
            self._failure = str(exc)
            return
        mode = str(phase).upper()
        if mode == "BASELINE":
            if self._back_proven:
                self._failure = "BASELINE_AFTER_BACK"
                return
            self._baseline_rois.append(roi.copy())
            self._baseline_global.append(global_luma)
            self._baseline_rois = self._baseline_rois[-self.config.baseline_frames:]
            self._baseline_global = self._baseline_global[-self.config.baseline_frames:]
            return
        if len(self._baseline_rois) < self.config.baseline_frames:
            self._failure = "BASELINE_INCOMPLETE"
            return
        if mode not in {"ACTIVE", "POST_BACK"}:
            self._failure = "LIGHT_PHASE_INVALID"
            return
        if mode == "POST_BACK" and not self._back_proven:
            self._failure = "POST_BACK_WITHOUT_PROOF"
            return
        baseline_roi = np.median(np.stack(self._baseline_rois, axis=0), axis=0)
        baseline_global = float(np.median(self._baseline_global))
        global_shift = abs(global_luma - baseline_global)
        if global_shift > self.config.maximum_global_luma_shift:
            self._failure = "EXPOSURE_SHIFT"
            return
        delta = roi - baseline_roi
        relative_rise = float(np.mean(delta))
        changed_fraction = float(np.mean(delta >= self.config.changed_pixel_relative_luma))
        on = bool(
            relative_rise >= self.config.minimum_relative_luma_rise
            and changed_fraction >= self.config.minimum_changed_fraction
        )
        if mode == "POST_BACK":
            self._decisions.append(on)
            if (
                len(self._decisions) >= self.config.latch_required
                and sum(self._decisions) >= self.config.latch_required
            ):
                self._latched = True
            self._post_back_count += 1
        self.metrics.append({
            "stamp_ns": int(stamp_ns),
            "phase": mode,
            "latch_eligible": mode == "POST_BACK",
            "relative_luma_rise": relative_rise,
            "changed_fraction": changed_fraction,
            "global_luma_shift": global_shift,
            "on": on,
        })

    def mark_back_proven(self, proven: bool) -> None:
        if proven is not True:
            self._failure = "BACK_UNPROVEN"
            return
        self._back_proven = True

    @property
    def baseline_ready(self) -> bool:
        return len(self._baseline_rois) >= self.config.baseline_frames

    @property
    def light_latched(self) -> bool:
        return self._latched

    def classify(self) -> str:
        if self._failure or not self._back_proven or self._post_back_count < self.config.post_back_frames:
            return UNOBSERVABLE
        return LIGHT_ON if self._latched else CLEAR_NO_LIGHT

    @property
    def post_back_ready(self) -> bool:
        return self._post_back_count >= self.config.post_back_frames

    @property
    def failure_reason(self) -> str:
        return self._failure
