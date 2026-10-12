"""GPU-free frame metrics and gating logic."""

from dataclasses import dataclass

import numpy as np


def resize_nearest(image, output_width, output_height):
    if output_width <= 0 or output_height <= 0:
        raise ValueError('resize dimensions must be positive')
    input_height, input_width = image.shape[:2]
    if input_width == output_width and input_height == output_height:
        return np.array(image, copy=True)
    y_indices = np.linspace(0, input_height - 1, output_height).astype(np.int64)
    x_indices = np.linspace(0, input_width - 1, output_width).astype(np.int64)
    return np.array(image[y_indices][:, x_indices], copy=True)


def grayscale(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('frame must be grayscale or RGB')
    rgb = image[..., :3].astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def laplacian_sharpness(image):
    gray = grayscale(image)
    padded = np.pad(gray, 1, mode='edge')
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(laplacian))


def mean_brightness(image):
    return float(np.mean(grayscale(image)))


def frame_motion(image, previous_image):
    if previous_image is None:
        return 0.0
    current = grayscale(image)
    previous = grayscale(previous_image)
    if previous.shape != current.shape:
        previous = resize_nearest(previous, current.shape[1], current.shape[0])
    return float(np.mean(np.abs(current - previous)))


@dataclass(frozen=True)
class FrameGateResult:
    accepted: bool
    sharpness: float
    brightness: float
    motion_score: float
    reason: str
    resized_image: object


class FrameGate:
    def __init__(
            self,
            output_width=640,
            output_height=480,
            metric_width=None,
            metric_height=None,
            min_sharpness=10.0,
            min_brightness=20.0,
            max_brightness=235.0,
            max_motion=50.0,
            min_interval_sec=0.1):
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.metric_width = int(
            output_width if metric_width is None else metric_width
        )
        self.metric_height = int(
            output_height if metric_height is None else metric_height
        )
        if self.metric_width <= 0 or self.metric_height <= 0:
            raise ValueError('metric dimensions must be positive')
        self.min_sharpness = float(min_sharpness)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)
        self.max_motion = float(max_motion)
        self.min_interval_sec = float(min_interval_sec)
        self._previous_frame = None
        self._last_accepted_time = None

    def evaluate(self, image, timestamp_sec):
        resized = resize_nearest(image, self.output_width, self.output_height)
        metric_image = resize_nearest(
            image,
            self.metric_width,
            self.metric_height,
        )
        sharpness = laplacian_sharpness(metric_image)
        brightness = mean_brightness(metric_image)
        motion_score = frame_motion(metric_image, self._previous_frame)
        self._previous_frame = np.array(metric_image, copy=True)

        accepted = True
        reason = 'accepted'
        if (
                self._last_accepted_time is not None
                and timestamp_sec - self._last_accepted_time < self.min_interval_sec):
            accepted = False
            reason = 'minimum_interval'
        elif sharpness < self.min_sharpness:
            accepted = False
            reason = 'low_sharpness'
        elif brightness < self.min_brightness:
            accepted = False
            reason = 'low_brightness'
        elif brightness > self.max_brightness:
            accepted = False
            reason = 'high_brightness'
        elif motion_score > self.max_motion:
            accepted = False
            reason = 'high_motion'

        if accepted:
            self._last_accepted_time = float(timestamp_sec)
        return FrameGateResult(
            accepted=accepted,
            sharpness=sharpness,
            brightness=brightness,
            motion_score=motion_score,
            reason=reason,
            resized_image=resized,
        )
