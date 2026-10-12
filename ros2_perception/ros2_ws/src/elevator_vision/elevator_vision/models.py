"""Data models shared by deterministic elevator vision backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Detector-owned geometry and confidence for one button candidate."""

    detection_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center_x: float
    center_y: float
    detector_confidence: float
    coarse_type: str

    @property
    def bbox(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)
