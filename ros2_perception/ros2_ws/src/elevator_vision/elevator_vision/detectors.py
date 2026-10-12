"""Detector backend contracts for mock and CUDA-only YOLO inference."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .models import Detection


class DetectorConfigurationError(ValueError):
    """Raised when a detector backend is configured incorrectly."""


class DetectorUnavailableError(RuntimeError):
    """Raised when an optional detector backend cannot be used."""


class BaseDetector(ABC):
    """Common interface for button detector backends."""

    @abstractmethod
    def detect(self, image):
        """Return detections for an RGB image."""


class MockDetector(BaseDetector):
    """Return four stable, deterministic, geometry-only detections."""

    _SPECS = (
        ('button_01', 0.20, 0.18, 0.36, 0.36, 0.96),
        ('button_02', 0.64, 0.18, 0.80, 0.36, 0.95),
        ('button_03', 0.20, 0.58, 0.36, 0.76, 0.94),
        ('button_04', 0.64, 0.58, 0.80, 0.76, 0.93),
    )

    def detect(self, image):
        if image is None or not hasattr(image, 'shape') or len(image.shape) < 2:
            raise ValueError('MockDetector requires an image with height and width')
        image_height = int(image.shape[0])
        image_width = int(image.shape[1])
        if image_width <= 0 or image_height <= 0:
            raise ValueError('image dimensions must be positive')

        detections = []
        for detection_id, left, top, right, bottom, confidence in self._SPECS:
            x_min = float(round(left * image_width, 3))
            y_min = float(round(top * image_height, 3))
            x_max = float(round(right * image_width, 3))
            y_max = float(round(bottom * image_height, 3))
            detections.append(Detection(
                detection_id=detection_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                center_x=(x_min + x_max) / 2.0,
                center_y=(y_min + y_max) / 2.0,
                detector_confidence=confidence,
                coarse_type='button_candidate',
            ))
        return detections


class YoloDetector(BaseDetector):
    """Ultralytics detector that is loaded and warmed once on CUDA."""

    def __init__(
        self,
        model_path,
        confidence_threshold=0.30,
        device=0,
        imgsz=640,
        iou_threshold=0.70,
        max_det=300,
    ):
        self.model_path = str(model_path or '')
        self.confidence_threshold = float(confidence_threshold)
        self.device = self._normalize_device(device)
        self.imgsz = int(imgsz)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.validate_configuration()
        self._torch, yolo_class = self._load_runtime()
        if not self._torch.cuda.is_available():
            raise DetectorUnavailableError(
                'CUDA is required for the YOLO backend; CPU fallback is disabled'
            )
        try:
            self._model = yolo_class(self.model_path, task='detect')
            self._warm_up()
        except DetectorUnavailableError:
            raise
        except Exception as exc:
            raise DetectorUnavailableError(
                'failed to initialize YOLO on CUDA:{}: {}'.format(
                    self.device,
                    exc,
                )
            ) from exc

    @staticmethod
    def _normalize_device(device):
        if isinstance(device, bool):
            raise DetectorConfigurationError('YOLO device must be CUDA device 0')
        if isinstance(device, int):
            normalized = device
        else:
            value = str(device).strip().lower()
            if value.startswith('cuda:'):
                value = value.split(':', 1)[1]
            if not value.isdigit():
                raise DetectorConfigurationError(
                    'YOLO device must be a CUDA device index; CPU fallback is disabled'
                )
            normalized = int(value)
        if normalized != 0:
            raise DetectorConfigurationError(
                'Phase AGX-V6-LIVE-1 requires CUDA device 0'
            )
        return normalized

    @staticmethod
    def _load_runtime():
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError(
                'YOLO backend requires torch and ultralytics'
            ) from exc
        return torch, YOLO

    def validate_configuration(self):
        if not self.model_path:
            raise DetectorConfigurationError('YOLO model_path must not be empty')
        if not Path(self.model_path).is_file():
            raise DetectorConfigurationError(
                'YOLO model file does not exist: {}'.format(self.model_path)
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise DetectorConfigurationError(
                'YOLO confidence_threshold must be in [0, 1]'
            )
        if self.imgsz <= 0:
            raise DetectorConfigurationError('YOLO imgsz must be positive')
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise DetectorConfigurationError(
                'YOLO iou_threshold must be in [0, 1]'
            )
        if self.max_det <= 0:
            raise DetectorConfigurationError('YOLO max_det must be positive')

    def _predict(self, image):
        return self._model.predict(
            source=image,
            task='detect',
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            classes=[0],
            verbose=False,
        )

    def _assert_cuda_model(self):
        try:
            model_device = next(self._model.model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise DetectorUnavailableError(
                'unable to verify the YOLO inference device'
            ) from exc
        if getattr(model_device, 'type', None) != 'cuda':
            raise DetectorUnavailableError(
                'YOLO initialized on {}; CPU fallback is disabled'.format(
                    model_device
                )
            )
        if getattr(model_device, 'index', self.device) not in (None, self.device):
            raise DetectorUnavailableError(
                'YOLO initialized on unexpected CUDA device {}'.format(model_device)
            )

    def _warm_up(self):
        warmup_image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._predict(warmup_image)
        self._torch.cuda.synchronize(self.device)
        self._assert_cuda_model()

    def detect(self, image):
        if image is None or not hasattr(image, 'shape') or len(image.shape) != 3:
            raise ValueError('YoloDetector requires an HxWxC RGB numpy image')
        image_height, image_width = image.shape[:2]
        if image_height <= 0 or image_width <= 0 or image.shape[2] < 3:
            raise ValueError('YoloDetector requires positive RGB image dimensions')

        # Ultralytics treats numpy input as OpenCV BGR. The public detector
        # contract is RGB, so convert without changing the caller's image.
        image_bgr = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
        results = self._predict(image_bgr)
        if not results:
            return []
        boxes = getattr(results[0], 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        max_x = float(image_width - 1)
        max_y = float(image_height - 1)
        candidates = []
        for coordinates, confidence, class_id in zip(xyxy, confidences, classes):
            if int(class_id) != 0:
                continue
            x_min = float(np.clip(coordinates[0], 0.0, max_x))
            y_min = float(np.clip(coordinates[1], 0.0, max_y))
            x_max = float(np.clip(coordinates[2], 0.0, max_x))
            y_max = float(np.clip(coordinates[3], 0.0, max_y))
            if x_max <= x_min or y_max <= y_min:
                continue
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            candidates.append((
                center_y,
                center_x,
                x_min,
                y_min,
                x_max,
                y_max,
                float(confidence),
            ))

        candidates.sort(key=lambda item: item[:-1] + (-item[-1],))
        return [
            Detection(
                detection_id='button_{:03d}'.format(index),
                x_min=item[2],
                y_min=item[3],
                x_max=item[4],
                y_max=item[5],
                center_x=item[1],
                center_y=item[0],
                detector_confidence=item[6],
                coarse_type='button_candidate',
            )
            for index, item in enumerate(candidates, start=1)
        ]
