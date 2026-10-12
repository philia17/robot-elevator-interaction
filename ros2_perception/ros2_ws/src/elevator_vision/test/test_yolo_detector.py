"""CPU-only contract tests for the CUDA-only Ultralytics wrapper."""

from types import SimpleNamespace

import numpy as np
import pytest

from elevator_vision.detectors import (
    DetectorConfigurationError,
    DetectorUnavailableError,
    YoloDetector,
)


class _FakeTensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBoxes:
    def __init__(self, xyxy=(), confidence=(), classes=()):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(confidence)
        self.cls = _FakeTensor(classes)

    def __len__(self):
        return len(self.conf._values)


class _FakeYolo:
    instances = []

    def __init__(self, model_path, task):
        self.model_path = model_path
        self.task = task
        self.calls = []
        parameter = SimpleNamespace(
            device=SimpleNamespace(type='cuda', index=0),
        )
        self.model = SimpleNamespace(parameters=lambda: iter([parameter]))
        self.instances.append(self)

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return [SimpleNamespace(boxes=_FakeBoxes())]
        return [SimpleNamespace(boxes=_FakeBoxes(
            xyxy=[
                [80.0, 50.0, 120.0, 90.0],
                [-5.0, -10.0, 25.0, 30.0],
                [10.0, 10.0, 20.0, 20.0],
                [90.0, 70.0, 140.0, 95.0],
            ],
            confidence=[0.80, 0.95, 0.99, 0.70],
            classes=[0.0, 0.0, 1.0, 0.0],
        ))]


class _FakeCuda:
    def __init__(self, available=True):
        self._available = available
        self.synchronize_calls = []

    def is_available(self):
        return self._available

    def synchronize(self, device):
        self.synchronize_calls.append(device)


def _make_detector(monkeypatch, tmp_path, cuda_available=True):
    _FakeYolo.instances = []
    fake_cuda = _FakeCuda(cuda_available)
    fake_torch = SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(
        YoloDetector,
        '_load_runtime',
        staticmethod(lambda: (fake_torch, _FakeYolo)),
    )
    model_path = tmp_path / 'best.pt'
    model_path.write_bytes(b'model-placeholder')
    detector = YoloDetector(
        model_path,
        confidence_threshold=0.30,
        device=0,
        imgsz=640,
        iou_threshold=0.70,
        max_det=300,
    )
    return detector, fake_cuda, _FakeYolo.instances[0]


def test_yolo_loads_once_warms_on_cuda_and_uses_fixed_contract(
    monkeypatch,
    tmp_path,
):
    detector, fake_cuda, model = _make_detector(monkeypatch, tmp_path)

    assert len(_FakeYolo.instances) == 1
    assert model.task == 'detect'
    assert len(model.calls) == 1
    assert model.calls[0]['source'].shape == (640, 640, 3)
    assert model.calls[0]['device'] == 0
    assert model.calls[0]['classes'] == [0]
    assert model.calls[0]['conf'] == 0.30
    assert model.calls[0]['imgsz'] == 640
    assert model.calls[0]['iou'] == 0.70
    assert model.calls[0]['max_det'] == 300
    assert fake_cuda.synchronize_calls == [0]

    rgb_image = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb_image[0, 0] = [10, 20, 30]
    detections = detector.detect(rgb_image)

    assert model.calls[1]['source'][0, 0].tolist() == [30, 20, 10]
    assert [item.detection_id for item in detections] == [
        'button_001', 'button_002', 'button_003'
    ]
    assert [item.bbox for item in detections] == [
        (0.0, 0.0, 25.0, 30.0),
        (80.0, 50.0, 99.0, 79.0),
        (90.0, 70.0, 99.0, 79.0),
    ]
    assert all(item.coarse_type == 'button_candidate' for item in detections)


def test_yolo_returns_empty_list_for_empty_result(monkeypatch, tmp_path):
    detector, _, model = _make_detector(monkeypatch, tmp_path)
    model.predict = lambda **kwargs: [SimpleNamespace(boxes=_FakeBoxes())]
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_yolo_rejects_cpu_device_before_loading_model(tmp_path):
    model_path = tmp_path / 'best.pt'
    model_path.write_bytes(b'model-placeholder')
    with pytest.raises(DetectorConfigurationError, match='CPU fallback'):
        YoloDetector(model_path, device='cpu')


def test_yolo_fails_when_cuda_is_unavailable(monkeypatch, tmp_path):
    with pytest.raises(DetectorUnavailableError, match='CPU fallback is disabled'):
        _make_detector(monkeypatch, tmp_path, cuda_available=False)
