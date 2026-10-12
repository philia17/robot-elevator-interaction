"""Asynchronous, CUDA-only Qwen3-VL button labeling for RGB-D display.

The ROS image/depth callbacks never call model loading or ``generate``.  This
module owns one worker thread, a latest-only queue of length one, strict
closed-set parsing, crop preparation, and short-lived stable-button caching.
"""

from dataclasses import dataclass, field
import csv
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from time import monotonic, time

import numpy as np
from PIL import Image


CLOSED_LABELS = frozenset({
    'alarm', 'b1', '4', '3', '2', '1', 'open', 'close', 'none',
})
UNKNOWN = 'UNKNOWN'
PROCESSING = 'PROCESSING'

PROMPT = """You are classifying ONE cropped elevator-panel button candidate detected by YOLO.
Choose exactly one label from this closed list:
alarm, b1, 4, 3, 2, 1, open, close, none

Definitions:

- alarm: emergency, alarm, bell or emergency-call button.
- b1: basement-floor button printed B1.
- 4, 3, 2, 1: floor-number buttons showing that exact digit.
- open: door-open icon, usually two arrows pointing away from each other.
- close: door-close icon, usually two arrows pointing toward each other.
- none: not one of the allowed labels, unreadable, reflection, keyhole, background, unrelated button or false detection.

Return ONLY one lowercase label. No sentence. No punctuation."""

# ── hall mode constants ──────────────────────────────────────────────────

HALL_CLOSED_LABELS = frozenset({'up', 'down', 'none'})

HALL_PROMPT_FEWSHOT_4B = """\
Image 1 is a verified UP elevator-button example.
Image 2 is a verified DOWN elevator-button example.
Image 3 is the query.
Compare the query icon's pointed tip, base, and chevron orientation with both
verified examples. Ignore floor displays, brightness, and panel position.
Return only up, down, or none."""


def hall_strict_parse_label(raw_response):
    """Accept only up/down/none from the hall closed set."""
    value = str(raw_response or '').strip().lower()
    return value if value in HALL_CLOSED_LABELS else UNKNOWN


def _clip_box_hall(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(x1, min(float(width), x2))
    y2 = max(y1, min(float(height), y2))
    return x1, y1, x2, y2


def prepare_crop_hall(rgb_image, bbox, padding_ratio=0.40, upscale=5.0):
    """Hall-mode crop with large padding, 5x upscale, and local enhancement."""
    from PIL import ImageEnhance, ImageOps
    height, width = rgb_image.shape[:2]
    x1, y1, x2, y2 = _clip_box_hall(bbox, width, height)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    padding = max(box_width, box_height) * float(padding_ratio)
    x1, y1, x2, y2 = _clip_box_hall(
        (x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height)
    crop = np.ascontiguousarray(rgb_image[int(y1):int(max(y1 + 1, y2)),
                                          int(x1):int(max(x1 + 1, x2)), :3])
    pil_crop = Image.fromarray(crop, mode='RGB')
    if upscale > 1.0:
        pil_crop = pil_crop.resize(
            (max(1, int(round(pil_crop.width * upscale))),
             max(1, int(round(pil_crop.height * upscale)))),
            Image.Resampling.BICUBIC)
    pil_crop = ImageOps.autocontrast(pil_crop, cutoff=1)
    pil_crop = ImageEnhance.Contrast(pil_crop).enhance(1.6)
    pil_crop = ImageEnhance.Sharpness(pil_crop).enhance(2.0)
    return pil_crop


def load_hall_reference_crops(up_path, down_path):
    """Load up/down reference images for few-shot hall inference."""
    references = []
    for path in (up_path, down_path):
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(
                'hall visual reference missing: {}'.format(resolved))
        references.append(Image.open(str(resolved)).convert('RGB'))
    return tuple(references)


def _build_hall_4b_loader(model_path):
    """Return a model-loader callable for the hall 4B model."""
    def loader():
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        import torch
        import transformers
        from transformers import AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
        model_directory = Path(model_path)
        if not model_directory.is_dir():
            raise RuntimeError(
                'local HF model directory missing: {}'.format(model_directory))
        processor = AutoProcessor.from_pretrained(
            str(model_directory), trust_remote_code=True, local_files_only=True)
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.padding_side = 'left'
        model_class = getattr(transformers, 'Qwen3VLForConditionalGeneration', None)
        if model_class is None:
            raise RuntimeError('Qwen3VLForConditionalGeneration is unavailable')
        model = model_class.from_pretrained(
            str(model_directory), torch_dtype=torch.float16,
            trust_remote_code=True, local_files_only=True)
        model = model.to(torch.device('cuda:0'))
        model.eval()
        return model, processor
    return loader


def _build_hall_infer_function(reference_up, reference_down, max_new_tokens=8):
    """Return an inference callable for hall few-shot 4B."""
    def infer(model, processor, task):
        import torch
        refs = load_hall_reference_crops(reference_up, reference_down)
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image'}, {'type': 'image'}, {'type': 'image'},
                {'type': 'text', 'text': HALL_PROMPT_FEWSHOT_4B},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[[refs[0], refs[1], task.crop]],
            return_tensors='pt', padding=True)
        inputs = {
            key: value.to('cuda:0') if hasattr(value, 'to') else value
            for key, value in inputs.items()}
        input_length = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        output_ids = output_ids[:, input_length:]
        return processor.batch_decode(
            output_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
    return infer



def strict_parse_label(raw_response):
    """Accept only a complete closed-set label; never fuzzy-extract text."""

    value = str(raw_response or '').strip().lower()
    return value if value in CLOSED_LABELS else UNKNOWN


def bbox_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(first, second):
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    return math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )


def _association_radius(first, second):
    sizes = []
    for bbox in (first, second):
        sizes.append(math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return max(20.0, max(sizes) * 0.75)


def prepare_crop(rgb_image, bbox, padding_ratio=0.15, minimum_size=30):
    """Crop from a clean RGB ndarray, pad, and upscale small candidates."""

    if rgb_image is None or not hasattr(rgb_image, 'shape') or len(rgb_image.shape) != 3:
        raise ValueError('prepare_crop requires an HxWxC RGB image')
    height, width = rgb_image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * float(padding_ratio)
    pad_y = box_height * float(padding_ratio)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError('empty semantic crop')
    crop = np.ascontiguousarray(rgb_image[top:bottom, left:right, :3])
    pil_crop = Image.fromarray(crop, mode='RGB')
    if pil_crop.width < minimum_size or pil_crop.height < minimum_size:
        scale = max(
            float(minimum_size) / max(1, pil_crop.width),
            float(minimum_size) / max(1, pil_crop.height),
        )
        pil_crop = pil_crop.resize(
            (
                max(minimum_size, int(round(pil_crop.width * scale))),
                max(minimum_size, int(round(pil_crop.height * scale))),
            ),
            Image.Resampling.BICUBIC,
        )
    return pil_crop


@dataclass(frozen=True)
class SemanticTask:
    track_id: str
    detection_id: str
    bbox: tuple
    crop: Image.Image
    submitted_monotonic: float


@dataclass(frozen=True)
class SemanticBatchTask:
    """One latest-frame queue item containing all currently due buttons."""

    batch_id: str
    frame_stamp_ns: int
    visible_track_ids: tuple
    tasks: tuple
    submitted_monotonic: float


@dataclass(frozen=True)
class SemanticResult:
    track_id: str
    detection_id: str
    label: str
    raw_response: str
    latency_sec: float
    error: str = ''


@dataclass
class _Track:
    track_id: str
    detection_id: str
    bbox: tuple
    last_seen: float
    label: str = PROCESSING
    label_time: float = 0.0
    last_submit: float = 0.0
    in_flight: bool = False
    bbox_changed: bool = False
    consecutive_failures: int = 0
    last_error: str = ''
    soft_expiry_counted: bool = False


class SemanticTrackCache:
    """Associate frame-local IDs to short-lived tracks and cache labels."""

    def __init__(
        self,
        cache_ttl_sec=30.0,
        hard_cache_ttl_sec=120.0,
        retry_cooldown_sec=3.0,
        track_ttl_sec=3.0,
        association_iou=0.30,
        bbox_retry_iou=0.65,
        maximum_consecutive_failures=3,
    ):
        self.cache_ttl_sec = float(cache_ttl_sec)
        self.hard_cache_ttl_sec = max(
            self.cache_ttl_sec,
            float(hard_cache_ttl_sec),
        )
        self.retry_cooldown_sec = float(retry_cooldown_sec)
        self.track_ttl_sec = float(track_ttl_sec)
        self.association_iou = float(association_iou)
        self.bbox_retry_iou = float(bbox_retry_iou)
        self.maximum_consecutive_failures = max(
            1,
            int(maximum_consecutive_failures),
        )
        self._lock = threading.Lock()
        self._tracks = {}
        self._next_track = 1
        self._expired_track_ids = []
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.cache_expired_count = 0
        self.association_failure_count = 0
        self.track_expired_count = 0

    @staticmethod
    def _compatible(first, second, minimum_iou):
        return (
            bbox_iou(first, second) >= minimum_iou
            or _center_distance(first, second) <= _association_radius(first, second)
        )

    def observe_frame(self, detections, now=None):
        now = monotonic() if now is None else float(now)
        with self._lock:
            stale = [
                key for key, track in self._tracks.items()
                if now - track.last_seen > self.track_ttl_sec and not track.in_flight
            ]
            for key in stale:
                self._expired_track_ids.append(key)
                self.track_expired_count += 1
                del self._tracks[key]

            used = set()
            resolved = []
            for detection_id, bbox in detections:
                bbox = tuple(float(value) for value in bbox)
                candidates = [track for track in self._tracks.values() if track.track_id not in used]
                exact = [
                    track for track in candidates
                    if track.detection_id == detection_id
                    and self._compatible(track.bbox, bbox, self.association_iou)
                ]
                track = exact[0] if exact else None
                if track is None:
                    compatible = [
                        track for track in candidates
                        if self._compatible(track.bbox, bbox, self.association_iou)
                    ]
                    if compatible:
                        track = max(
                            compatible,
                            key=lambda candidate: (
                                bbox_iou(candidate.bbox, bbox),
                                -_center_distance(candidate.bbox, bbox),
                            ),
                        )
                if track is None:
                    if candidates:
                        self.association_failure_count += 1
                    track_id = 'semantic_track_{:04d}'.format(self._next_track)
                    self._next_track += 1
                    track = _Track(track_id, detection_id, bbox, now)
                    self._tracks[track_id] = track
                    self.cache_miss_count += 1
                else:
                    self.cache_hit_count += 1
                    overlap = bbox_iou(track.bbox, bbox)
                    track.bbox_changed = track.bbox_changed or overlap < self.bbox_retry_iou
                    track.detection_id = detection_id
                    track.bbox = bbox
                    track.last_seen = now
                used.add(track.track_id)
                resolved.append(track.track_id)
            return resolved

    def display_label(self, track_id):
        with self._lock:
            track = self._tracks.get(track_id)
            return track.label if track else UNKNOWN

    def claim_batch(self, track_ids, now=None):
        """Atomically claim every due track from the current synchronized frame."""

        now = monotonic() if now is None else float(now)
        with self._lock:
            claimed = []
            for track_id in track_ids:
                track = self._tracks.get(track_id)
                if track is None or track.in_flight:
                    continue
                cache_expired = (
                    track.label_time > 0.0
                    and now - track.label_time >= self.cache_ttl_sec
                )
                if cache_expired and not track.soft_expiry_counted:
                    self.cache_expired_count += 1
                    track.soft_expiry_counted = True
                retry_due = (
                    track.label in ('none', UNKNOWN)
                    and now - max(track.label_time, track.last_submit) >= self.retry_cooldown_sec
                )
                first_request = track.label_time == 0.0 and track.last_submit == 0.0
                if not (first_request or cache_expired or retry_due or track.bbox_changed):
                    continue
                track.in_flight = True
                track.last_submit = now
                track.bbox_changed = False
                if track.label_time == 0.0:
                    track.label = PROCESSING
                claimed.append((
                    track.track_id,
                    track.detection_id,
                    track.bbox,
                ))
            return claimed

    def claim_next(self, track_ids, now=None):
        """Backward-compatible single-claim helper used by older tests/tools."""

        for track_id in track_ids:
            claimed = self.claim_batch([track_id], now=now)
            if claimed:
                return claimed[0]
        return None

    def release_dropped(self, track_id):
        with self._lock:
            track = self._tracks.get(track_id)
            if track:
                track.in_flight = False
                if track.label_time == 0.0:
                    track.label = PROCESSING

    def apply_result(self, result, now=None):
        now = monotonic() if now is None else float(now)
        with self._lock:
            track = self._tracks.get(result.track_id)
            if track is None:
                self.association_failure_count += 1
                return {
                    'accepted': False,
                    'retained': False,
                    'reason': 'track_missing',
                    'consecutive_failures': 0,
                }
            track.in_flight = False
            successful = (
                not result.error
                and result.label in CLOSED_LABELS
                and result.label != UNKNOWN
            )
            if successful:
                track.label = result.label
                track.label_time = now
                track.consecutive_failures = 0
                track.last_error = ''
                track.soft_expiry_counted = False
                return {
                    'accepted': True,
                    'retained': False,
                    'reason': 'success',
                    'consecutive_failures': 0,
                }

            track.consecutive_failures += 1
            track.last_error = result.error or 'invalid_result'
            has_valid_label = track.label in CLOSED_LABELS and track.label != 'none'
            retain = (
                has_valid_label
                and track.consecutive_failures < self.maximum_consecutive_failures
                and track.label_time > 0.0
                and now - track.label_time < self.hard_cache_ttl_sec
            )
            if not retain:
                track.label = UNKNOWN
            return {
                'accepted': not retain,
                'retained': retain,
                'reason': (
                    'failure_retained_cached_label'
                    if retain else 'consecutive_failure_limit'
                ),
                'consecutive_failures': track.consecutive_failures,
            }

    def consume_expired_track_ids(self):
        with self._lock:
            expired = tuple(self._expired_track_ids)
            self._expired_track_ids.clear()
            return expired

    def diagnostics(self):
        with self._lock:
            return {
                'current_track_count': len(self._tracks),
                'cache_hit_count': self.cache_hit_count,
                'cache_miss_count': self.cache_miss_count,
                'cache_expired_count': self.cache_expired_count,
                'association_failure_count': self.association_failure_count,
                'track_expired_count': self.track_expired_count,
            }

    def snapshot(self):
        with self._lock:
            return {
                key: {
                    'detection_id': track.detection_id,
                    'bbox': track.bbox,
                    'label': track.label,
                    'in_flight': track.in_flight,
                    'last_seen': track.last_seen,
                    'label_time': track.label_time,
                    'consecutive_failures': track.consecutive_failures,
                    'last_error': track.last_error,
                }
                for key, track in self._tracks.items()
            }


class QwenSemanticWorker:
    """One-load CUDA worker with one latest-frame batch waiting at most."""

    def __init__(
        self,
        model_path,
        result_callback,
        dropped_callback=None,
        event_callback=None,
        timeout_sec=5.0,
        max_new_tokens=8,
        prediction_log_path='',
        model_loader=None,
        infer_function=None,
        semantic_mode='cabin',
        hall_padding_ratio=0.40,
        hall_upscale=5.0,
    ):
        self.model_path = str(model_path)
        self.result_callback = result_callback
        self.dropped_callback = dropped_callback or (lambda _track_id: None)
        self.event_callback = event_callback or (lambda _level, _message: None)
        self.timeout_sec = float(timeout_sec)
        self.max_new_tokens = int(max_new_tokens)
        self.prediction_log_path = str(prediction_log_path or '')
        self._model_loader = model_loader
        self._infer_function = infer_function
        self.semantic_mode = str(semantic_mode)
        self.hall_padding_ratio = float(hall_padding_ratio)
        self.hall_upscale = float(hall_upscale)
        self._queue = Queue(maxsize=1)
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._busy = False
        self._state = 'LOADING'
        self._model = None
        self._processor = None
        self.model_load_count = 0
        self.cuda_verified = False
        self.load_error = ''
        self.queue_replaced_count = 0
        self.inference_timeout_count = 0
        self.parser_failure_count = 0
        self.inference_failure_count = 0
        self.last_successful_inference_time = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name='qwen3_vl_semantic_worker',
            daemon=True,
        )
        self._thread.start()

    @property
    def queue_size(self):
        return self._queue.qsize()

    @property
    def busy(self):
        with self._state_lock:
            return self._busy

    @property
    def state(self):
        with self._state_lock:
            return self._state

    @property
    def alive(self):
        return self._thread.is_alive()

    def can_accept_waiting(self):
        return self.alive and self._queue.qsize() == 0

    @staticmethod
    def _batch_tasks(batch):
        if isinstance(batch, SemanticBatchTask):
            return batch.tasks
        if isinstance(batch, SemanticTask):
            return (batch,)
        raise TypeError('Qwen queue accepts SemanticBatchTask or SemanticTask')

    def submit_latest(self, batch):
        dropped = None
        try:
            self._queue.put_nowait(batch)
        except Full:
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                dropped = None
            self._queue.put_nowait(batch)
        if dropped is not None:
            self.queue_replaced_count += 1
            dropped_tasks = self._batch_tasks(dropped)
            for task in dropped_tasks:
                self.dropped_callback(task.track_id)
            self.event_callback(
                'warning',
                'qwen_queue_replace replaced_count={} dropped_tracks={}'.format(
                    self.queue_replaced_count,
                    ','.join(task.track_id for task in dropped_tasks),
                ),
            )
        return dropped

    def _load_default(self):
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        import torch
        import transformers
        from transformers import AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback is forbidden')
        model_directory = Path(self.model_path)
        if not model_directory.is_dir():
            raise RuntimeError('local HF model directory missing: {}'.format(model_directory))
        processor = AutoProcessor.from_pretrained(
            str(model_directory),
            trust_remote_code=True,
            local_files_only=True,
        )
        model_class = getattr(transformers, 'Qwen3VLForConditionalGeneration', None)
        if model_class is None:
            raise RuntimeError('Qwen3VLForConditionalGeneration is unavailable')
        model = model_class.from_pretrained(
            str(model_directory),
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = model.to(torch.device('cuda:0'))
        model.eval()
        parameter_devices = {str(parameter.device) for parameter in model.parameters()}
        buffer_devices = {str(buffer.device) for buffer in model.buffers()}
        if parameter_devices != {'cuda:0'}:
            raise RuntimeError(
                'Qwen parameters not exclusively on cuda:0: {}'.format(sorted(parameter_devices))
            )
        if any(device != 'cuda:0' for device in buffer_devices):
            raise RuntimeError(
                'Qwen buffers not exclusively on cuda:0: {}'.format(sorted(buffer_devices))
            )
        return model, processor

    def _infer_default(self, model, processor, task):
        import torch

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': PROMPT},
            ],
        }]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
            images=[task.crop],
            return_tensors='pt',
            padding=True,
        )
        inputs = {
            key: value.to('cuda:0') if hasattr(value, 'to') else value
            for key, value in inputs.items()
        }
        input_length = inputs['input_ids'].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        output_ids = output_ids[:, input_length:]
        return processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _parse_label(self, raw_response):
        """Route to the correct parser based on mode."""
        if self.semantic_mode == 'hall':
            return hall_strict_parse_label(raw_response)
        return strict_parse_label(raw_response)

    def _append_prediction(self, task, result):
        if not self.prediction_log_path:
            return
        path = Path(self.prediction_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        fields = [
            'unix_time', 'track_id', 'detection_id', 'bbox', 'crop_width',
            'crop_height', 'raw_generation', 'parsed_label', 'latency_sec',
            'error', 'device', 'mode',
        ]
        with path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow({
                'unix_time': '{:.6f}'.format(time()),
                'track_id': task.track_id,
                'detection_id': task.detection_id,
                'bbox': ' '.join('{:.3f}'.format(value) for value in task.bbox),
                'crop_width': task.crop.width,
                'crop_height': task.crop.height,
                'raw_generation': result.raw_response,
                'parsed_label': result.label,
                'latency_sec': '{:.6f}'.format(result.latency_sec),
                'error': result.error,
                'device': 'cuda:0' if self.cuda_verified else 'UNAVAILABLE',
                'mode': self.semantic_mode,
            })

    def _run(self):
        try:
            loader = self._model_loader or self._load_default
            self._model, self._processor = loader()
            self.model_load_count += 1
            self.cuda_verified = True
            with self._state_lock:
                self._state = 'READY'
            self.event_callback('info', 'qwen_model_ready device=cuda:0 load_count=1')
        except Exception as exc:
            self.load_error = '{}: {}'.format(type(exc).__name__, exc)
            with self._state_lock:
                self._state = 'ERROR'
            self.event_callback('error', 'qwen_model_load_failed {}'.format(self.load_error))

        while not self._stop.is_set():
            try:
                batch = self._queue.get(timeout=0.1)
            except Empty:
                continue
            if batch is None:
                self._queue.task_done()
                break
            with self._state_lock:
                self._busy = True
                self._state = 'BUSY'
            tasks = self._batch_tasks(batch)
            batch_id = (
                batch.batch_id
                if isinstance(batch, SemanticBatchTask)
                else 'legacy_single'
            )
            self.event_callback(
                'info',
                'qwen_batch_start batch_id={} items={} visible={}'.format(
                    batch_id,
                    len(tasks),
                    (
                        len(batch.visible_track_ids)
                        if isinstance(batch, SemanticBatchTask) else len(tasks)
                    ),
                ),
            )
            for task in tasks:
                if self._stop.is_set():
                    self.dropped_callback(task.track_id)
                    continue
                self._process_task(task)
            with self._state_lock:
                self._busy = False
                self._state = 'ERROR' if self.load_error else 'READY'
            self._queue.task_done()

        with self._state_lock:
            self._busy = False
            self._state = 'STOPPED'

    def _process_task(self, task):
        started = monotonic()
        raw = ''
        error = ''
        label = UNKNOWN
        try:
            if self._model is None:
                raise RuntimeError(self.load_error or 'Qwen model unavailable')
            infer = self._infer_function or self._infer_default
            raw = infer(self._model, self._processor, task)
            elapsed = monotonic() - started
            if elapsed > self.timeout_sec:
                self.inference_timeout_count += 1
                error = 'timeout {:.3f}s > {:.3f}s'.format(
                    elapsed,
                    self.timeout_sec,
                )
            else:
                label = self._parse_label(raw)
                if label == UNKNOWN:
                    self.parser_failure_count += 1
                    error = 'strict_parse_failed'
                else:
                    self.last_successful_inference_time = monotonic()
        except Exception as exc:
            self.inference_failure_count += 1
            error = '{}: {}'.format(type(exc).__name__, exc)
            if 'out of memory' in str(exc).lower():
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        result = SemanticResult(
            track_id=task.track_id,
            detection_id=task.detection_id,
            label=label,
            raw_response=raw,
            latency_sec=monotonic() - started,
            error=error,
        )
        try:
            self._append_prediction(task, result)
            self.result_callback(result)
        except Exception as exc:
            self.event_callback(
                'error',
                'qwen_result_callback_failed {}'.format(exc),
            )

    def shutdown(self, join_timeout_sec=2.0):
        self._stop.set()
        try:
            pending = self._queue.get_nowait()
            self._queue.task_done()
            if pending is not None:
                for task in self._batch_tasks(pending):
                    self.dropped_callback(task.track_id)
        except Empty:
            pass
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        self._thread.join(timeout=float(join_timeout_sec))
        return not self._thread.is_alive()
