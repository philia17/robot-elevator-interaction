import threading
import time

import numpy as np
from PIL import Image

from elevator_vision.qwen_semantic_worker import (
    PROCESSING,
    UNKNOWN,
    QwenSemanticWorker,
    SemanticBatchTask,
    SemanticResult,
    SemanticTask,
    SemanticTrackCache,
    prepare_crop,
    strict_parse_label,
)


def _task(track_id, detection_id=None):
    return SemanticTask(
        track_id=track_id,
        detection_id=detection_id or track_id,
        bbox=(10.0, 10.0, 20.0, 20.0),
        crop=Image.new('RGB', (30, 30), 'black'),
        submitted_monotonic=time.monotonic(),
    )


def _batch(batch_id, *tasks):
    return SemanticBatchTask(
        batch_id=batch_id,
        frame_stamp_ns=123,
        visible_track_ids=tuple(task.track_id for task in tasks),
        tasks=tuple(tasks),
        submitted_monotonic=time.monotonic(),
    )


def test_strict_parser_accepts_only_complete_closed_label():
    assert strict_parse_label('open') == 'open'
    assert strict_parse_label('  B1\n') == 'b1'
    assert strict_parse_label('Label: open') == UNKNOWN
    assert strict_parse_label('open.') == UNKNOWN
    assert strict_parse_label('door_open') == UNKNOWN
    assert strict_parse_label('') == UNKNOWN


def test_crop_uses_clean_pixels_padding_and_minimum_size():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[40:50, 60:70] = (10, 20, 30)
    crop = prepare_crop(
        image,
        (60.0, 40.0, 70.0, 50.0),
        padding_ratio=0.20,
        minimum_size=30,
    )
    assert crop.width >= 30
    assert crop.height >= 30
    array = np.asarray(crop)
    assert array[:, :, 2].max() >= 30
    assert array[:, :, 2].max() > array[:, :, 1].max()


def test_tracker_prefers_id_then_recovers_with_geometry_and_cache():
    cache = SemanticTrackCache(
        cache_ttl_sec=30.0,
        retry_cooldown_sec=3.0,
        track_ttl_sec=5.0,
        association_iou=0.30,
        bbox_retry_iou=0.60,
    )
    first = cache.observe_frame([
        ('button_001', (10, 10, 30, 30)),
        ('button_002', (80, 10, 100, 30)),
    ], now=1.0)
    assert cache.display_label(first[0]) == PROCESSING
    claim = cache.claim_next(first, now=1.0)
    assert claim[0] == first[0]
    cache.apply_result(SemanticResult(
        track_id=first[0],
        detection_id='button_001',
        label='3',
        raw_response='3',
        latency_sec=0.2,
    ), now=1.2)
    swapped = cache.observe_frame([
        ('button_002', (11, 10, 31, 30)),
        ('button_001', (81, 10, 101, 30)),
    ], now=1.3)
    assert swapped[0] == first[0]
    assert swapped[1] == first[1]
    assert cache.display_label(swapped[0]) == '3'
    assert cache.claim_next(swapped, now=2.0)[0] == first[1]


def test_none_and_unknown_retry_only_after_cooldown():
    cache = SemanticTrackCache(retry_cooldown_sec=3.0)
    track_id = cache.observe_frame([('button_001', (0, 0, 20, 20))], now=10.0)[0]
    cache.claim_next([track_id], now=10.0)
    cache.apply_result(SemanticResult(
        track_id, 'button_001', 'none', 'none', 0.1,
    ), now=10.1)
    assert cache.claim_next([track_id], now=12.0) is None
    assert cache.claim_next([track_id], now=13.2)[0] == track_id


def test_frame_claim_contains_all_due_buttons_and_soft_ttl_keeps_labels():
    cache = SemanticTrackCache(
        cache_ttl_sec=30.0,
        hard_cache_ttl_sec=120.0,
    )
    track_ids = cache.observe_frame([
        ('button_001', (0, 0, 20, 20)),
        ('button_002', (40, 0, 60, 20)),
        ('button_003', (80, 0, 100, 20)),
    ], now=1.0)
    claimed = cache.claim_batch(track_ids, now=1.0)
    assert [item[0] for item in claimed] == track_ids
    for track_id, detection_id, _bbox in claimed:
        cache.apply_result(SemanticResult(
            track_id, detection_id, '3', '3', 0.1,
        ), now=1.2)
    refresh = cache.claim_batch(track_ids, now=31.3)
    assert [item[0] for item in refresh] == track_ids
    assert [cache.display_label(track_id) for track_id in track_ids] == [
        '3', '3', '3']
    assert cache.diagnostics()['cache_expired_count'] == 3


def test_single_failure_retains_valid_label_until_failure_limit():
    cache = SemanticTrackCache(
        hard_cache_ttl_sec=120.0,
        maximum_consecutive_failures=3,
    )
    track_id = cache.observe_frame([
        ('button_001', (0, 0, 20, 20)),
    ], now=1.0)[0]
    cache.claim_batch([track_id], now=1.0)
    cache.apply_result(SemanticResult(
        track_id, 'button_001', '4', '4', 0.1,
    ), now=1.1)
    cache.claim_batch([track_id], now=31.2)
    retained = cache.apply_result(SemanticResult(
        track_id, 'button_001', UNKNOWN, '', 0.1, 'strict_parse_failed',
    ), now=31.3)
    assert retained['retained']
    assert cache.display_label(track_id) == '4'


def test_track_survives_measured_frame_gap_but_expires_after_track_ttl():
    cache = SemanticTrackCache(track_ttl_sec=10.0)
    first = cache.observe_frame([
        ('button_001', (10, 10, 30, 30)),
    ], now=1.0)[0]
    after_seven_seconds = cache.observe_frame([
        ('button_009', (11, 10, 31, 30)),
    ], now=8.2)[0]
    assert after_seven_seconds == first
    assert cache.observe_frame([], now=18.3) == []
    assert first in cache.consume_expired_track_ids()


def test_latest_only_queue_model_load_once_and_submit_nonblocking():
    started = threading.Event()
    release = threading.Event()
    results = []
    dropped = []
    load_calls = []

    def loader():
        load_calls.append(1)
        return object(), object()

    def infer(_model, _processor, task):
        if task.track_id == 'track_1':
            started.set()
            assert release.wait(2.0)
        return '3'

    worker = QwenSemanticWorker(
        model_path='/unused-in-unit-test',
        result_callback=results.append,
        dropped_callback=dropped.append,
        model_loader=loader,
        infer_function=infer,
        timeout_sec=2.0,
    )
    submit_started = time.monotonic()
    worker.submit_latest(_task('track_1'))
    assert time.monotonic() - submit_started < 0.05
    assert started.wait(1.0)
    worker.submit_latest(_task('track_2'))
    worker.submit_latest(_task('track_3'))
    assert worker.queue_size == 1
    assert dropped == ['track_2']
    release.set()
    deadline = time.monotonic() + 2.0
    while len(results) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [result.track_id for result in results] == ['track_1', 'track_3']
    assert [result.label for result in results] == ['3', '3']
    assert worker.model_load_count == 1
    assert len(load_calls) == 1
    assert worker.queue_replaced_count == 1
    assert worker.shutdown()


def test_latest_queue_item_is_frame_batch_and_one_failure_is_isolated():
    results = []

    def infer(_model, _processor, task):
        if task.track_id == 'track_bad':
            raise RuntimeError('one crop failed')
        return '4'

    worker = QwenSemanticWorker(
        model_path='/unused-in-unit-test',
        result_callback=results.append,
        model_loader=lambda: (object(), object()),
        infer_function=infer,
    )
    worker.submit_latest(_batch(
        'frame_1',
        _task('track_good_1'),
        _task('track_bad'),
        _task('track_good_2'),
    ))
    deadline = time.monotonic() + 2.0
    while len(results) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [result.track_id for result in results] == [
        'track_good_1', 'track_bad', 'track_good_2']
    assert [result.label for result in results] == ['4', UNKNOWN, '4']
    assert worker.alive
    assert worker.shutdown()


def test_worker_exception_maps_to_unknown_without_thread_exit():
    results = []

    def failing_infer(_model, _processor, _task_value):
        raise RuntimeError('synthetic failure')

    worker = QwenSemanticWorker(
        model_path='/unused-in-unit-test',
        result_callback=results.append,
        model_loader=lambda: (object(), object()),
        infer_function=failing_infer,
    )
    worker.submit_latest(_task('track_failure'))
    deadline = time.monotonic() + 2.0
    while not results and time.monotonic() < deadline:
        time.sleep(0.01)
    assert results[0].label == UNKNOWN
    assert 'synthetic failure' in results[0].error
    assert worker.alive
    assert worker.shutdown()
