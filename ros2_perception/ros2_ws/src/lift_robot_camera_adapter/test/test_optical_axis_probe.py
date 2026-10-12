from types import SimpleNamespace

import numpy as np

from lift_robot_camera_adapter.optical_axis_core import (
    CANONICAL_FRAME,
    FreshTimestampGate,
    StreamState,
    assess_compressed_display,
    assess_streams,
    closest_timestamp_pair,
    evaluate_observations,
    fullres_to_phase2_preview,
    is_black_or_low_information,
    preview_layout,
    preview_to_source,
    phase2_preview_to_fullres,
)


def _observation(u, v, depth):
    return {'u_px': u, 'v_px': v, 'depth_device_units': depth}


def test_axis_probe_passes_only_for_right_down_and_toward_camera():
    result = evaluate_observations({
        'center': _observation(960, 540, 800),
        'right': _observation(1200, 540, 800),
        'down': _observation(960, 780, 800),
        'forward': _observation(960, 540, 600),
    })
    assert result['status'] == 'pass'
    assert all(result['checks'].values())


def test_axis_probe_rejects_opposite_signs():
    result = evaluate_observations({
        'center': _observation(960, 540, 800),
        'right': _observation(800, 540, 800),
        'down': _observation(960, 400, 800),
        'forward': _observation(960, 540, 1000),
    })
    assert result['status'] == 'fail'


def test_axis_probe_accepts_safe_40cm_baseline_with_5cm_forward_move():
    result = evaluate_observations({
        'center': _observation(960, 540, 400),
        'right': _observation(1200, 540, 400),
        'down': _observation(960, 780, 400),
        'forward': _observation(960, 540, 350),
    })
    assert result['status'] == 'pass'
    assert result['metrics']['forward_depth_relative_drop'] == 0.125


def test_axis_probe_rejects_small_depth_change_within_noise_margin():
    result = evaluate_observations({
        'center': _observation(960, 540, 400),
        'right': _observation(1200, 540, 400),
        'down': _observation(960, 780, 400),
        'forward': _observation(960, 540, 390),
    })
    assert result['status'] == 'fail'
    assert not result['checks']['optical_z_forward']


def _message(*, width=1920, height=1080, encoding='rgb8', stamp_ns=10_000_000_000):
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id=CANONICAL_FRAME,
            stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000),
        ),
        width=width,
        height=height,
        encoding=encoding,
        k=[1041.0, 0.0, 960.0, 0.0, 1042.0, 540.0, 0.0, 0.0, 1.0],
        step=width * (2 if encoding in {'16UC1', 'mono16'} else 3),
        data=bytes(width * height * (2 if encoding in {'16UC1', 'mono16'} else 3)),
    )


def test_preview_mapping_round_trips_full_resolution_coordinates():
    layout = preview_layout(1920, 1080)
    assert (layout.width, layout.height) == (960, 540)
    assert preview_to_source(layout.x, layout.y, layout) == (0, 0)
    assert preview_to_source(layout.x + 480, layout.y + 270, layout) == (960, 540)
    assert preview_to_source(10, 10, layout) is None


def test_phase2_half_resolution_click_maps_to_full_resolution_evidence():
    assert phase2_preview_to_fullres(160, 90) == (960, 540)
    assert fullres_to_phase2_preview(960, 540) == (160, 90)
    assert fullres_to_phase2_preview(965, 545) == (160, 90)


def test_compressed_display_must_be_fresh_expected_vendor_frame_and_nonempty():
    display = _message(encoding='jpeg')
    display.header.frame_id = 'gemini335_color_optical_frame'
    display.data = b'jpeg'
    ready = assess_compressed_display(
        display,
        received_monotonic=4.9,
        now_monotonic=5.0,
        now_ros_ns=10_100_000_000,
    )
    assert ready.status == 'READY'
    display.header.frame_id = 'unexpected_frame'
    invalid = assess_compressed_display(
        display,
        received_monotonic=4.9,
        now_monotonic=5.0,
        now_ros_ns=10_100_000_000,
    )
    assert invalid.status == 'INVALID'


def test_stream_state_ready_only_for_fresh_canonical_matching_streams():
    image = _message()
    depth = _message(encoding='16UC1', stamp_ns=9_980_000_000)
    info = _message(encoding='unused')
    state = assess_streams(
        image,
        depth,
        info,
        received_monotonic={'RGB': 4.9, 'depth': 4.9, 'CameraInfo': 4.9},
        now_monotonic=5.0,
        now_ros_ns=10_100_000_000,
    )
    assert state == StreamState('READY', 'fresh canonical RGB + aligned depth')


def test_stream_state_rejects_stale_wrong_frame_and_skew():
    image = _message()
    depth = _message(encoding='16UC1')
    info = _message(encoding='unused')
    common = dict(image=image, depth=depth, camera_info=info, now_ros_ns=10_100_000_000)
    stale = assess_streams(
        **common,
        received_monotonic={'RGB': 3.5, 'depth': 4.9, 'CameraInfo': 4.9},
        now_monotonic=5.0,
    )
    assert stale.status == 'STALE'
    image.header.frame_id = 'vendor_frame'
    invalid = assess_streams(
        **common,
        received_monotonic={'RGB': 4.9, 'depth': 4.9, 'CameraInfo': 4.9},
        now_monotonic=5.0,
    )
    assert invalid.status == 'INVALID'
    image.header.frame_id = CANONICAL_FRAME
    depth.header.stamp.nanosec = 900_000_000
    skew = assess_streams(
        **common,
        received_monotonic={'RGB': 4.9, 'depth': 4.9, 'CameraInfo': 4.9},
        now_monotonic=5.0,
    )
    assert skew.status == 'STALE'


def test_black_frame_detection_is_fail_closed_but_accepts_real_texture():
    assert is_black_or_low_information(np.zeros((100, 100, 3), dtype=np.uint8))
    textured = np.zeros((100, 100, 3), dtype=np.uint8)
    textured[::2] = 200
    assert not is_black_or_low_information(textured)


def test_closest_timestamp_pair_selects_synced_messages_not_latest_mismatch():
    images = [(_message(stamp_ns=10_000_000_000), 1.0), (_message(stamp_ns=10_100_000_000), 1.1)]
    depths = [
        (_message(encoding='16UC1', stamp_ns=10_001_000_000), 1.0),
        (_message(encoding='16UC1', stamp_ns=10_300_000_000), 1.1),
    ]
    image_pair, depth_pair = closest_timestamp_pair(images, depths)
    assert image_pair[0].header.stamp.sec == 10
    assert image_pair[0].header.stamp.nanosec == 0
    assert depth_pair[0].header.stamp.nanosec == 1_000_000


def test_fresh_timestamp_gate_requires_three_advancing_pairs_and_resets_on_stall():
    gate = FreshTimestampGate(required_pairs=3)
    ready = StreamState('READY', 'fresh canonical RGB + aligned depth')
    assert gate.update(ready, 1, 1).status == 'WAITING'
    assert gate.update(ready, 1, 1).status == 'WAITING'
    assert gate.update(ready, 2, 2).status == 'WAITING'
    assert gate.update(ready, 3, 3).status == 'READY'
    assert gate.update(StreamState('STALE', 'stalled'), 3, 3).status == 'STALE'
    assert gate.update(ready, 4, 4).status == 'WAITING'
