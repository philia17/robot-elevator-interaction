"""Pure validation and coordinate helpers for the Phase 2 optical-axis probe."""

from dataclasses import dataclass


CANONICAL_FRAME = 'camera_optical_frame'
RGB_ENCODINGS = {'rgb8', 'bgr8'}
DEPTH_ENCODINGS = {'16UC1', 'mono16'}
MAX_RECEIVE_AGE_S = 1.25
MAX_STAMP_AGE_S = 2.0
MAX_RGB_DEPTH_SKEW_MS = 50.0
DISPLAY_RAW_FRAME = 'gemini335_color_optical_frame'
CANVAS_WIDTH = 960
CANVAS_HEIGHT = 600
BANNER_HEIGHT = 60
PHASE2_PREVIEW_SCALE = 6


def phase2_preview_to_fullres(u: int, v: int) -> tuple[int, int]:
    return u * PHASE2_PREVIEW_SCALE, v * PHASE2_PREVIEW_SCALE


def fullres_to_phase2_preview(u: int, v: int) -> tuple[int, int]:
    return u // PHASE2_PREVIEW_SCALE, v // PHASE2_PREVIEW_SCALE


@dataclass(frozen=True)
class PreviewLayout:
    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int


@dataclass(frozen=True)
class StreamState:
    status: str
    reason: str


class FreshTimestampGate:
    """Require several advancing synchronized pairs before declaring READY."""

    def __init__(self, required_pairs: int = 3):
        self.required_pairs = required_pairs
        self.last_pair = None
        self.consecutive_pairs = 0

    def update(self, base_state: StreamState, rgb_stamp_ns: int, depth_stamp_ns: int):
        if base_state.status != 'READY':
            self.last_pair = None
            self.consecutive_pairs = 0
            return base_state
        pair = (rgb_stamp_ns, depth_stamp_ns)
        if pair != self.last_pair:
            self.last_pair = pair
            self.consecutive_pairs += 1
        if self.consecutive_pairs < self.required_pairs:
            return StreamState(
                'WAITING',
                f'freshening timestamps {self.consecutive_pairs}/{self.required_pairs}',
            )
        return base_state


def stamp_ns(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def assess_compressed_display(
    message,
    *,
    received_monotonic: float,
    now_monotonic: float,
    now_ros_ns: int,
) -> StreamState:
    """Validate the display-only high-resolution compressed RGB stream."""
    if message is None:
        return StreamState('WAITING', 'missing high-resolution display RGB')
    if not received_monotonic:
        return StreamState('WAITING', 'no receive time for display RGB')
    receive_age_s = now_monotonic - received_monotonic
    if receive_age_s > MAX_RECEIVE_AGE_S:
        return StreamState('STALE', f'no fresh display RGB for {receive_age_s:.2f}s')
    if message.header.frame_id != DISPLAY_RAW_FRAME:
        return StreamState('INVALID', f'display RGB frame_id={message.header.frame_id!r}')
    message_stamp_ns = stamp_ns(message)
    if message_stamp_ns <= 0:
        return StreamState('INVALID', 'display RGB timestamp is zero')
    stamp_age_s = (now_ros_ns - message_stamp_ns) / 1_000_000_000.0
    if stamp_age_s > MAX_STAMP_AGE_S or stamp_age_s < -0.1:
        return StreamState('STALE', f'display RGB timestamp age={stamp_age_s:.3f}s')
    if not message.data:
        return StreamState('INVALID', 'display RGB compressed payload is empty')
    return StreamState('READY', 'fresh high-resolution compressed display RGB')


def closest_timestamp_pair(images, depths):
    if not images or not depths:
        return None
    pairs = [(image, depth) for image in images for depth in depths]
    synchronized = [
        pair for pair in pairs
        if abs(stamp_ns(pair[0][0]) - stamp_ns(pair[1][0]))
        <= MAX_RGB_DEPTH_SKEW_MS * 1_000_000
    ]
    if synchronized:
        # Prefer the newest acceptable pair.  Choosing the absolute smallest
        # skew can otherwise pin the preview to an old exact match while newer
        # (<50 ms) synchronized frames are already available.
        return max(synchronized, key=lambda pair: min(pair[0][1], pair[1][1]))
    return min(
        pairs,
        key=lambda pair: abs(stamp_ns(pair[0][0]) - stamp_ns(pair[1][0])),
    )


def preview_layout(source_width: int, source_height: int) -> PreviewLayout:
    available_height = CANVAS_HEIGHT - BANNER_HEIGHT
    scale = min(CANVAS_WIDTH / source_width, available_height / source_height)
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    return PreviewLayout(
        x=(CANVAS_WIDTH - width) // 2,
        y=BANNER_HEIGHT + (available_height - height) // 2,
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
    )


def preview_to_source(x: int, y: int, layout: PreviewLayout):
    if not (
        layout.x <= x < layout.x + layout.width
        and layout.y <= y < layout.y + layout.height
    ):
        return None
    u = int((x - layout.x) * layout.source_width / layout.width)
    v = int((y - layout.y) * layout.source_height / layout.height)
    return (
        min(layout.source_width - 1, max(0, u)),
        min(layout.source_height - 1, max(0, v)),
    )


def is_black_or_low_information(image_bgr) -> bool:
    """Reject a black transport frame without rejecting normally dim scenes."""
    if image_bgr is None or image_bgr.size == 0:
        return True
    sample = image_bgr[::16, ::16]
    return float(sample.max()) < 8.0 or (
        float(sample.mean()) < 2.0 and float(sample.std()) < 1.0
    )


def assess_streams(
    image,
    depth,
    camera_info,
    *,
    received_monotonic: dict,
    now_monotonic: float,
    now_ros_ns: int,
) -> StreamState:
    messages = {'RGB': image, 'depth': depth, 'CameraInfo': camera_info}
    missing = [name for name, message in messages.items() if message is None]
    if missing:
        return StreamState('WAITING', f"missing {', '.join(missing)}")
    missing_receive = [name for name in messages if name not in received_monotonic]
    if missing_receive:
        return StreamState('WAITING', f"no receive time for {', '.join(missing_receive)}")
    stale_receive = [
        name for name in messages
        if now_monotonic - received_monotonic[name] > MAX_RECEIVE_AGE_S
    ]
    if stale_receive:
        return StreamState('STALE', f"no fresh {', '.join(stale_receive)}")

    for name, message in messages.items():
        if message.header.frame_id != CANONICAL_FRAME:
            return StreamState('INVALID', f'{name} frame_id={message.header.frame_id!r}')
        message_stamp_ns = stamp_ns(message)
        if message_stamp_ns <= 0:
            return StreamState('INVALID', f'{name} timestamp is zero')
        age_s = (now_ros_ns - message_stamp_ns) / 1_000_000_000.0
        if age_s > MAX_STAMP_AGE_S or age_s < -0.1:
            return StreamState('STALE', f'{name} timestamp age={age_s:.3f}s')

    if image.encoding not in RGB_ENCODINGS:
        return StreamState('INVALID', f'RGB encoding={image.encoding!r}')
    if depth.encoding not in DEPTH_ENCODINGS:
        return StreamState('INVALID', f'depth encoding={depth.encoding!r}')
    if image.width <= 0 or image.height <= 0:
        return StreamState('INVALID', 'RGB dimensions are empty')
    if (depth.width, depth.height) != (image.width, image.height):
        return StreamState('INVALID', 'RGB/depth dimensions differ')
    if (camera_info.width, camera_info.height) != (image.width, image.height):
        return StreamState('INVALID', 'CameraInfo dimensions differ from RGB')
    if len(camera_info.k) < 9 or camera_info.k[0] <= 0 or camera_info.k[4] <= 0:
        return StreamState('INVALID', 'CameraInfo intrinsics are missing/invalid')
    if image.step < image.width * 3 or len(image.data) < image.step * image.height:
        return StreamState('INVALID', 'RGB payload is truncated')
    if depth.step < depth.width * 2 or len(depth.data) < depth.step * depth.height:
        return StreamState('INVALID', 'depth payload is truncated')
    skew_ms = abs(stamp_ns(image) - stamp_ns(depth)) / 1_000_000.0
    if skew_ms > MAX_RGB_DEPTH_SKEW_MS:
        return StreamState('STALE', f'RGB-depth skew={skew_ms:.1f}ms')
    return StreamState('READY', 'fresh canonical RGB + aligned depth')


def evaluate_observations(observations: dict) -> dict:
    required = {'center', 'right', 'down', 'forward'}
    if set(observations) != required:
        return {'status': 'pending', 'reason': f'missing stages: {sorted(required - set(observations))}'}
    center = observations['center']
    right = observations['right']
    down = observations['down']
    forward = observations['forward']
    center_depth = float(center['depth_device_units'])
    forward_depth = float(forward['depth_device_units'])
    relative_depth_drop = (
        (center_depth - forward_depth) / center_depth
        if center_depth > 0.0 and forward_depth > 0.0
        else float('-inf')
    )
    checks = {
        'optical_x_right': right['u_px'] - center['u_px'] >= 100,
        'optical_y_down': down['v_px'] - center['v_px'] >= 100,
        # Depth scale is deliberately still pending.  A unitless 5% decrease
        # is large compared with static depth noise but works at the operator's
        # safe ~40 cm baseline with an approximately 5 cm forward move.
        'optical_z_forward': relative_depth_drop >= 0.05,
        'forward_lateral_hold': abs(forward['u_px'] - center['u_px']) <= 120
        and abs(forward['v_px'] - center['v_px']) <= 120,
    }
    return {
        'status': 'pass' if all(checks.values()) else 'fail',
        'checks': checks,
        'metrics': {'forward_depth_relative_drop': relative_depth_drop},
    }
