"""Interactive, no-motion physical optical-axis sign probe for Phase 2."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from .optical_axis_core import (
    BANNER_HEIGHT,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
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
    stamp_ns,
)
from .session_core import PROBE_STALE_EXIT, ProbeFreshnessDeadline

STAGE_KEYS = {
    ord(key): stage
    for key, stage in (
        ('c', 'center'), ('C', 'center'),
        ('r', 'right'), ('R', 'right'),
        ('d', 'down'), ('D', 'down'),
        ('f', 'forward'), ('F', 'forward'),
    )
}
PREVIEW_INTERVAL_S = 0.1
WINDOW_NAME = 'Phase 2 optical-axis probe (windowed only; click target; c/r/d/f; q quits)'


class OpticalAxisProbe(Node):
    def __init__(
        self, output_path: Path, *, startup_timeout_s: float = 30.0,
        stale_timeout_s: float = 30.0
    ) -> None:
        super().__init__('phase2_optical_axis_probe')
        self.output_path = output_path
        self.bridge = CvBridge()
        self.image = None
        self.depth = None
        self.image_queue = deque(maxlen=3)
        self.depth_queue = deque(maxlen=3)
        self.camera_info = None
        self.display_rgb = None
        self.display_received_monotonic = 0.0
        self.received_monotonic = {}
        self.source_click = None
        self.preview_click = None
        self.layout = None
        self.last_state = StreamState('WAITING', 'starting')
        self.last_logged_status = None
        self.last_heartbeat_monotonic = 0.0
        self.last_depth_valid_fraction = None
        self.cached_display_stamp_ns = None
        self.cached_display_bgr = None
        self.cached_display_layout = None
        self.cached_display_is_black = False
        self.cached_depth_stamp_ns = None
        self.fresh_timestamp_gate = FreshTimestampGate(required_pairs=3)
        self.freshness_deadline = ProbeFreshnessDeadline(
            time.monotonic(),
            startup_timeout_s=startup_timeout_s,
            stale_timeout_s=stale_timeout_s,
        )
        self.observations = {}
        self.completed = False
        self.stream_failure_reason = None
        preview_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, '/lift_robot/camera/phase2_preview/rgbd', self._on_rgbd, preview_qos
        )
        self.create_subscription(
            CameraInfo,
            '/lift_robot/camera/phase2_preview/color/camera_info',
            self._on_camera_info,
            preview_qos,
        )
        self.create_subscription(
            CompressedImage,
            '/gemini335/color/image_raw/compressed',
            self._on_display_rgb,
            preview_qos,
        )

    def _on_rgbd(self, message) -> None:
        if message.encoding != '8UC5' or message.step < message.width * 5:
            self.get_logger().error('dropping malformed atomic Phase 2 RGB-D preview')
            return
        try:
            packed = np.frombuffer(message.data, dtype=np.uint8).reshape(
                message.height, message.step
            )[:, :message.width * 5].reshape(message.height, message.width, 5)
        except ValueError:
            self.get_logger().error('dropping truncated atomic Phase 2 RGB-D preview')
            return
        rgb = Image()
        rgb.header = message.header
        rgb.height = message.height
        rgb.width = message.width
        rgb.encoding = 'rgb8'
        rgb.is_bigendian = 0
        rgb.step = message.width * 3
        rgb.data = np.ascontiguousarray(packed[:, :, :3]).tobytes()
        depth = Image()
        depth.header = message.header
        depth.height = message.height
        depth.width = message.width
        depth.encoding = '16UC1'
        depth.is_bigendian = 0
        depth.step = message.width * 2
        depth.data = np.ascontiguousarray(packed[:, :, 3:5]).tobytes()
        received = time.monotonic()
        self.image_queue.append((rgb, received))
        self.depth_queue.append((depth, received))

    def _on_camera_info(self, message) -> None:
        self.camera_info = message
        self.received_monotonic['CameraInfo'] = time.monotonic()

    def _on_display_rgb(self, message) -> None:
        self.display_rgb = message
        self.display_received_monotonic = time.monotonic()

    def _mouse(self, event, x, y, _flags, _parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.layout is None:
                self.get_logger().warning('preview is not ready; click ignored')
                return
            source = preview_to_source(int(x), int(y), self.layout)
            if source is None:
                self.get_logger().warning('click inside the camera image, not the status area')
                return
            self.preview_click = (int(x), int(y))
            self.source_click = source

    def _state(self) -> StreamState:
        pair = closest_timestamp_pair(self.image_queue, self.depth_queue)
        if pair is not None:
            (self.image, image_received), (self.depth, depth_received) = pair
            self.received_monotonic['RGB'] = image_received
            self.received_monotonic['depth'] = depth_received
        else:
            return StreamState(
                'WAITING',
                f'queued RGB={len(self.image_queue)} depth={len(self.depth_queue)}',
            )
        state = assess_streams(
            self.image,
            self.depth,
            self.camera_info,
            received_monotonic=self.received_monotonic,
            now_monotonic=time.monotonic(),
            now_ros_ns=self.get_clock().now().nanoseconds,
        )
        state = self.fresh_timestamp_gate.update(
            state, stamp_ns(self.image), stamp_ns(self.depth)
        )
        if state.status != 'READY':
            return state
        return assess_compressed_display(
            self.display_rgb,
            received_monotonic=self.display_received_monotonic,
            now_monotonic=time.monotonic(),
            now_ros_ns=self.get_clock().now().nanoseconds,
        )

    def _decode_rgb(self):
        image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
        if image.shape[:2] != (self.image.height, self.image.width):
            raise ValueError('decoded RGB dimensions differ from message metadata')
        return image

    def _decode_display_rgb(self):
        if self.display_rgb is None:
            raise ValueError('high-resolution display RGB is unavailable')
        encoded = np.frombuffer(self.display_rgb.data, dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError('high-resolution display RGB payload is empty')
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('OpenCV could not decode high-resolution display RGB')
        if image.shape[:2] != (1080, 1920):
            raise ValueError(f'display RGB dimensions {image.shape[1]}x{image.shape[0]} != 1920x1080')
        return image

    def _capture(self, stage: str) -> None:
        state = self._state()
        if state.status != 'READY':
            self.get_logger().error(f'capture rejected: {state.status}: {state.reason}')
            return
        if self.source_click is None:
            self.get_logger().error('capture rejected: click the target in the READY preview first')
            return
        try:
            rgb_array = self._decode_rgb()
        except (ValueError, cv2.error) as exc:
            self.get_logger().error(f'capture rejected: RGB decode failed: {exc}')
            return
        if is_black_or_low_information(rgb_array):
            self.get_logger().error('capture rejected: RGB frame is black/low-information')
            return
        depth_array = self.bridge.imgmsg_to_cv2(self.depth, desired_encoding='passthrough')
        if depth_array.shape[:2] != (self.depth.height, self.depth.width):
            self.get_logger().error('capture rejected: decoded depth dimensions differ')
            return
        u, v = self.source_click
        preview_u, preview_v = fullres_to_phase2_preview(u, v)
        preview_u = min(self.depth.width - 1, max(0, preview_u))
        preview_v = min(self.depth.height - 1, max(0, preview_v))
        radius = 5
        values = depth_array[
            max(0, preview_v - radius):preview_v + radius + 1,
            max(0, preview_u - radius):preview_u + radius + 1,
        ].reshape(-1)
        valid = [int(value) for value in values if int(value) > 0]
        if len(valid) < 20:
            self.get_logger().error('not enough valid aligned-depth pixels around click')
            return
        raw_depth = float(statistics.median(valid))
        valid_fraction = len(valid) / len(values)
        if valid_fraction < 0.5:
            self.get_logger().error(
                f'capture rejected: local valid-depth fraction {valid_fraction:.2f} < 0.50'
            )
            return
        skew_ms = abs(stamp_ns(self.image) - stamp_ns(self.depth)) / 1_000_000.0
        fx, fy = float(self.camera_info.k[0]), float(self.camera_info.k[4])
        cx, cy = float(self.camera_info.k[2]), float(self.camera_info.k[5])
        self.observations[stage] = {
            'u_px': u,
            'v_px': v,
            'preview_u_px': preview_u,
            'preview_v_px': preview_v,
            'depth_device_units': raw_depth,
            'projective_x_device_units': (preview_u - cx) * raw_depth / fx,
            'projective_y_device_units': (preview_v - cy) * raw_depth / fy,
            'rgb_stamp_ns': stamp_ns(self.image),
            'depth_stamp_ns': stamp_ns(self.depth),
            'rgb_depth_skew_ms': skew_ms,
            'local_valid_depth_fraction': valid_fraction,
            'frame_id': self.image.header.frame_id,
        }
        result = evaluate_observations(self.observations)
        document = {
            'schema_version': 1,
            'test': 'Gemini 335 canonical physical optical-axis sign probe',
            'depth_scale_status': 'pending_phase12',
            'depth_values_are_device_units_not_metres': True,
            'observations': self.observations,
            'result': result,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
        self.get_logger().info(f'recorded {stage}: u={u}, v={v}, raw_depth={raw_depth}; {result}')
        if result.get('status') == 'pass':
            self.completed = True

    def _render(self):
        state = self._state()
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        self.layout = None
        if self.display_rgb is not None:
            try:
                current_stamp_ns = stamp_ns(self.display_rgb)
                if current_stamp_ns != self.cached_display_stamp_ns:
                    rgb = self._decode_display_rgb()
                    self.cached_display_stamp_ns = current_stamp_ns
                    self.cached_display_bgr = None
                    self.cached_display_layout = None
                    self.cached_display_is_black = is_black_or_low_information(rgb)
                    if not self.cached_display_is_black:
                        layout = preview_layout(rgb.shape[1], rgb.shape[0])
                        self.cached_display_bgr = cv2.resize(
                            rgb, (layout.width, layout.height), interpolation=cv2.INTER_AREA
                        )
                        self.cached_display_layout = layout
                if self.cached_display_is_black:
                    state = StreamState('INVALID', 'display RGB is black/low-information')
                if self.cached_display_bgr is not None:
                    layout = self.cached_display_layout
                    canvas[
                        layout.y:layout.y + layout.height,
                        layout.x:layout.x + layout.width,
                    ] = self.cached_display_bgr
                    self.layout = layout
            except (ValueError, cv2.error) as exc:
                state = StreamState('INVALID', f'display RGB decode failed: {exc}')
        if state.status == 'READY':
            try:
                current_depth_stamp_ns = stamp_ns(self.depth)
                if current_depth_stamp_ns != self.cached_depth_stamp_ns:
                    depth = self.bridge.imgmsg_to_cv2(self.depth, desired_encoding='passthrough')
                    sample = depth[::24, ::24]
                    self.last_depth_valid_fraction = float(np.count_nonzero(sample)) / sample.size
                    self.cached_depth_stamp_ns = current_depth_stamp_ns
                if self.last_depth_valid_fraction < 0.01:
                    state = StreamState(
                        'INVALID',
                        f'depth valid fraction={self.last_depth_valid_fraction:.3f} < 0.01',
                    )
            except (ValueError, cv2.error) as exc:
                state = StreamState('INVALID', f'depth decode failed: {exc}')
        self.last_state = state
        now = time.monotonic()
        self.stream_failure_reason = self.freshness_deadline.update(now, state.status)
        if state.status != self.last_logged_status or now - self.last_heartbeat_monotonic >= 30.0:
            # Keep one fixed severity at this call site; rclpy rejects changing
            # logger severity dynamically for an otherwise identical call site.
            self.get_logger().info(f'preview_state={state.status}: {state.reason}')
            self.last_logged_status = state.status
            self.last_heartbeat_monotonic = now
        colors = {
            'READY': (0, 180, 0),
            'WAITING': (0, 180, 255),
            'STALE': (0, 100, 255),
            'INVALID': (0, 0, 255),
        }
        color = colors[state.status]
        if state.status != 'READY':
            self.source_click = None
            self.preview_click = None
            if self.layout is not None:
                region = canvas[
                    self.layout.y:self.layout.y + self.layout.height,
                    self.layout.x:self.layout.x + self.layout.width,
                ]
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                region[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 4
                display_age = (
                    time.monotonic() - self.display_received_monotonic
                    if self.display_received_monotonic else float('inf')
                )
                age_text = (
                    f'last display age {display_age:.1f}s'
                    if np.isfinite(display_age) else 'no display RGB'
                )
                cv2.putText(
                    canvas,
                    f'FROZEN / {state.status}',
                    (self.layout.x + 180, self.layout.y + self.layout.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    canvas,
                    age_text,
                    (self.layout.x + 300, self.layout.y + self.layout.height // 2 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 180, 255),
                    2,
                )
        cv2.rectangle(canvas, (0, 0), (CANVAS_WIDTH, BANNER_HEIGHT), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f'{state.status}: {state.reason}'[:105],
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )
        cv2.putText(
            canvas,
            f"Saved: {','.join(self.observations) or 'none'}. Click target; c/r/d/f. q quits."
            + (
                f' depth-valid={self.last_depth_valid_fraction:.2f}'
                if self.last_depth_valid_fraction is not None else ''
            ),
            (14, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
        )
        if self.preview_click is not None and self.layout is not None:
            cv2.drawMarker(canvas, self.preview_click, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.imshow(WINDOW_NAME, canvas)

    @staticmethod
    def _enforce_windowed_size() -> None:
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) > 0:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            _x, _y, width, height = cv2.getWindowImageRect(WINDOW_NAME)
            if width > CANVAS_WIDTH + 40 or height > CANVAS_HEIGHT + 80:
                cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)
        except cv2.error:
            pass

    def run(self) -> int:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse)
        next_preview = 0.0
        next_window_check = 0.0
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.005)
            now = time.monotonic()
            if now >= next_preview:
                self._render()
                next_preview = now + PREVIEW_INTERVAL_S
                if self.stream_failure_reason is not None:
                    self.get_logger().error(
                        f'persistent frozen preview: {self.stream_failure_reason}'
                    )
                    cv2.waitKey(250)
                    cv2.destroyAllWindows()
                    return PROBE_STALE_EXIT
                if self.completed:
                    cv2.destroyAllWindows()
                    return 0
            if now >= next_window_check:
                self._enforce_windowed_size()
                next_window_check = now + 0.5
            key = cv2.waitKey(1) & 0xFF
            if key in STAGE_KEYS:
                self._capture(STAGE_KEYS[key])
            elif key == ord('q'):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            time.sleep(0.002)
        cv2.destroyAllWindows()
        return 0


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--startup-timeout-s', type=float, default=30.0)
    parser.add_argument('--stale-timeout-s', type=float, default=30.0)
    parsed = parser.parse_args(remove_ros_args(args=args)[1:])
    if parsed.output.exists():
        raise SystemExit(f'refusing to overwrite existing evidence: {parsed.output}')
    rclpy.init(args=args)
    node = OpticalAxisProbe(
        parsed.output.expanduser().resolve(),
        startup_timeout_s=parsed.startup_timeout_s,
        stale_timeout_s=parsed.stale_timeout_s,
    )
    try:
        exit_code = node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)
