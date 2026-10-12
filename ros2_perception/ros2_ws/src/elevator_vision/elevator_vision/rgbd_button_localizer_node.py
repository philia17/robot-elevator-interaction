"""Synchronize 2-D detections with aligned depth and publish camera-frame XYZ."""

import json
import math
import gc
import hashlib
import threading
from time import monotonic

from cv_bridge import CvBridge
import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool

from elevator_interfaces.msg import (
    ButtonDetectionArray,
    ButtonObservation3D,
    ButtonObservation3DArray,
    ButtonSemantic,
    ButtonSemanticArray,
)

from .rgbd_geometry import (
    estimate_surface_geometry_from_bbox,
    project_pixel_to_camera,
    robust_depth_from_bbox,
    scale_pixel_between_images,
    surface_provenance_complete,
)
from .qwen_semantic_worker import (
    UNKNOWN,
    QwenSemanticWorker,
    SemanticBatchTask,
    SemanticResult,
    SemanticTask,
    SemanticTrackCache,
    _build_hall_4b_loader,
    _build_hall_infer_function,
    prepare_crop,
    prepare_crop_hall,
)
from .semantic_display import (
    CACHED,
    FRESH,
    PanelEntry,
    SemanticResultView,
    StableDisplayRegistry,
    SystemStatus,
    red_local_contrast_v1,
    render_semantic_display,
    semantic_display_state,
)


def _stamp_ns(message):
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def _camera_info_sha256(message):
    payload = {
        'frame_id': str(message.header.frame_id),
        'width': int(message.width),
        'height': int(message.height),
        'distortion_model': str(message.distortion_model),
        'd': [float(value) for value in message.d],
        'k': [float(value) for value in message.k],
        'r': [float(value) for value in message.r],
        'p': [float(value) for value in message.p],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _regions_overlap(first, second):
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _draw_text_block(image, lines, anchor_x, bbox_top, bbox_bottom, color, used):
    height, width = image.shape[:2]
    ui_scale = max(1.0, min(width / 640.0, height / 480.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 * ui_scale
    thickness = max(1, int(round(ui_scale)))
    padding = max(3, int(round(3 * ui_scale)))
    line_height = max(17, int(round(17 * ui_scale)))
    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    block_width = text_width + 2 * padding
    block_height = len(lines) * line_height + 2 * padding
    left = max(0, min(width - block_width, int(anchor_x)))
    candidates = [
        int(bbox_top) - block_height - 3,
        int(bbox_bottom) + 3,
        int(bbox_top),
    ]
    candidates.extend(range(0, max(1, height - block_height), block_height + 2))
    selected = None
    for top in candidates:
        top = max(0, min(height - block_height, top))
        region = (left, top, left + block_width, top + block_height)
        if not any(_regions_overlap(region, other) for other in used):
            selected = region
            break
    if selected is None:
        top = max(0, min(height - block_height, candidates[0]))
        selected = (left, top, left + block_width, top + block_height)
    used.append(selected)
    left, top, right, bottom = selected
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        baseline_y = top + padding + (index + 1) * line_height - 4
        cv2.putText(
            image,
            line,
            (left + padding, baseline_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


class RgbdButtonLocalizerNode(Node):
    def __init__(self):
        super().__init__('rgbd_button_localizer_node')
        self.declare_parameter('rgb_topic', '/elevator/selected_frame')
        self.declare_parameter('detections_topic', '/elevator/button_detections')
        self.declare_parameter(
            'aligned_depth_topic',
            '/lift_robot/camera/aligned_depth/image_raw',
        )
        self.declare_parameter(
            'camera_info_topic',
            '/lift_robot/camera/color/camera_info',
        )
        self.declare_parameter(
            'output_observations_topic',
            '/elevator/button_observations_3d',
        )
        self.declare_parameter('output_debug_image_topic', '/elevator/som_image_3d')
        self.declare_parameter('color_optical_frame', 'camera_optical_frame')
        self.declare_parameter('sync_queue_size', 5)
        self.declare_parameter('sync_slop_sec', 0.05)
        self.declare_parameter('maximum_sync_delta_ms', 50.0)
        self.declare_parameter('camera_info_max_age_sec', 1.0)
        self.declare_parameter('min_depth_m', 0.15)
        self.declare_parameter('max_depth_m', 5.0)
        self.declare_parameter('minimum_valid_pixels', 9)
        self.declare_parameter('center_roi_ratio', 0.30)
        self.declare_parameter('fallback_window_size', 9)
        self.declare_parameter('surface_geometry_enabled', True)
        self.declare_parameter('plane_annulus_expansion_ratio', 1.8)
        self.declare_parameter('plane_annulus_inner_ratio', 1.0)
        self.declare_parameter('maximum_plane_depth_delta_m', 0.08)
        self.declare_parameter('minimum_plane_points', 80)
        self.declare_parameter('maximum_plane_points', 2000)
        self.declare_parameter('minimum_plane_inlier_ratio', 0.65)
        self.declare_parameter('maximum_plane_rms_residual_m', 0.008)
        self.declare_parameter('minimum_plane_span_m', 0.005)
        self.declare_parameter('plane_mad_rejection_multiplier', 3.5)
        self.declare_parameter('surface_pixel_sigma_px', 1.0)
        self.declare_parameter('surface_depth_noise_floor_m', 0.001)
        self.declare_parameter(
            'maximum_normal_uncertainty_rad',
            math.radians(10.0),
        )
        self.declare_parameter(
            'surface_geometry_method_id',
            'expanded_bbox_annulus_svd_mad_v1',
        )
        self.declare_parameter(
            'surface_geometry_scope',
            'no_motion_diagnostic',
        )
        self.declare_parameter(
            'camera_optical_axis_status',
            'pending_physical_xyz_validation',
        )
        self.declare_parameter('canonical_frame_id', '')
        self.declare_parameter('raw_to_canonical_alias_id', '')
        self.declare_parameter('raw_to_canonical_alias_status', '')
        self.declare_parameter('raw_to_canonical_adapter_sha256', '')
        self.declare_parameter('section_b_calibration_id', '')
        self.declare_parameter(
            'rgb_depth_registration_id',
            'gemini335_aligned_depth_to_color_nominal',
        )
        self.declare_parameter(
            'rgb_depth_registration_status',
            'verified_config_and_observed_accuracy_pending',
        )
        self.declare_parameter('depth_scale_m_per_unit', 0.001)
        self.declare_parameter(
            'depth_scale_status',
            'verified_driver_sdk_metric_accuracy_pending',
        )
        self.declare_parameter('semantic_enabled', True)
        self.declare_parameter('semantic_mode', 'cabin')
        self.declare_parameter(
            'semantic_output_topic',
            '/elevator/som_image_semantic',
        )
        self.declare_parameter(
            'structured_semantics_topic',
            '/elevator/button_semantics',
        )
        self.declare_parameter(
            'qwen_model_path',
            '/data/models/qwen3-vl-2b-hf',
        )
        self.declare_parameter('qwen_padding_ratio', 0.15)
        self.declare_parameter('qwen_minimum_crop_size', 30)
        self.declare_parameter('qwen_timeout_sec', 5.0)
        self.declare_parameter('qwen_max_new_tokens', 8)
        self.declare_parameter('qwen_cache_ttl_sec', 30.0)
        self.declare_parameter('qwen_cache_hard_ttl_sec', 120.0)
        self.declare_parameter('qwen_retry_cooldown_sec', 3.0)
        self.declare_parameter('qwen_track_ttl_sec', 3.0)
        self.declare_parameter('qwen_association_iou', 0.30)
        self.declare_parameter('qwen_bbox_retry_iou', 0.65)
        self.declare_parameter('qwen_max_consecutive_failures', 3)
        self.declare_parameter('qwen_worker_recovery_cooldown_sec', 10.0)
        self.declare_parameter('qwen_worker_max_restarts', 1)
        self.declare_parameter('qwen_prediction_log_path', '')
        self.declare_parameter('hall_model_path', '/data/models/qwen3-vl-4b-hf')
        self.declare_parameter('hall_reference_up',
                               '/opt/robot-elevator-interaction/scripts/sprint3/assets/hall_reference_up.png')
        self.declare_parameter('hall_reference_down',
                               '/opt/robot-elevator-interaction/scripts/sprint3/assets/hall_reference_down.png')
        self.declare_parameter('hall_padding_ratio', 0.40)
        self.declare_parameter('hall_upscale', 5.0)
        self.declare_parameter('semantic_fresh_duration_sec', 1.5)
        self.declare_parameter('semantic_display_stale_timeout_sec', 1.5)
        self.declare_parameter('semantic_display_log_interval_sec', 5.0)

        self._bridge = CvBridge()
        self._camera_info = None
        self._frames_synchronized = 0
        self._frames_rejected = 0
        self._valid_observations = 0
        self._invalid_observations = 0
        self._semantic_frames_published = 0
        self._semantic_worker = None
        self._semantic_mode = 'cabin'
        self._semantic_result_lock = threading.Lock()
        self._semantic_results = {}
        self._semantic_last_latency_ms = math.nan
        self._semantic_dropped_tasks = 0
        self._semantic_cache_hits = 0
        self._semantic_cache_misses = 0
        self._semantic_frame_started = monotonic()
        self._semantic_display_log_state = {}
        self._semantic_previous_display = {}
        self._semantic_display_last_summary = 0.0
        self._semantic_health_last_log = 0.0
        self._semantic_crop_invalid_count = 0
        self._semantic_timestamp_mismatch_count = 0
        self._semantic_worker_generation = 0
        self._semantic_worker_restart_count = 0
        self._semantic_worker_last_restart_attempt = 0.0
        self._semantic_batch_sequence = 0

        self._maximum_sync_delta_ms = float(
            self.get_parameter('maximum_sync_delta_ms').value
        )
        self._camera_info_max_age_ns = int(
            float(self.get_parameter('camera_info_max_age_sec').value) * 1e9
        )
        self._expected_frame = str(
            self.get_parameter('color_optical_frame').value
        )

        image_qos = QoSProfile(depth=5)
        image_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        image_qos.durability = DurabilityPolicy.VOLATILE
        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE
        detections_qos = QoSProfile(depth=5)
        detections_qos.reliability = ReliabilityPolicy.RELIABLE
        detections_qos.durability = DurabilityPolicy.VOLATILE

        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self._on_camera_info,
            sensor_qos,
        )
        self._rgb_subscription = message_filters.Subscriber(
            self,
            Image,
            self.get_parameter('rgb_topic').value,
            qos_profile=image_qos,
        )
        self._detections_subscription = message_filters.Subscriber(
            self,
            ButtonDetectionArray,
            self.get_parameter('detections_topic').value,
            qos_profile=detections_qos,
        )
        self._depth_subscription = message_filters.Subscriber(
            self,
            Image,
            self.get_parameter('aligned_depth_topic').value,
            qos_profile=sensor_qos,
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [
                self._rgb_subscription,
                self._detections_subscription,
                self._depth_subscription,
            ],
            queue_size=int(self.get_parameter('sync_queue_size').value),
            slop=float(self.get_parameter('sync_slop_sec').value),
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._on_synchronized)

        self._observations_publisher = self.create_publisher(
            ButtonObservation3DArray,
            self.get_parameter('output_observations_topic').value,
            10,
        )
        self._debug_image_publisher = self.create_publisher(
            Image,
            self.get_parameter('output_debug_image_topic').value,
            image_qos,
        )
        self._semantic_image_publisher = self.create_publisher(
            Image,
            self.get_parameter('semantic_output_topic').value,
            image_qos,
        )
        self._structured_semantics_publisher = self.create_publisher(
            ButtonSemanticArray,
            self.get_parameter('structured_semantics_topic').value,
            10,
        )
        self._qwen_worker_service = self.create_service(
            SetBool,
            '/elevator/qwen_worker_enabled',
            self._set_qwen_worker_enabled,
        )
        self._semantic_tracks = SemanticTrackCache(
            cache_ttl_sec=self.get_parameter('qwen_cache_ttl_sec').value,
            hard_cache_ttl_sec=self.get_parameter(
                'qwen_cache_hard_ttl_sec'
            ).value,
            retry_cooldown_sec=self.get_parameter(
                'qwen_retry_cooldown_sec'
            ).value,
            track_ttl_sec=self.get_parameter('qwen_track_ttl_sec').value,
            association_iou=self.get_parameter('qwen_association_iou').value,
            bbox_retry_iou=self.get_parameter('qwen_bbox_retry_iou').value,
            maximum_consecutive_failures=self.get_parameter(
                'qwen_max_consecutive_failures'
            ).value,
        )
        self._display_registry = StableDisplayRegistry(
            stale_timeout_sec=self.get_parameter(
                'semantic_display_stale_timeout_sec'
            ).value,
        )
        self._semantic_enabled = bool(
            self.get_parameter('semantic_enabled').value
        )
        self._semantic_mode = str(
            self.get_parameter('semantic_mode').value
        )
        self._semantic_worker_requested = self._semantic_enabled
        if self._semantic_enabled:
            self._create_semantic_worker()
        self.get_logger().info(
            'RGB-D button localizer active rgb={} detections={} depth={} '
            'camera_info={} frame={} slop_sec={} max_delta_ms={} '
            'semantic_enabled={} semantic_mode={} semantic_topic={} '
            'qwen_queue_max=1'.format(
                self.get_parameter('rgb_topic').value,
                self.get_parameter('detections_topic').value,
                self.get_parameter('aligned_depth_topic').value,
                self.get_parameter('camera_info_topic').value,
                self._expected_frame,
                self.get_parameter('sync_slop_sec').value,
                self._maximum_sync_delta_ms,
                self._semantic_enabled,
                self._semantic_mode,
                self.get_parameter('semantic_output_topic').value,
            )
        )

    def _create_semantic_worker(self):
        if self._semantic_worker is not None:
            raise RuntimeError('refusing to create a second Qwen worker')
        self._semantic_worker_generation += 1
        mode = self._semantic_mode
        is_hall = mode == 'hall'
        worker_kwargs = dict(
            model_path=(
                self.get_parameter('hall_model_path').value if is_hall
                else self.get_parameter('qwen_model_path').value
            ),
            result_callback=self._on_qwen_result,
            dropped_callback=self._semantic_tracks.release_dropped,
            event_callback=self._on_qwen_event,
            timeout_sec=self.get_parameter('qwen_timeout_sec').value,
            max_new_tokens=self.get_parameter('qwen_max_new_tokens').value,
            prediction_log_path=self.get_parameter(
                'qwen_prediction_log_path').value,
            semantic_mode=mode,
        )
        if is_hall:
            worker_kwargs['model_loader'] = _build_hall_4b_loader(
                self.get_parameter('hall_model_path').value)
            worker_kwargs['infer_function'] = _build_hall_infer_function(
                self.get_parameter('hall_reference_up').value,
                self.get_parameter('hall_reference_down').value,
                self.get_parameter('qwen_max_new_tokens').value)
            worker_kwargs['hall_padding_ratio'] = (
                self.get_parameter('hall_padding_ratio').value)
            worker_kwargs['hall_upscale'] = (
                self.get_parameter('hall_upscale').value)
        self._semantic_worker = QwenSemanticWorker(**worker_kwargs)
        self.get_logger().info(
            'qwen_worker_created generation={} queue_task_type=frame_batch '
            'mode={}'.format(
                self._semantic_worker_generation, mode,
            )
        )

    def _release_semantic_worker(self):
        worker = self._semantic_worker
        if worker is None:
            return True
        stopped = worker.shutdown(join_timeout_sec=2.0)
        self.get_logger().info(
            'qwen_worker_release stopped={} state={} model_load_count={} '
            'cuda_verified={}'.format(
                stopped,
                worker.state,
                worker.model_load_count,
                worker.cuda_verified,
            )
        )
        if not stopped:
            return False
        self._semantic_worker = None
        del worker
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            self.get_logger().warning(
                'qwen_worker_release_cuda_cleanup_failed error={}'.format(exc)
            )
        return True

    def _ensure_semantic_worker(self, now):
        if not self._semantic_enabled or not self._semantic_worker_requested:
            return
        worker = self._semantic_worker
        if (
            worker is not None
            and worker.alive
            and not worker.load_error
        ):
            return
        maximum_restarts = int(
            self.get_parameter('qwen_worker_max_restarts').value
        )
        cooldown = float(
            self.get_parameter('qwen_worker_recovery_cooldown_sec').value
        )
        if now - self._semantic_worker_last_restart_attempt < cooldown:
            return
        if self._semantic_worker_restart_count >= maximum_restarts:
            self._semantic_worker_last_restart_attempt = now
            self.get_logger().error(
                'qwen_worker_recovery_required action=restart_localizer '
                'worker_state={} restart_count={} max_restarts={}'.format(
                    worker.state if worker is not None else 'MISSING',
                    self._semantic_worker_restart_count,
                    maximum_restarts,
                )
            )
            return
        self._semantic_worker_last_restart_attempt = now
        if not self._release_semantic_worker():
            self.get_logger().error(
                'qwen_worker_recovery_required action=restart_localizer '
                'reason=old_worker_did_not_stop'
            )
            return
        self._semantic_worker_restart_count += 1
        self.get_logger().warning(
            'qwen_worker_recovery_start restart_count={}'.format(
                self._semantic_worker_restart_count,
            )
        )
        self._create_semantic_worker()

    def _on_qwen_event(self, level, message):
        if str(message).startswith('qwen_queue_replace '):
            self._semantic_dropped_tasks += 1
        logger_method = getattr(self.get_logger(), str(level), None)
        if logger_method is None:
            logger_method = self.get_logger().info
        logger_method(message)

    def _on_qwen_result(self, result):
        outcome = self._semantic_tracks.apply_result(result)
        if outcome['reason'] == 'track_missing':
            self.get_logger().warning(
                'qwen_result_discarded track_id={} detection_id={} '
                'reason=track_missing'.format(
                    result.track_id,
                    result.detection_id,
                )
            )
            return
        received = monotonic()
        if not outcome['retained']:
            with self._semantic_result_lock:
                self._semantic_results[result.track_id] = SemanticResultView(
                    detection_id=result.detection_id,
                    label=result.label,
                    latency_ms=float(result.latency_sec) * 1000.0,
                    error=result.error,
                    received_monotonic=received,
                )
        with self._semantic_result_lock:
            self._semantic_last_latency_ms = float(result.latency_sec) * 1000.0
        if result.error:
            self.get_logger().warning(
                'qwen_result detection_id={} track_id={} label={} '
                'latency_sec={:.3f} error={} retained_cached_label={} '
                'consecutive_failures={}'.format(
                    result.detection_id,
                    result.track_id,
                    result.label,
                    result.latency_sec,
                    result.error,
                    outcome['retained'],
                    outcome['consecutive_failures'],
                )
            )
        else:
            self.get_logger().info(
                'qwen_result detection_id={} track_id={} label={} '
                'latency_sec={:.3f}'.format(
                    result.detection_id,
                    result.track_id,
                    result.label,
                    result.latency_sec,
                )
            )

    def shutdown_semantic_worker(self):
        if self._semantic_worker is None:
            return True
        return self._release_semantic_worker()

    def _set_qwen_worker_enabled(self, request, response):
        if request.data:
            if (
                self._semantic_worker_requested
                and self._semantic_worker is not None
                and self._semantic_worker.alive
                and not self._semantic_worker.load_error
            ):
                response.success = True
                response.message = 'Qwen worker is already enabled'
            else:
                if not self._semantic_enabled:
                    response.success = False
                    response.message = 'semantic_enabled parameter is false'
                    return response
                self._semantic_worker_requested = True
                if not self._release_semantic_worker():
                    response.success = False
                    response.message = (
                        'Old Qwen worker did not stop; restart localizer required'
                    )
                    return response
                self._create_semantic_worker()
                response.success = True
                response.message = (
                    'Qwen worker recovery started; RGB-D remained active'
                )
            return response

        if not self._semantic_worker_requested and self._semantic_worker is None:
            response.success = True
            response.message = 'Qwen worker is already disabled'
            return response

        self._semantic_worker_requested = False
        stopped = self.shutdown_semantic_worker()
        if stopped:
            response.success = True
            response.message = (
                'Qwen worker stopped; RGB-D localization remains active'
            )
        else:
            response.success = False
            response.message = 'Qwen worker did not stop within timeout'
        return response

    def _on_camera_info(self, message):
        if (
            message.width <= 0
            or message.height <= 0
            or len(message.k) < 9
            or not math.isfinite(float(message.k[0]))
            or not math.isfinite(float(message.k[4]))
            or float(message.k[0]) <= 0.0
            or float(message.k[4]) <= 0.0
        ):
            self.get_logger().warning('rejecting invalid color CameraInfo')
            return
        if message.header.frame_id != self._expected_frame:
            self.get_logger().warning(
                'rejecting CameraInfo frame_id={} expected={}'.format(
                    message.header.frame_id,
                    self._expected_frame,
                )
            )
            return
        self._camera_info = message

    def _camera_info_is_usable(self, rgb_message, depth_message):
        info = self._camera_info
        if info is None:
            return False, 'CameraInfo unavailable'
        if info.header.frame_id != self._expected_frame:
            return False, 'CameraInfo frame mismatch'
        if depth_message.header.frame_id != self._expected_frame:
            return False, 'aligned depth frame mismatch'
        if (int(info.width), int(info.height)) != (
            int(depth_message.width),
            int(depth_message.height),
        ):
            return False, 'CameraInfo/depth dimensions differ'
        age_ns = abs(_stamp_ns(rgb_message) - _stamp_ns(info))
        if age_ns > self._camera_info_max_age_ns:
            return False, 'CameraInfo is stale'
        return True, ''

    def _on_synchronized(self, rgb_message, detections_message, depth_message):
        started = monotonic()
        stamps = [
            _stamp_ns(rgb_message),
            _stamp_ns(detections_message),
            _stamp_ns(depth_message),
        ]
        sync_delta_ms = (max(stamps) - min(stamps)) / 1_000_000.0
        rgb_depth_delta_ms = abs(stamps[0] - stamps[2]) / 1_000_000.0
        if sync_delta_ms > self._maximum_sync_delta_ms:
            self._frames_rejected += 1
            self._semantic_timestamp_mismatch_count += 1
            self.get_logger().warning(
                'rgbd3d_reject reason=sync_delta sync_delta_ms={:.3f} '
                'semantic_cache_action=retain'.format(
                    sync_delta_ms,
                )
            )
            return
        usable, reason = self._camera_info_is_usable(rgb_message, depth_message)
        if not usable:
            self._frames_rejected += 1
            self.get_logger().warning('rgbd3d_reject reason={}'.format(reason))
            return
        if rgb_message.header.frame_id != self._expected_frame:
            self._frames_rejected += 1
            self.get_logger().warning(
                'rgbd3d_reject reason=rgb_frame frame_id={}'.format(
                    rgb_message.header.frame_id
                )
            )
            return

        try:
            rgb_image = self._bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding='rgb8',
            )
            depth_image = self._bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding='passthrough',
            )
        except Exception as exc:
            self._frames_rejected += 1
            self.get_logger().error('rgbd3d_reject reason=cv_bridge error={}'.format(exc))
            return

        info = self._camera_info
        try:
            camera_info_sha256 = _camera_info_sha256(info)
        except (TypeError, ValueError):
            self._frames_rejected += 1
            self.get_logger().warning(
                'rgbd3d_reject reason=invalid_camera_info_provenance'
            )
            return
        camera_info_stamp_ns = _stamp_ns(info)
        detection_width = int(detections_message.image_width)
        detection_height = int(detections_message.image_height)
        if detection_width <= 0 or detection_height <= 0:
            self._frames_rejected += 1
            self.get_logger().warning(
                'rgbd3d_reject reason=invalid_detection_dimensions'
            )
            return
        output = ButtonObservation3DArray()
        output.header.stamp = detections_message.header.stamp
        output.header.frame_id = self._expected_frame
        output.image_width = int(rgb_message.width)
        output.image_height = int(rgb_message.height)
        output.rgb_depth_delta_ms = float(rgb_depth_delta_ms)
        annotated = np.array(rgb_image, copy=True)
        used_label_regions = []
        semantic_items = []
        valid_count = 0
        invalid_count = 0
        valid_ratios = []
        annotation_scale = max(
            1.0,
            min(rgb_message.width / 640.0, rgb_message.height / 480.0),
        )

        for detection in detections_message.detections:
            center_u, center_v = scale_pixel_between_images(
                detection.center_x,
                detection.center_y,
                detection_width,
                detection_height,
                depth_message.width,
                depth_message.height,
            )
            bbox_min_u, bbox_min_v = scale_pixel_between_images(
                max(0.0, min(float(detection_width - 1), detection.x_min)),
                max(0.0, min(float(detection_height - 1), detection.y_min)),
                detection_width,
                detection_height,
                depth_message.width,
                depth_message.height,
            )
            bbox_max_u, bbox_max_v = scale_pixel_between_images(
                max(0.0, min(float(detection_width - 1), detection.x_max)),
                max(0.0, min(float(detection_height - 1), detection.y_max)),
                detection_width,
                detection_height,
                depth_message.width,
                depth_message.height,
            )
            display_center_x, display_center_y = scale_pixel_between_images(
                detection.center_x,
                detection.center_y,
                detection_width,
                detection_height,
                rgb_message.width,
                rgb_message.height,
            )
            display_min_x, display_min_y = scale_pixel_between_images(
                max(0.0, min(float(detection_width - 1), detection.x_min)),
                max(0.0, min(float(detection_height - 1), detection.y_min)),
                detection_width,
                detection_height,
                rgb_message.width,
                rgb_message.height,
            )
            display_max_x, display_max_y = scale_pixel_between_images(
                max(0.0, min(float(detection_width - 1), detection.x_max)),
                max(0.0, min(float(detection_height - 1), detection.y_max)),
                detection_width,
                detection_height,
                rgb_message.width,
                rgb_message.height,
            )
            sample = robust_depth_from_bbox(
                depth_image,
                (bbox_min_u, bbox_min_v, bbox_max_u, bbox_max_v),
                (center_u, center_v),
                depth_message.encoding,
                min_depth_m=float(self.get_parameter('min_depth_m').value),
                max_depth_m=float(self.get_parameter('max_depth_m').value),
                minimum_valid_pixels=int(
                    self.get_parameter('minimum_valid_pixels').value
                ),
                center_roi_ratio=float(
                    self.get_parameter('center_roi_ratio').value
                ),
                fallback_window_size=int(
                    self.get_parameter('fallback_window_size').value
                ),
            )
            geometry = None
            if bool(self.get_parameter('surface_geometry_enabled').value):
                geometry = estimate_surface_geometry_from_bbox(
                    depth_image,
                    (bbox_min_u, bbox_min_v, bbox_max_u, bbox_max_v),
                    (center_u, center_v),
                    depth_message.encoding,
                    info.k[0],
                    info.k[4],
                    info.k[2],
                    info.k[5],
                    min_depth_m=float(
                        self.get_parameter('min_depth_m').value
                    ),
                    max_depth_m=float(
                        self.get_parameter('max_depth_m').value
                    ),
                    minimum_surface_valid_pixels=int(
                        self.get_parameter('minimum_valid_pixels').value
                    ),
                    center_roi_ratio=float(
                        self.get_parameter('center_roi_ratio').value
                    ),
                    fallback_window_size=int(
                        self.get_parameter('fallback_window_size').value
                    ),
                    annulus_expansion_ratio=float(
                        self.get_parameter(
                            'plane_annulus_expansion_ratio'
                        ).value
                    ),
                    annulus_inner_ratio=float(
                        self.get_parameter(
                            'plane_annulus_inner_ratio'
                        ).value
                    ),
                    maximum_plane_depth_delta_m=float(
                        self.get_parameter(
                            'maximum_plane_depth_delta_m'
                        ).value
                    ),
                    minimum_plane_points=int(
                        self.get_parameter('minimum_plane_points').value
                    ),
                    maximum_plane_points=int(
                        self.get_parameter('maximum_plane_points').value
                    ),
                    minimum_plane_inlier_ratio=float(
                        self.get_parameter(
                            'minimum_plane_inlier_ratio'
                        ).value
                    ),
                    maximum_plane_rms_residual_m=float(
                        self.get_parameter(
                            'maximum_plane_rms_residual_m'
                        ).value
                    ),
                    minimum_plane_span_m=float(
                        self.get_parameter('minimum_plane_span_m').value
                    ),
                    mad_rejection_multiplier=float(
                        self.get_parameter(
                            'plane_mad_rejection_multiplier'
                        ).value
                    ),
                    pixel_sigma_px=float(
                        self.get_parameter('surface_pixel_sigma_px').value
                    ),
                    depth_noise_floor_m=float(
                        self.get_parameter(
                            'surface_depth_noise_floor_m'
                        ).value
                    ),
                    maximum_normal_uncertainty_rad=float(
                        self.get_parameter(
                            'maximum_normal_uncertainty_rad'
                        ).value
                    ),
                )

            observation = ButtonObservation3D()
            observation.detection_id = detection.detection_id
            observation.x_min = float(display_min_x)
            observation.y_min = float(display_min_y)
            observation.x_max = float(display_max_x)
            observation.y_max = float(display_max_y)
            observation.center_x = float(display_center_x)
            observation.center_y = float(display_center_y)
            observation.detector_confidence = detection.detector_confidence
            observation.depth_valid = sample.depth_valid
            observation.depth_m = float(sample.depth_m)
            observation.depth_valid_ratio = float(sample.valid_ratio)
            observation.coarse_type = detection.coarse_type
            surface = observation.surface_geometry
            registration_status = str(
                self.get_parameter('rgb_depth_registration_status').value
            )
            optical_axis_status = str(
                self.get_parameter('camera_optical_axis_status').value
            )
            depth_scale_status = str(
                self.get_parameter('depth_scale_status').value
            )
            surface.geometry_method_id = str(
                self.get_parameter('surface_geometry_method_id').value
            )
            surface.geometry_scope = str(
                self.get_parameter('surface_geometry_scope').value
            )
            surface.depth_sample_source = sample.source
            surface.camera_info_stamp_ns = int(camera_info_stamp_ns)
            surface.camera_info_sha256 = camera_info_sha256
            surface.source_frame_id = self._expected_frame
            surface.canonical_frame_id = str(
                self.get_parameter('canonical_frame_id').value
            )
            surface.raw_to_canonical_alias_id = str(
                self.get_parameter('raw_to_canonical_alias_id').value
            )
            surface.raw_to_canonical_alias_status = str(
                self.get_parameter('raw_to_canonical_alias_status').value
            )
            surface.raw_to_canonical_adapter_sha256 = str(
                self.get_parameter(
                    'raw_to_canonical_adapter_sha256'
                ).value
            )
            surface.section_b_calibration_id = str(
                self.get_parameter('section_b_calibration_id').value
            )
            surface.camera_optical_axis_status = optical_axis_status
            surface.rgb_depth_registration_id = str(
                self.get_parameter('rgb_depth_registration_id').value
            )
            surface.rgb_depth_registration_status = registration_status
            surface.depth_scale_m_per_unit = (
                1.0
                if str(depth_message.encoding).strip().upper() == '32FC1'
                else float(
                    self.get_parameter('depth_scale_m_per_unit').value
                )
            )
            surface.depth_scale_status = depth_scale_status
            surface.provenance_complete = surface_provenance_complete(
                geometry_valid=(
                    geometry is not None and geometry.geometry_valid
                ),
                camera_info_sha256=camera_info_sha256,
                canonical_frame_id=surface.canonical_frame_id,
                raw_to_canonical_alias_id=(
                    surface.raw_to_canonical_alias_id
                ),
                raw_to_canonical_alias_status=(
                    surface.raw_to_canonical_alias_status
                ),
                raw_to_canonical_adapter_sha256=(
                    surface.raw_to_canonical_adapter_sha256
                ),
                section_b_calibration_id=(
                    surface.section_b_calibration_id
                ),
                camera_optical_axis_status=optical_axis_status,
                rgb_depth_registration_status=registration_status,
                depth_scale_status=depth_scale_status,
            )
            if geometry is None:
                surface.estimate_valid = False
                surface.failure_reason = 'surface geometry disabled'
            else:
                surface.estimate_valid = bool(geometry.geometry_valid)
                surface.failure_reason = geometry.failure_reason
                surface.surface_point_x_m = float(
                    geometry.surface_point_m[0]
                )
                surface.surface_point_y_m = float(
                    geometry.surface_point_m[1]
                )
                surface.surface_point_z_m = float(
                    geometry.surface_point_m[2]
                )
                surface.surface_point_covariance_m2 = list(
                    geometry.surface_point_covariance_m2
                )
                surface.panel_outward_normal_x = float(
                    geometry.panel_outward_normal[0]
                )
                surface.panel_outward_normal_y = float(
                    geometry.panel_outward_normal[1]
                )
                surface.panel_outward_normal_z = float(
                    geometry.panel_outward_normal[2]
                )
                surface.normal_angular_uncertainty_rad = float(
                    geometry.normal_angular_uncertainty_rad
                )
                surface.plane_rms_residual_m = float(
                    geometry.plane_rms_residual_m
                )
                surface.plane_sample_count = int(
                    geometry.plane_sample_count
                )
                surface.plane_inlier_count = int(
                    geometry.plane_inlier_count
                )
                surface.plane_inlier_ratio = float(
                    geometry.plane_inlier_ratio
                )
                surface.depth_sample_source = geometry.depth_source
            valid_ratios.append(sample.valid_ratio)

            if sample.depth_valid:
                x_m, y_m, z_m = project_pixel_to_camera(
                    center_u,
                    center_v,
                    sample.depth_m,
                    info.k[0],
                    info.k[4],
                    info.k[2],
                    info.k[5],
                )
                observation.x_m = float(x_m)
                observation.y_m = float(y_m)
                observation.z_m = float(z_m)
                valid_count += 1
                color = (0, 255, 0)
                lines = [
                    '{} {:.2f}'.format(
                        observation.detection_id,
                        observation.detector_confidence,
                    ),
                    'X:{:+.3f} Y:{:+.3f} Z:{:.3f} m'.format(x_m, y_m, z_m),
                    (
                        'N:{:+.2f},{:+.2f},{:+.2f} DIAG'.format(
                            surface.panel_outward_normal_x,
                            surface.panel_outward_normal_y,
                            surface.panel_outward_normal_z,
                        )
                        if surface.estimate_valid
                        else 'SURFACE GEOMETRY INVALID'
                    ),
                ]
            else:
                observation.x_m = math.nan
                observation.y_m = math.nan
                observation.z_m = math.nan
                invalid_count += 1
                color = (255, 0, 0)
                lines = [
                    '{} {:.2f}'.format(
                        observation.detection_id,
                        observation.detector_confidence,
                    ),
                    'DEPTH INVALID',
                ]
            output.observations.append(observation)

            left = max(0, min(int(rgb_message.width - 1), int(round(display_min_x))))
            top = max(0, min(int(rgb_message.height - 1), int(round(display_min_y))))
            right = max(0, min(int(rgb_message.width - 1), int(round(display_max_x))))
            bottom = max(0, min(int(rgb_message.height - 1), int(round(display_max_y))))
            cv2.rectangle(
                annotated,
                (left, top),
                (right, bottom),
                color,
                max(2, int(round(2 * annotation_scale))),
            )
            cv2.circle(
                annotated,
                (int(round(display_center_x)), int(round(display_center_y))),
                max(4, int(round(4 * annotation_scale))),
                color,
                -1,
            )
            _draw_text_block(
                annotated,
                lines,
                left,
                top,
                bottom,
                color,
                used_label_regions,
            )
            semantic_items.append({
                'detection_id': observation.detection_id,
                'bbox': (
                    float(display_min_x),
                    float(display_min_y),
                    float(display_max_x),
                    float(display_max_y),
                ),
                'left': left,
                'top': top,
                'right': right,
                'bottom': bottom,
                'center_x': int(round(display_center_x)),
                'center_y': int(round(display_center_y)),
                'color': color,
                'detector_confidence': observation.detector_confidence,
                'depth_valid': observation.depth_valid,
                'x_m': observation.x_m,
                'y_m': observation.y_m,
                'z_m': observation.z_m,
            })

        observed_track_ids = self._semantic_tracks.observe_frame([
            (item['detection_id'], item['bbox']) for item in semantic_items
        ])
        expired_track_ids = self._semantic_tracks.consume_expired_track_ids()
        if expired_track_ids:
            with self._semantic_result_lock:
                for expired_track_id in expired_track_ids:
                    self._semantic_results.pop(expired_track_id, None)
                    self._semantic_display_log_state.pop(
                        expired_track_id,
                        None,
                    )
                    self._semantic_previous_display.pop(
                        expired_track_id,
                        None,
                    )
            self.get_logger().info(
                'semantic_tracks_removed reason=target_disappeared tracks={}'.format(
                    ','.join(expired_track_ids),
                )
            )
        for item, track_id in zip(semantic_items, observed_track_ids):
            item['track_id'] = track_id
            try:
                illumination = red_local_contrast_v1(
                    rgb_image,
                    item['bbox'],
                )
                item['illumination'] = illumination
                item['lit_state'] = {
                    'lit': True,
                    'unlit': False,
                    'uncertain': None,
                }.get(illumination['state'])
            except Exception as exc:
                item['illumination'] = {
                    'state': 'unavailable',
                    'error': '{}: {}'.format(type(exc).__name__, exc),
                }
                item['lit_state'] = None

        callback_now = monotonic()
        self._ensure_semantic_worker(callback_now)
        worker = self._semantic_worker
        if (
            self._semantic_worker_requested
            and worker is not None
            and worker.alive
            and not worker.load_error
            and worker.can_accept_waiting()
        ):
            claimed = self._semantic_tracks.claim_batch(observed_track_ids)
            if claimed:
                batch_tasks = []
                for track_id, detection_id, bbox in claimed:
                    try:
                        if self._semantic_mode == 'hall':
                            crop = prepare_crop_hall(
                                rgb_image, bbox,
                                padding_ratio=float(
                                    self.get_parameter('hall_padding_ratio').value),
                                upscale=float(
                                    self.get_parameter('hall_upscale').value))
                        else:
                            crop = prepare_crop(
                                rgb_image,
                                bbox,
                                padding_ratio=float(
                                    self.get_parameter('qwen_padding_ratio').value
                                ),
                                minimum_size=int(
                                    self.get_parameter(
                                        'qwen_minimum_crop_size'
                                    ).value
                                ),
                            )
                        batch_tasks.append(SemanticTask(
                            track_id=track_id,
                            detection_id=detection_id,
                            bbox=bbox,
                            crop=crop,
                            submitted_monotonic=monotonic(),
                        ))
                    except Exception as exc:
                        self._semantic_crop_invalid_count += 1
                        self._on_qwen_result(SemanticResult(
                            track_id=track_id,
                            detection_id=detection_id,
                            label=UNKNOWN,
                            raw_response='',
                            latency_sec=0.0,
                            error='crop_failed: {}'.format(exc),
                        ))
                        self.get_logger().warning(
                            'qwen_crop_failed detection_id={} track_id={} '
                            'error={} scope=single_button'.format(
                                detection_id,
                                track_id,
                                exc,
                            )
                        )
                if batch_tasks:
                    self._semantic_batch_sequence += 1
                    batch_id = 'frame_{:08d}'.format(
                        self._semantic_batch_sequence,
                    )
                    worker.submit_latest(SemanticBatchTask(
                        batch_id=batch_id,
                        frame_stamp_ns=stamps[0],
                        visible_track_ids=tuple(observed_track_ids),
                        tasks=tuple(batch_tasks),
                        submitted_monotonic=monotonic(),
                    ))
                    self.get_logger().info(
                        'qwen_batch_submit batch_id={} visible_buttons={} '
                        'inference_items={} frame_stamp_ns={}'.format(
                            batch_id,
                            len(observed_track_ids),
                            len(batch_tasks),
                            stamps[0],
                        )
                    )

        display_now = monotonic()
        track_snapshot = self._semantic_tracks.snapshot()
        active_cache_keys = []
        active_display_entries = []
        structured_semantics = []
        for item in semantic_items:
            cache_key = item['track_id']
            active_cache_keys.append(cache_key)
            cache_label = (
                self._semantic_tracks.display_label(item['track_id'])
                if (
                    self._semantic_enabled
                    and self._semantic_worker_requested
                )
                else UNKNOWN
            )
            track_view = track_snapshot.get(cache_key, {})
            with self._semantic_result_lock:
                result_view = self._semantic_results.get(cache_key)
            worker_available = (
                self._semantic_worker_requested
                and
                self._semantic_worker is not None
                and self._semantic_worker.alive
                and not self._semantic_worker.load_error
            )
            association_valid = (
                bool(track_view)
                and track_view.get('detection_id')
                == item['detection_id']
            )
            semantic_age_sec = (
                max(0.0, display_now - result_view.received_monotonic)
                if result_view is not None
                else math.nan
            )
            semantic, semantic_status, qwen_latency_ms, display_reason = (
                semantic_display_state(
                cache_label=cache_label,
                result=result_view,
                detection_id=item['detection_id'],
                semantic_enabled=(
                    self._semantic_enabled
                    and self._semantic_worker_requested
                ),
                worker_available=worker_available,
                cache_ttl_sec=self.get_parameter('qwen_cache_ttl_sec').value,
                hard_cache_ttl_sec=self.get_parameter(
                    'qwen_cache_hard_ttl_sec'
                ).value,
                fresh_duration_sec=self.get_parameter(
                    'semantic_fresh_duration_sec'
                ).value,
                consecutive_failures=track_view.get(
                    'consecutive_failures',
                    0,
                ),
                maximum_consecutive_failures=self.get_parameter(
                    'qwen_max_consecutive_failures'
                ).value,
                association_valid=association_valid,
                now=display_now,
                return_reason=True,
            ))
            if semantic_status in (FRESH, CACHED):
                self._semantic_cache_hits += 1
            else:
                self._semantic_cache_misses += 1
            entry = PanelEntry(
                display_id='',
                detection_id=item['detection_id'],
                semantic_cache_key=cache_key,
                bbox=item['bbox'],
                detector_confidence=item['detector_confidence'],
                depth_valid=item['depth_valid'],
                x_m=item['x_m'],
                y_m=item['y_m'],
                z_m=item['z_m'],
                semantic=semantic,
                semantic_status=semantic_status,
                qwen_latency_ms=qwen_latency_ms,
                active=True,
                lit_state=item['lit_state'],
            )
            entry = self._display_registry.update(cache_key, entry, display_now)
            active_display_entries.append(entry)
            structured_item = ButtonSemantic()
            structured_item.detection_id = entry.detection_id
            structured_item.label = entry.semantic
            if entry.semantic in ('open', 'close'):
                structured_item.button_type = 'door_control'
            elif entry.semantic == 'alarm':
                structured_item.button_type = 'alarm'
            elif entry.semantic and entry.semantic not in ('none', 'UNKNOWN'):
                structured_item.button_type = 'floor_button'
            else:
                structured_item.button_type = 'unknown'
            structured_item.pressable = bool(
                entry.semantic_status in (FRESH, CACHED)
                and entry.semantic
                and entry.semantic not in ('none', 'UNKNOWN')
            )
            # Qwen returns a discrete label but no calibrated probability.
            # Keep this explicitly unknown instead of inventing confidence.
            structured_item.semantic_confidence = math.nan
            structured_item.raw_response = json.dumps(
                {
                    'schema': 'inline_qwen_semantic_v1',
                    'source_stamp_ns': int(stamps[0]),
                    'track_id': entry.semantic_cache_key,
                    'display_id': entry.display_id,
                    'semantic_status': entry.semantic_status,
                    'semantic_age_sec': semantic_age_sec,
                    'association_valid': association_valid,
                    'qwen_latency_ms': entry.qwen_latency_ms,
                    'detector_confidence': entry.detector_confidence,
                    'depth_valid': entry.depth_valid,
                },
                sort_keys=True,
                separators=(',', ':'),
            )
            structured_semantics.append(structured_item)
            log_signature = (
                entry.display_id,
                entry.detection_id,
                entry.semantic_status,
                display_reason,
                entry.lit_state,
            )
            previous = self._semantic_display_log_state.get(cache_key)
            if previous != log_signature:
                self.get_logger().info(
                    'semantic_display display_id={} detection_id={} '
                    'bbox={:.1f},{:.1f},{:.1f},{:.1f} '
                    'semantic_cache_key={} semantic_status={} reason={} '
                    'lit_state={} '
                    'lit_method={} red_delta_p90={} red_fraction={}'.format(
                        entry.display_id,
                        entry.detection_id,
                        *entry.bbox,
                        entry.semantic_cache_key,
                        entry.semantic_status,
                        display_reason,
                        entry.lit_state,
                        item['illumination'].get('method', 'unavailable'),
                        item['illumination'].get(
                            'features', {}
                        ).get('red_delta_p90', 'unavailable'),
                        item['illumination'].get(
                            'features', {}
                        ).get('red_fraction', 'unavailable'),
                    )
                )
                if (
                    semantic_status in (FRESH, CACHED)
                    and semantic
                    and entry.lit_state is None
                ):
                    self.get_logger().warning(
                        'lit_unavailable detection_id={} '
                        'semantic_cache_key={} reason={}'.format(
                            entry.detection_id,
                            entry.semantic_cache_key,
                            item['illumination'].get(
                                'state',
                                item['illumination'].get(
                                    'error',
                                    'unavailable',
                                ),
                            ),
                        )
                    )
                self._semantic_display_log_state[cache_key] = log_signature

            previous_display = self._semantic_previous_display.get(cache_key)
            previous_semantic = (
                previous_display['semantic']
                if previous_display is not None else ''
            )
            if (
                previous_semantic
                and not semantic
                and semantic_status not in (FRESH, CACHED)
            ):
                self.get_logger().warning(
                    'semantic_unknown_transition detection_id={} '
                    'semantic_cache_key={} previous_label={} '
                    'new_status={} reason={} consecutive_failures={}'.format(
                        item['detection_id'],
                        cache_key,
                        previous_semantic,
                        semantic_status,
                        display_reason,
                        track_view.get('consecutive_failures', 0),
                    )
                )
            self._semantic_previous_display[cache_key] = {
                'semantic': semantic,
                'status': semantic_status,
                'reason': display_reason,
            }

        display_entries = self._display_registry.frame_entries(
            active_cache_keys,
            display_now,
        )
        semantic_elapsed = max(
            1e-6,
            display_now - self._semantic_frame_started,
        )
        worker = self._semantic_worker
        qwen_loaded = bool(
            worker is not None
            and worker.cuda_verified
            and worker.model_load_count == 1
            and not worker.load_error
        )
        queue_length = worker.queue_size if worker is not None else 0
        semantic_canvas, display_diagnostics = render_semantic_display(
            rgb_image,
            display_entries,
            SystemStatus(
                qwen_loaded=qwen_loaded,
                semantic_fps=self._semantic_frames_published / semantic_elapsed,
                last_latency_ms=self._semantic_last_latency_ms,
                cache_hits=self._semantic_cache_hits,
                cache_misses=self._semantic_cache_misses,
                queue_length=queue_length,
                dropped_tasks=self._semantic_dropped_tasks,
                mode_label=self._semantic_mode.upper(),
            ),
        )
        if (
            display_now - self._semantic_display_last_summary
            >= float(self.get_parameter(
                'semantic_display_log_interval_sec'
            ).value)
        ):
            self.get_logger().info(
                'semantic_inline_frame buttons={} output_width={} '
                'green={} orange={} red={} white={} text_overlap={} '
                'text_outside_image={} button_content_occluded={} '
                'unplaced={}'.format(
                    len(active_display_entries),
                    display_diagnostics['output_width'],
                    display_diagnostics['recognized_count'],
                    display_diagnostics['orange_count'],
                    display_diagnostics['unknown_count'],
                    display_diagnostics['processing_count'],
                    display_diagnostics['text_overlap'],
                    display_diagnostics['text_outside_image'],
                    display_diagnostics['button_content_occluded'],
                    display_diagnostics['unplaced_count'],
                )
            )
            track_diagnostics = self._semantic_tracks.diagnostics()
            worker_state = (
                worker.state
                if worker is not None else (
                    'MANUALLY_STOPPED'
                    if not self._semantic_worker_requested else 'MISSING'
                )
            )
            self.get_logger().info(
                'semantic_health worker_alive={} worker_state={} '
                'queue_replaced_count={} current_track_count={} '
                'cache_hit_count={} cache_miss_count={} '
                'cache_expired_count={} association_failure_count={} '
                'crop_invalid_count={} inference_timeout_count={} '
                'parser_failure_count={} last_successful_inference_time={} '
                'timestamp_mismatch_count={}'.format(
                    worker.alive if worker is not None else False,
                    worker_state,
                    self._semantic_dropped_tasks,
                    track_diagnostics['current_track_count'],
                    self._semantic_cache_hits,
                    self._semantic_cache_misses,
                    track_diagnostics['cache_expired_count'],
                    track_diagnostics['association_failure_count'],
                    self._semantic_crop_invalid_count,
                    (
                        worker.inference_timeout_count
                        if worker is not None else 0
                    ),
                    worker.parser_failure_count if worker is not None else 0,
                    (
                        '{:.6f}'.format(
                            worker.last_successful_inference_time
                        )
                        if (
                            worker is not None
                            and worker.last_successful_inference_time > 0.0
                        )
                        else 'NONE'
                    ),
                    self._semantic_timestamp_mismatch_count,
                )
            )
            self._semantic_display_last_summary = display_now

        self._observations_publisher.publish(output)
        semantic_output = ButtonSemanticArray()
        semantic_output.header = rgb_message.header
        semantic_output.semantics = structured_semantics
        semantic_output.vlm_latency_ms = float(self._semantic_last_latency_ms)
        semantic_output.backend = 'qwen3_vl_inline_{}'.format(
            self._semantic_mode,
        )
        self._structured_semantics_publisher.publish(semantic_output)
        debug_message = self._bridge.cv2_to_imgmsg(annotated, encoding='rgb8')
        debug_message.header = rgb_message.header
        self._debug_image_publisher.publish(debug_message)
        semantic_message = self._bridge.cv2_to_imgmsg(
            semantic_canvas,
            encoding='rgb8',
        )
        semantic_message.header = rgb_message.header
        self._semantic_image_publisher.publish(semantic_message)
        self._semantic_frames_published += 1

        self._frames_synchronized += 1
        self._valid_observations += valid_count
        self._invalid_observations += invalid_count
        latency_ms = (monotonic() - started) * 1000.0
        mean_valid_ratio = (
            sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0
        )
        self.get_logger().info(
            'rgbd3d_frame synchronized={} stamp_ns={} sync_delta_ms={:.3f} '
            'rgb_depth_delta_ms={:.3f} observations={} valid={} invalid={} '
            'mean_depth_valid_ratio={:.6f} latency_ms={:.3f}'.format(
                self._frames_synchronized,
                stamps[0],
                sync_delta_ms,
                rgb_depth_delta_ms,
                len(output.observations),
                valid_count,
                invalid_count,
                mean_valid_ratio,
                latency_ms,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = RgbdButtonLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_semantic_worker()
        node.get_logger().info(
            'rgbd3d_summary synchronized={} rejected={} valid={} invalid={} '
            'semantic_frames={}'.format(
                node._frames_synchronized,
                node._frames_rejected,
                node._valid_observations,
                node._invalid_observations,
                node._semantic_frames_published,
            )
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
