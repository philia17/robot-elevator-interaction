"""Run Gemini 335, V6 detection, camera-frame XYZ, and Qwen semantics.

The Gemini 335 driver must already be running. This launch deliberately does
not start target resolution, robot-frame transforms, or robot control.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share_directory = get_package_share_directory('elevator_bringup')
    config_directory = os.path.join(share_directory, 'config')
    pipeline_config = os.path.join(config_directory, 'pipeline.yaml')
    detector_config = os.path.join(config_directory, 'detector.yaml')
    rgbd_config = os.path.join(config_directory, 'rgbd_3d.yaml')

    semantic_mode = LaunchConfiguration('semantic_mode', default='cabin')

    return LaunchDescription([
        DeclareLaunchArgument(
            'semantic_mode',
            default_value='cabin',
            description='Semantic recognition mode: cabin or hall',
        ),
        Node(
            package='elevator_vision',
            executable='frame_gate_node',
            name='frame_gate_node',
            # Keep the detector branch light, but preserve Gemini 335's 16:9
            # geometry instead of stretching it to the legacy 640x480 shape.
            parameters=[
                pipeline_config,
                {
                    'output_width': 640,
                    'output_height': 360,
                    'metric_width': 640,
                    'metric_height': 360,
                    # This launch powers a live preview, so a dark or briefly
                    # blurred scene must still reach the detector.  The
                    # generic frame gate's quality thresholds are useful for
                    # dataset selection, but they otherwise make the live
                    # pipeline appear dead (no overlay is ever published).
                    'min_sharpness': 0.0,
                    'min_brightness': 0.0,
                    'max_brightness': 255.0,
                    'max_motion': 1.0e9,
                    'full_resolution_frame_topic': (
                        '/elevator/selected_frame_full'
                    ),
                },
            ],
            output='screen',
        ),
        Node(
            package='elevator_vision',
            executable='button_detector_node',
            name='button_detector_node',
            parameters=[detector_config],
            output='screen',
        ),
        Node(
            package='elevator_vision',
            executable='rgbd_button_localizer_node',
            name='rgbd_button_localizer_node',
            # Use the accepted native-resolution companion frame. It has the
            # detector frame's timestamp without flooding the synchronizer at
            # the camera's full 30 FPS rate.
            parameters=[
                rgbd_config,
                {
                    'rgb_topic': '/elevator/selected_frame_full',
                    # The desktop launch is the complete observation display:
                    # detector geometry/XYZ remain authoritative while Qwen
                    # contributes display-only semantics.
                    'semantic_enabled': True,
                    'semantic_mode': semantic_mode,
                },
            ],
            output='screen',
        ),
        Node(
            package='elevator_vision',
            executable='m0_target_observer_node',
            name='m0_target_observer_node',
            parameters=[{
                'target_label': '1',
                'required_consecutive_frames': 5,
                # Match the Qwen worker's 30 s soft cache TTL. Geometry and
                # track association remain live on every RGB-D frame; the
                # semantic label is refreshed by the worker at this boundary.
                'maximum_semantic_age_sec': 30.0,
                'maximum_non_target_semantic_age_sec': 30.0,
                # The detector is normally fast, but the live Qwen cache
                # refresh produced measured 4.0-7.3 s paired-message gaps.
                # Keep five confirmations and every spatial/quality gate;
                # only retain their window across that bounded refresh gap.
                'maximum_frame_gap_sec': 8.0,
                'minimum_detector_confidence': 0.7,
                'minimum_depth_valid_ratio': 0.5,
                'minimum_bbox_iou': 0.65,
                'maximum_position_spread_m': 0.015,
                # A target semantic refresh is never itself counted or
                # READY. Retain the already checked spatial window for at
                # most 10 s, then require a complete fresh target semantic
                # before READY can be published.
                'target_semantic_grace_ttl_sec': 10.0,
                # Button 1 remains subject to the exact-label, fresh
                # semantic, RGB-D, normal and five-frame gates.  A transient
                # non-target detector track is diagnostic only, so it cannot
                # repeatedly erase a valid Button-1 observation window.
                'require_complete_semantic_set': False,
            }],
            output='screen',
        ),
    ])
