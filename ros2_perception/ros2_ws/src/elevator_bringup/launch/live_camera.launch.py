"""Run all nine nodes against an already-running configurable RGB topic."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share_directory = get_package_share_directory('elevator_bringup')
    config_directory = os.path.join(share_directory, 'config')
    pipeline_config = os.path.join(config_directory, 'pipeline.yaml')
    detector_config = os.path.join(config_directory, 'detector.yaml')
    vlm_config = os.path.join(config_directory, 'vlm.yaml')
    default_floor_map = os.path.join(
        config_directory,
        'building_floor_map.yaml',
    )

    image_topic = LaunchConfiguration('image_topic')
    reliability = LaunchConfiguration('qos_reliability')
    durability = LaunchConfiguration('qos_durability')
    queue_depth = ParameterValue(LaunchConfiguration('queue_depth'), value_type=int)
    floor_map_path = LaunchConfiguration('floor_map_path')
    detector_backend = LaunchConfiguration('detector_backend')
    detector_model_path = LaunchConfiguration('detector_model_path')
    detector_confidence = ParameterValue(
        LaunchConfiguration('detector_confidence'),
        value_type=float,
    )
    detector_device = ParameterValue(
        LaunchConfiguration('detector_device'),
        value_type=int,
    )
    detector_imgsz = ParameterValue(
        LaunchConfiguration('detector_imgsz'),
        value_type=int,
    )
    vlm_backend = LaunchConfiguration('vlm_backend')
    confirmations = ParameterValue(
        LaunchConfiguration('required_confirmations'),
        value_type=int,
    )
    action_timeout = ParameterValue(
        LaunchConfiguration('action_timeout_sec'),
        value_type=float,
    )
    debug_enabled = ParameterValue(
        LaunchConfiguration('debug_image_enable'),
        value_type=bool,
    )
    diagnostics_enabled = ParameterValue(
        LaunchConfiguration('diagnostics_enable'),
        value_type=bool,
    )

    nodes = [
        Node(
            package='elevator_vision',
            executable='frame_gate_node',
            parameters=[pipeline_config, {
                'source_type': 'topic',
                'image_topic': image_topic,
                'qos_reliability': reliability,
                'qos_durability': durability,
                'queue_depth': queue_depth,
            }],
        ),
        Node(
            package='elevator_vision',
            executable='button_detector_node',
            parameters=[detector_config, {
                'backend': detector_backend,
                'model_path': detector_model_path,
                'confidence_threshold': detector_confidence,
                'device': detector_device,
                'imgsz': detector_imgsz,
            }],
        ),
        Node(
            package='elevator_semantics',
            executable='vlm_labeler_node',
            parameters=[vlm_config, {'backend': vlm_backend}],
        ),
        Node(
            package='elevator_pipeline',
            executable='observation_fusion_node',
            parameters=[pipeline_config],
        ),
        Node(
            package='elevator_pipeline',
            executable='target_resolver_node',
            parameters=[pipeline_config, {'floor_map_path': floor_map_path}],
        ),
        Node(
            package='elevator_pipeline',
            executable='temporal_tracker_node',
            parameters=[pipeline_config, {
                'required_confirmations': confirmations,
            }],
        ),
        Node(
            package='elevator_pipeline',
            executable='perception_manager_node',
            parameters=[pipeline_config, {
                'floor_map_path': floor_map_path,
                'maximum_action_timeout_sec': action_timeout,
            }],
        ),
        Node(
            package='elevator_pipeline',
            executable='visualization_node',
            parameters=[pipeline_config, {'enabled': debug_enabled}],
        ),
        Node(
            package='elevator_pipeline',
            executable='diagnostics_node',
            parameters=[pipeline_config, {'enabled': diagnostics_enabled}],
        ),
    ]
    arguments = [
        DeclareLaunchArgument(
            'image_topic',
            default_value='/gemini335/color/image_raw',
        ),
        DeclareLaunchArgument('qos_reliability', default_value='best_effort'),
        DeclareLaunchArgument('qos_durability', default_value='volatile'),
        DeclareLaunchArgument('queue_depth', default_value='10'),
        DeclareLaunchArgument('floor_map_path', default_value=default_floor_map),
        DeclareLaunchArgument('detector_backend', default_value='yolo'),
        DeclareLaunchArgument(
            'detector_model_path',
            default_value='/home/robot/models/v6/model/best.pt',
        ),
        DeclareLaunchArgument('detector_confidence', default_value='0.30'),
        DeclareLaunchArgument('detector_device', default_value='0'),
        DeclareLaunchArgument('detector_imgsz', default_value='640'),
        DeclareLaunchArgument('vlm_backend', default_value='mock'),
        DeclareLaunchArgument('required_confirmations', default_value='2'),
        DeclareLaunchArgument('action_timeout_sec', default_value='120.0'),
        DeclareLaunchArgument('debug_image_enable', default_value='true'),
        DeclareLaunchArgument('diagnostics_enable', default_value='true'),
    ]
    return LaunchDescription(arguments + nodes)
