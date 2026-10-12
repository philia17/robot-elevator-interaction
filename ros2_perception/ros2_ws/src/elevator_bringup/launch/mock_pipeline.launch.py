"""Start all nine nodes with deterministic Mock backends and file input."""

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
    sample_image = os.path.join(
        share_directory,
        'test_data',
        'sample_elevator_panel.jpg',
    )
    pipeline_config = os.path.join(config_directory, 'pipeline.yaml')
    detector_config = os.path.join(config_directory, 'detector.yaml')
    vlm_config = os.path.join(config_directory, 'vlm.yaml')
    default_floor_map = os.path.join(
        config_directory,
        'building_floor_map.yaml',
    )

    image_path = LaunchConfiguration('image_path')
    floor_map_path = LaunchConfiguration('floor_map_path')
    detector_backend = LaunchConfiguration('detector_backend')
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
                'source_type': 'file',
                'file_path': image_path,
            }],
        ),
        Node(
            package='elevator_vision',
            executable='button_detector_node',
            parameters=[detector_config, {'backend': detector_backend}],
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
        DeclareLaunchArgument('image_path', default_value=sample_image),
        DeclareLaunchArgument('floor_map_path', default_value=default_floor_map),
        DeclareLaunchArgument('detector_backend', default_value='mock'),
        DeclareLaunchArgument('vlm_backend', default_value='mock'),
        DeclareLaunchArgument('required_confirmations', default_value='2'),
        DeclareLaunchArgument('action_timeout_sec', default_value='120.0'),
        DeclareLaunchArgument('debug_image_enable', default_value='true'),
        DeclareLaunchArgument('diagnostics_enable', default_value='true'),
    ]
    return LaunchDescription(arguments + nodes)
