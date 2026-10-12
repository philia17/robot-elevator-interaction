"""Run the complete ROS pipeline from a configurable JPEG or PNG file."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share_directory = get_package_share_directory('elevator_bringup')
    mock_launch = os.path.join(
        share_directory,
        'launch',
        'mock_pipeline.launch.py',
    )
    sample_image = os.path.join(
        share_directory,
        'test_data',
        'sample_elevator_panel.jpg',
    )
    image_path = LaunchConfiguration('image_path')
    return LaunchDescription([
        DeclareLaunchArgument('image_path', default_value=sample_image),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mock_launch),
            launch_arguments={'image_path': image_path}.items(),
        ),
    ])
