"""Launch the inert pre-contact admission adapter."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_path = str(
        Path(get_package_share_directory("roarm_precontact_adapter"))
        / "config"
        / "precontact_adapter.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="roarm_precontact_adapter",
                executable="roarm_precontact_adapter",
                name="roarm_precontact_adapter",
                output="screen",
                parameters=[config_path],
            )
        ]
    )
