"""Read-only feedback launch for operator-controlled Web UI calibration."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("roarm_safe_driver"),
        "config",
        "webui_monitor.yaml",
    )
    return LaunchDescription([
        Node(
            package="roarm_safe_driver",
            executable="feedback_monitor",
            name="roarm_feedback_monitor",
            output="screen",
            parameters=[config],
        ),
    ])
