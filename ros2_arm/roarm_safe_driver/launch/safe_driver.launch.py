from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("roarm_safe_driver"),
        "config",
        "safe_defaults.yaml",
    )
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config),
        Node(
            package="roarm_safe_driver",
            executable="safe_driver",
            name="roarm_safe_driver",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
    ])
