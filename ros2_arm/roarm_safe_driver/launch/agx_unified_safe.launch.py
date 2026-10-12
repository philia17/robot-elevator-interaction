"""Hardware-active AGX entry point requiring operator and runtime safety gates.

The default profile opens the configured serial device.  Starting this launch
file is a physical-hardware action and never constitutes motion authorization.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("roarm_safe_driver"),
        "config",
        "agx_unified_safe.yaml",
    )
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config),
        DeclareLaunchArgument("auto_torque_recovery_enabled", default_value="false"),
        Node(
            package="roarm_safe_driver",
            executable="safe_driver",
            name="roarm_safe_driver",
            output="screen",
            parameters=[LaunchConfiguration("config"), {
                "auto_torque_recovery_enabled": ParameterValue(
                    LaunchConfiguration("auto_torque_recovery_enabled"), value_type=bool,
                ),
            }],
        ),
    ])
