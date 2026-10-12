"""Read-only Phase-4 RoArm FK/TF bridge.

This launch description has exactly two nodes: robot_state_publisher consumes
the measured /joint_states already produced by feedback_monitor, and a static
identity adapter supplies roarm_base_link -> base_link.  It starts no serial
transport, joint-state publisher, driver, MoveIt component, or RViz process.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from roarm_safe_driver.tf_bridge import build_phase4_robot_description


def generate_launch_description():
    vendor_share = get_package_share_directory("roarm_description")
    vendor_xacro = os.path.join(
        vendor_share, "urdf", "roarm_m3", "roarm_m3.xacro"
    )
    with open(vendor_xacro, "r", encoding="utf-8") as handle:
        robot_description = build_phase4_robot_description(handle.read())

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="roarm_phase4_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="roarm_vendor_base_identity_adapter",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                "--frame-id", "roarm_base_link",
                "--child-frame-id", "base_link",
            ],
            output="screen",
        ),
    ])
