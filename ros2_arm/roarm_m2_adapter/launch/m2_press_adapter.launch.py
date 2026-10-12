"""Launch the M2 press adapter — inert until explicitly authorized."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="roarm_m2_adapter",
                executable="m2_press_adapter",
                name="roarm_m2_press_adapter",
                output="screen",
                parameters=[],
            )
        ]
    )
