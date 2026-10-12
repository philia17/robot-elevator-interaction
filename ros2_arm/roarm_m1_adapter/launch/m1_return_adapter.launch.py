"""Launch the M1 return adapter — inert until explicitly authorized."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="roarm_m1_adapter",
                executable="m1_return_adapter",
                name="roarm_m1_return_adapter",
                output="screen",
                parameters=[],
            )
        ]
    )
