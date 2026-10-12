from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("artifact_path", default_value=""),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("execution_enabled", default_value="false"),
        Node(
            package="roarm_press_sequencer",
            executable="press_sequencer",
            name="roarm_press_sequencer",
            output="screen",
            parameters=[{
                "artifact_path": LaunchConfiguration("artifact_path"),
                "dry_run": LaunchConfiguration("dry_run"),
                "execution_enabled": LaunchConfiguration("execution_enabled"),
            }],
        ),
    ])
