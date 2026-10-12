"""Start Gemini 335 plus the native canonical adapter (separate-process fallback)."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    vendor = os.path.join(get_package_share_directory('orbbec_camera'), 'launch', 'gemini_330_series.launch.py')
    config = os.path.join(get_package_share_directory('lift_robot_camera_adapter_cpp'), 'config', 'gemini335_canonical_cpp.yaml')
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(vendor),
        launch_arguments={
            'camera_name': 'gemini335', 'enable_color': 'true',
            'color_width': '1920', 'color_height': '1080', 'color_fps': '30', 'color_format': 'MJPG',
            'enable_depth': 'true', 'depth_width': '1280', 'depth_height': '800',
            'depth_fps': '30', 'depth_format': 'Y16', 'depth_registration': 'true',
            'align_mode': 'SW', 'align_target_stream': 'COLOR', 'enable_point_cloud': 'false',
            'enable_colored_point_cloud': 'false', 'enable_accel': 'false', 'enable_gyro': 'false',
            'publish_tf': 'false',
        }.items())
    adapter = Node(package='lift_robot_camera_adapter_cpp', executable='gemini335_canonical_adapter_cpp',
                   name='gemini335_canonical_adapter', parameters=[config], output='screen')
    return LaunchDescription([camera, adapter])
