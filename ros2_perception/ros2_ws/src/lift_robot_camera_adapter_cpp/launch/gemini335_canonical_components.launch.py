"""Run the Orbbec driver and canonical adapter as intra-process components."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    adapter_config = os.path.join(
        get_package_share_directory('lift_robot_camera_adapter_cpp'),
        'config', 'gemini335_canonical_cpp.yaml')
    common_extra = [{'use_intra_process_comms': True}]
    camera_parameters = {
        'camera_name': 'gemini335',
        'enable_color': True,
        'color_width': 1920, 'color_height': 1080, 'color_fps': 30, 'color_format': 'MJPG',
        'enable_depth': True,
        'depth_width': 1280, 'depth_height': 800, 'depth_fps': 30, 'depth_format': 'Y16',
        'depth_registration': True, 'align_mode': 'SW', 'align_target_stream': 'COLOR',
        # Gemini 330 publishes incomplete FrameSets when aggregation is left at
        # ANY.  In SW D2C mode those depth-only sets bypass alignment and leak
        # native 1280x800 frames onto the nominally aligned 1920x1080 topic.
        # Require synchronized, complete RGB-D sets before software alignment.
        'enable_frame_sync': True, 'frame_aggregate_mode': 'full_frame',
        # Match Orbbec's low-CPU Gemini 330 profile.  Running both software and
        # hardware noise removal needlessly competes with 1080p SW alignment.
        'enable_noise_removal_filter': False,
        'enable_hardware_noise_removal_filter': True,
        'enable_point_cloud': False, 'enable_colored_point_cloud': False,
        'enable_accel': False, 'enable_gyro': False, 'publish_tf': False,
    }
    container = ComposableNodeContainer(
        name='gemini335_canonical_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[
            ComposableNode(
                package='orbbec_camera', plugin='orbbec_camera::OBCameraNodeDriver',
                namespace='gemini335', name='gemini335', parameters=[camera_parameters],
                extra_arguments=common_extra),
            ComposableNode(
                package='lift_robot_camera_adapter_cpp',
                plugin='lift_robot_camera_adapter::Gemini335CanonicalAdapter',
                name='gemini335_canonical_adapter', parameters=[adapter_config],
                extra_arguments=common_extra),
        ])
    return LaunchDescription([container])
