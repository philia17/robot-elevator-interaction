import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'lift_robot_camera_adapter'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lift Robot Maintainer',
    maintainer_email='maintainer@example.com',
    description='Fail-closed Gemini 335 canonical camera adapter.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'apriltag_axis_validator = lift_robot_camera_adapter.apriltag_axis_validator:main',
            'apriltag_pose_sampler = lift_robot_camera_adapter.apriltag_pose_sampler:main',
            'phase2_optical_axis_probe = lift_robot_camera_adapter.optical_axis_probe:main',
            'phase2_camera_session = lift_robot_camera_adapter.phase2_session:main',
        ],
    },
)
