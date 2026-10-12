from setuptools import find_packages, setup


package_name = 'elevator_vision'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elevator Perception Maintainer',
    maintainer_email='maintainer@example.com',
    description='Vision-stage skeleton nodes for elevator perception.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frame_gate_node = elevator_vision.nodes:frame_gate_main',
            'button_detector_node = elevator_vision.nodes:button_detector_main',
            'rgbd_button_localizer_node = elevator_vision.nodes:rgbd_button_localizer_main',
            'm0_target_observer_node = elevator_vision.nodes:m0_target_observer_main',
        ],
    },
)
