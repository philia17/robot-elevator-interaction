from setuptools import find_packages, setup


package_name = 'elevator_pipeline'


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
    description='Pipeline orchestration skeleton nodes for elevator perception.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'observation_fusion_node = elevator_pipeline.nodes:observation_fusion_main',
            'target_resolver_node = elevator_pipeline.nodes:target_resolver_main',
            'temporal_tracker_node = elevator_pipeline.nodes:temporal_tracker_main',
            'perception_manager_node = elevator_pipeline.nodes:perception_manager_main',
            'visualization_node = elevator_pipeline.nodes:visualization_main',
            'diagnostics_node = elevator_pipeline.nodes:diagnostics_main',
        ],
    },
)
