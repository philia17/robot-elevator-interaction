from setuptools import find_packages, setup


package_name = 'elevator_semantics'


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
    description='Semantic-labeling skeleton node for elevator perception.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_labeler_node = elevator_semantics.node:main',
        ],
    },
)
