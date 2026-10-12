import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'elevator_bringup'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (
            os.path.join('share', package_name, 'test_data'),
            glob('../../../test_data/*.png')
            + glob('../../../test_data/*.jpg')
            + glob('../../../test_data/*.jpeg'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elevator Perception Maintainer',
    maintainer_email='maintainer@example.com',
    description='Launch and configuration skeletons for elevator perception.',
    license='Apache-2.0',
    tests_require=['pytest'],
)
