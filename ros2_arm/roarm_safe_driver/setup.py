from glob import glob
from setuptools import find_packages, setup

package_name = "roarm_safe_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Lift Robot team",
    maintainer_email="noreply@example.com",
    description="Fail-closed ROS 2 serial driver for supervised RoArm-M3 joint validation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "feedback_monitor = roarm_safe_driver.monitor_node:main",
            "safe_driver = roarm_safe_driver.node:main",
            "calibration_recorder = roarm_safe_driver.calibration_recorder:main",
        ]
    },
)
