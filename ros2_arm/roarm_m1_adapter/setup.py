from glob import glob
from setuptools import find_packages, setup

package_name = "roarm_m1_adapter"

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
    install_requires=["setuptools", "PyYAML"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Lift Robot team",
    maintainer_email="noreply@example.com",
    description="Inert-by-default M1 no-contact return adapter for RoArm-M3.",
    license="MIT",
    entry_points={"console_scripts": ["m1_return_adapter = roarm_m1_adapter.node:main"]},
)
