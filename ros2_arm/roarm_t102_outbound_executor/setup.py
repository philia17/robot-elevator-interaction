from glob import glob
from setuptools import find_packages, setup


package_name = "roarm_t102_outbound_executor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lift Robot team",
    maintainer_email="noreply@example.com",
    description="Inert-by-default UI owner for a future outbound-only T102 route.",
    license="MIT",
    entry_points={"console_scripts": [
        "roarm_t102_outbound_executor = roarm_t102_outbound_executor.node:main",
        "t102_supervised_oneshot = roarm_t102_outbound_executor.oneshot_client:main",
    ]},
)
