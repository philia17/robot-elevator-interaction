from setuptools import find_packages, setup


package_name = "roarm_post_precontact_rehearsal_executor"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lift Robot team",
    maintainer_email="noreply@example.com",
    description="Disabled-by-default typed 70 mm contact-approach OUT+BACK owner.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "roarm_post_precontact_rehearsal_executor = roarm_post_precontact_rehearsal_executor.node:main",
        ],
    },
)
