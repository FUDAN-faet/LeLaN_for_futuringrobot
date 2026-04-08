from glob import glob
from setuptools import find_packages, setup


package_name = "lelan_ros2"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zme",
    maintainer_email="zme@example.com",
    description="ROS 2 Jazzy deployment wrapper for LeLaN.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lelan_policy_node = lelan_ros2.lelan_policy_node:main",
        ],
    },
)
