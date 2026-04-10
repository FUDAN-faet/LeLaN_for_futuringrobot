from glob import glob
from pathlib import Path

from setuptools import setup


package_name = "lelan_deployment"
deployment_root = Path(__file__).parent


data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (f"share/{package_name}", ["package.xml"]),
    (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    (f"share/{package_name}/config", glob("config/*.yaml")),
    (f"share/{package_name}/config/ros2", glob("config/ros2/*.yaml")),
]

topomap_files = glob("topomaps/images/*")
if topomap_files:
    data_files.append((f"share/{package_name}/topomaps/images", topomap_files))


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="support@example.com",
    description="ROS 2 Jazzy deployment package for learning-language-navigation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cmd_vel_mux = lelan_deployment.cmd_vel_mux:main",
            "create_topomap = lelan_deployment.create_topomap:main",
            "explore = lelan_deployment.explore:main",
            "joy_teleop = lelan_deployment.joy_teleop:main",
            "lelan_policy_col = lelan_deployment.lelan_policy_col:main",
            "navigate = lelan_deployment.navigate:main",
            "navigate_lelan = lelan_deployment.navigate_lelan:main",
            "pd_controller = lelan_deployment.pd_controller:main",
            "pd_controller_lelan = lelan_deployment.pd_controller_lelan:main",
        ],
    },
)
