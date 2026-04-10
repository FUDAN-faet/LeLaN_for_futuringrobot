from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("lelan_deployment")

    camera_config = LaunchConfiguration("camera_config")
    joy_config = LaunchConfiguration("joy_config")
    enable_camera = LaunchConfiguration("enable_camera")
    enable_joy = LaunchConfiguration("enable_joy")
    enable_cmd_vel_mux = LaunchConfiguration("enable_cmd_vel_mux")
    camera_namespace = LaunchConfiguration("camera_namespace")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_config",
                default_value=PathJoinSubstitution([package_share, "config", "ros2", "camera_front.yaml"]),
            ),
            DeclareLaunchArgument(
                "joy_config",
                default_value=PathJoinSubstitution([package_share, "config", "ros2", "joy_driver.yaml"]),
            ),
            DeclareLaunchArgument("enable_camera", default_value="true"),
            DeclareLaunchArgument("enable_joy", default_value="true"),
            DeclareLaunchArgument("enable_cmd_vel_mux", default_value="true"),
            DeclareLaunchArgument("camera_namespace", default_value="usb_cam"),
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                namespace=camera_namespace,
                name="usb_cam",
                output="screen",
                parameters=[camera_config],
                condition=IfCondition(enable_camera),
            ),
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                output="screen",
                parameters=[joy_config],
                condition=IfCondition(enable_joy),
            ),
            Node(
                package="lelan_deployment",
                executable="cmd_vel_mux",
                name="cmd_vel_mux",
                output="screen",
                condition=IfCondition(enable_cmd_vel_mux),
            ),
        ]
    )
