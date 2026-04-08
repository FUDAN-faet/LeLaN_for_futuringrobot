from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("prompt", default_value="chair"),
            DeclareLaunchArgument("use_ricoh", default_value="false"),
            DeclareLaunchArgument("timer_period", default_value="0.1"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("model_path", default_value=""),
            Node(
                package="lelan_ros2",
                executable="lelan_policy_node",
                name="lelan_policy_node",
                output="screen",
                parameters=[
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "prompt": LaunchConfiguration("prompt"),
                        "use_ricoh": LaunchConfiguration("use_ricoh"),
                        "timer_period": LaunchConfiguration("timer_period"),
                        "config_path": LaunchConfiguration("config_path"),
                        "model_path": LaunchConfiguration("model_path"),
                    }
                ],
            ),
        ]
    )
