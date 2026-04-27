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
            DeclareLaunchArgument("apply_velocity_limits", default_value="false"),
            DeclareLaunchArgument("config_path", default_value=""),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument("server_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("server_port", default_value="8765"),
            DeclareLaunchArgument("inference_backend", default_value="socket"),
            DeclareLaunchArgument("request_timeout", default_value="2.0"),
            DeclareLaunchArgument("jpeg_quality", default_value="90"),
            DeclareLaunchArgument("max_linear_vel", default_value="0.3"),
            DeclareLaunchArgument("max_angular_vel", default_value="0.5"),
            DeclareLaunchArgument("control_mode", default_value="first_step"),
            DeclareLaunchArgument("rollout_steps", default_value="1"),
            DeclareLaunchArgument("replan_on_new_image", default_value="true"),
            DeclareLaunchArgument("adaptive_turn_rollout", default_value="false"),
            DeclareLaunchArgument("turn_replan_threshold", default_value="0.35"),
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
                        "apply_velocity_limits": LaunchConfiguration("apply_velocity_limits"),
                        "config_path": LaunchConfiguration("config_path"),
                        "model_path": LaunchConfiguration("model_path"),
                        "server_host": LaunchConfiguration("server_host"),
                        "server_port": LaunchConfiguration("server_port"),
                        "inference_backend": LaunchConfiguration("inference_backend"),
                        "request_timeout": LaunchConfiguration("request_timeout"),
                        "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                        "max_linear_vel": LaunchConfiguration("max_linear_vel"),
                        "max_angular_vel": LaunchConfiguration("max_angular_vel"),
                        "control_mode": LaunchConfiguration("control_mode"),
                        "rollout_steps": LaunchConfiguration("rollout_steps"),
                        "replan_on_new_image": LaunchConfiguration("replan_on_new_image"),
                        "adaptive_turn_rollout": LaunchConfiguration("adaptive_turn_rollout"),
                        "turn_replan_threshold": LaunchConfiguration("turn_replan_threshold"),
                    }
                ],
            ),
        ]
    )
