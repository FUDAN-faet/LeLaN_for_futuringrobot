from __future__ import annotations

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from .paths import resolve_readonly_path
from .topic_names import JOY_BUMPER_TOPIC


with open(resolve_readonly_path("config", "robot.yaml"), "r", encoding="utf-8") as f:
    ROBOT_CONFIG = yaml.safe_load(f)
with open(resolve_readonly_path("config", "joystick.yaml"), "r", encoding="utf-8") as f:
    JOY_CONFIG = yaml.safe_load(f)

MAX_V = 0.4
MAX_W = 0.8
VEL_TOPIC = ROBOT_CONFIG["vel_teleop_topic"]
DEADMAN_SWITCH = JOY_CONFIG["deadman_switch"]
LIN_VEL_BUTTON = JOY_CONFIG["lin_vel_button"]
ANG_VEL_BUTTON = JOY_CONFIG["ang_vel_button"]
RATE = 9


class JoyTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("joy_teleop")
        self.vel_msg = Twist()
        self.button_pressed = False
        self.bumper_pressed = False

        self.create_subscription(Joy, "joy", self._on_joy, 10)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, 1)
        self.bumper_pub = self.create_publisher(Bool, JOY_BUMPER_TOPIC, 1)
        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info("Waiting for joystick input...")

    def _button(self, buttons, idx: int) -> bool:
        return 0 <= idx < len(buttons) and bool(buttons[idx])

    def _axis(self, axes, idx: int) -> float:
        return float(axes[idx]) if 0 <= idx < len(axes) else 0.0

    def _on_joy(self, data: Joy) -> None:
        self.button_pressed = self._button(data.buttons, DEADMAN_SWITCH)
        self.bumper_pressed = self._button(data.buttons, DEADMAN_SWITCH - 1)
        if self.button_pressed:
            self.vel_msg.linear.x = MAX_V * self._axis(data.axes, LIN_VEL_BUTTON)
            self.vel_msg.angular.z = MAX_W * self._axis(data.axes, ANG_VEL_BUTTON)
        else:
            self.vel_msg = Twist()

    def _tick(self) -> None:
        self.vel_pub.publish(self.vel_msg)
        bumper_msg = Bool()
        bumper_msg.data = self.bumper_pressed
        self.bumper_pub.publish(bumper_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JoyTeleopNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
