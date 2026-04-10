from __future__ import annotations

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from typing import Tuple

from .paths import resolve_readonly_path
from .ros_data import ROSData
from .topic_names import REACHED_GOAL_TOPIC, VELOCITY_LELAN_TOPIC, WAYPOINT_TOPIC
from .utils import clip_angle


with open(resolve_readonly_path("config", "robot.yaml"), "r", encoding="utf-8") as f:
    ROBOT_CONFIG = yaml.safe_load(f)

MAX_V = ROBOT_CONFIG["max_v"]
MAX_W = ROBOT_CONFIG["max_w"]
DT = 1 / ROBOT_CONFIG["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1.0
VEL_TOPIC = "/cmd_vel"


def pd_controller(waypoint: np.ndarray) -> Tuple[float, float]:
    assert len(waypoint) in {2, 4}, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0.0
        w = clip_angle(np.arctan2(hy, hx)) / DT
    elif np.abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = dx / DT
        w = np.arctan(dy / dx) / DT
    v = np.clip(v, 0, MAX_V)
    w = np.clip(w, -MAX_W, MAX_W)
    return float(v), float(w)


class PDControllerLeLanNode(Node):
    def __init__(self) -> None:
        super().__init__("pd_controller_lelan")
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self.reverse_mode = False
        self.v_lelan = 0.0
        self.w_lelan = 0.0

        self.create_subscription(Float32MultiArray, WAYPOINT_TOPIC, self._on_waypoint, 1)
        self.create_subscription(Bool, REACHED_GOAL_TOPIC, self._on_reached_goal, 1)
        self.create_subscription(Twist, VELOCITY_LELAN_TOPIC, self._on_lelan_velocity, 1)
        self.vel_out = self.create_publisher(Twist, VEL_TOPIC, 1)
        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info("Waiting for waypoints and LeLaN velocities...")

    def _on_waypoint(self, waypoint_msg: Float32MultiArray) -> None:
        self.waypoint.set(np.array(waypoint_msg.data, dtype=np.float32))

    def _on_reached_goal(self, reached_goal_msg: Bool) -> None:
        self.reached_goal = reached_goal_msg.data

    def _on_lelan_velocity(self, vel_msg: Twist) -> None:
        self.v_lelan = vel_msg.linear.x
        self.w_lelan = vel_msg.angular.z

    def _tick(self) -> None:
        vel_msg = Twist()
        if self.reached_goal:
            vel_msg.linear.x = self.v_lelan
            vel_msg.angular.z = self.w_lelan
            self.vel_out.publish(vel_msg)
            return
        if self.waypoint.is_valid(verbose=True, log_fn=self.get_logger().warning):
            v, w = pd_controller(self.waypoint.get())
            if self.reverse_mode:
                v *= -1.0
            vel_msg.linear.x = v
            vel_msg.angular.z = w
        self.vel_out.publish(vel_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PDControllerLeLanNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
