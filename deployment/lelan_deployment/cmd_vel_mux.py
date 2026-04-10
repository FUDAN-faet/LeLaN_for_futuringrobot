from __future__ import annotations

import time
from dataclasses import dataclass

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .paths import resolve_readonly_path


with open(resolve_readonly_path("config", "cmd_vel_mux.yaml"), "r", encoding="utf-8") as f:
    MUX_CONFIG = yaml.safe_load(f)

PUBLISHER_TOPIC = MUX_CONFIG["publisher"]
SUBSCRIBERS = MUX_CONFIG["subscribers"]
RATE = 20.0


@dataclass
class VelocitySource:
    name: str
    topic: str
    timeout: float
    priority: int
    msg: Twist | None = None
    stamp: float = float("-inf")

    def active(self, now: float) -> bool:
        return self.msg is not None and (now - self.stamp) <= self.timeout


class CmdVelMuxNode(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_mux")
        self.sources = {}
        self.last_selected = None

        for source_cfg in SUBSCRIBERS:
            source = VelocitySource(
                name=source_cfg["name"],
                topic=source_cfg["topic"],
                timeout=float(source_cfg["timeout"]),
                priority=int(source_cfg["priority"]),
            )
            self.sources[source.name] = source
            self.create_subscription(
                Twist,
                source.topic,
                lambda msg, source_name=source.name: self._on_velocity(source_name, msg),
                10,
            )

        self.publisher = self.create_publisher(Twist, PUBLISHER_TOPIC, 10)
        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info(f"Publishing muxed velocity to {PUBLISHER_TOPIC}")

    def _on_velocity(self, source_name: str, msg: Twist) -> None:
        source = self.sources[source_name]
        source.msg = msg
        source.stamp = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        active_sources = [source for source in self.sources.values() if source.active(now)]
        if not active_sources:
            self.publisher.publish(Twist())
            self.last_selected = None
            return

        selected = max(active_sources, key=lambda source: source.priority)
        self.publisher.publish(selected.msg)
        self.last_selected = selected.name


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
