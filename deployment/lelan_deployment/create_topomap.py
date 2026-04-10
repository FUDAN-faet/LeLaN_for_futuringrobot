from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy

from .paths import resolve_writable_dir
from .topic_names import IMAGE_TOPIC
from .utils import msg_to_pil


def remove_files_in_dir(dir_path: Path) -> None:
    for path in dir_path.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


class CreateTopomapNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("create_topomap")
        self.dt = args.dt
        self.obs_img: PILImage.Image | None = None
        self.last_image_time: float | None = None
        self.image_index = 0
        self.topomap_dir = resolve_writable_dir("topomaps", "images", args.dir)

        if any(self.topomap_dir.iterdir()):
            self.get_logger().warning(f"{self.topomap_dir} already exists. Removing previous images...")
            remove_files_in_dir(self.topomap_dir)

        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.create_subscription(Joy, "joy", self._on_joy, 10)
        self.create_timer(self.dt, self._tick)

        self.get_logger().info(f"Saving topomap images to {self.topomap_dir}")

    def _on_image(self, msg: Image) -> None:
        self.obs_img = msg_to_pil(msg)
        self.last_image_time = time.monotonic()

    def _on_joy(self, msg: Joy) -> None:
        if msg.buttons and msg.buttons[0]:
            self.get_logger().info("Shutdown requested from joystick")
            if rclpy.ok():
                rclpy.shutdown()

    def _tick(self) -> None:
        if self.obs_img is not None:
            save_path = self.topomap_dir / f"{self.image_index}.png"
            self.obs_img.save(save_path)
            self.get_logger().info(f"Saved image {self.image_index}")
            self.image_index += 1
            self.obs_img = None
            return

        if self.last_image_time is not None and (time.monotonic() - self.last_image_time) > (2 * self.dt):
            self.get_logger().warning(f"Topic {IMAGE_TOPIC} stopped publishing. Shutting down...")
            if rclpy.ok():
                rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Generate topomaps from the {IMAGE_TOPIC} topic")
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="directory name inside deployment/topomaps/images",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help=f"time between images sampled from the {IMAGE_TOPIC} topic",
    )
    return parser


def main(args=None) -> None:
    argv = args if args is not None else None
    parsed_args, _ = build_parser().parse_known_args(argv)
    if parsed_args.dt <= 0:
        raise ValueError("dt must be positive")

    rclpy.init(args=args)
    node = CreateTopomapNode(parsed_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
