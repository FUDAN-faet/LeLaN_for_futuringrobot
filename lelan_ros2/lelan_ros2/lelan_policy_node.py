import json
from pathlib import Path
import socket
import struct
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_SRC_DIR = REPO_ROOT / "deployment" / "src"

if str(DEPLOYMENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_SRC_DIR))


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def recv_message(sock: socket.socket):
    header_size = struct.unpack("!I", recv_exact(sock, 4))[0]
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
    payload_size = int(header.get("payload_size", 0))
    payload = recv_exact(sock, payload_size) if payload_size else b""
    return header, payload


def send_message(sock: socket.socket, header, payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_size"] = len(payload)
    header_bytes = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


def preprocess_cv_image(cv_image: np.ndarray, use_ricoh: bool) -> np.ndarray:
    if not use_ricoh:
        return cv_image

    xc = 310
    yc = 321
    xyoffset = 280
    x1 = xc - xyoffset
    x2 = xc + xyoffset
    y1 = yc - xyoffset
    y2 = yc + xyoffset

    if x1 < 0 or y1 < 0 or x2 > cv_image.shape[1] or y2 > cv_image.shape[0]:
        raise ValueError(
            "Ricoh crop is outside the incoming image bounds. "
            f"image_shape={cv_image.shape}"
        )

    mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (xc, yc), (xyoffset, xyoffset), 0, 0, 360, 255, -1)
    masked = cv2.bitwise_and(cv_image, cv_image, mask=mask)
    cropped = masked[y1:y2, x1:x2]
    cropped = cv2.transpose(cropped)
    return cv2.flip(cropped, 1)


class SocketPolicyBackend:
    def __init__(self, host: str, port: int, prompt: str, timeout: float, jpeg_quality: int) -> None:
        self.host = host
        self.port = port
        self.prompt = prompt
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

    def predict(self, cv_image: np.ndarray) -> Tuple[float, float]:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode image for inference request")

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            send_message(
                sock,
                {
                    "encoding": "jpeg",
                    "prompt": self.prompt,
                },
                encoded.tobytes(),
            )
            response, _ = recv_message(sock)

        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Inference server returned an unknown error"))

        return float(response["linear_vel"]), float(response["angular_vel"])


class LocalPolicyBackend:
    def __init__(self, config_path: str, model_path: str, prompt: str) -> None:
        try:
            from lelan_runtime import LeLaNInferenceRuntime
        except Exception as exc:  # pragma: no cover - depends on local Python env
            raise RuntimeError(
                "Failed to load local LeLaN runtime. "
                "For ROS 2 Jazzy + conda Python 3.8, use inference_backend:=socket."
            ) from exc

        self.runtime = LeLaNInferenceRuntime(
            config_path=config_path or None,
            model_path=model_path or None,
            prompt=prompt,
        )

    def predict(self, cv_image: np.ndarray) -> Tuple[float, float]:
        return self.runtime.predict_from_cv2(cv_image)


class LeLaNPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("lelan_policy_node")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("prompt", "chair")
        self.declare_parameter("config_path", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("inference_backend", "socket")
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8765)
        self.declare_parameter("request_timeout", 2.0)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("use_ricoh", False)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 0.5)

        self.image_topic = self.get_parameter("image_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.prompt = self.get_parameter("prompt").value
        self.config_path = self.get_parameter("config_path").value
        self.model_path = self.get_parameter("model_path").value
        self.inference_backend = self.get_parameter("inference_backend").value
        self.server_host = self.get_parameter("server_host").value
        self.server_port = int(self.get_parameter("server_port").value)
        self.request_timeout = float(self.get_parameter("request_timeout").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.use_ricoh = bool(self.get_parameter("use_ricoh").value)
        self.timer_period = float(self.get_parameter("timer_period").value)
        self.max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        self.max_angular_vel = float(self.get_parameter("max_angular_vel").value)

        self.bridge = CvBridge()
        self.latest_processed_image: Optional[np.ndarray] = None
        self.last_backend_error: Optional[str] = None
        self.backend = self._create_backend()

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info("LeLaN policy node ready")
        self.get_logger().info(f"inference_backend={self.inference_backend}")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"cmd_vel_topic={self.cmd_vel_topic}")
        self.get_logger().info(f"prompt={self.prompt}")
        if self.inference_backend == "socket":
            self.get_logger().info(f"server={self.server_host}:{self.server_port}")
        else:
            self.get_logger().info(f"config_path={self.config_path or '<auto>'}")
            self.get_logger().info(f"model_path={self.model_path or '<auto>'}")

    def _create_backend(self):
        if self.inference_backend == "socket":
            return SocketPolicyBackend(
                host=self.server_host,
                port=self.server_port,
                prompt=self.prompt,
                timeout=self.request_timeout,
                jpeg_quality=self.jpeg_quality,
            )
        if self.inference_backend == "local":
            return LocalPolicyBackend(
                config_path=self.config_path,
                model_path=self.model_path,
                prompt=self.prompt,
            )
        raise ValueError(f"Unsupported inference_backend: {self.inference_backend}")

    def image_callback(self, msg: Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.latest_processed_image = preprocess_cv_image(cv_image, self.use_ricoh)

    def timer_callback(self) -> None:
        if self.latest_processed_image is None:
            return

        processed_image = self.latest_processed_image
        self.latest_processed_image = None

        try:
            linear_vel, angular_vel = self.backend.predict(processed_image)
            self.last_backend_error = None
        except Exception as exc:
            message = str(exc)
            if message != self.last_backend_error:
                self.get_logger().error(f"Policy inference failed: {message}")
                self.last_backend_error = message
            return

        self.get_logger().debug(f"Predicted velocity v={linear_vel:.3f}, w={angular_vel:.3f}")
        self.cmd_pub.publish(self._limit_velocity(linear_vel, angular_vel))

    def _limit_velocity(self, vt: float, wt: float) -> Twist:
        msg = Twist()
        maxv = self.max_linear_vel
        maxw = self.max_angular_vel

        if abs(vt) <= maxv:
            if abs(wt) <= maxw:
                msg.linear.x = vt
                msg.angular.z = wt
            else:
                rd = vt / wt
                msg.linear.x = maxw * np.sign(vt) * abs(rd)
                msg.angular.z = maxw * np.sign(wt)
        else:
            if abs(wt) <= 1e-3:
                msg.linear.x = maxv * np.sign(vt)
                msg.angular.z = 0.0
            else:
                rd = vt / wt
                if abs(rd) >= maxv / maxw:
                    msg.linear.x = maxv * np.sign(vt)
                    msg.angular.z = maxv * np.sign(wt) / abs(rd)
                else:
                    msg.linear.x = maxw * np.sign(vt) * abs(rd)
                    msg.angular.z = maxw * np.sign(wt)

        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeLaNPolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
