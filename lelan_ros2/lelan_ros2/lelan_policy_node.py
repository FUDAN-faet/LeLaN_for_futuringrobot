from dataclasses import dataclass
import json
from pathlib import Path
import socket
import struct
import sys
from typing import List, Optional, Tuple

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


@dataclass
class PolicyPrediction:
    linear_vels: List[float]
    angular_vels: List[float]

    def commands(self) -> List[Tuple[float, float]]:
        return list(zip(self.linear_vels, self.angular_vels))


class SocketPolicyBackend:
    def __init__(self, host: str, port: int, prompt: str, timeout: float, jpeg_quality: int) -> None:
        self.host = host
        self.port = port
        self.prompt = prompt
        self.timeout = timeout
        self.jpeg_quality = jpeg_quality

    def predict(self, cv_image: np.ndarray) -> PolicyPrediction:
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

        linear_vels = response.get("linear_vels")
        angular_vels = response.get("angular_vels")
        if linear_vels is None or angular_vels is None:
            linear_vels = [float(response["linear_vel"])]
            angular_vels = [float(response["angular_vel"])]

        if len(linear_vels) != len(angular_vels):
            raise RuntimeError("Inference server returned mismatched control horizon lengths")

        return PolicyPrediction(
            linear_vels=[float(v) for v in linear_vels],
            angular_vels=[float(v) for v in angular_vels],
        )


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

    def predict(self, cv_image: np.ndarray) -> PolicyPrediction:
        linear_vels, angular_vels = self.runtime.predict_sequence_from_cv2(cv_image)
        return PolicyPrediction(
            linear_vels=[float(v) for v in linear_vels],
            angular_vels=[float(v) for v in angular_vels],
        )


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
        self.declare_parameter("apply_velocity_limits", True)
        self.declare_parameter("control_mode", "first_step")
        self.declare_parameter("rollout_steps", 1)
        self.declare_parameter("replan_on_new_image", True)
        self.declare_parameter("adaptive_turn_rollout", False)
        self.declare_parameter("turn_replan_threshold", 0.35)

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
        self.apply_velocity_limits = bool(self.get_parameter("apply_velocity_limits").value)
        self.control_mode = str(self.get_parameter("control_mode").value).strip().lower()
        self.rollout_steps = max(1, int(self.get_parameter("rollout_steps").value))
        self.replan_on_new_image = bool(self.get_parameter("replan_on_new_image").value)
        self.adaptive_turn_rollout = bool(self.get_parameter("adaptive_turn_rollout").value)
        self.turn_replan_threshold = max(
            0.0,
            float(self.get_parameter("turn_replan_threshold").value),
        )

        if self.control_mode not in {"first_step", "rollout"}:
            raise ValueError("control_mode must be 'first_step' or 'rollout'")

        self.bridge = CvBridge()
        self.latest_processed_image: Optional[np.ndarray] = None
        self.last_backend_error: Optional[str] = None
        self.pending_commands: List[Tuple[float, float]] = []
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
        self.get_logger().info(f"apply_velocity_limits={self.apply_velocity_limits}")
        self.get_logger().info(f"control_mode={self.control_mode}")
        self.get_logger().info(f"rollout_steps={self.rollout_steps}")
        self.get_logger().info(f"replan_on_new_image={self.replan_on_new_image}")
        self.get_logger().info(f"adaptive_turn_rollout={self.adaptive_turn_rollout}")
        self.get_logger().info(f"turn_replan_threshold={self.turn_replan_threshold}")
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
        processed_image: Optional[np.ndarray] = None
        if self.latest_processed_image is not None:
            processed_image = self.latest_processed_image
            self.latest_processed_image = None

        if processed_image is None:
            if self.control_mode == "rollout" and self.pending_commands:
                linear_vel, angular_vel = self.pending_commands.pop(0)
                self.cmd_pub.publish(self._limit_velocity(linear_vel, angular_vel))
            return

        if self.control_mode == "rollout" and self.pending_commands and not self.replan_on_new_image:
            linear_vel, angular_vel = self.pending_commands.pop(0)
            self.cmd_pub.publish(self._limit_velocity(linear_vel, angular_vel))
            return

        try:
            prediction = self.backend.predict(processed_image)
            self.last_backend_error = None
        except Exception as exc:
            message = str(exc)
            if message != self.last_backend_error:
                self.get_logger().error(f"Policy inference failed: {message}")
                self.last_backend_error = message
            return

        commands = prediction.commands()
        if not commands:
            return

        if self.control_mode == "rollout":
            rollout_end = self._effective_rollout_end(commands)
            self.pending_commands = commands[1:rollout_end]
            linear_vel, angular_vel = commands[0]
        else:
            linear_vel, angular_vel = commands[0]

        self.get_logger().debug(f"Predicted velocity v={linear_vel:.3f}, w={angular_vel:.3f}")
        self.cmd_pub.publish(self._limit_velocity(linear_vel, angular_vel))

    def _limit_velocity(self, vt: float, wt: float) -> Twist:
        msg = Twist()
        if not self.apply_velocity_limits:
            msg.linear.x = vt
            msg.angular.z = wt
            return msg

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

    def _effective_rollout_end(self, commands: List[Tuple[float, float]]) -> int:
        rollout_end = min(self.rollout_steps, len(commands))
        if rollout_end <= 1 or not self.adaptive_turn_rollout:
            return rollout_end

        _, first_angular = commands[0]
        if abs(first_angular) >= self.turn_replan_threshold:
            self.get_logger().debug(
                "Reducing rollout to 1 step because "
                f"|w0|={abs(first_angular):.3f} >= {self.turn_replan_threshold:.3f}"
            )
            return 1

        return rollout_end


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
