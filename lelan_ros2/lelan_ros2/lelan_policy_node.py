from pathlib import Path
import sys
from typing import Optional

import clip
import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from PIL import ImageDraw

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "train"
DEPLOYMENT_SRC_DIR = REPO_ROOT / "deployment" / "src"

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(DEPLOYMENT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_SRC_DIR))

from utils import load_model, pil2cv, cv2pil, transform_images_lelan  # noqa: E402


class LeLaNPolicyNode(Node):
    """ROS 2 Jazzy node for LeLaN last-mile navigation."""

    def __init__(self) -> None:
        super().__init__("lelan_policy_node")

        default_config_path = str(REPO_ROOT / "train" / "config" / "lelan.yaml")
        default_model_path = str(REPO_ROOT / "deployment" / "model_weights" / "wo_col_loss_wo_temp.pth")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("prompt", "chair")
        self.declare_parameter("config_path", default_config_path)
        self.declare_parameter("model_path", default_model_path)
        self.declare_parameter("use_ricoh", False)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 0.5)

        self.image_topic = self.get_parameter("image_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.prompt = self.get_parameter("prompt").value
        raw_config_path = self.get_parameter("config_path").value
        raw_model_path = self.get_parameter("model_path").value
        self.config_path = raw_config_path or default_config_path
        self.model_path = raw_model_path or default_model_path
        self.use_ricoh = self.get_parameter("use_ricoh").value
        self.timer_period = float(self.get_parameter("timer_period").value)
        self.max_linear_vel = float(self.get_parameter("max_linear_vel").value)
        self.max_angular_vel = float(self.get_parameter("max_angular_vel").value)

        self.bridge = CvBridge()
        self.latest_processed_image: Optional[PILImage.Image] = None
        self.image_hist = []
        self.history_len = 10

        self.xc = 310
        self.yc = 321
        self.xyoffset = 280
        self.mask_xy = [
            (self.xc - self.xyoffset, self.yc - self.xyoffset),
            (self.xc + self.xyoffset, self.yc + self.xyoffset),
        ]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_params = self._load_model_params(self.config_path)
        self.model = self._load_policy_model(self.model_path)
        self.feat_text = self._encode_prompt(self.prompt)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.get_logger().info(f"LeLaN policy node ready on {self.device}")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"cmd_vel_topic={self.cmd_vel_topic}")
        self.get_logger().info(f"prompt={self.prompt}")
        self.get_logger().info(f"config_path={self.config_path}")
        self.get_logger().info(f"model_path={self.model_path}")

    def _load_model_params(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_policy_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        model = load_model(str(path), self.model_params, self.device)
        model = model.to(self.device)
        model.eval()
        return model

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        with torch.no_grad():
            tokenized = clip.tokenize(prompt).to(self.device)
            return self.model("text_encoder", inst_ref=tokenized)

    def image_callback(self, msg: Image) -> None:
        self.latest_processed_image = self._preprocess_image(msg)

    def timer_callback(self) -> None:
        if self.latest_processed_image is None:
            return

        processed_image = self.latest_processed_image
        self.latest_processed_image = None
        twist = self._run_policy(processed_image)
        self.cmd_pub.publish(twist)

    def _preprocess_image(self, msg: Image) -> PILImage.Image:
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if not self.use_ricoh:
            return cv2pil(cv_image)

        pil_img = cv2pil(cv_image)
        fg_img = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 255))
        draw = ImageDraw.Draw(fg_img)
        draw.ellipse(self.mask_xy, fill=(0, 0, 0, 0))
        pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
        cv2_img = pil2cv(pil_img)
        cv_cutimg = cv2_img[
            self.yc - self.xyoffset:self.yc + self.xyoffset,
            self.xc - self.xyoffset:self.xc + self.xyoffset,
        ]
        cv_cutimg = cv2.transpose(cv_cutimg)
        cv_cutimg = cv2.flip(cv_cutimg, 1)
        return cv2pil(cv_cutimg)

    def _run_policy(self, image: PILImage.Image) -> Twist:
        if not self.image_hist:
            self.image_hist = [image for _ in range(self.history_len)]

        hist_image = self.image_hist[self.history_len - 1]
        im_obs = [hist_image, image]

        obs_images, obs_current = transform_images_lelan(
            im_obs,
            self.model_params["image_size"],
            center_crop=False,
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        batch_obs_images = obs_images.to(self.device)
        batch_obs_current = obs_current.to(self.device)

        with torch.no_grad():
            if self.model_params["model_type"] == "lelan_col":
                obsgoal_cond = self.model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    feat_text=self.feat_text.to(torch.float32),
                    current_img=batch_obs_current,
                )
            elif self.model_params["model_type"] == "lelan":
                obsgoal_cond = self.model(
                    "vision_encoder",
                    obs_img=batch_obs_current,
                    feat_text=self.feat_text.to(torch.float32),
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_params['model_type']}")

            linear_vel, angular_vel = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)

        vt = float(linear_vel.cpu().numpy()[0, 0])
        wt = float(angular_vel.cpu().numpy()[0, 0])
        self.get_logger().debug(f"Predicted velocity v={vt:.3f}, w={wt:.3f}")

        self.image_hist = [image] + self.image_hist[: self.history_len - 1]
        return self._limit_velocity(vt, wt)

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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
