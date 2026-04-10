from __future__ import annotations

import argparse
from distutils.util import strtobool

import clip
import cv2
import numpy as np
import rclpy
import torch
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from PIL import ImageDraw
from rclpy.node import Node
from sensor_msgs.msg import Image

from .paths import resolve_from_deployment
from .utils import clamp_velocity, cv2pil, load_model, pil2cv, transform_images_lelan


XC = 310
YC = 321
XYOFFSET = 280
MASK_BOX = [(XC - XYOFFSET, YC - XYOFFSET), (XC + XYOFFSET, YC + XYOFFSET)]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _bool_arg(value: str) -> bool:
    return bool(strtobool(value))


class LeLanPolicyColNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("lelan_col")
        self.args = args
        self.bridge = CvBridge()
        self.latest_image: PILImage.Image | None = None
        self.image_hist: list[PILImage.Image] = []
        self.feat_text = None

        model_config_path = resolve_from_deployment(args.config)
        with open(model_config_path, "r", encoding="utf-8") as f:
            self.model_params = yaml.safe_load(f)

        ckpt_path = resolve_from_deployment(args.model)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

        self.model = load_model(str(ckpt_path), self.model_params, DEVICE).to(DEVICE)
        self.model.eval()

        self.pub_vel = self.create_publisher(Twist, args.cmd_vel_topic, 1)
        self.pub_processed = self.create_publisher(Image, args.processed_topic, 1)
        self.create_subscription(Image, args.camera_topic, self._preprocess_camera, 1)
        self.create_timer(args.timer_period, self._tick)

        self.get_logger().info(f"Using device: {DEVICE}")
        self.get_logger().info(f"Listening to camera topic {args.camera_topic}")

    def _preprocess_camera(self, msg: Image) -> None:
        if self.args.ricoh:
            cv2_msg_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_img = cv2pil(cv2_msg_img)
            fg_img = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 255))
            draw = ImageDraw.Draw(fg_img)
            draw.ellipse(MASK_BOX, fill=(0, 0, 0, 0))
            pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
            cv2_img = pil2cv(pil_img)
            cv_cutimg = cv2_img[YC - XYOFFSET : YC + XYOFFSET, XC - XYOFFSET : XC + XYOFFSET]
            cv_cutimg = cv2.transpose(cv_cutimg)
            cv_cutimg = cv2.flip(cv_cutimg, 1)
        else:
            cv_cutimg = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        self.latest_image = cv2pil(cv_cutimg)
        msg_img = self.bridge.cv2_to_imgmsg(cv_cutimg, "bgr8")
        msg_img.header = msg.header
        self.pub_processed.publish(msg_img)

    def _ensure_history(self, current_image: PILImage.Image) -> None:
        if not self.image_hist:
            self.image_hist = [current_image for _ in range(10)]

    def _encode_text_once(self) -> None:
        if self.feat_text is not None:
            return
        batch_obj_inst = clip.tokenize(self.args.prompt).to(DEVICE)
        with torch.no_grad():
            self.feat_text = self.model("text_encoder", inst_ref=batch_obj_inst)

    def _tick(self) -> None:
        if self.latest_image is None:
            return

        current_image = self.latest_image
        self.latest_image = None
        self._ensure_history(current_image)
        self._encode_text_once()

        im_obs = [self.image_hist[9], current_image]
        obs_images, obs_current = transform_images_lelan(im_obs, self.model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(DEVICE)
        obs_current = obs_current.to(DEVICE)

        with torch.no_grad():
            if self.model_params["model_type"] == "lelan_col":
                obsgoal_cond = self.model(
                    "vision_encoder",
                    obs_img=obs_images,
                    feat_text=self.feat_text.to(torch.float32),
                    current_img=obs_current,
                )
            elif self.model_params["model_type"] == "lelan":
                obsgoal_cond = self.model(
                    "vision_encoder",
                    obs_img=obs_current,
                    feat_text=self.feat_text.to(torch.float32),
                )
            else:
                raise ValueError(f"Unsupported LeLaN model_type: {self.model_params['model_type']}")
            linear_vel, angular_vel = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)

        vt = float(linear_vel.cpu().numpy()[0, 0])
        wt = float(angular_vel.cpu().numpy()[0, 0])
        self.pub_vel.publish(clamp_velocity(vt, wt))
        self.image_hist = [current_image] + self.image_hist[:9]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LeLaN final-approach policy")
    parser.add_argument("-p", "--prompt", required=True, type=str, help="target object prompt")
    parser.add_argument(
        "-c",
        "--config",
        default="../../train/config/lelan.yaml",
        type=str,
        help="path to the model config file relative to deployment/",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="../model_weights/wo_col_loss_wo_temp.pth",
        type=str,
        help="path to the model checkpoint relative to deployment/",
    )
    parser.add_argument(
        "-r",
        "--ricoh",
        default=True,
        type=_bool_arg,
        help="whether the input camera is a Ricoh Theta S spherical camera",
    )
    parser.add_argument("--camera-topic", default="/cv_camera_node/image_raw", type=str, help="input camera topic")
    parser.add_argument("--processed-topic", default="/image_processed", type=str, help="debug processed image topic")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel", type=str, help="output velocity topic")
    parser.add_argument("--timer-period", default=0.1, type=float, help="policy update period in seconds")
    return parser


def main(args=None) -> None:
    parsed_args, _ = build_parser().parse_known_args(args)
    rclpy.init(args=args)
    node = LeLanPolicyColNode(parsed_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
