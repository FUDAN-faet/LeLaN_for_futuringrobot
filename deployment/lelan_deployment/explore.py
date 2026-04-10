from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from .paths import resolve_from_deployment, resolve_readonly_path
from .topic_names import IMAGE_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC
from .utils import load_model, msg_to_pil, to_numpy, transform_images

from vint_train.training.train_utils import get_action


with open(resolve_readonly_path("config", "robot.yaml"), "r", encoding="utf-8") as f:
    ROBOT_CONFIG = yaml.safe_load(f)

MAX_V = ROBOT_CONFIG["max_v"]
RATE = ROBOT_CONFIG["frame_rate"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExploreNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("exploration")
        self.args = args
        self.context_queue = []

        with open(resolve_readonly_path("config", "models.yaml"), "r", encoding="utf-8") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = resolve_from_deployment(model_paths[args.model]["config_path"])
        with open(model_config_path, "r", encoding="utf-8") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]
        ckpt_path = resolve_from_deployment(model_paths[args.model]["ckpt_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

        self.model = load_model(str(ckpt_path), self.model_params, DEVICE).to(DEVICE)
        self.model.eval()

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info(f"Using device: {DEVICE}")
        self.get_logger().info("Waiting for image observations...")

    def _on_image(self, msg: Image) -> None:
        obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

    def _tick(self) -> None:
        if len(self.context_queue) <= self.context_size:
            return

        obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = obs_images.to(DEVICE)
        fake_goal = torch.randn((1, 3, *self.model_params["image_size"]), device=DEVICE)
        mask = torch.ones(1, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            obs_cond = self.model("vision_encoder", obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

            naction = torch.randn(
                (self.args.num_samples, self.model_params["len_traj_pred"], 2),
                device=DEVICE,
            )
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            start_time = time.time()
            for timestep in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=timestep,
                    global_cond=obs_cond,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=naction,
                ).prev_sample
            self.get_logger().debug(f"Diffusion loop took {time.time() - start_time:.3f}s")

        naction = to_numpy(get_action(naction))

        sampled_actions_msg = Float32MultiArray()
        sampled_actions_msg.data = np.concatenate((np.array([0.0]), naction.flatten())).astype(np.float32).tolist()
        self.sampled_actions_pub.publish(sampled_actions_msg)

        chosen_waypoint = naction[0][self.args.waypoint]
        if self.model_params["normalize"]:
            chosen_waypoint *= MAX_V / RATE

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = np.asarray(chosen_waypoint, dtype=np.float32).tolist()
        self.waypoint_pub.publish(waypoint_msg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NoMaD exploration on the robot")
    parser.add_argument("--model", "-m", default="nomad", type=str, help="model name from config/models.yaml")
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="index of the waypoint used for navigation",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="number of actions sampled from the exploration model",
    )
    return parser


def main(args=None) -> None:
    parsed_args, _ = build_parser().parse_known_args(args)
    rclpy.init(args=args)
    node = ExploreNode(parsed_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
