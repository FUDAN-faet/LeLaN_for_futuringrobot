from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

from .paths import resolve_from_deployment, resolve_readonly_path
from .topic_names import IMAGE_TOPIC, REACHED_GOAL_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC
from .utils import load_model, msg_to_pil, to_numpy, transform_images

from vint_train.training.train_utils import get_action


with open(resolve_readonly_path("config", "robot.yaml"), "r", encoding="utf-8") as f:
    ROBOT_CONFIG = yaml.safe_load(f)

MAX_V = ROBOT_CONFIG["max_v"]
RATE = ROBOT_CONFIG["frame_rate"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_topomap_dir(path_or_name: str) -> Path:
    path = Path(path_or_name).expanduser()
    if path.is_absolute():
        return path
    candidate = resolve_from_deployment(str(Path("topomaps") / "images" / path_or_name))
    if candidate.exists():
        return candidate
    return (Path.cwd() / path_or_name).resolve()


class NavigateNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("navigate")
        self.args = args
        self.context_queue = []
        self.closest_node = 0
        self.reached_goal = False

        with open(resolve_readonly_path("config", "models.yaml"), "r", encoding="utf-8") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = resolve_from_deployment(model_paths[args.model]["config_path"])
        with open(model_config_path, "r", encoding="utf-8") as f:
            self.model_params = yaml.safe_load(f)

        ckpt_path = resolve_from_deployment(model_paths[args.model]["ckpt_path"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

        self.model = load_model(str(ckpt_path), self.model_params, DEVICE).to(DEVICE)
        self.model.eval()

        topomap_dir = resolve_topomap_dir(args.dir)
        if not topomap_dir.is_dir():
            raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")

        topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda name: int(name.split(".")[0]))
        self.topomap = [PILImage.open(topomap_dir / filename) for filename in topomap_filenames]

        if not self.topomap:
            raise ValueError(f"No topomap images found in {topomap_dir}")

        if args.goal_node == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            if not (-1 <= args.goal_node < len(self.topomap)):
                raise ValueError("Invalid goal index")
            self.goal_node = args.goal_node

        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )
        else:
            self.num_diffusion_iters = None
            self.noise_scheduler = None

        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info(f"Using device: {DEVICE}")
        self.get_logger().info(f"Loaded {len(self.topomap)} topomap nodes from {topomap_dir}")

    def _on_image(self, msg: Image) -> None:
        obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.model_params["context_size"] + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

    def _publish_goal(self) -> None:
        goal_msg = Bool()
        goal_msg.data = self.reached_goal
        self.goal_pub.publish(goal_msg)

    def _tick(self) -> None:
        chosen_waypoint = np.zeros(4, dtype=np.float32)

        if len(self.context_queue) > self.model_params["context_size"]:
            if self.model_params["model_type"] == "nomad":
                chosen_waypoint = self._tick_nomad()
            else:
                chosen_waypoint = self._tick_goal_conditioned()

        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.astype(np.float32).tolist()
        self.waypoint_pub.publish(waypoint_msg)

        self.reached_goal = self.closest_node == self.goal_node
        self._publish_goal()

    def _window(self) -> tuple[int, int]:
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        return start, end

    def _tick_nomad(self) -> np.ndarray:
        obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(DEVICE)
        mask = torch.zeros(1, dtype=torch.long, device=DEVICE)

        start, end = self._window()
        goal_image = [
            transform_images(goal_img, self.model_params["image_size"], center_crop=False).to(DEVICE)
            for goal_img in self.topomap[start : end + 1]
        ]
        goal_image = torch.concat(goal_image, dim=0)

        with torch.no_grad():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                goal_img=goal_image,
                input_goal_mask=mask.repeat(len(goal_image)),
            )
            dists = to_numpy(self.model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())

        min_idx = int(np.argmin(dists))
        self.closest_node = min_idx + start
        sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        with torch.no_grad():
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

        return np.asarray(naction[0][self.args.waypoint], dtype=np.float32)

    def _tick_goal_conditioned(self) -> np.ndarray:
        start, end = self._window()
        batch_obs_imgs = []
        batch_goal_data = []
        for goal_img in self.topomap[start : end + 1]:
            batch_obs_imgs.append(transform_images(self.context_queue, self.model_params["image_size"]))
            batch_goal_data.append(transform_images(goal_img, self.model_params["image_size"]))

        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(DEVICE)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(DEVICE)

        with torch.no_grad():
            distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)

        min_dist_idx = int(np.argmin(distances))
        if distances[min_dist_idx] > self.args.close_threshold:
            chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
            self.closest_node = start + min_dist_idx
        else:
            chosen_waypoint = waypoints[min(min_dist_idx + 1, len(waypoints) - 1)][self.args.waypoint]
            self.closest_node = min(start + min_dist_idx + 1, self.goal_node)
        return np.asarray(chosen_waypoint, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run topological navigation on the robot")
    parser.add_argument("--model", "-m", default="nomad", type=str, help="model name from config/models.yaml")
    parser.add_argument("--waypoint", "-w", default=2, type=int, help="waypoint index used for navigation")
    parser.add_argument("--dir", "-d", default="topomap", type=str, help="topomap directory name or absolute path")
    parser.add_argument("--goal-node", "-g", default=-1, type=int, help="goal node index in the topomap")
    parser.add_argument("--close-threshold", "-t", default=3, type=int, help="localization threshold")
    parser.add_argument("--radius", "-r", default=4, type=int, help="number of nearby nodes to evaluate")
    parser.add_argument("--num-samples", "-n", default=8, type=int, help="number of sampled actions for NoMaD")
    return parser


def main(args=None) -> None:
    parsed_args, _ = build_parser().parse_known_args(args)
    rclpy.init(args=args)
    node = NavigateNode(parsed_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
