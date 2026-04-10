from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import clip
import numpy as np
import rclpy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from .paths import resolve_from_deployment, resolve_readonly_path
from .topic_names import (
    IMAGE_TOPIC,
    REACHED_GOAL_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    VELOCITY_LELAN_TOPIC,
    WAYPOINT_TOPIC,
)
from .utils import clamp_velocity, load_model, msg_to_pil, to_numpy, transform_images, transform_images_lelan

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


def argmax(values: list[float]) -> int:
    if not values:
        raise ValueError("List is empty")
    max_index = 0
    for idx in range(1, len(values)):
        if values[idx] > values[max_index]:
            max_index = idx
    return max_index


def load_owlvit(checkpoint_path: str = "owlv2-base-patch16-ensemble"):
    processor = Owlv2Processor.from_pretrained(f"google/{checkpoint_path}")
    model = Owlv2ForObjectDetection.from_pretrained(f"google/{checkpoint_path}")
    model.to(DEVICE)
    model.eval()
    return model, processor


def find_target_object(topomap: list[PILImage.Image], texts: list[list[str]]) -> tuple[int, int]:
    model_owl, processor_owl = load_owlvit()
    size_list = []
    score_list = []

    for image in topomap:
        with torch.no_grad():
            inputs_owl = processor_owl(text=texts, images=image, return_tensors="pt").to(DEVICE)
            outputs_owl = model_owl(**inputs_owl)

        target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
        results = processor_owl.post_process_object_detection(
            outputs=outputs_owl,
            threshold=0.0,
            target_sizes=target_sizes,
        )

        scores = torch.sigmoid(outputs_owl.logits)
        topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)
        topk_idxs = topk_idxs.squeeze(1).tolist()
        topk_boxes = results[0]["boxes"][topk_idxs]
        topk_scores = topk_scores.view(len(texts[0]), -1)
        topk_labels = results[0]["labels"][topk_idxs]

        for box, score, _label in zip(topk_boxes, topk_scores, topk_labels):
            box = [round(value, 2) for value in box.tolist()]
            size_list.append((box[2] - box[0]) * (box[3] - box[1]))
            score_list.append(float(score.item()))

    if not score_list:
        fallback = max(0, len(topomap) - 1)
        return fallback, fallback
    return argmax(score_list), argmax(size_list)


class NavigateLeLanNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("navigate_lelan")
        self.args = args
        self.context_queue = []
        self.context_size = None
        self.obs_img: PILImage.Image | None = None
        self.closest_node = 0
        self.reached_goal = False
        self.lelan_image_hist: list[PILImage.Image] = []
        self.switch_logged = False

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

        self.lelan_config_path = resolve_from_deployment(args.config)
        with open(self.lelan_config_path, "r", encoding="utf-8") as f:
            self.lelan_params = yaml.safe_load(f)

        lelan_ckpt_path = resolve_from_deployment(args.lelan_model)
        if not lelan_ckpt_path.exists():
            raise FileNotFoundError(f"LeLaN model weights not found at {lelan_ckpt_path}")
        self.model_lelan = load_model(str(lelan_ckpt_path), self.lelan_params, DEVICE).to(DEVICE)
        self.model_lelan.eval()

        topomap_dir = resolve_topomap_dir(args.dir)
        if not topomap_dir.is_dir():
            raise FileNotFoundError(f"Topomap directory not found: {topomap_dir}")
        topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda name: int(name.split(".")[0]))
        self.topomap = []
        for filename in topomap_filenames:
            with PILImage.open(topomap_dir / filename) as image:
                self.topomap.append(image.copy())

        if not self.topomap:
            raise ValueError(f"No topomap images found in {topomap_dir}")

        texts = [[args.prompt]]
        with torch.no_grad():
            batch_obj_inst = clip.tokenize(texts[0][0]).to(DEVICE)
            self.feat_text = self.model_lelan("text_encoder", inst_ref=batch_obj_inst)

        nodeid_score, _nodeid_size = find_target_object(self.topomap, texts)
        self.goal_node = nodeid_score
        self.get_logger().info(f"Selected target node {self.goal_node} for prompt '{args.prompt}'")

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
        self.velocity_lelan_pub = self.create_publisher(Twist, VELOCITY_LELAN_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.create_timer(1.0 / RATE, self._tick)

        self.get_logger().info(f"Using device: {DEVICE}")

    def _on_image(self, msg: Image) -> None:
        self.obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(self.obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(self.obs_img)

    def _window(self) -> tuple[int, int]:
        start = max(self.closest_node - self.args.radius, 0)
        end = min(self.closest_node + self.args.radius + 1, self.goal_node)
        return start, end

    def _publish_goal(self) -> None:
        msg = Bool()
        msg.data = self.reached_goal
        self.goal_pub.publish(msg)

    def _tick(self) -> None:
        chosen_waypoint = np.zeros(4, dtype=np.float32)

        if len(self.context_queue) > self.context_size:
            if self.model_params["model_type"] == "nomad":
                chosen_waypoint = self._tick_nomad()
            else:
                chosen_waypoint = self._tick_goal_conditioned()

        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.astype(np.float32).tolist()
        self.waypoint_pub.publish(waypoint_msg)

        if not self.reached_goal:
            self.reached_goal = self.closest_node >= self.goal_node
        self._publish_goal()

        if self.reached_goal and not self.switch_logged:
            self.get_logger().info("Reached target node. Switching to LeLaN final approach.")
            self.switch_logged = True

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
        if not self.reached_goal:
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

            local_idx = int(np.argmin(distances))
            if distances[local_idx] > self.args.close_threshold:
                chosen_waypoint = waypoints[local_idx][self.args.waypoint]
                self.closest_node = start + local_idx
            else:
                next_idx = min(local_idx + 1, len(waypoints) - 1)
                chosen_waypoint = waypoints[next_idx][self.args.waypoint]
                self.closest_node = min(start + next_idx, self.goal_node)
            return np.asarray(chosen_waypoint, dtype=np.float32)

        self._publish_lelan_velocity()
        return np.zeros(4, dtype=np.float32)

    def _publish_lelan_velocity(self) -> None:
        if self.obs_img is None:
            return

        if self.lelan_params["model_type"] == "lelan_col":
            if not self.lelan_image_hist:
                self.lelan_image_hist = [self.obs_img for _ in range(10)]
            im_obs = [self.lelan_image_hist[9], self.obs_img]
            obs_images, obs_current = transform_images_lelan(im_obs, self.lelan_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(DEVICE)
            obs_current = obs_current.to(DEVICE)
            with torch.no_grad():
                obsgoal_cond = self.model_lelan(
                    "vision_encoder",
                    obs_img=obs_images,
                    feat_text=self.feat_text.float(),
                    current_img=obs_current,
                )
                linear_vel, angular_vel = self.model_lelan("dist_pred_net", obsgoal_cond=obsgoal_cond)
            self.lelan_image_hist = [self.obs_img] + self.lelan_image_hist[:9]
        else:
            _obs_images, obs_current = transform_images_lelan([self.obs_img], self.lelan_params["image_size"], center_crop=False)
            obs_current = obs_current.to(DEVICE)
            with torch.no_grad():
                obsgoal_cond = self.model_lelan(
                    "vision_encoder",
                    obs_img=obs_current,
                    feat_text=self.feat_text.float(),
                )
                linear_vel, angular_vel = self.model_lelan("dist_pred_net", obsgoal_cond=obsgoal_cond)

        vt = float(linear_vel.cpu().numpy()[0, 0])
        wt = float(angular_vel.cpu().numpy()[0, 0])
        self.velocity_lelan_pub.publish(clamp_velocity(vt, wt))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run long-distance LeLaN navigation with a topological map")
    parser.add_argument("--model", "-m", default="vint", type=str, help="navigation model name from config/models.yaml")
    parser.add_argument("--waypoint", "-w", default=2, type=int, help="waypoint index used for navigation")
    parser.add_argument("--dir", "-d", default="topomap", type=str, help="topomap directory name or absolute path")
    parser.add_argument("--goal-node", "-g", default=-1, type=int, help="reserved goal node override")
    parser.add_argument("--close-threshold", "-t", default=3, type=int, help="localization threshold")
    parser.add_argument("--radius", "-r", default=4, type=int, help="number of nearby nodes to evaluate")
    parser.add_argument("--num-samples", "-n", default=8, type=int, help="number of sampled actions for NoMaD")
    parser.add_argument("--prompt", "-p", default="person", type=str, help="target object prompt")
    parser.add_argument(
        "--config",
        "-c",
        default="../../train/config/lelan.yaml",
        type=str,
        help="LeLaN config path relative to deployment/",
    )
    parser.add_argument(
        "--lelan-model",
        default="../model_weights/wo_col_loss_wo_temp.pth",
        type=str,
        help="LeLaN checkpoint path relative to deployment/",
    )
    return parser


def main(args=None) -> None:
    parsed_args, _ = build_parser().parse_known_args(args)
    rclpy.init(args=args)
    node = NavigateLeLanNode(parsed_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
