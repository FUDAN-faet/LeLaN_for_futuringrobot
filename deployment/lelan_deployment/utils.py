"""Shared helpers for the ROS 2 deployment nodes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from torchvision import transforms
from typing import List

from .paths import ensure_train_on_path


ensure_train_on_path()

from vint_train.data.data_utils import IMAGE_ASPECT_RATIO
from vint_train.models.gnm.gnm import GNM
from vint_train.models.lelan.lelan import LeLaN_clip, LeLaN_clip_temp, DenseNetwork_lelan
from vint_train.models.lelan.lelan_comp import LeLaN_clip_FiLM, LeLaN_clip_FiLM_temp
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT

import clip
import cv2
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load a model from a checkpoint file."""
    model_type = config["model_type"]

    if model_type == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif model_type == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif model_type == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    elif model_type == "lelan":
        if config["vision_encoder"] != "lelan_clip_film":
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        vision_encoder = LeLaN_clip_FiLM(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            feature_size=config["feature_size"],
            clip_type=config["clip_type"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        text_encoder, _ = clip.load(config["clip_type"])
        dist_pred_network = DenseNetwork_lelan(
            embedding_dim=config["encoding_size"],
            control_horizon=config["len_traj_pred"],
        )
        model = LeLaN_clip(
            vision_encoder=vision_encoder,
            dist_pred_net=dist_pred_network,
            text_encoder=text_encoder,
        )
    elif model_type in {"lelan_col", "lelan_col2"}:
        if config["vision_encoder"] != "lelan_clip_film":
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        vision_encoder = LeLaN_clip_FiLM_temp(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            feature_size=config["feature_size"],
            clip_type=config["clip_type"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        text_encoder, _ = clip.load(config["clip_type"])
        text_encoder.to(torch.float32)
        dist_pred_network = DenseNetwork_lelan(
            embedding_dim=config["encoding_size"],
            control_horizon=config["len_traj_pred"],
        )
        model = LeLaN_clip_temp(
            vision_encoder=vision_encoder,
            dist_pred_net=dist_pred_network,
            text_encoder=text_encoder,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device)
    if model_type in {"nomad", "lelan", "lelan_col", "lelan_col2"}:
        model.load_state_dict(checkpoint, strict=True)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
        except AttributeError:
            state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def pil2cv(image: PILImage.Image) -> np.ndarray:
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        return new_image
    if new_image.shape[2] == 3:
        return cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    if new_image.shape[2] == 4:
        return cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image: np.ndarray) -> PILImage.Image:
    new_image = image.copy()
    if new_image.ndim == 2:
        return PILImage.fromarray(new_image)
    if new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    return PILImage.fromarray(new_image)


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8)
    channels = max(1, int(msg.step / msg.width))
    img = img.reshape(msg.height, msg.width, channels)
    return PILImage.fromarray(img)


def pil_to_msg(pil_img: PILImage.Image, encoding: str = "rgb8") -> Image:
    img = np.asarray(pil_img)
    ros_image = Image()
    ros_image.encoding = encoding
    ros_image.height = img.shape[0]
    ros_image.width = img.shape[1]
    ros_image.step = int(img.strides[0])
    ros_image.data = img.tobytes()
    return ros_image


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def transform_images(
    pil_imgs: List[PILImage.Image],
    image_size: List[int],
    center_crop: bool = False,
) -> torch.Tensor:
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_imgs.append(torch.unsqueeze(transf_img, 0))
    return torch.cat(transf_imgs, dim=1)


def transform_images_lelan(
    pil_imgs: List[PILImage.Image],
    image_size: List[int],
    center_crop: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_size_small = (96, 96)

    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size_small)
        transf_imgs.append(torch.unsqueeze(transform_type(pil_img), 0))

    cur_img = torch.unsqueeze(transform_type(pil_imgs[-1].resize(image_size)), 0)
    return torch.cat(transf_imgs, dim=1), cur_img


def clip_angle(angle: float) -> float:
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def clamp_velocity(vt: float, wt: float, max_v: float = 0.3, max_w: float = 0.5) -> Twist:
    msg = Twist()
    if np.absolute(vt) <= max_v:
        if np.absolute(wt) <= max_w:
            msg.linear.x = float(vt)
            msg.angular.z = float(wt)
        else:
            radius = vt / wt
            msg.linear.x = float(max_w * np.sign(vt) * np.absolute(radius))
            msg.angular.z = float(max_w * np.sign(wt))
    else:
        if np.absolute(wt) <= 0.001:
            msg.linear.x = float(max_v * np.sign(vt))
            msg.angular.z = 0.0
        else:
            radius = vt / wt
            if np.absolute(radius) >= max_v / max_w:
                msg.linear.x = float(max_v * np.sign(vt))
                msg.angular.z = float(max_v * np.sign(wt) / np.absolute(radius))
            else:
                msg.linear.x = float(max_w * np.sign(vt) * np.absolute(radius))
                msg.angular.z = float(max_w * np.sign(wt))
    return msg
