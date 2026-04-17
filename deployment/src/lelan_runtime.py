from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import warnings

import clip
import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "train"
DEPLOYMENT_SRC_DIR = REPO_ROOT / "deployment" / "src"
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"

for _path in (TRAIN_DIR, DEPLOYMENT_SRC_DIR, DIFFUSION_POLICY_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from utils import cv2pil, load_model, transform_images_lelan  # noqa: E402


MODEL_CANDIDATES = (
    "deployment/model_weights/with_col_loss.pth",
    "deployment/model_weights/wo_col_loss.pth",
    "deployment/model_weights/wo_col_loss_wo_temp.pth",
)


def _candidate_paths(relative_paths: List[str]) -> List[Path]:
    return [REPO_ROOT / relative_path for relative_path in relative_paths]


def _resolve_existing_path(
    requested_path: Optional[str],
    default_candidates: List[Path],
    label: str,
) -> Path:
    if requested_path:
        resolved = Path(requested_path).expanduser().resolve()
        if resolved.exists():
            return resolved
        searched = [resolved]
    else:
        searched = []

    for candidate in default_candidates:
        resolved = candidate.expanduser().resolve()
        searched.append(resolved)
        if resolved.exists():
            return resolved

    searched_text = "\n".join(f"  - {path}" for path in searched)
    raise FileNotFoundError(
        f"{label} not found. Checked:\n{searched_text}"
    )


def _check_cuda_device(device_index: int = 0) -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"

    try:
        capability = torch.cuda.get_device_capability(device_index)
        device_arch = f"sm_{capability[0]}{capability[1]}"
        supported_arches = sorted(set(torch.cuda.get_arch_list()))
        if supported_arches and device_arch not in supported_arches:
            supported_text = ", ".join(supported_arches)
            return (
                False,
                f"Detected CUDA device arch {device_arch}, but this PyTorch build only supports: {supported_text}",
            )

        torch.empty(1, device=f"cuda:{device_index}")
        return True, ""
    except Exception as exc:
        return False, f"CUDA device validation failed on cuda:{device_index}: {exc}"


def _resolve_runtime_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        resolved = torch.device(requested_device)
        if resolved.type != "cuda":
            return resolved

        device_index = 0 if resolved.index is None else resolved.index
        ok, reason = _check_cuda_device(device_index)
        if ok:
            return resolved

        raise RuntimeError(
            f"Requested CUDA device '{resolved}' is not usable: {reason}. "
            "Use '--device cpu' for now, or upgrade PyTorch to a build that supports this GPU."
        )

    ok, reason = _check_cuda_device(0)
    if ok:
        return torch.device("cuda:0")

    warnings.warn(
        f"{reason}. Falling back to CPU for LeLaN inference.",
        RuntimeWarning,
    )
    return torch.device("cpu")


class LeLaNInferenceRuntime:
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        prompt: str = "chair",
        device: Optional[str] = None,
    ) -> None:
        self.device = _resolve_runtime_device(device)
        self.model_path = _resolve_existing_path(
            model_path,
            _candidate_paths(list(MODEL_CANDIDATES)),
            "Model weights",
        )
        self.config_path = _resolve_existing_path(
            config_path,
            self._default_config_candidates(self.model_path),
            "Model config",
        )
        self.model_params = self._load_model_params(self.config_path)
        self.model = self._load_policy_model(self.model_path)
        self.prompt_features: Dict[str, torch.Tensor] = {}
        self.prompt = ""
        self.feat_text: Optional[torch.Tensor] = None
        self.history_len = 10
        self.image_hist = []
        self.set_prompt(prompt)

    @staticmethod
    def _default_config_candidates(model_path: Path) -> List[Path]:
        if model_path.name == "wo_col_loss_wo_temp.pth":
            return [REPO_ROOT / "train/config/lelan.yaml"]
        return [
            REPO_ROOT / "train/config/lelan_col.yaml",
            REPO_ROOT / "train/config/lelan.yaml",
        ]

    @staticmethod
    def _load_model_params(config_path: Path) -> dict:
        with config_path.open("r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj)

    def _load_policy_model(self, model_path: Path):
        model = load_model(str(model_path), self.model_params, self.device)
        model = model.to(self.device)
        model.eval()
        return model

    def set_prompt(self, prompt: str) -> None:
        prompt = prompt or "chair"
        if prompt not in self.prompt_features:
            with torch.no_grad():
                tokenized = clip.tokenize(prompt).to(self.device)
                self.prompt_features[prompt] = self.model("text_encoder", inst_ref=tokenized)
        self.prompt = prompt
        self.feat_text = self.prompt_features[prompt].to(torch.float32)

    def predict_from_cv2(self, cv_image, prompt: Optional[str] = None) -> Tuple[float, float]:
        if prompt and prompt != self.prompt:
            self.set_prompt(prompt)

        image = cv2pil(cv_image)
        if not self.image_hist:
            self.image_hist = [image for _ in range(self.history_len)]

        hist_image = self.image_hist[self.history_len - 1]
        obs_images, obs_current = transform_images_lelan(
            [hist_image, image],
            self.model_params["image_size"],
            center_crop=False,
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        batch_obs_images = obs_images.to(self.device)
        batch_obs_current = obs_current.to(self.device)

        with torch.no_grad():
            model_type = self.model_params["model_type"]
            if model_type in {"lelan_col", "lelan_col2"}:
                obsgoal_cond = self.model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    feat_text=self.feat_text,
                    current_img=batch_obs_current,
                )
            elif model_type == "lelan":
                obsgoal_cond = self.model(
                    "vision_encoder",
                    obs_img=batch_obs_current,
                    feat_text=self.feat_text,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            linear_vel, angular_vel = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)

        self.image_hist = [image] + self.image_hist[: self.history_len - 1]
        return (
            float(linear_vel.cpu().numpy()[0, 0]),
            float(angular_vel.cpu().numpy()[0, 0]),
        )


def decode_jpeg_image(payload: bytes):
    image_array = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image_array is None:
        raise ValueError("Failed to decode JPEG payload")
    return image_array
