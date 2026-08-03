"""Stroke-aware image channels and box-safe augmentation for direct supervision."""
from __future__ import annotations

import copy
import math
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Subset


_TRAIN_ORIGINAL_BUILD_DATALOADERS = None
_EVAL_ORIGINAL_BUILD_TRANSFORM = None
_EVAL_ORIGINAL_LOAD_MODELS = None


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def preprocessing_config(default_enabled: bool = True) -> dict[str, Any]:
    enabled = _flag("DIRECT_SUBWORD_STROKE_INPUT", default_enabled)
    return {
        "stroke_aware_input": enabled,
        "stroke_aware_channels": "soft_ink,distance_proximity,sobel_magnitude",
        "stroke_aware_normalize": _flag("STROKE_AWARE_NORMALIZE", True),
        "stroke_aware_distance_clip": _number("STROKE_AWARE_DISTANCE_CLIP", 8.0),
        "stroke_aware_ink_threshold": _number("STROKE_AWARE_INK_THRESHOLD", 0.08),
        "stroke_aware_ink_ratio_threshold": _number(
            "STROKE_AWARE_INK_RATIO_THRESHOLD", 0.08
        ),
        "stroke_aware_augment": _flag("DIRECT_SUBWORD_STROKE_AUGMENT", True),
        "stroke_aware_morph_probability": _number("STROKE_AUG_MORPH_PROB", 0.40),
        "stroke_aware_gamma_probability": _number("STROKE_AUG_GAMMA_PROB", 0.30),
        "stroke_aware_gamma_min": _number("STROKE_AUG_GAMMA_MIN", 0.70),
        "stroke_aware_gamma_max": _number("STROKE_AUG_GAMMA_MAX", 1.40),
        "stroke_aware_contrast_min": _number("STROKE_AUG_CONTRAST_MIN", 0.80),
        "stroke_aware_contrast_max": _number("STROKE_AUG_CONTRAST_MAX", 1.20),
        "stroke_aware_blur_probability": _number("STROKE_AUG_BLUR_PROB", 0.20),
        "stroke_aware_blur_sigma_min": _number("STROKE_AUG_BLUR_SIGMA_MIN", 0.30),
        "stroke_aware_blur_sigma_max": _number("STROKE_AUG_BLUR_SIGMA_MAX", 1.00),
        "stroke_aware_noise_probability": _number("STROKE_AUG_NOISE_PROB", 0.15),
        "stroke_aware_noise_sigma_max": _number("STROKE_AUG_NOISE_SIGMA_MAX", 0.03),
        "stroke_aware_vertical_shift_probability": _number(
            "STROKE_AUG_VERTICAL_SHIFT_PROB", 0.20
        ),
        "stroke_aware_vertical_shift_max": _integer("STROKE_AUG_VERTICAL_SHIFT_MAX", 4),
    }


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ordered_bounds(left: float, right: float) -> tuple[float, float]:
    return (float(left), float(right)) if left <= right else (float(right), float(left))


def _morphology(ink: np.ndarray, dilate: bool) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(ink)).float()[None, None]
    if dilate:
        result = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    else:
        result = -F.max_pool2d(-tensor, kernel_size=3, stride=1, padding=1)
    return result[0, 0].clamp(0.0, 1.0).numpy()


def _gaussian_blur(ink: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(1e-3, float(sigma))
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    tensor = torch.from_numpy(np.ascontiguousarray(ink)).float()[None, None]
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    tensor = F.pad(tensor, (radius, radius, 0, 0), mode="replicate")
    tensor = F.conv2d(tensor, horizontal)
    tensor = F.pad(tensor, (0, 0, radius, radius), mode="replicate")
    tensor = F.conv2d(tensor, vertical)
    return tensor[0, 0].clamp(0.0, 1.0).numpy()


def _shift_vertical(ink: np.ndarray, offset: int) -> np.ndarray:
    offset = int(offset)
    if offset == 0:
        return ink
    shifted = np.zeros_like(ink)
    if offset > 0:
        shifted[offset:, :] = ink[:-offset, :]
    else:
        shifted[:offset, :] = ink[-offset:, :]
    return shifted


def _distance_proximity(ink: np.ndarray, threshold: float, clip: float) -> np.ndarray:
    mask = np.asarray(ink >= float(threshold), dtype=bool)
    clip = max(1.0, float(clip))
    if not mask.any():
        return np.zeros_like(ink, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        distance = distance_transform_edt(~mask).astype(np.float32, copy=False)
    except Exception:
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        reached = mask_tensor
        distance_tensor = torch.full_like(mask_tensor, float(clip))
        distance_tensor[mask_tensor > 0] = 0.0
        for step in range(1, int(math.ceil(clip)) + 1):
            expanded = F.max_pool2d(reached, kernel_size=3, stride=1, padding=1)
            newly_reached = (expanded > 0) & (reached <= 0)
            distance_tensor[newly_reached] = float(step)
            reached = expanded
        distance = distance_tensor[0, 0].numpy()
    return 1.0 - np.clip(distance / clip, 0.0, 1.0)


def _sobel_magnitude(ink: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(ink)).float()[None, None]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)
    padded = F.pad(tensor, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, kernel_x)
    grad_y = F.conv2d(padded, kernel_y)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square())[0, 0]
    maximum = magnitude.amax().clamp_min(1e-6)
    return (magnitude / maximum).clamp(0.0, 1.0).numpy()


class StrokeAwareLineTransform:
    """Return soft-ink, distance-proximity, and Sobel channels.

    Training augmentation is deliberately horizontal-geometry preserving so the
    renderer-derived subword x0/x1 intervals remain valid.
    """

    def __init__(
        self,
        size: tuple[int, int] = (128, 1024),
        *,
        training: bool = False,
        prepare_real: bool = False,
    ) -> None:
        self.height = int(size[0])
        self.width = int(size[1])
        self.training = bool(training)
        self.prepare_real = bool(prepare_real)
        self.config = preprocessing_config(default_enabled=True)

    def _prepare_grayscale(self, image: Image.Image) -> np.ndarray:
        image = image.convert("L").resize((self.width, self.height), Image.BILINEAR)
        if self.prepare_real:
            image = ImageOps.autocontrast(image)
            array_u8 = np.asarray(image, dtype=np.uint8)
            border_width = max(1, int(round(self.width * 0.01)))
            border_height = max(1, int(round(self.height * 0.05)))
            border = np.concatenate(
                [
                    array_u8[:border_height, :].reshape(-1),
                    array_u8[-border_height:, :].reshape(-1),
                    array_u8[:, :border_width].reshape(-1),
                    array_u8[:, -border_width:].reshape(-1),
                ]
            )
            if border.size and float(border.mean()) < 127.5:
                array_u8 = 255 - array_u8
            return array_u8.astype(np.float32) / 255.0
        return np.asarray(image, dtype=np.float32) / 255.0

    def _augment_ink(self, ink: np.ndarray) -> np.ndarray:
        cfg = self.config
        if random.random() < _clamp_probability(cfg["stroke_aware_gamma_probability"]):
            gamma_min, gamma_max = _ordered_bounds(
                cfg["stroke_aware_gamma_min"], cfg["stroke_aware_gamma_max"]
            )
            contrast_min, contrast_max = _ordered_bounds(
                cfg["stroke_aware_contrast_min"], cfg["stroke_aware_contrast_max"]
            )
            gamma = random.uniform(max(1e-3, gamma_min), max(1e-3, gamma_max))
            contrast = random.uniform(contrast_min, contrast_max)
            ink = np.clip(np.power(ink, gamma) * contrast, 0.0, 1.0)

        if random.random() < _clamp_probability(cfg["stroke_aware_morph_probability"]):
            ink = _morphology(ink, dilate=random.random() < 0.5)

        if random.random() < _clamp_probability(cfg["stroke_aware_blur_probability"]):
            sigma_min, sigma_max = _ordered_bounds(
                cfg["stroke_aware_blur_sigma_min"],
                cfg["stroke_aware_blur_sigma_max"],
            )
            ink = _gaussian_blur(
                ink,
                random.uniform(max(0.01, sigma_min), max(0.01, sigma_max)),
            )

        if random.random() < _clamp_probability(cfg["stroke_aware_noise_probability"]):
            sigma = random.uniform(0.0, max(0.0, cfg["stroke_aware_noise_sigma_max"]))
            if sigma > 0.0:
                noise = np.random.normal(0.0, sigma, size=ink.shape).astype(np.float32)
                ink = np.clip(ink + noise, 0.0, 1.0)

        if random.random() < _clamp_probability(
            cfg["stroke_aware_vertical_shift_probability"]
        ):
            maximum = max(0, int(cfg["stroke_aware_vertical_shift_max"]))
            if maximum > 0:
                ink = _shift_vertical(ink, random.randint(-maximum, maximum))
        return np.asarray(ink, dtype=np.float32)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        grayscale = self._prepare_grayscale(image)
        ink = np.clip(1.0 - grayscale, 0.0, 1.0).astype(np.float32, copy=False)
        if self.training and self.config["stroke_aware_augment"]:
            ink = self._augment_ink(ink)

        distance = _distance_proximity(
            ink,
            threshold=self.config["stroke_aware_ink_threshold"],
            clip=self.config["stroke_aware_distance_clip"],
        )
        edge = _sobel_magnitude(ink)
        channels = np.stack([ink, distance, edge], axis=0).astype(np.float32, copy=False)
        tensor = torch.from_numpy(np.ascontiguousarray(channels))
        if self.config["stroke_aware_normalize"]:
            tensor = (tensor - 0.5) / 0.5
        return tensor


def stroke_aware_window_ink_ratio_from_patches(
    patches: torch.Tensor, contrast_threshold: float | None = None
) -> torch.Tensor:
    """Compute per-window ink directly from the normalized soft-ink channel."""
    threshold = (
        _number("STROKE_AWARE_INK_RATIO_THRESHOLD", 0.08)
        if contrast_threshold is None
        else float(contrast_threshold)
    )
    threshold = max(0.0, min(1.0, threshold))
    with torch.no_grad():
        work = patches.detach().float()
        if work.ndim != 5 or work.shape[2] < 1:
            raise ValueError(
                "Expected stroke-aware patches with shape "
                "[batch, windows, channels, height, width]"
            )
        first_channel = work[:, :, 0]
        ink = (
            (first_channel + 1.0) * 0.5
            if _flag("STROKE_AWARE_NORMALIZE", True)
            else first_channel
        ).clamp(0.0, 1.0)
        return ink.ge(threshold).float().mean(dim=(2, 3))


def _clone_dataset_with_transform(dataset, transform):
    if isinstance(dataset, Subset):
        base = copy.copy(dataset.dataset)
        base.transform = transform
        return Subset(base, list(dataset.indices))
    clone = copy.copy(dataset)
    clone.transform = transform
    return clone


def stroke_aware_build_dataloaders(data_dir=None):
    if _TRAIN_ORIGINAL_BUILD_DATALOADERS is None:
        raise RuntimeError("Stroke-aware DataLoader patch is not installed")
    loaders = _TRAIN_ORIGINAL_BUILD_DATALOADERS(data_dir)
    if not _flag("DIRECT_SUBWORD_STROKE_INPUT", True):
        return loaders

    import DataLoader as loader_module

    train_loader, valid_loader, test_loader = loaders
    train_dataset = _clone_dataset_with_transform(
        train_loader.dataset,
        StrokeAwareLineTransform(training=True),
    )
    valid_dataset = _clone_dataset_with_transform(
        valid_loader.dataset,
        StrokeAwareLineTransform(training=False),
    )
    test_dataset = _clone_dataset_with_transform(
        test_loader.dataset,
        StrokeAwareLineTransform(training=False),
    )
    print(
        "Stroke-aware image input: channels=soft_ink,distance_proximity,sobel "
        f"train_augmentation={_flag('DIRECT_SUBWORD_STROKE_AUGMENT', True)} "
        "validation_augmentation=False",
        flush=True,
    )
    return (
        loader_module._make_loader(train_dataset, shuffle=True),
        loader_module._make_loader(valid_dataset, shuffle=False),
        loader_module._make_loader(test_dataset, shuffle=False),
    )


def install_training_preprocessing() -> None:
    global _TRAIN_ORIGINAL_BUILD_DATALOADERS
    import DataLoader as loader_module

    if getattr(loader_module, "_stroke_aware_preprocessing_installed", False):
        return
    _TRAIN_ORIGINAL_BUILD_DATALOADERS = loader_module.build_dataloaders
    loader_module.synthetic_transform = StrokeAwareLineTransform(training=False)
    loader_module.transform = loader_module.synthetic_transform
    loader_module.build_dataloaders = stroke_aware_build_dataloaders
    loader_module._stroke_aware_preprocessing_installed = True


def _set_env_value(name: str, value: Any) -> None:
    if isinstance(value, bool):
        os.environ[name] = "1" if value else "0"
    else:
        os.environ[name] = str(value)


def apply_checkpoint_preprocessing_config(config: dict[str, Any] | None) -> None:
    config = config or {}
    mapping = {
        "stroke_aware_input": "DIRECT_SUBWORD_STROKE_INPUT",
        "stroke_aware_normalize": "STROKE_AWARE_NORMALIZE",
        "stroke_aware_distance_clip": "STROKE_AWARE_DISTANCE_CLIP",
        "stroke_aware_ink_threshold": "STROKE_AWARE_INK_THRESHOLD",
        "stroke_aware_ink_ratio_threshold": "STROKE_AWARE_INK_RATIO_THRESHOLD",
        "stroke_aware_augment": "DIRECT_SUBWORD_STROKE_AUGMENT",
        "stroke_aware_morph_probability": "STROKE_AUG_MORPH_PROB",
        "stroke_aware_gamma_probability": "STROKE_AUG_GAMMA_PROB",
        "stroke_aware_gamma_min": "STROKE_AUG_GAMMA_MIN",
        "stroke_aware_gamma_max": "STROKE_AUG_GAMMA_MAX",
        "stroke_aware_contrast_min": "STROKE_AUG_CONTRAST_MIN",
        "stroke_aware_contrast_max": "STROKE_AUG_CONTRAST_MAX",
        "stroke_aware_blur_probability": "STROKE_AUG_BLUR_PROB",
        "stroke_aware_blur_sigma_min": "STROKE_AUG_BLUR_SIGMA_MIN",
        "stroke_aware_blur_sigma_max": "STROKE_AUG_BLUR_SIGMA_MAX",
        "stroke_aware_noise_probability": "STROKE_AUG_NOISE_PROB",
        "stroke_aware_noise_sigma_max": "STROKE_AUG_NOISE_SIGMA_MAX",
        "stroke_aware_vertical_shift_probability": "STROKE_AUG_VERTICAL_SHIFT_PROB",
        "stroke_aware_vertical_shift_max": "STROKE_AUG_VERTICAL_SHIFT_MAX",
    }
    for key, environment_name in mapping.items():
        if key in config:
            _set_env_value(environment_name, config[key])
    if _flag("DIRECT_SUBWORD_STROKE_INPUT", False):
        import embeddingModel as embedding_model_module

        embedding_model_module.window_ink_ratio_from_patches = (
            stroke_aware_window_ink_ratio_from_patches
        )
    # Evaluation must always be deterministic even when the checkpoint records
    # that training augmentation was enabled.
    os.environ["DIRECT_SUBWORD_STROKE_AUGMENT"] = "0"


def stroke_aware_evaluation_transform(dataset_type: str = "synthetic"):
    if not _flag("DIRECT_SUBWORD_STROKE_INPUT", False):
        if _EVAL_ORIGINAL_BUILD_TRANSFORM is None:
            raise RuntimeError("Evaluation preprocessing patch is not installed")
        return _EVAL_ORIGINAL_BUILD_TRANSFORM(dataset_type)
    return StrokeAwareLineTransform(
        training=False,
        prepare_real=str(dataset_type).lower() == "real",
    )


def stroke_aware_load_evaluation_models(*args, **kwargs):
    if _EVAL_ORIGINAL_LOAD_MODELS is None:
        raise RuntimeError("Evaluation model-loader patch is not installed")
    models = _EVAL_ORIGINAL_LOAD_MODELS(*args, **kwargs)
    apply_checkpoint_preprocessing_config(getattr(models, "config", None))
    return models


def install_evaluation_preprocessing(*implementation_modules) -> None:
    global _EVAL_ORIGINAL_BUILD_TRANSFORM, _EVAL_ORIGINAL_LOAD_MODELS
    from Evaluation import _eval_utils as eval_utils
    from Evaluation import sw_runner

    if not getattr(eval_utils, "_stroke_aware_preprocessing_installed", False):
        _EVAL_ORIGINAL_BUILD_TRANSFORM = eval_utils.build_transform
        _EVAL_ORIGINAL_LOAD_MODELS = eval_utils.load_evaluation_models
        eval_utils.build_transform = stroke_aware_evaluation_transform
        eval_utils.load_evaluation_models = stroke_aware_load_evaluation_models
        eval_utils._stroke_aware_preprocessing_installed = True

    sw_runner.load_evaluation_models = stroke_aware_load_evaluation_models
    sw_runner.get_image_features = eval_utils.get_image_features
    for module in implementation_modules:
        module.load_evaluation_models = stroke_aware_load_evaluation_models
        module.get_image_features = eval_utils.get_image_features
