"""Strict zero-shot preprocessing and visual-training helpers.

This module deliberately uses only synthetic images during training.  It makes
synthetic and real line images enter the visual encoder through the same
geometry and binary-image pipeline, while randomizing synthetic appearance so
real manuscript scans are less out-of-distribution.
"""
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8)
    histogram = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    total = float(values.size)
    if total <= 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    total_sum = float(np.dot(levels, histogram))
    left_weight = 0.0
    left_sum = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(256):
        left_weight += histogram[threshold]
        if left_weight <= 0:
            continue
        right_weight = total - left_weight
        if right_weight <= 0:
            break
        left_sum += threshold * histogram[threshold]
        left_mean = left_sum / left_weight
        right_mean = (total_sum - left_sum) / right_weight
        variance = left_weight * right_weight * (left_mean - right_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def _border_mean(values: np.ndarray) -> float:
    height, width = values.shape
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.01)))
    border = np.concatenate(
        [
            values[:border_h, :].reshape(-1),
            values[-border_h:, :].reshape(-1),
            values[:, :border_w].reshape(-1),
            values[:, -border_w:].reshape(-1),
        ]
    )
    return float(border.mean()) if border.size else 255.0


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    threshold = otsu_threshold(gray)
    dark_ink = gray <= threshold
    light_ink = gray > threshold
    # The border normally represents page background.  Select the foreground
    # polarity that disagrees with it.
    return dark_ink if _border_mean(gray) >= 127.5 else light_ink


def foreground_crop(image: Image.Image, margin_x=0.025, margin_y=0.15) -> Image.Image:
    gray_image = ImageOps.autocontrast(image.convert("L"))
    gray = np.asarray(gray_image, dtype=np.uint8)
    mask = _ink_mask(gray)
    ys, xs = np.nonzero(mask)
    if xs.size < 4 or ys.size < 4:
        return gray_image
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int(round((x1 - x0) * float(margin_x))))
    pad_y = max(2, int(round((y1 - y0) * float(margin_y))))
    x0 = max(0, x0 - pad_x)
    x1 = min(gray.shape[1], x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(gray.shape[0], y1 + pad_y)
    return gray_image.crop((x0, y0, x1, y1))


def aspect_preserving_pad(
    image: Image.Image,
    size=(128, 1024),
    target_ink_height_ratio=0.72,
    horizontal_jitter=0.0,
) -> Image.Image:
    target_h, target_w = map(int, size)
    image = image.convert("L")
    desired_h = max(8, int(round(target_h * float(target_ink_height_ratio))))
    scale = min(
        desired_h / max(1, image.height),
        target_w / max(1, image.width),
    )
    new_w = max(1, min(target_w, int(round(image.width * scale))))
    new_h = max(1, min(target_h, int(round(image.height * scale))))
    resized = image.resize((new_w, new_h), _BILINEAR)
    canvas = Image.new("L", (target_w, target_h), color=255)
    max_x = max(0, target_w - new_w)
    centered_x = max_x // 2
    jitter = int(round(max_x * max(0.0, float(horizontal_jitter))))
    x = min(max_x, max(0, centered_x + random.randint(-jitter, jitter))) if jitter else centered_x
    y = max(0, (target_h - new_h) // 2)
    canvas.paste(resized, (x, y))
    return canvas


def _random_resize(image: Image.Image) -> Image.Image:
    width_scale = random.uniform(0.84, 1.16)
    height_scale = random.uniform(0.90, 1.10)
    width = max(8, int(round(image.width * width_scale)))
    height = max(8, int(round(image.height * height_scale)))
    return image.resize((width, height), _BILINEAR)


def _add_gray_noise(image: Image.Image, std: float) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    array += np.random.normal(0.0, float(std), size=array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


def _add_scan_artifacts(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    height, width = array.shape

    # Broken/faded strokes: short white interruptions.
    for _ in range(random.randint(0, 4)):
        block_w = random.randint(1, max(2, width // 100))
        block_h = random.randint(1, max(2, height // 12))
        x0 = random.randint(0, max(0, width - block_w))
        y0 = random.randint(0, max(0, height - block_h))
        array[y0 : y0 + block_h, x0 : x0 + block_w] = 255

    # Dust and bleed-through-like speckles.
    count = int(array.size * random.uniform(0.0, 0.0025))
    if count > 0:
        ys = np.random.randint(0, height, size=count)
        xs = np.random.randint(0, width, size=count)
        array[ys, xs] = np.random.choice([0, 40, 210, 255], size=count)
    return Image.fromarray(array, mode="L")


class ManuscriptLinePreprocessor:
    """Crop, geometrically normalize, degrade, and binarize a line image."""

    def __init__(
        self,
        size=(128, 1024),
        *,
        training=False,
        augment=False,
        binarize=True,
        method="otsu",
        fixed_threshold=180,
        threshold_jitter=0,
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
        auto_invert=True,
        autocontrast=True,
        augment_probability=0.85,
        clean_probability=0.20,
    ):
        self.size = tuple(map(int, size))
        self.training = bool(training)
        self.augment = bool(augment)
        self.binarize = bool(binarize)
        self.method = str(method).lower()
        self.fixed_threshold = int(fixed_threshold)
        self.threshold_jitter = max(0, int(threshold_jitter))
        self.preserve_aspect = bool(preserve_aspect)
        self.crop_foreground = bool(crop_foreground)
        self.target_ink_height_ratio = float(target_ink_height_ratio)
        self.auto_invert = bool(auto_invert)
        self.autocontrast = bool(autocontrast)
        self.augment_probability = float(augment_probability)
        self.clean_probability = float(clean_probability)
        if self.method not in {"otsu", "fixed", "random"}:
            raise ValueError("binarization method must be otsu, fixed, or random")

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.training or not self.augment:
            return image
        if random.random() < self.clean_probability:
            return image
        if random.random() > self.augment_probability:
            return image

        image = _random_resize(image)
        angle = random.uniform(-2.5, 2.5)
        image = image.rotate(angle, resample=_BILINEAR, expand=False, fillcolor=255)

        if random.random() < 0.45:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.25, 1.15)))
        if random.random() < 0.35:
            # Black ink: MinFilter expands strokes, MaxFilter erodes them.
            image = image.filter(
                ImageFilter.MinFilter(3) if random.random() < 0.5 else ImageFilter.MaxFilter(3)
            )
        if random.random() < 0.70:
            image = _add_gray_noise(image, random.uniform(2.0, 14.0))
        if random.random() < 0.65:
            image = _add_scan_artifacts(image)
        return image

    def _threshold(self, gray: np.ndarray) -> int:
        if self.method == "fixed":
            return max(0, min(255, self.fixed_threshold))
        base = otsu_threshold(gray)
        if self.method == "random" and self.training and self.threshold_jitter > 0:
            base += random.randint(-self.threshold_jitter, self.threshold_jitter)
        return max(0, min(255, int(base)))

    def __call__(self, image: Image.Image) -> Image.Image:
        work = image.convert("L")
        if self.autocontrast:
            work = ImageOps.autocontrast(work)
        if self.crop_foreground:
            work = foreground_crop(work)
        work = self._augment(work)
        if self.preserve_aspect:
            work = aspect_preserving_pad(
                work,
                self.size,
                self.target_ink_height_ratio,
                horizontal_jitter=0.08 if self.training and self.augment else 0.0,
            )
        else:
            work = work.resize((self.size[1], self.size[0]), _BILINEAR)

        if not self.binarize:
            return work.convert("RGB")
        gray = np.asarray(ImageOps.autocontrast(work), dtype=np.uint8)
        threshold = self._threshold(gray)
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        if self.auto_invert and _border_mean(binary) < 127.5:
            binary = 255 - binary
        return Image.fromarray(binary, mode="L").convert("RGB")


def build_preprocessor(dataset_type: str, training: bool) -> ManuscriptLinePreprocessor:
    synthetic = str(dataset_type).lower() == "synthetic"
    enabled = env_flag("ZERO_SHOT_PREPROCESS", True)
    if not enabled:
        return ManuscriptLinePreprocessor(
            training=False,
            augment=False,
            binarize=False,
            preserve_aspect=False,
            crop_foreground=False,
        )
    return ManuscriptLinePreprocessor(
        training=bool(training),
        augment=synthetic and env_flag("SYNTHETIC_MANUSCRIPT_AUGMENT", True),
        binarize=(
            env_flag("SYNTHETIC_BINARIZE", True)
            if synthetic
            else env_flag("REAL_BINARIZE", True)
        ),
        method=(
            os.environ.get("SYNTHETIC_BINARIZE_METHOD", "random")
            if synthetic
            else os.environ.get("REAL_BINARIZE_METHOD", "otsu")
        ),
        fixed_threshold=env_int(
            "SYNTHETIC_BINARIZE_THRESHOLD" if synthetic else "REAL_BINARIZE_THRESHOLD",
            180,
        ),
        threshold_jitter=env_int("SYNTHETIC_THRESHOLD_JITTER", 24) if synthetic else 0,
        preserve_aspect=env_flag("ZERO_SHOT_PRESERVE_ASPECT", True),
        crop_foreground=env_flag("ZERO_SHOT_FOREGROUND_CROP", True),
        target_ink_height_ratio=env_float("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72),
        auto_invert=env_flag("REAL_BINARIZE_AUTO_INVERT", True),
        autocontrast=env_flag("REAL_BINARIZE_AUTOCONTRAST", True),
        augment_probability=env_float("SYNTHETIC_AUGMENT_PROBABILITY", 0.85),
        clean_probability=env_float("SYNTHETIC_CLEAN_PROBABILITY", 0.20),
    )


def build_tensor_transform(dataset_type: str, training: bool):
    return transforms.Compose(
        [
            build_preprocessor(dataset_type, training),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TransformViewDataset(Dataset):
    """Apply a split-specific PIL transform without changing the base records."""

    def __init__(self, dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _transform_image(self, image):
        if torch.is_tensor(image):
            return image
        return self.transform(image)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, dict):
            result = dict(item)
            if "image1" in result:
                result["image1"] = self._transform_image(result["image1"])
            if "image2" in result:
                result["image2"] = self._transform_image(result["image2"])
            return result
        if isinstance(item, tuple) and len(item) == 2:
            text, image = item
            return text, self._transform_image(image)
        return item


def install_dataloader_profile() -> None:
    """Install augmented train and clean validation transforms for synthetic data."""
    import DataLoader as data_loader

    if getattr(data_loader, "_zero_shot_profile_installed", False):
        return
    original_build = data_loader.build_dataloaders

    def build_dataloaders(data_dir=None):
        resolved = data_dir or data_loader._default_data_dir
        if data_loader._detect_dataset_type(resolved) != "synthetic":
            return original_build(resolved)

        # Build the base dataset with decoded PIL images.  Augmentation is applied
        # only by the training wrapper; validation and test remain deterministic.
        previous = data_loader.synthetic_transform
        data_loader.synthetic_transform = None
        try:
            full_dataset = data_loader._build_synthetic_dataset(resolved)
        finally:
            data_loader.synthetic_transform = previous
        train_subset, valid_subset, test_subset = data_loader._random_split_seeded(
            full_dataset
        )
        train_dataset = TransformViewDataset(
            train_subset, build_tensor_transform("synthetic", training=True)
        )
        clean_transform = build_tensor_transform("synthetic", training=False)
        valid_dataset = TransformViewDataset(valid_subset, clean_transform)
        test_dataset = TransformViewDataset(test_subset, clean_transform)
        return (
            data_loader._make_loader(train_dataset, shuffle=True),
            data_loader._make_loader(valid_dataset, shuffle=False),
            data_loader._make_loader(test_dataset, shuffle=False),
        )

    data_loader.build_dataloaders = build_dataloaders
    data_loader._zero_shot_profile_installed = True


def install_embedding_profile(train_module) -> None:
    """Use grouped+local features in the existing local hard-negative objective."""
    if getattr(train_module, "_zero_shot_embedding_profile_installed", False):
        return

    def compute_embeddings(image_embedder, images):
        with train_module.autocast(
            dtype=train_module.AMP_DTYPE, enabled=train_module.USE_AMP
        ):
            contextual, local, grouped, ink = image_embedder(
                images,
                return_local=True,
                return_grouped=True,
                return_ink=True,
            )
        grouped_weight = max(0.0, min(1.0, env_float("ZERO_SHOT_GROUPED_BLEND", 0.50)))
        local_grouped = (1.0 - grouped_weight) * local + grouped_weight * grouped
        return (
            F.normalize(contextual.float(), p=2, dim=-1),
            F.normalize(local_grouped.float(), p=2, dim=-1),
            ink,
            local_grouped,
        )

    train_module.compute_embeddings = compute_embeddings
    train_module._zero_shot_embedding_profile_installed = True


def _force_batch_norm_eval(module, _inputs) -> None:
    module.training = False


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _replace_batch_norm(parent: nn.Module) -> int:
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            group_norm = nn.GroupNorm(_group_count(child.num_features), child.num_features)
            if child.affine:
                with torch.no_grad():
                    group_norm.weight.copy_(child.weight)
                    group_norm.bias.copy_(child.bias)
            setattr(parent, name, group_norm)
            replaced += 1
        else:
            replaced += _replace_batch_norm(child)
    return replaced


def configure_domain_robust_normalization(model: nn.Module) -> dict:
    """Prevent BatchNorm running statistics from specializing to synthetic scans."""
    mode = os.environ.get("ZERO_SHOT_NORM_MODE", "frozen-bn").strip().lower()
    if mode in {"none", "train-bn", "batchnorm"}:
        return {"zero_shot_norm_mode": "train-bn", "zero_shot_norm_layers": 0}
    if mode == "groupnorm":
        count = _replace_batch_norm(model)
        return {"zero_shot_norm_mode": "groupnorm", "zero_shot_norm_layers": count}
    if mode not in {"frozen", "frozen-bn", "frozen_batchnorm"}:
        raise ValueError("ZERO_SHOT_NORM_MODE must be frozen-bn, groupnorm, or train-bn")

    count = 0
    handles = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            if module.affine:
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
            handles.append(module.register_forward_pre_hook(_force_batch_norm_eval))
            count += 1
    model._zero_shot_batch_norm_handles = handles
    return {"zero_shot_norm_mode": "frozen-bn", "zero_shot_norm_layers": count}


def zero_shot_config() -> dict:
    return {
        "zero_shot_profile": True,
        "zero_shot_preserve_aspect": env_flag("ZERO_SHOT_PRESERVE_ASPECT", True),
        "zero_shot_foreground_crop": env_flag("ZERO_SHOT_FOREGROUND_CROP", True),
        "zero_shot_target_ink_height_ratio": env_float(
            "ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72
        ),
        "synthetic_manuscript_augment": env_flag(
            "SYNTHETIC_MANUSCRIPT_AUGMENT", True
        ),
        "synthetic_binarize": env_flag("SYNTHETIC_BINARIZE", True),
        "synthetic_binarize_method": os.environ.get(
            "SYNTHETIC_BINARIZE_METHOD", "random"
        ),
        "synthetic_threshold_jitter": env_int("SYNTHETIC_THRESHOLD_JITTER", 24),
        "zero_shot_grouped_blend": env_float("ZERO_SHOT_GROUPED_BLEND", 0.50),
        "zero_shot_norm_mode": os.environ.get("ZERO_SHOT_NORM_MODE", "frozen-bn"),
    }
