"""Checkpoint-owned evaluation pixels, tensorization, and window runtime flags.

Historical preprocessing/model code reads environment variables. Evaluation
scopes those reads to a resolved contract; training code and weights are untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import os
from pathlib import Path
from threading import RLock

import torch
from torchvision import transforms

from zero_shot_preprocessing import (
    IMAGENET_GRAY_MEAN, IMAGENET_GRAY_STD, IMAGENET_MEAN, IMAGENET_STD,
)

_ENVIRONMENT_LOCK = RLock()


def flag(value):
    return value.strip().lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


@contextmanager
def evaluation_environment(values):
    """Restore the caller's environment, including on a failed forward."""
    with _ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update({key: str(value) for key, value in values.items()})
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@dataclass(frozen=True)
class EvaluationContract:
    config: dict
    architecture_family: str
    backend_variant: str
    visual_input_channels: int
    visual_grayscale: bool
    normalization_mean: tuple
    normalization_std: tuple
    line_height: int
    line_width: int
    line_geometry_mode: str
    real_bbox_crop: bool
    foreground_crop: bool
    preserve_aspect: bool
    padding: bool
    real_binarize: bool
    synthetic_binarize: bool
    autocontrast: bool
    inversion: bool
    window_size: int
    stride: int
    pack_valid_windows: bool
    full_image_no_padding: bool
    real_all_page_lines: bool
    split_seed: int
    fallback_fields: tuple

    @property
    def color_mode(self):
        return "L" if self.visual_grayscale else "RGB"

    @property
    def window_count(self):
        return 1 + (self.line_width - self.window_size) // self.stride

    @property
    def split_strategy(self):
        return "all_page_lines_source_page_split" if self.real_all_page_lines else "legacy_dataset_split"

    def environment(self):
        c = self.config
        values = {
            "VISUAL_INPUT_CHANNELS": self.visual_input_channels,
            "VISUAL_GRAYSCALE": int(self.visual_grayscale),
            "REAL_GRAYSCALE": int(self.visual_grayscale),
            "LINE_HEIGHT": self.line_height, "LINE_WIDTH": self.line_width,
            "LINE_GEOMETRY_MODE": self.line_geometry_mode,
            "REAL_BBOX_CROP": int(self.real_bbox_crop),
            "REAL_BBOX_CROP_STRICT": 1,
            "REAL_BBOX_MARGIN_RATIO": c.get("real_bbox_margin_ratio", 0.05),
            "REAL_BBOX_MIN_MARGIN_PX": c.get("real_bbox_min_margin_px", 2),
            "ZERO_SHOT_PREPROCESS": int(flag(c.get("zero_shot_preprocess", True))),
            "ZERO_SHOT_FOREGROUND_CROP": int(self.foreground_crop),
            "ZERO_SHOT_PRESERVE_ASPECT": int(self.preserve_aspect),
            "ZERO_SHOT_SOURCE_GEOMETRY": int(self.line_geometry_mode == "source-compatible-height"),
            "ZERO_SHOT_CROP_MODE": c.get("zero_shot_crop_mode", "vertical_borders" if self.line_geometry_mode == "crop-aspect-preserving-rgb" else "legacy_otsu"),
            "TARGET_INK_HEIGHT_RATIO": c.get("target_ink_height_ratio", 0.72),
            "ZERO_SHOT_TARGET_INK_HEIGHT_RATIO": c.get("target_ink_height_ratio", 0.72),
            "REAL_BINARIZE": int(self.real_binarize),
            "SYNTHETIC_BINARIZE": int(self.synthetic_binarize),
            "REAL_BINARIZE_AUTOCONTRAST": int(self.autocontrast),
            "REAL_BINARIZE_AUTO_INVERT": int(self.inversion),
            "REAL_BINARIZE_METHOD": c.get("real_binarize_method", "otsu"),
            "REAL_BINARIZE_THRESHOLD": c.get("real_binarize_threshold", 180),
            "SYNTHETIC_BINARIZE_METHOD": c.get("synthetic_binarize_method", "random"),
            "SYNTHETIC_BINARIZE_THRESHOLD": c.get("synthetic_binarize_threshold", 180),
            "REAL_SYNTHETIC_STYLE": int(flag(c.get("real_synthetic_style", False))),
            "PACK_VALID_WINDOWS": int(self.pack_valid_windows),
            "FULL_IMAGE_NO_PADDING": int(self.full_image_no_padding),
            "REAL_ALL_PAGE_LINES": int(self.real_all_page_lines),
            "REAL_MANIFEST_NAME": c.get("real_manifest_name", "dataset_manifest.jsonl"),
            "DATASET_SPLIT_SEED": self.split_seed,
        }
        return {key: str(value) for key, value in values.items()}

    def install(self):
        """Install defaults for legacy callers; explicit operations also scope them."""
        os.environ.update(self.environment())
        from unified_line_geometry import install_evaluation_geometry
        return install_evaluation_geometry()

    def prepare_line(self, path, domain="real", preprocessing="training"):
        from unified_line_geometry import install_evaluation_geometry
        from Evaluation.yelda_geometry import prepare_line
        with evaluation_environment(self.environment()):
            install_evaluation_geometry()
            image, geometry = prepare_line(path, domain, preprocessing)
        image = image.convert(self.color_mode)
        geometry.update(color_mode=image.mode, grayscale=self.visual_grayscale)
        return image, geometry

    def tensor_transform(self):
        """Tensorize prepared pixels only: no crop, resize, padding, or threshold."""
        return transforms.Compose([
            transforms.Lambda(lambda image: image.convert(self.color_mode)),
            transforms.ToTensor(),
            transforms.Normalize(self.normalization_mean, self.normalization_std),
        ])

    def dummy(self, device="cpu"):
        return torch.zeros((1, self.visual_input_channels, self.line_height, self.line_width), device=device)

    def bind_model(self, model):
        """Bind environment-reading sequence methods on this evaluation instance."""
        values = {key: self.environment()[key] for key in ("PACK_VALID_WINDOWS", "FULL_IMAGE_NO_PADDING")}

        def scoped(method):
            @wraps(method)
            def call(*args, **kwargs):
                with evaluation_environment(values):
                    return method(*args, **kwargs)
            return call

        for name in ("encode_restoration_sequence", "encode_physical_window_sequence"):
            method = getattr(model.vit_encoder, name, None)
            if method is not None:
                setattr(model.vit_encoder, name, scoped(method))
        model.evaluation_contract = self
        return model

    def metadata(self, checkpoint_path=None, preprocessing="training", *, split_strategy=None):
        return {
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None,
            "architecture_family": self.architecture_family,
            "backend_variant": self.backend_variant,
            "visual_input_channels": self.visual_input_channels,
            "visual_grayscale": self.visual_grayscale,
            "normalization": {"mean": list(self.normalization_mean), "std": list(self.normalization_std)},
            "preprocessing_mode": preprocessing,
            "diagnostic_preprocessing_override": preprocessing != "training",
            "line_geometry_mode": self.line_geometry_mode,
            "line_width": self.line_width, "line_height": self.line_height,
            "preserve_aspect": self.preserve_aspect, "real_bbox_crop": self.real_bbox_crop,
            "foreground_crop": self.foreground_crop, "padding": self.padding,
            "real_binarize": self.real_binarize, "synthetic_binarize": self.synthetic_binarize,
            "autocontrast": self.autocontrast, "inversion": self.inversion,
            "window_size": self.window_size, "stride": self.stride,
            "physical_window_count": self.window_count,
            "pack_valid_windows": self.pack_valid_windows,
            "full_image_no_padding": self.full_image_no_padding,
            "resolved_split_strategy": split_strategy or self.split_strategy,
            "split_seed": self.split_seed,
            "split_seed_source": "checkpoint" if any(k in self.config for k in ("dataset_split_seed", "split_seed")) else "historical_training_default_42",
            "fallback_fields": list(self.fallback_fields),
        }


def resolve_evaluation_contract(config):
    """Resolve recorded values, then fixed historical defaults, never ambient env.

    Old RGB checkpoints omit channel fields; XML-gray geometry records imply
    one channel. The historical training split seed is 42 when not serialized.
    Every defaulted field is included in evaluation metadata.
    """
    c = dict(config)
    fallback = []

    def value(key, default):
        if key not in c:
            fallback.append(key)
        return c.get(key, default)

    mode = str(value("line_geometry_mode", "source-compatible-height"))
    if mode not in {"source-compatible-height", "crop-aspect-preserving-rgb",
                    "xml-bbox-gray-aspect-preserving", "xml-bbox-gray-full-resize"}:
        raise ValueError(f"Unsupported checkpoint geometry: {mode}")
    xml_gray = mode in {"xml-bbox-gray-aspect-preserving", "xml-bbox-gray-full-resize"}
    channels = int(value("visual_input_channels", 1 if flag(c.get("visual_grayscale", xml_gray)) else 3))
    grayscale = flag(value("visual_grayscale", channels == 1))
    if channels not in {1, 3} or grayscale != (channels == 1):
        raise ValueError(f"Inconsistent checkpoint color contract: channels={channels}, grayscale={grayscale}")
    if xml_gray and not grayscale:
        raise ValueError(f"Checkpoint geometry {mode!r} requires true grayscale")
    mean = tuple(value("normalization_mean", IMAGENET_GRAY_MEAN if grayscale else IMAGENET_MEAN))
    std = tuple(value("normalization_std", IMAGENET_GRAY_STD if grayscale else IMAGENET_STD))
    if len(mean) != channels or len(std) != channels or any(s <= 0 for s in std):
        raise ValueError("Checkpoint normalization must match the visual channel count")
    full = flag(value("full_image_no_padding", mode == "xml-bbox-gray-full-resize"))
    pack = flag(value("pack_valid_windows", False))
    preserve = flag(value("zero_shot_preserve_aspect", mode != "xml-bbox-gray-full-resize"))
    foreground = flag(value("zero_shot_foreground_crop", not xml_gray))
    bbox = flag(value("real_bbox_crop", xml_gray))
    if full and (pack or preserve):
        raise ValueError("Checkpoint full-image/no-padding contract conflicts with packing/aspect padding")
    window = int(value("window_size", 32))
    ratio = float(c.get("stride_ratio", 0.5))
    stride_default = {"no_overlap": window, "light_overlap": max(1, window // 2), "dense_overlap": max(1, window // 4)}.get(c.get("window_overlap_mode"), max(1, int(window * ratio)))
    stride = int(value("stride", c.get("window_stride", stride_default)))
    height = int(value("line_height", c.get("vit_input_height", 128)))
    width = int(value("line_width", 1024))
    if height <= 0 or width < window or window <= 0 or stride <= 0:
        raise ValueError("Invalid checkpoint line/window dimensions")
    variant = str(value("model_backend_variant", "physical_window" if c.get("context_input") == "same_physical_rgb_128x32_window" or c.get("model_backend") == "resnet18_physical_window_tinyvit_positive_dtw" else "resnet_token"))
    if variant not in {"resnet_token", "physical_window"}:
        raise ValueError(f"Unsupported checkpoint backend variant: {variant}")
    binary = flag(value("real_binarize", not (xml_gray or mode == "crop-aspect-preserving-rgb")))
    synthetic_binary = flag(value("synthetic_binarize", binary))
    autocontrast = flag(value("real_binarize_autocontrast", binary))
    inversion = flag(value("real_binarize_auto_invert", binary))
    all_pages = flag(value("real_all_page_lines", False))
    seed = int(value("dataset_split_seed", c.get("split_seed", 42)))
    return EvaluationContract(c, str(c.get("architecture_family", "")), variant,
        channels, grayscale, mean, std, height, width, mode, bbox, foreground,
        preserve, preserve, binary, synthetic_binary, autocontrast, inversion,
        window, stride, pack, full, all_pages, seed, tuple(fallback))


def visual_preflight(models):
    """The public launcher's checkpoint-driven smoke test, also usable on CPU."""
    contract = models.contract
    dummy = contract.dummy(models.device)
    with torch.inference_mode():
        contextual, local, grouped, valid = models.image_model(
            dummy, return_local=True, return_grouped=True, return_ink=True)
    if not all(torch.isfinite(t).all() for t in (contextual, local, grouped, valid)):
        raise RuntimeError("Non-finite checkpoint preflight output")
    if not contract.pack_valid_windows and contextual.shape[1] != contract.window_count:
        raise RuntimeError("Checkpoint preflight window count disagrees with its geometry")
    if contract.full_image_no_padding and not bool(valid.bool().all()):
        raise RuntimeError("Full-image checkpoint preflight masked valid windows")
    return {"input_shape": list(dummy.shape), "feature_shape": list(contextual.shape),
            "token_count": int(contextual.shape[1]), "valid_token_count": int(valid.bool().sum()),
            "finite": True}


def evaluation_metadata(models, checkpoint_path=None, preprocessing="training", dataset=None):
    checkpoint_path = checkpoint_path or models.checkpoint.get("_evaluation_path")
    metadata = models.contract.metadata(checkpoint_path, preprocessing)
    if dataset is not None:
        root = Path(dataset)
        root = root.parent if root.is_file() else root
        if (root / "images").is_dir():
            metadata["resolved_split_strategy"] = "synthetic_torch_60_20_20"
        elif models.contract.real_all_page_lines:
            from Evaluation.sw_dataset import _all_page_line_split_map
            with evaluation_environment(models.contract.environment()):
                _, lines = _all_page_line_split_map(
                    root, text_key=models.config.get("real_text_key", "text_original_path"),
                    seed=models.contract.split_seed)
            metadata["split_diagnostics"] = lines.evaluation_split_diagnostics
    return metadata
