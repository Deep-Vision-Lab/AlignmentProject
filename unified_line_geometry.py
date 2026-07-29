"""Install one line-image geometry for all training and evaluation paths.

Both synthetic and real manuscript lines are normalized to the same fixed canvas
and foreground-ink height.  The default output is width=1024, height=128, with
the measured ink occupying 72% of the canvas height.  Long lines retain the
selected vertical ink scale and are compressed only along the horizontal axis.
"""
from __future__ import annotations

import os
from typing import Any


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def resolved_geometry() -> dict[str, Any]:
    height = max(16, _integer("LINE_HEIGHT", 128))
    width = max(32, _integer("LINE_WIDTH", 1024))
    ink_ratio = min(0.95, max(0.20, _number("TARGET_INK_HEIGHT_RATIO", 0.72)))
    return {
        "line_height": height,
        "line_width": width,
        "target_ink_height_ratio": ink_ratio,
        "target_ink_height_pixels": int(round(height * ink_ratio)),
        "line_geometry_mode": "source-compatible-height",
    }


def _install_preprocessor_factory():
    geometry = resolved_geometry()

    # Keep old environment names working while exposing neutral names that apply
    # to synthetic, real, training, validation, test, and evaluation alike.
    os.environ.setdefault("ZERO_SHOT_PREPROCESS", "1")
    os.environ.setdefault("ZERO_SHOT_PRESERVE_ASPECT", "1")
    os.environ.setdefault("ZERO_SHOT_FOREGROUND_CROP", "1")
    os.environ.setdefault("ZERO_SHOT_SOURCE_GEOMETRY", "1")
    os.environ["ZERO_SHOT_TARGET_INK_HEIGHT_RATIO"] = str(
        geometry["target_ink_height_ratio"]
    )
    os.environ.setdefault("SYNTHETIC_BINARIZE", "1")
    os.environ.setdefault("REAL_BINARIZE", "1")
    os.environ.setdefault("REAL_BINARIZE_AUTO_INVERT", "1")
    os.environ.setdefault("REAL_BINARIZE_AUTOCONTRAST", "1")

    from zero_shot_geometry import install_source_compatible_geometry

    install_source_compatible_geometry()

    import zero_shot_preprocessing as preprocessing

    if not hasattr(preprocessing, "_unified_geometry_original_build_preprocessor"):
        preprocessing._unified_geometry_original_build_preprocessor = (
            preprocessing.build_preprocessor
        )

        def build_preprocessor(dataset_type: str, training: bool):
            current = resolved_geometry()
            factory = preprocessing._unified_geometry_original_build_preprocessor
            preprocessor = factory(dataset_type, training)
            preprocessor.size = (
                int(current["line_height"]),
                int(current["line_width"]),
            )
            preprocessor.target_ink_height_ratio = float(
                current["target_ink_height_ratio"]
            )
            return preprocessor

        preprocessing.build_preprocessor = build_preprocessor
        preprocessing._unified_line_geometry_installed = True

    return preprocessing, geometry


def install_training_geometry() -> dict[str, Any]:
    """Patch every training loader to use the shared line geometry."""
    preprocessing, geometry = _install_preprocessor_factory()

    import DataLoader as data_loader

    # Standard synthetic and real loaders use the same deterministic geometry for
    # validation/test.  ZERO_SHOT_PROFILE replaces the synthetic training view
    # with an augmented transform built from the same factory.
    data_loader.synthetic_transform = preprocessing.build_tensor_transform(
        "synthetic", training=False
    )
    data_loader.real_transform = preprocessing.build_tensor_transform(
        "real", training=False
    )

    # The opt-in augmented real loader has its own training transform. Replace it
    # without changing its pair stitching or post-binary ink augmentation logic.
    try:
        import AugmentedRealDataLoader as augmented_real
        from torchvision import transforms

        def train_real_transform():
            return transforms.Compose(
                [
                    preprocessing.build_preprocessor("real", training=True),
                    augmented_real.BinaryInkAugment.from_env(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        preprocessing.IMAGENET_MEAN,
                        preprocessing.IMAGENET_STD,
                    ),
                ]
            )

        augmented_real._train_real_transform = train_real_transform
    except ImportError:
        # Standard real training still uses data_loader.real_transform.
        pass

    return dict(geometry)


def install_evaluation_geometry() -> dict[str, Any]:
    """Install the identical geometry before evaluation modules are imported."""
    _preprocessing, geometry = _install_preprocessor_factory()
    return dict(geometry)
