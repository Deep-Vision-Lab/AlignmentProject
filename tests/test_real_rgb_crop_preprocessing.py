import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from zero_shot_preprocessing import (
    aspect_preserving_pad_with_metadata,
    foreground_crop_with_metadata,
    foreground_detection_mask_with_metadata,
    build_preprocessor,
)


def test_robust_crop_ignores_isolated_border_noise_and_preserves_rgb(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_CROP_MODE", "robust_projection")
    monkeypatch.setenv("ZERO_SHOT_CROP_MIN_CONTRAST", "12")

    image = Image.new("RGB", (600, 140), (232, 225, 205))
    draw = ImageDraw.Draw(image)
    # Main handwriting-like dark band.
    draw.rectangle((120, 50, 470, 82), fill=(55, 43, 35))
    draw.ellipse((170, 38, 178, 46), fill=(55, 43, 35))
    draw.ellipse((380, 88, 388, 96), fill=(55, 43, 35))
    # Sparse scan noise at all four borders that should NOT force full crop.
    for x, y in [(0, 0), (599, 0), (0, 139), (599, 139), (4, 70), (595, 20)]:
        draw.point((x, y), fill=(20, 20, 20))

    mask, detector = foreground_detection_mask_with_metadata(image)
    cropped, meta = foreground_crop_with_metadata(image)

    assert detector["crop_detector"] == "adaptive-local-background-projection"
    assert int(mask.sum()) > 0
    assert meta["crop_mode"] == "robust_projection"
    assert meta["crop_left"] > 0
    assert meta["crop_right"] < image.width
    assert meta["crop_top"] > 0
    assert meta["crop_bottom"] < image.height

    # Crop is taken from the untouched RGB source, not a thresholded copy.
    source = np.asarray(image)
    got = np.asarray(cropped)
    expected = source[
        meta["crop_top"] : meta["crop_bottom"],
        meta["crop_left"] : meta["crop_right"],
    ]
    assert np.array_equal(got, expected)


def test_rgb_crop_to_model_canvas_keeps_continuous_intensity(monkeypatch):
    monkeypatch.setenv("ZERO_SHOT_CROP_MODE", "robust_projection")
    image = Image.new("RGB", (500, 120), (240, 232, 214))
    draw = ImageDraw.Draw(image)
    draw.rectangle((90, 40, 410, 85), fill=(72, 55, 41))

    cropped, _ = foreground_crop_with_metadata(image)
    model_input, meta = aspect_preserving_pad_with_metadata(
        cropped,
        size=(128, 1024),
        target_ink_height_ratio=0.72,
        horizontal_jitter=0.0,
    )

    assert model_input.size == (1024, 128)
    values = np.asarray(model_input)
    # RGB path must contain interpolated/original non-binary values.
    unique = np.unique(values.reshape(-1, 3), axis=0)
    assert len(unique) > 2
    assert meta["canvas_width"] == 1024
    assert meta["canvas_height"] == 128


def test_real_training_preprocessor_outputs_true_grayscale(monkeypatch):
    monkeypatch.setenv("VISUAL_GRAYSCALE", "1")
    monkeypatch.setenv("REAL_GRAYSCALE", "1")
    monkeypatch.setenv("REAL_SYNTHETIC_STYLE", "0")
    monkeypatch.setenv("REAL_BINARIZE", "0")
    monkeypatch.setenv("REAL_BINARIZE_AUTOCONTRAST", "0")
    monkeypatch.setenv("ZERO_SHOT_PREPROCESS", "1")
    monkeypatch.setenv("ZERO_SHOT_FOREGROUND_CROP", "0")
    monkeypatch.setenv("ZERO_SHOT_PRESERVE_ASPECT", "1")

    image = Image.new("RGB", (520, 90), (238, 228, 207))
    draw = ImageDraw.Draw(image)
    draw.rectangle((95, 24, 430, 68), fill=(63, 48, 37))
    draw.rectangle((205, 33, 280, 56), fill=(108, 82, 60))

    preprocessor = build_preprocessor("real", training=False)
    model_input, meta = preprocessor.preprocess_with_metadata(image)

    assert preprocessor.binarize is False
    assert preprocessor.crop_foreground is False
    assert preprocessor.preserve_aspect is True
    assert preprocessor.grayscale is True
    assert meta["binarize"] is False
    assert meta["grayscale"] is True
    assert model_input.mode == "L"
    assert model_input.size == (1024, 128)

    values = np.asarray(model_input)
    assert values.ndim == 2
    assert len(np.unique(values)) > 3
    assert np.any(values == 255)
