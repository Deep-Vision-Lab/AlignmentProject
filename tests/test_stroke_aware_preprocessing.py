import os
import pickle
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from stroke_aware_preprocessing import (
    StrokeAwareLineTransform,
    _shift_vertical,
    apply_checkpoint_preprocessing_config,
    preprocessing_config,
)


def _sample_line(width=128, height=32):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((12, height // 2, width - 12, height // 2), fill="black", width=2)
    draw.ellipse((35, 6, 38, 9), fill="black")
    return image


def test_stroke_aware_transform_returns_three_distinct_channels(monkeypatch):
    monkeypatch.setenv("DIRECT_SUBWORD_STROKE_INPUT", "1")
    monkeypatch.setenv("DIRECT_SUBWORD_STROKE_AUGMENT", "0")
    transform = StrokeAwareLineTransform(size=(32, 128), training=False)

    tensor = transform(_sample_line())

    assert tensor.shape == (3, 32, 128)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0
    assert not torch.allclose(tensor[0], tensor[1])
    assert not torch.allclose(tensor[0], tensor[2])


def test_training_augmentation_preserves_tensor_geometry(monkeypatch):
    monkeypatch.setenv("DIRECT_SUBWORD_STROKE_INPUT", "1")
    monkeypatch.setenv("DIRECT_SUBWORD_STROKE_AUGMENT", "1")
    monkeypatch.setenv("STROKE_AUG_MORPH_PROB", "1")
    monkeypatch.setenv("STROKE_AUG_GAMMA_PROB", "1")
    monkeypatch.setenv("STROKE_AUG_BLUR_PROB", "1")
    monkeypatch.setenv("STROKE_AUG_NOISE_PROB", "1")
    monkeypatch.setenv("STROKE_AUG_VERTICAL_SHIFT_PROB", "1")
    random.seed(7)
    np.random.seed(7)

    tensor = StrokeAwareLineTransform(size=(32, 128), training=True)(_sample_line())

    assert tensor.shape == (3, 32, 128)
    assert torch.isfinite(tensor).all()


def test_vertical_shift_never_changes_horizontal_projection_location():
    ink = np.zeros((16, 40), dtype=np.float32)
    ink[5:8, 11:19] = 1.0

    shifted = _shift_vertical(ink, 4)

    assert shifted.shape == ink.shape
    assert np.flatnonzero(shifted.max(axis=0)).tolist() == list(range(11, 19))


def test_transform_is_spawn_picklable():
    pickle.dumps(StrokeAwareLineTransform(training=True))


def test_checkpoint_config_enables_channels_but_disables_eval_augmentation(monkeypatch):
    monkeypatch.delenv("DIRECT_SUBWORD_STROKE_INPUT", raising=False)
    monkeypatch.delenv("DIRECT_SUBWORD_STROKE_AUGMENT", raising=False)

    apply_checkpoint_preprocessing_config(
        {
            "stroke_aware_input": True,
            "stroke_aware_augment": True,
            "stroke_aware_distance_clip": 6.0,
        }
    )

    assert os.environ["DIRECT_SUBWORD_STROKE_INPUT"] == "1"
    assert os.environ["DIRECT_SUBWORD_STROKE_AUGMENT"] == "0"
    assert os.environ["STROKE_AWARE_DISTANCE_CLIP"] == "6.0"


def test_preprocessing_config_records_reproducible_defaults(monkeypatch):
    monkeypatch.setenv("DIRECT_SUBWORD_STROKE_INPUT", "1")
    config = preprocessing_config(default_enabled=False)

    assert config["stroke_aware_input"] is True
    assert config["stroke_aware_channels"] == (
        "soft_ink,distance_proximity,sobel_magnitude"
    )
    assert config["stroke_aware_vertical_shift_max"] == 4
