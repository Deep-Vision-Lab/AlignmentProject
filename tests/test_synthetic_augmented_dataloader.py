import random

import numpy as np
from PIL import Image

from SyntheticAugmentedDataLoader import (
    SyntheticScaleTranslateAugment,
    _split_indices,
)


def test_split_uses_exact_disjoint_training_count():
    train, valid, test = _split_indices(
        total=5000,
        train_count=3000,
        valid_requested=500,
        test_requested=500,
        seed=17,
    )

    assert len(train) == 3000
    assert len(valid) == 500
    assert len(test) == 500
    assert not (set(train) & set(valid))
    assert not (set(train) & set(test))
    assert not (set(valid) & set(test))


def test_augmentation_keeps_shape_and_border_polarity():
    image = Image.new("RGB", (1024, 128), (255, 255, 255))
    pixels = np.asarray(image).copy()
    pixels[48:80, 250:780] = 0
    image = Image.fromarray(pixels)

    random.seed(5)
    augment = SyntheticScaleTranslateAugment(
        scale_min=0.85,
        scale_max=0.85,
        translate_pct=0.05,
        probability=1.0,
    )
    result = augment(image)
    result_array = np.asarray(result)

    assert result.size == (1024, 128)
    assert np.median(result_array[0]) == 255
    assert result_array.min() == 0
