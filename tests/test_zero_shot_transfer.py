import random
from types import SimpleNamespace

import numpy as np
from PIL import Image

from Evaluation.balanced_sampling import balanced_group_split_pairs
from Evaluation.sw_core import ImagePair
from Evaluation.zero_shot_sw import balanced_batch_pairs, ink_aware_match_scores
from zero_shot_preprocessing import ManuscriptLinePreprocessor, foreground_crop


def _line_image(width=420, height=70):
    array = np.full((height, width), 255, dtype=np.uint8)
    array[20:48, 45:360] = 0
    array[15:25, 90:120] = 0
    return Image.fromarray(array, mode="L").convert("RGB")


def _ink_bbox(array):
    gray = np.asarray(array.convert("L"), dtype=np.uint8)
    ys, xs = np.nonzero(gray < 128)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def test_foreground_crop_removes_large_white_margins():
    image = Image.new("L", (800, 200), 255)
    image.paste(_line_image(420, 70).convert("L"), (210, 65))
    cropped = foreground_crop(image)
    assert cropped.width < 520
    assert cropped.height < 120


def test_zero_shot_preprocessor_preserves_aspect_and_outputs_binary():
    preprocessor = ManuscriptLinePreprocessor(
        training=False,
        augment=False,
        binarize=True,
        method="otsu",
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
    )
    source = _line_image()
    output = preprocessor(source)
    assert output.size == (1024, 128)
    assert set(np.unique(np.asarray(output)).tolist()).issubset({0, 255})

    source_box = _ink_bbox(source)
    output_box = _ink_bbox(output)
    source_ratio = (source_box[2] - source_box[0]) / (source_box[3] - source_box[1])
    output_ratio = (output_box[2] - output_box[0]) / (output_box[3] - output_box[1])
    assert abs(output_ratio - source_ratio) / source_ratio < 0.15


def test_augmented_synthetic_preprocessor_remains_model_compatible():
    random.seed(7)
    np.random.seed(7)
    preprocessor = ManuscriptLinePreprocessor(
        training=True,
        augment=True,
        binarize=True,
        method="random",
        threshold_jitter=24,
        preserve_aspect=True,
        crop_foreground=True,
        augment_probability=1.0,
        clean_probability=0.0,
    )
    output = preprocessor(_line_image())
    assert output.size == (1024, 128)
    assert set(np.unique(np.asarray(output)).tolist()).issubset({0, 255})
    assert np.asarray(output.convert("L")).mean() > 127.5


def test_ink_aware_scores_penalize_blank_matches(monkeypatch):
    monkeypatch.setenv("SW_INK_AWARE", "1")
    monkeypatch.setenv("SW_MIN_INK", "0.02")
    monkeypatch.setenv("SW_BLANK_BLANK_SCORE", "-0.20")
    monkeypatch.setenv("SW_BLANK_INK_SCORE", "-0.50")
    scores = np.asarray([[0.8, 0.7], [0.6, 0.9]], dtype=np.float32)
    adjusted = ink_aware_match_scores(
        scores,
        ink1=np.asarray([0.0, 0.20], dtype=np.float32),
        ink2=np.asarray([0.0, 0.30], dtype=np.float32),
    )
    assert adjusted[0, 0] == np.float32(-0.20)
    assert adjusted[0, 1] == np.float32(-0.50)
    assert adjusted[1, 0] == np.float32(-0.50)
    assert adjusted[1, 1] == np.float32(0.9)


def _pair(index, pair_id):
    return ImagePair(index=index, image1=None, image2=None, pair_id=pair_id)


def test_balanced_group_split_keeps_groups_disjoint_and_diverse():
    pairs = []
    index = 1
    for group, size in enumerate((12, 10, 8, 6, 5, 4, 3, 2)):
        for _ in range(size):
            pairs.append(_pair(index, f"pair_{group}"))
            index += 1
    train, valid, test = balanced_group_split_pairs(
        pairs,
        seed=42,
        fallback=lambda values, _seed: (values, [], []),
    )
    train_ids = {pair.pair_id for pair in train}
    valid_ids = {pair.pair_id for pair in valid}
    test_ids = {pair.pair_id for pair in test}
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    assert len(valid_ids) >= 2
    assert len(test_ids) >= 2


def test_balanced_batch_round_robins_pair_ids(monkeypatch):
    monkeypatch.setenv("REAL_EVAL_BALANCED", "1")
    pairs = [
        _pair(1, "a"),
        _pair(2, "a"),
        _pair(3, "a"),
        _pair(4, "b"),
        _pair(5, "b"),
        _pair(6, "c"),
    ]
    args = SimpleNamespace(start_index=1, n_samples=5)
    selected = balanced_batch_pairs(args, pairs)
    assert [pair.pair_id for pair in selected[:3]] == ["a", "b", "c"]
