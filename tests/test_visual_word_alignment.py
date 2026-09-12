import numpy as np
import torch

from Evaluation._eval_utils import ImageFeatures
from Evaluation.visual_word_alignment import (
    detect_visual_word_boxes,
    matched_word_pairs,
    pool_visual_words,
)


def _line_image():
    image = np.full((128, 320, 3), 255, dtype=np.uint8)
    # Physical left-to-right: word A has two pieces separated by a small
    # intra-word gap; word B is separated by a much larger inter-word gap.
    image[45:82, 35:75] = 0
    image[45:82, 81:112] = 0
    image[45:82, 155:220] = 0
    image[45:82, 226:255] = 0
    return image


def test_visual_word_detection_bridges_small_gaps_but_not_word_spaces(monkeypatch):
    monkeypatch.setenv("WORD_ALIGNMENT_MAX_INTRAWORD_GAP_PX", "10")
    boxes = detect_visual_word_boxes(_line_image(), use_flip=False)
    assert len(boxes) == 2
    assert boxes[0][0] <= 35
    assert boxes[0][1] < boxes[1][0]
    assert boxes[1][1] >= 255


def test_arabic_word_regions_follow_right_to_left_reading_order(monkeypatch):
    monkeypatch.setenv("WORD_ALIGNMENT_MAX_INTRAWORD_GAP_PX", "10")
    boxes = detect_visual_word_boxes(_line_image(), use_flip=True)
    assert len(boxes) == 2
    assert boxes[0][0] > boxes[1][0]


def test_pool_visual_words_returns_one_vector_per_detected_word(monkeypatch):
    monkeypatch.setenv("WORD_ALIGNMENT_MAX_INTRAWORD_GAP_PX", "10")
    count, dim = 19, 8
    base = torch.arange(count * dim, dtype=torch.float32).reshape(count, dim)
    features = ImageFeatures(
        contextual=torch.nn.functional.normalize(base + 1.0, dim=-1),
        local=torch.nn.functional.normalize(base + 2.0, dim=-1),
        grouped=torch.nn.functional.normalize(base + 3.0, dim=-1),
        ink=torch.ones(count),
        image_size=(320, 128),
    )
    pooled, regions = pool_visual_words(
        features,
        _line_image(),
        use_flip=False,
        window_size=32,
        stride=16,
    )
    assert len(regions) == 2
    assert pooled.local.shape == (2, dim)
    assert pooled.contextual.shape == (2, dim)
    assert all(region.window_indices for region in regions)


def test_supported_word_matches_are_whole_nw_word_cells():
    class Step:
        def __init__(self, i, j):
            self.index1 = i
            self.index2 = j

    class Result:
        steps = [Step(0, 0), Step(1, 1), Step(2, 2)]

    scores = np.asarray([[0.5, 0, 0], [0, -0.1, 0], [0, 0, 0.7]], dtype=np.float32)
    assert matched_word_pairs(Result(), scores, support_floor=0.0) == [(0, 0), (2, 2)]
