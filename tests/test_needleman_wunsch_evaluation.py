import numpy as np
import torch

from Evaluation._eval_utils import needleman_wunsch, normalize_word


def test_nw_recovers_diagonal_alignment():
    similarity = torch.tensor(
        [
            [0.9, -0.2, -0.3],
            [-0.1, 0.8, -0.2],
            [-0.4, -0.1, 0.95],
        ],
        dtype=torch.float32,
    )
    result = needleman_wunsch(similarity, gap_penalty=-0.3)
    assert result.pairs == [(0, 0), (1, 1), (2, 2)]
    assert result.normalized_score > 0


def test_nw_can_insert_gap():
    similarity = np.asarray(
        [[0.9, -0.8, -0.8], [-0.8, -0.8, 0.9]],
        dtype=np.float32,
    )
    result = needleman_wunsch(similarity, gap_penalty=-0.2)
    assert (0, 0) in result.pairs
    assert (1, 2) in result.pairs
    assert any(step.operation.startswith("gap") for step in result.steps)


def test_arabic_word_normalization_removes_tatweel_and_diacritics():
    assert normalize_word("كِتـاب") == normalize_word("كتاب")
