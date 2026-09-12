import numpy as np
import torch

from Evaluation.analyze_restoration_line import (
    char_index,
    deterministic_char_codebook,
    hard_monotonic_dtw,
)


def test_hard_monotonic_dtw_follows_low_cost_diagonal():
    costs = np.asarray(
        [
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    path, value = hard_monotonic_dtw(costs, step_penalty=0.02)
    assert path == [(0, 0), (1, 1), (2, 2)]
    assert abs(value) < 1e-8


def test_hard_monotonic_dtw_allows_multiple_windows_for_one_letter():
    costs = np.asarray(
        [
            [0.0, 2.0],
            [0.1, 2.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    path, _ = hard_monotonic_dtw(costs, step_penalty=0.02)
    assert (0, 0) in path
    assert (1, 0) in path
    assert (2, 1) in path


def test_deterministic_codebook_matches_expected_shape_and_seed():
    config = {
        "vector_size": 16,
        "letter_codebook_vocab_size": 64,
        "letter_codebook_seed": 1234,
    }
    first = deterministic_char_codebook(config, torch.device("cpu"))
    second = deterministic_char_codebook(config, torch.device("cpu"))
    assert first.shape == (64, 16)
    assert torch.allclose(first, second)
    assert char_index("ا", 64) == (ord("ا") % 62) + 2
