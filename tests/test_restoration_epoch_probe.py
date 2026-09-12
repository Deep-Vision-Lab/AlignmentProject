import numpy as np
import torch

from restoration_epoch_probe import _hard_dtw, _matrix_correlation


def test_epoch_probe_hard_dtw_tracks_diagonal():
    costs = np.asarray(
        [
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    path, cost = _hard_dtw(costs, 0.02)
    assert path == [(0, 0), (1, 1), (2, 2)]
    assert abs(cost) < 1e-8


def test_epoch_probe_similarity_correlation_detects_matching_geometry():
    matrix = torch.tensor(
        [
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )
    assert _matrix_correlation(matrix, matrix) > 0.999
    reversed_geometry = torch.tensor(
        [
            [1.0, 0.1, 0.8],
            [0.1, 1.0, 0.2],
            [0.8, 0.2, 1.0],
        ]
    )
    assert _matrix_correlation(matrix, reversed_geometry) < 0.0
