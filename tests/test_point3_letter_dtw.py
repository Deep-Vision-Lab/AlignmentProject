import numpy as np

from Evaluation import eval_point3_hard_paths as point3


def test_letter_dtw_helpers_are_installed():
    assert callable(point3._letter_dtw_side)
    assert callable(point3._hard_letter_dtw_path)
    assert callable(point3.save_letter_dtw_overview)


def test_hard_letter_dtw_monotonic_and_complete():
    costs = np.asarray(
        [
            [0.1, 2.0, 3.0],
            [0.2, 0.1, 2.0],
            [1.5, 0.2, 0.1],
            [2.0, 1.0, 0.2],
        ],
        dtype=np.float32,
    )
    path, matrix = point3._hard_letter_dtw_path(
        costs,
        vertical_penalty=0.05,
        horizontal_penalty=0.30,
        position_prior_weight=0.15,
        disable_horizontal_when_feasible=True,
    )
    assert matrix.shape == (4, 3)
    assert path[0] == (0, 0)
    assert path[-1] == (3, 2)
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert (i1 - i0, j1 - j0) in {(1, 0), (1, 1), (0, 1)}
