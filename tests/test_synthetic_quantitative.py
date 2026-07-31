import numpy as np

from Evaluation.synthetic_quantitative import (
    interval_metrics,
    localization_metrics,
    roc_auc,
    seeded_split_indices,
)


def test_seeded_split_is_complete_disjoint_and_60_20_20():
    train, valid, test = seeded_split_indices(100, seed=42)
    assert (len(train), len(valid), len(test)) == (60, 20, 20)
    assert len(set(train) & set(valid)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(valid) & set(test)) == 0
    assert sorted(train + valid + test) == list(range(100))


def test_interval_metrics_are_exact_for_identical_regions():
    metrics = interval_metrics((2, 8), (2, 8))
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["start_error"] == 0.0
    assert metrics["end_error"] == 0.0


def test_localization_respects_arabic_logical_flip():
    normal = localization_metrics(
        pred_start=2,
        pred_end=5,
        gt_pixels=(20, 60),
        n_windows=10,
        width=100,
        flipped=False,
    )
    flipped = localization_metrics(
        pred_start=4,
        pred_end=7,
        gt_pixels=(20, 60),
        n_windows=10,
        width=100,
        flipped=True,
    )
    assert normal["window_iou"] == 1.0
    assert flipped["window_iou"] == 1.0
    assert flipped["pixel_iou"] == 1.0


def test_roc_auc_extremes():
    assert np.isclose(roc_auc([2.0, 3.0], [-1.0, 0.0]), 1.0)
    assert np.isclose(roc_auc([-1.0, 0.0], [2.0, 3.0]), 0.0)
