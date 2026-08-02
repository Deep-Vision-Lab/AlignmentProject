import torch

from direct_subword_data import window_overlap_weights
from direct_subword_loss import (
    interval_localization_loss,
    multi_positive_info_nce,
)


def test_window_overlap_respects_rtl_flip():
    physical = window_overlap_weights(
        16,
        48,
        num_windows=5,
        window_size=32,
        stride=16,
        use_flip=False,
    )
    rtl = window_overlap_weights(
        16,
        48,
        num_windows=5,
        window_size=32,
        stride=16,
        use_flip=True,
    )
    assert torch.equal(rtl, torch.flip(physical, dims=[0]))
    assert physical.tolist() == [16.0, 32.0, 16.0, 0.0, 0.0]


def test_multi_positive_infonce_accepts_duplicate_subwords():
    visual = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    text = visual.clone()
    loss, stats = multi_positive_info_nce(
        visual, text, ["كتب", "كتب", "قرأ"], temperature=0.1
    )
    assert torch.isfinite(loss)
    assert loss.item() < 0.01
    assert stats["positive_mask"].tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]


def test_interval_localization_prefers_overlapping_windows():
    windows = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    text = torch.tensor([1.0, 0.0])
    correct = interval_localization_loss(
        windows, text, torch.tensor([1.0, 1.0, 0.0]), temperature=0.1
    )
    wrong = interval_localization_loss(
        windows, text, torch.tensor([0.0, 0.0, 1.0]), temperature=0.1
    )
    assert correct.item() < wrong.item()
