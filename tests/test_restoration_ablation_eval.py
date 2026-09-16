import numpy as np
import torch

from Evaluation.eval_restoration_ablation import (
    binary_iou,
    mask_token_indices,
    nearest_neighbor_hit_rates,
    predicted_columns_from_tokens,
)


def test_binary_iou_exact_match():
    left = np.zeros(16, dtype=bool)
    right = np.zeros(16, dtype=bool)
    left[4:10] = True
    right[4:10] = True
    assert binary_iou(left, right) == 1.0


def test_token_mask_roundtrip_keeps_region():
    mask = np.zeros(64, dtype=bool)
    mask[16:48] = True
    tokens = mask_token_indices(
        mask,
        3,
        window_size=32,
        stride=16,
        flipped=False,
    )
    predicted = predicted_columns_from_tokens(
        set(tokens),
        3,
        64,
        window_size=32,
        stride=16,
        flipped=False,
    )
    assert binary_iou(predicted, mask) > 0.45


def test_nearest_neighbor_hits_ground_truth_target_region():
    sim = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.2, 0.8, 0.3],
            [0.9, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    hit1, hit5 = nearest_neighbor_hit_rates(
        sim,
        source_gt=[0, 1],
        target_gt=[1],
        source_is_first=True,
        top_k=2,
    )
    assert hit1 == 1.0
    assert hit5 == 1.0


def test_flipped_token_order_maps_back_to_physical_region():
    mask = np.zeros(64, dtype=bool)
    mask[:32] = True
    tokens = mask_token_indices(
        mask,
        3,
        window_size=32,
        stride=16,
        flipped=True,
    )
    # Physical left-most window is last in the RTL-flipped sequence.
    assert 2 in tokens
