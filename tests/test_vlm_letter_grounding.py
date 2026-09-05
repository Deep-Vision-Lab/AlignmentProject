import torch

from vlm_letter_grounding import (
    apply_branch_config,
    monotonic_letter_dtw_cost,
)


def test_letter_dtw_prefers_matching_order():
    # Three orthogonal letter prototypes and a visual sequence in the same order.
    prototypes = torch.eye(3, dtype=torch.float32)
    visual = prototypes.clone().requires_grad_(True)

    matching = monotonic_letter_dtw_cost(
        visual,
        prototypes,
        gamma=0.01,
        step_penalty=0.02,
    )
    reversed_cost = monotonic_letter_dtw_cost(
        visual,
        torch.flip(prototypes, dims=[0]),
        gamma=0.01,
        step_penalty=0.02,
    )

    assert matching < reversed_cost
    matching.backward()
    assert visual.grad is not None
    assert torch.isfinite(visual.grad).all()


def test_letter_dtw_allows_multiple_letters_in_one_window():
    # One broad visual window may depict strokes from two adjacent letters.  The
    # horizontal DTW transition must make T < L feasible and differentiable.
    visual = torch.tensor([[1.0, 1.0]], requires_grad=True)
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = monotonic_letter_dtw_cost(
        visual,
        text,
        gamma=0.01,
        step_penalty=0.02,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert visual.grad is not None


def test_branch_config_is_quality_first_and_keeps_10k():
    class P:
        pass

    apply_branch_config(P)
    assert P.num_samples == 10000
    assert P.vit_layers == 4
    assert P.vit_binarize_input is False
    assert P.num_negatives == 10
    assert P.span_dtw_active_negatives_per_sample == 0
    assert P.local_hard_negative_every_n_batches == 1
    assert P.local_hard_negative_max_samples_per_batch == 0
    assert P.image_pair_every_n_batches == 1
    assert P.image_pair_max_samples_per_batch == 0
    assert P.span_feature_cache_size == 0
