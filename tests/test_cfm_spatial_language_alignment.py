import torch

from vlm_spatial_language_alignment import (
    LocalPreservingContextFusion,
    SpatialAffinityLanguageAdapter,
    _cfm_hard_monotonic_path,
    monotonic_letter_dtw_cost,
)


def test_spatial_affinity_is_local_and_normalized():
    torch.manual_seed(7)
    module = SpatialAffinityLanguageAdapter(
        16,
        radius=1,
        temperature=0.2,
        distance_penalty=0.1,
        initial_gate=0.15,
    )
    x = torch.randn(2, 6, 16)
    y, weights = module(x)
    assert y.shape == x.shape
    assert weights.shape == (2, 6, 6)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 6))
    for i in range(6):
        for j in range(6):
            if abs(i - j) > 1:
                assert float(
                    weights[:, i, j].detach().abs().max()
                ) < 1e-7


def test_context_fusion_keeps_one_token_per_location():
    torch.manual_seed(11)
    fusion = LocalPreservingContextFusion(16, initial_gate=0.25)
    local = torch.randn(3, 9, 16)
    context = torch.randn(3, 9, 16)
    fused = fusion(local, context)
    assert fused.shape == local.shape
    assert 0.0 < float(fusion.gate.detach()) < 1.0


def test_monotonic_letter_alignment_cost_and_path_are_valid():
    torch.manual_seed(13)
    visual = torch.randn(7, 12)
    text = torch.randn(4, 12)
    cost = monotonic_letter_dtw_cost(
        visual, text, gamma=0.05, step_penalty=0.02
    )
    assert cost.ndim == 0
    assert torch.isfinite(cost)

    path = _cfm_hard_monotonic_path(
        visual, text, step_penalty=0.02
    )
    assert path
    assert path[0] == (0, 0)
    assert path[-1] == (
        visual.shape[0] - 1,
        text.shape[0] - 1,
    )
    assert all(
        i2 >= i1 and j2 >= j1
        for (i1, j1), (i2, j2) in zip(path, path[1:])
    )
