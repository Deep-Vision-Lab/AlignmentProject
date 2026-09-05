import torch

from vlm_pair_cross_attention import (
    SymmetricPairCrossAttention,
    apply_cross_attention_config,
)


def test_bidirectional_cross_attention_shapes_and_gradients():
    module = SymmetricPairCrossAttention(
        16,
        num_heads=4,
        dropout=0.0,
        ff_multiplier=2,
        initial_gate=0.2,
    )
    left = torch.randn(2, 7, 16, requires_grad=True)
    right = torch.randn(2, 9, 16, requires_grad=True)
    ink_left = torch.ones(2, 7)
    ink_right = torch.ones(2, 9)

    fused_left, fused_right, attn_lr, attn_rl = module(
        left,
        right,
        ink1=ink_left,
        ink2=ink_right,
        return_weights=True,
    )

    assert fused_left.shape == left.shape
    assert fused_right.shape == right.shape
    assert attn_lr.shape == (2, 4, 7, 9)
    assert attn_rl.shape == (2, 4, 9, 7)

    loss = fused_left.square().mean() + fused_right.square().mean()
    loss.backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
    trainable_grads = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None for gradient in trainable_grads)
    assert all(torch.isfinite(gradient).all() for gradient in trainable_grads)


def test_blank_query_windows_remain_independent():
    module = SymmetricPairCrossAttention(
        8,
        num_heads=2,
        dropout=0.0,
        initial_gate=0.5,
    )
    left = torch.randn(1, 4, 8)
    right = torch.randn(1, 5, 8)
    ink_left = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    ink_right = torch.ones(1, 5)

    fused_left, _fused_right, _a, _b = module(
        left,
        right,
        ink1=ink_left,
        ink2=ink_right,
        min_ink=0.01,
        return_weights=False,
    )

    assert torch.allclose(fused_left[:, 1], left[:, 1])
    assert torch.allclose(fused_left[:, 3], left[:, 3])


def test_all_memory_masked_is_safe():
    module = SymmetricPairCrossAttention(8, num_heads=2, dropout=0.0)
    left = torch.randn(1, 3, 8)
    right = torch.randn(1, 4, 8)
    ink_left = torch.ones(1, 3)
    ink_right = torch.zeros(1, 4)

    fused_left, fused_right, _a, _b = module(
        left,
        right,
        ink1=ink_left,
        ink2=ink_right,
        return_weights=False,
    )
    assert torch.isfinite(fused_left).all()
    assert torch.isfinite(fused_right).all()


def test_cross_attention_config_is_quality_first():
    class P:
        vector_size = 128

    apply_cross_attention_config(P)
    assert P.experiment_name == "vit_vlm_letter_depiction_cross_attention"
    assert P.cross_attention_enabled is True
    assert P.cross_attention_heads == 4
    assert P.local_hard_negative_every_n_batches == 1
    assert P.local_hard_negative_max_samples_per_batch == 0
    assert P.image_pair_every_n_batches == 1
    assert P.image_pair_max_samples_per_batch == 0
