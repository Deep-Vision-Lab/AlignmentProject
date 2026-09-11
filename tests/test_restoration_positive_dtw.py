import torch

from types import SimpleNamespace

from embeddingModel import EmbeddingModel
from vlm_restoration_positive_dtw import (
    StrokeRestorationDecoder,
    _clean_letters,
    attach_restoration_dtw_stages,
    positive_monotonic_letter_dtw_cost,
)


def test_restoration_decoder_shape_and_range():
    decoder = StrokeRestorationDecoder(
        dim=128,
        output_height=128,
        output_width=32,
        channels=32,
    )
    tokens = torch.randn(2, 7, 128)
    restored = decoder(tokens)
    assert restored.shape == (2, 7, 1, 128, 32)
    assert torch.isfinite(restored).all()
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_positive_dtw_is_differentiable():
    visual = torch.randn(9, 128, requires_grad=True)
    target = torch.randn(4, 128)
    loss = positive_monotonic_letter_dtw_cost(
        visual,
        target,
        gamma=0.05,
        step_penalty=0.02,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert visual.grad is not None
    assert torch.isfinite(visual.grad).all()


def test_matching_order_costs_less_than_reversed_order():
    generator = torch.Generator().manual_seed(7)
    letters = torch.randn(4, 128, generator=generator)
    letters = torch.nn.functional.normalize(letters, p=2, dim=-1)

    # Two windows per letter in the correct monotonic order.
    visual = torch.repeat_interleave(letters, repeats=2, dim=0)
    correct = positive_monotonic_letter_dtw_cost(
        visual, letters, gamma=0.01, step_penalty=0.02
    )
    reversed_target = positive_monotonic_letter_dtw_cost(
        visual, torch.flip(letters, dims=[0]), gamma=0.01, step_penalty=0.02
    )
    assert float(correct) < float(reversed_target)


def test_unicode_arabic_cleaning_keeps_wasla_and_decomposes_ligature():
    cleaned = _clean_letters(" ٱلْﻻ! ")
    assert cleaned == ["ٱ", "ل", "ل", "ا"]


def test_full_model_training_bundle_has_one_token_per_window():
    model = EmbeddingModel(
        window_size=32,
        stride=16,
        vector_size=128,
        device="cpu",
        use_flip=True,
        input_height=128,
        vit_layers=1,
        vit_heads=4,
        vit_mlp_dim=256,
        vit_dropout=0.0,
        vit_max_tokens=64,
        vit_position_base_tokens=7,
        vit_binarize_input=False,
    )
    config = SimpleNamespace(
        restoration_decoder_channels=16,
        restoration_contrast_scale=0.15,
    )
    model = attach_restoration_dtw_stages(model, config)
    image = torch.randn(1, 3, 128, 128)
    bundle = model(image, return_training_bundle=True)

    assert bundle["primitive"].shape == (1, 7, 128)
    assert bundle["semantic"].shape == (1, 7, 128)
    assert bundle["ink"].shape == (1, 7)
    assert bundle["restoration"].shape == (1, 7, 1, 128, 32)
    assert bundle["restoration_target"].shape == (1, 7, 1, 128, 32)
