import torch

from vlm_restoration_positive_dtw import (
    StrokeRestorationDecoder,
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
