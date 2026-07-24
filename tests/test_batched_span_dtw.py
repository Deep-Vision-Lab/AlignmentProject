from types import SimpleNamespace

import pytest
import torch

from span_alignment_loss import SpanContrastiveSoftDTW


def _encoding(offset=0.0):
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    if offset:
        vectors = vectors + offset
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return SimpleNamespace(
        embeddings=vectors,
        context_embeddings=vectors,
        starts=[0, 1, -1],
        lengths=[1, 1, 0],
        texts=["a", "b", "<BLANK>"],
        surface_texts=["a", "b", "<BLANK>"],
        raw_texts=["a", "b", "<BLANK>"],
        is_space=[False, False, False],
        is_blank=[False, False, True],
        blank_index=2,
        text_length=2,
        max_span_chars=1,
    )


def _images(requires_grad=False):
    basis = torch.eye(3, dtype=torch.float32)
    images = torch.stack(
        [
            # visible a, blank, visible b
            torch.stack([basis[0], basis[2], basis[1]]),
            torch.tensor(
                [
                    [0.95, 0.05, 0.0],
                    [0.0, 0.05, 0.95],
                    [0.05, 0.95, 0.0],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    return torch.nn.functional.normalize(images, dim=-1).requires_grad_(requires_grad)


def _criterion(backend):
    return SpanContrastiveSoftDTW(
        gamma=0.1,
        margin=1.0,
        temperature=0.1,
        max_windows_per_span=1,
        window_count_penalty=0.0,
        backend=backend,
    )


def test_torch_batch_matches_individual(monkeypatch):
    monkeypatch.setenv("SPAN_USE_BLANK_TRANSITIONS", "1")
    monkeypatch.setenv("SPAN_BLANK_PENALTY", "0.1")
    criterion = _criterion("torch")
    encodings = [_encoding(), _encoding(0.01)]
    images = _images(requires_grad=True)

    batched = criterion._span_dtw_costs(encodings, images)
    individual = torch.stack(
        [
            criterion._span_dtw_cost_torch(encoding, image)
            for encoding, image in zip(encodings, images)
        ]
    )
    torch.testing.assert_close(batched, individual, rtol=1e-5, atol=1e-5)
    batched.sum().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def test_jax_batch_matches_single_when_available(monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.setenv("SPAN_USE_BLANK_TRANSITIONS", "1")
    monkeypatch.setenv("SPAN_BLANK_PENALTY", "0.1")
    criterion = _criterion("jax")
    encodings = [_encoding(), _encoding(0.01)]
    images = _images(requires_grad=True)

    batched = criterion._span_dtw_costs(encodings, images)
    singles = torch.stack(
        [
            criterion._span_dtw_cost_jax(encoding, image)
            for encoding, image in zip(encodings, images)
        ]
    )
    torch.testing.assert_close(batched, singles, rtol=2e-4, atol=2e-4)
    batched.sum().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
