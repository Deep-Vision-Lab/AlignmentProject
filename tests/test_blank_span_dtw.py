import os
from types import SimpleNamespace

import torch

from span_alignment_loss import SpanContrastiveSoftDTW, hard_span_dtw_path


def _encoding():
    vectors = torch.eye(3, dtype=torch.float32)
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


def _image_windows(requires_grad=False):
    # visible a, empty/background, visible b
    return torch.eye(3, dtype=torch.float32).requires_grad_(requires_grad)


def _configure_blank(monkeypatch):
    monkeypatch.setenv("SPAN_USE_BLANK_TRANSITIONS", "1")
    monkeypatch.setenv("SPAN_BLANK_PENALTY", "0.10")
    monkeypatch.setenv("SPAN_EXTRA_WINDOWS_PER_CORE", "0")
    monkeypatch.setenv("SPAN_SPACE_MAX_WINDOWS", "1")


def test_hard_path_blank_consumes_window_without_text(monkeypatch):
    _configure_blank(monkeypatch)
    path = hard_span_dtw_path(
        _encoding(),
        _image_windows(),
        temperature=0.1,
        max_windows=1,
        window_count_penalty=0.0,
        include_blank_steps=True,
    )

    assert [step["text"] for step in path] == ["a", "<BLANK>", "b"]
    blank = path[1]
    assert blank["is_blank"] is True
    assert blank["text_start"] == blank["text_end"] == 1
    assert blank["window_start"] == 1
    assert blank["window_end"] == 2


def test_training_path_filters_blank_steps(monkeypatch):
    _configure_blank(monkeypatch)
    path = hard_span_dtw_path(
        _encoding(),
        _image_windows(),
        temperature=0.1,
        max_windows=1,
        window_count_penalty=0.0,
        include_blank_steps=False,
    )

    assert [step["text"] for step in path] == ["a", "b"]
    assert [(step["window_start"], step["window_end"]) for step in path] == [
        (0, 1),
        (2, 3),
    ]


def test_torch_soft_cost_backpropagates_through_blank_path(monkeypatch):
    _configure_blank(monkeypatch)
    image = _image_windows(requires_grad=True)
    criterion = SpanContrastiveSoftDTW(
        gamma=0.1,
        margin=1.0,
        temperature=0.1,
        max_windows_per_span=1,
        window_count_penalty=0.0,
        backend="torch",
    )

    cost = criterion._span_dtw_cost(_encoding(), image)
    assert torch.isfinite(cost)
    cost.backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
