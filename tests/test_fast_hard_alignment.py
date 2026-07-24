from types import SimpleNamespace

import torch

from fast_hard_alignment import hard_span_dtw_path_fast
from span_alignment_loss import hard_span_dtw_path


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


def _signature(path):
    return [
        (
            step["text_start"],
            step["text_end"],
            step["window_start"],
            step["window_end"],
            step["span_idx"],
            step["text"],
            step["is_blank"],
        )
        for step in path
    ]


def test_fast_decoder_matches_reference_with_blank(monkeypatch):
    monkeypatch.setenv("SPAN_USE_BLANK_TRANSITIONS", "1")
    monkeypatch.setenv("SPAN_BLANK_PENALTY", "0.10")
    monkeypatch.setenv("SPAN_EXTRA_WINDOWS_PER_CORE", "0")
    images = torch.eye(3, dtype=torch.float32)
    kwargs = dict(
        temperature=0.1,
        max_windows=1,
        window_count_penalty=0.0,
        include_blank_steps=True,
    )
    reference = hard_span_dtw_path(_encoding(), images, **kwargs)
    optimized = hard_span_dtw_path_fast(_encoding(), images, **kwargs)
    assert _signature(optimized) == _signature(reference)


def test_fast_decoder_filters_blank_for_training(monkeypatch):
    monkeypatch.setenv("SPAN_USE_BLANK_TRANSITIONS", "1")
    monkeypatch.setenv("SPAN_BLANK_PENALTY", "0.10")
    monkeypatch.setenv("SPAN_EXTRA_WINDOWS_PER_CORE", "0")
    path = hard_span_dtw_path_fast(
        _encoding(),
        torch.eye(3, dtype=torch.float32),
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
