from types import SimpleNamespace

import pytest

from training_optimizations import split_embeddings, validate_resolved_configuration


def _safe_config():
    return SimpleNamespace(
        max_text_span_chars=2,
        max_text_token_chars=2,
        max_windows_per_span=3,
        span_include_space_context=False,
        span_allow_character_space_surfaces=False,
    )


def test_safe_span_configuration_is_accepted():
    validate_resolved_configuration(_safe_config())


def test_unsafe_span_configuration_is_rejected(monkeypatch):
    config = _safe_config()
    config.max_text_span_chars = 3
    monkeypatch.delenv("ALLOW_UNSAFE_SPAN_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="MAX_TEXT_SPAN_CHARS"):
        validate_resolved_configuration(config)


def test_explicit_ablation_can_override_guardrail(monkeypatch):
    config = _safe_config()
    config.span_include_space_context = True
    monkeypatch.setenv("ALLOW_UNSAFE_SPAN_CONFIG", "1")
    validate_resolved_configuration(config)


def test_combined_visual_embeddings_split_without_copying_values():
    import torch

    combined = (
        torch.arange(24).view(4, 2, 3),
        torch.arange(24, 48).view(4, 2, 3),
        torch.arange(8).view(4, 2),
        torch.arange(48, 72).view(4, 2, 3),
    )
    split = split_embeddings(combined, 2)
    for original, (first, second) in zip(combined, split):
        torch.testing.assert_close(first, original[:2])
        torch.testing.assert_close(second, original[2:])
