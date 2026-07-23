import os

import torch
import torch.nn.functional as F

import span_alignment_loss_legacy as _legacy
from span_alignment_loss_legacy import *  # noqa: F401,F403


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _alignment_embeddings(span_encoding):
    value = getattr(span_encoding, "context_embeddings", None)
    return value if value is not None else span_encoding.embeddings


def _per_span_window_limits(span_encoding, global_max, device):
    """Return a realistic maximum window count for every text span.

    A global maximum alone allowed one whitespace token or one short character
    span to absorb four neighboring windows. With 32-pixel windows and stride 16,
    that creates broad repeated SPACE/three-character labels. A non-space core of
    length L may use L plus one overlap window; a literal text space has its own
    smaller cap.
    """
    lengths = torch.as_tensor(
        getattr(span_encoding, "lengths", []), dtype=torch.long, device=device
    )
    if lengths.numel() == 0:
        return None
    extra = max(0, _env_int("SPAN_EXTRA_WINDOWS_PER_CORE", 1))
    limits = (lengths + extra).clamp(min=1, max=max(1, int(global_max)))
    spaces = getattr(span_encoding, "is_space", None)
    if spaces is not None:
        space_mask = torch.as_tensor(spaces, dtype=torch.bool, device=device)
        space_cap = max(
            1,
            min(
                int(global_max),
                _env_int("SPAN_SPACE_MAX_WINDOWS", 2),
            ),
        )
        limits = torch.where(space_mask, torch.full_like(limits, space_cap), limits)
    return limits


def _precompute_transition_costs(
    span_encoding,
    image_embeddings,
    temperature,
    max_windows_per_span,
    window_count_penalty,
):
    span_embeddings = F.normalize(
        _alignment_embeddings(span_encoding).float(), p=2, dim=-1
    )
    image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=-1)
    image_steps = image_embeddings.shape[0]
    max_windows = min(max_windows_per_span, image_steps)
    window_costs = (
        1.0 - torch.matmul(span_embeddings, image_embeddings.T)
    ) / temperature
    prefix = torch.cat(
        [
            torch.zeros(
                window_costs.shape[0],
                1,
                device=window_costs.device,
                dtype=window_costs.dtype,
            ),
            window_costs.cumsum(dim=1),
        ],
        dim=1,
    )
    limits = _per_span_window_limits(
        span_encoding, max_windows, window_costs.device
    )
    result = {}
    for count in range(1, max_windows + 1):
        means = (prefix[:, count:] - prefix[:, :-count]) / count
        costs = means + window_count_penalty * (count - 1)
        if limits is not None:
            invalid = limits < count
            if invalid.any():
                costs = costs.masked_fill(invalid.unsqueeze(1), float("inf"))
        result[count] = costs
    return result


# Methods defined in the legacy module resolve this global dynamically.
_legacy._precompute_transition_costs = _precompute_transition_costs


def hard_span_dtw_path(
    span_encoding,
    image_embeddings,
    temperature=contrastive_temperature,
    max_windows=max_windows_per_span,
    window_count_penalty=None,
):
    if window_count_penalty is None:
        window_count_penalty = _env_float(
            "SPAN_WINDOW_COUNT_PENALTY", 0.05
        )
    path = _legacy.hard_span_dtw_path(
        span_encoding,
        image_embeddings,
        temperature=temperature,
        max_windows=max_windows,
        window_count_penalty=window_count_penalty,
    )
    surfaces = getattr(span_encoding, "surface_texts", None)
    raw_texts = getattr(span_encoding, "raw_texts", None)
    spaces = getattr(span_encoding, "is_space", None)
    for step in path:
        index = int(step["span_idx"])
        step["text"] = span_encoding.texts[index]
        step["surface_text"] = (
            surfaces[index] if surfaces is not None else step["text"]
        )
        step["raw_text"] = (
            raw_texts[index] if raw_texts is not None else step["text"]
        )
        step["is_space"] = bool(spaces[index]) if spaces is not None else False
    return path


class SpanContrastiveSoftDTW(_legacy.SpanContrastiveSoftDTW):
    def __init__(
        self,
        gamma=contrastive_soft_dtw_gamma,
        margin=contrastive_margin,
        temperature=contrastive_temperature,
        max_windows_per_span=max_windows_per_span,
        window_count_penalty=None,
        negative_grad_mode="hardest",
        backend="torch",
    ):
        if window_count_penalty is None:
            window_count_penalty = _env_float(
                "SPAN_WINDOW_COUNT_PENALTY", 0.05
            )
        super().__init__(
            gamma=gamma,
            margin=margin,
            temperature=temperature,
            max_windows_per_span=max_windows_per_span,
            window_count_penalty=window_count_penalty,
            negative_grad_mode=negative_grad_mode,
            backend=backend,
        )
