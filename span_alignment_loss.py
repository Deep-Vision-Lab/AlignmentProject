import os

import torch
import torch.nn.functional as F

import span_alignment_loss_legacy as _legacy
from span_alignment_loss_legacy import *  # noqa: F401,F403


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _blank_available(span_encoding):
    blank_index = getattr(span_encoding, "blank_index", None)
    return (
        _env_flag("SPAN_USE_BLANK_TRANSITIONS", True)
        and blank_index is not None
        and 0 <= int(blank_index) < int(_alignment_embeddings(span_encoding).shape[0])
    )


def _blank_transition_costs(span_encoding, image_embeddings, temperature):
    """Cost for consuming one image window without consuming transcript text."""
    if not _blank_available(span_encoding):
        return None
    blank_index = int(span_encoding.blank_index)
    span_embeddings = F.normalize(
        _alignment_embeddings(span_encoding).float(), p=2, dim=-1
    )
    image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=-1)
    blank_vector = span_embeddings[blank_index]
    similarities = torch.matmul(image_embeddings, blank_vector)
    penalty = _env_float("SPAN_BLANK_PENALTY", 0.35)
    return (1.0 - similarities) / float(temperature) + float(penalty)


def _per_span_window_limits(span_encoding, global_max, device):
    """Return a realistic maximum window count for every text span."""
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


def _dense_transition_costs(
    span_encoding,
    image_embeddings,
    temperature,
    max_windows_per_span,
    window_count_penalty,
    text_steps_padded=None,
):
    actual_text_steps = int(span_encoding.text_length)
    dense_text_steps = int(text_steps_padded or actual_text_steps)
    if dense_text_steps < actual_text_steps:
        raise ValueError(
            f"text_steps_padded={dense_text_steps} is smaller than actual text length "
            f"{actual_text_steps}."
        )
    image_steps = int(image_embeddings.shape[0])
    max_span_chars = max(
        int(getattr(span_encoding, "max_span_chars", 0)),
        max((int(length) for length in span_encoding.lengths), default=0),
    )
    transition_costs = _precompute_transition_costs(
        span_encoding,
        image_embeddings,
        temperature,
        max_windows_per_span,
        window_count_penalty,
    )
    dense = image_embeddings.new_full(
        (
            max_span_chars + 1,
            max_windows_per_span + 1,
            dense_text_steps + 1,
            image_steps + 1,
        ),
        float("inf"),
    )

    for span_idx, (start, span_len) in enumerate(
        zip(span_encoding.starts, span_encoding.lengths)
    ):
        start = int(start)
        span_len = int(span_len)
        if span_len <= 0 or start < 0:
            continue
        if span_len > max_span_chars or start + span_len > actual_text_steps:
            continue
        for window_count, costs in transition_costs.items():
            if costs.shape[1] == 0:
                continue
            dense[span_len, window_count, start, : costs.shape[1]] = costs[span_idx]

    blank_costs = _blank_transition_costs(
        span_encoding, image_embeddings, temperature
    )
    if blank_costs is not None and max_windows_per_span >= 1:
        for text_position in range(actual_text_steps + 1):
            dense[0, 1, text_position, :image_steps] = blank_costs
    return dense


# Methods defined in the legacy module resolve these globals dynamically.
_legacy._precompute_transition_costs = _precompute_transition_costs
_legacy._dense_transition_costs = _dense_transition_costs


def hard_span_dtw_path(
    span_encoding,
    image_embeddings,
    temperature=contrastive_temperature,
    max_windows=max_windows_per_span,
    window_count_penalty=None,
    include_blank_steps=None,
):
    if window_count_penalty is None:
        window_count_penalty = _env_float(
            "SPAN_WINDOW_COUNT_PENALTY", 0.05
        )
    if include_blank_steps is None:
        include_blank_steps = _env_flag("SPAN_RETURN_BLANK_STEPS", False)

    span_lookup = _legacy.span_index_by_start_and_length(span_encoding)
    text_steps = int(span_encoding.text_length)
    image_steps = int(image_embeddings.shape[0])
    min_required = _legacy._min_required_spans(span_encoding)
    if min_required == float("inf") or image_steps < min_required:
        raise ValueError(
            _legacy._span_no_path_message(
                span_encoding,
                image_steps,
                max_windows,
                min_required,
                reason="too_few_windows",
            )
        )
    if not _blank_available(span_encoding):
        max_possible_windows = text_steps * int(max_windows)
        if image_steps > max_possible_windows:
            raise ValueError(
                _legacy._span_no_path_message(
                    span_encoding,
                    image_steps,
                    max_windows,
                    min_required,
                    reason="too_many_windows",
                )
            )

    transition_costs = _precompute_transition_costs(
        span_encoding,
        image_embeddings,
        temperature,
        max_windows,
        window_count_penalty,
    )
    blank_costs = _blank_transition_costs(
        span_encoding, image_embeddings, temperature
    )
    blank_index = int(getattr(span_encoding, "blank_index", -1))

    dp = torch.full((text_steps + 1, image_steps + 1), float("inf"))
    back = [[None for _ in range(image_steps + 1)] for _ in range(text_steps + 1)]
    dp[0, 0] = 0.0

    for i in range(text_steps + 1):
        for j in range(image_steps + 1):
            if not torch.isfinite(dp[i, j]):
                continue

            if blank_costs is not None and j < image_steps:
                candidate = dp[i, j] + blank_costs[j].detach().cpu()
                if candidate < dp[i, j + 1]:
                    dp[i, j + 1] = candidate
                    back[i][j + 1] = (i, j, blank_index, True)

            for span_len in range(1, text_steps - i + 1):
                span_idx = span_lookup.get((i, span_len))
                if span_idx is None:
                    continue
                for window_count in range(
                    1, min(int(max_windows), image_steps - j) + 1
                ):
                    cost = transition_costs[window_count][span_idx, j]
                    if not torch.isfinite(cost):
                        continue
                    next_i = i + span_len
                    next_j = j + window_count
                    candidate = dp[i, j] + cost.detach().cpu()
                    if candidate < dp[next_i, next_j]:
                        dp[next_i, next_j] = candidate
                        back[next_i][next_j] = (i, j, span_idx, False)

    if back[text_steps][image_steps] is None:
        raise ValueError(
            _legacy._span_no_path_message(
                span_encoding,
                image_steps,
                max_windows,
                min_required,
                reason="generic",
            )
        )

    path = []
    i, j = text_steps, image_steps
    while back[i][j] is not None:
        prev_i, prev_j, span_idx, is_blank = back[i][j]
        path.append(
            {
                "text_start": prev_i,
                "text_end": i,
                "window_start": prev_j,
                "window_end": j,
                "span_idx": int(span_idx),
                "is_blank": bool(is_blank),
            }
        )
        i, j = prev_i, prev_j
    path.reverse()

    surfaces = getattr(span_encoding, "surface_texts", None)
    raw_texts = getattr(span_encoding, "raw_texts", None)
    spaces = getattr(span_encoding, "is_space", None)
    blanks = getattr(span_encoding, "is_blank", None)
    enriched = []
    for step in path:
        index = int(step["span_idx"])
        is_blank = bool(step.get("is_blank", False))
        step["text"] = span_encoding.texts[index]
        step["surface_text"] = (
            surfaces[index] if surfaces is not None else step["text"]
        )
        step["raw_text"] = (
            raw_texts[index] if raw_texts is not None else step["text"]
        )
        step["is_space"] = bool(spaces[index]) if spaces is not None else False
        if blanks is not None:
            step["is_blank"] = bool(blanks[index]) or is_blank
        if include_blank_steps or not step["is_blank"]:
            enriched.append(step)
    return enriched


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

    def _check_path_feasible(self, span_encoding, image_steps):
        min_required = _legacy._min_required_spans(span_encoding)
        if min_required == float("inf") or image_steps < min_required:
            raise ValueError(
                _legacy._span_no_path_message(
                    span_encoding,
                    image_steps,
                    self.max_windows_per_span,
                    min_required,
                    reason="too_few_windows",
                )
            )
        if not _blank_available(span_encoding):
            max_possible_windows = int(span_encoding.text_length) * int(
                self.max_windows_per_span
            )
            if image_steps > max_possible_windows:
                raise ValueError(
                    _legacy._span_no_path_message(
                        span_encoding,
                        image_steps,
                        self.max_windows_per_span,
                        min_required,
                        reason="too_many_windows",
                    )
                )
        return min_required

    def _span_dtw_cost_torch(self, span_encoding, image_embeddings):
        span_lookup = _legacy.span_index_by_start_and_length(span_encoding)
        text_steps = int(span_encoding.text_length)
        image_steps = int(image_embeddings.shape[0])
        device = image_embeddings.device
        min_required = self._check_path_feasible(span_encoding, image_steps)
        transition_costs = _precompute_transition_costs(
            span_encoding,
            image_embeddings,
            self.temperature,
            self.max_windows_per_span,
            self.window_count_penalty,
        )
        blank_costs = _blank_transition_costs(
            span_encoding, image_embeddings, self.temperature
        )

        zero = torch.zeros((), device=device, dtype=image_embeddings.dtype)
        dp = [[None for _ in range(image_steps + 1)] for _ in range(text_steps + 1)]
        dp[0][0] = zero

        for i in range(text_steps + 1):
            for j in range(image_steps + 1):
                current = dp[i][j]
                if current is None:
                    continue

                if blank_costs is not None and j < image_steps:
                    candidate = current + blank_costs[j]
                    previous = dp[i][j + 1]
                    dp[i][j + 1] = (
                        candidate
                        if previous is None
                        else _legacy.softmin(
                            torch.stack((previous, candidate)), self.gamma
                        )
                    )

                for span_len in range(1, text_steps - i + 1):
                    span_idx = span_lookup.get((i, span_len))
                    if span_idx is None:
                        continue
                    for window_count in range(
                        1,
                        min(self.max_windows_per_span, image_steps - j) + 1,
                    ):
                        transition = transition_costs[window_count][span_idx, j]
                        if not torch.isfinite(transition):
                            continue
                        next_i = i + span_len
                        next_j = j + window_count
                        candidate = current + transition
                        previous = dp[next_i][next_j]
                        dp[next_i][next_j] = (
                            candidate
                            if previous is None
                            else _legacy.softmin(
                                torch.stack((previous, candidate)), self.gamma
                            )
                        )

        if dp[text_steps][image_steps] is None:
            raise ValueError(
                _legacy._span_no_path_message(
                    span_encoding,
                    image_steps,
                    self.max_windows_per_span,
                    min_required,
                    reason="generic",
                )
            )
        return dp[text_steps][image_steps]
