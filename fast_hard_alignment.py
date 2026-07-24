from __future__ import annotations

import numpy as np
import torch

import span_alignment_loss as span_loss
import span_alignment_loss_legacy as legacy


def hard_span_dtw_path_fast(
    span_encoding,
    image_embeddings,
    temperature=span_loss.contrastive_temperature,
    max_windows=span_loss.max_windows_per_span,
    window_count_penalty=None,
    include_blank_steps=None,
):
    """Hard path with one bulk GPU→CPU transfer instead of scalar syncs.

    The previous Python decoder called ``cost.detach().cpu()`` in the innermost
    transition loop. On CUDA that synchronizes the device thousands of times per
    path. This implementation transfers each complete transition-cost tensor once
    and performs the dynamic program and backtrace in NumPy.
    """
    if window_count_penalty is None:
        window_count_penalty = span_loss._env_float(
            "SPAN_WINDOW_COUNT_PENALTY", 0.05
        )
    if include_blank_steps is None:
        include_blank_steps = span_loss._env_flag(
            "SPAN_RETURN_BLANK_STEPS", False
        )

    text_steps = int(span_encoding.text_length)
    image_steps = int(image_embeddings.shape[0])
    min_required = legacy._min_required_spans(span_encoding)
    if min_required == float("inf") or image_steps < min_required:
        raise ValueError(
            legacy._span_no_path_message(
                span_encoding,
                image_steps,
                max_windows,
                min_required,
                reason="too_few_windows",
            )
        )
    if not span_loss._blank_available(span_encoding):
        max_possible = text_steps * int(max_windows)
        if image_steps > max_possible:
            raise ValueError(
                legacy._span_no_path_message(
                    span_encoding,
                    image_steps,
                    max_windows,
                    min_required,
                    reason="too_many_windows",
                )
            )

    with torch.no_grad():
        transition_gpu = span_loss._precompute_transition_costs(
            span_encoding,
            image_embeddings,
            temperature,
            max_windows,
            window_count_penalty,
        )
        transition = {
            count: costs.detach().float().cpu().numpy()
            for count, costs in transition_gpu.items()
        }
        blank_gpu = span_loss._blank_transition_costs(
            span_encoding, image_embeddings, temperature
        )
        blank = (
            blank_gpu.detach().float().cpu().numpy()
            if blank_gpu is not None
            else None
        )

    lookup = legacy.span_index_by_start_and_length(span_encoding)
    blank_index = int(getattr(span_encoding, "blank_index", -1))
    dp = np.full((text_steps + 1, image_steps + 1), np.inf, dtype=np.float64)
    back_i = np.full_like(dp, -1, dtype=np.int32)
    back_j = np.full_like(dp, -1, dtype=np.int32)
    back_span = np.full_like(dp, -1, dtype=np.int32)
    back_blank = np.zeros_like(dp, dtype=np.bool_)
    dp[0, 0] = 0.0

    for i in range(text_steps + 1):
        for j in range(image_steps + 1):
            current = dp[i, j]
            if not np.isfinite(current):
                continue
            if blank is not None and j < image_steps:
                candidate = current + float(blank[j])
                if candidate < dp[i, j + 1]:
                    dp[i, j + 1] = candidate
                    back_i[i, j + 1] = i
                    back_j[i, j + 1] = j
                    back_span[i, j + 1] = blank_index
                    back_blank[i, j + 1] = True

            for span_len in range(1, text_steps - i + 1):
                span_index = lookup.get((i, span_len))
                if span_index is None:
                    continue
                maximum = min(int(max_windows), image_steps - j)
                for window_count in range(1, maximum + 1):
                    cost = float(transition[window_count][span_index, j])
                    if not np.isfinite(cost):
                        continue
                    next_i = i + span_len
                    next_j = j + window_count
                    candidate = current + cost
                    if candidate < dp[next_i, next_j]:
                        dp[next_i, next_j] = candidate
                        back_i[next_i, next_j] = i
                        back_j[next_i, next_j] = j
                        back_span[next_i, next_j] = int(span_index)
                        back_blank[next_i, next_j] = False

    if back_span[text_steps, image_steps] < 0:
        raise ValueError(
            legacy._span_no_path_message(
                span_encoding,
                image_steps,
                max_windows,
                min_required,
                reason="generic",
            )
        )

    path = []
    i, j = text_steps, image_steps
    while i != 0 or j != 0:
        previous_i = int(back_i[i, j])
        previous_j = int(back_j[i, j])
        span_index = int(back_span[i, j])
        if previous_i < 0 or previous_j < 0 or span_index < 0:
            raise ValueError("Broken hard Span-DTW backtrace")
        path.append(
            {
                "text_start": previous_i,
                "text_end": i,
                "window_start": previous_j,
                "window_end": j,
                "span_idx": span_index,
                "is_blank": bool(back_blank[i, j]),
            }
        )
        i, j = previous_i, previous_j
    path.reverse()

    surfaces = getattr(span_encoding, "surface_texts", None)
    raw_texts = getattr(span_encoding, "raw_texts", None)
    spaces = getattr(span_encoding, "is_space", None)
    blanks = getattr(span_encoding, "is_blank", None)
    result = []
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
        if blanks is not None:
            step["is_blank"] = bool(blanks[index]) or bool(step["is_blank"])
        if include_blank_steps or not step["is_blank"]:
            result.append(step)
    return result
