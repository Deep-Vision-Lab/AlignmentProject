import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from Parameters import (
    contrastive_margin,
    contrastive_soft_dtw_gamma,
    contrastive_temperature,
    max_windows_per_span,
    span_dtw_bucket_text_lengths,
    span_dtw_max_text_bucket,
    span_dtw_text_bucket_size,
)


_SPAN_DTW_MEM_DEBUG = os.environ.get("SPAN_DTW_MEM_DEBUG", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SPAN_DTW_MEM_DEBUG_INTERVAL = max(1, int(os.environ.get("SPAN_DTW_MEM_DEBUG_INTERVAL", "50")))
_JAX_DENSE_CALL_COUNT = 0
_JAX_DENSE_SHAPE_COUNTS = {}


def softmin(values, gamma):
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def _current_memory_summary():
    parts = []
    try:
        import psutil

        process = psutil.Process(os.getpid())
        parts.append(f"rss_mb={process.memory_info().rss / (1024 ** 2):.1f}")
    except Exception:
        pass

    if torch.cuda.is_available():
        parts.extend(
            [
                f"cuda_alloc_mb={torch.cuda.memory_allocated() / (1024 ** 2):.1f}",
                f"cuda_reserved_mb={torch.cuda.memory_reserved() / (1024 ** 2):.1f}",
                f"cuda_max_alloc_mb={torch.cuda.max_memory_allocated() / (1024 ** 2):.1f}",
            ]
        )
    return " ".join(parts)


def _bucket_length(length, bucket_size, max_bucket, enabled=True):
    length = int(length)
    if not enabled:
        return length
    bucket_size = max(1, int(bucket_size))
    max_bucket = max(1, int(max_bucket))
    bucketed = ((length + bucket_size - 1) // bucket_size) * bucket_size
    if length <= max_bucket:
        bucketed = min(bucketed, max_bucket)
    return max(length, bucketed)


def _debug_jax_dense_tensor(span_encoding, image_embeddings, dense, text_steps_padded=None):
    global _JAX_DENSE_CALL_COUNT
    _JAX_DENSE_CALL_COUNT += 1
    shape = tuple(dense.shape)
    previous_shape_count = _JAX_DENSE_SHAPE_COUNTS.get(shape, 0)
    _JAX_DENSE_SHAPE_COUNTS[shape] = previous_shape_count + 1

    should_print = (
        previous_shape_count == 0
        or (_JAX_DENSE_CALL_COUNT % _SPAN_DTW_MEM_DEBUG_INTERVAL == 0)
    )
    if not should_print:
        return

    dense_mb = dense.numel() * dense.element_size() / (1024 ** 2)
    print(
        "[SPAN_MEM_DEBUG] "
        f"jax_dense_call={_JAX_DENSE_CALL_COUNT} "
        f"new_shape={previous_shape_count == 0} "
        f"unique_dense_shapes={len(_JAX_DENSE_SHAPE_COUNTS)} "
        f"dense_shape={shape} dense_mb={dense_mb:.2f} "
        f"actual_text_length={span_encoding.text_length} "
        f"bucketed_text_length={text_steps_padded or span_encoding.text_length} "
        f"num_valid_spans={len(span_encoding.starts)} "
        f"image_windows={image_embeddings.shape[0]} "
        f"{_current_memory_summary()}",
        flush=True,
    )


def _debug_jax_cost(span_encoding, image_embeddings, cost):
    if not _SPAN_DTW_MEM_DEBUG:
        return
    detached = cost.detach()
    is_bad = (not torch.isfinite(detached).all().item()) or detached.item() < -1e-3
    if not is_bad:
        return
    print(
        "[SPAN_MEM_DEBUG_BAD_COST] "
        f"cost={detached.item()} "
        f"text_length={span_encoding.text_length} "
        f"image_windows={image_embeddings.shape[0]} "
        f"num_valid_spans={len(span_encoding.starts)} "
        f"{_current_memory_summary()}",
        flush=True,
    )


def span_index_by_start_and_length(span_encoding):
    return {
        (start, length): idx
        for idx, (start, length) in enumerate(
            zip(span_encoding.starts, span_encoding.lengths)
        )
    }


def _spans_by_start(span_encoding):
    spans = {}
    for start, length in zip(span_encoding.starts, span_encoding.lengths):
        spans.setdefault(start, []).append(length)
    return spans


def _min_required_spans(span_encoding):
    text_length = span_encoding.text_length
    inf = float("inf")
    min_spans = [inf] * (text_length + 1)
    min_spans[0] = 0
    spans_by_start = _spans_by_start(span_encoding)

    for i in range(text_length):
        if min_spans[i] == inf:
            continue
        for span_length in spans_by_start.get(i, []):
            next_i = i + span_length
            if next_i <= text_length:
                min_spans[next_i] = min(min_spans[next_i], min_spans[i] + 1)

    return min_spans[text_length]


def _span_no_path_message(span_encoding, image_steps, max_windows_per_span, min_required_spans, reason=None):
    max_span_chars = getattr(span_encoding, "max_span_chars", "unknown")
    prefix = "No valid span-DTW path. "
    if reason == "too_few_windows":
        detail = "There are too few image windows for the text length. "
        fix = (
            "Fix by: "
            "1. decreasing STRIDE_RATIO, for example 0.5 instead of 1.0; "
            "2. increasing MAX_TEXT_SPAN_CHARS, for example 3; "
            "3. using a smaller WINDOW_SIZE only if it increases usable windows. "
            "Do not fix this by increasing MAX_WINDOWS_PER_SPAN."
        )
    elif reason == "too_many_windows":
        detail = (
            "There are too many image windows for this text under MAX_WINDOWS_PER_SPAN. "
        )
        fix = (
            "Fix by increasing MAX_WINDOWS_PER_SPAN for this mode, reducing image windows, "
            "or treating this transcript as an infeasible/easy negative."
        )
    else:
        detail = (
            "This usually means the text/window constraints are infeasible. "
        )
        fix = (
            "Check STRIDE_RATIO, MAX_TEXT_SPAN_CHARS, MAX_WINDOWS_PER_SPAN, and transcript length."
        )
    return (
        prefix
        + detail
        + f"text_length={span_encoding.text_length}, image_windows={image_steps}, "
        f"minimum_required_spans={min_required_spans}, "
        f"max_text_span_chars={max_span_chars}, "
        f"max_windows_per_span={max_windows_per_span}. "
        + fix
    )


def _infeasible_negative_cost(image_embeddings, text_length, temperature):
    return image_embeddings.new_tensor(
        max(1, int(image_embeddings.shape[0]), int(text_length)) * (2.0 / temperature)
    )


def _transition_cost(span_embedding, image_window_embeddings, temperature, window_count_penalty):
    # Non-negative cosine distance: good match -> near 0, bad match -> larger cost.
    cosine_similarities = torch.matmul(image_window_embeddings, span_embedding)
    window_costs = (1.0 - cosine_similarities) / temperature
    return window_costs.mean() + window_count_penalty * (image_window_embeddings.shape[0] - 1)


def _precompute_transition_costs(
    span_encoding,
    image_embeddings,
    temperature,
    max_windows_per_span,
    window_count_penalty,
):
    span_embeddings = F.normalize(span_encoding.embeddings.float(), p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=-1)
    image_steps = image_embeddings.shape[0]
    max_windows = min(max_windows_per_span, image_steps)

    cosine_similarities = torch.matmul(span_embeddings, image_embeddings.T)
    window_costs = (1.0 - cosine_similarities) / temperature
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

    costs_by_window_count = {}
    for window_count in range(1, max_windows + 1):
        range_sums = prefix[:, window_count:] - prefix[:, :-window_count]
        range_means = range_sums / window_count
        costs_by_window_count[window_count] = (
            range_means + window_count_penalty * (window_count - 1)
        )
    return costs_by_window_count


def _dense_transition_costs(
    span_encoding,
    image_embeddings,
    temperature,
    max_windows_per_span,
    window_count_penalty,
    text_steps_padded=None,
):
    actual_text_steps = span_encoding.text_length
    dense_text_steps = int(text_steps_padded or actual_text_steps)
    if dense_text_steps < actual_text_steps:
        raise ValueError(
            f"text_steps_padded={dense_text_steps} is smaller than actual text length "
            f"{actual_text_steps}."
        )
    image_steps = image_embeddings.shape[0]
    max_span_chars = max(
        int(getattr(span_encoding, "max_span_chars", 0)),
        max(span_encoding.lengths, default=0),
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

    for span_idx, (start, span_len) in enumerate(zip(span_encoding.starts, span_encoding.lengths)):
        if span_len > max_span_chars or start + span_len > actual_text_steps:
            continue
        for window_count, costs in transition_costs.items():
            if costs.shape[1] == 0:
                continue
            dense[span_len, window_count, start, : costs.shape[1]] = costs[span_idx]

    return dense


def hard_span_dtw_path(
    span_encoding,
    image_embeddings,
    temperature=contrastive_temperature,
    max_windows=max_windows_per_span,
    window_count_penalty=0.01,
):
    span_lookup = span_index_by_start_and_length(span_encoding)
    text_steps = span_encoding.text_length
    image_steps = image_embeddings.shape[0]
    min_required = _min_required_spans(span_encoding)
    if min_required == float("inf") or image_steps < min_required:
        raise ValueError(
            _span_no_path_message(
                span_encoding,
                image_steps,
                max_windows,
                min_required,
                reason="too_few_windows",
            )
        )
    max_possible_windows = span_encoding.text_length * max_windows
    if image_steps > max_possible_windows:
        raise ValueError(
            _span_no_path_message(
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

    dp = torch.full((text_steps + 1, image_steps + 1), float("inf"))
    back = [[None for _ in range(image_steps + 1)] for _ in range(text_steps + 1)]
    dp[0, 0] = 0.0

    for i in range(text_steps + 1):
        for j in range(image_steps + 1):
            if not torch.isfinite(dp[i, j]):
                continue
            for span_len in range(1, text_steps - i + 1):
                span_idx = span_lookup.get((i, span_len))
                if span_idx is None:
                    continue
                for window_count in range(1, min(max_windows, image_steps - j) + 1):
                    next_i = i + span_len
                    next_j = j + window_count
                    cost = transition_costs[window_count][span_idx, j]
                    candidate = dp[i, j] + cost.detach().cpu()
                    if candidate < dp[next_i, next_j]:
                        dp[next_i, next_j] = candidate
                        back[next_i][next_j] = (i, j, span_idx)

    if back[text_steps][image_steps] is None:
        raise ValueError(
            _span_no_path_message(
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
        prev_i, prev_j, span_idx = back[i][j]
        path.append(
            {
                "text_start": prev_i,
                "text_end": i,
                "window_start": prev_j,
                "window_end": j,
                "span_idx": span_idx,
                "text": span_encoding.texts[span_idx],
            }
        )
        i, j = prev_i, prev_j
    path.reverse()
    return path


class SpanContrastiveSoftDTW(nn.Module):
    def __init__(
        self,
        gamma=contrastive_soft_dtw_gamma,
        margin=contrastive_margin,
        temperature=contrastive_temperature,
        max_windows_per_span=max_windows_per_span,
        window_count_penalty=0.01,
        negative_grad_mode="hardest",
        backend="torch",
    ):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.temperature = temperature
        self.max_windows_per_span = max_windows_per_span
        self.window_count_penalty = window_count_penalty
        self.negative_grad_mode = str(negative_grad_mode).lower()
        self.backend = str(backend).lower()
        self._warned_jax_backend = False
        if self.negative_grad_mode not in {"all", "hardest", "none"}:
            raise ValueError(
                "negative_grad_mode must be one of: all, hardest, none. "
                f"Got {negative_grad_mode!r}."
            )
        if self.backend not in {"torch", "jax"}:
            raise ValueError("backend must be 'torch' or 'jax'")

    def _span_dtw_cost(self, span_encoding, image_embeddings):
        if self.backend == "torch":
            return self._span_dtw_cost_torch(span_encoding, image_embeddings)
        return self._span_dtw_cost_jax(span_encoding, image_embeddings)

    def _check_path_feasible(self, span_encoding, image_steps):
        text_steps = span_encoding.text_length
        min_required = _min_required_spans(span_encoding)
        if min_required == float("inf") or image_steps < min_required:
            raise ValueError(
                _span_no_path_message(
                    span_encoding,
                    image_steps,
                    self.max_windows_per_span,
                    min_required,
                    reason="too_few_windows",
                )
            )
        max_possible_windows = text_steps * self.max_windows_per_span
        if image_steps > max_possible_windows:
            raise ValueError(
                _span_no_path_message(
                    span_encoding,
                    image_steps,
                    self.max_windows_per_span,
                    min_required,
                    reason="too_many_windows",
                )
            )
        return min_required

    def _span_dtw_cost_torch(self, span_encoding, image_embeddings):
        span_lookup = span_index_by_start_and_length(span_encoding)
        text_steps = span_encoding.text_length
        image_steps = image_embeddings.shape[0]
        device = image_embeddings.device
        min_required = self._check_path_feasible(span_encoding, image_steps)
        transition_costs = _precompute_transition_costs(
            span_encoding,
            image_embeddings,
            self.temperature,
            self.max_windows_per_span,
            self.window_count_penalty,
        )

        zero = torch.zeros((), device=device, dtype=image_embeddings.dtype)
        # Keep one accumulated soft-min value per DP cell instead of a list of
        # every incoming candidate. The previous implementation created:
        #
        #   candidates = [[[] for image_step] for text_step]
        #
        # for every positive and negative transcript, then appended graph
        # tensors to those lists. With B samples and N negatives this creates
        # B*(1+N) large list grids per batch and keeps many duplicate transition
        # tensors alive until backward, which is exactly the kind of object that
        # causes host RAM OOM in span-D3TW.
        #
        # Incremental softmin is equivalent because:
        #   softmin(softmin(a, b), c) == softmin(a, b, c)
        # for the log-sum-exp definition used here.
        dp = [[None for _ in range(image_steps + 1)] for _ in range(text_steps + 1)]
        dp[0][0] = zero

        for i in range(text_steps + 1):
            for j in range(image_steps + 1):
                current = dp[i][j]
                if current is None:
                    continue
                for span_len in range(1, text_steps - i + 1):
                    span_idx = span_lookup.get((i, span_len))
                    if span_idx is None:
                        continue
                    for window_count in range(1, min(self.max_windows_per_span, image_steps - j) + 1):
                        next_i = i + span_len
                        next_j = j + window_count
                        transition = transition_costs[window_count][span_idx, j]
                        candidate = current + transition
                        previous = dp[next_i][next_j]
                        if previous is None:
                            dp[next_i][next_j] = candidate
                        else:
                            dp[next_i][next_j] = softmin(
                                torch.stack((previous, candidate)),
                                self.gamma,
                            )

        if dp[text_steps][image_steps] is None:
            raise ValueError(
                _span_no_path_message(
                    span_encoding,
                    image_steps,
                    self.max_windows_per_span,
                    min_required,
                    reason="generic",
                )
            )
        return dp[text_steps][image_steps]

    def _span_dtw_cost_jax(self, span_encoding, image_embeddings):
        if not self._warned_jax_backend:
            print(
                "SPAN_DTW_BACKEND=jax compiles by dense tensor shape. "
                "Text lengths are bucketed when SPAN_DTW_BUCKET_TEXT_LENGTHS=1; "
                "also keep WINDOW_SIZE, STRIDE_RATIO, MAX_TEXT_SPAN_CHARS, "
                "and MAX_WINDOWS_PER_SPAN fixed for best reuse.",
                flush=True,
            )
            self._warned_jax_backend = True

        image_steps = image_embeddings.shape[0]
        self._check_path_feasible(span_encoding, image_steps)
        actual_text_steps = int(span_encoding.text_length)
        text_steps_padded = _bucket_length(
            actual_text_steps,
            span_dtw_text_bucket_size,
            span_dtw_max_text_bucket,
            enabled=span_dtw_bucket_text_lengths,
        )
        transition_costs_dense = _dense_transition_costs(
            span_encoding,
            image_embeddings,
            self.temperature,
            self.max_windows_per_span,
            self.window_count_penalty,
            text_steps_padded=text_steps_padded,
        )
        if _SPAN_DTW_MEM_DEBUG:
            _debug_jax_dense_tensor(
                span_encoding,
                image_embeddings,
                transition_costs_dense,
                text_steps_padded=text_steps_padded,
            )

        try:
            from jax_span_dtw import JaxSpanDTWFunction
        except RuntimeError:
            raise
        except ImportError as exc:
            raise RuntimeError(
                "SPAN_DTW_BACKEND=jax requires JAX to be installed. "
                "Use SPAN_DTW_BACKEND=torch or install the JAX packages from requirements.txt."
            ) from exc
        needs_gradient = torch.is_grad_enabled() and transition_costs_dense.requires_grad
        cost = JaxSpanDTWFunction.apply(
            transition_costs_dense,
            actual_text_steps,
            int(image_steps),
            self.gamma,
            needs_gradient,
        )
        _debug_jax_cost(span_encoding, image_embeddings, cost)
        return cost

    def forward_varlen(self, text_encoder, norm_img, pos_texts, neg_texts):
        losses = []
        pos_cost_values = []
        neg_cost_values = []
        norm_pos_values = []
        norm_neg_values = []
        pos_prob_values = []
        gap_values = []

        for sample_idx, pos_text in enumerate(pos_texts):
            pos_encoding = text_encoder(pos_text, use_cache=False if text_encoder.training else None)
            pos_cost = self._span_dtw_cost(pos_encoding, norm_img[sample_idx])
            norm_pos = pos_cost / max(pos_encoding.text_length, norm_img[sample_idx].shape[0])

            neg_costs = []
            norm_negs = []
            neg_text_list = list(neg_texts[sample_idx])

            if self.negative_grad_mode == "all":
                for neg_text in neg_text_list:
                    neg_encoding = text_encoder(neg_text, use_cache=False)
                    try:
                        neg_cost = self._span_dtw_cost(neg_encoding, norm_img[sample_idx])
                    except ValueError:
                        neg_cost = _infeasible_negative_cost(
                            norm_img[sample_idx],
                            neg_encoding.text_length,
                            self.temperature,
                        )
                    neg_costs.append(neg_cost)
                    norm_negs.append(neg_cost / max(neg_encoding.text_length, norm_img[sample_idx].shape[0]))
                neg_costs = torch.stack(neg_costs)
                norm_negs = torch.stack(norm_negs)
            else:
                # Memory-critical path: each span-DTW cost builds a non-trivial
                # autograd graph. Keeping all negative graphs for a full batch
                # creates B*num_negatives repeated DP graphs before backward.
                # Score every negative without grad, then optionally recompute
                # only the hardest negative with grad. This keeps the hard
                # contrastive signal while avoiding the repeated object/graph
                # explosion that was OOM-killing the SLURM job in epoch 1.
                scored_neg_costs = []
                scored_norm_negs = []
                neg_feasible = []
                with torch.no_grad():
                    for neg_text in neg_text_list:
                        neg_encoding = text_encoder(neg_text, use_cache=False)
                        try:
                            neg_cost = self._span_dtw_cost(neg_encoding, norm_img[sample_idx])
                            feasible = True
                        except ValueError:
                            neg_cost = _infeasible_negative_cost(
                                norm_img[sample_idx],
                                neg_encoding.text_length,
                                self.temperature,
                            )
                            feasible = False
                        scored_neg_costs.append(neg_cost.detach())
                        scored_norm_negs.append(
                            (neg_cost / max(neg_encoding.text_length, norm_img[sample_idx].shape[0])).detach()
                        )
                        neg_feasible.append(feasible)

                if not scored_norm_negs:
                    raise ValueError("SpanContrastiveSoftDTW requires at least one negative text per sample.")

                scored_neg_costs = torch.stack(scored_neg_costs)
                scored_norm_negs = torch.stack(scored_norm_negs)
                hard_idx = int(torch.argmin(scored_norm_negs).item())

                if self.negative_grad_mode == "hardest":
                    hard_neg_encoding = text_encoder(neg_text_list[hard_idx], use_cache=False)
                    if neg_feasible[hard_idx]:
                        hard_neg_cost = self._span_dtw_cost(hard_neg_encoding, norm_img[sample_idx])
                    else:
                        hard_neg_cost = scored_neg_costs[hard_idx]
                    hard_norm_neg = hard_neg_cost / max(
                        hard_neg_encoding.text_length,
                        norm_img[sample_idx].shape[0],
                    )
                    neg_costs = hard_neg_cost.view(1)
                    norm_negs = hard_norm_neg.view(1)
                else:
                    # "none": no negative branch graph is kept. The margin still
                    # gates positive alignment updates using the hardest cached
                    # negative score.
                    neg_costs = scored_neg_costs[hard_idx].view(1)
                    norm_negs = scored_norm_negs[hard_idx].view(1)

                stats_neg_costs = scored_neg_costs
                stats_norm_negs = scored_norm_negs

            sample_loss = torch.clamp(norm_pos - norm_negs + self.margin, min=0.0).mean()
            losses.append(sample_loss)

            if self.negative_grad_mode == "all":
                stats_neg_costs = neg_costs.detach()
                stats_norm_negs = norm_negs.detach()

            all_costs = torch.cat([norm_pos.detach().view(1), stats_norm_negs], dim=0)
            pos_cost_values.append(pos_cost.detach())
            neg_cost_values.append(stats_neg_costs.mean().detach())
            norm_pos_values.append(norm_pos.detach())
            norm_neg_values.append(stats_norm_negs.mean().detach())
            pos_prob_values.append(torch.softmax(-all_costs, dim=0)[0].detach())
            gap_values.append((norm_pos.detach() - stats_norm_negs.mean()).detach())

        loss = torch.stack(losses).mean()
        return loss, {
            "cost_pos": torch.stack(pos_cost_values).mean().item(),
            "cost_neg": torch.stack(neg_cost_values).mean().item(),
            "pos_prob": torch.stack(pos_prob_values).mean().item(),
            "gap": torch.stack(gap_values).mean().item(),
            "norm_pos": torch.stack(norm_pos_values).mean().item(),
            "norm_neg": torch.stack(norm_neg_values).mean().item(),
            "contrastive": loss.item(),
        }
