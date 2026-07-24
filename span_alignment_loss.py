import os
from collections import defaultdict

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
    blanks = getattr(span_encoding, "is_blank", None)
    if blanks is not None:
        blank_mask = torch.as_tensor(blanks, dtype=torch.bool, device=device)
        limits = torch.where(blank_mask, torch.ones_like(limits), limits)
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

    starts = torch.as_tensor(
        span_encoding.starts, dtype=torch.long, device=image_embeddings.device
    )
    lengths = torch.as_tensor(
        span_encoding.lengths, dtype=torch.long, device=image_embeddings.device
    )
    valid = (
        (starts >= 0)
        & (lengths > 0)
        & (lengths <= max_span_chars)
        & (starts + lengths <= actual_text_steps)
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        valid_starts = starts.index_select(0, valid_indices)
        valid_lengths = lengths.index_select(0, valid_indices)
        for window_count, costs in transition_costs.items():
            width = int(costs.shape[1])
            if width <= 0:
                continue
            dense[
                valid_lengths,
                int(window_count),
                valid_starts,
                :width,
            ] = costs.index_select(0, valid_indices)

    blank_costs = _blank_transition_costs(
        span_encoding, image_embeddings, temperature
    )
    if blank_costs is not None and max_windows_per_span >= 1:
        dense[0, 1, : actual_text_steps + 1, :image_steps] = (
            blank_costs.view(1, image_steps).expand(actual_text_steps + 1, image_steps)
        )
    return dense


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
        self.last_positive_encodings = None

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

    def _dense_bucket_key(self, span_encoding, image_embeddings):
        padded_text = _legacy._bucket_length(
            int(span_encoding.text_length),
            span_dtw_text_bucket_size,
            span_dtw_max_text_bucket,
            enabled=span_dtw_bucket_text_lengths,
        )
        return (
            int(getattr(span_encoding, "max_span_chars", 0)) + 1,
            int(self.max_windows_per_span) + 1,
            int(padded_text) + 1,
            int(image_embeddings.shape[0]) + 1,
        )

    def _span_dtw_costs_jax(self, encodings, image_embeddings):
        from jax_span_dtw import JaxBatchedSpanDTWFunction

        if not encodings:
            return image_embeddings.new_empty((0,))
        buckets = defaultdict(list)
        for index, (encoding, image) in enumerate(zip(encodings, image_embeddings)):
            self._check_path_feasible(encoding, int(image.shape[0]))
            buckets[self._dense_bucket_key(encoding, image)].append(index)

        result = [None] * len(encodings)
        for _shape, indices in buckets.items():
            dense_items = []
            text_lengths = []
            image_steps = []
            for index in indices:
                encoding = encodings[index]
                image = image_embeddings[index]
                padded_text = _legacy._bucket_length(
                    int(encoding.text_length),
                    span_dtw_text_bucket_size,
                    span_dtw_max_text_bucket,
                    enabled=span_dtw_bucket_text_lengths,
                )
                dense_items.append(
                    _dense_transition_costs(
                        encoding,
                        image,
                        self.temperature,
                        self.max_windows_per_span,
                        self.window_count_penalty,
                        text_steps_padded=padded_text,
                    )
                )
                text_lengths.append(int(encoding.text_length))
                image_steps.append(int(image.shape[0]))
            dense_batch = torch.stack(dense_items, dim=0)
            text_lengths_tensor = torch.as_tensor(
                text_lengths, device=dense_batch.device, dtype=torch.int32
            )
            image_steps_tensor = torch.as_tensor(
                image_steps, device=dense_batch.device, dtype=torch.int32
            )
            needs_gradient = (
                torch.is_grad_enabled() and dense_batch.requires_grad
            )
            costs = JaxBatchedSpanDTWFunction.apply(
                dense_batch,
                text_lengths_tensor,
                image_steps_tensor,
                self.gamma,
                needs_gradient,
            )
            for local_index, original_index in enumerate(indices):
                result[original_index] = costs[local_index]
        return torch.stack(result)

    def _span_dtw_costs(self, encodings, image_embeddings):
        if self.backend == "jax":
            return self._span_dtw_costs_jax(encodings, image_embeddings)
        return torch.stack(
            [
                self._span_dtw_cost_torch(encoding, image)
                for encoding, image in zip(encodings, image_embeddings)
            ]
        )

    def _span_dtw_cost_jax(self, span_encoding, image_embeddings):
        return self._span_dtw_costs_jax(
            [span_encoding], image_embeddings.unsqueeze(0)
        )[0]

    @staticmethod
    def _encode_many(text_encoder, texts, use_cache=True):
        if hasattr(text_encoder, "encode_many"):
            return text_encoder.encode_many(texts, use_cache=use_cache)
        return [
            text_encoder(
                text,
                use_cache=use_cache if hasattr(text_encoder, "clear_cache") else None,
            )
            for text in texts
        ]

    def _costs_allowing_infeasible(self, encodings, images):
        valid_indices = []
        result = [None] * len(encodings)
        for index, (encoding, image) in enumerate(zip(encodings, images)):
            try:
                self._check_path_feasible(encoding, int(image.shape[0]))
                valid_indices.append(index)
            except ValueError:
                result[index] = _legacy._infeasible_negative_cost(
                    image, encoding.text_length, self.temperature
                )
        if valid_indices:
            index_tensor = torch.as_tensor(
                valid_indices, device=images.device, dtype=torch.long
            )
            valid_images = images.index_select(0, index_tensor)
            valid_encodings = [encodings[index] for index in valid_indices]
            valid_costs = self._span_dtw_costs(valid_encodings, valid_images)
            for local_index, original_index in enumerate(valid_indices):
                result[original_index] = valid_costs[local_index]
        return torch.stack(result)

    def forward_varlen(self, text_encoder, norm_img, pos_texts, neg_texts):
        pos_texts = list(pos_texts)
        pos_encodings = self._encode_many(text_encoder, pos_texts, use_cache=True)
        self.last_positive_encodings = pos_encodings
        pos_costs = self._span_dtw_costs(pos_encodings, norm_img)
        image_steps = int(norm_img.shape[1])
        pos_denoms = pos_costs.new_tensor(
            [max(int(enc.text_length), image_steps) for enc in pos_encodings]
        )
        norm_pos = pos_costs / pos_denoms

        flat_neg_texts = []
        owners = []
        owner_slices = []
        cursor = 0
        for sample_index, sample_negatives in enumerate(neg_texts):
            sample_negatives = list(sample_negatives)
            if not sample_negatives:
                raise ValueError(
                    "SpanContrastiveSoftDTW requires at least one negative text per sample."
                )
            flat_neg_texts.extend(sample_negatives)
            owners.extend([sample_index] * len(sample_negatives))
            owner_slices.append(slice(cursor, cursor + len(sample_negatives)))
            cursor += len(sample_negatives)
        owner_tensor = torch.as_tensor(
            owners, device=norm_img.device, dtype=torch.long
        )
        flat_images = norm_img.index_select(0, owner_tensor)

        if self.negative_grad_mode == "all":
            neg_encodings = self._encode_many(
                text_encoder, flat_neg_texts, use_cache=True
            )
            scored_costs = self._costs_allowing_infeasible(
                neg_encodings, flat_images
            )
            grad_costs = scored_costs
            grad_encodings = neg_encodings
            selected_flat_indices = None
        else:
            with torch.no_grad():
                scored_encodings = self._encode_many(
                    text_encoder, flat_neg_texts, use_cache=True
                )
                scored_costs = self._costs_allowing_infeasible(
                    scored_encodings, flat_images
                ).detach()
            selected_flat_indices = []
            for sample_index, sample_slice in enumerate(owner_slices):
                local_costs = scored_costs[sample_slice]
                local_encodings = scored_encodings[sample_slice]
                local_denoms = local_costs.new_tensor(
                    [
                        max(int(enc.text_length), image_steps)
                        for enc in local_encodings
                    ]
                )
                selected_flat_indices.append(
                    sample_slice.start
                    + int(torch.argmin(local_costs / local_denoms).item())
                )
            if self.negative_grad_mode == "hardest":
                selected_texts = [
                    flat_neg_texts[index] for index in selected_flat_indices
                ]
                grad_encodings = self._encode_many(
                    text_encoder, selected_texts, use_cache=True
                )
                selected_owner_tensor = torch.arange(
                    len(pos_texts), device=norm_img.device, dtype=torch.long
                )
                grad_images = norm_img.index_select(0, selected_owner_tensor)
                grad_costs = self._costs_allowing_infeasible(
                    grad_encodings, grad_images
                )
            else:
                grad_encodings = [
                    scored_encodings[index] for index in selected_flat_indices
                ]
                grad_costs = torch.stack(
                    [scored_costs[index] for index in selected_flat_indices]
                )

        scored_norms = []
        for cost, encoding in zip(
            scored_costs,
            neg_encodings if self.negative_grad_mode == "all" else scored_encodings,
        ):
            scored_norms.append(
                cost / max(int(encoding.text_length), image_steps)
            )
        scored_norms = torch.stack(scored_norms)

        losses = []
        norm_neg_for_loss = []
        for sample_index, sample_slice in enumerate(owner_slices):
            if self.negative_grad_mode == "all":
                sample_costs = grad_costs[sample_slice]
                sample_encodings = grad_encodings[sample_slice]
                sample_norms = torch.stack(
                    [
                        cost / max(int(enc.text_length), image_steps)
                        for cost, enc in zip(sample_costs, sample_encodings)
                    ]
                )
            else:
                cost = grad_costs[sample_index]
                encoding = grad_encodings[sample_index]
                sample_norms = (
                    cost / max(int(encoding.text_length), image_steps)
                ).view(1)
            norm_neg_for_loss.append(sample_norms.mean())
            losses.append(
                torch.clamp(
                    norm_pos[sample_index] - sample_norms + self.margin,
                    min=0.0,
                ).mean()
            )

        pos_probabilities = []
        mean_scored_norms = []
        mean_scored_costs = []
        for sample_index, sample_slice in enumerate(owner_slices):
            sample_norms = scored_norms[sample_slice]
            sample_costs = scored_costs[sample_slice]
            all_costs = torch.cat(
                [norm_pos[sample_index].detach().view(1), sample_norms.detach()]
            )
            pos_probabilities.append(torch.softmax(-all_costs, dim=0)[0])
            mean_scored_norms.append(sample_norms.mean().detach())
            mean_scored_costs.append(sample_costs.mean().detach())

        loss = torch.stack(losses).mean()
        mean_scored_norms_tensor = torch.stack(mean_scored_norms)
        return loss, {
            "cost_pos": float(pos_costs.detach().mean().item()),
            "cost_neg": float(torch.stack(mean_scored_costs).mean().item()),
            "pos_prob": float(torch.stack(pos_probabilities).mean().item()),
            "gap": float(
                (norm_pos.detach() - mean_scored_norms_tensor).mean().item()
            ),
            "norm_pos": float(norm_pos.detach().mean().item()),
            "norm_neg": float(mean_scored_norms_tensor.mean().item()),
            "contrastive": float(loss.detach().item()),
            "dtw_batch_items": float(len(pos_encodings) + len(flat_neg_texts)),
        }
