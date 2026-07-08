import torch
import torch.nn as nn
import torch.nn.functional as F

from Parameters import (
    contrastive_margin,
    contrastive_soft_dtw_gamma,
    contrastive_temperature,
    max_windows_per_span,
)


def softmin(values, gamma):
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def span_index_by_start_and_length(span_encoding):
    return {
        (start, length): idx
        for idx, (start, length) in enumerate(
            zip(span_encoding.starts, span_encoding.lengths)
        )
    }


def _transition_cost(span_embedding, image_window_embeddings, temperature, window_count_penalty):
    # Non-negative cosine distance: good match -> near 0, bad match -> larger cost.
    similarities = torch.matmul(image_window_embeddings, span_embedding)
    window_costs = (1.0 - similarities) / temperature
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

    similarities = torch.matmul(span_embeddings, image_embeddings.T)
    window_costs = (1.0 - similarities) / temperature
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
            f"No valid span-DTW path for text length {text_steps} and {image_steps} image windows."
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
    ):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.temperature = temperature
        self.max_windows_per_span = max_windows_per_span
        self.window_count_penalty = window_count_penalty
        self.negative_grad_mode = str(negative_grad_mode).lower()
        if self.negative_grad_mode not in {"all", "hardest", "none"}:
            raise ValueError(
                "negative_grad_mode must be one of: all, hardest, none. "
                f"Got {negative_grad_mode!r}."
            )

    def _span_dtw_cost(self, span_encoding, image_embeddings):
        span_lookup = span_index_by_start_and_length(span_encoding)
        text_steps = span_encoding.text_length
        image_steps = image_embeddings.shape[0]
        device = image_embeddings.device
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
                f"No valid span-DTW path for text length {text_steps} and {image_steps} image windows. "
                f"Increase MAX_WINDOWS_PER_SPAN or reduce image windows."
            )
        return dp[text_steps][image_steps]

    def forward_varlen(self, text_encoder, norm_img, pos_texts, neg_texts):
        losses = []
        pos_cost_values = []
        neg_cost_values = []
        norm_pos_values = []
        norm_neg_values = []
        pos_prob_values = []
        gap_values = []

        for sample_idx, pos_text in enumerate(pos_texts):
            pos_encoding = text_encoder(pos_text)
            pos_cost = self._span_dtw_cost(pos_encoding, norm_img[sample_idx])
            norm_pos = pos_cost / max(pos_encoding.text_length, norm_img[sample_idx].shape[0])

            neg_costs = []
            norm_negs = []
            neg_text_list = list(neg_texts[sample_idx])

            if self.negative_grad_mode == "all":
                for neg_text in neg_text_list:
                    neg_encoding = text_encoder(neg_text)
                    neg_cost = self._span_dtw_cost(neg_encoding, norm_img[sample_idx])
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
                with torch.no_grad():
                    for neg_text in neg_text_list:
                        neg_encoding = text_encoder(neg_text)
                        neg_cost = self._span_dtw_cost(neg_encoding, norm_img[sample_idx])
                        scored_neg_costs.append(neg_cost.detach())
                        scored_norm_negs.append(
                            (neg_cost / max(neg_encoding.text_length, norm_img[sample_idx].shape[0])).detach()
                        )

                if not scored_norm_negs:
                    raise ValueError("SpanContrastiveSoftDTW requires at least one negative text per sample.")

                scored_neg_costs = torch.stack(scored_neg_costs)
                scored_norm_negs = torch.stack(scored_norm_negs)
                hard_idx = int(torch.argmin(scored_norm_negs).item())

                if self.negative_grad_mode == "hardest":
                    hard_neg_encoding = text_encoder(neg_text_list[hard_idx])
                    hard_neg_cost = self._span_dtw_cost(hard_neg_encoding, norm_img[sample_idx])
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
