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
    window_costs = -torch.matmul(image_window_embeddings, span_embedding) / temperature
    return window_costs.mean() + window_count_penalty * (image_window_embeddings.shape[0] - 1)


def hard_span_dtw_path(
    span_encoding,
    image_embeddings,
    temperature=contrastive_temperature,
    max_windows=max_windows_per_span,
    window_count_penalty=0.01,
):
    span_embeddings = F.normalize(span_encoding.embeddings.float(), p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=-1)
    span_lookup = span_index_by_start_and_length(span_encoding)
    text_steps = span_encoding.text_length
    image_steps = image_embeddings.shape[0]

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
                    cost = _transition_cost(
                        span_embeddings[span_idx],
                        image_embeddings[j:next_j],
                        temperature,
                        window_count_penalty,
                    )
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
    ):
        super().__init__()
        self.gamma = gamma
        self.margin = margin
        self.temperature = temperature
        self.max_windows_per_span = max_windows_per_span
        self.window_count_penalty = window_count_penalty

    def _span_dtw_cost(self, span_encoding, image_embeddings):
        span_embeddings = F.normalize(span_encoding.embeddings.float(), p=2, dim=-1)
        span_lookup = span_index_by_start_and_length(span_encoding)
        text_steps = span_encoding.text_length
        image_steps = image_embeddings.shape[0]
        device = image_embeddings.device

        zero = torch.zeros((), device=device, dtype=image_embeddings.dtype)
        candidates = [[[] for _ in range(image_steps + 1)] for _ in range(text_steps + 1)]
        candidates[0][0].append(zero)

        for i in range(text_steps + 1):
            for j in range(image_steps + 1):
                if not candidates[i][j]:
                    continue
                current = softmin(torch.stack(candidates[i][j]), self.gamma)
                for span_len in range(1, text_steps - i + 1):
                    span_idx = span_lookup.get((i, span_len))
                    if span_idx is None:
                        continue
                    for window_count in range(1, min(self.max_windows_per_span, image_steps - j) + 1):
                        next_i = i + span_len
                        next_j = j + window_count
                        transition = _transition_cost(
                            span_embeddings[span_idx],
                            image_embeddings[j:next_j],
                            self.temperature,
                            self.window_count_penalty,
                        )
                        candidates[next_i][next_j].append(current + transition)

        if not candidates[text_steps][image_steps]:
            raise ValueError(
                f"No valid span-DTW path for text length {text_steps} and {image_steps} image windows. "
                f"Increase MAX_WINDOWS_PER_SPAN or reduce image windows."
            )
        return softmin(torch.stack(candidates[text_steps][image_steps]), self.gamma)

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
            for neg_text in neg_texts[sample_idx]:
                neg_encoding = text_encoder(neg_text)
                neg_cost = self._span_dtw_cost(neg_encoding, norm_img[sample_idx])
                neg_costs.append(neg_cost)
                norm_negs.append(neg_cost / max(neg_encoding.text_length, norm_img[sample_idx].shape[0]))

            neg_costs = torch.stack(neg_costs)
            norm_negs = torch.stack(norm_negs)
            sample_loss = torch.clamp(norm_pos - norm_negs + self.margin, min=0.0).mean()
            losses.append(sample_loss)

            all_costs = torch.cat([norm_pos.view(1), norm_negs], dim=0)
            pos_cost_values.append(pos_cost.detach())
            neg_cost_values.append(neg_costs.mean().detach())
            norm_pos_values.append(norm_pos.detach())
            norm_neg_values.append(norm_negs.mean().detach())
            pos_prob_values.append(torch.softmax(-all_costs, dim=0)[0].detach())
            gap_values.append((norm_pos - norm_negs.mean()).detach())

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
