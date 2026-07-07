import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_dtw_cuda
from Parameters import (
    contrastive_margin,
    contrastive_soft_dtw_gamma,
    contrastive_temperature,
)


class ContrastiveSoftDTW(nn.Module):
    """Margin contrastive Soft-DTW over positive and negative similarity matrices."""

    def __init__(
        self,
        gamma=contrastive_soft_dtw_gamma,
        use_cuda=True,
        margin=contrastive_margin,
        temperature=contrastive_temperature,
    ):
        super().__init__()
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.margin = margin
        self.temperature = temperature

    def _compute_dtw_on_similarity(self, sim_matrix):
        # Convert cosine similarity to a per-column character NLL distance.
        centered = sim_matrix - sim_matrix.mean(dim=1, keepdim=True)
        dist_matrix = -F.log_softmax(centered / self.temperature, dim=1)

        if self.use_cuda and dist_matrix.is_cuda:
            return soft_dtw_cuda._SoftDTWCUDA.apply(dist_matrix, self.gamma, 0.0)
        return soft_dtw_cuda._SoftDTW.apply(dist_matrix, self.gamma, 0.0)

    def forward(self, sim_pos, sim_neg_all):
        batch_size, num_negatives = sim_neg_all.shape[:2]

        pos_cost = self._compute_dtw_on_similarity(sim_pos)
        neg_flat = sim_neg_all.reshape(
            batch_size * num_negatives,
            sim_neg_all.shape[2],
            sim_neg_all.shape[3],
        )
        neg_costs = self._compute_dtw_on_similarity(neg_flat).view(batch_size, num_negatives)

        seq_len = max(sim_pos.size(1), sim_pos.size(2))
        norm_pos = pos_cost / seq_len
        norm_neg = neg_costs / seq_len

        per_negative = torch.clamp(norm_pos.unsqueeze(1) - norm_neg + self.margin, min=0.0)
        loss = per_negative.mean()

        all_costs = torch.cat([norm_pos.unsqueeze(1), norm_neg], dim=1)
        pos_prob = torch.softmax(-all_costs, dim=1)[:, 0].mean().item()
        return loss, self._stats(pos_cost, neg_costs, norm_pos, norm_neg, pos_prob, loss)

    def forward_varlen(self, sim_pos_list, sim_neg_list):
        losses = []
        pos_cost_values = []
        neg_cost_values = []
        norm_pos_values = []
        norm_neg_values = []
        pos_prob_values = []
        gap_values = []

        for sim_pos, neg_sims in zip(sim_pos_list, sim_neg_list):
            pos_cost = self._compute_dtw_on_similarity(sim_pos.unsqueeze(0))[0]
            norm_pos = pos_cost / max(sim_pos.shape[0], sim_pos.shape[1])

            neg_costs = []
            norm_negs = []
            for sim_neg in neg_sims:
                neg_cost = self._compute_dtw_on_similarity(sim_neg.unsqueeze(0))[0]
                neg_costs.append(neg_cost)
                norm_negs.append(neg_cost / max(sim_neg.shape[0], sim_neg.shape[1]))

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

    @staticmethod
    def _stats(pos_cost, neg_costs, norm_pos, norm_neg, pos_prob, loss):
        return {
            "cost_pos": pos_cost.mean().item(),
            "cost_neg": neg_costs.mean().item(),
            "pos_prob": pos_prob,
            "gap": (norm_pos.unsqueeze(1) - norm_neg).mean().item(),
            "norm_pos": norm_pos.mean().item(),
            "norm_neg": norm_neg.mean().item(),
            "contrastive": loss.item(),
        }


def contrastive_soft_dtw_alignment_loss(sim_pos, sim_neg, gamma=0.1, use_cuda=True):
    criterion = ContrastiveSoftDTW(gamma=gamma, use_cuda=use_cuda)
    return criterion(sim_pos, sim_neg)
