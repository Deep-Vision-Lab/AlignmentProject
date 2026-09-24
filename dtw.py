"""Standalone compact letter-DTW mathematics in PyTorch.

Preserved launcher defaults: gamma=.05, vertical=.05, horizontal=.30,
absolute normalized-position prior=.15, alphabet temperature=.10. Horizontal
moves cost 1e4 when T>=L; the final objective is divided by T+L, NOT path length.
These positions belong to DTW, independently of Transformer positional encoding.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


def cosine_similarity_matrix(visual, letters):
    if visual.ndim != 2 or letters.ndim != 2 or visual.shape[1] != letters.shape[1]:
        raise ValueError('Expected compatible [T,D] and [L,D] vectors')
    return F.normalize(visual.float(), dim=-1) @ F.normalize(letters.float(), dim=-1).T


def cosine_cost_matrix(visual, letters):
    return 1. - cosine_similarity_matrix(visual, letters)


def letter_cost_matrix(visual, letters, *, alphabet=None, letter_ids=None,
                       temperature=.10, mode='full_alphabet_nll'):
    if mode == 'cosine':
        return cosine_cost_matrix(visual, letters)
    if mode != 'full_alphabet_nll' or alphabet is None or letter_ids is None:
        raise ValueError('full_alphabet_nll requires complete alphabet vectors and transcript letter_ids')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    nll = -F.log_softmax(cosine_similarity_matrix(visual, alphabet) / max(1e-4, temperature), dim=-1)
    ids = torch.as_tensor(letter_ids, device=visual.device, dtype=torch.long)
    if len(ids) != len(letters):
        raise ValueError('letter_ids must match transcript columns')
    return nll.index_select(1, ids)


def effective_costs(costs, position_prior=.15):
    if costs.ndim != 2 or min(costs.shape) == 0:
        raise ValueError('DTW requires a nonempty [T,L] cost matrix')
    if not torch.isfinite(costs).all():
        raise ValueError('Nonfinite DTW costs')
    t, l = costs.shape
    if position_prior > 0 and t > 1 and l > 1:
        a = torch.linspace(0, 1, t, device=costs.device, dtype=costs.dtype)
        b = torch.linspace(0, 1, l, device=costs.device, dtype=costs.dtype)
        return costs + position_prior * (a[:, None] - b[None, :]).abs()
    return costs


def soft_dtw(costs, gamma=.05, vertical_penalty=.05, horizontal_penalty=.30,
             position_prior=.15, disable_horizontal_when_feasible=True):
    """Anti-diagonal recurrence: vectorized cells, differentiable predecessors."""
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    costs = effective_costs(costs.float(), position_prior)
    t, l = costs.shape
    horizontal = 1e4 if disable_horizontal_when_feasible and t >= l else horizontal_penalty
    previous = previous2 = None
    previous_start = previous2_start = 0
    for diagonal_index in range(t + l - 1):
        start = max(0, diagonal_index - l + 1)
        i = torch.arange(start, min(t - 1, diagonal_index) + 1, device=costs.device)
        j = diagonal_index - i
        missing = costs.new_full(i.shape, 1e4)
        vertical = horizontal_cost = diagonal = missing
        if previous is not None:
            vi = (i - 1 - previous_start).clamp(0, len(previous) - 1)
            hi = (i - previous_start).clamp(0, len(previous) - 1)
            vertical = torch.where(i > 0, previous[vi], missing)
            horizontal_cost = torch.where(j > 0, previous[hi], missing)
        if previous2 is not None:
            di = (i - 1 - previous2_start).clamp(0, len(previous2) - 1)
            diagonal = torch.where((i > 0) & (j > 0), previous2[di], missing)
        diagonal = torch.where((i == 0) & (j == 0), torch.zeros_like(diagonal), diagonal)
        choices = torch.stack((diagonal, vertical + vertical_penalty, horizontal_cost + horizontal))
        current = costs[i, j] - max(gamma, 1e-5) * torch.logsumexp(-choices / max(gamma, 1e-5), dim=0)
        previous2, previous2_start = previous, previous_start
        previous, previous_start = current, start
    return previous[0] / (t + l)


@dataclass
class HardDTW:
    path: list[tuple[int, int]]
    total: float
    normalized: float


def hard_dtw_path(costs, vertical_penalty=.05, horizontal_penalty=.30,
                  position_prior=.15, disable_horizontal_when_feasible=True):
    """Hard argmin, ties diagonal then vertical; scalar includes transition costs."""
    cells = effective_costs(costs.detach().double(), position_prior).cpu().tolist()
    t, l = len(cells), len(cells[0])
    horizontal = 1e4 if disable_horizontal_when_feasible and t >= l else horizontal_penalty
    dp = [[math.inf] * (l + 1) for _ in range(t + 1)]
    dp[0][0] = 0.
    back = {}
    for i in range(1, t + 1):
        for j in range(1, l + 1):
            choices = [(dp[i-1][j-1], (i-1,j-1)),
                       (dp[i-1][j] + vertical_penalty, (i-1,j)),
                       (dp[i][j-1] + horizontal, (i,j-1))]
            best, predecessor = min(choices, key=lambda x: x[0])
            dp[i][j] = cells[i-1][j-1] + best
            back[i,j] = predecessor
    path, position = [], (t,l)
    while position != (0,0):
        i,j = position
        path.append((i-1,j-1))
        position = back[position]
    return HardDTW(path[::-1], dp[t][l], dp[t][l] / (t+l))
