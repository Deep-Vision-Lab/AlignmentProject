"""Standalone compact letter-DTW mathematics in PyTorch.

Preserved launcher defaults: gamma=.05, vertical=.05, horizontal=.30,
absolute normalized-position prior=.15, alphabet temperature=.10. Horizontal
moves cost 1e4 when T>=L. Legacy normalizes free energy by T+L;
aligned_mean reports occupancy-weighted mean cell NLL.
These positions belong to DTW, independently of Transformer positional encoding.
"""
from dataclasses import dataclass, field
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F


def cosine_similarity_matrix(visual, letters):
    if visual.ndim != 2 or letters.ndim != 2 or visual.shape[1] != letters.shape[1]:
        raise ValueError(f'Embedding dimensions must agree: got {tuple(visual.shape)} and {tuple(letters.shape)}; sequence lengths may differ')
    return F.normalize(visual.float(), dim=-1) @ F.normalize(letters.float(), dim=-1).T


def cosine_cost_matrix(visual, letters):
    return 1. - cosine_similarity_matrix(visual, letters)


def bidirectional_window_scores(cosine, temperature=.10, reward_offset=0.):
    """Float32 directional window evidence on an already compact valid grid.

    Directional softmax scores are relative normalized scores, not calibrated
    match probabilities. Never pass padded rows/columns into this function.
    Empty rectangular grids retain their shape without taking log(0).
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('window_match_temperature must be finite and > 0')
    if not math.isfinite(reward_offset):
        raise ValueError('window_match_reward_offset must be finite')
    c = torch.as_tensor(cosine).float()
    if c.ndim != 2 or not torch.isfinite(c).all():
        raise ValueError('Window cosine must be a finite 2D valid-window matrix')
    if ((c < -1.00001) | (c > 1.00001)).any():
        raise ValueError('Window cosine must be in [-1,1]')
    # PyTorch 2.4 has no MPS autocast context, even with enabled=False.
    context = (torch.autocast(device_type=c.device.type, enabled=False)
               if torch.amp.autocast_mode.is_autocast_available(c.device.type) else nullcontext())
    with context:
        if not c.numel():
            row, column, evidence = c.clone(), c.clone(), c.clone()
        else:
            logits = c / temperature
            if not torch.isfinite(logits).all():
                raise ValueError('Temperature produces nonfinite float32 window logits')
            row = F.log_softmax(logits, dim=1)
            column = F.log_softmax(logits, dim=0)
            n, m = c.shape
            evidence = .5 * (row + math.log(m) + column + math.log(n))
        rewards = evidence - reward_offset
    if not torch.isfinite(rewards).all():
        raise ValueError('Window parameters produce nonfinite float32 rewards')
    return dict(log_p_B_given_A=row, log_p_A_given_B=column,
                bidirectional_evidence=evidence, rewards=rewards)


def letter_cost_matrix(visual, letters, *, alphabet=None, letter_ids=None,
                       temperature=.10, mode='full_alphabet_nll'):
    if visual.ndim != 2 or letters.ndim != 2 or visual.shape[1] != letters.shape[1]:
        raise ValueError(f'Embedding dimensions must agree: visual {tuple(visual.shape)}, text {tuple(letters.shape)}')
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


def alphabet_log_probabilities(visual, alphabet, temperature=.1, blank_logit=None):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Alphabet temperature must be finite and positive')
    logits = cosine_similarity_matrix(visual, alphabet) / max(1e-4, temperature)
    if blank_logit is not None:
        logits = torch.cat((logits, logits.new_full((len(logits), 1), blank_logit)), dim=1)
    return F.log_softmax(logits, dim=-1)


def letter_evidence_matrix(log_a, log_b, prior=None, acceptance_offset=0., prior_floor=1e-4):
    """Rectangular log shared-letter likelihood ratio; never an alignment probability.

    Inputs exclude any blank column WITHOUT renormalizing away blank mass.
    Flooring then renormalizing bounds inverse-prior amplification. Chunking
    bounds the [T_A,T_B,K] temporary while retaining all letter uncertainty.
    """
    if log_a.ndim != 2 or log_b.ndim != 2 or log_a.shape[1] != log_b.shape[1]:
        raise ValueError('Alphabet dimensions/order must agree')
    k = log_a.shape[1]
    if not k or not 0 < prior_floor < 1/k:
        raise ValueError('Require 0 < prior_floor < 1/alphabet_size')
    # Validate/floor BEFORE float32 conversion (tiny positive priors may underflow).
    pi = torch.full((k,), 1/k, dtype=torch.float64) if prior is None else torch.as_tensor(prior, device='cpu', dtype=torch.float64)
    if pi.shape != (k,) or not torch.isfinite(pi).all() or (pi <= 0).any():
        raise ValueError('Prior must be finite, positive, and match alphabet order')
    pi = pi / pi.sum()
    pi = pi.clamp_min(prior_floor); pi = (pi / pi.sum()).to(device=log_a.device,dtype=log_a.dtype)
    if not len(log_a) or not len(log_b):
        return log_a.new_empty((len(log_a),len(log_b)))
    return torch.cat([torch.logsumexp(a[:, None, :] + log_b[None, :, :] - pi.log(), dim=-1)
                      for a in log_a.split(64)], dim=0) - acceptance_offset


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
             position_prior=.15, disable_horizontal_when_feasible=True,
             normalization='legacy', return_occupancy=False, *, space_columns=None,
             space_skip_penalty=0., return_space_skip_count=False):
    """Soft path energy; aligned_mean uses d(total)/d(cell cost) as occupancy.

    The derivative of the log-partition Soft-DTW total is the expected number
    of visits to each cell. Its weighted NLL divided by expected visit count
    is an average evidence cost, not the free energy (which includes entropy,
    transition penalties and the position prior). Differentiation through the
    occupancy is retained during training. Legacy keeps total/(T+L).
    """
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    if normalization not in {'legacy', 'aligned_mean'}:
        raise ValueError('Unknown DTW normalization')
    if space_columns is not None:
        space_columns = torch.as_tensor(space_columns, device=costs.device, dtype=torch.bool)
        if space_columns.shape != (costs.shape[1],):
            raise ValueError('space_columns must have one entry per transcript token')
        if not math.isfinite(space_skip_penalty) or space_skip_penalty <= 0:
            raise ValueError('space_skip_penalty must be finite and positive')
    outer_grad = torch.is_grad_enabled()
    with torch.enable_grad():
        raw = costs.float()
        if (normalization == 'aligned_mean' or return_occupancy or return_space_skip_count) and not raw.requires_grad:
            raw = raw.detach().requires_grad_(True)
        skip_penalty = (raw.new_tensor(space_skip_penalty, requires_grad=return_space_skip_count)
                        if space_columns is not None and bool(space_columns.any()) else None)
        total = _soft_dtw_total(raw, gamma, vertical_penalty, horizontal_penalty,
                                position_prior, disable_horizontal_when_feasible,
                                space_columns, skip_penalty)
        occupancy = expected_skips = None
        if normalization == 'aligned_mean' or return_occupancy or return_space_skip_count:
            parameters = (raw, skip_penalty) if return_space_skip_count and skip_penalty is not None else (raw,)
            derivatives = torch.autograd.grad(total, parameters, create_graph=outer_grad,
                                              retain_graph=outer_grad)
            occupancy = derivatives[0]
            expected_skips = derivatives[1] if len(derivatives) == 2 else raw.new_zeros(())
            if not torch.isfinite(occupancy).all() or occupancy.sum() <= 0:
                raise FloatingPointError('Invalid Soft-DTW alignment occupancy')
        value = ((occupancy * raw).sum() / occupancy.sum() if normalization == 'aligned_mean'
                 else total / sum(raw.shape))
    if not outer_grad:
        value = value.detach()
        if occupancy is not None:
            occupancy = occupancy.detach()
        if expected_skips is not None:
            expected_skips = expected_skips.detach()
    if return_space_skip_count:
        return (value, occupancy, expected_skips) if return_occupancy else (value, expected_skips)
    return (value, occupancy) if return_occupancy else value


def soft_drop_dtw(costs, background_confidence, *, valid_windows=None, gamma=.05,
                  vertical_penalty=.05, horizontal_penalty=.30, position_prior=.15,
                  disable_horizontal_when_feasible=True, space_drop_cost=.1,
                  text_drop_cost=1., enable_space_drop=True,
                  min_text_coverage=.9, enforce_coverage=True,
                  space_columns=None, space_skip_penalty=0., return_details=False):
    """Differentiable global transcript alignment with optional real-window drops.

    Every non-space transcript column must be emitted by at least one match.
    Invalid batch rows are *mandatory zero-cost PAD drops*: they are removed
    before the recurrence, so neither their values nor gradients enter it.
    The reported energy is expected match NLL plus expected real-window drop
    penalty, divided by expected matched-cell mass. This is an evidence energy,
    not the entropy-containing soft-path free energy. Optional transcript spaces
    retain the existing zero-window space skip operation.
    """
    if costs.ndim != 2 or background_confidence.ndim != 1 or len(background_confidence) != len(costs):
        raise ValueError('Require [T,L] costs and [T] background confidence')
    if valid_windows is None:
        valid = torch.ones(len(costs), device=costs.device, dtype=torch.bool)
    else:
        valid = torch.as_tensor(valid_windows, device=costs.device, dtype=torch.bool)
        if valid.shape != (len(costs),):
            raise ValueError('valid_windows must have one entry per cost row')
    pad_count = int((~valid).sum())
    raw = costs[valid].float()
    background = background_confidence[valid].float()
    if raw.numel() == 0 or not torch.isfinite(raw).all() or not torch.isfinite(background).all():
        raise ValueError('Drop-DTW requires finite nonempty valid costs and confidence')
    if (background < 0).any() or (background > 1).any():
        raise ValueError('Background confidence must be in [0,1]')
    if not math.isfinite(gamma) or gamma <= 0 or not 0 <= space_drop_cost < text_drop_cost:
        raise ValueError('Invalid gamma or real-window drop costs')
    spaces = torch.zeros(raw.shape[1], device=raw.device, dtype=torch.bool) if space_columns is None else torch.as_tensor(
        space_columns, device=raw.device, dtype=torch.bool)
    if spaces.shape != (raw.shape[1],):
        raise ValueError('space_columns must match transcript length')
    if spaces.any() and space_skip_penalty <= 0:
        raise ValueError('Optional transcript-space skips require a positive penalty')
    drop = (space_drop_cost + (text_drop_cost-space_drop_cost)*(1-background)
            if enable_space_drop else torch.full_like(background, text_drop_cost))
    outer_grad = torch.is_grad_enabled()
    with torch.enable_grad():
        if not raw.requires_grad:
            raw = raw.detach().requires_grad_(True)
        if not drop.requires_grad:
            drop = drop.detach().requires_grad_(True)
        skip = raw.new_tensor(space_skip_penalty, requires_grad=True) if bool(spaces.any()) else None
        total = _soft_drop_total(raw, drop, gamma, vertical_penalty, horizontal_penalty,
                                 position_prior, disable_horizontal_when_feasible,
                                 spaces, skip if skip is not None else 0.)
        derivatives = torch.autograd.grad(
            total, (raw, drop, skip) if skip is not None else (raw, drop),
            create_graph=outer_grad, retain_graph=outer_grad)
        match_mass, drop_mass = derivatives[:2]
        skip_mass = derivatives[2] if skip is not None else raw.new_zeros(())
        matched = match_mass.sum()
        if not torch.isfinite(total) or not torch.isfinite(match_mass).all() or matched <= 0:
            raise FloatingPointError('Invalid Drop-DTW alignment or no transcript coverage')
        foreground = 1-background
        ink_mass = foreground.sum()
        coverage = ((match_mass.sum(1).clamp(max=1) * foreground).sum() /
                    ink_mass.clamp_min(1e-8)) if ink_mass > 1e-8 else raw.new_zeros(())
        energy = ((match_mass*raw).sum() + (drop_mass*drop).sum()
                  + skip_mass*(skip if skip is not None else 0.)) / matched
        if enforce_coverage:
            energy = energy + text_drop_cost * F.relu(min_text_coverage-coverage)
    if not outer_grad:
        energy, match_mass, drop_mass, coverage, skip_mass = [x.detach() for x in
                                      (energy, match_mass, drop_mass, coverage, skip_mass)]
    details = dict(matched_cells=matched.detach(), dropped_real=drop_mass.sum().detach(),
                   dropped_pad=pad_count, text_coverage=coverage.detach(),
                   skipped_transcript_spaces_expected=skip_mass.detach(),
                   real_windows=len(raw))
    return (energy, match_mass, details) if return_details else energy


def _soft_drop_total(raw, drop, gamma, vertical_penalty, horizontal_penalty,
                     position_prior, disable_horizontal_when_feasible,
                     spaces, space_skip_penalty):
    cells = effective_costs(raw, position_prior)
    t, l = cells.shape
    horizontal = 1e4 if disable_horizontal_when_feasible and t >= l else horizontal_penalty
    previous = previous2 = None
    previous_start = previous2_start = 0
    infinity = 1e6
    for diagonal_index in range(t+l+1):
        start = max(0, diagonal_index-l)
        i = torch.arange(start, min(t, diagonal_index)+1, device=cells.device)
        j = diagonal_index-i
        missing = cells.new_full(i.shape, infinity)
        up = left = diagonal = missing
        if previous is not None:
            ui = (i-1-previous_start).clamp(0,len(previous)-1)
            li = (i-previous_start).clamp(0,len(previous)-1)
            up = torch.where(i>0, previous[ui], missing)
            left = torch.where(j>0, previous[li], missing)
        if previous2 is not None:
            di = (i-1-previous2_start).clamp(0,len(previous2)-1)
            diagonal = torch.where((i>0)&(j>0),previous2[di],missing)
        match = cells[(i-1).clamp_min(0).clamp_max(t-1),(j-1).clamp_min(0).clamp_max(l-1)] - gamma*torch.logsumexp(
            -torch.stack((diagonal,up+vertical_penalty,left+horizontal))/gamma,dim=0)
        match = torch.where((i>0)&(j>0),match,missing)
        skipped_image = torch.where(i>0,up+drop[(i-1).clamp_min(0).clamp_max(t-1)],missing)
        skipped_text = torch.where((j>0)&spaces[(j-1).clamp_min(0)],left+space_skip_penalty,missing)
        current = -gamma*torch.logsumexp(-torch.stack((match,skipped_image,skipped_text))/gamma,dim=0)
        current = torch.where((i==0)&(j==0),torch.zeros_like(current),current)
        previous2,previous2_start = previous,previous_start
        previous,previous_start = current,start
    return previous[0]


def _soft_dtw_total(raw_costs, gamma, vertical_penalty, horizontal_penalty,
                    position_prior, disable_horizontal_when_feasible,
                    space_columns=None, space_skip_penalty=None):
    costs = effective_costs(raw_costs, position_prior)
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
        if space_columns is not None and space_skip_penalty is not None and previous is not None:
            # A space alone may advance the text without consuming a visual
            # window or paying its NLL. All Arabic letters retain old moves.
            skip = torch.where((j > 0) & space_columns[j],
                               previous[(i - previous_start).clamp(0, len(previous) - 1)]
                               + space_skip_penalty, missing)
            current = -gamma * torch.logsumexp(torch.stack((-current / gamma, -skip / gamma)), dim=0)
        previous2, previous2_start = previous, previous_start
        previous, previous_start = current, start
    return previous[0]


@dataclass
class HardDTW:
    path: list[tuple[int, int]]
    total: float
    normalized: float  # legacy total/(T+L), not the aligned_mean training energy
    skipped_columns: list[int] = field(default_factory=list)


def hard_dtw_path(costs, vertical_penalty=.05, horizontal_penalty=.30,
                  position_prior=.15, disable_horizontal_when_feasible=True, *,
                  space_columns=None, space_skip_penalty=0.):
    """Hard counterpart of training Soft-DTW, including zero-window space skips.

    A skip advances text on the same visual row, pays the saved skip penalty,
    and emits no alignment cell. Ties prefer diagonal, vertical, horizontal,
    then skipping. Horizontal suppression uses the full transcript length,
    exactly as in the training recurrence.
    """
    cells = effective_costs(costs.detach().cpu().double(), position_prior).tolist()
    t, l = len(cells), len(cells[0])
    spaces = [False] * l
    if space_columns is not None:
        mask = torch.as_tensor(space_columns, device='cpu', dtype=torch.bool)
        if mask.shape != (l,):
            raise ValueError('space_columns must have one entry per transcript token')
        if not math.isfinite(space_skip_penalty) or space_skip_penalty <= 0:
            raise ValueError('space_skip_penalty must be finite and positive')
        spaces = mask.tolist()
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
            skipped = False
            if j > 1 and spaces[j-1] and dp[i][j-1] + space_skip_penalty < dp[i][j]:
                dp[i][j] = dp[i][j-1] + space_skip_penalty
                predecessor, skipped = (i, j-1), True
            back[i,j] = predecessor, skipped
    path, skipped_columns, position = [], [], (t,l)
    while position != (0,0):
        i,j = position
        predecessor, skipped = back[position]
        if skipped:
            skipped_columns.append(j-1)
        else:
            path.append((i-1,j-1))
        position = predecessor
    return HardDTW(path[::-1], dp[t][l], dp[t][l] / (t+l), skipped_columns[::-1])
