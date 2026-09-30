"""Positive letter-DTW, optional transcript margin, and pre-L2 ECF SIGReg."""
import torch
from collections import Counter
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F

from dtw import letter_cost_matrix, soft_dtw, alphabet_log_probabilities
from text_embedding import ARABIC_LETTERS, clean_letters


def _line_loss(visual, letters, text_embedding, config):
    inventory = list(dict.fromkeys(config.alphabet_inventory or
                     (ARABIC_LETTERS if config.alignment_objective=='ctc' else ARABIC_LETTERS + ''.join(letters))))
    lookup = {c: i for i, c in enumerate(inventory)}
    if any(c not in lookup for c in letters):
        raise AlignmentInfeasible('unsupported_character')
    if config.alignment_objective == 'ctc':
        required = len(letters) + sum(a == b for a, b in zip(letters, letters[1:]))
        if len(visual) < required:
            raise AlignmentInfeasible('ctc_repeated_label_feasibility')
        logp = alphabet_log_probabilities(visual, text_embedding.encode(''.join(inventory)),
                                         config.competition_temperature, config.ctc_blank_logit)
        targets = torch.tensor([lookup[c] for c in letters], device=visual.device)
        return F.ctc_loss(logp[:, None, :], targets, [len(visual)], [len(letters)],
                          blank=len(inventory), reduction='sum', zero_infinity=False) / len(letters)
    costs = letter_cost_matrix(visual, text_embedding.encode(''.join(letters)),
                              alphabet=text_embedding.encode(''.join(inventory)),
                              letter_ids=[lookup[c] for c in letters],
                              temperature=config.competition_temperature, mode=config.dtw_cost_mode)
    return soft_dtw(costs, config.dtw_gamma, config.vertical_penalty, config.horizontal_penalty,
                    config.position_prior, config.disable_horizontal_when_feasible)


class AlignmentInfeasible(ValueError):
    """A transcript cannot be explained by the configured sequence objective."""


def positive_dtw_loss(vectors, texts, text_embedding, config, token_valid=None, *, return_costs=False):
    if len(texts) != len(vectors):
        raise ValueError('One own transcript is required per image')
    values = []
    lengths = []
    costs, reasons = [], Counter()
    for index, text in enumerate(texts):
        letters = clean_letters(text)
        visual = vectors[index] if token_valid is None else vectors[index][token_valid[index]]
        if not letters or len(visual) == 0:
            reasons['empty_transcript' if not letters else 'no_valid_windows'] += 1
            costs.append(None)
            continue
        try:
            cost = _line_loss(visual, letters, text_embedding, config)
        except AlignmentInfeasible as exc:
            reasons[str(exc)] += 1
            costs.append(None)
            continue
        values.append(cost)
        costs.append(cost)
        lengths.append((len(visual), len(letters)))
    loss = torch.stack(values).mean() if values else (vectors if token_valid is None else vectors[token_valid]).sum() * 0.
    stats = dict(evaluated=len(values), skipped=len(texts)-len(values), skip_reasons=dict(reasons),
                 dtw_sum=sum(float(v.detach()) for v in values), lengths=lengths)
    return (loss, stats, costs) if return_costs else (loss, stats)


def negative_dtw_margin_loss(vectors, texts, negative_texts, text_embedding, config, token_valid=None,
                             *, positive_costs=None, return_stats=False):
    """Mean candidate hinge per line, then mean over ranked lines. Both costs differentiate."""
    if negative_texts is None or len(negative_texts) != len(texts):
        raise ValueError('Active negative loss requires explicit per-line negative transcripts')
    losses = []
    correct = candidates = 0
    margin_sum = 0.
    reasons = Counter()
    for i, text in enumerate(texts):
        letters = clean_letters(text)
        visual = vectors[i] if token_valid is None else vectors[i][token_valid[i]]
        negatives = list(dict.fromkeys(''.join(clean_letters(t)) for t in negative_texts[i]))
        negatives = [n for n in negatives if n and n != ''.join(letters)]
        if letters and negatives and len(visual):
            try:
                pos = positive_costs[i] if positive_costs is not None else _line_loss(visual, letters, text_embedding, config)
                if pos is None:
                    continue
                values = []
                for n in negatives:
                    try:
                        neg = _line_loss(visual, n, text_embedding, config)
                    except AlignmentInfeasible as exc:
                        reasons[str(exc)] += 1
                        continue
                    values.append(F.relu(config.negative_margin + pos - neg))
                    delta = float((neg-pos).detach())
                    margin_sum += delta
                    correct += delta > 0
                    candidates += 1
                if values:
                    losses.append(torch.stack(values).mean())
            except AlignmentInfeasible as exc:
                reasons[str(exc)] += 1
    loss = torch.stack(losses).mean() if losses else (vectors if token_valid is None else vectors[token_valid]).sum() * 0.
    stats = dict(negative_sum=sum(float(x.detach()) for x in losses), ranked_lines=len(losses),
                 ranking_candidates=candidates, ranking_correct=correct, ranking_margin_sum=margin_sum,
                 negative_skip_reasons=dict(reasons))
    return (loss, stats) if return_stats else loss


def sigreg_loss(embeddings, token_valid=None, *, sketch_dim=1024, num_knots=17,
                min_samples=32, slice_chunk=128, distributed_statistics=False, seed=None):
    """Sliced Epps-Pulley statistic: original t=[0,3] Gaussian-weighted quadrature.

    Uses valid pre-L2 h, float32, N times ECF discrepancy. DDP uses global
    moments, differentiable all-reduce and shared directions. Rank-zero evaluation
    explicitly passes distributed_statistics=False. A seed fixes evaluation sketches.
    """
    if embeddings.ndim != 3 or sketch_dim < 1 or num_knots < 3:
        raise ValueError('SIGReg requires [B,T,D], positive sketches and >=3 knots')
    distributed = distributed_statistics and dist.is_available() and dist.is_initialized()
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        values = embeddings.float()
        values = values.reshape(-1, values.shape[-1]) if token_valid is None else values[token_valid.bool()]
        count = values.new_tensor(float(len(values)))
        if distributed:
            dist.all_reduce(count)
        n = int(count.item())
        if n < min_samples:
            return values.sum() * 0.
        t = torch.linspace(0, 3, num_knots, device=values.device)
        dt = 3. / (num_knots - 1)
        weights = torch.full_like(t, 2 * dt)
        weights[0] = weights[-1] = dt
        gaussian = torch.exp(-.5 * t.square())
        weights = weights * gaussian
        generator = None if seed is None else torch.Generator(device=values.device).manual_seed(seed)
        total = values.new_zeros(())
        for start in range(0, sketch_dim, slice_chunk):
            directions = values.new_empty(values.shape[-1], min(slice_chunk, sketch_dim-start))
            if not distributed or dist.get_rank() == 0:
                directions.normal_(generator=generator)
                directions.div_(directions.norm(dim=0, keepdim=True).clamp_min(1e-6))
            if distributed:
                dist.broadcast(directions, src=0)
            args = (values @ directions).unsqueeze(-1) * t
            moments = torch.stack((args.cos().sum(0), args.sin().sum(0)))
            if distributed:
                moments = all_reduce(moments, op=dist.ReduceOp.SUM)
            error = (moments[0] / n - gaussian).square() + (moments[1] / n).square()
            total = total + ((error * weights).sum(dim=-1) * n).sum()
        return total / sketch_dim


def compute_loss(output, texts, text_embedding, config, *, negative_texts=None,
                 distributed_statistics=False, sketch_seed=None):
    vectors, valid = output['fused'], output['token_valid']
    positive, counts, costs = positive_dtw_loss(vectors, texts, text_embedding, config, valid, return_costs=True)
    positive_for_gradient = positive
    global_evaluated = counts['evaluated']
    if distributed_statistics and dist.is_available() and dist.is_initialized():
        count = vectors.new_tensor(float(counts['evaluated']))
        dist.all_reduce(count)
        global_evaluated = int(count.item())
        # DDP averages gradients: compensate for differing per-rank valid counts.
        positive_for_gradient = positive * counts['evaluated'] * dist.get_world_size() / count.clamp_min(1)
    negative = vectors[valid].sum() * 0.
    ranking = dict(negative_sum=0., ranked_lines=0, ranking_candidates=0, ranking_correct=0,
                   ranking_margin_sum=0., negative_skip_reasons={})
    if config.negative_dtw_weight:
        negative, ranking = negative_dtw_margin_loss(vectors, texts, negative_texts, text_embedding, config, valid,
                                                    positive_costs=costs, return_stats=True)
    negative_for_gradient = negative
    if config.negative_dtw_weight and distributed_statistics and dist.is_available() and dist.is_initialized():
        count = vectors.new_tensor(float(ranking['ranked_lines']))
        dist.all_reduce(count)
        negative_for_gradient = negative * ranking['ranked_lines'] * dist.get_world_size() / count.clamp_min(1)
    sigreg = vectors[valid].sum() * 0.
    if config.sigreg_weight:
        sigreg = sigreg_loss(output['fused_pre_l2'], valid, sketch_dim=config.sigreg_sketch_dim,
                             num_knots=config.sigreg_num_knots, min_samples=config.sigreg_min_samples,
                             distributed_statistics=distributed_statistics, seed=sketch_seed)
    total = (config.positive_dtw_weight * positive_for_gradient
             + config.negative_dtw_weight * negative_for_gradient
             + config.sigreg_weight * sigreg)
    stats = dict(total=float(total.detach()), positive_dtw=float(positive.detach()) if counts['evaluated'] else None,
                 negative_dtw=float(negative.detach()) if config.negative_dtw_weight else None,
                 sigreg=float(sigreg.detach()) if config.sigreg_weight else None,
                 weighted_sigreg=float(sigreg.detach()) * config.sigreg_weight,
                 valid_tokens=int(valid.sum()), global_evaluated=global_evaluated, **counts, **ranking)
    return total, stats
