"""Positive letter-DTW, optional transcript margin, and pre-L2 ECF SIGReg."""
import torch
from collections import Counter
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F

from dtw import letter_cost_matrix, soft_dtw, alphabet_log_probabilities
from text_embedding import ARABIC_LETTERS, clean_letters


def _line_loss(visual, letters, text_embedding, config, *, return_occupancy=False,
               inventory_override=None):
    inventory = list(dict.fromkeys(inventory_override or config.alphabet_inventory or
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
                    config.position_prior, config.disable_horizontal_when_feasible,
                    normalization=config.dtw_normalization, return_occupancy=return_occupancy)


class AlignmentInfeasible(ValueError):
    """A transcript cannot be explained by the configured sequence objective."""


def positive_dtw_loss(vectors, texts, text_embedding, config, token_valid=None, *, return_costs=False,
                      return_alignments=False):
    if len(texts) != len(vectors):
        raise ValueError('One own transcript is required per image')
    values = []
    lengths = []
    costs, alignments, reasons = [], [], Counter()
    for index, text in enumerate(texts):
        letters = clean_letters(text)
        visual = vectors[index] if token_valid is None else vectors[index][token_valid[index]]
        if not letters or len(visual) == 0:
            reasons['empty_transcript' if not letters else 'no_valid_windows'] += 1
            costs.append(None)
            alignments.append(None)
            continue
        try:
            result = _line_loss(visual, letters, text_embedding, config,
                                return_occupancy=True) if return_alignments else _line_loss(visual, letters, text_embedding, config)
            cost, occupancy = result if return_alignments else (result, None)
        except AlignmentInfeasible as exc:
            reasons[str(exc)] += 1
            costs.append(None)
            alignments.append(None)
            continue
        values.append(cost)
        costs.append(cost)
        alignments.append(occupancy)
        lengths.append((len(visual), len(letters)))
    loss = torch.stack(values).mean() if values else (vectors if token_valid is None else vectors[token_valid]).sum() * 0.
    stats = dict(evaluated=len(values), skipped=len(texts)-len(values), skip_reasons=dict(reasons),
                 dtw_sum=sum(float(v.detach()) for v in values),
                 positive_cost_min=min((float(v.detach()) for v in values), default=None),
                 positive_cost_max=max((float(v.detach()) for v in values), default=None),
                 positive_below_reference=sum(float(v.detach()) < config.negative_target_min for v in values),
                 lengths=lengths)
    if return_alignments:
        return loss, stats, costs, alignments
    return (loss, stats, costs) if return_costs else (loss, stats)


def negative_dtw_margin_loss(vectors, texts, negative_texts, text_embedding, config, token_valid=None,
                             *, positive_costs=None, return_stats=False):
    """Mean candidate hinge per line, then mean over ranked lines. Both costs differentiate."""
    if negative_texts is None or len(negative_texts) != len(texts):
        raise ValueError('Active negative loss requires explicit per-line negative transcripts')
    losses = []
    correct = candidates = hard_negatives = 0
    margin_sum = negative_cost_sum = 0.
    negative_cost_min, negative_cost_max = float('inf'), float('-inf')
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
                    negative_cost = float(neg.detach())
                    negative_cost_sum += negative_cost
                    negative_cost_min = min(negative_cost_min, negative_cost)
                    negative_cost_max = max(negative_cost_max, negative_cost)
                    hard_negatives += float(values[-1].detach()) > 0
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
                 negative_cost_sum=negative_cost_sum,
                 negative_cost_min=negative_cost_min if candidates else None,
                 negative_cost_max=negative_cost_max if candidates else None,
                 hard_negatives=hard_negatives,
                 negative_skip_reasons=dict(reasons))
    return (loss, stats) if return_stats else loss


def absolute_rejection_loss(energies, ratios, *, target_min, target_max, softness, hard_k):
    """Reject the lowest-energy candidates; selection is by cost, not loss size."""
    if energies.ndim != 1 or len(energies) != len(ratios) or not 1 <= hard_k <= len(energies):
        raise ValueError('Require one ratio per energy and 1 <= hard_k <= number of energies')
    ratios = torch.as_tensor(ratios, device=energies.device, dtype=energies.dtype)
    if not torch.isfinite(ratios).all() or (ratios < 0).any() or (ratios > 1).any():
        raise ValueError('Corruption ratios must be finite and in [0,1]')
    indices = torch.topk(energies.detach(), hard_k, largest=False).indices
    targets = target_min + ratios * (target_max - target_min)
    loss = softness * F.softplus((targets[indices] - energies[indices]) / softness).mean()
    return loss, indices, targets


def negative_sequence_objective(vectors, texts, negative_texts, text_embedding, config,
                                token_valid=None, *, positive_costs, positive_alignments=None,
                                negative_metadata=None):
    """Per-line absolute rejection of lowest-energy negatives; old hinge is diagnostic/auxiliary.

    No candidate is assigned a visual-window label. The optional substitution
    unlikelihood uses the POSITIVE soft alignment occupancy at the known
    substituted transcript column, never a hard guessed window index.
    """
    if negative_texts is None or len(negative_texts) != len(texts):
        raise ValueError('Active negative loss requires explicit per-line negative transcripts')
    zero = vectors[token_valid].sum() * 0 if token_valid is not None else vectors.sum() * 0
    line_losses, absolute_losses, ranking_losses, wrong_losses = [], [], [], []
    costs_seen, hard_seen, targets_seen = [], [], []
    stats = Counter()
    reasons = Counter()
    operations = Counter()
    for i, text in enumerate(texts):
        positive = positive_costs[i]
        if positive is None:
            continue
        original = ''.join(clean_letters(text))
        visual = vectors[i] if token_valid is None else vectors[i][token_valid[i]]
        entries = (negative_metadata[i] if negative_metadata is not None else None)
        if entries is not None and len(entries) != len(negative_texts[i]):
            raise ValueError('Negative metadata must match negative transcript count')
        candidates = []
        seen = set()
        for j, candidate in enumerate(negative_texts[i]):
            candidate = ''.join(clean_letters(candidate))
            if not candidate or candidate == original or candidate in seen:
                reasons['empty_identical_or_duplicate'] += 1
                continue
            seen.add(candidate)
            try:
                fixed_inventory = list(dict.fromkeys(config.alphabet_inventory or ARABIC_LETTERS + original))
                energy = (_line_loss(visual, list(candidate), text_embedding, config,
                                     inventory_override=fixed_inventory)
                          if config.negative_loss_type == 'absolute' else
                          _line_loss(visual, list(candidate), text_embedding, config))
            except AlignmentInfeasible as exc:
                reasons[str(exc)] += 1
                continue
            meta = entries[j] if entries is not None else {}
            ratio = meta.get('corruption_ratio',
                             sum(a != b for a, b in zip(original, candidate)) / max(len(original), len(candidate)))
            if not 0 <= ratio <= 1:
                raise ValueError('Negative corruption_ratio must be within [0,1]')
            target = config.negative_target_min + ratio * (config.negative_target_max - config.negative_target_min)
            candidates.append((energy, target, meta, candidate, ratio))
            costs_seen.append(float(energy.detach()))
            targets_seen.append(float(target))
            stats['gap_sum'] += float((energy - positive).detach())
            stats['ranking_correct'] += float((energy > positive).detach())
            stats['target_success'] += float((energy >= target).detach())
            stats['margin_violations'] += float((energy < positive + config.negative_margin).detach())
            if meta.get('operation'):
                operations[meta['operation']] += 1
        if not candidates:
            continue
        values = torch.stack([item[0] for item in candidates])
        hinge = F.relu(config.negative_margin + positive - values).mean()
        ranking_losses.append(hinge)
        absolute, selected, _ = absolute_rejection_loss(
            values, [item[4] for item in candidates],
            target_min=config.negative_target_min, target_max=config.negative_target_max,
            softness=config.negative_softness,
            hard_k=min(config.hard_negative_k, len(values)))
        hard_indices = selected.tolist()
        hard = [candidates[k] for k in hard_indices]
        hard_seen.extend(float(item[0].detach()) for item in hard)
        absolute_losses.append(absolute)
        wrong = zero
        if config.wrong_letter_unlikelihood_weight:
            occupancy = positive_alignments[i]
            inventory = list(dict.fromkeys(config.alphabet_inventory or ARABIC_LETTERS + original))
            lookup = {c: k for k, c in enumerate(inventory)}
            logits = alphabet_log_probabilities(visual, text_embedding.encode(''.join(inventory)),
                                                config.competition_temperature)
            terms = []
            for _, _, meta, candidate, _ in candidates:
                if meta.get('operation') != 'substitute' or not meta.get('changed_positions'):
                    continue
                for column in meta['changed_positions']:
                    if not 0 <= column < len(original) or len(candidate) != len(original):
                        continue
                    letter = candidate[column]
                    if letter == original[column] or letter not in lookup:
                        continue
                    weights = occupancy[:, column]
                    probability = logits[:, lookup[letter]].exp().clamp(max=1-1e-6)
                    terms.append((weights * -torch.log1p(-probability)).sum() / weights.sum().clamp_min(1e-8))
            if terms:
                wrong = torch.stack(terms).mean()
        wrong_losses.append(wrong)
        active = hinge if config.negative_loss_type == 'ranking' else absolute
        line_losses.append(active)
        stats['ranked_lines'] += 1
        stats['ranking_candidates'] += len(candidates)
    def average(values):
        return torch.stack(values).mean() if values else zero
    objective = average(line_losses)
    absolute = average(absolute_losses)
    ranking = average(ranking_losses)
    wrong = average(wrong_losses)
    detail = dict(objective_sum=sum(float(x.detach()) for x in line_losses),
                  absolute_sum=sum(float(x.detach()) for x in absolute_losses),
                  ranking_sum=sum(float(x.detach()) for x in ranking_losses),
                  wrong_sum=sum(float(x.detach()) for x in wrong_losses),
                  ranked_lines=stats['ranked_lines'], ranking_candidates=stats['ranking_candidates'],
                  ranking_correct=stats['ranking_correct'], ranking_margin_sum=stats['gap_sum'],
                  target_success=stats['target_success'], margin_violations=stats['margin_violations'],
                  negative_cost_sum=sum(costs_seen),
                  negative_cost_min=min(costs_seen, default=None),
                  negative_cost_max=max(costs_seen, default=None),
                  hard_cost_sum=sum(hard_seen), hard_cost_min=min(hard_seen, default=None),
                  hard_count=len(hard_seen), target_sum=sum(targets_seen),
                  negative_skip_reasons=dict(reasons), operations=dict(operations),
                  hard_negatives=stats['margin_violations'],
                  negative_sum=sum(float(x.detach()) for x in line_losses),
                  absolute_negative_loss=float(absolute.detach()) if line_losses else None,
                  ranking_loss=float(ranking.detach()) if line_losses else None,
                  wrong_letter_loss=float(wrong.detach()) if line_losses else None)
    return objective, ranking, wrong, detail


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


def compute_loss(output, texts, text_embedding, config, *, negative_texts=None, negative_metadata=None,
                 distributed_statistics=False, sketch_seed=None):
    vectors, valid = output['fused'], output['token_valid']
    if config.wrong_letter_unlikelihood_weight and config.negative_dtw_weight:
        positive, counts, costs, alignments = positive_dtw_loss(
            vectors, texts, text_embedding, config, valid, return_alignments=True)
    else:
        positive, counts, costs = positive_dtw_loss(vectors, texts, text_embedding, config, valid, return_costs=True)
        alignments = None
    positive_for_gradient = positive
    global_evaluated = counts['evaluated']
    if distributed_statistics and dist.is_available() and dist.is_initialized():
        count = vectors.new_tensor(float(counts['evaluated']))
        dist.all_reduce(count)
        global_evaluated = int(count.item())
        # DDP averages gradients: compensate for differing per-rank valid counts.
        positive_for_gradient = positive * counts['evaluated'] * dist.get_world_size() / count.clamp_min(1)
    negative = vectors[valid].sum() * 0.
    ranking = dict(objective_sum=0., absolute_sum=0., ranking_sum=0., wrong_sum=0.,
                   ranked_lines=0, ranking_candidates=0, ranking_correct=0,
                   ranking_margin_sum=0., negative_cost_sum=0., negative_cost_min=None,
                   negative_cost_max=None, hard_cost_sum=0., hard_cost_min=None, hard_count=0,
                   target_sum=0., target_success=0., margin_violations=0.,
                   negative_skip_reasons={}, operations={}, absolute_negative_loss=None,
                   ranking_loss=None, wrong_letter_loss=None)
    if config.negative_dtw_weight:
        negative, ranking_aux, wrong_aux, ranking = negative_sequence_objective(
            vectors, texts, negative_texts, text_embedding, config, valid,
            positive_costs=costs, positive_alignments=alignments,
            negative_metadata=negative_metadata)
    else:
        ranking_aux = wrong_aux = negative
    negative_for_gradient = negative
    ranking_for_gradient = ranking_aux
    wrong_for_gradient = wrong_aux
    global_ranking_candidates = ranking['ranking_candidates']
    if config.negative_dtw_weight and distributed_statistics and dist.is_available() and dist.is_initialized():
        counts_global = vectors.new_tensor([ranking['ranked_lines'], ranking['ranking_candidates']])
        dist.all_reduce(counts_global)
        factor = ranking['ranked_lines'] * dist.get_world_size() / counts_global[0].clamp_min(1)
        negative_for_gradient = negative * factor
        ranking_for_gradient = ranking_aux * factor
        wrong_for_gradient = wrong_aux * factor
        global_ranking_candidates = int(counts_global[1].item())
    sigreg = vectors[valid].sum() * 0.
    if config.sigreg_weight:
        sigreg = sigreg_loss(output['fused_pre_l2'], valid, sketch_dim=config.sigreg_sketch_dim,
                             num_knots=config.sigreg_num_knots, min_samples=config.sigreg_min_samples,
                             distributed_statistics=distributed_statistics, seed=sketch_seed)
    total = (config.positive_dtw_weight * positive_for_gradient
             + config.negative_dtw_weight * negative_for_gradient
             + config.ranking_aux_weight * ranking_for_gradient
             + config.wrong_letter_unlikelihood_weight * wrong_for_gradient
             + config.sigreg_weight * sigreg)
    stats = dict(total=float(total.detach()), positive_dtw=float(positive.detach()) if counts['evaluated'] else None,
                 positive_cost=float(positive.detach()) if counts['evaluated'] else None,
                 negative_objective=float(negative.detach()) if config.negative_dtw_weight else None,
                 # Deprecated machine-readable alias for old downstream consumers only.
                 negative_dtw=float(negative.detach()) if config.negative_loss_type == 'ranking' and config.negative_dtw_weight else None,
                 sigreg=float(sigreg.detach()) if config.sigreg_weight else None,
                 weighted_sigreg=float(sigreg.detach()) * config.sigreg_weight,
                 valid_tokens=int(valid.sum()), global_evaluated=global_evaluated,
                 global_ranking_candidates=global_ranking_candidates, **counts, **ranking)
    return total, stats
