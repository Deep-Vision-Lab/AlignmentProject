"""Positive letter-DTW, optional transcript margin, and pre-L2 ECF SIGReg."""
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F

from dtw import letter_cost_matrix, soft_dtw
from text_embedding import ARABIC_LETTERS, clean_letters


def _line_loss(visual, letters, text_embedding, config):
    inventory = list(dict.fromkeys(ARABIC_LETTERS + ''.join(letters)))
    lookup = {c: i for i, c in enumerate(inventory)}
    costs = letter_cost_matrix(visual, text_embedding.encode(''.join(letters)),
                              alphabet=text_embedding.encode(''.join(inventory)),
                              letter_ids=[lookup[c] for c in letters],
                              temperature=config.competition_temperature, mode=config.dtw_cost_mode)
    return soft_dtw(costs, config.dtw_gamma, config.vertical_penalty, config.horizontal_penalty,
                    config.position_prior, config.disable_horizontal_when_feasible)


def positive_dtw_loss(vectors, texts, text_embedding, config, token_valid=None):
    if len(texts) != len(vectors):
        raise ValueError('One own transcript is required per image')
    values = []
    lengths = []
    for index, text in enumerate(texts):
        letters = clean_letters(text)
        visual = vectors[index] if token_valid is None else vectors[index][token_valid[index]]
        if not letters or len(visual) == 0:
            continue
        values.append(_line_loss(visual, letters, text_embedding, config))
        lengths.append((len(visual), len(letters)))
    loss = torch.stack(values).mean() if values else vectors.sum() * 0.
    return loss, dict(evaluated=len(values), skipped=len(texts)-len(values),
                      dtw_sum=sum(float(v.detach()) for v in values), lengths=lengths)


def negative_dtw_margin_loss(vectors, texts, negative_texts, text_embedding, config, token_valid=None):
    """Optional hardest-negative transcript margin; never enabled implicitly."""
    if negative_texts is None or len(negative_texts) != len(texts):
        raise ValueError('Active negative loss requires explicit per-line negative transcripts')
    losses = []
    for i, text in enumerate(texts):
        letters = clean_letters(text)
        visual = vectors[i] if token_valid is None else vectors[i][token_valid[i]]
        negatives = [clean_letters(t) for t in negative_texts[i] if clean_letters(t)]
        if letters and negatives and len(visual):
            pos = _line_loss(visual, letters, text_embedding, config)
            neg = torch.stack([_line_loss(visual, n, text_embedding, config) for n in negatives]).min()
            losses.append(F.relu(config.negative_margin + pos - neg))
    return torch.stack(losses).mean() if losses else vectors.sum() * 0.


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
            return embeddings.float().sum() * 0.
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
    positive, counts = positive_dtw_loss(vectors, texts, text_embedding, config, valid)
    positive_for_gradient = positive
    if distributed_statistics and dist.is_available() and dist.is_initialized():
        count = vectors.new_tensor(float(counts['evaluated']))
        dist.all_reduce(count)
        # DDP averages gradients: compensate for differing per-rank valid counts.
        positive_for_gradient = positive * counts['evaluated'] * dist.get_world_size() / count.clamp_min(1)
    negative = vectors.sum() * 0.
    if config.negative_dtw_weight:
        negative = negative_dtw_margin_loss(vectors, texts, negative_texts, text_embedding, config, valid)
    sigreg = vectors.sum() * 0.
    if config.sigreg_weight:
        sigreg = sigreg_loss(output['fused_pre_l2'], valid, sketch_dim=config.sigreg_sketch_dim,
                             num_knots=config.sigreg_num_knots, min_samples=config.sigreg_min_samples,
                             distributed_statistics=distributed_statistics, seed=sketch_seed)
    total = (config.positive_dtw_weight * positive_for_gradient
             + config.negative_dtw_weight * negative
             + config.sigreg_weight * sigreg)
    stats = dict(total=float(total.detach()), positive_dtw=float(positive.detach()) if counts['evaluated'] else None,
                 negative_dtw=float(negative.detach()) if config.negative_dtw_weight else None,
                 sigreg=float(sigreg.detach()) if config.sigreg_weight else None,
                 weighted_sigreg=float(sigreg.detach()) * config.sigreg_weight,
                 valid_tokens=int(valid.sum()), **counts)
    return total, stats
