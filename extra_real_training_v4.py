"""Expanded-real training with direct local sequence-alignment ranking.

This objective operates on the same normalized local image-window embeddings used
by the Smith-Waterman evaluator. Positive high/medium pairs are ranked above
``no_shared_content`` pairs using a differentiable soft approximation of local
Smith-Waterman score and matched-path fraction.

The established image-text, positive span-pair, and order losses remain active.
"""
from __future__ import annotations

import os

import torch

import extra_real_training as legacy
import extra_real_training_v2_absolute as shared


def _flag(name, default):
    value = os.environ.get(name)
    return bool(default) if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


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


def _zero_stats(reference):
    zero = reference.new_tensor(0.0)
    return zero, {
        "sequence_rank_loss": 0.0,
        "sequence_path_rank_loss": 0.0,
        "sequence_score_rank_loss": 0.0,
        "sequence_positive_floor_loss": 0.0,
        "sequence_pos_fraction": 0.0,
        "sequence_neg_fraction": 0.0,
        "sequence_hard_fraction_gap": 0.0,
        "sequence_pos_score": 0.0,
        "sequence_neg_score": 0.0,
        "sequence_hard_score_gap": 0.0,
        "sequence_positive_samples": 0.0,
        "sequence_negative_samples": 0.0,
        "sequence_active": 0.0,
    }


def _lookup(previous, previous_start, target_i, valid, reference):
    if previous is None or previous.numel() == 0:
        return reference.new_zeros(target_i.shape)
    indices = target_i - int(previous_start)
    safe = indices.clamp(0, previous.numel() - 1)
    values = previous.index_select(0, safe)
    return torch.where(valid, values, torch.zeros_like(values))


def soft_local_alignment_metrics(similarity, threshold, gap_penalty, temperature):
    """Return soft SW score and soft diagonal-step fraction."""
    if similarity.ndim != 2:
        raise ValueError(f"Expected 2D similarity, got {tuple(similarity.shape)}")
    n, m = map(int, similarity.shape)
    if n <= 0 or m <= 0:
        zero = similarity.new_tensor(0.0)
        return zero, zero

    tau = max(float(temperature), 1e-4)
    prev2_score = prev2_len = None
    prev2_start = 0
    prev1_score = prev1_len = None
    prev1_start = 0
    all_scores, all_lengths = [], []

    # Cells on one anti-diagonal are conditionally independent because their
    # diag/up/left predecessors live on the previous two anti-diagonals.
    for diagonal in range(2, n + m + 1):
        start_i = max(1, diagonal - m)
        end_i = min(n, diagonal - 1)
        i = torch.arange(
            start_i, end_i + 1, device=similarity.device, dtype=torch.long
        )
        j = diagonal - i
        reference = similarity[i - 1, j - 1]

        diag_valid = (i > 1) & (j > 1)
        up_valid = i > 1
        left_valid = j > 1

        diag_score = _lookup(prev2_score, prev2_start, i - 1, diag_valid, reference)
        up_score = _lookup(prev1_score, prev1_start, i - 1, up_valid, reference)
        left_score = _lookup(prev1_score, prev1_start, i, left_valid, reference)
        diag_len = _lookup(prev2_len, prev2_start, i - 1, diag_valid, reference)
        up_len = _lookup(prev1_len, prev1_start, i - 1, up_valid, reference)
        left_len = _lookup(prev1_len, prev1_start, i, left_valid, reference)

        match = reference - float(threshold)
        candidates = torch.stack(
            (
                torch.zeros_like(match),
                diag_score + match,
                up_score + float(gap_penalty),
                left_score + float(gap_penalty),
            ),
            dim=0,
        )
        probabilities = torch.softmax(candidates / tau, dim=0)
        current_score = (probabilities * candidates).sum(dim=0)
        current_len = (
            probabilities[1] * (diag_len + 1.0)
            + probabilities[2] * up_len
            + probabilities[3] * left_len
        )

        all_scores.append(current_score)
        all_lengths.append(current_len)
        prev2_score, prev2_len, prev2_start = prev1_score, prev1_len, prev1_start
        prev1_score, prev1_len, prev1_start = current_score, current_len, start_i

    scores = torch.cat(all_scores)
    lengths = torch.cat(all_lengths)
    terminal_probabilities = torch.softmax(scores / tau, dim=0)
    best_score = (terminal_probabilities * scores).sum()
    expected_steps = (terminal_probabilities * lengths).sum()
    return best_score, expected_steps / float(max(1, min(n, m)))


def _sample_metrics(norm_local1, norm_local2, index):
    similarity = torch.matmul(norm_local1[index], norm_local2[index].T)
    return soft_local_alignment_metrics(
        similarity,
        _env_float("SEQUENCE_RANKING_THRESHOLD", 0.65),
        _env_float("SEQUENCE_RANKING_GAP", -0.30),
        _env_float("SEQUENCE_RANKING_TEMPERATURE", 0.03),
    )


def sequence_ranking_loss(base, emb1, emb2, labels):
    if (
        not _flag("USE_SEQUENCE_ALIGNMENT_RANKING", True)
        or _env_float("SEQUENCE_RANKING_WEIGHT", 0.10) <= 0
        or not torch.is_grad_enabled()
    ):
        return _zero_stats(emb1[0])

    positive_indices = [
        i for i, label in enumerate(labels)
        if str(label) in legacy.POSITIVE_LABELS
    ]
    negative_indices = [
        i for i, label in enumerate(labels)
        if str(label) == legacy.EXTRA_LABEL
    ]
    max_pos = _env_int("SEQUENCE_RANKING_MAX_POS_SAMPLES_PER_BATCH", 4)
    max_neg = _env_int("SEQUENCE_RANKING_MAX_NEG_SAMPLES_PER_BATCH", 4)
    if max_pos > 0:
        positive_indices = positive_indices[:max_pos]
    if max_neg > 0:
        negative_indices = negative_indices[:max_neg]
    if not positive_indices or not negative_indices:
        return _zero_stats(emb1[0])

    _ctx1, norm_local1, _ink1, _raw1 = emb1
    _ctx2, norm_local2, _ink2, _raw2 = emb2

    positive = [
        _sample_metrics(norm_local1, norm_local2, i) for i in positive_indices
    ]
    negative = [
        _sample_metrics(norm_local1, norm_local2, i) for i in negative_indices
    ]
    pos_scores = torch.stack([item[0] for item in positive])
    pos_fractions = torch.stack([item[1] for item in positive])
    neg_scores = torch.stack([item[0] for item in negative])
    neg_fractions = torch.stack([item[1] for item in negative])

    # Pair every positive against every no-shared sample. This directly attacks
    # distribution overlap (AUROC), rather than only separating batch means.
    path_loss = torch.relu(
        _env_float("SEQUENCE_RANKING_PATH_MARGIN", 0.03)
        - pos_fractions[:, None]
        + neg_fractions[None, :]
    ).mean()
    score_loss = torch.relu(
        _env_float("SEQUENCE_RANKING_SCORE_MARGIN", 0.25)
        - pos_scores[:, None]
        + neg_scores[None, :]
    ).mean()
    positive_floor_loss = torch.relu(
        _env_float("SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR", 0.15)
        - pos_fractions
    ).mean()

    loss = (
        _env_float("SEQUENCE_RANKING_PATH_COMPONENT_WEIGHT", 1.0) * path_loss
        + _env_float("SEQUENCE_RANKING_SCORE_COMPONENT_WEIGHT", 0.20) * score_loss
        + _env_float("SEQUENCE_RANKING_POSITIVE_FLOOR_WEIGHT", 0.50)
        * positive_floor_loss
    )

    return loss, {
        "sequence_rank_loss": float(loss.detach().item()),
        "sequence_path_rank_loss": float(path_loss.detach().item()),
        "sequence_score_rank_loss": float(score_loss.detach().item()),
        "sequence_positive_floor_loss": float(positive_floor_loss.detach().item()),
        "sequence_pos_fraction": float(pos_fractions.mean().detach().item()),
        "sequence_neg_fraction": float(neg_fractions.mean().detach().item()),
        "sequence_hard_fraction_gap": float(
            (pos_fractions.min() - neg_fractions.max()).detach().item()
        ),
        "sequence_pos_score": float(pos_scores.mean().detach().item()),
        "sequence_neg_score": float(neg_scores.mean().detach().item()),
        "sequence_hard_score_gap": float(
            (pos_scores.min() - neg_scores.max()).detach().item()
        ),
        "sequence_positive_samples": float(pos_scores.numel()),
        "sequence_negative_samples": float(neg_scores.numel()),
        "sequence_active": 1.0,
    }


def install(base):
    legacy.install(base)
    original_model_config = base.model_config

    def model_config(stride, args):
        config = dict(original_model_config(stride, args))
        config.update(
            {
                "no_shared_image_objective": "sequence_ranking",
                "sequence_ranking_weight": _env_float(
                    "SEQUENCE_RANKING_WEIGHT", 0.10
                ),
                "sequence_ranking_threshold": _env_float(
                    "SEQUENCE_RANKING_THRESHOLD", 0.65
                ),
                "sequence_ranking_gap": _env_float(
                    "SEQUENCE_RANKING_GAP", -0.30
                ),
                "sequence_ranking_temperature": _env_float(
                    "SEQUENCE_RANKING_TEMPERATURE", 0.03
                ),
                "sequence_ranking_path_margin": _env_float(
                    "SEQUENCE_RANKING_PATH_MARGIN", 0.03
                ),
                "sequence_ranking_score_margin": _env_float(
                    "SEQUENCE_RANKING_SCORE_MARGIN", 0.25
                ),
                "sequence_ranking_positive_fraction_floor": _env_float(
                    "SEQUENCE_RANKING_POSITIVE_FRACTION_FLOOR", 0.15
                ),
            }
        )
        return config

    base.model_config = model_config

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        if not isinstance(batch, dict):
            return base._ORIGINAL_COMPUTE_BATCH_LOSS(
                image_embedder, text_encoder, criterion, batch
            )

        if torch.is_grad_enabled():
            base._BATCH_COUNTER += 1
        local_every = max(
            1,
            base._env_int(
                "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
                getattr(base.P, "local_hard_negative_every_n_batches", 1),
            ),
        )
        local_enabled = torch.is_grad_enabled() and (
            base._BATCH_COUNTER % local_every == 0
        )

        images1 = batch["images1"].to(base.P.device, non_blocking=True)
        images2 = batch["images2"].to(base.P.device, non_blocking=True)
        texts1, texts2 = batch["texts1"], batch["texts2"]
        neg_texts1, neg_texts2 = batch["neg_texts1"], batch["neg_texts2"]
        labels = batch.get("pair_labels") or [
            legacy.EXTRA_LABEL if not bool(value) else "high_match"
            for value in batch.get("pair_positive_mask", [])
        ]

        if base.P.text_encoder_type == "arabic_span":
            emb1 = base.compute_embeddings(image_embedder, images1)
            emb2 = base.compute_embeddings(image_embedder, images2)
        else:
            emb1 = emb2 = None

        loss1, stats1, emb1 = base.compute_single_image_text_loss(
            image_embedder,
            text_encoder,
            criterion,
            images1,
            texts1,
            neg_texts1,
            emb1,
            local_enabled=local_enabled,
        )
        if bool(getattr(base.P, "image_text_loss_on_both_lines", True)):
            loss2, stats2, emb2 = base.compute_single_image_text_loss(
                image_embedder,
                text_encoder,
                criterion,
                images2,
                texts2,
                neg_texts2,
                emb2,
                local_enabled=local_enabled,
            )
            loss = 0.5 * (loss1 + loss2)
            stats = base.average_stats([stats1, stats2])
        else:
            loss, stats = loss1, dict(stats1)

        pair_every = max(
            1,
            base._env_int(
                "IMAGE_PAIR_EVERY_N_BATCHES",
                getattr(base.P, "image_pair_every_n_batches", 1),
            ),
        )
        pair_enabled = torch.is_grad_enabled() and (
            base._BATCH_COUNTER % pair_every == 0
        )
        if pair_enabled and emb1 is not None and emb2 is not None:
            pair_loss, order_loss, pair_stats = shared._positive_pair_loss(
                base,
                text_encoder,
                criterion,
                texts1,
                texts2,
                emb1,
                emb2,
                labels,
            )
            loss = loss + base.P.image_pair_loss_weight * pair_loss
            if base.P.sequence_consistency_loss_weight > 0:
                loss = loss + base.P.sequence_consistency_loss_weight * order_loss

            seq_loss, seq_stats = sequence_ranking_loss(base, emb1, emb2, labels)
            loss = loss + _env_float("SEQUENCE_RANKING_WEIGHT", 0.10) * seq_loss
            stats.update(pair_stats)
            stats.update(seq_stats)

            if base.CTX.is_main and seq_stats.get("sequence_active", 0.0) > 0:
                print(
                    "sequence_batch "
                    f"loss={seq_stats['sequence_rank_loss']:.4f} "
                    f"path_loss={seq_stats['sequence_path_rank_loss']:.4f} "
                    f"score_loss={seq_stats['sequence_score_rank_loss']:.4f} "
                    f"pos_frac={seq_stats['sequence_pos_fraction']:.4f} "
                    f"neg_frac={seq_stats['sequence_neg_fraction']:.4f} "
                    f"hard_frac_gap={seq_stats['sequence_hard_fraction_gap']:.4f} "
                    f"pos_score={seq_stats['sequence_pos_score']:.4f} "
                    f"neg_score={seq_stats['sequence_neg_score']:.4f} "
                    f"hard_score_gap={seq_stats['sequence_hard_score_gap']:.4f}",
                    flush=True,
                )
        else:
            stats.update(
                {
                    "image_pair_loss": 0.0,
                    "order_loss": 0.0,
                    "pair_terms": 0.0,
                    "pair_positive_samples": 0.0,
                }
            )
            stats.update(_zero_stats(images1)[1])

        stats["total"] = float(loss.detach().item())
        return loss, stats

    base.compute_batch_loss = compute_batch_loss

    if getattr(base.CTX, "is_main", True):
        print(
            "Expanded-real image-pair objective: "
            f"objective=sequence_ranking "
            f"weight={_env_float('SEQUENCE_RANKING_WEIGHT', 0.10):.3f} "
            f"threshold={_env_float('SEQUENCE_RANKING_THRESHOLD', 0.65):.3f} "
            f"gap={_env_float('SEQUENCE_RANKING_GAP', -0.30):.3f} "
            f"temperature={_env_float('SEQUENCE_RANKING_TEMPERATURE', 0.03):.3f}",
            flush=True,
        )
