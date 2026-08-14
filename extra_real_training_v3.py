"""Expanded-real training with relative positive-vs-no-shared ranking.

Keep the established image-text, positive image-image, and order objectives. Add
only a relative image-image constraint:

    true positive similarity >= hard no-shared similarity + margin

Equal-text groups inside ``no_shared_content`` pairs are never negatives.
"""
from __future__ import annotations

import os

import torch

import extra_real_training as legacy
import extra_real_training_v2 as shared


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _zero_stats(reference: torch.Tensor) -> tuple[torch.Tensor, dict]:
    zero = reference.new_tensor(0.0)
    return zero, {
        "no_shared_rank_loss": 0.0,
        "positive_pair_similarity": 0.0,
        "no_shared_hard_similarity": 0.0,
        "no_shared_max_similarity": 0.0,
        "ranking_gap": 0.0,
        "ranking_positive_samples": 0.0,
        "ranking_negative_samples": 0.0,
        "ranking_negative_terms": 0.0,
        "ranking_active": 0.0,
    }


def _groups(base, text_encoder, criterion, text, embeddings, index, min_chars):
    norm_ctx, norm_loc, ink, _raw = embeddings
    regions = base.extract_aligned_span_regions(
        text_encoder,
        criterion,
        text,
        norm_ctx[index],
        norm_loc[index],
        ink[index],
    )
    return shared._eligible_groups(base, regions, min_chars)


def _same_text_scores(groups_a, groups_b):
    if not groups_a or not groups_b:
        return []
    vectors_a = torch.stack([group["vec"] for group in groups_a])
    vectors_b = torch.stack([group["vec"] for group in groups_b])
    similarities = torch.matmul(vectors_a, vectors_b.T)
    scores = []
    for index_a, group_a in enumerate(groups_a):
        matches = [
            index_b
            for index_b, group_b in enumerate(groups_b)
            if group_b["text"] == group_a["text"]
        ]
        if matches:
            scores.append(similarities[index_a, matches].max())
    return scores


def _positive_score(base, text_encoder, criterion, texts1, texts2, emb1, emb2, index, min_chars):
    groups1 = _groups(base, text_encoder, criterion, texts1[index], emb1, index, min_chars)
    groups2 = _groups(base, text_encoder, criterion, texts2[index], emb2, index, min_chars)
    scores = _same_text_scores(groups1, groups2) + _same_text_scores(groups2, groups1)
    if not scores:
        return None
    return torch.stack(scores).mean()


def _negative_score(base, text_encoder, criterion, texts1, texts2, emb1, emb2, index, top_k, min_chars):
    groups1 = _groups(base, text_encoder, criterion, texts1[index], emb1, index, min_chars)
    groups2 = _groups(base, text_encoder, criterion, texts2[index], emb2, index, min_chars)
    if not groups1 or not groups2:
        return None
    vectors1 = torch.stack([group["vec"] for group in groups1])
    vectors2 = torch.stack([group["vec"] for group in groups2])
    similarities = torch.matmul(vectors1, vectors2.T)
    mismatch = torch.tensor(
        [
            [group1["text"] != group2["text"] for group2 in groups2]
            for group1 in groups1
        ],
        dtype=torch.bool,
        device=similarities.device,
    )
    pool = similarities[mismatch]
    if pool.numel() == 0:
        return None
    k = min(max(1, int(top_k)), int(pool.numel()))
    hard = torch.topk(pool, k=k).values
    return hard.mean(), hard.max(), k


def ranking_loss(base, text_encoder, criterion, texts1, texts2, emb1, emb2, labels):
    if (
        not _flag("USE_NO_SHARED_IMAGE_RANKING", True)
        or _env_float("NO_SHARED_RANKING_WEIGHT", 0.20) <= 0
        or base.P.text_encoder_type != "arabic_span"
        or not torch.is_grad_enabled()
    ):
        return _zero_stats(emb1[0])

    positive_indices = [i for i, label in enumerate(labels) if str(label) in legacy.POSITIVE_LABELS]
    negative_indices = [i for i, label in enumerate(labels) if str(label) == legacy.EXTRA_LABEL]
    max_pos = _env_int("NO_SHARED_RANKING_MAX_POS_SAMPLES_PER_BATCH", 8)
    max_neg = _env_int("NO_SHARED_RANKING_MAX_NEG_SAMPLES_PER_BATCH", 8)
    positive_indices = positive_indices[:max_pos] if max_pos > 0 else positive_indices
    negative_indices = negative_indices[:max_neg] if max_neg > 0 else negative_indices
    if not positive_indices or not negative_indices:
        return _zero_stats(emb1[0])

    margin = _env_float("NO_SHARED_RANKING_MARGIN", 0.10)
    top_k = _env_int("NO_SHARED_RANKING_TOP_K", 8)
    min_chars = _env_int("NO_SHARED_RANKING_MIN_CHARS", 2)

    positive_scores = []
    for index in positive_indices:
        try:
            score = _positive_score(
                base, text_encoder, criterion, texts1, texts2, emb1, emb2, index, min_chars
            )
        except ValueError:
            continue
        if score is not None:
            positive_scores.append(score)

    negative_scores = []
    negative_maxes = []
    terms = 0
    for index in negative_indices:
        try:
            result = _negative_score(
                base, text_encoder, criterion, texts1, texts2, emb1, emb2, index, top_k, min_chars
            )
        except ValueError:
            continue
        if result is None:
            continue
        hard_mean, hard_max, sample_terms = result
        negative_scores.append(hard_mean)
        negative_maxes.append(hard_max)
        terms += int(sample_terms)

    if not positive_scores or not negative_scores:
        return _zero_stats(emb1[0])

    positive_similarity = torch.stack(positive_scores).mean()
    negative_similarity = torch.stack(negative_scores).mean()
    gap = positive_similarity - negative_similarity
    loss = torch.relu(float(margin) - gap)
    return loss, {
        "no_shared_rank_loss": float(loss.detach().item()),
        "positive_pair_similarity": float(positive_similarity.detach().item()),
        "no_shared_hard_similarity": float(negative_similarity.detach().item()),
        "no_shared_max_similarity": float(torch.stack(negative_maxes).max().detach().item()),
        "ranking_gap": float(gap.detach().item()),
        "ranking_positive_samples": float(len(positive_scores)),
        "ranking_negative_samples": float(len(negative_scores)),
        "ranking_negative_terms": float(terms) / len(negative_scores),
        "ranking_active": 1.0,
    }


def install(base) -> None:
    """Install expanded-real loading plus the relative ranking objective."""
    legacy.install(base)
    original_model_config = base.model_config

    def model_config(stride, args):
        config = dict(original_model_config(stride, args))
        config.update(
            {
                "no_shared_image_objective": "ranking",
                "use_no_shared_image_ranking": _flag("USE_NO_SHARED_IMAGE_RANKING", True),
                "no_shared_ranking_weight": _env_float("NO_SHARED_RANKING_WEIGHT", 0.20),
                "no_shared_ranking_margin": _env_float("NO_SHARED_RANKING_MARGIN", 0.10),
                "no_shared_ranking_top_k": _env_int("NO_SHARED_RANKING_TOP_K", 8),
                "no_shared_ranking_min_chars": _env_int("NO_SHARED_RANKING_MIN_CHARS", 2),
                "no_shared_ranking_max_pos_samples_per_batch": _env_int(
                    "NO_SHARED_RANKING_MAX_POS_SAMPLES_PER_BATCH", 8
                ),
                "no_shared_ranking_max_neg_samples_per_batch": _env_int(
                    "NO_SHARED_RANKING_MAX_NEG_SAMPLES_PER_BATCH", 8
                ),
            }
        )
        return config

    base.model_config = model_config

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        if not isinstance(batch, dict):
            return base._ORIGINAL_COMPUTE_BATCH_LOSS(image_embedder, text_encoder, criterion, batch)

        if torch.is_grad_enabled():
            base._BATCH_COUNTER += 1
        local_every = max(
            1,
            base._env_int(
                "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
                getattr(base.P, "local_hard_negative_every_n_batches", 1),
            ),
        )
        local_enabled = torch.is_grad_enabled() and (base._BATCH_COUNTER % local_every == 0)

        images1 = batch["images1"].to(base.P.device, non_blocking=True)
        images2 = batch["images2"].to(base.P.device, non_blocking=True)
        texts1, texts2 = batch["texts1"], batch["texts2"]
        neg_texts1, neg_texts2 = batch["neg_texts1"], batch["neg_texts2"]
        labels = batch.get("pair_labels") or [
            legacy.EXTRA_LABEL if not bool(value) else "high_match"
            for value in batch.get("pair_positive_mask", [])
        ]

        emb1 = base.compute_embeddings(image_embedder, images1) if base.P.text_encoder_type == "arabic_span" else None
        emb2 = base.compute_embeddings(image_embedder, images2) if base.P.text_encoder_type == "arabic_span" else None

        loss1, stats1, emb1 = base.compute_single_image_text_loss(
            image_embedder, text_encoder, criterion, images1, texts1, neg_texts1, emb1,
            local_enabled=local_enabled,
        )
        if bool(getattr(base.P, "image_text_loss_on_both_lines", True)):
            loss2, stats2, emb2 = base.compute_single_image_text_loss(
                image_embedder, text_encoder, criterion, images2, texts2, neg_texts2, emb2,
                local_enabled=local_enabled,
            )
            loss = 0.5 * (loss1 + loss2)
            stats = base.average_stats([stats1, stats2])
        else:
            loss, stats = loss1, dict(stats1)

        pair_every = max(
            1,
            base._env_int("IMAGE_PAIR_EVERY_N_BATCHES", getattr(base.P, "image_pair_every_n_batches", 1)),
        )
        pair_enabled = torch.is_grad_enabled() and (base._BATCH_COUNTER % pair_every == 0)
        if pair_enabled and emb1 is not None and emb2 is not None:
            pair_loss, order_loss, pair_stats = shared._positive_pair_loss(
                base, text_encoder, criterion, texts1, texts2, emb1, emb2, labels
            )
            loss = loss + base.P.image_pair_loss_weight * pair_loss
            if base.P.sequence_consistency_loss_weight > 0:
                loss = loss + base.P.sequence_consistency_loss_weight * order_loss

            rank_loss, rank_stats = ranking_loss(
                base, text_encoder, criterion, texts1, texts2, emb1, emb2, labels
            )
            rank_weight = _env_float("NO_SHARED_RANKING_WEIGHT", 0.20)
            if rank_weight > 0:
                loss = loss + rank_weight * rank_loss
            stats.update(pair_stats)
            stats.update(rank_stats)

            if base.CTX.is_main and rank_stats.get("ranking_active", 0.0) > 0:
                print(
                    "ranking_batch "
                    f"loss={rank_stats['no_shared_rank_loss']:.4f} "
                    f"positive={rank_stats['positive_pair_similarity']:.4f} "
                    f"no_shared={rank_stats['no_shared_hard_similarity']:.4f} "
                    f"gap={rank_stats['ranking_gap']:.4f} "
                    f"max_no_shared={rank_stats['no_shared_max_similarity']:.4f} "
                    f"positive_samples={rank_stats['ranking_positive_samples']:.0f} "
                    f"negative_samples={rank_stats['ranking_negative_samples']:.0f}",
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
            f"objective=ranking enabled={_flag('USE_NO_SHARED_IMAGE_RANKING', True)} "
            f"weight={_env_float('NO_SHARED_RANKING_WEIGHT', 0.20):.3f} "
            f"margin={_env_float('NO_SHARED_RANKING_MARGIN', 0.10):.3f} "
            f"top_k={_env_int('NO_SHARED_RANKING_TOP_K', 8)} "
            f"min_chars={_env_int('NO_SHARED_RANKING_MIN_CHARS', 2)}",
            flush=True,
        )
