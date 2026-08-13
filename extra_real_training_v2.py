"""Expanded-real training with explicit no-shared image-image negatives.

This module layers on top of ``extra_real_training``. Positive high/medium-match
pairs keep the existing span-region contrastive and order objectives. Rows
labelled ``no_shared_content`` keep image-text supervision and additionally
contribute a hard image-image negative loss over *different-text* multi-character
span groups. Equal-text groups are ignored so common Arabic letters/subwords are
not incorrectly pushed apart.
"""
from __future__ import annotations

import os

import torch

import extra_real_training as legacy


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
        "no_shared_image_neg_loss": 0.0,
        "no_shared_hard_cosine": 0.0,
        "no_shared_max_cosine": 0.0,
        "no_shared_negative_terms": 0.0,
        "no_shared_negative_samples": 0.0,
    }


def _select_embeddings(values, indices: list[int]):
    if not indices:
        return values
    index_tensor = torch.as_tensor(
        indices, dtype=torch.long, device=values[0].device
    )
    return tuple(
        value.index_select(0, index_tensor)
        if torch.is_tensor(value) and value.ndim > 0
        else value
        for value in values
    )


def _positive_pair_loss(
    base,
    text_encoder,
    criterion,
    texts1,
    texts2,
    emb1,
    emb2,
    labels,
):
    keep = [
        index
        for index, label in enumerate(labels)
        if str(label) in legacy.POSITIVE_LABELS
    ]
    if not keep:
        zero = emb1[0].new_tensor(0.0)
        return zero, zero, {
            "image_pair_loss": 0.0,
            "order_loss": 0.0,
            "pair_terms": 0.0,
            "pair_positive_samples": 0.0,
        }
    selected1 = _select_embeddings(emb1, keep)
    selected2 = _select_embeddings(emb2, keep)
    pair_loss, order_loss, stats = base.compute_image_pair_loss(
        text_encoder,
        criterion,
        [texts1[index] for index in keep],
        [texts2[index] for index in keep],
        selected1,
        selected2,
    )
    stats["pair_positive_samples"] = float(len(keep))
    return pair_loss, order_loss, stats


def _eligible_groups(base, regions, min_chars: int):
    groups = base._build_composition_groups(regions)
    return [
        group
        for group in groups
        if len(str(group.get("text", "")).strip()) >= int(min_chars)
    ]


def _one_no_shared_loss(
    base,
    text_encoder,
    criterion,
    text1,
    text2,
    emb1,
    emb2,
    batch_index: int,
    *,
    max_cosine: float,
    top_k: int,
    min_chars: int,
):
    norm_ctx1, norm_loc1, ink1, _raw1 = emb1
    norm_ctx2, norm_loc2, ink2, _raw2 = emb2
    regions1 = base.extract_aligned_span_regions(
        text_encoder,
        criterion,
        text1,
        norm_ctx1[batch_index],
        norm_loc1[batch_index],
        ink1[batch_index],
    )
    regions2 = base.extract_aligned_span_regions(
        text_encoder,
        criterion,
        text2,
        norm_ctx2[batch_index],
        norm_loc2[batch_index],
        ink2[batch_index],
    )
    groups1 = _eligible_groups(base, regions1, min_chars)
    groups2 = _eligible_groups(base, regions2, min_chars)
    if not groups1 or not groups2:
        return None

    vectors1 = torch.stack([group["vec"] for group in groups1], dim=0)
    vectors2 = torch.stack([group["vec"] for group in groups2], dim=0)
    similarities = torch.matmul(vectors1, vectors2.T)

    mismatch = torch.tensor(
        [
            [group1["text"] != group2["text"] for group2 in groups2]
            for group1 in groups1
        ],
        dtype=torch.bool,
        device=similarities.device,
    )
    hard_pool = similarities[mismatch]
    if hard_pool.numel() == 0:
        return None
    k = min(max(1, int(top_k)), int(hard_pool.numel()))
    hard = torch.topk(hard_pool, k=k).values
    loss = torch.relu(hard - float(max_cosine)).mean()
    return loss, hard.mean(), hard.max(), k


def no_shared_image_negative_loss(
    base,
    text_encoder,
    criterion,
    texts1,
    texts2,
    emb1,
    emb2,
    labels,
):
    if (
        not _flag("USE_NO_SHARED_IMAGE_NEGATIVES", True)
        or _env_float("NO_SHARED_IMAGE_NEGATIVE_WEIGHT", 0.25) <= 0
        or base.P.text_encoder_type != "arabic_span"
        or not torch.is_grad_enabled()
    ):
        return _zero_stats(emb1[0])

    indices = [
        index
        for index, label in enumerate(labels)
        if str(label) == legacy.EXTRA_LABEL
    ]
    max_samples = _env_int("NO_SHARED_IMAGE_NEGATIVE_MAX_SAMPLES_PER_BATCH", 8)
    if max_samples > 0:
        indices = indices[:max_samples]
    if not indices:
        return _zero_stats(emb1[0])

    max_cosine = _env_float("NO_SHARED_IMAGE_NEGATIVE_MAX_COSINE", 0.45)
    top_k = _env_int("NO_SHARED_IMAGE_NEGATIVE_TOP_K", 8)
    min_chars = _env_int("NO_SHARED_IMAGE_NEGATIVE_MIN_CHARS", 2)

    losses = []
    hard_means = []
    hard_maxes = []
    terms = 0
    for index in indices:
        try:
            result = _one_no_shared_loss(
                base,
                text_encoder,
                criterion,
                texts1[index],
                texts2[index],
                emb1,
                emb2,
                index,
                max_cosine=max_cosine,
                top_k=top_k,
                min_chars=min_chars,
            )
        except ValueError:
            continue
        if result is None:
            continue
        sample_loss, hard_mean, hard_max, sample_terms = result
        losses.append(sample_loss)
        hard_means.append(hard_mean)
        hard_maxes.append(hard_max)
        terms += int(sample_terms)

    if not losses:
        return _zero_stats(emb1[0])

    loss = torch.stack(losses).mean()
    return loss, {
        "no_shared_image_neg_loss": float(loss.detach().item()),
        "no_shared_hard_cosine": float(
            torch.stack(hard_means).mean().detach().item()
        ),
        "no_shared_max_cosine": float(
            torch.stack(hard_maxes).max().detach().item()
        ),
        "no_shared_negative_terms": float(terms) / len(losses),
        "no_shared_negative_samples": float(len(losses)),
    }


def install(base) -> None:
    """Install legacy expanded-real loading plus explicit negative-pair training."""
    legacy.install(base)

    original_model_config = base.model_config

    def model_config(stride, args):
        config = dict(original_model_config(stride, args))
        config.update(
            {
                "use_no_shared_image_negatives": _flag(
                    "USE_NO_SHARED_IMAGE_NEGATIVES", True
                ),
                "no_shared_image_negative_weight": _env_float(
                    "NO_SHARED_IMAGE_NEGATIVE_WEIGHT", 0.25
                ),
                "no_shared_image_negative_max_cosine": _env_float(
                    "NO_SHARED_IMAGE_NEGATIVE_MAX_COSINE", 0.45
                ),
                "no_shared_image_negative_top_k": _env_int(
                    "NO_SHARED_IMAGE_NEGATIVE_TOP_K", 8
                ),
                "no_shared_image_negative_min_chars": _env_int(
                    "NO_SHARED_IMAGE_NEGATIVE_MIN_CHARS", 2
                ),
                "no_shared_image_negative_max_samples_per_batch": _env_int(
                    "NO_SHARED_IMAGE_NEGATIVE_MAX_SAMPLES_PER_BATCH", 8
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
        texts1 = batch["texts1"]
        texts2 = batch["texts2"]
        neg_texts1 = batch["neg_texts1"]
        neg_texts2 = batch["neg_texts2"]
        labels = batch.get("pair_labels") or [
            legacy.EXTRA_LABEL if not bool(value) else "high_match"
            for value in batch.get("pair_positive_mask", [])
        ]

        if base.P.text_encoder_type == "arabic_span":
            emb1 = base.compute_embeddings(image_embedder, images1)
            emb2 = base.compute_embeddings(image_embedder, images2)
        else:
            emb1 = None
            emb2 = None

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
            loss = loss1
            stats = dict(stats1)

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
            pair_loss, order_loss, pair_stats = _positive_pair_loss(
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

            negative_loss, negative_stats = no_shared_image_negative_loss(
                base,
                text_encoder,
                criterion,
                texts1,
                texts2,
                emb1,
                emb2,
                labels,
            )
            negative_weight = _env_float(
                "NO_SHARED_IMAGE_NEGATIVE_WEIGHT", 0.25
            )
            if negative_weight > 0:
                loss = loss + negative_weight * negative_loss
            stats.update(pair_stats)
            stats.update(negative_stats)
        else:
            stats.update(
                {
                    "image_pair_loss": 0.0,
                    "order_loss": 0.0,
                    "pair_terms": 0.0,
                    "pair_positive_samples": 0.0,
                    "no_shared_image_neg_loss": 0.0,
                    "no_shared_hard_cosine": 0.0,
                    "no_shared_max_cosine": 0.0,
                    "no_shared_negative_terms": 0.0,
                    "no_shared_negative_samples": 0.0,
                }
            )
        stats["total"] = float(loss.detach().item())
        return loss, stats

    base.compute_batch_loss = compute_batch_loss

    if getattr(base.CTX, "is_main", True):
        print(
            "Expanded-real negative-pair objective: "
            f"enabled={_flag('USE_NO_SHARED_IMAGE_NEGATIVES', True)} "
            f"weight={_env_float('NO_SHARED_IMAGE_NEGATIVE_WEIGHT', 0.25):.3f} "
            f"max_cosine={_env_float('NO_SHARED_IMAGE_NEGATIVE_MAX_COSINE', 0.45):.3f} "
            f"top_k={_env_int('NO_SHARED_IMAGE_NEGATIVE_TOP_K', 8)} "
            f"min_chars={_env_int('NO_SHARED_IMAGE_NEGATIVE_MIN_CHARS', 2)}",
            flush=True,
        )
