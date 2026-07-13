#!/usr/bin/env python3
"""Fast training entry point for improve_neg image-pair experiments.

This wrapper keeps the original train.py code intact, but monkey-patches the
batch-loss function to avoid the biggest new bottlenecks introduced by paired
image-image training:

1. Do not run full Span-DTW image-text loss on line2 by default. Line2 still
   participates in the image-image pair loss, so it receives local image-space
   gradients for the retrieval task.
2. Limit image-pair pseudo-alignment mining to a subset of samples per batch.
3. Optionally run local hard-negative mining every N batches.

The defaults are controlled through Parameters.py / environment variables and
can be overridden from the sbatch script.
"""
from __future__ import annotations

import os

import torch

import Parameters as P
import train as base

_BATCH_COUNTER = 0
_original_compute_batch_loss = base.compute_batch_loss
_original_model_config = base.model_config


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _slice_embeddings(embeddings, max_samples: int):
    """Slice the batch dimension of an embeddings tuple returned by train.compute_embeddings."""
    if embeddings is None or max_samples <= 0:
        return embeddings
    return tuple(x[:max_samples] if torch.is_tensor(x) and x.shape[0] >= max_samples else x for x in embeddings)


def _zero_pair_stats(tensor):
    zero = tensor.new_tensor(0.0)
    return zero, zero, {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}


def compute_image_pair_loss_fast(text_encoder, criterion, texts1, texts2, emb1, emb2):
    """Same objective as train.compute_image_pair_loss, but with sample limit and optional order skip."""
    if (
        not P.use_image_pair_contrastive
        or P.image_pair_loss_weight <= 0
        or P.text_encoder_type != "arabic_span"
        or not torch.is_grad_enabled()
    ):
        return _zero_pair_stats(emb1[0])

    max_samples = _get_int_env(
        "IMAGE_PAIR_MAX_SAMPLES_PER_BATCH",
        getattr(P, "image_pair_max_samples_per_batch", 8),
    )
    if max_samples > 0:
        max_samples = min(max_samples, len(texts1), int(emb1[0].shape[0]), int(emb2[0].shape[0]))
        texts1 = list(texts1[:max_samples])
        texts2 = list(texts2[:max_samples])
        emb1 = _slice_embeddings(emb1, max_samples)
        emb2 = _slice_embeddings(emb2, max_samples)

    norm_ctx1, norm_loc1, ink1, _raw1 = emb1
    norm_ctx2, norm_loc2, ink2, _raw2 = emb2
    pair_losses = []
    order_losses = []
    terms = 0
    compute_order = P.sequence_consistency_loss_weight > 0 and torch.is_grad_enabled()

    for b in range(norm_ctx1.shape[0]):
        try:
            regions1 = base.extract_aligned_span_regions(
                text_encoder,
                criterion,
                texts1[b],
                norm_ctx1[b],
                norm_loc1[b],
                ink1[b],
            )
            regions2 = base.extract_aligned_span_regions(
                text_encoder,
                criterion,
                texts2[b],
                norm_ctx2[b],
                norm_loc2[b],
                ink2[b],
            )
        except ValueError:
            continue

        pair_loss, matched_pairs = base.image_image_span_contrastive_loss(
            regions1,
            regions2,
            margin=P.image_pair_margin,
            top_k=P.image_pair_top_k,
        )
        if pair_loss is None:
            continue

        pair_losses.append(pair_loss)
        terms += len(matched_pairs)
        if compute_order:
            order_losses.append(base.image_image_order_consistency_loss(regions1, regions2, matched_pairs))

    if not pair_losses:
        return _zero_pair_stats(norm_ctx1)

    pair_loss = torch.stack(pair_losses).mean()
    order_loss = torch.stack(order_losses).mean() if order_losses else pair_loss.new_tensor(0.0)
    return pair_loss, order_loss, {
        "image_pair_loss": float(pair_loss.detach().item()),
        "order_loss": float(order_loss.detach().item()),
        "pair_terms": float(terms) / max(1, len(pair_losses)),
    }


def _with_local_hard_negative_frequency(fn, enabled_this_batch: bool):
    old_value = P.use_local_hard_negatives
    P.use_local_hard_negatives = bool(old_value and enabled_this_batch)
    try:
        return fn()
    finally:
        P.use_local_hard_negatives = old_value


def compute_batch_loss_fast(image_embedder, text_encoder, criterion, batch):
    """Optimized replacement for train.compute_batch_loss."""
    global _BATCH_COUNTER
    _BATCH_COUNTER += 1

    # Keep old non-paired behavior unchanged, except for optional local-loss frequency.
    if not isinstance(batch, dict):
        local_every = max(1, _get_int_env(
            "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
            getattr(P, "local_hard_negative_every_n_batches", 1),
        ))
        local_enabled = (_BATCH_COUNTER % local_every) == 0
        return _with_local_hard_negative_frequency(
            lambda: _original_compute_batch_loss(image_embedder, text_encoder, criterion, batch),
            local_enabled,
        )

    images1 = batch["images1"].to(P.device, non_blocking=True)
    images2 = batch["images2"].to(P.device, non_blocking=True)
    texts1 = batch["texts1"]
    texts2 = batch["texts2"]
    neg_texts1 = batch["neg_texts1"]
    neg_texts2 = batch["neg_texts2"]

    local_every = max(1, _get_int_env(
        "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
        getattr(P, "local_hard_negative_every_n_batches", 1),
    ))
    local_enabled = (_BATCH_COUNTER % local_every) == 0

    if P.text_encoder_type == "arabic_span":
        emb1 = base.compute_embeddings(image_embedder, images1)
        emb2 = base.compute_embeddings(image_embedder, images2)
    else:
        emb1 = None
        emb2 = None

    loss1, stats1, emb1 = _with_local_hard_negative_frequency(
        lambda: base.compute_single_image_text_loss(
            image_embedder,
            text_encoder,
            criterion,
            images1,
            texts1,
            neg_texts1,
            emb1,
        ),
        local_enabled,
    )

    # Full Span-DTW on line2 doubles the number of expensive JAX DTW calls.
    # Keep it optional; by default line2 is trained through image-pair loss.
    train_line2_text = bool(getattr(P, "image_text_loss_on_both_lines", False))
    if train_line2_text:
        loss2, stats2, emb2 = _with_local_hard_negative_frequency(
            lambda: base.compute_single_image_text_loss(
                image_embedder,
                text_encoder,
                criterion,
                images2,
                texts2,
                neg_texts2,
                emb2,
            ),
            local_enabled,
        )
        loss = 0.5 * (loss1 + loss2)
        stats = base.average_stats([stats1, stats2])
    else:
        loss = loss1
        stats = dict(stats1)

    pair_every = max(1, _get_int_env(
        "IMAGE_PAIR_EVERY_N_BATCHES",
        getattr(P, "image_pair_every_n_batches", 1),
    ))
    pair_enabled = (_BATCH_COUNTER % pair_every) == 0

    if pair_enabled and emb1 is not None and emb2 is not None:
        pair_loss, order_loss, pair_stats = compute_image_pair_loss_fast(
            text_encoder,
            criterion,
            texts1,
            texts2,
            emb1,
            emb2,
        )
        loss = loss + P.image_pair_loss_weight * pair_loss
        if P.sequence_consistency_loss_weight > 0 and torch.is_grad_enabled():
            loss = loss + P.sequence_consistency_loss_weight * order_loss
        stats.update(pair_stats)
    else:
        stats.update({"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0})

    stats["total"] = float(loss.detach().item())
    return loss, stats


def model_config_fast(stride):
    cfg = _original_model_config(stride)
    cfg.update(
        {
            "image_text_loss_on_both_lines": getattr(P, "image_text_loss_on_both_lines", False),
            "image_pair_every_n_batches": getattr(P, "image_pair_every_n_batches", 1),
            "image_pair_max_samples_per_batch": getattr(P, "image_pair_max_samples_per_batch", 8),
            "local_hard_negative_every_n_batches": getattr(P, "local_hard_negative_every_n_batches", 1),
        }
    )
    return cfg


base.compute_batch_loss = compute_batch_loss_fast
base.compute_image_pair_loss = compute_image_pair_loss_fast
base.model_config = model_config_fast

if __name__ == "__main__":
    base.main()
