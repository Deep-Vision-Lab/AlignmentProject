#!/usr/bin/env python3
"""Fast training entry point for improve_neg image-pair experiments.

This wrapper keeps the original train.py code intact, but monkey-patches the
batch-loss function to avoid the biggest bottlenecks introduced by paired
image-image training:

1. Do not run full Span-DTW image-text loss on line2 by default. Line2 still
   participates in the image-image pair loss, so it receives local image-space
   gradients for the retrieval task.
2. Limit image-pair pseudo-alignment mining to a subset of samples per batch.
3. Run local hard-negative mining every N batches and only on a subset of the
   batch when it runs.
4. Keep NUM_NEGATIVES as the full generated pool, but optionally score only a
   rotating subset of negatives with Span-DTW each batch. This preserves varied
   negatives across training while cutting the dominant JAX DTW calls.

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
    return tuple(
        x[:max_samples] if torch.is_tensor(x) and x.shape[0] >= max_samples else x
        for x in embeddings
    )


def _select_cyclic_indices(batch_size: int, max_samples: int, device):
    """Choose a small deterministic subset that rotates across batches."""
    if max_samples <= 0 or max_samples >= batch_size:
        return None
    start = ((_BATCH_COUNTER - 1) * max_samples) % batch_size
    idx = [(start + offset) % batch_size for offset in range(max_samples)]
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def _select_local_subset(pos_texts, norm_context_img, norm_local_img, ink_ratios, max_samples: int):
    batch_size = int(norm_context_img.shape[0])
    indices = _select_cyclic_indices(batch_size, max_samples, norm_context_img.device)
    if indices is None:
        return pos_texts, norm_context_img, norm_local_img, ink_ratios

    cpu_indices = [int(i) for i in indices.detach().cpu().tolist()]
    selected_texts = [pos_texts[i] for i in cpu_indices]
    selected_context = norm_context_img.index_select(0, indices)
    selected_local = norm_local_img.index_select(0, indices)
    selected_ink = ink_ratios.index_select(0, indices) if torch.is_tensor(ink_ratios) else ink_ratios
    return selected_texts, selected_context, selected_local, selected_ink


def _select_active_negatives(neg_texts, active_per_sample: int):
    """Keep a rotating subset of generated negatives for expensive Span-DTW scoring.

    This does NOT reduce NUM_NEGATIVES in the dataloader. It keeps the generated
    negative pool and rotates which negatives are passed to Span-DTW each batch.
    With NUM_NEGATIVES=4 and active_per_sample=2, each batch uses two negatives,
    then the next batches use different offsets.
    """
    if active_per_sample <= 0:
        return neg_texts

    selected = []
    for sample_idx, sample_negs in enumerate(neg_texts):
        sample_negs = list(sample_negs)
        n = len(sample_negs)
        if n == 0 or active_per_sample >= n:
            selected.append(sample_negs)
            continue
        start = (_BATCH_COUNTER + sample_idx) % n
        selected.append([sample_negs[(start + offset) % n] for offset in range(active_per_sample)])
    return selected


def _zero_pair_stats(tensor):
    zero = tensor.new_tensor(0.0)
    return zero, zero, {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}


def _zero_local_stats(tensor):
    zero = tensor.new_tensor(0.0)
    return zero, {
        "local_hard_neg": 0.0,
        "local_pos_sim": 0.0,
        "local_neg_sim": 0.0,
        "local_terms": 0.0,
    }


def compute_local_hard_negative_loss_fast(
    text_encoder,
    criterion,
    norm_context_img,
    norm_local_img,
    pos_texts,
    ink_ratios=None,
):
    """Run local hard-negative mining on only a subset of the batch."""
    if (
        not P.use_local_hard_negatives
        or P.local_hard_negative_weight <= 0
        or not torch.is_grad_enabled()
    ):
        return _zero_local_stats(norm_local_img)

    max_samples = _get_int_env(
        "LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH",
        getattr(P, "local_hard_negative_max_samples_per_batch", 8),
    )
    selected_texts, selected_context, selected_local, selected_ink = _select_local_subset(
        pos_texts,
        norm_context_img,
        norm_local_img,
        ink_ratios,
        max_samples,
    )
    return base.local_hard_negative_loss_for_batch(
        text_encoder,
        criterion,
        selected_context,
        selected_local,
        selected_texts,
        ink_ratios=selected_ink,
    )


def compute_single_image_text_loss_fast(
    image_embedder,
    text_encoder,
    criterion,
    images,
    pos_texts,
    neg_texts,
    embeddings=None,
):
    """Like train.compute_single_image_text_loss, with active negatives and subset-limited local mining."""
    if P.text_encoder_type != "arabic_span":
        return base.compute_single_image_text_loss(
            image_embedder,
            text_encoder,
            criterion,
            images,
            pos_texts,
            neg_texts,
            embeddings,
        )

    if embeddings is None:
        norm_img, norm_local_img, ink_ratio, local_img_emb = base.compute_embeddings(image_embedder, images)
    else:
        norm_img, norm_local_img, ink_ratio, local_img_emb = embeddings

    active_negatives = _get_int_env(
        "SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE",
        getattr(P, "span_dtw_active_negatives_per_sample", 0),
    )
    neg_texts_for_dtw = _select_active_negatives(neg_texts, active_negatives)

    loss, stats = criterion.forward_varlen(text_encoder, norm_img, pos_texts, neg_texts_for_dtw)
    stats["active_negatives"] = float(
        len(neg_texts_for_dtw[0]) if neg_texts_for_dtw and len(neg_texts_for_dtw[0]) > 0 else 0
    )

    local_loss, local_stats = compute_local_hard_negative_loss_fast(
        text_encoder,
        criterion,
        norm_img,
        norm_local_img,
        pos_texts,
        ink_ratios=ink_ratio,
    )
    if P.use_local_hard_negatives and P.local_hard_negative_weight > 0 and torch.is_grad_enabled():
        loss = loss + P.local_hard_negative_weight * local_loss
    stats.update(local_stats)

    var_loss = base.image_embedding_variance_loss(local_img_emb, P.image_variance_target_std)
    if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
        loss = loss + P.image_variance_loss_weight * var_loss
    stats["img_var_loss"] = float(var_loss.detach().item())
    stats["total"] = float(loss.detach().item())
    return loss, stats, (norm_img, norm_local_img, ink_ratio, local_img_emb)


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

    local_every = max(1, _get_int_env(
        "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
        getattr(P, "local_hard_negative_every_n_batches", 1),
    ))
    local_enabled = (_BATCH_COUNTER % local_every) == 0

    # Keep old non-paired behavior, but use active negatives and subset-limited local loss.
    if not isinstance(batch, dict):
        images, pos_texts, neg_texts = batch
        images = images.to(P.device, non_blocking=True)
        return _with_local_hard_negative_frequency(
            lambda: compute_single_image_text_loss_fast(
                image_embedder,
                text_encoder,
                criterion,
                images,
                pos_texts,
                neg_texts,
            )[:2],
            local_enabled,
        )

    images1 = batch["images1"].to(P.device, non_blocking=True)
    images2 = batch["images2"].to(P.device, non_blocking=True)
    texts1 = batch["texts1"]
    texts2 = batch["texts2"]
    neg_texts1 = batch["neg_texts1"]
    neg_texts2 = batch["neg_texts2"]

    if P.text_encoder_type == "arabic_span":
        emb1 = base.compute_embeddings(image_embedder, images1)
        emb2 = base.compute_embeddings(image_embedder, images2)
    else:
        emb1 = None
        emb2 = None

    loss1, stats1, emb1 = _with_local_hard_negative_frequency(
        lambda: compute_single_image_text_loss_fast(
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
            lambda: compute_single_image_text_loss_fast(
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
            "local_hard_negative_max_samples_per_batch": getattr(P, "local_hard_negative_max_samples_per_batch", 8),
            "span_dtw_active_negatives_per_sample": getattr(P, "span_dtw_active_negatives_per_sample", 0),
        }
    )
    return cfg


base.compute_batch_loss = compute_batch_loss_fast
base.compute_image_pair_loss = compute_image_pair_loss_fast
base.model_config = model_config_fast

if __name__ == "__main__":
    base.main()
