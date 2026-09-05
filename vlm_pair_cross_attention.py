"""Bidirectional cross-line attention for the VLM letter-depiction hierarchy.

The two manuscript lines are still encoded by two completely independent image
forwards.  Cross-attention happens only *after* each line has produced its own
4-layer contextual sequence.

Training hierarchy
------------------
    line1 pixels -> depiction -> self-context C1 --+--> Cross(C1 <- C2) -> F1
                                                    |
    line2 pixels -> depiction -> self-context C2 --+--> Cross(C2 <- C1) -> F2

Independent C1/C2 continue to receive the image-text Span-DTW objective.  The
existing local/image-pair losses are retained.  F1/F2 receive an additional
cosine-based region contrastive + order objective, so cross-attention improves
pair matching without becoming a shortcut for the independent text grounding.

Implementation note
-------------------
The pair-fusion module is registered on the Arabic text encoder.  This is only a
runtime ownership choice: that module is *not* part of text encoding.  The text
encoder is already outside image DDP and its trainable gradients are explicitly
all-reduced by the training runtime, so this keeps the two image forwards clean
and avoids adding occasionally-unused pair parameters to the image DDP graph.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import training_optimizations as opt


class SymmetricPairCrossAttention(nn.Module):
    """One shared residual cross-attention block used in both directions."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.10,
        ff_multiplier: int = 2,
        initial_gate: float = 0.20,
    ) -> None:
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"cross attention requires dim divisible by heads, got {dim}/{num_heads}"
            )
        hidden = max(dim, int(ff_multiplier) * dim)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.post_attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
        )
        self.ff_dropout = nn.Dropout(float(dropout))
        self.output_norm = nn.LayerNorm(dim)

        initial_gate = min(max(float(initial_gate), 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    @staticmethod
    def _safe_key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.bool().clone()
        if mask.ndim != 2:
            raise ValueError(f"key padding mask must be [B,S], got {tuple(mask.shape)}")
        all_masked = mask.all(dim=1)
        if bool(all_masked.any()):
            mask[all_masked] = False
        return mask

    def attend(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_padding_mask: torch.Tensor | None = None,
        query_active_mask: torch.Tensor | None = None,
        return_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("cross attention expects [B,T,D] query and [B,S,D] memory")
        if query.shape[0] != memory.shape[0] or query.shape[2] != memory.shape[2]:
            raise ValueError(
                "cross-attention batch/feature dimensions must match: "
                f"{tuple(query.shape)} vs {tuple(memory.shape)}"
            )

        q = self.query_norm(query)
        kv = self.memory_norm(memory)
        attended, weights = self.attention(
            q,
            kv,
            kv,
            key_padding_mask=self._safe_key_padding_mask(memory_padding_mask),
            need_weights=bool(return_weights),
            average_attn_weights=False,
        )
        mixed = self.post_attn_norm(
            query + self.gate.to(dtype=query.dtype, device=query.device) * self.attn_dropout(attended)
        )
        fused = self.output_norm(
            mixed + self.ff_dropout(self.ff(self.ff_norm(mixed)))
        )

        # Empty/background query windows should not acquire semantic content only
        # because they were allowed to read from the opposite line.
        if query_active_mask is not None:
            active = query_active_mask.to(query.device).bool().unsqueeze(-1)
            fused = torch.where(active, fused, query)
        return fused, weights

    def forward(
        self,
        context1: torch.Tensor,
        context2: torch.Tensor,
        *,
        ink1: torch.Tensor | None = None,
        ink2: torch.Tensor | None = None,
        min_ink: float = 0.01,
        return_weights: bool = True,
    ):
        active1 = None if ink1 is None else ink1.float() >= float(min_ink)
        active2 = None if ink2 is None else ink2.float() >= float(min_ink)
        fused1, attn12 = self.attend(
            context1,
            context2,
            memory_padding_mask=None if active2 is None else ~active2,
            query_active_mask=active1,
            return_weights=return_weights,
        )
        fused2, attn21 = self.attend(
            context2,
            context1,
            memory_padding_mask=None if active1 is None else ~active1,
            query_active_mask=active2,
            return_weights=return_weights,
        )
        return fused1, fused2, attn12, attn21


def apply_cross_attention_config(P) -> None:
    P.experiment_name = "vit_vlm_letter_depiction_cross_attention"
    P.cross_attention_enabled = True
    P.cross_attention_heads = 4
    P.cross_attention_dropout = 0.10
    P.cross_attention_ff_multiplier = 2
    P.cross_attention_initial_gate = 0.20
    P.cross_attention_min_ink = 0.01

    # Keep every original hierarchy loss and add cross-aware pair supervision.
    P.cross_attention_pair_weight = 0.30
    P.cross_attention_order_weight = 0.05

    # Quality-first guards.  No alternating batches or sample caps.
    P.local_hard_negative_every_n_batches = 1
    P.local_hard_negative_max_samples_per_batch = 0
    P.image_pair_every_n_batches = 1
    P.image_pair_max_samples_per_batch = 0


def attach_pair_cross_attention(text_encoder, P):
    if hasattr(text_encoder, "pair_cross_attention"):
        return text_encoder
    dim = int(P.vector_size)
    module = SymmetricPairCrossAttention(
        dim,
        num_heads=int(P.cross_attention_heads),
        dropout=float(P.cross_attention_dropout),
        ff_multiplier=int(P.cross_attention_ff_multiplier),
        initial_gate=float(P.cross_attention_initial_gate),
    ).to(P.device)
    text_encoder.add_module("pair_cross_attention", module)
    return text_encoder


def cross_fuse_pair(
    text_encoder,
    P,
    context1: torch.Tensor,
    context2: torch.Tensor,
    ink1: torch.Tensor | None,
    ink2: torch.Tensor | None,
    *,
    return_weights: bool = True,
):
    module = getattr(text_encoder, "pair_cross_attention", None)
    if module is None:
        raise RuntimeError("pair_cross_attention is not attached to the text encoder")
    return module(
        context1,
        context2,
        ink1=ink1,
        ink2=ink2,
        min_ink=float(P.cross_attention_min_ink),
        return_weights=return_weights,
    )


def _attention_entropy(weights: torch.Tensor | None) -> float:
    if weights is None or weights.numel() == 0:
        return 0.0
    probs = weights.float().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.detach().item())


def _mean_row_best_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = F.normalize(left.float(), p=2, dim=-1)
    right = F.normalize(right.float(), p=2, dim=-1)
    matrix = torch.matmul(left, right.transpose(-1, -2))
    return float(matrix.max(dim=-1).values.mean().detach().item())


def _cross_regions(train_module, item, fused, ink):
    # Reuse transcript-derived independent hard boundaries, but pool the
    # cross-aware representation as both the matching and contextual vector.
    return opt.regions_from_alignment(
        train_module,
        item,
        fused,
        fused,
        ink,
    )


def install_pair_cross_attention(train_module) -> None:
    """Attach pair fusion and install a full-quality paired batch loss."""
    P = train_module.P
    apply_cross_attention_config(P)
    P.export_environment()

    original_build_text_encoder = train_module.build_text_encoder

    def build_text_encoder():
        encoder = original_build_text_encoder()
        return attach_pair_cross_attention(encoder, P)

    train_module.build_text_encoder = build_text_encoder
    original_batch_loss = train_module.compute_batch_loss

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        if not isinstance(batch, dict) or P.text_encoder_type != "arabic_span":
            return original_batch_loss(image_embedder, text_encoder, criterion, batch)

        train_module._BATCH_COUNTER += int(torch.is_grad_enabled())
        counter = max(1, train_module._BATCH_COUNTER)
        images1 = batch["images1"].to(P.device, non_blocking=True)
        images2 = batch["images2"].to(P.device, non_blocking=True)
        if torch.cuda.is_available() and str(
            __import__("os").environ.get("USE_CHANNELS_LAST", "1")
        ).lower() in {"1", "true", "yes", "on"}:
            images1 = images1.contiguous(memory_format=torch.channels_last)
            images2 = images2.contiguous(memory_format=torch.channels_last)
        texts1, texts2 = batch["texts1"], batch["texts2"]
        neg1, neg2 = batch["neg_texts1"], batch["neg_texts2"]
        batch_size = int(images1.shape[0])

        # Critical invariant: two independent image forwards.  No [2B,...]
        # concatenation and no cross-line information is available yet.
        with opt.PROFILER.section("visual_forward_line1"):
            emb1 = train_module.compute_embeddings(image_embedder, images1)
        with opt.PROFILER.section("visual_forward_line2"):
            emb2 = train_module.compute_embeddings(image_embedder, images2)

        # Independent semantic grounding is deliberately computed BEFORE cross
        # attention so each line must be meaningful on its own.
        with opt.PROFILER.section("image_text_line1"):
            loss1, stats1, emb1 = train_module.compute_single_image_text_loss(
                image_embedder,
                text_encoder,
                criterion,
                images1,
                texts1,
                neg1,
                emb1,
                local_enabled=False,
            )
        if bool(getattr(P, "image_text_loss_on_both_lines", True)):
            with opt.PROFILER.section("image_text_line2"):
                loss2, stats2, emb2 = train_module.compute_single_image_text_loss(
                    image_embedder,
                    text_encoder,
                    criterion,
                    images2,
                    texts2,
                    neg2,
                    emb2,
                    local_enabled=False,
                )
            loss = 0.5 * (loss1 + loss2)
            stats = train_module.average_stats([stats1, stats2])
        else:
            loss = loss1
            stats = dict(stats1)

        local_enabled = (
            torch.is_grad_enabled()
            and bool(P.use_local_hard_negatives)
            and float(P.local_hard_negative_weight) > 0
        )
        pair_enabled = (
            torch.is_grad_enabled()
            and bool(P.use_image_pair_contrastive)
            and float(P.image_pair_loss_weight) > 0
        )
        local_indices = list(range(batch_size)) if local_enabled else []
        pair_indices = list(range(batch_size)) if pair_enabled else []
        union_indices = sorted(set(local_indices) | set(pair_indices))

        ctx1, loc1, ink1, _raw1 = emb1
        ctx2, loc2, ink2, _raw2 = emb2

        # Text-to-image hard paths are derived from independent line context.
        # Cross attention therefore cannot choose its own supervision boundaries.
        cache1 = opt.build_alignment_cache(
            train_module, text_encoder, criterion, texts1, ctx1, union_indices
        )
        cache2 = opt.build_alignment_cache(
            train_module, text_encoder, criterion, texts2, ctx2, union_indices
        )

        if local_enabled:
            with opt.PROFILER.section("local_hard_negative"):
                local1, local_stats1 = opt.local_loss_from_cache(
                    train_module, cache1, loc1, ink1, local_indices
                )
                local2, local_stats2 = opt.local_loss_from_cache(
                    train_module, cache2, loc2, ink2, local_indices
                )
                local_loss = 0.5 * (local1 + local2)
                loss = loss + float(P.local_hard_negative_weight) * local_loss
                stats.update(
                    train_module.average_stats([local_stats1, local_stats2])
                )
        else:
            stats.update(train_module._zero_local_stats(loc1)[1])

        # Pair interaction starts only here, after both independent lines exist.
        with opt.PROFILER.section("bidirectional_cross_attention"):
            fused1, fused2, attn12, attn21 = cross_fuse_pair(
                text_encoder,
                P,
                ctx1,
                ctx2,
                ink1,
                ink2,
                return_weights=True,
            )
            fused1 = F.normalize(fused1.float(), p=2, dim=-1)
            fused2 = F.normalize(fused2.float(), p=2, dim=-1)

        stats.update(
            {
                "cross_attention_gate": float(
                    text_encoder.pair_cross_attention.gate.detach().item()
                ),
                "cross_attention_entropy_12": _attention_entropy(attn12),
                "cross_attention_entropy_21": _attention_entropy(attn21),
                "pair_cosine_best_before": 0.5
                * (
                    _mean_row_best_cosine(ctx1, ctx2)
                    + _mean_row_best_cosine(ctx2, ctx1)
                ),
                "pair_cosine_best_after": 0.5
                * (
                    _mean_row_best_cosine(fused1, fused2)
                    + _mean_row_best_cosine(fused2, fused1)
                ),
            }
        )

        if pair_enabled:
            with opt.PROFILER.section("image_pair_and_order"):
                base_pair_losses = []
                base_order_losses = []
                cross_pair_losses = []
                cross_order_losses = []
                base_terms = 0
                cross_terms = 0

                for index in pair_indices:
                    if index not in cache1 or index not in cache2:
                        continue

                    # Original hierarchy pair objective: retained unchanged.
                    regions1 = opt.regions_from_alignment(
                        train_module,
                        cache1[index],
                        ctx1[index],
                        loc1[index],
                        ink1[index],
                    )
                    regions2 = opt.regions_from_alignment(
                        train_module,
                        cache2[index],
                        ctx2[index],
                        loc2[index],
                        ink2[index],
                    )
                    base_pair, base_matched = (
                        train_module.image_image_span_contrastive_loss(
                            regions1,
                            regions2,
                            margin=P.image_pair_margin,
                            top_k=P.image_pair_top_k,
                        )
                    )
                    if base_pair is not None:
                        base_pair_losses.append(base_pair)
                        base_terms += len(base_matched)
                        if float(P.sequence_consistency_loss_weight) > 0:
                            base_order_losses.append(
                                train_module.image_image_order_consistency_loss(
                                    regions1, regions2, base_matched
                                )
                            )

                    # New objective: cosine matching on cross-aware contextual
                    # regions using the SAME independent transcript boundaries.
                    cross_regions1 = _cross_regions(
                        train_module, cache1[index], fused1[index], ink1[index]
                    )
                    cross_regions2 = _cross_regions(
                        train_module, cache2[index], fused2[index], ink2[index]
                    )
                    cross_pair, cross_matched = (
                        train_module.image_image_span_contrastive_loss(
                            cross_regions1,
                            cross_regions2,
                            margin=P.image_pair_margin,
                            top_k=P.image_pair_top_k,
                        )
                    )
                    if cross_pair is not None:
                        cross_pair_losses.append(cross_pair)
                        cross_terms += len(cross_matched)
                        cross_order_losses.append(
                            train_module.image_image_order_consistency_loss(
                                cross_regions1, cross_regions2, cross_matched
                            )
                        )

                if base_pair_losses:
                    base_pair_loss = torch.stack(base_pair_losses).mean()
                    base_order_loss = (
                        torch.stack(base_order_losses).mean()
                        if base_order_losses
                        else base_pair_loss.new_tensor(0.0)
                    )
                    loss = loss + float(P.image_pair_loss_weight) * base_pair_loss
                    if float(P.sequence_consistency_loss_weight) > 0:
                        loss = loss + float(P.sequence_consistency_loss_weight) * base_order_loss
                else:
                    base_pair_loss = loss.new_tensor(0.0)
                    base_order_loss = loss.new_tensor(0.0)

                if cross_pair_losses:
                    cross_pair_loss = torch.stack(cross_pair_losses).mean()
                    cross_order_loss = (
                        torch.stack(cross_order_losses).mean()
                        if cross_order_losses
                        else cross_pair_loss.new_tensor(0.0)
                    )
                    loss = loss + float(P.cross_attention_pair_weight) * cross_pair_loss
                    if float(P.cross_attention_order_weight) > 0:
                        loss = loss + float(P.cross_attention_order_weight) * cross_order_loss
                else:
                    cross_pair_loss = loss.new_tensor(0.0)
                    cross_order_loss = loss.new_tensor(0.0)

                stats.update(
                    {
                        "image_pair_loss": float(base_pair_loss.detach().item()),
                        "order_loss": float(base_order_loss.detach().item()),
                        "pair_terms": float(base_terms) / max(1, len(base_pair_losses)),
                        "cross_pair_loss": float(cross_pair_loss.detach().item()),
                        "cross_order_loss": float(cross_order_loss.detach().item()),
                        "cross_pair_terms": float(cross_terms) / max(1, len(cross_pair_losses)),
                    }
                )
        else:
            stats.update(
                {
                    "image_pair_loss": 0.0,
                    "order_loss": 0.0,
                    "pair_terms": 0.0,
                    "cross_pair_loss": 0.0,
                    "cross_order_loss": 0.0,
                    "cross_pair_terms": 0.0,
                }
            )

        if hasattr(text_encoder, "cache_stats"):
            stats.update(text_encoder.cache_stats())
        try:
            from jax_span_dtw import bridge_stats

            stats.update(
                {f"jax_{key}": float(value) for key, value in bridge_stats().items()}
            )
        except Exception:
            pass
        stats["total"] = float(loss.detach().item())
        return loss, stats

    train_module.compute_batch_loss = compute_batch_loss


def model_config(P) -> dict[str, Any]:
    return {
        "cross_attention_enabled": bool(P.cross_attention_enabled),
        "cross_attention_stage": "after_independent_4layer_context_before_pair_cosine",
        "cross_attention_direction": "bidirectional_shared_weights",
        "cross_attention_heads": int(P.cross_attention_heads),
        "cross_attention_dropout": float(P.cross_attention_dropout),
        "cross_attention_ff_multiplier": int(P.cross_attention_ff_multiplier),
        "cross_attention_initial_gate": float(P.cross_attention_initial_gate),
        "cross_attention_min_ink": float(P.cross_attention_min_ink),
        "cross_attention_pair_weight": float(P.cross_attention_pair_weight),
        "cross_attention_order_weight": float(P.cross_attention_order_weight),
        "cross_attention_final_match": "cosine_on_fused_context",
        "cross_attention_image_text_policy": "independent_context_only",
        "paired_visual_forward": "separate",
    }
