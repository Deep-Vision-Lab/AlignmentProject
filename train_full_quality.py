#!/usr/bin/env python3
"""Full-quality training entry point for the Arabic manuscript alignment model.

This entry point keeps the memory-conscious machinery from
``train_fast_image_pair.py`` but configures it for quality rather than speed and
patches two objectives that were incomplete in the previous experiment:

1. Compositional image-pair positives
   A region labelled ``AB`` can match either another ``AB`` region or two
   consecutive regions labelled ``A`` and ``B``. The comparison is symmetric,
   so ``A+B`` can also match ``AB``.

2. Differentiable contextual order consistency
   Hard Span-DTW still supplies pseudo regions, but each region also retains a
   pooled contextual embedding. A soft cross-image position distribution is
   computed from contextual cosine similarities. Position and monotonicity
   penalties therefore backpropagate into the CNN+BiLSTM contextual features.

The launcher ``scripts/train/sbatch_span_d3tw_full_quality.sbatch`` enables full
image-text training on both lines, all negatives, all local-hard-negative
samples, and all image-pair samples by default.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn.functional as F

# Importing the fast wrapper first applies its efficient batch implementation to
# train.py. This full-quality entry point then upgrades the objectives below.
import train_fast_image_pair as fast  # noqa: F401
import train as base


_ORIGINAL_MODEL_CONFIG = base.model_config


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _normalised_mean(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(vectors), dim=0)
    return F.normalize(stacked.mean(dim=0), p=2, dim=-1)


def extract_aligned_span_regions_full(
    text_encoder,
    criterion,
    text,
    norm_context_img,
    norm_local_img,
    ink_ratio=None,
):
    """Extract hard pseudo regions while retaining trainable local/context vectors."""
    encoding = text_encoder(text, use_cache=False if text_encoder.training else None)
    with torch.no_grad():
        path = base.hard_span_dtw_path(
            encoding,
            norm_context_img,
            temperature=criterion.temperature,
            max_windows=criterion.max_windows_per_span,
            window_count_penalty=criterion.window_count_penalty,
        )

    regions = []
    ink_ratio = ink_ratio.to(norm_local_img.device).float() if ink_ratio is not None else None
    for step in path:
        span_idx = int(step["span_idx"])
        w0 = int(step["window_start"])
        w1 = int(step["window_end"])
        if w1 <= w0 or w0 < 0 or w1 > norm_local_img.shape[0]:
            continue

        span_text = str(encoding.texts[span_idx])
        if not span_text.strip():
            continue

        region_ink = ink_ratio[w0:w1] if ink_ratio is not None else None
        if (
            region_ink is not None
            and region_ink.max().item() < base.P.local_hard_negative_min_ink
        ):
            continue

        local_vec = F.normalize(
            base.ink_weighted_mean(norm_local_img[w0:w1], region_ink),
            p=2,
            dim=-1,
        )
        context_vec = F.normalize(
            base.ink_weighted_mean(norm_context_img[w0:w1], region_ink),
            p=2,
            dim=-1,
        )
        regions.append(
            {
                "span_text": span_text,
                "span_idx": span_idx,
                "window_start": w0,
                "window_end": w1,
                "center": norm_local_img.new_tensor(0.5 * (w0 + w1 - 1)),
                "vec": local_vec,
                "context_vec": context_vec,
            }
        )
    return regions


def _build_composition_groups(regions):
    """Build one-region and consecutive multi-region compositional candidates."""
    max_regions = max(1, _env_int("PAIR_COMPOSITION_MAX_REGIONS", 2))
    max_chars = max(1, _env_int("PAIR_COMPOSITION_MAX_CHARS", 3))
    groups = []

    for start in range(len(regions)):
        for count in range(1, max_regions + 1):
            end = start + count
            if end > len(regions):
                break
            members = regions[start:end]
            texts = [str(region["span_text"]) for region in members]
            if any((not text.strip()) or any(ch.isspace() for ch in text) for text in texts):
                break
            text = "".join(texts)
            if len(text) > max_chars:
                break

            groups.append(
                {
                    "text": text,
                    "region_start": start,
                    "region_end": end,
                    "window_start": int(members[0]["window_start"]),
                    "window_end": int(members[-1]["window_end"]),
                    "center": torch.stack([member["center"] for member in members]).mean(),
                    "vec": _normalised_mean(member["vec"] for member in members),
                    "context_vec": _normalised_mean(
                        member.get("context_vec", member["vec"]) for member in members
                    ),
                }
            )
    return groups


def _directional_group_contrastive(groups_a, groups_b, margin, top_k):
    if not groups_a or not groups_b:
        return [], []

    vecs_a = torch.stack([group["vec"] for group in groups_a], dim=0)
    vecs_b = torch.stack([group["vec"] for group in groups_b], dim=0)
    similarities = torch.matmul(vecs_a, vecs_b.T)
    losses = []
    matched = []

    for index_a, group_a in enumerate(groups_a):
        positive_indices = [
            index_b
            for index_b, group_b in enumerate(groups_b)
            if group_b["text"] == group_a["text"]
        ]
        if not positive_indices:
            continue

        positive_index = max(
            positive_indices,
            key=lambda index_b: float(similarities[index_a, index_b].detach()),
        )
        positive_similarity = similarities[index_a, positive_index]

        # All groups spelling the same text are valid positives, not negatives.
        negative_indices = [
            index_b
            for index_b, group_b in enumerate(groups_b)
            if group_b["text"] != group_a["text"]
        ]
        if not negative_indices:
            continue

        negative_values = similarities[index_a, negative_indices]
        k = min(max(1, int(top_k)), int(negative_values.numel()))
        hard_negatives = torch.topk(negative_values, k=k).values
        losses.append(
            torch.relu(float(margin) - positive_similarity + hard_negatives).mean()
        )
        matched.append((group_a, groups_b[positive_index]))

    return losses, matched


def image_image_span_contrastive_loss_compositional(
    regions1,
    regions2,
    margin=0.35,
    top_k=8,
):
    """Symmetric exact/compositional image-image region contrastive loss."""
    groups1 = _build_composition_groups(regions1)
    groups2 = _build_composition_groups(regions2)
    if not groups1 or not groups2:
        return None, []

    losses12, matched12 = _directional_group_contrastive(
        groups1, groups2, margin, top_k
    )
    losses21, matched21_reverse = _directional_group_contrastive(
        groups2, groups1, margin, top_k
    )
    matched21 = [(group1, group2) for group2, group1 in matched21_reverse]
    losses = losses12 + losses21
    matched = matched12 + matched21

    if not losses:
        return groups1[0]["vec"].new_tensor(0.0), matched
    return torch.stack(losses).mean(), matched


def _normalise_centers(centers: torch.Tensor) -> torch.Tensor:
    minimum = centers.min()
    maximum = centers.max()
    scale = (maximum - minimum).clamp_min(1.0)
    return (centers - minimum) / scale


def _soft_contextual_position_loss(anchor_regions, candidate_regions):
    """Differentiable position/order loss using contextual region similarities."""
    if len(anchor_regions) < 2 or len(candidate_regions) < 2:
        device = (
            anchor_regions[0]["vec"].device
            if anchor_regions
            else candidate_regions[0]["vec"].device
        )
        return torch.tensor(0.0, device=device)

    anchor_vectors = torch.stack(
        [region.get("context_vec", region["vec"]) for region in anchor_regions],
        dim=0,
    )
    candidate_vectors = torch.stack(
        [region.get("context_vec", region["vec"]) for region in candidate_regions],
        dim=0,
    )
    anchor_centers = _normalise_centers(
        torch.stack([region["center"] for region in anchor_regions]).to(anchor_vectors.device)
    )
    candidate_centers = _normalise_centers(
        torch.stack([region["center"] for region in candidate_regions]).to(anchor_vectors.device)
    )

    temperature = max(1e-4, _env_float("ORDER_TEMPERATURE", 0.07))
    similarities = torch.matmul(anchor_vectors, candidate_vectors.T) / temperature
    probabilities = torch.softmax(similarities, dim=1)
    expected_positions = torch.matmul(probabilities, candidate_centers)

    # Same-content paired lines should agree in relative normalized position.
    position_loss = F.smooth_l1_loss(expected_positions, anchor_centers)

    # Preserve monotonic ordering after sorting by the source-line positions.
    order = torch.argsort(anchor_centers)
    ordered_expected = expected_positions.index_select(0, order)
    differences = ordered_expected[1:] - ordered_expected[:-1]
    margin = _env_float("ORDER_MONOTONIC_MARGIN", 0.02)
    monotonic_loss = torch.relu(float(margin) - differences).mean()

    position_weight = _env_float("ORDER_POSITION_COMPONENT_WEIGHT", 1.0)
    monotonic_weight = _env_float("ORDER_MONOTONIC_COMPONENT_WEIGHT", 1.0)
    return position_weight * position_loss + monotonic_weight * monotonic_loss


def image_image_order_consistency_loss_differentiable(
    regions1,
    regions2,
    matched_pairs,
):
    """Symmetric contextual position/order loss with real embedding gradients."""
    del matched_pairs  # Pair count is logged separately; ordering uses all regions.
    if not regions1 or not regions2:
        device = regions1[0]["vec"].device if regions1 else regions2[0]["vec"].device
        return torch.tensor(0.0, device=device)

    forward_loss = _soft_contextual_position_loss(regions1, regions2)
    backward_loss = _soft_contextual_position_loss(regions2, regions1)
    return 0.5 * (forward_loss + backward_loss)


def model_config_full_quality(stride):
    config = _ORIGINAL_MODEL_CONFIG(stride)
    config.update(
        {
            "training_profile": "full_quality_compositional",
            "pair_composition_max_regions": _env_int(
                "PAIR_COMPOSITION_MAX_REGIONS", 2
            ),
            "pair_composition_max_chars": _env_int(
                "PAIR_COMPOSITION_MAX_CHARS", 3
            ),
            "order_temperature": _env_float("ORDER_TEMPERATURE", 0.07),
            "order_monotonic_margin": _env_float(
                "ORDER_MONOTONIC_MARGIN", 0.02
            ),
            "order_position_component_weight": _env_float(
                "ORDER_POSITION_COMPONENT_WEIGHT", 1.0
            ),
            "order_monotonic_component_weight": _env_float(
                "ORDER_MONOTONIC_COMPONENT_WEIGHT", 1.0
            ),
            "differentiable_contextual_order": True,
            "compositional_pair_matching": True,
        }
    )
    return config


# Apply patches before train.main() creates the optimizer and starts training.
base.extract_aligned_span_regions = extract_aligned_span_regions_full
base.image_image_span_contrastive_loss = (
    image_image_span_contrastive_loss_compositional
)
base.image_image_order_consistency_loss = (
    image_image_order_consistency_loss_differentiable
)
base.model_config = model_config_full_quality


if __name__ == "__main__":
    base.main()
