"""Adapt the bridge-specific real-image/text loss to V2 multi-island positives.

Positive synthetic lines now contain unrelated distractors, so the bridge-specific
text ranking must not treat the entire synthetic transcript as content that should
align with the real anchor. This runtime extracts the exact shared word runs between
real and positive synthetic transcripts and scores each island separately. For the
rare single-token fallback it uses >=3-character matching blocks; distractors cannot
create such a block under the V2 negative guarantee.
"""
from __future__ import annotations

from difflib import SequenceMatcher

import torch


def _word_islands(anchor_text: str, synthetic_text: str) -> list[str]:
    anchor_words = set(str(anchor_text).strip().split())
    islands: list[str] = []
    current: list[str] = []
    for word in str(synthetic_text).strip().split():
        if word in anchor_words:
            current.append(word)
        elif current:
            islands.append(" ".join(current))
            current = []
    if current:
        islands.append(" ".join(current))
    return [item for item in islands if item.strip()]


def _character_fallback(anchor_text: str, synthetic_text: str, min_chars: int = 3) -> list[str]:
    anchor = "".join(str(anchor_text).strip().split())
    synthetic = "".join(str(synthetic_text).strip().split())
    if not anchor or not synthetic:
        return []
    blocks = SequenceMatcher(None, anchor, synthetic, autojunk=False).get_matching_blocks()
    return [synthetic[block.b:block.b + block.size] for block in blocks if block.size >= int(min_chars)]


def shared_islands(anchor_text: str, synthetic_text: str) -> list[str]:
    islands = _word_islands(anchor_text, synthetic_text)
    return islands or _character_fallback(anchor_text, synthetic_text, min_chars=3)


def install(bridge_module) -> None:
    def direct_cross_text_loss(base, text_encoder, texts2, emb1, labels, texts1=None):
        reference = emb1[0]
        zero = reference.new_tensor(0.0)
        empty_stats = {
            "bridge_cross_text_loss": 0.0,
            "bridge_text_path_rank_loss": 0.0,
            "bridge_text_score_rank_loss": 0.0,
            "bridge_text_positive_floor_loss": 0.0,
            "bridge_text_negative_ceiling_loss": 0.0,
            "bridge_text_pos_fraction": 0.0,
            "bridge_text_neg_fraction": 0.0,
            "bridge_text_fraction_gap": 0.0,
            "bridge_text_pos_score": 0.0,
            "bridge_text_neg_score": 0.0,
            "bridge_text_shared_islands_mean": 0.0,
            "bridge_text_active": 0.0,
        }
        if not torch.is_grad_enabled() or bridge_module._env_float("BRIDGE_CROSS_TEXT_WEIGHT", 0.10) <= 0:
            return zero, empty_stats
        if texts1 is None:
            raise RuntimeError("Bridge V2 multi-island loss requires real anchor texts1")

        positive_indices = [
            i for i, label in enumerate(labels)
            if str(label) in bridge_module.legacy.POSITIVE_LABELS
        ]
        negative_indices = [
            i for i, label in enumerate(labels)
            if str(label) == bridge_module.legacy.EXTRA_LABEL
        ]
        max_pos = bridge_module._env_int("BRIDGE_CROSS_TEXT_MAX_POS_PER_BATCH", 4)
        max_neg = bridge_module._env_int("BRIDGE_CROSS_TEXT_MAX_NEG_PER_BATCH", 4)
        if max_pos > 0:
            positive_indices = positive_indices[:max_pos]
        if max_neg > 0:
            negative_indices = negative_indices[:max_neg]
        if not positive_indices or not negative_indices:
            return zero, empty_stats

        _context1, norm_local1, _ink1, _raw1 = emb1

        def one_text_metric(index: int, text: str):
            text_embedding = base.embed_single_text(text_encoder, text).detach()
            if text_embedding.ndim == 1:
                text_embedding = text_embedding.unsqueeze(0)
            similarity = torch.matmul(norm_local1[index], text_embedding.T)
            return bridge_module.sequence.soft_local_alignment_metrics(
                similarity,
                bridge_module._env_float("BRIDGE_CROSS_TEXT_THRESHOLD", 0.50),
                bridge_module._env_float("BRIDGE_CROSS_TEXT_GAP", -0.30),
                bridge_module._env_float("BRIDGE_CROSS_TEXT_TEMPERATURE", 0.03),
            )

        positive_metrics = []
        island_counts = []
        for index in positive_indices:
            islands = shared_islands(texts1[index], texts2[index])
            if not islands:
                continue
            metrics = [one_text_metric(index, island) for island in islands]
            positive_metrics.append(
                (
                    torch.stack([item[0] for item in metrics]).mean(),
                    torch.stack([item[1] for item in metrics]).mean(),
                )
            )
            island_counts.append(float(len(islands)))

        negative_metrics = [one_text_metric(index, texts2[index]) for index in negative_indices]
        if not positive_metrics or not negative_metrics:
            return zero, empty_stats

        pos_scores = torch.stack([item[0] for item in positive_metrics])
        pos_fractions = torch.stack([item[1] for item in positive_metrics])
        neg_scores = torch.stack([item[0] for item in negative_metrics])
        neg_fractions = torch.stack([item[1] for item in negative_metrics])

        path_rank_loss = torch.relu(
            bridge_module._env_float("BRIDGE_CROSS_TEXT_PATH_MARGIN", 0.10)
            - pos_fractions[:, None] + neg_fractions[None, :]
        ).mean()
        score_rank_loss = torch.relu(
            bridge_module._env_float("BRIDGE_CROSS_TEXT_SCORE_MARGIN", 0.10)
            - pos_scores[:, None] + neg_scores[None, :]
        ).mean()
        positive_floor_loss = torch.relu(
            bridge_module._env_float("BRIDGE_CROSS_TEXT_POSITIVE_FLOOR", 0.20) - pos_fractions
        ).mean()
        negative_ceiling_loss = torch.relu(
            neg_fractions - bridge_module._env_float("BRIDGE_CROSS_TEXT_NEGATIVE_CEILING", 0.15)
        ).mean()
        loss = (
            path_rank_loss
            + bridge_module._env_float("BRIDGE_CROSS_TEXT_SCORE_COMPONENT_WEIGHT", 0.20) * score_rank_loss
            + bridge_module._env_float("BRIDGE_CROSS_TEXT_POSITIVE_FLOOR_WEIGHT", 0.50) * positive_floor_loss
            + bridge_module._env_float("BRIDGE_CROSS_TEXT_NEGATIVE_CEILING_WEIGHT", 0.75) * negative_ceiling_loss
        )
        stats = {
            "bridge_cross_text_loss": float(loss.detach().item()),
            "bridge_text_path_rank_loss": float(path_rank_loss.detach().item()),
            "bridge_text_score_rank_loss": float(score_rank_loss.detach().item()),
            "bridge_text_positive_floor_loss": float(positive_floor_loss.detach().item()),
            "bridge_text_negative_ceiling_loss": float(negative_ceiling_loss.detach().item()),
            "bridge_text_pos_fraction": float(pos_fractions.mean().detach().item()),
            "bridge_text_neg_fraction": float(neg_fractions.mean().detach().item()),
            "bridge_text_fraction_gap": float((pos_fractions.mean() - neg_fractions.mean()).detach().item()),
            "bridge_text_pos_score": float(pos_scores.mean().detach().item()),
            "bridge_text_neg_score": float(neg_scores.mean().detach().item()),
            "bridge_text_shared_islands_mean": float(sum(island_counts) / max(1, len(island_counts))),
            "bridge_text_active": 1.0,
        }
        return loss, stats

    bridge_module._direct_cross_text_loss_v2 = direct_cross_text_loss

    # Replace the hook installer because the original v1 helper only forwards texts2.
    def install_direct_cross_text(base):
        original_pair_loss = bridge_module.sequence.shared._positive_pair_loss

        def bridge_pair_loss(base_arg, text_encoder, criterion, texts1, texts2, emb1, emb2, labels):
            pair_loss, order_loss, stats = original_pair_loss(
                base_arg, text_encoder, criterion, texts1, texts2, emb1, emb2, labels
            )
            direct_loss, direct_stats = direct_cross_text_loss(
                base_arg, text_encoder, texts2, emb1, labels, texts1=texts1
            )
            requested_weight = bridge_module._env_float("BRIDGE_CROSS_TEXT_WEIGHT", 0.10)
            outer_weight = float(getattr(base_arg.P, "image_pair_loss_weight", 0.0))
            if requested_weight > 0:
                if outer_weight <= 0:
                    raise RuntimeError(
                        "BRIDGE_CROSS_TEXT_WEIGHT requires IMAGE_PAIR_LOSS_WEIGHT > 0"
                    )
                pair_loss = pair_loss + (requested_weight / outer_weight) * direct_loss
            stats.update(direct_stats)
            return pair_loss, order_loss, stats

        bridge_module.sequence.shared._positive_pair_loss = bridge_pair_loss

    bridge_module._install_direct_cross_text = install_direct_cross_text
