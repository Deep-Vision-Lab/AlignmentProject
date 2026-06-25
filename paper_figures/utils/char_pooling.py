"""Shared D3TW-guided character pooling helpers for paper figures.

Figure scripts must use the same D3TW assignment/pooling implementation as
training.  This module is intentionally a thin wrapper around
``alignment_pooling.py``.
"""

import warnings

import torch
import torch.nn.functional as F

from alignment_pooling import (
    hard_d3tw_path_from_similarity,
    path_to_assignment,
    pool_visual_by_assignment,
    groups_from_assignment,
)


def compute_d3tw_char_pool_for_sample(
    visual_emb,
    text_emb,
    transcript_chars,
    detach_assignment=True,
):
    """
    Compute similarity, hard restricted-D3TW assignment, and character pooling.

    Args:
        visual_emb: [S, D] normalized visual window embeddings.
        text_emb: [T, D] normalized transcript-character embeddings.
        transcript_chars: list[str] with length T.

    Returns:
        dict with sim [T,S], path, assignment [T,S], groups, pooled_visual [T,D],
        valid_mask [T], and counts [T].
    """
    if visual_emb.dim() != 2:
        raise ValueError(f"visual_emb must have shape [S,D], got {tuple(visual_emb.shape)}")
    if text_emb.dim() != 2:
        raise ValueError(f"text_emb must have shape [T,D], got {tuple(text_emb.shape)}")
    if visual_emb.shape[1] != text_emb.shape[1]:
        raise ValueError(
            f"Embedding dims differ: visual D={visual_emb.shape[1]} text D={text_emb.shape[1]}"
        )

    sim = text_emb @ visual_emb.T
    T, S = sim.shape
    if len(transcript_chars) != T:
        warnings.warn(
            f"Transcript length {len(transcript_chars)} does not match similarity T={T}.",
            RuntimeWarning,
        )

    path = hard_d3tw_path_from_similarity(sim)
    if not path:
        warnings.warn(
            f"Restricted D3TW path is empty for T={T}, S={S}; character pooling unavailable.",
            RuntimeWarning,
        )
        assignment = torch.zeros((T, S), device=sim.device, dtype=sim.dtype)
        pooled_visual = visual_emb.new_zeros((T, visual_emb.shape[1]))
        valid_mask = torch.zeros((T,), device=sim.device, dtype=torch.bool)
        counts = torch.zeros((T,), device=sim.device, dtype=sim.dtype)
        groups = [[] for _ in range(T)]
    else:
        assignment = path_to_assignment(path, T, S, device=sim.device)
        pooled_visual, valid_mask, counts = pool_visual_by_assignment(
            visual_emb=visual_emb,
            assignment=assignment,
            detach_assignment=detach_assignment,
        )
        groups = groups_from_assignment(assignment)

    if bool((counts == 0).any().item()):
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten().cpu().tolist()
        warnings.warn(f"Empty D3TW assignment rows for character indices: {empty}", RuntimeWarning)

    assert pooled_visual.shape[0] == T
    assert pooled_visual.shape[1] == visual_emb.shape[1]
    return {
        "sim": sim,
        "path": path,
        "assignment": assignment,
        "groups": groups,
        "pooled_visual": pooled_visual,
        "valid_mask": valid_mask,
        "counts": counts,
    }


def char_pool_predictions(
    pooled_visual,
    transcript_chars,
    char_bank_embeddings,
    char_to_idx,
    idx_to_char,
    valid_mask=None,
    topk=5,
):
    """Return per-character top-k predictions and aggregate accuracy stats."""
    if char_bank_embeddings is None or char_to_idx is None or idx_to_char is None:
        return None, {
            "char_pool_top1": None,
            "char_pool_top5": None,
            "char_pool_valid_chars": 0,
        }

    pooled = F.normalize(pooled_visual, dim=-1)
    bank = F.normalize(char_bank_embeddings.to(pooled.device), dim=-1)
    logits = pooled @ bank.T
    k = min(int(topk), logits.shape[-1])
    top_idx = logits.topk(k, dim=-1).indices.detach().cpu().tolist()

    rows = []
    top1_hits = []
    top5_hits = []
    for j, char in enumerate(transcript_chars):
        valid = True if valid_mask is None else bool(valid_mask[j].item())
        pred_chars = [idx_to_char[idx] for idx in top_idx[j]]
        target_known = char in char_to_idx
        correct_top1 = bool(valid and target_known and pred_chars and pred_chars[0] == char)
        correct_top5 = bool(valid and target_known and char in pred_chars)
        rows.append({
            "char_index": j,
            "char": char,
            "top1_pred": pred_chars[0] if pred_chars else None,
            "top5_pred": pred_chars,
            "correct": correct_top1,
            "valid": valid,
        })
        if valid and target_known:
            top1_hits.append(float(correct_top1))
            top5_hits.append(float(correct_top5))

    return rows, {
        "char_pool_top1": float(sum(top1_hits) / len(top1_hits)) if top1_hits else 0.0,
        "char_pool_top5": float(sum(top5_hits) / len(top5_hits)) if top5_hits else 0.0,
        "char_pool_valid_chars": int(len(top1_hits)),
    }


def group_records(transcript_chars, groups, predictions=None):
    """Build JSON-serializable character-window group records."""
    pred_by_idx = {}
    if predictions:
        pred_by_idx = {row["char_index"]: row for row in predictions}

    records = []
    for j, char in enumerate(transcript_chars):
        pred = pred_by_idx.get(j, {})
        group = list(groups[j]) if j < len(groups) else []
        records.append({
            "char_index": int(j),
            "char": char,
            "assigned_windows": [int(i) for i in group],
            "num_windows": int(len(group)),
            "top1_pred": pred.get("top1_pred"),
            "top5_pred": pred.get("top5_pred"),
            "correct": pred.get("correct"),
        })
    return records
