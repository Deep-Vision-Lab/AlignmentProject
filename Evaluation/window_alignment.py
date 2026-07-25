"""Window-level alignment and clustering primitives used by evaluation scripts."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


UNALIGNED_TOKEN = "<UNALIGNED>"
BLANK_TOKEN = "<BLANK>"
SPACE_TOKEN = "<SPACE>"


@dataclass(frozen=True)
class ClusterTokenSummary:
    cluster: int
    token: str
    count: int
    purity: float
    token_counts: dict[str, int]


def token_for_alignment_step(step: dict) -> str:
    """Return a display token for one hard Span-DTW transition."""
    if bool(step.get("is_blank", False)):
        return BLANK_TOKEN
    if bool(step.get("is_space", False)):
        return SPACE_TOKEN
    raw = step.get("text", step.get("surface_text", step.get("raw_text", "")))
    token = str(raw).strip()
    return token or SPACE_TOKEN


def window_token_labels(
    path: Sequence[dict],
    n_windows: int,
    default: str = UNALIGNED_TOKEN,
) -> list[str]:
    """Assign every image window the token consumed by its Span-DTW step."""
    labels = [str(default) for _ in range(max(0, int(n_windows)))]
    for step in path:
        start = max(0, int(step.get("window_start", 0)))
        end = min(len(labels), int(step.get("window_end", start)))
        token = token_for_alignment_step(step)
        for index in range(start, max(start, end)):
            labels[index] = token
    return labels


def alignment_score_matrix(
    similarity: torch.Tensor | np.ndarray,
    mode: str = "mutual_z",
    clip: float = 4.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Convert raw window cosine values into discriminative NW match scores.

    Raw cosine matrices from contextual encoders often have a large positive
    background. With a negative gap score, ordinary Needleman-Wunsch then prefers
    a smooth diagonal even when a brighter off-diagonal ridge is more distinctive.

    ``mutual_z`` removes the per-row and per-column baselines and averages their
    z-scores. A cell is rewarded when it is unusually strong for both windows,
    while the order constraint still comes exclusively from Needleman-Wunsch.
    The raw cosine matrix remains available for reporting and display.
    """
    matrix = torch.as_tensor(similarity, dtype=torch.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D similarity matrix, got {tuple(matrix.shape)}")
    if matrix.numel() == 0:
        return matrix.clone()

    value = str(mode).strip().lower().replace("-", "_")
    if value == "raw":
        return matrix.clone()

    row_centered = matrix - matrix.mean(dim=1, keepdim=True)
    column_centered = matrix - matrix.mean(dim=0, keepdim=True)
    if value == "centered":
        return 0.5 * (row_centered + column_centered)
    if value != "mutual_z":
        raise ValueError("score mode must be raw, centered, or mutual-z")

    row_scale = torch.sqrt(row_centered.square().mean(dim=1, keepdim=True) + float(eps))
    column_scale = torch.sqrt(
        column_centered.square().mean(dim=0, keepdim=True) + float(eps)
    )
    scores = 0.5 * (
        row_centered / row_scale
        + column_centered / column_scale
    )
    if float(clip) > 0:
        scores = scores.clamp(min=-float(clip), max=float(clip))
    return scores


def attach_raw_similarities(result, raw_similarity: torch.Tensor | np.ndarray):
    """Keep a score-derived NW path but report the original cosine per match."""
    raw = (
        raw_similarity.detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(raw_similarity)
        else np.asarray(raw_similarity, dtype=np.float32)
    )
    steps = []
    for step in result.steps:
        similarity = None
        if step.index1 is not None and step.index2 is not None:
            similarity = float(raw[int(step.index1), int(step.index2)])
        steps.append(replace(step, similarity=similarity))
    return replace(result, steps=steps)


def consecutive_match_segments(
    result,
    min_similarity: float = -float("inf"),
) -> list[list]:
    """Split an NW traceback into strict consecutive diagonal match blocks.

    A block continues only for adjacent diagonal transitions, for example
    ``(i, j) -> (i+1, j+1)``. Any gap transition, non-adjacent match, or match below
    ``min_similarity`` closes the current block. This makes one visual mask/color
    correspond to one continuous aligned window region in both line images.
    """
    segments: list[list] = []
    current: list = []

    def flush() -> None:
        nonlocal current
        if current:
            segments.append(current)
            current = []

    for step in result.steps:
        is_match = (
            step.index1 is not None
            and step.index2 is not None
            and step.similarity is not None
            and float(step.similarity) >= float(min_similarity)
        )
        if not is_match:
            flush()
            continue

        if current:
            previous = current[-1]
            adjacent = (
                int(step.index1) == int(previous.index1) + 1
                and int(step.index2) == int(previous.index2) + 1
            )
            if not adjacent:
                flush()
        current.append(step)

    flush()
    return segments


def pca_project_2d(features: torch.Tensor | np.ndarray) -> np.ndarray:
    """Project embeddings to two dimensions with deterministic PCA/SVD."""
    array = (
        features.detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(features)
        else np.asarray(features, dtype=np.float32)
    )
    if array.ndim != 2:
        raise ValueError(f"Expected [N,D] features, got {array.shape}")
    if array.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    centered = array - array.mean(axis=0, keepdims=True)
    if array.shape[0] == 1 or np.allclose(centered, 0):
        return np.zeros((array.shape[0], 2), dtype=np.float32)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[: min(2, vh.shape[0])].T
    projected = centered @ components
    if projected.shape[1] == 1:
        projected = np.concatenate(
            [projected, np.zeros((projected.shape[0], 1), dtype=projected.dtype)],
            axis=1,
        )
    return projected[:, :2].astype(np.float32, copy=False)


def cosine_kmeans(
    features: torch.Tensor | np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic cosine K-means without a scikit-learn dependency."""
    tensor = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2:
        raise ValueError(f"Expected [N,D] features, got {tuple(tensor.shape)}")
    n_samples = int(tensor.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot cluster zero windows")
    k = max(1, min(int(n_clusters), n_samples))
    normalized = F.normalize(tensor, p=2, dim=-1)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(n_samples, (1,), generator=generator).item())
    chosen = [first]
    centers = [normalized[first]]
    while len(centers) < k:
        sims = normalized @ torch.stack(centers).T
        nearest_distance = 1.0 - sims.max(dim=1).values
        nearest_distance[torch.as_tensor(chosen, dtype=torch.long)] = -1.0
        next_index = int(torch.argmax(nearest_distance).item())
        chosen.append(next_index)
        centers.append(normalized[next_index])
    centers_t = torch.stack(centers)

    labels = torch.full((n_samples,), -1, dtype=torch.long)
    for _ in range(max(1, int(max_iter))):
        new_labels = torch.argmax(normalized @ centers_t.T, dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        new_centers = []
        assigned_similarity = (normalized * centers_t[labels]).sum(dim=1)
        for cluster in range(k):
            members = normalized[labels == cluster]
            if members.numel() == 0:
                replacement = int(torch.argmin(assigned_similarity).item())
                center = normalized[replacement]
            else:
                center = F.normalize(members.mean(dim=0), p=2, dim=0)
            new_centers.append(center)
        centers_t = torch.stack(new_centers)

    return labels.numpy(), centers_t.numpy()


def cluster_representatives(
    features: torch.Tensor | np.ndarray,
    labels: Sequence[int],
    centers: torch.Tensor | np.ndarray,
) -> dict[int, int]:
    """Choose the member closest in cosine similarity to each cluster center."""
    tensor = F.normalize(torch.as_tensor(features, dtype=torch.float32).cpu(), p=2, dim=-1)
    labels_np = np.asarray(labels, dtype=np.int64)
    centers_t = F.normalize(torch.as_tensor(centers, dtype=torch.float32).cpu(), p=2, dim=-1)
    representatives: dict[int, int] = {}
    for cluster in range(int(centers_t.shape[0])):
        members = np.flatnonzero(labels_np == cluster)
        if not len(members):
            continue
        member_tensor = tensor[torch.as_tensor(members, dtype=torch.long)]
        local = int(torch.argmax(member_tensor @ centers_t[cluster]).item())
        representatives[cluster] = int(members[local])
    return representatives


def summarize_cluster_tokens(
    labels: Sequence[int],
    tokens: Sequence[str],
) -> list[ClusterTokenSummary]:
    """Label each visual cluster by its majority aligned text token."""
    labels_np = np.asarray(labels, dtype=np.int64)
    if len(labels_np) != len(tokens):
        raise ValueError("labels and tokens must have the same length")
    summaries = []
    for cluster in sorted(set(int(value) for value in labels_np.tolist())):
        member_tokens = [str(tokens[i]) for i in np.flatnonzero(labels_np == cluster)]
        counts = Counter(member_tokens)
        token, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        summaries.append(
            ClusterTokenSummary(
                cluster=cluster,
                token=token,
                count=len(member_tokens),
                purity=float(count) / max(1, len(member_tokens)),
                token_counts=dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
            )
        )
    return summaries


def window_alignment_metrics(result, labels1: Sequence[str], labels2: Sequence[str]) -> dict:
    """Summarize an image-only window NW path, with optional token agreement."""
    matched = [
        step
        for step in result.steps
        if step.index1 is not None and step.index2 is not None
    ]
    segments = consecutive_match_segments(result)
    segment_lengths = [len(segment) for segment in segments]
    similarities = [float(step.similarity) for step in matched if step.similarity is not None]
    comparable = []
    ignored = {UNALIGNED_TOKEN, BLANK_TOKEN}
    for step in matched:
        left = str(labels1[int(step.index1)])
        right = str(labels2[int(step.index2)])
        if left in ignored or right in ignored:
            continue
        comparable.append(left == right)
    n1, n2 = len(labels1), len(labels2)
    return {
        "nw_score": float(result.score),
        "nw_normalized_score": float(result.normalized_score),
        "matched_window_pairs": len(matched),
        "consecutive_segments": len(segments),
        "longest_segment_pairs": max(segment_lengths, default=0),
        "mean_segment_pairs": float(np.mean(segment_lengths)) if segment_lengths else 0.0,
        "gap_in_line1": sum(step.index1 is None for step in result.steps),
        "gap_in_line2": sum(step.index2 is None for step in result.steps),
        "mean_matched_cosine": float(np.mean(similarities)) if similarities else 0.0,
        "line1_window_coverage": len({int(step.index1) for step in matched}) / max(1, n1),
        "line2_window_coverage": len({int(step.index2) for step in matched}) / max(1, n2),
        "token_agreement": float(np.mean(comparable)) if comparable else 0.0,
        "token_comparisons": len(comparable),
    }
