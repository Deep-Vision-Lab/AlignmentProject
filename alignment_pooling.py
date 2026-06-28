"""D3TW-guided character pooling and frozen character-bank utilities."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


Path = Sequence[Tuple[int, int]]


def hard_d3tw_path_from_similarity(sim: torch.Tensor) -> List[Tuple[int, int]]:
    """Extract a hard global path with diagonal and character-stay moves.

    Args:
        sim: Similarity matrix with shape ``[T, S]``.

    Returns:
        A forward-ordered list of ``(char_idx, visual_idx)`` cells. The path
        starts at ``(0, 0)`` and ends at ``(T - 1, S - 1)``.

    A diagonal move advances both axes. A stay move keeps the same character
    and consumes the next visual position, allowing consecutive windows to be
    assigned to one character. Hard path extraction is intentionally detached.
    """
    if sim.dim() != 2:
        raise ValueError(f"sim must have shape [T, S], got {tuple(sim.shape)}")
    num_chars, num_visual = sim.shape
    if num_chars == 0 or num_visual == 0:
        return []
    if num_chars > num_visual:
        # This restricted topology cannot advance the character axis without
        # also consuming a visual position.
        return []

    dist = -sim.detach()
    # Path decisions do not need to remain on the accelerator. Float32 also
    # avoids half-precision accumulation when AMP produced the similarities.
    dist = dist.to(device="cpu", dtype=torch.float32)
    inf = torch.tensor(float("inf"), dtype=dist.dtype)
    cost = torch.full_like(dist, inf)
    # 0 = diagonal, 1 = stay. Only reachable cells are backtracked.
    backptr = torch.zeros((num_chars, num_visual), dtype=torch.uint8)

    cost[0, 0] = dist[0, 0]
    for visual_idx in range(1, num_visual):
        cost[0, visual_idx] = cost[0, visual_idx - 1] + dist[0, visual_idx]
        backptr[0, visual_idx] = 1

    for char_idx in range(1, num_chars):
        for visual_idx in range(char_idx, num_visual):
            diagonal_cost = cost[char_idx - 1, visual_idx - 1]
            stay_cost = cost[char_idx, visual_idx - 1]
            if diagonal_cost <= stay_cost:
                predecessor = diagonal_cost
            else:
                predecessor = stay_cost
                backptr[char_idx, visual_idx] = 1
            cost[char_idx, visual_idx] = predecessor + dist[char_idx, visual_idx]

    char_idx = num_chars - 1
    visual_idx = num_visual - 1
    path: List[Tuple[int, int]] = []
    while True:
        path.append((char_idx, visual_idx))
        if char_idx == 0 and visual_idx == 0:
            break
        if backptr[char_idx, visual_idx].item() == 1:
            visual_idx -= 1
        else:
            char_idx -= 1
            visual_idx -= 1

    path.reverse()
    assert path[0] == (0, 0)
    assert path[-1] == (num_chars - 1, num_visual - 1)
    return path


def path_to_assignment(
    path: Path,
    num_chars: int,
    num_visual: int,
    device=None,
) -> torch.Tensor:
    """Convert a hard path to a binary assignment matrix of shape ``[T, S]``."""
    assignment = torch.zeros(
        (num_chars, num_visual), device=device, dtype=torch.float32
    )
    for char_idx, visual_idx in path:
        if not (0 <= char_idx < num_chars and 0 <= visual_idx < num_visual):
            raise ValueError(
                f"Path cell {(char_idx, visual_idx)} is outside assignment "
                f"shape {(num_chars, num_visual)}"
            )
        assignment[char_idx, visual_idx] = 1.0
    return assignment


def normalize_assignment(assignment: torch.Tensor, eps: float = 1e-8):
    """Row-normalize ``[T, S]`` assignments and return validity metadata."""
    if assignment.dim() != 2:
        raise ValueError(
            f"assignment must have shape [T, S], got {tuple(assignment.shape)}"
        )
    counts = assignment.sum(dim=1)
    weights = assignment / (counts.unsqueeze(1) + eps)
    valid_mask = counts > 0
    return weights, valid_mask, counts


def pool_visual_by_assignment(
    visual_emb: torch.Tensor,
    assignment: torch.Tensor,
    detach_assignment: bool = True,
    min_windows_per_char: int = 1,
    eps: float = 1e-8,
):
    """Merge visual windows assigned to each transcript character.

    Args:
        visual_emb: Visual window embeddings with shape ``[S, D]``.
        assignment: Character-to-window assignment with shape ``[T, S]``.

    Returns:
        ``pooled_visual [T, D]``, ``valid_mask [T]``, and ``counts [T]``.
    """
    assert visual_emb.dim() == 2
    assert assignment.dim() == 2
    assert assignment.shape[1] == visual_emb.shape[0]
    if min_windows_per_char < 1:
        raise ValueError("min_windows_per_char must be >= 1")

    if detach_assignment:
        assignment = assignment.detach()
    assignment = assignment.to(device=visual_emb.device, dtype=visual_emb.dtype)
    weights, nonempty_mask, counts = normalize_assignment(assignment, eps)
    valid_mask = nonempty_mask & (counts >= min_windows_per_char)

    # F.normalize(..., dim=-1) only normalizes each vector; it does not merge
    # windows. This matrix multiplication is the actual D3TW-guided merge:
    # assignment[j, i] states whether window i belongs to character j, so
    # pooled_visual[j] is the mean of only the windows assigned to character j.
    pooled_visual = weights @ visual_emb
    pooled_visual = F.normalize(pooled_visual, dim=-1, eps=eps)

    assert pooled_visual.shape[0] == assignment.shape[0]
    assert pooled_visual.shape[1] == visual_emb.shape[1]
    return pooled_visual, valid_mask, counts


def groups_from_assignment(assignment: torch.Tensor) -> List[List[int]]:
    """Return the visual indices assigned to each character row."""
    if assignment.dim() != 2:
        raise ValueError("assignment must have shape [T, S]")
    return [
        torch.nonzero(row > 0, as_tuple=False).flatten().cpu().tolist()
        for row in assignment
    ]


def collect_unique_chars(texts: Iterable[str], skip_spaces: bool = False) -> List[str]:
    """Collect and sort unique characters from training transcripts."""
    chars = {char for text in texts for char in text}
    if skip_spaces:
        chars = {char for char in chars if char != " "}
    return sorted(chars)


def build_char_bank(text_embedder, chars: Sequence[str], device):
    """Build a normalized, frozen character bank with shape ``[V, D]``."""
    chars = list(chars)
    if not chars:
        raise ValueError("Cannot build a character bank from an empty vocabulary")
    if any(len(char) != 1 for char in chars):
        raise ValueError("Every character-bank entry must be one character")

    text_embedder.eval()
    for parameter in text_embedder.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        embeddings = text_embedder("".join(chars)).to(device=device, dtype=torch.float32)
        embeddings = F.normalize(embeddings, dim=-1).detach()

    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    return embeddings, char_to_idx, chars


def compute_char_pool_contrastive_loss(
    pooled_visual: torch.Tensor,
    transcript_chars: list,
    char_bank_embeddings: torch.Tensor,
    char_to_idx: dict,
    tau: float = 0.07,
    valid_mask: torch.Tensor = None,
    skip_spaces: bool = False,
):
    """Classify each valid pooled visual vector against the frozen char bank."""
    if pooled_visual.dim() != 2:
        raise ValueError("pooled_visual must have shape [T, D]")
    if len(transcript_chars) != pooled_visual.shape[0]:
        raise ValueError("transcript_chars length must equal pooled_visual rows")
    if char_bank_embeddings.dim() != 2:
        raise ValueError("char_bank_embeddings must have shape [V, D]")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    device = pooled_visual.device
    keep_indices = []
    target_ids = []
    for char_idx, char in enumerate(transcript_chars):
        if skip_spaces and char == " ":
            continue
        if char not in char_to_idx:
            continue
        if valid_mask is not None and not bool(valid_mask[char_idx].item()):
            continue
        keep_indices.append(char_idx)
        target_ids.append(char_to_idx[char])

    if not keep_indices:
        return pooled_visual.new_tensor(0.0), {
            "char_pool_valid_chars": 0,
            "char_pool_acc_top1": 0.0,
            "char_pool_acc_top5": 0.0,
        }

    keep = torch.tensor(keep_indices, device=device, dtype=torch.long)
    targets = torch.tensor(target_ids, device=device, dtype=torch.long)
    pooled_kept = F.normalize(pooled_visual[keep], dim=-1)
    bank = F.normalize(
        char_bank_embeddings.detach().to(device=device, dtype=pooled_visual.dtype),
        dim=-1,
    )
    logits = pooled_kept @ bank.T
    logits = logits / tau
    loss = F.cross_entropy(logits, targets)

    pred = logits.argmax(dim=-1)
    top1 = (pred == targets).float().mean()
    top5_idx = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    top5 = (top5_idx == targets.unsqueeze(1)).any(dim=1).float().mean()
    return loss, {
        "char_pool_loss": float(loss.detach().cpu()),
        "char_pool_valid_chars": len(keep_indices),
        "char_pool_acc_top1": float(top1.detach().cpu()),
        "char_pool_acc_top5": float(top5.detach().cpu()),
    }


def get_char_pool_weight(epoch, base_weight, warmup_epochs, ramp_epochs):
    """Apply warmup followed by a linear ramp to the character-loss weight."""
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return base_weight
    progress = min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
    return base_weight * progress


# Backward-compatible aliases for earlier local experiments.
extract_restricted_d3tw_path = hard_d3tw_path_from_similarity
hard_path_to_assignment = path_to_assignment
assignment_to_groups = groups_from_assignment


def pool_windows_by_assignment(
    visual_emb,
    assignment,
    method="hard_mean",
    detach_alignment=True,
    eps=1e-8,
    min_windows_per_char=1,
):
    if method != "hard_mean":
        raise ValueError("Only hard_mean is supported by hard D3TW assignments")
    pooled, valid, _ = pool_visual_by_assignment(
        visual_emb,
        assignment,
        detach_assignment=detach_alignment,
        min_windows_per_char=min_windows_per_char,
        eps=eps,
    )
    return pooled, valid


@dataclass(frozen=True)
class CharacterBank:
    """Compatibility container used by older callers."""

    char_to_idx: Dict[str, int]
    idx_to_char: List[str]
    embeddings: torch.Tensor
