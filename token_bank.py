"""Frozen bigram-token bank and auxiliary bigram supervision utilities."""

from collections import Counter
from dataclasses import dataclass
import json
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


COMMON_ARABIC_LIGATURE_TOKENS = ["لا", "لل", "ال"]


def collect_bigram_tokens(
    texts,
    skip_spaces=True,
    min_freq=1,
    max_vocab_size=5000,
    include_ligatures=True,
):
    """Collect adjacent 2-character tokens from training transcripts."""
    counts = Counter()
    for text in texts:
        chars = list(text)
        for idx in range(len(chars) - 1):
            token = chars[idx] + chars[idx + 1]
            if skip_spaces and (" " in token):
                continue
            counts[token] += 1

    min_freq = max(1, int(min_freq))
    tokens = [token for token, count in counts.items() if count >= min_freq]
    tokens.sort(key=lambda token: (-counts[token], token))

    if include_ligatures:
        for token in COMMON_ARABIC_LIGATURE_TOKENS:
            if (not skip_spaces or " " not in token) and token not in tokens:
                tokens.append(token)

    max_vocab_size = int(max_vocab_size)
    if max_vocab_size > 0:
        tokens = tokens[:max_vocab_size]
    return tokens


def _embed_token_with_fallback(text_embedder, token, device):
    """Embed a 2-char token; fallback to mean(char embeddings) if needed."""
    try:
        emb = text_embedder(token)
        if emb.dim() == 2 and emb.shape[0] == 1:
            return emb[0]
        if emb.dim() == 2 and emb.shape[0] == len(token):
            return emb.mean(dim=0)
        if emb.dim() == 1:
            return emb
    except Exception:
        pass

    char_embs = text_embedder(token)
    if char_embs.dim() != 2:
        raise RuntimeError(f"Could not embed bigram token {token!r}")
    return char_embs.mean(dim=0)


def build_token_bank(text_embedder, tokens, device):
    """Build a normalized frozen token bank with shape [V_token, D]."""
    tokens = list(tokens)
    if not tokens:
        raise ValueError("Cannot build token bank from an empty token list")
    if any(len(token) != 2 for token in tokens):
        raise ValueError("Bigram token bank entries must be exactly two characters")

    text_embedder.eval()
    for parameter in text_embedder.parameters():
        parameter.requires_grad_(False)

    vectors = []
    with torch.no_grad():
        for token in tokens:
            vec = _embed_token_with_fallback(text_embedder, token, device)
            vectors.append(vec.to(device=device, dtype=torch.float32))
        embeddings = F.normalize(torch.stack(vectors, dim=0), dim=-1).detach()

    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    return embeddings, token_to_idx, tokens


def save_token_bank_json(path, token_to_idx, idx_to_token):
    payload = {
        "idx_to_token": list(idx_to_token),
        "token_to_idx": dict(token_to_idx),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


class BigramFusionMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def build_adjacent_pair_visuals(
    pooled_visual,
    transcript_chars,
    token_to_idx,
    fusion="mean",
    skip_spaces=True,
    fusion_mlp=None,
):
    """Build adjacent visual bigram vectors from pooled character vectors.

    This function never pools over the whole line. The only visual inputs are
    neighboring D3TW-pooled character vectors M[j] and M[j+1].
    """
    if pooled_visual.dim() != 2:
        raise ValueError("pooled_visual must have shape [T,D]")
    if len(transcript_chars) != pooled_visual.shape[0]:
        raise ValueError("transcript_chars length must match pooled_visual rows")

    device = pooled_visual.device
    pair_vecs = []
    target_ids = []
    metadata = []
    fusion = str(fusion).lower()

    for char_idx in range(len(transcript_chars) - 1):
        token = transcript_chars[char_idx] + transcript_chars[char_idx + 1]
        if skip_spaces and " " in token:
            continue
        if token not in token_to_idx:
            continue
        left = pooled_visual[char_idx]
        right = pooled_visual[char_idx + 1]
        if fusion == "mean":
            pair_vec = F.normalize((left + right) / 2.0, dim=-1)
        elif fusion == "mlp":
            if fusion_mlp is None:
                raise ValueError("fusion='mlp' requires fusion_mlp")
            pair_vec = fusion_mlp(torch.cat([left, right], dim=-1).unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f"Unknown bigram fusion mode: {fusion}")
        pair_vecs.append(pair_vec)
        target_ids.append(token_to_idx[token])
        metadata.append({
            "start_char_index": char_idx,
            "end_char_index": char_idx + 1,
            "token": token,
            "chars": [transcript_chars[char_idx], transcript_chars[char_idx + 1]],
        })

    if not pair_vecs:
        empty_vecs = pooled_visual.new_empty((0, pooled_visual.shape[1]))
        empty_ids = torch.empty((0,), device=device, dtype=torch.long)
        return empty_vecs, empty_ids, metadata

    return (
        torch.stack(pair_vecs, dim=0),
        torch.tensor(target_ids, device=device, dtype=torch.long),
        metadata,
    )


def compute_bigram_token_contrastive_loss(
    pair_visuals,
    target_token_ids,
    token_bank_embeddings,
    tau=0.07,
):
    """Classify adjacent pooled-character visual pairs against token bank."""
    if pair_visuals.numel() == 0:
        return pair_visuals.new_tensor(0.0), {
            "bigram_token_valid_pairs": 0,
            "bigram_token_acc_top1": 0.0,
            "bigram_token_acc_top5": 0.0,
        }
    if tau <= 0:
        raise ValueError("tau must be > 0")

    pair_visuals = F.normalize(pair_visuals, dim=-1)
    bank = F.normalize(
        token_bank_embeddings.detach().to(
            device=pair_visuals.device,
            dtype=pair_visuals.dtype,
        ),
        dim=-1,
    )
    logits = pair_visuals @ bank.T
    logits = logits / tau
    loss = F.cross_entropy(logits, target_token_ids)

    pred = logits.argmax(dim=-1)
    top1 = (pred == target_token_ids).float().mean()
    top5_idx = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    top5 = (top5_idx == target_token_ids.unsqueeze(1)).any(dim=1).float().mean()
    return loss, {
        "bigram_token_loss": float(loss.detach().cpu()),
        "bigram_token_valid_pairs": int(pair_visuals.shape[0]),
        "bigram_token_acc_top1": float(top1.detach().cpu()),
        "bigram_token_acc_top5": float(top5.detach().cpu()),
    }


def get_aux_weight(epoch, base_weight, warmup_epochs, ramp_epochs):
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return base_weight
    progress = min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
    return base_weight * progress


@dataclass(frozen=True)
class TokenBank:
    token_to_idx: Dict[str, int]
    idx_to_token: List[str]
    embeddings: torch.Tensor
