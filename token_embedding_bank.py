"""Frozen n-gram token bank and token-level pooling losses."""

import torch
import torch.nn.functional as F


def embed_token_with_fallback(text_embedder, token, device):
    """Embed a token directly when possible, otherwise mean-pool characters."""
    token = str(token)
    try:
        emb = text_embedder(token)
        if emb.dim() == 1:
            return emb
        if emb.dim() == 2:
            if emb.shape[0] == 1:
                return emb[0]
            # Character-only embedders return one row per character. Mean is
            # the intended frozen fallback for multi-character text units.
            return emb.mean(dim=0)
    except Exception:
        pass

    char_vectors = []
    for ch in token:
        emb = text_embedder(ch)
        if emb.dim() == 2:
            emb = emb[0]
        char_vectors.append(emb)
    if not char_vectors:
        raise ValueError("Cannot embed an empty token")
    return torch.stack(char_vectors, dim=0).mean(dim=0).to(device)


def build_token_embedding_bank(text_embedder, tokens, device):
    """Build normalized frozen embeddings for n-gram text units."""
    tokens = list(tokens)
    if not tokens:
        raise ValueError("Cannot build token bank from an empty vocabulary")

    text_embedder.eval()
    for parameter in text_embedder.parameters():
        parameter.requires_grad_(False)

    vectors = []
    with torch.no_grad():
        for token in tokens:
            vec = embed_token_with_fallback(text_embedder, token, device)
            vectors.append(vec.to(device=device, dtype=torch.float32))
        embeddings = F.normalize(torch.stack(vectors, dim=0), dim=-1).detach()

    token_to_idx = {token: idx for idx, token in enumerate(tokens)}
    return embeddings, token_to_idx, tokens


def encode_text_units(
    text,
    text_unit_type,
    text_embedder,
    device,
    ngram_tokenizer=None,
):
    """Encode transcript as characters or n-gram tokens with frozen embeddings."""
    text_unit_type = str(text_unit_type).lower()
    if text_unit_type == "char":
        units = list(text)
        spans = [(idx, idx + 1) for idx in range(len(units))]
    elif text_unit_type == "ngram":
        if ngram_tokenizer is None:
            raise ValueError("ngram_tokenizer is required when text_unit_type='ngram'")
        units, spans = ngram_tokenizer.tokenize(text)
    else:
        raise ValueError(f"Unknown text_unit_type: {text_unit_type}")

    if not units:
        return units, spans, torch.empty((0, 0), device=device)

    text_embedder.eval()
    vectors = []
    with torch.no_grad():
        for unit in units:
            vec = embed_token_with_fallback(text_embedder, unit, device)
            vectors.append(vec.to(device=device, dtype=torch.float32))
        embeddings = F.normalize(torch.stack(vectors, dim=0), dim=-1)
    return units, spans, embeddings


def compute_token_pool_contrastive_loss(
    pooled_visual,
    units,
    token_bank_embeddings,
    token_to_idx,
    tau=0.07,
    valid_mask=None,
    counts=None,
):
    """Classify pooled token visual vectors against a frozen token bank."""
    if pooled_visual.dim() != 2:
        raise ValueError("pooled_visual must have shape [K,D]")
    if len(units) != pooled_visual.shape[0]:
        raise ValueError("units length must match pooled_visual rows")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    keep_indices = []
    target_ids = []
    for idx, unit in enumerate(units):
        if unit not in token_to_idx:
            continue
        if valid_mask is not None and not bool(valid_mask[idx].item()):
            continue
        keep_indices.append(idx)
        target_ids.append(token_to_idx[unit])

    if not keep_indices:
        return pooled_visual.new_tensor(0.0), {
            "token_pool_valid_tokens": 0,
            "token_pool_acc_top1": 0.0,
            "token_pool_acc_top5": 0.0,
            "mean_windows_per_token": 0.0,
            "min_windows_per_token": 0,
            "max_windows_per_token": 0,
        }

    keep = torch.tensor(keep_indices, device=pooled_visual.device, dtype=torch.long)
    targets = torch.tensor(target_ids, device=pooled_visual.device, dtype=torch.long)
    pooled_kept = F.normalize(pooled_visual[keep], dim=-1)
    bank = F.normalize(
        token_bank_embeddings.detach().to(
            device=pooled_visual.device,
            dtype=pooled_visual.dtype,
        ),
        dim=-1,
    )
    logits = pooled_kept @ bank.T
    logits = logits / tau
    loss = F.cross_entropy(logits, targets)

    pred = logits.argmax(dim=-1)
    top1 = (pred == targets).float().mean()
    top5_idx = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    top5 = (top5_idx == targets.unsqueeze(1)).any(dim=1).float().mean()

    if counts is not None:
        kept_counts = counts[keep].detach().float()
        mean_windows = float(kept_counts.mean().cpu()) if kept_counts.numel() else 0.0
        min_windows = int(kept_counts.min().cpu()) if kept_counts.numel() else 0
        max_windows = int(kept_counts.max().cpu()) if kept_counts.numel() else 0
    else:
        mean_windows = 0.0
        min_windows = 0
        max_windows = 0

    return loss, {
        "token_pool_loss": float(loss.detach().cpu()),
        "token_pool_valid_tokens": len(keep_indices),
        "token_pool_acc_top1": float(top1.detach().cpu()),
        "token_pool_acc_top5": float(top5.detach().cpu()),
        "mean_windows_per_token": mean_windows,
        "min_windows_per_token": min_windows,
        "max_windows_per_token": max_windows,
    }


def compute_char_aux_loss_from_token_pool(
    pooled_token_visual,
    units,
    spans,
    original_text,
    char_bank_embeddings,
    char_to_idx,
    tau=0.07,
    valid_mask=None,
):
    """Auxiliary CE for n-gram tokens that cover exactly one character."""
    if len(units) != pooled_token_visual.shape[0] or len(spans) != len(units):
        raise ValueError("units/spans length must match pooled_token_visual rows")
    keep_indices = []
    target_ids = []
    chars = list(original_text)
    for unit_idx, span in enumerate(spans):
        start, end = span
        if end - start != 1:
            continue
        if not (0 <= start < len(chars)):
            continue
        ch = chars[start]
        if ch not in char_to_idx:
            continue
        if valid_mask is not None and not bool(valid_mask[unit_idx].item()):
            continue
        keep_indices.append(unit_idx)
        target_ids.append(char_to_idx[ch])

    if not keep_indices:
        return pooled_token_visual.new_tensor(0.0), {
            "char_aux_valid_chars": 0,
            "char_aux_acc_top1": 0.0,
            "char_aux_acc_top5": 0.0,
        }

    keep = torch.tensor(keep_indices, device=pooled_token_visual.device, dtype=torch.long)
    targets = torch.tensor(target_ids, device=pooled_token_visual.device, dtype=torch.long)
    pooled_kept = F.normalize(pooled_token_visual[keep], dim=-1)
    bank = F.normalize(
        char_bank_embeddings.detach().to(
            device=pooled_token_visual.device,
            dtype=pooled_token_visual.dtype,
        ),
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
        "char_aux_loss": float(loss.detach().cpu()),
        "char_aux_valid_chars": len(keep_indices),
        "char_aux_acc_top1": float(top1.detach().cpu()),
        "char_aux_acc_top5": float(top5.detach().cpu()),
    }
