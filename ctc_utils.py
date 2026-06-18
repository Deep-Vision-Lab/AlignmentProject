import json
import os
import warnings

import torch
import torch.nn.functional as F

from Parameters import (
    ctc_blank_token,
    ctc_reduction,
    ctc_zero_infinity,
)


class CTCVocabulary:
    """Character vocabulary for PyTorch CTCLoss targets."""

    def __init__(self, chars=None, blank_token=ctc_blank_token):
        self.blank_token = blank_token
        chars = list(chars or [])
        chars = [ch for ch in chars if ch != blank_token]
        seen = set()
        unique_chars = []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                unique_chars.append(ch)

        self.idx_to_char = [blank_token] + unique_chars
        self.char_to_idx = {ch: i for i, ch in enumerate(self.idx_to_char)}
        self.blank_idx = self.char_to_idx[blank_token]

    @classmethod
    def from_texts(cls, texts, blank_token=ctc_blank_token):
        chars = []
        seen = set()
        for text in texts:
            for ch in text:
                if ch == blank_token or ch in seen:
                    continue
                seen.add(ch)
                chars.append(ch)
        return cls(chars=chars, blank_token=blank_token)

    def __len__(self):
        return len(self.idx_to_char)

    def to_dict(self):
        return {
            "blank_token": self.blank_token,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "blank_idx": self.blank_idx,
        }

    @classmethod
    def from_dict(cls, data):
        blank_token = data.get("blank_token", ctc_blank_token)
        idx_to_char = list(data["idx_to_char"])
        chars = [ch for ch in idx_to_char if ch != blank_token]
        vocab = cls(chars=chars, blank_token=blank_token)
        if data.get("blank_idx", vocab.blank_idx) != vocab.blank_idx:
            raise ValueError("Loaded CTC vocabulary has an inconsistent blank_idx.")
        return vocab

    def save_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def encode_text(self, text):
        ids = []
        for ch in text:
            if ch == self.blank_token:
                continue
            if ch not in self.char_to_idx:
                raise KeyError(f"Character {ch!r} is not in the CTC vocabulary.")
            idx = self.char_to_idx[ch]
            if idx != self.blank_idx:
                ids.append(idx)
        encoded = torch.tensor(ids, dtype=torch.long)
        assert self.blank_idx not in encoded.tolist()
        return encoded

    def encode_batch(self, texts):
        encoded = [self.encode_text(text) for text in texts]
        lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
        if encoded:
            targets = torch.cat(encoded, dim=0)
        else:
            targets = torch.empty(0, dtype=torch.long)
        assert self.blank_idx not in targets.tolist()
        return targets, lengths


def _ctc_loss_module(blank_idx, reduction=None, zero_infinity=None):
    return torch.nn.CTCLoss(
        blank=blank_idx,
        reduction=ctc_reduction if reduction is None else reduction,
        zero_infinity=ctc_zero_infinity if zero_infinity is None else zero_infinity,
    )


def _filter_valid_ctc_texts(texts, input_length, ctc_vocab, device):
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        try:
            target = ctc_vocab.encode_text(text)
        except KeyError as exc:
            warnings.warn(
                f"Skipping CTC sample {i}: {exc}",
                RuntimeWarning,
            )
            continue
        if input_length < len(target):
            warnings.warn(
                f"Skipping CTC sample {i}: input_length={input_length} "
                f"< target_length={len(target)}",
                RuntimeWarning,
            )
            continue
        valid_texts.append(text)
        valid_indices.append(i)

    if not valid_texts:
        return None, None, None

    targets, target_lengths = ctc_vocab.encode_batch(valid_texts)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)
    input_lengths = torch.full(
        (len(valid_texts),), input_length, dtype=torch.long, device=device
    )
    assert torch.all(input_lengths >= target_lengths)
    return valid_indices, targets, target_lengths


def compute_ctc_loss(image_embeddings, pos_texts, ctc_head, ctc_vocab, device):
    logits = ctc_head(image_embeddings)
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
    input_length = log_probs.size(0)

    valid_indices, targets, target_lengths = _filter_valid_ctc_texts(
        pos_texts, input_length, ctc_vocab, device
    )
    if valid_indices is None:
        zero = image_embeddings.sum() * 0.0
        return zero, {
            "ctc_loss": 0.0,
            "mean_target_length": 0.0,
            "input_length": input_length,
            "vocab_size": len(ctc_vocab),
            "ctc_skipped": len(pos_texts),
        }

    valid_log_probs = log_probs[:, valid_indices, :]
    input_lengths = torch.full(
        (len(valid_indices),), input_length, dtype=torch.long, device=device
    )
    criterion = _ctc_loss_module(ctc_vocab.blank_idx)
    loss = criterion(valid_log_probs, targets, input_lengths, target_lengths)
    return loss, {
        "ctc_loss": loss.detach().item(),
        "mean_target_length": target_lengths.float().mean().detach().item(),
        "input_length": input_length,
        "vocab_size": len(ctc_vocab),
        "ctc_skipped": len(pos_texts) - len(valid_indices),
    }


def _single_ctc_cost(log_probs_sc, text, ctc_vocab, device):
    try:
        target = ctc_vocab.encode_text(text).to(device)
    except KeyError as exc:
        warnings.warn(f"Skipping CTC candidate: {exc}", RuntimeWarning)
        return None
    target_length = torch.tensor([target.numel()], dtype=torch.long, device=device)
    input_length = torch.tensor([log_probs_sc.size(0)], dtype=torch.long, device=device)
    if input_length.item() < target_length.item():
        warnings.warn(
            f"Skipping CTC candidate: input_length={input_length.item()} "
            f"< target_length={target_length.item()}",
            RuntimeWarning,
        )
        return None
    assert ctc_vocab.blank_idx not in target.tolist()
    criterion = _ctc_loss_module(ctc_vocab.blank_idx, reduction="none")
    return criterion(log_probs_sc.unsqueeze(1), target, input_length, target_length)[0]


@torch.no_grad()
def compute_ctc_cost(image_embeddings, text, ctc_head, ctc_vocab, device):
    """Return the scalar CTC cost for one image embedding sequence and transcript.

    Args:
        image_embeddings: [S, D] or [1, S, D] visual sequence.
        text: Transcript string in logical order.
        ctc_head: Linear head mapping D -> vocab size.
        ctc_vocab: CTCVocabulary instance.
        device: torch device/string.
    """
    if image_embeddings.dim() == 2:
        emb = image_embeddings.unsqueeze(0)
    elif image_embeddings.dim() == 3 and image_embeddings.size(0) == 1:
        emb = image_embeddings
    else:
        raise ValueError(
            "compute_ctc_cost expects image_embeddings with shape [S, D] or [1, S, D]."
        )

    emb = emb.to(device)
    logits = ctc_head(emb)
    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
    cost = _single_ctc_cost(log_probs, text, ctc_vocab, device)
    if cost is None:
        return float("nan")
    return float(cost.detach().cpu().item())


def compute_contrastive_ctc_loss(
    image_embeddings,
    pos_texts,
    neg_texts,
    ctc_head,
    ctc_vocab,
    tau,
    margin,
    loss_type,
    device,
):
    logits = ctc_head(image_embeddings)
    log_probs = F.log_softmax(logits, dim=-1)

    losses = []
    pos_cost_values = []
    neg_cost_values = []
    gap_values = []
    rank_values = []
    top1_values = []

    for i, pos_text in enumerate(pos_texts):
        pos_cost = _single_ctc_cost(log_probs[i], pos_text, ctc_vocab, device)
        if pos_cost is None:
            continue

        sample_neg_costs = []
        for neg_text in neg_texts[i]:
            neg_cost = _single_ctc_cost(log_probs[i], neg_text, ctc_vocab, device)
            if neg_cost is not None:
                sample_neg_costs.append(neg_cost)
        if not sample_neg_costs:
            continue

        neg_costs = torch.stack(sample_neg_costs)
        mode = loss_type.lower()
        if mode == "infonce":
            costs = torch.cat([pos_cost.view(1), neg_costs], dim=0)
            logits_i = -costs / max(tau, 1e-6)
            labels = torch.zeros(1, dtype=torch.long, device=device)
            sample_loss = F.cross_entropy(logits_i.view(1, -1), labels)
        elif mode == "margin":
            sample_loss = torch.clamp(pos_cost - neg_costs + margin, min=0).mean()
        else:
            raise ValueError(f"Unknown contrastive CTC loss type: {loss_type}")

        rank = 1 + (neg_costs < pos_cost).sum()

        losses.append(sample_loss)
        pos_cost_values.append(pos_cost.detach())
        neg_cost_values.append(neg_costs.detach().mean())
        gap_values.append((neg_costs.detach().mean() - pos_cost.detach()))
        rank_values.append(rank.detach().float())
        top1_values.append((rank == 1).detach().float())

    if not losses:
        zero = image_embeddings.sum() * 0.0
        return zero, {
            "ctc_pos_cost": 0.0,
            "ctc_neg_cost": 0.0,
            "ctc_gap": 0.0,
            "ctc_pos_rank": 0.0,
            "ctc_top1": 0.0,
            "ctc_skipped": len(pos_texts),
        }

    loss = torch.stack(losses).mean()
    return loss, {
        "ctc_pos_cost": torch.stack(pos_cost_values).mean().item(),
        "ctc_neg_cost": torch.stack(neg_cost_values).mean().item(),
        "ctc_gap": torch.stack(gap_values).mean().item(),
        "ctc_pos_rank": torch.stack(rank_values).mean().item(),
        "ctc_top1": torch.stack(top1_values).mean().item(),
        "ctc_skipped": len(pos_texts) - len(losses),
    }
