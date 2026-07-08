from dataclasses import dataclass
import os

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


@dataclass
class SpanEncoding:
    embeddings: torch.Tensor
    starts: list[int]
    lengths: list[int]
    texts: list[str]
    text_length: int


class ArabicSpanTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="aubmindlab/bert-base-arabertv02",
        output_dim=128,
        max_span_chars=3,
        freeze_backbone=True,
        device="cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_span_chars = max_span_chars
        self.freeze_backbone = freeze_backbone
        self.device = torch.device(device)

        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "transformers is required for ArabicSpanTextEncoder. "
                "Install it or use TEXT_ENCODER_TYPE=char."
            )

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None
        local_files_only = (
            os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
            or os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

        self.projection = nn.Linear(hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self._span_feature_cache = {}
        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def enumerate_spans(self, text):
        starts = []
        lengths = []
        texts = []

        for start, char in enumerate(text):
            if char.isspace():
                starts.append(start)
                lengths.append(1)
                texts.append(char)
                continue

            max_end = min(len(text), start + self.max_span_chars)
            for end in range(start + 1, max_end + 1):
                span = text[start:end]
                if any(ch.isspace() for ch in span):
                    break
                starts.append(start)
                lengths.append(end - start)
                texts.append(span)

        return starts, lengths, texts

    def _encoded_inputs(self, spans):
        encoded = self.tokenizer(
            spans,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _pool_non_special_tokens(self, hidden, encoded):
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        if "special_tokens_mask" in encoded:
            non_special = (1 - encoded["special_tokens_mask"]).unsqueeze(-1).float()
            pool_mask = attention_mask * non_special
            empty_rows = pool_mask.sum(dim=1, keepdim=True) == 0
            pool_mask = torch.where(empty_rows, attention_mask, pool_mask)
        else:
            pool_mask = attention_mask

        summed = (hidden * pool_mask).sum(dim=1)
        counts = pool_mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def clear_cache(self):
        self._span_feature_cache.clear()

    def _get_frozen_span_features(self, text):
        cached = self._span_feature_cache.get(text)
        if cached is not None:
            starts, lengths, spans, pooled_cpu = cached
            return starts, lengths, spans, pooled_cpu.to(self.device, non_blocking=True)

        starts, lengths, spans = self.enumerate_spans(text)
        if not spans:
            pooled = torch.empty(
                0,
                self.projection.in_features,
                device=self.device,
            )
            return starts, lengths, spans, pooled

        encoded = self._encoded_inputs(spans)
        backbone_inputs = {
            key: value
            for key, value in encoded.items()
            if key != "special_tokens_mask"
        }

        with torch.no_grad():
            outputs = self.backbone(**backbone_inputs)
            pooled = self._pool_non_special_tokens(outputs.last_hidden_state, encoded)

        self._span_feature_cache[text] = (
            starts,
            lengths,
            spans,
            pooled.detach().cpu(),
        )
        return starts, lengths, spans, pooled

    def forward(self, text):
        if self.freeze_backbone:
            starts, lengths, spans, pooled = self._get_frozen_span_features(text)
            projected = self.projection(pooled)
            return SpanEncoding(
                embeddings=self.norm(projected),
                starts=starts,
                lengths=lengths,
                texts=spans,
                text_length=len(text),
            )

        starts, lengths, spans = self.enumerate_spans(text)
        if not spans:
            return SpanEncoding(
                embeddings=torch.empty(0, self.projection.out_features, device=self.device),
                starts=[],
                lengths=[],
                texts=[],
                text_length=len(text),
            )

        encoded = self._encoded_inputs(spans)
        backbone_inputs = {
            key: value
            for key, value in encoded.items()
            if key != "special_tokens_mask"
        }

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**backbone_inputs)
        else:
            outputs = self.backbone(**backbone_inputs)

        pooled = self._pool_non_special_tokens(outputs.last_hidden_state, encoded)
        projected = self.projection(pooled)
        return SpanEncoding(
            embeddings=self.norm(projected),
            starts=starts,
            lengths=lengths,
            texts=spans,
            text_length=len(text),
        )
