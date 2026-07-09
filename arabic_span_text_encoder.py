from collections import OrderedDict
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
    max_span_chars: int


class ArabicSpanTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="aubmindlab/bert-base-arabertv02",
        output_dim=128,
        max_span_chars=3,
        freeze_backbone=True,
        device="cpu",
        strip_text_edges=True,
        cache_size=2048,
        cache_dtype="float16",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_span_chars = max_span_chars
        self.freeze_backbone = freeze_backbone
        self.device = torch.device(device)
        self.strip_text_edges = strip_text_edges
        self.cache_size = int(cache_size)
        self.cache_dtype = str(cache_dtype).lower()

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
        # LRU cache for frozen AraBERT pooled span features. Important:
        # generated hard negatives are usually unique and should not be cached
        # during training. By default, forward(..., use_cache=None) disables this
        # cache while self.training=True and enables it for eval/inference only.
        self._span_feature_cache = OrderedDict()
        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_text(self, text):
        if self.strip_text_edges:
            return text.strip()
        return text

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

    def cache_size_current(self):
        return len(self._span_feature_cache)

    def _should_use_cache(self, use_cache):
        if use_cache is not None:
            return bool(use_cache)
        # Training texts include many generated negatives that are nearly never
        # reused. Caching them causes RAM growth across batches, so the safe
        # default is: no cache while training, cache only for eval/inference.
        return not self.training

    def _cache_storage_dtype(self):
        if self.cache_dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if self.cache_dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.cache_dtype in {"float32", "fp32", "full"}:
            return torch.float32
        raise ValueError(
            "SPAN_FEATURE_CACHE_DTYPE must be one of float16, bfloat16, or float32. "
            f"Got {self.cache_dtype!r}."
        )

    def _cache_get(self, text, use_cache=True):
        if not use_cache or self.cache_size == 0:
            return None
        cached = self._span_feature_cache.get(text)
        if cached is None:
            return None
        self._span_feature_cache.move_to_end(text)
        starts, lengths, spans, pooled_cpu = cached
        pooled = pooled_cpu.to(self.device, non_blocking=True)
        # Projection weights are normally float32. Convert cached half precision
        # features back before Linear to avoid dtype mismatch outside autocast.
        if pooled.dtype != self.projection.weight.dtype:
            pooled = pooled.to(dtype=self.projection.weight.dtype)
        return starts, lengths, spans, pooled

    def _cache_put(self, text, starts, lengths, spans, pooled, use_cache=True):
        if not use_cache or self.cache_size == 0:
            return
        pooled_cpu = pooled.detach().to(device="cpu", dtype=self._cache_storage_dtype())
        self._span_feature_cache[text] = (starts, lengths, spans, pooled_cpu)
        self._span_feature_cache.move_to_end(text)
        if self.cache_size > 0:
            while len(self._span_feature_cache) > self.cache_size:
                self._span_feature_cache.popitem(last=False)

    def _get_frozen_span_features(self, text, use_cache=None):
        text = self._prepare_text(text)
        use_cache = self._should_use_cache(use_cache)
        cached = self._cache_get(text, use_cache=use_cache)
        if cached is not None:
            return cached

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

        self._cache_put(text, starts, lengths, spans, pooled, use_cache=use_cache)
        return starts, lengths, spans, pooled

    def forward(self, text, use_cache=None):
        if self.freeze_backbone:
            text = self._prepare_text(text)
            starts, lengths, spans, pooled = self._get_frozen_span_features(
                text,
                use_cache=use_cache,
            )
            projected = self.projection(pooled)
            return SpanEncoding(
                embeddings=self.norm(projected),
                starts=starts,
                lengths=lengths,
                texts=spans,
                text_length=len(text),
                max_span_chars=self.max_span_chars,
            )

        text = self._prepare_text(text)
        starts, lengths, spans = self.enumerate_spans(text)
        if not spans:
            return SpanEncoding(
                embeddings=torch.empty(0, self.projection.out_features, device=self.device),
                starts=[],
                lengths=[],
                texts=[],
                text_length=len(text),
                max_span_chars=self.max_span_chars,
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
            max_span_chars=self.max_span_chars,
        )
