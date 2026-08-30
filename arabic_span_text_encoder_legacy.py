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


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SpanEncoding:
    embeddings: torch.Tensor
    starts: list[int]
    lengths: list[int]
    texts: list[str]
    text_length: int
    max_span_chars: int


class ArabicSpanTextEncoder(nn.Module):
    """Encode monotonic text spans with overlap-aware boundary context.

    The DP coverage remains non-overlapping: ``starts`` and ``lengths`` describe
    only the core text consumed by each transition. The encoded surface may add
    one look-ahead character. For example, consecutive one-character cores
    ``ب`` and ``ا`` in ``بات`` can be represented by the surfaces ``با`` and
    ``ات``. This matches overlapping image windows without changing the current
    Torch or JAX Span-DTW transition topology.
    """

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
        boundary_context_chars=None,
        include_space_context=None,
        space_token="<SPACE>",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_span_chars = max_span_chars
        self.freeze_backbone = freeze_backbone
        self.device = torch.device(device)
        self.strip_text_edges = strip_text_edges
        self.cache_size = int(cache_size)
        self.cache_dtype = str(cache_dtype).lower()
        self.space_token = str(space_token)

        if boundary_context_chars is None:
            boundary_context_chars = int(os.environ.get("SPAN_BOUNDARY_CONTEXT_CHARS", "1"))
        if include_space_context is None:
            include_space_context = _env_flag("SPAN_INCLUDE_SPACE_CONTEXT", True)
        boundary_context_chars = max(0, int(boundary_context_chars))
        self.register_buffer(
            "_boundary_context_chars_state",
            torch.tensor(boundary_context_chars, dtype=torch.int16),
        )
        self.register_buffer(
            "_include_space_context_state",
            torch.tensor(1 if include_space_context else 0, dtype=torch.uint8),
        )

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
        self.space_embedding = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.space_embedding, mean=0.0, std=0.02)

        # Cache only frozen backbone features. The trainable projection and space
        # embedding are applied after cache retrieval and therefore stay current.
        self._span_feature_cache = OrderedDict()
        self.to(self.device)

    @property
    def boundary_context_chars(self):
        return int(self._boundary_context_chars_state.item())

    @property
    def include_space_context(self):
        return bool(int(self._include_space_context_state.item()))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Disable new span semantics when loading a legacy checkpoint.

        New checkpoints store the context configuration as buffers. Legacy
        checkpoints do not, so they retain the old non-overlapping span behavior
        unless FORCE_OVERLAP_AWARE_SPANS=1 is explicitly requested.
        """
        force_overlap = _env_flag("FORCE_OVERLAP_AWARE_SPANS", False)
        boundary_key = prefix + "_boundary_context_chars_state"
        space_context_key = prefix + "_include_space_context_state"
        space_embedding_key = prefix + "space_embedding"

        if boundary_key not in state_dict:
            self._boundary_context_chars_state.fill_(1 if force_overlap else 0)
            state_dict[boundary_key] = self._boundary_context_chars_state.detach().clone()
        if space_context_key not in state_dict:
            self._include_space_context_state.fill_(1 if force_overlap else 0)
            state_dict[space_context_key] = self._include_space_context_state.detach().clone()
        if space_embedding_key not in state_dict:
            state_dict[space_embedding_key] = self.space_embedding.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.clear_cache()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_text(self, text):
        if self.strip_text_edges:
            return text.strip()
        return text

    def _display_surface(self, surface):
        return "".join(self.space_token if char.isspace() else char for char in surface)

    def _surface_for_core(self, text, start, core_length):
        core_end = start + core_length
        surface_end = core_end
        if self.boundary_context_chars > 0 and core_end < len(text):
            candidate_end = min(len(text), core_end + self.boundary_context_chars)
            context = text[core_end:candidate_end]
            if self.include_space_context or not any(char.isspace() for char in context):
                surface_end = candidate_end
        return text[start:surface_end]

    def enumerate_spans(self, text):
        starts = []
        lengths = []
        display_texts = []
        raw_surfaces = []

        for start, char in enumerate(text):
            if char.isspace():
                core_lengths = [1]
            else:
                core_lengths = []
                max_end = min(len(text), start + self.max_span_chars)
                for end in range(start + 1, max_end + 1):
                    core = text[start:end]
                    if any(ch.isspace() for ch in core):
                        break
                    core_lengths.append(end - start)

            for core_length in core_lengths:
                surface = self._surface_for_core(text, start, core_length)
                starts.append(start)
                lengths.append(core_length)
                raw_surfaces.append(surface)
                display_texts.append(self._display_surface(surface))

        return starts, lengths, display_texts, raw_surfaces

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
        if not use_cache or self.cache_size <= 0:
            return None
        cached = self._span_feature_cache.get(text)
        if cached is None:
            return None
        self._span_feature_cache.move_to_end(text)
        (
            starts,
            lengths,
            display_texts,
            raw_surfaces,
            pooled_visible_cpu,
            visible_counts,
            space_counts,
        ) = cached
        pooled_visible = pooled_visible_cpu.to(self.device, non_blocking=True)
        if pooled_visible.dtype != self.projection.weight.dtype:
            pooled_visible = pooled_visible.to(dtype=self.projection.weight.dtype)
        return (
            starts,
            lengths,
            display_texts,
            raw_surfaces,
            pooled_visible,
            visible_counts,
            space_counts,
        )

    def _cache_put(
        self,
        text,
        starts,
        lengths,
        display_texts,
        raw_surfaces,
        pooled_visible,
        visible_counts,
        space_counts,
        use_cache=True,
    ):
        if not use_cache or self.cache_size <= 0:
            return
        pooled_visible_cpu = pooled_visible.detach().to(
            device="cpu", dtype=self._cache_storage_dtype()
        )
        self._span_feature_cache[text] = (
            starts,
            lengths,
            display_texts,
            raw_surfaces,
            pooled_visible_cpu,
            visible_counts,
            space_counts,
        )
        self._span_feature_cache.move_to_end(text)
        while len(self._span_feature_cache) > self.cache_size:
            self._span_feature_cache.popitem(last=False)

    def _surface_backbone_features(self, raw_surfaces, no_grad):
        hidden_size = self.projection.in_features
        pooled_visible = torch.zeros(
            len(raw_surfaces), hidden_size, device=self.device, dtype=self.projection.weight.dtype
        )
        visible_texts = []
        visible_positions = []
        visible_counts = []
        space_counts = []

        for index, surface in enumerate(raw_surfaces):
            visible = "".join(char for char in surface if not char.isspace())
            visible_count = sum(not char.isspace() for char in surface)
            space_count = sum(char.isspace() for char in surface)
            visible_counts.append(visible_count)
            space_counts.append(space_count)
            if visible:
                visible_positions.append(index)
                visible_texts.append(visible)

        if visible_texts:
            encoded = self._encoded_inputs(visible_texts)
            backbone_inputs = {
                key: value for key, value in encoded.items() if key != "special_tokens_mask"
            }
            if no_grad:
                with torch.no_grad():
                    outputs = self.backbone(**backbone_inputs)
            else:
                outputs = self.backbone(**backbone_inputs)
            pooled = self._pool_non_special_tokens(outputs.last_hidden_state, encoded)
            pooled_visible[torch.as_tensor(visible_positions, device=self.device)] = pooled

        return pooled_visible, visible_counts, space_counts

    def _get_frozen_span_features(self, text, use_cache=None):
        text = self._prepare_text(text)
        use_cache = self._should_use_cache(use_cache)
        cached = self._cache_get(text, use_cache=use_cache)
        if cached is not None:
            return cached

        starts, lengths, display_texts, raw_surfaces = self.enumerate_spans(text)
        pooled_visible, visible_counts, space_counts = self._surface_backbone_features(
            raw_surfaces,
            no_grad=True,
        )
        self._cache_put(
            text,
            starts,
            lengths,
            display_texts,
            raw_surfaces,
            pooled_visible,
            visible_counts,
            space_counts,
            use_cache=use_cache,
        )
        return (
            starts,
            lengths,
            display_texts,
            raw_surfaces,
            pooled_visible,
            visible_counts,
            space_counts,
        )

    def _compose_projected(self, pooled_visible, visible_counts, space_counts):
        if pooled_visible.shape[0] == 0:
            return torch.empty(
                0,
                self.projection.out_features,
                device=self.device,
                dtype=self.projection.weight.dtype,
            )
        visible_counts_tensor = pooled_visible.new_tensor(visible_counts).unsqueeze(-1)
        space_counts_tensor = pooled_visible.new_tensor(space_counts).unsqueeze(-1)
        projected_visible = self.projection(pooled_visible)
        projected_visible = torch.where(
            visible_counts_tensor > 0,
            projected_visible,
            torch.zeros_like(projected_visible),
        )
        projected_space = self.space_embedding.unsqueeze(0).expand_as(projected_visible)
        total = (visible_counts_tensor + space_counts_tensor).clamp_min(1.0)
        return (
            projected_visible * visible_counts_tensor
            + projected_space * space_counts_tensor
        ) / total

    def forward(self, text, use_cache=None):
        text = self._prepare_text(text)
        if self.freeze_backbone:
            (
                starts,
                lengths,
                display_texts,
                _raw_surfaces,
                pooled_visible,
                visible_counts,
                space_counts,
            ) = self._get_frozen_span_features(text, use_cache=use_cache)
        else:
            starts, lengths, display_texts, raw_surfaces = self.enumerate_spans(text)
            pooled_visible, visible_counts, space_counts = self._surface_backbone_features(
                raw_surfaces,
                no_grad=False,
            )

        projected = self._compose_projected(
            pooled_visible,
            visible_counts,
            space_counts,
        )
        return SpanEncoding(
            embeddings=self.norm(projected),
            starts=starts,
            lengths=lengths,
            texts=display_texts,
            text_length=len(text),
            max_span_chars=self.max_span_chars,
        )
