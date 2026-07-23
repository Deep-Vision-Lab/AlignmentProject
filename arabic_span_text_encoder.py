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


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


@dataclass
class SpanEncoding:
    # Core embeddings are used by local window losses and Grad-CAM targets.
    embeddings: torch.Tensor
    starts: list[int]
    lengths: list[int]
    texts: list[str]
    text_length: int
    max_span_chars: int
    # Context embeddings are used only by global Span-DTW. They may contain one
    # following character for one-character cores, but never an implicit space.
    context_embeddings: torch.Tensor | None = None
    surface_texts: list[str] | None = None
    raw_texts: list[str] | None = None
    raw_surface_texts: list[str] | None = None
    is_space: list[bool] | None = None


class ArabicSpanTextEncoder(nn.Module):
    """Encode Arabic core spans and optional overlap context separately.

    ``starts`` and ``lengths`` always describe the non-overlapping core consumed
    by Span-DTW. ``embeddings`` and ``texts`` describe that visible core. A
    separate ``context_embeddings`` tensor may include one following character
    for a one-character core so the global aligner can model overlap between
    adjacent image windows without teaching the local CNN that an unseen third
    character or a textual space is physically present in the current window.
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
        boundary_context_max_core_chars=None,
        allow_character_space_surfaces=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_span_chars = int(max_span_chars)
        self.freeze_backbone = bool(freeze_backbone)
        self.device = torch.device(device)
        self.strip_text_edges = bool(strip_text_edges)
        self.cache_size = int(cache_size)
        self.cache_dtype = str(cache_dtype).lower()
        self.space_token = str(space_token)

        if boundary_context_chars is None:
            boundary_context_chars = _env_int("SPAN_BOUNDARY_CONTEXT_CHARS", 1)
        if include_space_context is None:
            include_space_context = _env_flag("SPAN_INCLUDE_SPACE_CONTEXT", False)
        if boundary_context_max_core_chars is None:
            boundary_context_max_core_chars = _env_int(
                "SPAN_BOUNDARY_CONTEXT_MAX_CORE_CHARS", 1
            )
        if allow_character_space_surfaces is None:
            allow_character_space_surfaces = _env_flag(
                "SPAN_ALLOW_CHARACTER_SPACE_SURFACES", False
            )

        self.register_buffer(
            "_boundary_context_chars_state",
            torch.tensor(max(0, int(boundary_context_chars)), dtype=torch.int16),
        )
        self.register_buffer(
            "_include_space_context_state",
            torch.tensor(1 if include_space_context else 0, dtype=torch.uint8),
        )
        self.register_buffer(
            "_boundary_context_max_core_chars_state",
            torch.tensor(
                max(0, int(boundary_context_max_core_chars)), dtype=torch.int16
            ),
        )
        self.register_buffer(
            "_allow_character_space_surfaces_state",
            torch.tensor(
                1 if allow_character_space_surfaces else 0, dtype=torch.uint8
            ),
        )

        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "transformers is required for ArabicSpanTextEncoder. "
                "Install it or use TEXT_ENCODER_TYPE=char."
            )

        cache_dir = (
            os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
            or None
        )
        local_files_only = (
            _env_flag("HF_HUB_OFFLINE", False)
            or _env_flag("TRANSFORMERS_OFFLINE", False)
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

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

        self.projection = nn.Linear(hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        # Retained for checkpoint compatibility and standalone textual spaces.
        self.space_embedding = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.space_embedding, mean=0.0, std=0.02)

        # Cache frozen backbone features only. Projection/norm stay trainable.
        self._span_feature_cache = OrderedDict()
        self.to(self.device)

    @property
    def boundary_context_chars(self):
        return int(self._boundary_context_chars_state.item())

    @property
    def include_space_context(self):
        return bool(int(self._include_space_context_state.item()))

    @property
    def boundary_context_max_core_chars(self):
        return int(self._boundary_context_max_core_chars_state.item())

    @property
    def allow_character_space_surfaces(self):
        return bool(int(self._allow_character_space_surfaces_state.item()))

    def configure_context(
        self,
        boundary_context_chars=None,
        include_space_context=None,
        boundary_context_max_core_chars=None,
        allow_character_space_surfaces=None,
    ):
        """Apply explicit runtime semantics and invalidate cached span features."""
        if boundary_context_chars is not None:
            self._boundary_context_chars_state.fill_(
                max(0, int(boundary_context_chars))
            )
        if include_space_context is not None:
            self._include_space_context_state.fill_(
                1 if bool(include_space_context) else 0
            )
        if boundary_context_max_core_chars is not None:
            self._boundary_context_max_core_chars_state.fill_(
                max(0, int(boundary_context_max_core_chars))
            )
        if allow_character_space_surfaces is not None:
            self._allow_character_space_surfaces_state.fill_(
                1 if bool(allow_character_space_surfaces) else 0
            )
        self.clear_cache()

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
        """Load old checkpoints while keeping new semantics configurable."""
        force_overlap = _env_flag("FORCE_OVERLAP_AWARE_SPANS", False)
        defaults = {
            "_boundary_context_chars_state": torch.tensor(
                1 if force_overlap else self.boundary_context_chars,
                dtype=torch.int16,
                device=self._boundary_context_chars_state.device,
            ),
            "_include_space_context_state": torch.tensor(
                0,
                dtype=torch.uint8,
                device=self._include_space_context_state.device,
            ),
            "_boundary_context_max_core_chars_state": torch.tensor(
                self.boundary_context_max_core_chars,
                dtype=torch.int16,
                device=self._boundary_context_max_core_chars_state.device,
            ),
            "_allow_character_space_surfaces_state": torch.tensor(
                0,
                dtype=torch.uint8,
                device=self._allow_character_space_surfaces_state.device,
            ),
            "space_embedding": self.space_embedding.detach().clone(),
        }
        for name, value in defaults.items():
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = value

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # New launchers can make their explicit environment configuration
        # authoritative even when initializing from an older checkpoint.
        if _env_flag("OVERRIDE_SPAN_CONTEXT_FROM_ENV", False):
            self.configure_context(
                boundary_context_chars=_env_int(
                    "SPAN_BOUNDARY_CONTEXT_CHARS", self.boundary_context_chars
                ),
                include_space_context=_env_flag(
                    "SPAN_INCLUDE_SPACE_CONTEXT", False
                ),
                boundary_context_max_core_chars=_env_int(
                    "SPAN_BOUNDARY_CONTEXT_MAX_CORE_CHARS",
                    self.boundary_context_max_core_chars,
                ),
                allow_character_space_surfaces=_env_flag(
                    "SPAN_ALLOW_CHARACTER_SPACE_SURFACES", False
                ),
            )
        else:
            self.clear_cache()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_text(self, text):
        text = str(text)
        return text.strip() if self.strip_text_edges else text

    def _display_surface(self, surface):
        return "".join(
            self.space_token if character.isspace() else character
            for character in surface
        )

    def _surface_for_core(self, text, start, core_length):
        core_end = start + core_length
        core = text[start:core_end]

        # A textual space is already a complete core. Never turn it into
        # <SPACE>+character context.
        if not core or any(character.isspace() for character in core):
            return core
        if self.boundary_context_chars <= 0:
            return core
        if core_length > self.boundary_context_max_core_chars:
            return core
        if core_end >= len(text):
            return core

        candidate_end = min(len(text), core_end + self.boundary_context_chars)
        context = text[core_end:candidate_end]
        if any(character.isspace() for character in context):
            # Character+space surfaces were the main source of false local
            # labels. They now require a separate explicit opt-in gate.
            if not (
                self.include_space_context
                and self.allow_character_space_surfaces
            ):
                return core
        return text[start:candidate_end]

    def enumerate_spans(self, text):
        starts = []
        lengths = []
        core_texts = []
        surface_texts = []
        raw_cores = []
        raw_surfaces = []
        is_space = []

        for start, character in enumerate(text):
            if character.isspace():
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
                core = text[start : start + core_length]
                surface = self._surface_for_core(text, start, core_length)
                starts.append(start)
                lengths.append(core_length)
                raw_cores.append(core)
                raw_surfaces.append(surface)
                core_texts.append(self._display_surface(core))
                surface_texts.append(self._display_surface(surface))
                is_space.append(all(ch.isspace() for ch in core))

        return (
            starts,
            lengths,
            core_texts,
            surface_texts,
            raw_cores,
            raw_surfaces,
            is_space,
        )

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
            non_special = (
                1 - encoded["special_tokens_mask"]
            ).unsqueeze(-1).float()
            pool_mask = attention_mask * non_special
            empty_rows = pool_mask.sum(dim=1, keepdim=True) == 0
            pool_mask = torch.where(empty_rows, attention_mask, pool_mask)
        else:
            pool_mask = attention_mask
        return (hidden * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp_min(1.0)

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

    def _backbone_features_for_strings(self, strings, no_grad):
        """Encode unique visible strings once and scatter them back."""
        hidden_size = self.projection.in_features
        pooled = torch.zeros(
            len(strings),
            hidden_size,
            device=self.device,
            dtype=self.projection.weight.dtype,
        )
        visible_counts = [
            sum(not ch.isspace() for ch in value) for value in strings
        ]
        space_counts = [sum(ch.isspace() for ch in value) for value in strings]

        visible_values = [
            "".join(ch for ch in value if not ch.isspace()) for value in strings
        ]
        unique_visible = []
        unique_index = {}
        scatter = []
        for visible in visible_values:
            if not visible:
                scatter.append(-1)
                continue
            if visible not in unique_index:
                unique_index[visible] = len(unique_visible)
                unique_visible.append(visible)
            scatter.append(unique_index[visible])

        if unique_visible:
            encoded = self._encoded_inputs(unique_visible)
            backbone_inputs = {
                key: value
                for key, value in encoded.items()
                if key != "special_tokens_mask"
            }
            if no_grad:
                with torch.no_grad():
                    outputs = self.backbone(**backbone_inputs)
            else:
                outputs = self.backbone(**backbone_inputs)
            unique_pooled = self._pool_non_special_tokens(
                outputs.last_hidden_state, encoded
            )
            for output_index, unique_pos in enumerate(scatter):
                if unique_pos >= 0:
                    pooled[output_index] = unique_pooled[unique_pos]

        return pooled, visible_counts, space_counts

    def _cache_get(self, key, use_cache=True):
        if not use_cache or self.cache_size <= 0:
            return None
        cached = self._span_feature_cache.get(key)
        if cached is None:
            return None
        self._span_feature_cache.move_to_end(key)
        metadata, tensors = cached
        restored = []
        for tensor in tensors:
            tensor = tensor.to(self.device, non_blocking=True)
            if tensor.dtype != self.projection.weight.dtype:
                tensor = tensor.to(dtype=self.projection.weight.dtype)
            restored.append(tensor)
        return (*metadata, *restored)

    def _cache_put(self, key, metadata, tensors, use_cache=True):
        if not use_cache or self.cache_size <= 0:
            return
        stored = tuple(
            tensor.detach().to(
                device="cpu", dtype=self._cache_storage_dtype()
            )
            for tensor in tensors
        )
        self._span_feature_cache[key] = (metadata, stored)
        self._span_feature_cache.move_to_end(key)
        while len(self._span_feature_cache) > self.cache_size:
            self._span_feature_cache.popitem(last=False)

    def _get_frozen_span_features(self, text, use_cache=None):
        text = self._prepare_text(text)
        use_cache = self._should_use_cache(use_cache)
        cache_key = (
            text,
            self.boundary_context_chars,
            self.boundary_context_max_core_chars,
            self.include_space_context,
            self.allow_character_space_surfaces,
        )
        cached = self._cache_get(cache_key, use_cache=use_cache)
        if cached is not None:
            return cached

        (
            starts,
            lengths,
            core_texts,
            surface_texts,
            raw_cores,
            raw_surfaces,
            is_space,
        ) = self.enumerate_spans(text)
        combined = raw_cores + raw_surfaces
        pooled, visible_counts, space_counts = self._backbone_features_for_strings(
            combined, no_grad=True
        )
        split = len(raw_cores)
        core_pooled, surface_pooled = pooled[:split], pooled[split:]
        core_visible, surface_visible = visible_counts[:split], visible_counts[split:]
        core_spaces, surface_spaces = space_counts[:split], space_counts[split:]

        metadata = (
            starts,
            lengths,
            core_texts,
            surface_texts,
            raw_cores,
            raw_surfaces,
            is_space,
            core_visible,
            core_spaces,
            surface_visible,
            surface_spaces,
        )
        self._cache_put(
            cache_key,
            metadata,
            (core_pooled, surface_pooled),
            use_cache=use_cache,
        )
        return (*metadata, core_pooled, surface_pooled)

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
                core_texts,
                surface_texts,
                raw_cores,
                raw_surfaces,
                is_space,
                core_visible,
                core_spaces,
                surface_visible,
                surface_spaces,
                core_pooled,
                surface_pooled,
            ) = self._get_frozen_span_features(text, use_cache=use_cache)
        else:
            (
                starts,
                lengths,
                core_texts,
                surface_texts,
                raw_cores,
                raw_surfaces,
                is_space,
            ) = self.enumerate_spans(text)
            combined = raw_cores + raw_surfaces
            pooled, visible_counts, space_counts = self._backbone_features_for_strings(
                combined, no_grad=False
            )
            split = len(raw_cores)
            core_pooled, surface_pooled = pooled[:split], pooled[split:]
            core_visible, surface_visible = visible_counts[:split], visible_counts[split:]
            core_spaces, surface_spaces = space_counts[:split], space_counts[split:]

        core_projected = self._compose_projected(
            core_pooled, core_visible, core_spaces
        )
        context_projected = self._compose_projected(
            surface_pooled, surface_visible, surface_spaces
        )
        core_embeddings = self.norm(core_projected)
        context_embeddings = self.norm(context_projected)

        return SpanEncoding(
            embeddings=core_embeddings,
            context_embeddings=context_embeddings,
            starts=starts,
            lengths=lengths,
            texts=core_texts,
            surface_texts=surface_texts,
            raw_texts=raw_cores,
            raw_surface_texts=raw_surfaces,
            is_space=is_space,
            text_length=len(text),
            max_span_chars=self.max_span_chars,
        )
