from collections import OrderedDict
from dataclasses import dataclass
import os

import torch

import arabic_span_text_encoder_legacy as _legacy


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
    embeddings: torch.Tensor
    starts: list[int]
    lengths: list[int]
    texts: list[str]
    text_length: int
    max_span_chars: int
    context_embeddings: torch.Tensor | None = None
    surface_texts: list[str] | None = None
    raw_texts: list[str] | None = None
    raw_surface_texts: list[str] | None = None
    is_space: list[bool] | None = None
    is_blank: list[bool] | None = None
    blank_index: int | None = None


class ArabicSpanTextEncoder(_legacy.ArabicSpanTextEncoder):
    """Arabic span encoder with persistent frozen-backbone surface caching.

    The AraBERT backbone is frozen, so its pooled output for a visible surface is
    immutable.  Cache those pooled features by unique visible surface and apply
    the trainable projection, ``<SPACE>`` and ``<BLANK>`` embeddings after cache
    retrieval.  This keeps gradients correct while removing repeated transformer
    calls across cores, context surfaces, positive texts and negative texts.
    """

    def __init__(
        self,
        *args,
        boundary_context_max_core_chars=None,
        allow_character_space_surfaces=None,
        **kwargs,
    ):
        if kwargs.get("include_space_context") is None:
            kwargs["include_space_context"] = _env_flag(
                "SPAN_INCLUDE_SPACE_CONTEXT", False
            )
        super().__init__(*args, **kwargs)
        if boundary_context_max_core_chars is None:
            boundary_context_max_core_chars = _env_int(
                "SPAN_BOUNDARY_CONTEXT_MAX_CORE_CHARS", 1
            )
        if allow_character_space_surfaces is None:
            allow_character_space_surfaces = _env_flag(
                "SPAN_ALLOW_CHARACTER_SPACE_SURFACES", False
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

        output_dim = int(self.projection.out_features)
        self.blank_embedding = torch.nn.Parameter(
            torch.empty(
                output_dim,
                device=self.space_embedding.device,
                dtype=self.space_embedding.dtype,
            )
        )
        with torch.no_grad():
            self.blank_embedding.copy_(self.space_embedding.detach())

        # The legacy cache is transcript-level and disabled while training.  The
        # optimized cache is surface-level and remains valid in training because
        # it stores only frozen backbone outputs on CPU.
        self._surface_feature_cache = OrderedDict()
        self._surface_cache_hits = 0
        self._surface_cache_misses = 0

    @property
    def boundary_context_max_core_chars(self):
        return int(self._boundary_context_max_core_chars_state.item())

    @property
    def allow_character_space_surfaces(self):
        return bool(int(self._allow_character_space_surfaces_state.item()))

    @property
    def max_visible_core_chars(self):
        cap = _env_int("SPAN_MAX_CORE_CHARS_CAP", 2)
        if cap <= 0:
            return int(self.max_span_chars)
        return min(int(self.max_span_chars), cap)

    def configure_context(
        self,
        boundary_context_chars=None,
        include_space_context=None,
        boundary_context_max_core_chars=None,
        allow_character_space_surfaces=None,
    ):
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
        self.clear_cache(force=True)

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
        defaults = {
            prefix + "_boundary_context_max_core_chars_state": (
                self._boundary_context_max_core_chars_state.detach().clone()
            ),
            prefix + "_allow_character_space_surfaces_state": (
                self._allow_character_space_surfaces_state.detach().clone()
            ),
        }
        for key, value in defaults.items():
            if key not in state_dict:
                state_dict[key] = value

        blank_key = prefix + "blank_embedding"
        if blank_key not in state_dict:
            space_key = prefix + "space_embedding"
            source = state_dict.get(space_key, self.blank_embedding.detach())
            state_dict[blank_key] = source.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
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

    def _surface_for_core(self, text, start, core_length):
        core_end = start + core_length
        core = text[start:core_end]
        if not core or any(ch.isspace() for ch in core):
            return core
        if self.boundary_context_chars <= 0:
            return core
        if core_length > self.boundary_context_max_core_chars:
            return core
        if core_end >= len(text):
            return core
        candidate_end = min(len(text), core_end + self.boundary_context_chars)
        context = text[core_end:candidate_end]
        if any(ch.isspace() for ch in context) and not (
            self.include_space_context and self.allow_character_space_surfaces
        ):
            return core
        return text[start:candidate_end]

    def enumerate_spans(self, text):
        starts, lengths = [], []
        core_texts, surface_texts = [], []
        raw_cores, raw_surfaces, is_space = [], [], []
        max_core_chars = self.max_visible_core_chars
        for start, character in enumerate(text):
            if character.isspace():
                core_lengths = [1]
            else:
                core_lengths = []
                for end in range(
                    start + 1,
                    min(len(text), start + max_core_chars) + 1,
                ):
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

    def _cache_storage_dtype(self):
        if self.cache_dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if self.cache_dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float32

    def clear_cache(self, force=False):
        # Epoch-level calls should not erase immutable AraBERT features.  Force a
        # clear when span semantics change or when explicitly requested.
        if force or _env_flag("CLEAR_FROZEN_SPAN_CACHE", False):
            self._surface_feature_cache.clear()
            self._surface_cache_hits = 0
            self._surface_cache_misses = 0
            try:
                super().clear_cache()
            except AttributeError:
                pass

    def cache_size_current(self):
        return len(self._surface_feature_cache)

    def cache_stats(self):
        total = self._surface_cache_hits + self._surface_cache_misses
        return {
            "surface_cache_size": float(len(self._surface_feature_cache)),
            "surface_cache_hits": float(self._surface_cache_hits),
            "surface_cache_misses": float(self._surface_cache_misses),
            "surface_cache_hit_rate": (
                float(self._surface_cache_hits) / total if total else 0.0
            ),
        }

    @staticmethod
    def _visible_surface(surface):
        return "".join(character for character in surface if not character.isspace())

    def _cache_get_surface(self, visible):
        if self.cache_size <= 0:
            return None
        value = self._surface_feature_cache.get(visible)
        if value is None:
            self._surface_cache_misses += 1
            return None
        self._surface_cache_hits += 1
        self._surface_feature_cache.move_to_end(visible)
        return value.to(
            device=self.device,
            dtype=self.projection.weight.dtype,
            non_blocking=True,
        )

    def _cache_put_surface(self, visible, pooled):
        if self.cache_size <= 0:
            return
        self._surface_feature_cache[visible] = pooled.detach().to(
            device="cpu", dtype=self._cache_storage_dtype()
        )
        self._surface_feature_cache.move_to_end(visible)
        while len(self._surface_feature_cache) > self.cache_size:
            self._surface_feature_cache.popitem(last=False)

    def _encode_missing_visible(self, visible_texts):
        if not visible_texts:
            return {}
        results = {}
        batch_size = max(1, _env_int("SPAN_BACKBONE_BATCH_SIZE", 512))
        for start in range(0, len(visible_texts), batch_size):
            chunk = visible_texts[start : start + batch_size]
            encoded = self._encoded_inputs(chunk)
            backbone_inputs = {
                key: value
                for key, value in encoded.items()
                if key != "special_tokens_mask"
            }
            with torch.no_grad():
                output = self.backbone(**backbone_inputs)
            pooled = self._pool_non_special_tokens(
                output.last_hidden_state, encoded
            ).to(dtype=self.projection.weight.dtype)
            for text, vector in zip(chunk, pooled):
                results[text] = vector
                self._cache_put_surface(text, vector)
        return results

    def _surface_backbone_features_cached(self, surfaces):
        hidden_size = int(self.projection.in_features)
        visible_counts = [
            sum(not character.isspace() for character in surface)
            for surface in surfaces
        ]
        space_counts = [
            sum(character.isspace() for character in surface)
            for surface in surfaces
        ]
        visible_keys = [self._visible_surface(surface) for surface in surfaces]

        vectors = {}
        missing = []
        seen_missing = set()
        for visible in visible_keys:
            if not visible:
                continue
            cached = self._cache_get_surface(visible)
            if cached is not None:
                vectors[visible] = cached
            elif visible not in seen_missing:
                seen_missing.add(visible)
                missing.append(visible)
        vectors.update(self._encode_missing_visible(missing))

        zero = torch.zeros(
            hidden_size,
            device=self.device,
            dtype=self.projection.weight.dtype,
        )
        if surfaces:
            pooled = torch.stack(
                [vectors.get(visible, zero) if visible else zero for visible in visible_keys],
                dim=0,
            )
        else:
            pooled = zero.new_empty((0, hidden_size))
        return pooled, visible_counts, space_counts

    def _project_surfaces(self, surfaces):
        pooled, visible_counts, space_counts = (
            self._surface_backbone_features_cached(surfaces)
        )
        return self.norm(
            self._compose_projected(pooled, visible_counts, space_counts)
        )

    def encode_many(self, texts, use_cache=None):
        del use_cache
        prepared = [self._prepare_text(text) for text in texts]
        metadata = []
        all_surfaces = []
        for text in prepared:
            enumerated = self.enumerate_spans(text)
            metadata.append(enumerated)
            all_surfaces.extend(enumerated[4])
            all_surfaces.extend(enumerated[5])

        # Populate the unique surface cache once for the whole batch.  Projection
        # remains outside the cache and therefore receives gradients.
        if all_surfaces:
            self._surface_backbone_features_cached(all_surfaces)

        blank_vector = self.norm(self.blank_embedding.view(1, -1))
        encodings = []
        for text, enumerated in zip(prepared, metadata):
            (
                starts,
                lengths,
                core_texts,
                surface_texts,
                raw_cores,
                raw_surfaces,
                is_space,
            ) = enumerated
            core_embeddings = self._project_surfaces(raw_cores)
            context_embeddings = self._project_surfaces(raw_surfaces)

            blank_index = len(core_texts)
            core_embeddings = torch.cat([core_embeddings, blank_vector], dim=0)
            context_embeddings = torch.cat(
                [context_embeddings, blank_vector], dim=0
            )
            starts = list(starts) + [-1]
            lengths = list(lengths) + [0]
            core_texts = list(core_texts) + ["<BLANK>"]
            surface_texts = list(surface_texts) + ["<BLANK>"]
            raw_cores = list(raw_cores) + ["<BLANK>"]
            raw_surfaces = list(raw_surfaces) + ["<BLANK>"]
            is_space = list(is_space) + [False]
            is_blank = [False] * blank_index + [True]

            encodings.append(
                SpanEncoding(
                    embeddings=core_embeddings,
                    context_embeddings=context_embeddings,
                    starts=starts,
                    lengths=lengths,
                    texts=core_texts,
                    surface_texts=surface_texts,
                    raw_texts=raw_cores,
                    raw_surface_texts=raw_surfaces,
                    is_space=is_space,
                    is_blank=is_blank,
                    blank_index=blank_index,
                    text_length=len(text),
                    max_span_chars=self.max_visible_core_chars,
                )
            )
        return encodings

    def forward(self, text, use_cache=None):
        return self.encode_many([text], use_cache=use_cache)[0]
