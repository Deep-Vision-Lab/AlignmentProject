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
    """Keep visible core spans separate from overlap-only and blank context.

    ``<SPACE>`` remains a real transcript position. ``<BLANK>`` is a learned
    visual background prototype appended as a zero-length pseudo span. Span-DTW
    may consume an image window with that prototype without advancing through
    the transcript.
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
        if hasattr(self, "space_embedding"):
            initial_blank = self.space_embedding.detach().clone()
        else:
            initial_blank = self.projection.weight.new_empty(output_dim)
            torch.nn.init.normal_(initial_blank, mean=0.0, std=0.02)
        self.blank_embedding = torch.nn.Parameter(initial_blank)

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

    def _encode_surfaces(self, surfaces, no_grad):
        pooled, visible_counts, space_counts = self._surface_backbone_features(
            surfaces, no_grad=no_grad
        )
        return self.norm(
            self._compose_projected(pooled, visible_counts, space_counts)
        )

    def forward(self, text, use_cache=None):
        del use_cache
        text = self._prepare_text(text)
        (
            starts,
            lengths,
            core_texts,
            surface_texts,
            raw_cores,
            raw_surfaces,
            is_space,
        ) = self.enumerate_spans(text)
        no_grad = bool(self.freeze_backbone)
        core_embeddings = self._encode_surfaces(raw_cores, no_grad=no_grad)
        context_embeddings = self._encode_surfaces(raw_surfaces, no_grad=no_grad)

        blank_vector = self.norm(self.blank_embedding.view(1, -1))
        blank_index = len(core_texts)
        core_embeddings = torch.cat([core_embeddings, blank_vector], dim=0)
        context_embeddings = torch.cat([context_embeddings, blank_vector], dim=0)
        starts.append(-1)
        lengths.append(0)
        core_texts.append("<BLANK>")
        surface_texts.append("<BLANK>")
        raw_cores.append("<BLANK>")
        raw_surfaces.append("<BLANK>")
        is_space.append(False)
        is_blank = [False] * blank_index + [True]

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
            is_blank=is_blank,
            blank_index=blank_index,
            text_length=len(text),
            max_span_chars=self.max_visible_core_chars,
        )