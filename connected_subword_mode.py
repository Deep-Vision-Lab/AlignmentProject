"""Arabic connected-subword tokenization for Span-DTW alignment.

This experimental mode replaces character-span search with a deterministic
sequence of Arabic joining runs. Each connected run is one semantic text state
that may consume several consecutive image windows. A learned
``<SUBWORD_BOUNDARY>`` state is inserted between disconnected runs inside a
word, while the existing learned ``<SPACE>`` state separates complete words.

The ordinary free ``<BLANK>`` transition is retained for background and unused
image windows. It is not the same as the explicit subword-boundary state.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import unicodedata

import torch

MODE_NAME = "connected_subword"
BOUNDARY_TOKEN = "<SUBWORD_BOUNDARY>"
SPACE_TOKEN = "<SPACE>"

# Unicode Joining_Type=Right_Joining letters commonly found in Arabic and
# Arabic-script manuscript datasets. All other Arabic letters default to
# Dual_Joining. HAMZA is explicitly non-joining.
_RIGHT_JOINING = frozenset(
    "اآأؤإدذرزوةٱٲٳٵٶٷڈډڊڋڌڍڎڏڐڑڒړڔڕږژڙۀۃۄۅۆۇۈۉۊۋۍۏےۓە"
)
_NON_JOINING = frozenset({"ء", "\u200c"})
_JOIN_CAUSING = frozenset({"\u0640", "\u200d"})


@dataclass(frozen=True)
class ConnectedUnit:
    text: str
    kind: str  # subword | boundary | space
    char_length: int


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def connected_mode_enabled() -> bool:
    return os.environ.get("SPAN_TOKENIZATION_MODE", "character_span").strip().lower() in {
        MODE_NAME,
        "connected-subword",
        "joining_run",
        "joining-run",
    }


def _base_count(text: str) -> int:
    return max(1, sum(unicodedata.combining(char) == 0 for char in text))


def _grapheme_clusters(word: str) -> list[str]:
    clusters: list[str] = []
    for char in unicodedata.normalize("NFKC", word):
        if unicodedata.combining(char) and clusters:
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _joining_type(cluster: str) -> str:
    if not cluster:
        return "U"
    char = cluster[0]
    dual_overrides = set(os.environ.get("SPAN_CONNECTED_DUAL_OVERRIDES", ""))
    right_overrides = set(os.environ.get("SPAN_CONNECTED_RIGHT_OVERRIDES", ""))
    non_joining_overrides = set(
        os.environ.get("SPAN_CONNECTED_NON_JOINING_OVERRIDES", "")
    )
    if char in non_joining_overrides:
        return "U"
    if char in dual_overrides:
        return "D"
    if char in right_overrides:
        return "R"
    if char in _JOIN_CAUSING:
        return "C"
    if char in _NON_JOINING or char.isspace():
        return "U"
    if char in _RIGHT_JOINING:
        return "R"
    category = unicodedata.category(char)
    if category.startswith("L") and "ARABIC" in unicodedata.name(char, ""):
        return "D"
    return "U"


def _connects(previous_cluster: str, next_cluster: str) -> bool:
    """Return whether adjacent logical-order clusters share one ink component."""
    previous_type = _joining_type(previous_cluster)
    next_type = _joining_type(next_cluster)
    # In logical Arabic order, the previous character must join toward its left
    # (D/C), while the next character must accept a join from its right (D/R/C).
    return previous_type in {"D", "C"} and next_type in {"D", "R", "C"}


def split_connected_word(word: str) -> list[str]:
    clusters = _grapheme_clusters(word)
    if not clusters:
        return []
    runs = [clusters[0]]
    previous_cluster = clusters[0]
    for cluster in clusters[1:]:
        if _connects(previous_cluster, cluster):
            runs[-1] += cluster
        else:
            runs.append(cluster)
        previous_cluster = cluster
    return runs


def connected_units(text: str) -> list[ConnectedUnit]:
    words = str(text).strip().split()
    units: list[ConnectedUnit] = []
    for word_index, word in enumerate(words):
        if word_index:
            units.append(ConnectedUnit(" ", "space", 1))
        runs = split_connected_word(word)
        for run_index, run in enumerate(runs):
            if run_index:
                units.append(ConnectedUnit("", "boundary", 1))
            units.append(ConnectedUnit(run, "subword", _base_count(run)))
    return units


def render_connected_units(text: str) -> list[str]:
    rendered = []
    for unit in connected_units(text):
        if unit.kind == "boundary":
            rendered.append(BOUNDARY_TOKEN)
        elif unit.kind == "space":
            rendered.append(SPACE_TOKEN)
        else:
            rendered.append(unit.text)
    return rendered


_INSTALLED = False
_TRAINING_PATCHED = False


def install_connected_subword_mode() -> bool:
    """Patch the current span encoder and transition limits for this experiment."""
    global _INSTALLED
    if _INSTALLED or not connected_mode_enabled():
        return _INSTALLED

    import arabic_span_text_encoder as encoder_module
    import span_alignment_loss as loss_module

    encoder_cls = encoder_module.ArabicSpanTextEncoder
    span_encoding_cls = encoder_module.SpanEncoding
    original_init = encoder_cls.__init__
    original_encode_many = encoder_cls.encode_many
    original_limits = loss_module._per_span_window_limits

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        output_dim = int(self.projection.out_features)
        self.subword_boundary_embedding = torch.nn.Parameter(
            torch.empty(
                output_dim,
                device=self.space_embedding.device,
                dtype=self.space_embedding.dtype,
            )
        )
        with torch.no_grad():
            source = getattr(self, "blank_embedding", self.space_embedding)
            self.subword_boundary_embedding.copy_(source.detach())
            self.subword_boundary_embedding.add_(
                torch.randn_like(self.subword_boundary_embedding) * 0.005
            )

    def connected_encode_many(self, texts, use_cache=None):
        if not connected_mode_enabled():
            return original_encode_many(self, texts, use_cache=use_cache)
        del use_cache
        prepared = [self._prepare_text(text) for text in texts]
        sequences = [connected_units(text) for text in prepared]
        visible_surfaces = [
            unit.text
            for sequence in sequences
            for unit in sequence
            if unit.kind == "subword" and unit.text
        ]
        if visible_surfaces:
            self._surface_backbone_features_cached(visible_surfaces)

        blank_vector = self.norm(self.blank_embedding.view(1, -1))
        boundary_vector = self.norm(self.subword_boundary_embedding.view(1, -1))
        encodings = []
        for source_text, units in zip(prepared, sequences):
            raw_cores = [unit.text if unit.kind != "boundary" else "" for unit in units]
            if raw_cores:
                embeddings = self._project_surfaces(raw_cores)
                boundary_mask = torch.as_tensor(
                    [unit.kind == "boundary" for unit in units],
                    dtype=torch.bool,
                    device=embeddings.device,
                )
                if boundary_mask.any():
                    embeddings = torch.where(
                        boundary_mask.unsqueeze(-1),
                        boundary_vector.expand(embeddings.shape[0], -1),
                        embeddings,
                    )
            else:
                embeddings = boundary_vector.new_empty((0, boundary_vector.shape[-1]))

            display_texts = [
                BOUNDARY_TOKEN
                if unit.kind == "boundary"
                else SPACE_TOKEN
                if unit.kind == "space"
                else unit.text
                for unit in units
            ]
            starts = list(range(len(units)))
            lengths = [1] * len(units)
            is_space = [unit.kind == "space" for unit in units]
            is_boundary = [unit.kind == "boundary" for unit in units]
            unit_char_lengths = [unit.char_length for unit in units]

            blank_index = len(units)
            embeddings = torch.cat([embeddings, blank_vector], dim=0)
            starts.append(-1)
            lengths.append(0)
            display_texts.append("<BLANK>")
            raw_cores.append("<BLANK>")
            is_space.append(False)
            is_boundary.append(False)
            unit_char_lengths.append(0)
            is_blank = [False] * blank_index + [True]

            encoding = span_encoding_cls(
                embeddings=embeddings,
                context_embeddings=embeddings,
                starts=starts,
                lengths=lengths,
                texts=display_texts,
                surface_texts=list(display_texts),
                raw_texts=raw_cores,
                raw_surface_texts=list(raw_cores),
                is_space=is_space,
                is_blank=is_blank,
                blank_index=blank_index,
                text_length=len(units),
                max_span_chars=1,
            )
            encoding.is_boundary = is_boundary
            encoding.unit_char_lengths = unit_char_lengths
            encoding.unit_kinds = [unit.kind for unit in units] + ["blank"]
            encoding.tokenization_mode = MODE_NAME
            encoding.source_text = source_text
            encodings.append(encoding)
        return encodings

    def connected_limits(span_encoding, global_max, device):
        if getattr(span_encoding, "tokenization_mode", "") != MODE_NAME:
            return original_limits(span_encoding, global_max, device)
        char_lengths = torch.as_tensor(
            getattr(span_encoding, "unit_char_lengths", []),
            dtype=torch.long,
            device=device,
        )
        if char_lengths.numel() == 0:
            return None
        windows_per_char = max(
            1, int(os.environ.get("SPAN_CONNECTED_WINDOWS_PER_CHAR", "3"))
        )
        extra = max(0, int(os.environ.get("SPAN_CONNECTED_EXTRA_WINDOWS", "1")))
        limits = (char_lengths * windows_per_char + extra).clamp(
            min=1, max=max(1, int(global_max))
        )
        spaces = torch.as_tensor(
            getattr(span_encoding, "is_space", [False] * len(char_lengths)),
            dtype=torch.bool,
            device=device,
        )
        boundaries = torch.as_tensor(
            getattr(span_encoding, "is_boundary", [False] * len(char_lengths)),
            dtype=torch.bool,
            device=device,
        )
        blanks = torch.as_tensor(
            getattr(span_encoding, "is_blank", [False] * len(char_lengths)),
            dtype=torch.bool,
            device=device,
        )
        space_cap = max(
            1,
            min(int(global_max), int(os.environ.get("SPAN_SPACE_MAX_WINDOWS", "3"))),
        )
        boundary_cap = max(
            1,
            min(
                int(global_max),
                int(os.environ.get("SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS", "2")),
            ),
        )
        limits = torch.where(spaces, torch.full_like(limits, space_cap), limits)
        limits = torch.where(
            boundaries, torch.full_like(limits, boundary_cap), limits
        )
        limits = torch.where(blanks, torch.ones_like(limits), limits)
        return limits

    encoder_cls.__init__ = patched_init
    encoder_cls.encode_many = connected_encode_many
    loss_module._per_span_window_limits = connected_limits
    _INSTALLED = True
    return True


def install_connected_subword_training(train_module) -> bool:
    """Remove structural boundary/space states from semantic pair matching."""
    global _TRAINING_PATCHED
    if _TRAINING_PATCHED or not connected_mode_enabled():
        return _TRAINING_PATCHED

    special = {BOUNDARY_TOKEN, SPACE_TOKEN, "<BLANK>"}
    original_extract = train_module.extract_aligned_span_regions

    def filtered_extract(*args, **kwargs):
        return [
            region
            for region in original_extract(*args, **kwargs)
            if str(region.get("span_text", "")) not in special
        ]

    train_module.extract_aligned_span_regions = filtered_extract

    # The optimized trainer has its own cached region extractor.
    try:
        import training_optimizations

        original_cached = training_optimizations.regions_from_alignment

        def filtered_cached(*args, **kwargs):
            return [
                region
                for region in original_cached(*args, **kwargs)
                if str(region.get("span_text", "")) not in special
            ]

        training_optimizations.regions_from_alignment = filtered_cached
    except (ImportError, AttributeError):
        pass

    _TRAINING_PATCHED = True
    return True


_EVALUATION_PATCHED = False


def install_connected_subword_evaluation(utils_module) -> bool:
    """Make word-level evaluation understand unit-index rather than char-index paths."""
    global _EVALUATION_PATCHED
    if _EVALUATION_PATCHED or not connected_mode_enabled():
        return _EVALUATION_PATCHED

    import re
    import torch.nn.functional as F

    original_extract = utils_module.extract_word_regions

    def connected_extract_word_regions(models, text, image_features, feature="local"):
        prepared, _encoding_unused, path = utils_module.align_text_to_windows(
            models, text, image_features, True
        )
        encoding = models.text_model(prepared)
        if getattr(encoding, "tokenization_mode", "") != MODE_NAME:
            return original_extract(models, text, image_features, feature=feature)

        word_matches = list(re.finditer(r"\S+", prepared))
        unit_word_indices: list[int | None] = []
        word_index = 0
        for kind in getattr(encoding, "unit_kinds", [])[:-1]:
            if kind == "space":
                unit_word_indices.append(None)
                word_index += 1
            elif kind == "subword":
                unit_word_indices.append(word_index)
            else:
                unit_word_indices.append(None)

        visual = image_features.select(feature)
        regions = []
        for index, match in enumerate(word_matches):
            overlapping = []
            for step in path:
                if step.get("is_blank", False):
                    continue
                span_index = int(step["span_idx"])
                if not 0 <= span_index < len(unit_word_indices):
                    continue
                if unit_word_indices[span_index] == index:
                    overlapping.append(step)
            if not overlapping:
                continue
            w0 = min(int(step["window_start"]) for step in overlapping)
            w1 = max(int(step["window_end"]) for step in overlapping)
            w0 = max(0, min(w0, int(visual.shape[0]) - 1))
            w1 = max(w0 + 1, min(w1, int(visual.shape[0])))
            ink = image_features.ink[w0:w1].clamp_min(0.0)
            if ink.numel() and float(ink.sum()) > 1e-8:
                weights = ink / ink.sum()
                pooled = (visual[w0:w1] * weights.unsqueeze(-1)).sum(dim=0)
                mean_ink = float(ink.mean().item())
            else:
                pooled = visual[w0:w1].mean(dim=0)
                mean_ink = 0.0
            regions.append(
                utils_module.WordRegion(
                    index=index,
                    text=match.group(0),
                    char_start=match.start(),
                    char_end=match.end(),
                    window_start=w0,
                    window_end=w1,
                    embedding=F.normalize(pooled.float(), p=2, dim=-1),
                    ink=mean_ink,
                )
            )
        return regions, path

    utils_module.extract_word_regions = connected_extract_word_regions
    _EVALUATION_PATCHED = True
    return True
