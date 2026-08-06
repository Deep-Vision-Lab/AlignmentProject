"""Arabic connected-subword tokenization for Span-DTW alignment.

Connected Arabic runs, explicit intra-word boundaries, and spaces form the
source coordinate system. The encoder exposes variable spans of one or more
connected units, so a single image window may represent multiple neighboring
units when the visual sequence is shorter than the structural token sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import unicodedata

import torch

MODE_NAME = "connected_subword"
BOUNDARY_TOKEN = "<SUBWORD_BOUNDARY>"
SPACE_TOKEN = "<SPACE>"

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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def connected_mode_enabled() -> bool:
    return os.environ.get("SPAN_TOKENIZATION_MODE", "character_span").strip().lower() in {
        MODE_NAME,
        "connected-subword",
        "joining_run",
        "joining-run",
    }


def connected_max_units_per_span() -> int:
    fallback = _env_int("MAX_TEXT_SPAN_CHARS", 3)
    return max(1, _env_int("SPAN_CONNECTED_MAX_UNITS_PER_SPAN", fallback))


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
    non_joining_overrides = set(os.environ.get("SPAN_CONNECTED_NON_JOINING_OVERRIDES", ""))
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
    previous_type = _joining_type(previous_cluster)
    next_type = _joining_type(next_cluster)
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


def connected_span_slices(units: list[ConnectedUnit], max_units: int | None = None) -> list[tuple[int, int]]:
    cap = connected_max_units_per_span() if max_units is None else max(1, int(max_units))
    result: list[tuple[int, int]] = []
    for start, unit in enumerate(units):
        if unit.kind == "space":
            result.append((start, 1))
            continue
        for length in range(1, cap + 1):
            end = start + length
            if end > len(units):
                break
            candidate = units[start:end]
            if any(item.kind == "space" for item in candidate):
                break
            result.append((start, length))
    return result


def minimum_connected_spans(text: str, max_units: int | None = None) -> int:
    units = connected_units(text)
    if not units:
        return 0
    slices = connected_span_slices(units, max_units=max_units)
    by_start: dict[int, list[int]] = {}
    for start, length in slices:
        by_start.setdefault(start, []).append(length)
    inf = len(units) + 1
    best = [inf] * (len(units) + 1)
    best[0] = 0
    for start in range(len(units)):
        if best[start] >= inf:
            continue
        for length in by_start.get(start, []):
            best[start + length] = min(best[start + length], best[start] + 1)
    return best[-1]


_INSTALLED = False
_TRAINING_PATCHED = False


def install_connected_subword_mode() -> bool:
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
        self.subword_boundary_embedding = torch.nn.Parameter(torch.empty(output_dim, device=self.space_embedding.device, dtype=self.space_embedding.dtype))
        with torch.no_grad():
            source = getattr(self, "blank_embedding", self.space_embedding)
            self.subword_boundary_embedding.copy_(source.detach())
            self.subword_boundary_embedding.add_(torch.randn_like(self.subword_boundary_embedding) * 0.005)

    def connected_encode_many(self, texts, use_cache=None):
        if not connected_mode_enabled():
            return original_encode_many(self, texts, use_cache=use_cache)
        del use_cache
        prepared = [self._prepare_text(text) for text in texts]
        sequences = [connected_units(text) for text in prepared]
        max_units = connected_max_units_per_span()
        span_metadata = []
        all_raw_surfaces: list[str] = []
        for units in sequences:
            source_word_indices: list[int | None] = []
            word_index = 0
            for unit in units:
                if unit.kind == "space":
                    source_word_indices.append(None)
                    word_index += 1
                else:
                    source_word_indices.append(word_index)
            candidates = []
            for start, length in connected_span_slices(units, max_units=max_units):
                covered = units[start:start + length]
                raw = "".join(unit.text for unit in covered if unit.kind != "boundary")
                display = "".join(BOUNDARY_TOKEN if unit.kind == "boundary" else SPACE_TOKEN if unit.kind == "space" else unit.text for unit in covered)
                kinds = [unit.kind for unit in covered]
                semantic_words = {source_word_indices[index] for index in range(start, start + length) if source_word_indices[index] is not None}
                candidates.append({
                    "start": start,
                    "length": length,
                    "raw": raw,
                    "display": display,
                    "kind": kinds[0] if length == 1 else "compound",
                    "is_space": all(kind == "space" for kind in kinds),
                    "is_boundary": all(kind == "boundary" for kind in kinds),
                    "boundary_count": sum(kind == "boundary" for kind in kinds),
                    "char_length": max(1, sum(unit.char_length for unit in covered if unit.kind == "subword")),
                    "word_index": next(iter(semantic_words)) if len(semantic_words) == 1 else None,
                })
                all_raw_surfaces.append(raw)
            span_metadata.append((units, candidates, source_word_indices))

        if all_raw_surfaces:
            self._surface_backbone_features_cached(all_raw_surfaces)
        blank_vector = self.norm(self.blank_embedding.view(1, -1))
        boundary_vector = self.norm(self.subword_boundary_embedding.view(1, -1))
        boundary_blend = float(os.environ.get("SPAN_SUBWORD_BOUNDARY_EMBEDDING_WEIGHT", "0.25"))
        encodings = []
        for source_text, (units, candidates, source_word_indices) in zip(prepared, span_metadata):
            raw_cores = [candidate["raw"] for candidate in candidates]
            if raw_cores:
                embeddings = self._project_surfaces(raw_cores)
                boundary_counts = torch.as_tensor([candidate["boundary_count"] for candidate in candidates], dtype=embeddings.dtype, device=embeddings.device).unsqueeze(-1)
                if boundary_blend and boundary_counts.any():
                    embeddings = self.norm(embeddings + boundary_counts * boundary_blend * boundary_vector.expand(embeddings.shape[0], -1))
                pure_boundaries = torch.as_tensor([candidate["is_boundary"] for candidate in candidates], dtype=torch.bool, device=embeddings.device)
                if pure_boundaries.any():
                    embeddings = torch.where(pure_boundaries.unsqueeze(-1), boundary_vector.expand(embeddings.shape[0], -1), embeddings)
            else:
                embeddings = boundary_vector.new_empty((0, boundary_vector.shape[-1]))

            starts = [candidate["start"] for candidate in candidates]
            lengths = [candidate["length"] for candidate in candidates]
            display_texts = [candidate["display"] for candidate in candidates]
            is_space = [candidate["is_space"] for candidate in candidates]
            is_boundary = [candidate["is_boundary"] for candidate in candidates]
            unit_char_lengths = [candidate["char_length"] for candidate in candidates]
            unit_kinds = [candidate["kind"] for candidate in candidates]
            span_word_indices = [candidate["word_index"] for candidate in candidates]
            blank_index = len(candidates)
            embeddings = torch.cat([embeddings, blank_vector], dim=0)
            starts.append(-1); lengths.append(0); display_texts.append("<BLANK>"); raw_cores.append("<BLANK>")
            is_space.append(False); is_boundary.append(False); unit_char_lengths.append(0); unit_kinds.append("blank"); span_word_indices.append(None)
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
                max_span_chars=max_units,
            )
            encoding.is_boundary = is_boundary
            encoding.unit_char_lengths = unit_char_lengths
            encoding.unit_kinds = unit_kinds
            encoding.span_word_indices = span_word_indices
            encoding.source_unit_kinds = [unit.kind for unit in units]
            encoding.source_unit_word_indices = source_word_indices
            encoding.tokenization_mode = MODE_NAME
            encoding.source_text = source_text
            encodings.append(encoding)
        return encodings

    def connected_limits(span_encoding, global_max, device):
        if getattr(span_encoding, "tokenization_mode", "") != MODE_NAME:
            return original_limits(span_encoding, global_max, device)
        char_lengths = torch.as_tensor(getattr(span_encoding, "unit_char_lengths", []), dtype=torch.long, device=device)
        if char_lengths.numel() == 0:
            return None
        windows_per_char = max(1, int(os.environ.get("SPAN_CONNECTED_WINDOWS_PER_CHAR", "3")))
        extra = max(0, int(os.environ.get("SPAN_CONNECTED_EXTRA_WINDOWS", "1")))
        limits = (char_lengths * windows_per_char + extra).clamp(min=1, max=max(1, int(global_max)))
        spaces = torch.as_tensor(getattr(span_encoding, "is_space", [False] * len(char_lengths)), dtype=torch.bool, device=device)
        boundaries = torch.as_tensor(getattr(span_encoding, "is_boundary", [False] * len(char_lengths)), dtype=torch.bool, device=device)
        blanks = torch.as_tensor(getattr(span_encoding, "is_blank", [False] * len(char_lengths)), dtype=torch.bool, device=device)
        space_cap = max(1, min(int(global_max), int(os.environ.get("SPAN_SPACE_MAX_WINDOWS", "3"))))
        boundary_cap = max(1, min(int(global_max), int(os.environ.get("SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS", "2"))))
        limits = torch.where(spaces, torch.full_like(limits, space_cap), limits)
        limits = torch.where(boundaries, torch.full_like(limits, boundary_cap), limits)
        limits = torch.where(blanks, torch.ones_like(limits), limits)
        return limits

    encoder_cls.__init__ = patched_init
    encoder_cls.encode_many = connected_encode_many
    loss_module._per_span_window_limits = connected_limits
    _INSTALLED = True
    return True


def install_connected_subword_training(train_module) -> bool:
    global _TRAINING_PATCHED
    if _TRAINING_PATCHED or not connected_mode_enabled():
        return _TRAINING_PATCHED
    special = {BOUNDARY_TOKEN, SPACE_TOKEN, "<BLANK>"}
    original_extract = train_module.extract_aligned_span_regions
    def filtered_extract(*args, **kwargs):
        return [region for region in original_extract(*args, **kwargs) if str(region.get("span_text", "")) not in special]
    train_module.extract_aligned_span_regions = filtered_extract
    try:
        import training_optimizations
        original_cached = training_optimizations.regions_from_alignment
        def filtered_cached(*args, **kwargs):
            return [region for region in original_cached(*args, **kwargs) if str(region.get("span_text", "")) not in special]
        training_optimizations.regions_from_alignment = filtered_cached
    except (ImportError, AttributeError):
        pass
    _TRAINING_PATCHED = True
    return True


_EVALUATION_PATCHED = False


def install_connected_subword_evaluation(utils_module) -> bool:
    global _EVALUATION_PATCHED
    if _EVALUATION_PATCHED or not connected_mode_enabled():
        return _EVALUATION_PATCHED
    import re
    import torch.nn.functional as F
    original_extract = utils_module.extract_word_regions
    def connected_extract_word_regions(models, text, image_features, feature="local"):
        prepared, _encoding_unused, path = utils_module.align_text_to_windows(models, text, image_features, True)
        encoding = models.text_model(prepared)
        if getattr(encoding, "tokenization_mode", "") != MODE_NAME:
            return original_extract(models, text, image_features, feature=feature)
        word_matches = list(re.finditer(r"\S+", prepared))
        span_word_indices = getattr(encoding, "span_word_indices", None)
        if span_word_indices is None:
            span_word_indices = []
            word_index = 0
            for kind in getattr(encoding, "unit_kinds", [])[:-1]:
                if kind == "space":
                    span_word_indices.append(None); word_index += 1
                elif kind == "subword":
                    span_word_indices.append(word_index)
                else:
                    span_word_indices.append(None)
        visual = image_features.select(feature)
        regions = []
        for index, match in enumerate(word_matches):
            overlapping = []
            for step in path:
                if step.get("is_blank", False):
                    continue
                span_index = int(step["span_idx"])
                if 0 <= span_index < len(span_word_indices) and span_word_indices[span_index] == index:
                    overlapping.append(step)
            if not overlapping:
                continue
            w0 = min(int(step["window_start"]) for step in overlapping)
            w1 = max(int(step["window_end"]) for step in overlapping)
            w0 = max(0, min(w0, int(visual.shape[0]) - 1)); w1 = max(w0 + 1, min(w1, int(visual.shape[0])))
            ink = image_features.ink[w0:w1].clamp_min(0.0)
            if ink.numel() and float(ink.sum()) > 1e-8:
                weights = ink / ink.sum(); pooled = (visual[w0:w1] * weights.unsqueeze(-1)).sum(dim=0); mean_ink = float(ink.mean().item())
            else:
                pooled = visual[w0:w1].mean(dim=0); mean_ink = 0.0
            regions.append(utils_module.WordRegion(index=index, text=match.group(0), char_start=match.start(), char_end=match.end(), window_start=w0, window_end=w1, embedding=F.normalize(pooled.float(), p=2, dim=-1), ink=mean_ink))
        return regions, path
    utils_module.extract_word_regions = connected_extract_word_regions
    _EVALUATION_PATCHED = True
    return True
