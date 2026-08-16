"""Online partial-overlap pair synthesis from leakage-safe real training lines.

The generator never guesses character-to-pixel boundaries inside a line. Instead,
it stitches complete real line crops whose transcripts are known exactly.

A synthetic pair contains one to three *shared islands*. Each shared island comes
from an existing high/medium A/B real pair, so side A and side B preserve genuine
cross-writer handwriting. Independent no-shared A/B lines are inserted as
unmatched distractor islands before, between, or after the shared islands.

Example logical layout (Arabic RTL is handled by ``stitch_rtl_lines``):

    synthetic A: distractor_A0 | shared_A0 | distractor_A1 | shared_A1
    synthetic B: distractor_B0 | shared_B0 | distractor_B1 | shared_B1

Only indices supplied by the caller are used. The training loader is responsible
for giving this dataset train-only positive and no-shared indices after page/pair
leakage exclusion; validation/test are never synthesized.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset

from RealDataAugmentation import _strip_text_boundaries, _with_text_boundaries, stitch_rtl_lines


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _read_raw_image(dataset, side: dict) -> Image.Image:
    path = dataset._resolve(side["line_image_path"])
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _read_clean_text(dataset, side: dict) -> str:
    return _strip_text_boundaries(dataset._read_text(side[dataset.text_key]))


def _stitch_many(images: list[Image.Image], gap_min: float, gap_max: float) -> Image.Image:
    if not images:
        raise ValueError("Cannot stitch an empty segment list")
    result = images[0]
    for image in images[1:]:
        result = stitch_rtl_lines(
            result,
            image,
            gap_ratio_min=gap_min,
            gap_ratio_max=gap_max,
            vertical_jitter_ratio=0.02,
        )
    return result


class PartialOverlapRealPairDataset(Dataset):
    """Generate train-only partial-overlap A/B pairs online.

    ``positive_dataset`` contains high/medium real A/B pairs. ``distractor_dataset``
    contains leakage-safe no-shared A/B rows. The returned sample uses the normal
    paired dictionary contract and deliberately reports ``medium_match`` so the
    existing text-equality span contrastive and sequence-positive objectives apply.

    Global order consistency should be disabled for this dataset because unmatched
    distractor regions intentionally have no cross-line partner.
    """

    def __init__(
        self,
        positive_dataset,
        positive_indices: Sequence[int],
        distractor_dataset,
        distractor_indices: Sequence[int],
        transform,
        target_length: int,
    ):
        self.positive_dataset = positive_dataset
        self.positive_indices = [int(i) for i in positive_indices]
        self.distractor_dataset = distractor_dataset
        self.distractor_indices = [int(i) for i in distractor_indices]
        self.transform = transform
        self.target_length = max(1, int(target_length))
        if not self.positive_indices:
            raise ValueError("Partial-overlap synthesis requires positive training rows")
        if not self.distractor_indices:
            raise ValueError("Partial-overlap synthesis requires no-shared distractor rows")

        self.max_shared_islands = max(1, min(3, _env_int("REAL_PARTIAL_OVERLAP_MAX_SHARED_ISLANDS", 3)))
        self.max_text_chars = max(32, _env_int("REAL_PARTIAL_OVERLAP_MAX_TEXT_CHARS", 150))
        self.multi_island_probability = max(
            0.0, min(1.0, _env_float("REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB", 0.65))
        )
        self.three_island_probability = max(
            0.0, min(1.0, _env_float("REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB", 0.15))
        )
        self.edge_distractor_probability = max(
            0.0, min(1.0, _env_float("REAL_PARTIAL_OVERLAP_EDGE_DISTRACTOR_PROB", 0.45))
        )
        self.gap_min = max(0.0, _env_float("REAL_AUG_STITCH_GAP_MIN", 0.05))
        self.gap_max = max(self.gap_min, _env_float("REAL_AUG_STITCH_GAP_MAX", 0.12))
        self.max_attempts = max(4, _env_int("REAL_PARTIAL_OVERLAP_MAX_ATTEMPTS", 24))

        # Short distractors make multi-island composites fit the fixed 1024px canvas
        # without crushing each real handwriting segment too aggressively.
        distractor_lengths = []
        for index in self.distractor_indices:
            sample = self.distractor_dataset.samples[index]
            length = max(
                len(_read_clean_text(self.distractor_dataset, sample["A"])),
                len(_read_clean_text(self.distractor_dataset, sample["B"])),
            )
            distractor_lengths.append((length, index))
        distractor_lengths.sort()
        keep = max(1, int(round(len(distractor_lengths) * 0.40)))
        self.short_distractor_indices = [index for _length, index in distractor_lengths[:keep]]

    def __len__(self):
        return self.target_length

    def _shared_count(self) -> int:
        if self.max_shared_islands <= 1 or random.random() > self.multi_island_probability:
            return 1
        if self.max_shared_islands >= 3 and random.random() < self.three_island_probability:
            return 3
        return 2

    def _choose_shared(self, anchor: int, count: int) -> list[int]:
        chosen = [int(anchor)]
        pool = [index for index in self.positive_indices if int(index) != int(anchor)]
        need = max(0, count - 1)
        if need:
            if len(pool) >= need:
                chosen.extend(random.sample(pool, need))
            else:
                chosen.extend(random.choice(pool or [anchor]) for _ in range(need))
        random.shuffle(chosen)
        return chosen

    def _choose_distractor(self) -> int:
        return int(random.choice(self.short_distractor_indices))

    def _build_once(self, anchor: int, shared_count: int):
        shared_indices = self._choose_shared(anchor, shared_count)
        segments_a: list[tuple[Image.Image, str, str]] = []
        segments_b: list[tuple[Image.Image, str, str]] = []
        shared_texts: list[tuple[str, str]] = []

        def add_distractor():
            index = self._choose_distractor()
            sample = self.distractor_dataset.samples[index]
            segments_a.append((
                _read_raw_image(self.distractor_dataset, sample["A"]),
                _read_clean_text(self.distractor_dataset, sample["A"]),
                "distractor",
            ))
            segments_b.append((
                _read_raw_image(self.distractor_dataset, sample["B"]),
                _read_clean_text(self.distractor_dataset, sample["B"]),
                "distractor",
            ))

        if random.random() < self.edge_distractor_probability:
            add_distractor()

        for shared_position, index in enumerate(shared_indices):
            sample = self.positive_dataset.samples[index]
            text_a = _read_clean_text(self.positive_dataset, sample["A"])
            text_b = _read_clean_text(self.positive_dataset, sample["B"])
            segments_a.append((
                _read_raw_image(self.positive_dataset, sample["A"]), text_a, "shared"
            ))
            segments_b.append((
                _read_raw_image(self.positive_dataset, sample["B"]), text_b, "shared"
            ))
            shared_texts.append((text_a, text_b))
            if shared_position + 1 < len(shared_indices):
                # Mandatory unmatched content separates two shared islands.
                add_distractor()

        if random.random() < self.edge_distractor_probability:
            add_distractor()

        text_a = " ".join(text for _image, text, _kind in segments_a if text).strip()
        text_b = " ".join(text for _image, text, _kind in segments_b if text).strip()
        if not text_a or not text_b:
            return None
        if max(len(text_a), len(text_b)) > self.max_text_chars:
            return None

        image_a = _stitch_many([image for image, _text, _kind in segments_a], self.gap_min, self.gap_max)
        image_b = _stitch_many([image for image, _text, _kind in segments_b], self.gap_min, self.gap_max)
        if self.transform is not None:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        shared_islands = sum(1 for _image, _text, kind in segments_a if kind == "shared")
        distractor_islands = sum(1 for _image, _text, kind in segments_a if kind == "distractor")
        return {
            "text1": _with_text_boundaries(text_a),
            "image1": image_a,
            "text2": _with_text_boundaries(text_b),
            "image2": image_b,
            # Existing pair loss matches equal transcript spans; labeling as medium
            # lets shared spans train while non-shared distractors receive no false
            # positive span match.
            "label_type": "medium_match",
            "pair_id": f"partial_overlap_{anchor}_{random.randrange(1 << 30)}",
            "sample_type": "synthetic_partial_overlap",
            "partial_overlap_shared_islands": shared_islands,
            "partial_overlap_distractor_islands": distractor_islands,
            "partial_overlap_shared_texts": shared_texts,
            "text_score": 0.5,
            "avg_sim": 0.5,
            "coverage_A": 0.5,
            "coverage_B": 0.5,
            "line1_index": -1,
            "line2_index": -1,
        }

    def __getitem__(self, index):
        anchor = self.positive_indices[int(index) % len(self.positive_indices)]
        requested = self._shared_count()
        for attempt in range(self.max_attempts):
            # If longer composites repeatedly fail the character cap, gracefully
            # fall back 3 -> 2 -> 1 while preserving a valid training sample.
            shared_count = max(1, requested - (attempt // max(1, self.max_attempts // 3)))
            result = self._build_once(anchor, shared_count)
            if result is not None:
                return result
        # Last-resort one-island sample with no edge distractor probability.
        old_probability = self.edge_distractor_probability
        try:
            self.edge_distractor_probability = 0.0
            result = self._build_once(anchor, 1)
            if result is None:
                raise RuntimeError("Could not construct a feasible partial-overlap sample")
            return result
        finally:
            self.edge_distractor_probability = old_probability
