"""Dense aligned/unaligned augmentation extensions.

Importing this module patches ``SyntheticAugmentedDataLoader`` in place. This
keeps the augmentation entry points shared by all experiment branches while
adding a denser layout and an aligned-plus-unaligned paired mode.
"""
from __future__ import annotations

import os
import random

import SyntheticAugmentedDataLoader as loader


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _dense_compose(
    first,
    second,
    size=(128, 1024),
    rtl=True,
    min_gap_fraction=0.08,
):
    """Place two wide fragments as one centered group with little empty space."""
    height, width = int(size[0]), int(size[1])
    canvas = loader.Image.new(
        "RGB",
        (width, height),
        loader._background(first, second),
    )
    patch_height = random.randint(
        max(1, int(round(height * 0.94))),
        max(1, int(round(height * 0.99))),
    )
    max_patch_width = int(round(width * 0.46))
    first = loader._fit_patch(first, patch_height, max_patch_width)
    second = loader._fit_patch(second, patch_height, max_patch_width)

    margin = max(2, int(round(width * 0.02)))
    gap = max(4, int(round(width * min_gap_fraction)))
    usable = width - (2 * margin) - gap
    combined = first.width + second.width
    if combined > usable:
        factor = usable / max(1, combined)
        first = first.resize(
            (
                max(1, int(round(first.width * factor))),
                max(1, int(round(first.height * factor))),
            ),
            loader._BILINEAR,
        )
        second = second.resize(
            (
                max(1, int(round(second.width * factor))),
                max(1, int(round(second.height * factor))),
            ),
            loader._BILINEAR,
        )

    group_width = first.width + gap + second.width
    free_width = max(0, width - (2 * margin) - group_width)
    start = margin + free_width // 2
    jitter = min(free_width // 2, int(round(width * 0.025)))
    start += random.randint(-jitter, jitter)
    start = max(margin, min(start, width - margin - group_width))

    first_y = random.randint(0, max(0, height - first.height))
    second_y = random.randint(0, max(0, height - second.height))
    if rtl:
        first_position = (start + second.width + gap, first_y)
        second_position = (start, second_y)
    else:
        first_position = (start, first_y)
        second_position = (start + first.width + gap, second_y)

    canvas.paste(first, first_position)
    canvas.paste(second, second_position)
    return canvas


def _different_span(first, min_fraction, max_fraction, min_distance):
    first_center = (first[0] + first[1]) / 2.0
    candidate = loader._random_span(min_fraction, max_fraction)
    for _ in range(60):
        candidate = loader._random_span(min_fraction, max_fraction)
        center = (candidate[0] + candidate[1]) / 2.0
        if abs(center - first_center) >= min_distance:
            break
    return candidate


_original_init = loader.SyntheticPairAugmentor.__init__
_original_apply = loader.SyntheticPairAugmentor.apply


def _patched_init(self):
    _original_init(self)
    self.injection_probability = _env_float(
        "SYNTHETIC_INJECTION_PROB",
        0.35,
    )
    self.two_region_probability = _env_float(
        "SYNTHETIC_TWO_REGION_PROB",
        0.30,
    )
    self.aligned_unaligned_probability = _env_float(
        "SYNTHETIC_ALIGNED_UNALIGNED_PROB",
        0.25,
    )
    self.min_fraction = _env_float(
        "SYNTHETIC_FRAGMENT_MIN_FRACTION",
        0.20,
    )
    self.max_fraction = _env_float(
        "SYNTHETIC_FRAGMENT_MAX_FRACTION",
        0.40,
    )
    self.source_gap = _env_float(
        "SYNTHETIC_SOURCE_GAP_FRACTION",
        0.04,
    )
    self.canvas_gap = _env_float(
        "SYNTHETIC_CANVAS_GAP_FRACTION",
        0.08,
    )
    self.mismatch_span_distance = _env_float(
        "SYNTHETIC_MISMATCH_SPAN_DISTANCE",
        0.12,
    )
    probabilities = (
        self.injection_probability,
        self.two_region_probability,
        self.aligned_unaligned_probability,
    )
    if min(probabilities) < 0 or sum(probabilities) > 1:
        raise ValueError(
            "Injection, two-region, and aligned-unaligned probabilities "
            "must be non-negative and sum to at most 1."
        )
    if not 0 < self.min_fraction <= self.max_fraction <= 1:
        raise ValueError("Expected 0 < fragment min <= fragment max <= 1.")


def _choose_mode(self):
    draw = random.random()
    if draw < self.injection_probability:
        return "cross_injection"
    threshold = self.injection_probability + self.two_region_probability
    if draw < threshold:
        return "two_region"
    threshold += self.aligned_unaligned_probability
    if draw < threshold:
        return "aligned_unaligned"
    return "full_line"


def _aligned_unaligned(self, target_images, target_texts, donor_images, donor_texts):
    image1, image2 = target_images
    text1, text2 = target_texts
    if image2 is None or text2 is None:
        return _original_apply(
            self,
            target_images,
            target_texts,
            donor_images,
            donor_texts,
            "cross_injection",
        )

    shared_span = loader._random_span(self.min_fraction, self.max_fraction)
    donor_span1 = loader._random_span(self.min_fraction, self.max_fraction)
    donor_span2 = _different_span(
        donor_span1,
        self.min_fraction,
        self.max_fraction,
        self.mismatch_span_distance,
    )
    donor_image2 = donor_images[1] or donor_images[0]
    donor_text2 = donor_texts[1] or donor_texts[0]
    shared_first = random.random() < 0.5

    if shared_first:
        output1, out_text1 = self._compose(
            image1,
            donor_images[0],
            text1,
            donor_texts[0],
            shared_span,
            donor_span1,
        )
        output2, out_text2 = self._compose(
            image2,
            donor_image2,
            text2,
            donor_text2,
            shared_span,
            donor_span2,
        )
    else:
        output1, out_text1 = self._compose(
            donor_images[0],
            image1,
            donor_texts[0],
            text1,
            donor_span1,
            shared_span,
        )
        output2, out_text2 = self._compose(
            donor_image2,
            image2,
            donor_text2,
            text2,
            donor_span2,
            shared_span,
        )

    return {
        "mode": "aligned_unaligned",
        "image1": self.appearance(output1),
        "text1": out_text1,
        "image2": self.appearance(output2),
        "text2": out_text2,
    }


def _patched_apply(
    self,
    target_images,
    target_texts,
    donor_images=None,
    donor_texts=None,
    mode=None,
):
    mode = mode or self.choose_mode()
    if mode != "aligned_unaligned":
        return _original_apply(
            self,
            target_images,
            target_texts,
            donor_images,
            donor_texts,
            mode,
        )
    if donor_images is None or donor_texts is None:
        raise ValueError("aligned_unaligned requires a donor sample")
    return _aligned_unaligned(
        self,
        target_images,
        target_texts,
        donor_images,
        donor_texts,
    )


def _patched_getitem(self, index):
    base_index = int(index) % len(self.records)
    target_images, target_texts = self._load(self.records[base_index])
    mode = self.augmentor.choose_mode()

    donor_images = donor_texts = None
    if mode in {"cross_injection", "aligned_unaligned"}:
        donor_index = (
            random.randrange(len(self.records) - 1)
            if len(self.records) > 1
            else 0
        )
        if len(self.records) > 1 and donor_index >= base_index:
            donor_index += 1
        donor_images, donor_texts = self._load(self.records[donor_index])

    result = self.augmentor.apply(
        target_images,
        target_texts,
        donor_images,
        donor_texts,
        mode,
    )
    if result["image2"] is None:
        return result["text1"], loader._to_tensor(result["image1"])
    return {
        "text1": result["text1"],
        "image1": loader._to_tensor(result["image1"]),
        "text2": result["text2"],
        "image2": loader._to_tensor(result["image2"]),
    }


loader.compose_separated_regions = _dense_compose
loader.SyntheticPairAugmentor.MODES = (
    "full_line",
    "two_region",
    "cross_injection",
    "aligned_unaligned",
)
loader.SyntheticPairAugmentor.__init__ = _patched_init
loader.SyntheticPairAugmentor.choose_mode = _choose_mode
loader.SyntheticPairAugmentor.apply = _patched_apply
loader.ExpandedAugmentedSyntheticDataset.__getitem__ = _patched_getitem
