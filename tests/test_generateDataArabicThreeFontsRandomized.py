from pathlib import Path
import random
from types import SimpleNamespace

from PIL import Image
import pytest

import generateDataArabicThreeFonts as generator
from generateDataArabicThreeFontsRandomized import (
    MODES,
    apply_noise,
    build_plan,
    mode_schedule,
)


FONTS = (Path("a.ttf"), Path("b.ttf"), Path("c.ttf"))


@pytest.mark.parametrize(
    ("mode", "expected_regions", "minimum_shared_chars"),
    [
        ("original", 1, 18),
        ("cross_injection", 1, 18),
        ("aligned_unaligned", 1, 18),
        ("two_aligned_parts", 2, 24),
        ("three_aligned_parts", 3, 30),
    ],
)
def test_randomized_modes_generate_long_exact_aligned_regions(
    mode,
    expected_regions,
    minimum_shared_chars,
):
    plan = build_plan(
        mode,
        random.Random(17),
        generator.BASE_PHRASES,
        generator.CONTEXT_PHRASES,
        FONTS,
        FONTS[0],
        FONTS[1],
        0.5,
        120,
        85,
    )

    line1_shared = [
        segment.text for segment in plan.line1.segments if segment.role == "shared"
    ]
    line2_shared = [
        segment.text for segment in plan.line2.segments if segment.role == "shared"
    ]
    assert line1_shared == line2_shared
    assert len(line1_shared) == expected_regions
    assert sum(len(value) for value in line1_shared) >= minimum_shared_chars
    assert 85 <= len(plan.line1.text) <= 120
    assert 85 <= len(plan.line2.text) <= 120

    if expected_regions > 1:
        for line in (plan.line1, plan.line2):
            shared_indexes = [
                index
                for index, segment in enumerate(line.segments)
                if segment.role == "shared"
            ]
            assert all(
                right - left >= 2
                for left, right in zip(shared_indexes, shared_indexes[1:])
            )


def test_noise_changes_image_and_records_operation():
    args = SimpleNamespace(
        gaussian_noise_prob=1.0,
        gaussian_noise_std_min=8.0,
        gaussian_noise_std_max=8.0,
        salt_pepper_noise_prob=0.0,
        salt_pepper_density_min=0.0,
        salt_pepper_density_max=0.0,
        blur_prob=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
    )
    image = Image.new("RGB", (32, 16), 0)
    noisy, metadata = apply_noise(image, random.Random(3), args, probability=1.0)

    assert noisy.tobytes() != image.tobytes()
    assert metadata["applied"] is True
    assert metadata["operations"][0]["type"] == "gaussian"


def test_nine_sample_smoke_schedule_contains_every_mode():
    schedule, counts = mode_schedule(
        random.Random(3),
        9,
        {
            "original": 0.10,
            "cross_injection": 0.25,
            "aligned_unaligned": 0.20,
            "two_aligned_parts": 0.25,
            "three_aligned_parts": 0.20,
        },
    )
    assert len(schedule) == 9
    assert set(schedule) == set(MODES)
    assert all(counts[mode] >= 1 for mode in MODES)
