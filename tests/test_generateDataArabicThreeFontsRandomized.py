from pathlib import Path
import random

import pytest

import generateDataArabicThreeFonts as generator
from generateDataArabicThreeFontsRandomized import MODES, build_plan, mode_schedule


FONTS = (Path("a.ttf"), Path("b.ttf"), Path("c.ttf"))


@pytest.mark.parametrize(
    ("mode", "expected_regions"),
    [
        ("original", 1),
        ("cross_injection", 1),
        ("aligned_unaligned", 1),
        ("two_aligned_parts", 2),
        ("three_aligned_parts", 3),
    ],
)
def test_randomized_modes_preserve_exact_aligned_regions(mode, expected_regions):
    plan = build_plan(
        mode,
        random.Random(17),
        generator.BASE_PHRASES,
        generator.CONTEXT_PHRASES,
        FONTS,
        FONTS[0],
        FONTS[1],
        0.5,
        63,
    )

    line1_shared = [
        segment.text for segment in plan.line1.segments if segment.role == "shared"
    ]
    line2_shared = [
        segment.text for segment in plan.line2.segments if segment.role == "shared"
    ]
    assert line1_shared == line2_shared
    assert len(line1_shared) == expected_regions
    assert len(plan.line1.text) <= 63
    assert len(plan.line2.text) <= 63

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
