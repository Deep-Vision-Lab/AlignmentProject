from pathlib import Path
import random

from generateDataArabicThreeFonts import build_plan, compact_layout, join_text


def test_generated_text_is_exact_segment_join():
    fonts = (Path("a.ttf"), Path("b.ttf"), Path("c.ttf"))
    plan = build_plan(
        "aligned_unaligned",
        random.Random(7),
        ("القراءة غذاء العقل", "الشمس مشرقة اليوم", "العلم نور المستقبل"),
        (
            "في هذا المكان",
            "بعد قليل",
            "بكل هدوء",
            "مع نسيم الصباح",
            "بدون شك",
        ),
        fonts,
        fonts[0],
        fonts[1],
        0.5,
        63,
    )
    assert plan.line1.text == join_text(
        segment.text for segment in plan.line1.segments
    )
    assert plan.line2.text == join_text(
        segment.text for segment in plan.line2.segments
    )
    assert plan.shared in plan.line1.text and plan.shared in plan.line2.text
    assert any(segment.role == "injected" for segment in plan.line1.segments)
    assert any(segment.role == "injected" for segment in plan.line2.segments)


def test_compact_layout_preserves_small_requested_gaps():
    sizes, positions = compact_layout(
        [(220, 60), (180, 55), (160, 58)],
        1024,
        128,
        [6, 8],
        8,
        5,
        0.94,
    )
    assert positions[0][0] > positions[1][0] > positions[2][0]
    assert positions[0][0] - (positions[1][0] + sizes[1][0]) == 6
    assert positions[1][0] - (positions[2][0] + sizes[2][0]) == 8
