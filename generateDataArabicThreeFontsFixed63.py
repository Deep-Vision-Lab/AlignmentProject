#!/usr/bin/env python3
"""Generate the three-font augmented Arabic dataset with exactly 63 characters per line."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

import generateDataArabicThreeFontsRandomized as base

TARGET_TEXT_CHARS = 63
_ORIGINAL_BUILD_PLAN = base.build_plan


def _normalized_pool(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(text for text in (base.compatible.normalize(value) for value in values) if text))


def _line_length(segments) -> int:
    return len(base.compatible.join_text(segment.text for segment in segments))


def _fit_line_exact(plan, rng, pool: Sequence[str], fallback_font):
    segments = [
        base.generator.Segment(base.compatible.normalize(segment.text), segment.role, segment.font)
        for segment in plan.segments
        if base.compatible.normalize(segment.text)
    ]
    current = _line_length(segments)
    if current > TARGET_TEXT_CHARS:
        raise ValueError(f"Line already exceeds fixed target: {current}>{TARGET_TEXT_CHARS}")
    candidates = _normalized_pool(pool)
    rng.shuffle(candidates)
    used_texts = {segment.text for segment in segments}
    while current < TARGET_TEXT_CHARS:
        gap = TARGET_TEXT_CHARS - current
        fitting = [
            value for value in candidates
            if value not in used_texts and len(value) + (1 if segments else 0) <= gap
        ]
        if not fitting:
            break
        value = max(fitting, key=len)
        segments.append(base.generator.Segment(value, "context", fallback_font))
        used_texts.add(value)
        current = _line_length(segments)
    gap = TARGET_TEXT_CHARS - current
    if gap:
        editable = [i for i, segment in enumerate(segments) if segment.role != "shared"]
        if not editable:
            raise ValueError("Cannot reach fixed length without modifying a shared region")
        letters = "".join(
            character for value in candidates for character in value if not character.isspace()
        ) or "ا"
        suffix = (letters * math.ceil(gap / len(letters)))[:gap]
        index = rng.choice(editable)
        segment = segments[index]
        segments[index] = base.generator.Segment(segment.text + suffix, segment.role, segment.font)
    fitted = base.generator.LinePlan(tuple(segments))
    if len(fitted.text) != TARGET_TEXT_CHARS:
        raise RuntimeError(
            f"Fixed-length construction failed: got {len(fitted.text)}, expected {TARGET_TEXT_CHARS}"
        )
    return fitted


def build_plan_fixed63(
    mode: str,
    rng,
    base_pool,
    context_pool,
    fonts,
    primary,
    secondary,
    mixed_font_probability: float,
    max_chars: int,
    min_chars: int = 0,
):
    del max_chars, min_chars
    plan = _ORIGINAL_BUILD_PLAN(
        mode, rng, base_pool, context_pool, fonts, primary, secondary,
        mixed_font_probability, TARGET_TEXT_CHARS, 0,
    )
    filler_pool = tuple(dict.fromkeys((*context_pool, *base_pool)))
    line1 = _fit_line_exact(plan.line1, rng, filler_pool, primary)
    line2 = _fit_line_exact(plan.line2, rng, filler_pool, secondary)
    shared_values = [segment.text for segment in line1.segments if segment.role == "shared"]
    for value in shared_values:
        if value not in line2.text:
            raise RuntimeError(f"Shared region lost while fixing text length: {value}")
    return base.generator.PairPlan(
        plan.mode, plan.shared, line1, line2, plan.primary_font, plan.secondary_font
    )


def _replace_cli_option(flag: str, value: str) -> None:
    while flag in sys.argv:
        index = sys.argv.index(flag)
        del sys.argv[index:index + 2]
    sys.argv.extend([flag, value])


def _cli_option(flag: str, default: str) -> str:
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for {flag}")
    return sys.argv[index + 1]


def _validate_output(root: Path) -> None:
    text_files = sorted((root / "texts").glob("text[12]_*.txt"))
    if not text_files:
        raise RuntimeError(f"No generated transcript files found under {root / 'texts'}")
    invalid = []
    for path in text_files:
        length = len(path.read_text(encoding="utf-8").strip())
        if length != TARGET_TEXT_CHARS:
            invalid.append((path.name, length))
            if len(invalid) >= 10:
                break
    if invalid:
        details = ", ".join(f"{name}={length}" for name, length in invalid)
        raise RuntimeError(f"Fixed-length validation failed: {details}")
    summary_path = root / "generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary["text_length_range"] = [TARGET_TEXT_CHARS, TARGET_TEXT_CHARS]
    summary["fixed_text_length"] = TARGET_TEXT_CHARS
    summary["validated_transcript_files"] = len(text_files)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Validated {len(text_files)} transcript files: "
        f"every line has exactly {TARGET_TEXT_CHARS} characters."
    )


def main() -> None:
    base.build_plan = build_plan_fixed63
    _replace_cli_option("--min-text-chars", str(TARGET_TEXT_CHARS))
    _replace_cli_option("--max-text-chars", str(TARGET_TEXT_CHARS))
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "DataSet/AugmentedArabicDataset63"])
    output_dir = Path(_cli_option("--output-dir", "DataSet/AugmentedArabicDataset63"))
    base.main()
    _validate_output(output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
