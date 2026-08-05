#!/usr/bin/env python3
"""Generate randomized exact-text Arabic pairs with 1-3 aligned regions."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

import generateDataArabicThreeFonts as generator
import generateDataArabicThreeFontsCompatible as compatible

MODES = (
    "original",
    "cross_injection",
    "aligned_unaligned",
    "two_aligned_parts",
    "three_aligned_parts",
)


def fragments(phrases: Sequence[str], max_words: int, max_chars: int) -> tuple[str, ...]:
    values: set[str] = set()
    for phrase in phrases:
        words = compatible.normalize(phrase).split()
        for size in range(1, min(max_words, len(words)) + 1):
            for start in range(len(words) - size + 1):
                value = " ".join(words[start : start + size])
                if 2 <= len(value) <= max_chars:
                    values.add(value)
    return tuple(sorted(values))


def choose(
    rng: random.Random,
    pool: Sequence[str],
    count: int,
    used: set[str] | None = None,
) -> list[str]:
    used = set() if used is None else set(used)
    candidates = list(
        dict.fromkeys(
            compatible.normalize(value)
            for value in pool
            if compatible.normalize(value) not in used
        )
    )
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct phrases; found {len(candidates)}")
    return rng.sample(candidates, count)


def trim(parts: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    parts = [
        (compatible.normalize(text), role)
        for text, role in parts
        if compatible.normalize(text)
    ]
    while max_chars > 0 and len(compatible.join_text(text for text, _ in parts)) > max_chars:
        shrinkable = [
            index
            for index, (text, role) in enumerate(parts)
            if role != "shared" and len(text.split()) > 1
        ]
        if shrinkable:
            index = max(shrinkable, key=lambda item: len(parts[item][0]))
            text, role = parts[index]
            parts[index] = (" ".join(text.split()[:-1]), role)
            continue
        removable = [
            index
            for index, (_, role) in enumerate(parts)
            if role in {"context", "injected"}
        ]
        if not removable:
            break
        parts.pop(max(removable, key=lambda item: len(parts[item][0])))
    if len(compatible.join_text(text for text, _ in parts)) > max_chars:
        raise ValueError("Shared regions exceed --max-text-chars")
    return parts


def make_line(
    parts: list[tuple[str, str]],
    base_font: Path,
    fonts: Sequence[Path],
    rng: random.Random,
    mixed_font_probability: float,
    max_chars: int,
) -> generator.LinePlan:
    segments = []
    for text, role in trim(parts, max_chars):
        font = base_font
        if role in {"injected", "separator"} and rng.random() < mixed_font_probability:
            font = rng.choice(tuple(fonts))
        segments.append(generator.Segment(text, role, font))
    return generator.LinePlan(tuple(segments))


def insert_shared(
    rng: random.Random,
    shared: str,
    other: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    result = list(other)
    result.insert(rng.randrange(len(result) + 1), (shared, "shared"))
    return result


def build_plan(
    mode: str,
    rng: random.Random,
    base_pool: Sequence[str],
    context_pool: Sequence[str],
    fonts: Sequence[Path],
    primary: Path,
    secondary: Path,
    mixed_font_probability: float,
    max_chars: int,
) -> generator.PairPlan:
    shared_count = {
        "original": 1,
        "cross_injection": 1,
        "aligned_unaligned": 1,
        "two_aligned_parts": 2,
        "three_aligned_parts": 3,
    }[mode]
    shared_pool = fragments(
        base_pool,
        2 if shared_count < 3 else 1,
        {1: 20, 2: 14, 3: 10}[shared_count],
    )
    shared = choose(rng, shared_pool, shared_count)
    rng.shuffle(shared)
    used = set(shared)
    contexts = fragments(context_pool, 2, 14)
    separators = fragments(context_pool, 1, 9)

    if mode == "original":
        first = choose(rng, contexts, rng.randint(1, 2), used)
        second = choose(rng, contexts, rng.randint(1, 2), used | set(first))
        parts1 = insert_shared(rng, shared[0], [(value, "context") for value in first])
        parts2 = insert_shared(rng, shared[0], [(value, "context") for value in second])
    elif mode == "cross_injection":
        first = choose(rng, contexts, rng.randint(1, 2), used)
        second = choose(rng, contexts, rng.randint(1, 2), used | set(first))
        parts1 = insert_shared(rng, shared[0], [(value, "injected") for value in first])
        parts2 = insert_shared(rng, shared[0], [(value, "injected") for value in second])
    elif mode == "aligned_unaligned":
        values = choose(rng, contexts, 4, used)
        parts1 = insert_shared(
            rng, shared[0], [(values[0], "context"), (values[2], "injected")]
        )
        parts2 = insert_shared(
            rng, shared[0], [(values[1], "context"), (values[3], "injected")]
        )
    elif mode == "two_aligned_parts":
        sep = choose(rng, separators, 2, used)
        parts1 = [(shared[0], "shared"), (sep[0], "separator"), (shared[1], "shared")]
        parts2 = [(shared[0], "shared"), (sep[1], "separator"), (shared[1], "shared")]
        optional = choose(rng, contexts, 2, used | set(sep))
        if rng.random() < 0.5:
            parts1.insert(0, (optional[0], "context"))
        if rng.random() < 0.5:
            parts2.append((optional[1], "context"))
    elif mode == "three_aligned_parts":
        sep = choose(rng, separators, 4, used)
        parts1 = [
            (shared[0], "shared"), (sep[0], "separator"),
            (shared[1], "shared"), (sep[1], "separator"),
            (shared[2], "shared"),
        ]
        parts2 = [
            (shared[0], "shared"), (sep[2], "separator"),
            (shared[1], "shared"), (sep[3], "separator"),
            (shared[2], "shared"),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    line1 = make_line(parts1, primary, fonts, rng, mixed_font_probability, max_chars)
    line2 = make_line(parts2, secondary, fonts, rng, mixed_font_probability, max_chars)
    for value in shared:
        if value not in line1.text or value not in line2.text:
            raise RuntimeError(f"Shared region lost: {value}")
    for line in (line1, line2):
        if sum(segment.role == "shared" for segment in line.segments) != shared_count:
            raise RuntimeError("Incorrect number of aligned regions")
    return generator.PairPlan(
        mode, " || ".join(shared), line1, line2, primary, secondary
    )


def mode_schedule(
    rng: random.Random,
    total: int,
    ratios: dict[str, float],
) -> tuple[list[str], dict[str, int]]:
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Mode ratios must sum to 1.0, got {sum(ratios.values())}")
    raw = {mode: total * ratios[mode] for mode in MODES}
    counts = {mode: int(raw[mode]) for mode in MODES}
    remainder = total - sum(counts.values())
    order = sorted(MODES, key=lambda mode: raw[mode] - counts[mode], reverse=True)
    for mode in order[:remainder]:
        counts[mode] += 1
    active = [mode for mode in MODES if ratios[mode] > 0]
    if total >= len(active):
        for mode in active:
            if counts[mode] == 0:
                donor = max(active, key=lambda item: counts[item])
                counts[donor] -= 1
                counts[mode] += 1
    schedule = [mode for mode in MODES for _ in range(counts[mode])]
    rng.shuffle(schedule)
    return schedule, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("DataSet/Synthetic_Arabic_Three_Font_Augmented"))
    parser.add_argument("--font-dir", type=Path, default=Path("Fonts"))
    parser.add_argument("--fonts", nargs="*", default=None)
    parser.add_argument("--font-count", type=int, default=3)
    parser.add_argument("--samples-per-font", type=int, default=3000)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--font-size", type=int, default=90)
    parser.add_argument("--segment-gap-min", type=int, default=2)
    parser.add_argument("--segment-gap-max", type=int, default=6)
    parser.add_argument("--outer-margin", type=int, default=8)
    parser.add_argument("--vertical-margin", type=int, default=5)
    parser.add_argument("--target-fill-ratio", type=float, default=0.96)
    parser.add_argument("--max-text-chars", type=int, default=63)
    parser.add_argument("--original-ratio", type=float, default=0.10)
    parser.add_argument("--cross-injection-ratio", type=float, default=0.25)
    parser.add_argument("--aligned-unaligned-ratio", type=float, default=0.20)
    parser.add_argument("--two-aligned-parts-ratio", type=float, default=0.25)
    parser.add_argument("--three-aligned-parts-ratio", type=float, default=0.20)
    parser.add_argument("--mixed-font-injection-prob", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {root}; pass --overwrite")
    if args.samples_per_font <= 0:
        raise ValueError("--samples-per-font must be positive")
    root.mkdir(parents=True, exist_ok=True)

    generator.clean_text = compatible.clean_text
    generator.normalize = compatible.normalize
    generator.join_text = compatible.join_text
    generator.visual_text = compatible.visual_text

    paths = compatible.output_dirs(root)
    fonts = generator.discover_fonts(args.font_dir, args.fonts, args.font_count)
    external = generator.read_corpus(args.corpus)
    base_pool = tuple(dict.fromkeys((*generator.BASE_PHRASES, *external)))
    context_pool = tuple(dict.fromkeys((*generator.CONTEXT_PHRASES, *external)))
    rng = random.Random(args.seed)
    total = len(fonts) * args.samples_per_font
    ratios = {
        "original": args.original_ratio,
        "cross_injection": args.cross_injection_ratio,
        "aligned_unaligned": args.aligned_unaligned_ratio,
        "two_aligned_parts": args.two_aligned_parts_ratio,
        "three_aligned_parts": args.three_aligned_parts_ratio,
    }
    modes, counts = mode_schedule(rng, total, ratios)
    primary_fonts = [font for font in fonts for _ in range(args.samples_per_font)]
    rng.shuffle(primary_fonts)
    primary_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()

    print("Selected fonts:", *(f"\n  - {font}" for font in fonts), sep="")
    print("Mode counts:", json.dumps(counts, ensure_ascii=False))
    with (root / "metadata.jsonl").open("w", encoding="utf-8") as manifest:
        for offset in tqdm(range(total), desc="Generating randomized pairs"):
            primary = primary_fonts[offset]
            secondary = rng.choice(tuple(fonts))
            plan = build_plan(
                modes[offset], rng, base_pool, context_pool, fonts,
                primary, secondary, args.mixed_font_injection_prob,
                args.max_text_chars,
            )
            record = compatible.save(
                offset + 1,
                plan,
                compatible.render_line(plan.line1, rng, args),
                compatible.render_line(plan.line2, rng, args),
                paths,
            )
            record["shared_parts"] = [
                segment["text"]
                for segment in record["line1_segments"]
                if segment["role"] == "shared"
            ]
            record["aligned_region_count"] = len(record["shared_parts"])
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            primary_counts[primary.name] += 1
            secondary_counts[secondary.name] += 1

    summary = {
        "total_samples": total,
        "samples_per_primary_font": args.samples_per_font,
        "fonts": [font.name for font in fonts],
        "primary_font_counts": dict(primary_counts),
        "secondary_font_counts": dict(secondary_counts),
        "randomized_font_order": True,
        "random_secondary_font_per_pair": True,
        "mode_counts": counts,
        "mode_ratios": ratios,
        "aligned_regions": [1, 2, 3],
        "exact_transcripts": True,
        "connected_arabic_shaping": True,
        "full_height_masks": True,
        "image_size": [args.image_width, args.image_height],
        "font_size": args.font_size,
        "segment_gap_range": [args.segment_gap_min, args.segment_gap_max],
        "target_fill_ratio": args.target_fill_ratio,
        "generated_directories": ["images", "masks", "texts"],
        "omitted_outputs": ["matrices", "similarity_matrices", "subword_boxes"],
    }
    (root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
