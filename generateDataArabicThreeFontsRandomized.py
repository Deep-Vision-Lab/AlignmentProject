#!/usr/bin/env python3
"""Generate noisy, long, randomized Arabic pairs with 1-3 aligned regions."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter
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


def fragments(
    phrases: Sequence[str],
    min_words: int,
    max_words: int,
    min_chars: int,
    max_chars: int,
) -> tuple[str, ...]:
    values: set[str] = set()
    for phrase in phrases:
        words = compatible.normalize(phrase).split()
        for size in range(min_words, min(max_words, len(words)) + 1):
            for start in range(len(words) - size + 1):
                value = " ".join(words[start : start + size])
                if min_chars <= len(value) <= max_chars:
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


def text_length(parts: Sequence[tuple[str, str]]) -> int:
    return len(compatible.join_text(text for text, _ in parts))


def extend_parts(
    parts: list[tuple[str, str]],
    rng: random.Random,
    pool: Sequence[str],
    used: set[str],
    min_chars: int,
    max_chars: int,
    role: str = "context",
) -> list[tuple[str, str]]:
    result = list(parts)
    candidates = [
        compatible.normalize(value)
        for value in pool
        if compatible.normalize(value) not in used
    ]
    rng.shuffle(candidates)
    while text_length(result) < min_chars and candidates:
        value = candidates.pop()
        positions = list(range(len(result) + 1))
        rng.shuffle(positions)
        for position in positions:
            candidate = list(result)
            candidate.insert(position, (value, role))
            if text_length(candidate) <= max_chars:
                result = candidate
                used.add(value)
                break
    return result


def trim(parts: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    parts = [
        (compatible.normalize(text), role)
        for text, role in parts
        if compatible.normalize(text)
    ]
    while max_chars > 0 and text_length(parts) > max_chars:
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
    if text_length(parts) > max_chars:
        raise ValueError("Shared regions and separators exceed --max-text-chars")
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
        if role in {"context", "injected", "separator"} and rng.random() < mixed_font_probability:
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
    min_chars: int = 0,
) -> generator.PairPlan:
    shared_count = {
        "original": 1,
        "cross_injection": 1,
        "aligned_unaligned": 1,
        "two_aligned_parts": 2,
        "three_aligned_parts": 3,
    }[mode]
    shared_settings = {
        1: (3, 5, 18, 44),
        2: (2, 4, 12, 30),
        3: (2, 3, 10, 24),
    }
    shared_pool = fragments(base_pool, *shared_settings[shared_count])
    shared = choose(rng, shared_pool, shared_count)
    rng.shuffle(shared)
    used = set(shared)

    long_pool = fragments(
        tuple(dict.fromkeys((*base_pool, *context_pool))),
        2,
        5,
        8,
        38,
    )
    contexts = fragments(
        tuple(dict.fromkeys((*context_pool, *base_pool))),
        2,
        4,
        8,
        30,
    )
    separators = fragments(context_pool, 1, 3, 4, 20)

    if mode == "original":
        first = choose(rng, contexts, rng.randint(2, 3), used)
        used.update(first)
        second = choose(rng, contexts, rng.randint(2, 3), used)
        used.update(second)
        parts1 = insert_shared(rng, shared[0], [(value, "context") for value in first])
        parts2 = insert_shared(rng, shared[0], [(value, "context") for value in second])
    elif mode == "cross_injection":
        first = choose(rng, contexts, rng.randint(2, 3), used)
        used.update(first)
        second = choose(rng, contexts, rng.randint(2, 3), used)
        used.update(second)
        parts1 = insert_shared(rng, shared[0], [(value, "injected") for value in first])
        parts2 = insert_shared(rng, shared[0], [(value, "injected") for value in second])
    elif mode == "aligned_unaligned":
        values = choose(rng, contexts, 6, used)
        used.update(values)
        parts1 = insert_shared(
            rng,
            shared[0],
            [
                (values[0], "context"),
                (values[2], "injected"),
                (values[4], "context"),
            ],
        )
        parts2 = insert_shared(
            rng,
            shared[0],
            [
                (values[1], "context"),
                (values[3], "injected"),
                (values[5], "context"),
            ],
        )
    elif mode == "two_aligned_parts":
        sep = choose(rng, separators, 2, used)
        used.update(sep)
        parts1 = [
            (shared[0], "shared"),
            (sep[0], "separator"),
            (shared[1], "shared"),
        ]
        parts2 = [
            (shared[0], "shared"),
            (sep[1], "separator"),
            (shared[1], "shared"),
        ]
        optional = choose(rng, contexts, 4, used)
        used.update(optional)
        parts1.insert(0, (optional[0], "context"))
        parts1.append((optional[1], "context"))
        parts2.insert(0, (optional[2], "context"))
        parts2.append((optional[3], "context"))
    elif mode == "three_aligned_parts":
        sep = choose(rng, separators, 4, used)
        used.update(sep)
        parts1 = [
            (shared[0], "shared"),
            (sep[0], "separator"),
            (shared[1], "shared"),
            (sep[1], "separator"),
            (shared[2], "shared"),
        ]
        parts2 = [
            (shared[0], "shared"),
            (sep[2], "separator"),
            (shared[1], "shared"),
            (sep[3], "separator"),
            (shared[2], "shared"),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    parts1 = extend_parts(parts1, rng, long_pool, used, min_chars, max_chars)
    parts2 = extend_parts(parts2, rng, long_pool, used, min_chars, max_chars)
    line1 = make_line(parts1, primary, fonts, rng, mixed_font_probability, max_chars)
    line2 = make_line(parts2, secondary, fonts, rng, mixed_font_probability, max_chars)

    for value in shared:
        if value not in line1.text or value not in line2.text:
            raise RuntimeError(f"Shared region lost: {value}")
    for line in (line1, line2):
        if sum(segment.role == "shared" for segment in line.segments) != shared_count:
            raise RuntimeError("Incorrect number of aligned regions")
        if len(line.text) < min_chars:
            raise ValueError(
                f"Could not reach --min-text-chars={min_chars}; got {len(line.text)}"
            )
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


def apply_noise(
    image: Image.Image,
    rng: random.Random,
    args: argparse.Namespace,
    probability: float,
) -> tuple[Image.Image, dict]:
    info: dict[str, object] = {"applied": False, "operations": []}
    if probability <= 0 or rng.random() >= probability:
        return image, info

    local_rng = np.random.default_rng(rng.randrange(2**32))
    array = np.asarray(image).astype(np.float32)
    operations: list[dict[str, float | str]] = []

    if rng.random() < args.gaussian_noise_prob:
        std = rng.uniform(args.gaussian_noise_std_min, args.gaussian_noise_std_max)
        array += local_rng.normal(0.0, std, size=array.shape)
        operations.append({"type": "gaussian", "std": round(std, 4)})

    if rng.random() < args.salt_pepper_noise_prob:
        density = rng.uniform(
            args.salt_pepper_density_min,
            args.salt_pepper_density_max,
        )
        height, width = array.shape[:2]
        random_map = local_rng.random((height, width))
        pepper = random_map < density / 2.0
        salt = (random_map >= density / 2.0) & (random_map < density)
        array[pepper] = 0
        array[salt] = 255
        operations.append({"type": "salt_pepper", "density": round(density, 6)})

    noisy = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode=image.mode)
    if rng.random() < args.blur_prob:
        radius = rng.uniform(args.blur_radius_min, args.blur_radius_max)
        noisy = noisy.filter(ImageFilter.GaussianBlur(radius=radius))
        operations.append({"type": "blur", "radius": round(radius, 4)})

    if not operations:
        std = rng.uniform(args.gaussian_noise_std_min, args.gaussian_noise_std_max)
        array = np.asarray(image).astype(np.float32)
        array += local_rng.normal(0.0, std, size=array.shape)
        noisy = Image.fromarray(
            np.clip(array, 0, 255).astype(np.uint8),
            mode=image.mode,
        )
        operations.append({"type": "gaussian", "std": round(std, 4)})

    info["applied"] = True
    info["operations"] = operations
    return noisy, info


def render_with_noise(
    plan: generator.LinePlan,
    mode: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[tuple[Image.Image, Image.Image, list[dict]], dict]:
    image, mask, metadata = compatible.render_line(plan, rng, args)
    probability = args.noise_prob
    if mode == "original":
        probability *= args.original_noise_scale
    image, noise = apply_noise(image, rng, args, probability)
    return (image, mask, metadata), noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DataSet/Synthetic_Arabic_Three_Font_Augmented"),
    )
    parser.add_argument("--font-dir", type=Path, default=Path("Fonts"))
    parser.add_argument("--fonts", nargs="*", default=None)
    parser.add_argument("--font-count", type=int, default=3)
    parser.add_argument("--samples-per-font", type=int, default=3000)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--font-size", type=int, default=84)
    parser.add_argument("--segment-gap-min", type=int, default=2)
    parser.add_argument("--segment-gap-max", type=int, default=6)
    parser.add_argument("--outer-margin", type=int, default=8)
    parser.add_argument("--vertical-margin", type=int, default=5)
    parser.add_argument("--target-fill-ratio", type=float, default=0.98)
    parser.add_argument("--min-text-chars", type=int, default=85)
    parser.add_argument("--max-text-chars", type=int, default=120)
    parser.add_argument("--original-ratio", type=float, default=0.10)
    parser.add_argument("--cross-injection-ratio", type=float, default=0.25)
    parser.add_argument("--aligned-unaligned-ratio", type=float, default=0.20)
    parser.add_argument("--two-aligned-parts-ratio", type=float, default=0.25)
    parser.add_argument("--three-aligned-parts-ratio", type=float, default=0.20)
    parser.add_argument("--mixed-font-injection-prob", type=float, default=0.50)
    parser.add_argument("--noise-prob", type=float, default=0.75)
    parser.add_argument("--original-noise-scale", type=float, default=0.20)
    parser.add_argument("--gaussian-noise-prob", type=float, default=0.90)
    parser.add_argument("--gaussian-noise-std-min", type=float, default=2.0)
    parser.add_argument("--gaussian-noise-std-max", type=float, default=10.0)
    parser.add_argument("--salt-pepper-noise-prob", type=float, default=0.55)
    parser.add_argument("--salt-pepper-density-min", type=float, default=0.0005)
    parser.add_argument("--salt-pepper-density-max", type=float, default=0.0035)
    parser.add_argument("--blur-prob", type=float, default=0.25)
    parser.add_argument("--blur-radius-min", type=float, default=0.15)
    parser.add_argument("--blur-radius-max", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {root}; pass --overwrite"
        )
    if args.samples_per_font <= 0:
        raise ValueError("--samples-per-font must be positive")
    if not 0 < args.target_fill_ratio <= 1:
        raise ValueError("--target-fill-ratio must be in (0, 1]")
    if not 0 <= args.min_text_chars <= args.max_text_chars:
        raise ValueError("Require 0 <= --min-text-chars <= --max-text-chars")
    for name in (
        "noise_prob",
        "original_noise_scale",
        "gaussian_noise_prob",
        "salt_pepper_noise_prob",
        "blur_prob",
        "mixed_font_injection_prob",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.gaussian_noise_std_min < 0 or (
        args.gaussian_noise_std_max < args.gaussian_noise_std_min
    ):
        raise ValueError("Invalid Gaussian noise standard-deviation range")
    if args.salt_pepper_density_min < 0 or (
        args.salt_pepper_density_max < args.salt_pepper_density_min
    ):
        raise ValueError("Invalid salt-and-pepper density range")
    if args.blur_radius_min < 0 or args.blur_radius_max < args.blur_radius_min:
        raise ValueError("Invalid blur radius range")

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
    noisy_line_count = 0

    print("Selected fonts:", *(f"\n  - {font}" for font in fonts), sep="")
    print("Mode counts:", json.dumps(counts, ensure_ascii=False))
    print(
        "Text length range:",
        f"{args.min_text_chars}-{args.max_text_chars} characters",
    )
    with (root / "metadata.jsonl").open("w", encoding="utf-8") as manifest:
        for offset in tqdm(range(total), desc="Generating randomized pairs"):
            primary = primary_fonts[offset]
            secondary = rng.choice(tuple(fonts))
            last_error: Exception | None = None
            for _ in range(40):
                try:
                    plan = build_plan(
                        modes[offset],
                        rng,
                        base_pool,
                        context_pool,
                        fonts,
                        primary,
                        secondary,
                        args.mixed_font_injection_prob,
                        args.max_text_chars,
                        args.min_text_chars,
                    )
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                raise RuntimeError(
                    f"Could not construct a long {modes[offset]} sample"
                ) from last_error

            rendered1, noise1 = render_with_noise(plan.line1, plan.mode, rng, args)
            rendered2, noise2 = render_with_noise(plan.line2, plan.mode, rng, args)
            record = compatible.save(
                offset + 1,
                plan,
                rendered1,
                rendered2,
                paths,
            )
            record["shared_parts"] = [
                segment["text"]
                for segment in record["line1_segments"]
                if segment["role"] == "shared"
            ]
            record["aligned_region_count"] = len(record["shared_parts"])
            record["text1_length"] = len(plan.line1.text)
            record["text2_length"] = len(plan.line2.text)
            record["noise1"] = noise1
            record["noise2"] = noise2
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            primary_counts[primary.name] += 1
            secondary_counts[secondary.name] += 1
            noisy_line_count += int(bool(noise1["applied"])) + int(
                bool(noise2["applied"])
            )

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
        "text_length_range": [args.min_text_chars, args.max_text_chars],
        "long_shared_regions": True,
        "exact_transcripts": True,
        "connected_arabic_shaping": True,
        "full_height_masks": True,
        "noise": {
            "line_probability": args.noise_prob,
            "original_mode_scale": args.original_noise_scale,
            "gaussian_probability": args.gaussian_noise_prob,
            "gaussian_std_range": [
                args.gaussian_noise_std_min,
                args.gaussian_noise_std_max,
            ],
            "salt_pepper_probability": args.salt_pepper_noise_prob,
            "salt_pepper_density_range": [
                args.salt_pepper_density_min,
                args.salt_pepper_density_max,
            ],
            "blur_probability": args.blur_prob,
            "blur_radius_range": [
                args.blur_radius_min,
                args.blur_radius_max,
            ],
            "noisy_lines": noisy_line_count,
        },
        "image_size": [args.image_width, args.image_height],
        "font_size": args.font_size,
        "segment_gap_range": [args.segment_gap_min, args.segment_gap_max],
        "target_fill_ratio": args.target_fill_ratio,
        "generated_directories": ["images", "masks", "texts"],
        "omitted_outputs": ["matrices", "similarity_matrices", "subword_boxes"],
    }
    (root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
