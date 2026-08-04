#!/usr/bin/env python3
"""Generate exact-text Arabic pairs with three fonts and compact augmentation.

All augmentation is performed before rendering. Each visible segment therefore
has a known text string, and the saved transcript is assembled from those same
segments. This avoids the label drift caused by estimating text from cropped
pixel fractions.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from Parameters import gapScore, matchScore, mismatchScore
except Exception:
    matchScore, mismatchScore, gapScore = 2, -1, -1

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}

BASE_PHRASES = (
    "الشمس مشرقة اليوم",
    "القهوة صباحا تعدل المزاج",
    "الحديقة مليئة بالزهور",
    "القراءة غذاء العقل والروح",
    "السفر يفتح آفاقا جديدة",
    "الرياضة تحافظ على الصحة",
    "العمل الجاد يؤتي ثماره",
    "التعاون يحقق أفضل النتائج",
    "العائلة هي السند الحقيقي",
    "العلم نور يضيء المستقبل",
    "الصداقة الحقيقية كنز ثمين",
    "المدينة جميلة عند الغروب",
    "الأمانة أساس كل علاقة ناجحة",
    "التفكير الإيجابي يغير الحياة",
    "التعليم يرفع مستوى المجتمعات",
    "الحضارة تبنى بالعلم والعمل",
)

CONTEXT_PHRASES = (
    "بكل تأكيد",
    "في بعض الأحيان",
    "بشكل عام",
    "في الواقع",
    "لهذا السبب",
    "لحسن الحظ",
    "بعد قليل",
    "في المساء الباكر",
    "قبل الظهر",
    "ببطء وحذر",
    "بسرعة كبيرة",
    "بدون أدنى شك",
    "مرة أخرى",
    "في هذا المكان",
    "على سبيل المثال",
    "بعيدا عن الضوضاء",
    "وسط الطبيعة",
    "مع طلوع الفجر",
    "بعد تفكير عميق",
    "تحت سماء صافية",
    "بصوت منخفض",
    "بكل هدوء",
    "على مدار اليوم",
    "بكثير من الصبر",
    "في الوقت المحدد",
    "مع مرور الأيام",
    "خلف الأفق",
    "رغم الصعاب",
    "بعد طول انتظار",
    "مع نسيم الصباح",
)


@dataclass(frozen=True)
class Segment:
    text: str
    role: str
    font: Path


@dataclass(frozen=True)
class LinePlan:
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        return join_text(segment.text for segment in self.segments)


@dataclass(frozen=True)
class PairPlan:
    mode: str
    shared: str
    line1: LinePlan
    line2: LinePlan
    primary_font: Path
    secondary_font: Path


def clean_text(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS.sub("", text)
    return " ".join(ch for ch in text if ch.isprintable()).split()


def normalize(text: str) -> str:
    return " ".join(clean_text(text))


def join_text(parts: Iterable[str]) -> str:
    return " ".join(part for part in (normalize(value) for value in parts) if part)


def visual_text(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise RuntimeError(
            "Install arabic-reshaper and python-bidi before generation."
        ) from exc
    reshaper = arabic_reshaper.ArabicReshaper(
        configuration={
            "delete_harakat": True,
            "support_zwj": False,
            "use_unshaped_instead_of_isolated": True,
        }
    )
    shaped = reshaper.reshape(normalize(text))
    shaped = "".join(
        ch for ch in shaped if ch.isprintable() and ord(ch) not in {0x200C, 0x200D}
    )
    return get_display(shaped)


def discover_fonts(
    font_dir: Path,
    requested: Sequence[str] | None,
    count: int,
) -> tuple[Path, ...]:
    font_dir = font_dir.expanduser().resolve()
    if requested:
        fonts = []
        for value in requested:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (font_dir / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Font not found: {value}")
            fonts.append(path)
    else:
        fonts = sorted(
            (
                path.resolve()
                for path in font_dir.iterdir()
                if path.suffix.lower() in FONT_SUFFIXES
            ),
            key=lambda path: path.name.lower(),
        )[:count]
    fonts = tuple(dict.fromkeys(fonts))
    if len(fonts) != count:
        raise ValueError(
            f"Expected exactly {count} fonts, found {[path.name for path in fonts]}"
        )
    for path in fonts:
        ImageFont.truetype(str(path), 32)
    return fonts


def read_corpus(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(
        text
        for text in (
            normalize(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        if len(text.split()) >= 2
    )


def distinct(rng: random.Random, pool: Sequence[str], used: set[str]) -> str:
    candidates = [normalize(value) for value in pool if normalize(value) not in used]
    if not candidates:
        raise ValueError("Not enough distinct phrases in the corpus")
    return rng.choice(candidates)


def trim_parts(parts: list[tuple[str, str]], max_chars: int) -> list[tuple[str, str]]:
    parts = [(normalize(text), role) for text, role in parts if normalize(text)]
    while max_chars > 0 and len(join_text(text for text, _ in parts)) > max_chars:
        candidates = [
            index
            for index, (text, role) in enumerate(parts)
            if role != "shared" and len(text.split()) > 1
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: len(parts[item][0]))
        text, role = parts[index]
        parts[index] = (" ".join(text.split()[:-1]), role)
    return [(text, role) for text, role in parts if text]


def make_line(
    parts: list[tuple[str, str]],
    line_font: Path,
    donor_font: Path,
    rng: random.Random,
    mixed_font_probability: float,
    max_chars: int,
) -> LinePlan:
    segments = []
    for text, role in trim_parts(parts, max_chars):
        font = (
            donor_font
            if role == "injected" and rng.random() < mixed_font_probability
            else line_font
        )
        segments.append(Segment(text, role, font))
    return LinePlan(tuple(segments))


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
) -> PairPlan:
    shared = normalize(rng.choice(base_pool))
    used = {shared}
    values = []
    for _ in range(4):
        value = distinct(rng, context_pool, used)
        used.add(value)
        values.append(value)
    first, second, third, fourth = values
    donor1 = rng.choice([font for font in fonts if font != primary] or [primary])
    donor2 = rng.choice([font for font in fonts if font != secondary] or [secondary])

    if mode == "original":
        if rng.random() < 0.5:
            parts1 = [(shared, "shared"), (first, "context")]
            parts2 = [(shared, "shared"), (second, "context")]
        else:
            parts1 = [(first, "context"), (shared, "shared")]
            parts2 = [(second, "context"), (shared, "shared")]
    elif mode == "cross_injection":
        layouts = (
            (
                [(shared, "shared"), (first, "injected")],
                [(second, "injected"), (shared, "shared")],
            ),
            (
                [(first, "injected"), (shared, "shared")],
                [(shared, "shared"), (second, "injected")],
            ),
            (
                [(shared, "shared"), (first, "injected")],
                [(shared, "shared"), (second, "injected")],
            ),
            (
                [(first, "injected"), (shared, "shared")],
                [(second, "injected"), (shared, "shared")],
            ),
        )
        parts1, parts2 = rng.choice(layouts)
    elif mode == "aligned_unaligned":
        parts1 = [(first, "context"), (third, "injected")]
        parts2 = [(second, "context"), (fourth, "injected")]
        parts1.insert(rng.randrange(3), (shared, "shared"))
        parts2.insert(rng.randrange(3), (shared, "shared"))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    line1 = make_line(
        parts1, primary, donor1, rng, mixed_font_probability, max_chars
    )
    line2 = make_line(
        parts2, secondary, donor2, rng, mixed_font_probability, max_chars
    )
    if shared not in line1.text or shared not in line2.text:
        raise RuntimeError("Shared text was lost during plan construction")
    return PairPlan(mode, shared, line1, line2, primary, secondary)


def render_patch(segment: Segment, font_size: int) -> Image.Image:
    font = ImageFont.truetype(str(segment.font), font_size)
    text = visual_text(segment.text)
    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    patch = Image.new(
        "L",
        (max(1, right - left) + 6, max(1, bottom - top) + 6),
        0,
    )
    ImageDraw.Draw(patch).text((3 - left, 3 - top), text, font=font, fill=255)
    return patch.crop(patch.getbbox()) if patch.getbbox() else patch


def compact_layout(
    sizes: Sequence[tuple[int, int]],
    width: int,
    height: int,
    gaps: Sequence[int],
    outer_margin: int,
    vertical_margin: int,
    fill_ratio: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    source_width = sum(item[0] for item in sizes)
    source_height = max(item[1] for item in sizes)
    target_width = min(width - 2 * outer_margin, int(width * fill_ratio))
    scale = min(
        (target_width - sum(gaps)) / max(1, source_width),
        (height - 2 * vertical_margin) / max(1, source_height),
    )
    scaled = tuple(
        (max(1, round(patch_width * scale)), max(1, round(patch_height * scale)))
        for patch_width, patch_height in sizes
    )
    content_width = sum(patch_width for patch_width, _ in scaled) + sum(gaps)
    cursor = (width - content_width) // 2 + content_width
    positions = []
    for index, (patch_width, patch_height) in enumerate(scaled):
        cursor -= patch_width
        positions.append((cursor, (height - patch_height) // 2))
        if index < len(gaps):
            cursor -= gaps[index]
    return scaled, tuple(positions)


def render_line(
    plan: LinePlan,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[Image.Image, Image.Image, list[dict]]:
    patches = [render_patch(segment, args.font_size) for segment in plan.segments]
    gaps = [
        rng.randint(args.segment_gap_min, args.segment_gap_max)
        for _ in range(len(patches) - 1)
    ]
    sizes, positions = compact_layout(
        [patch.size for patch in patches],
        args.image_width,
        args.image_height,
        gaps,
        args.outer_margin,
        args.vertical_margin,
        args.target_fill_ratio,
    )
    image = Image.new("L", (args.image_width, args.image_height), 0)
    mask = Image.new("L", image.size, 0)
    metadata = []
    for segment, patch, size, position in zip(
        plan.segments, patches, sizes, positions
    ):
        patch = patch.resize(size, RESAMPLE)
        image.paste(patch, position, patch)
        box = [
            position[0],
            position[1],
            position[0] + size[0],
            position[1] + size[1],
        ]
        if segment.role == "shared":
            ImageDraw.Draw(mask).rectangle(box, fill=255)
        metadata.append(
            {
                "text": segment.text,
                "role": segment.role,
                "font": segment.font.name,
                "box": box,
            }
        )
    return image.convert("RGB"), mask, metadata


def nw_matrix(seq1: str, seq2: str) -> np.ndarray:
    matrix = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=np.int16)
    matrix[:, 0] = np.arange(len(seq1) + 1) * int(gapScore)
    matrix[0, :] = np.arange(len(seq2) + 1) * int(gapScore)
    for row in range(1, len(seq1) + 1):
        for col in range(1, len(seq2) + 1):
            diagonal = matrix[row - 1, col - 1] + (
                int(matchScore)
                if seq1[row - 1] == seq2[col - 1]
                else int(mismatchScore)
            )
            matrix[row, col] = max(
                diagonal,
                matrix[row - 1, col] + int(gapScore),
                matrix[row, col - 1] + int(gapScore),
            )
    return matrix[1:, 1:]


def select_mode(rng: random.Random, args: argparse.Namespace) -> str:
    total = (
        args.original_ratio
        + args.cross_injection_ratio
        + args.aligned_unaligned_ratio
    )
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Mode ratios must sum to 1.0, got {total}")
    value = rng.random()
    if value < args.original_ratio:
        return "original"
    if value < args.original_ratio + args.cross_injection_ratio:
        return "cross_injection"
    return "aligned_unaligned"


def output_dirs(root: Path) -> dict[str, Path]:
    paths = {
        name: root / name
        for name in (
            "images",
            "texts",
            "masks",
            "matrices",
            "similarity_matrices",
        )
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save(
    index: int,
    plan: PairPlan,
    rendered1,
    rendered2,
    paths,
    skip_matrices: bool,
) -> dict:
    image1, mask1, meta1 = rendered1
    image2, mask2, meta2 = rendered2
    image1.save(paths["images"] / f"img1_{index}.png")
    image2.save(paths["images"] / f"img2_{index}.png")
    mask1.save(paths["masks"] / f"mask1_{index}.png")
    mask2.save(paths["masks"] / f"mask2_{index}.png")
    (paths["texts"] / f"text1_{index}.txt").write_text(
        plan.line1.text, encoding="utf-8"
    )
    (paths["texts"] / f"text2_{index}.txt").write_text(
        plan.line2.text, encoding="utf-8"
    )
    seq1 = plan.line1.text.replace(" ", "")
    seq2 = plan.line2.text.replace(" ", "")
    if not skip_matrices:
        np.save(paths["matrices"] / f"scoreMatrix_{index}.npy", nw_matrix(seq1, seq2))
        equality = (
            np.asarray(list(seq1))[:, None] == np.asarray(list(seq2))[None, :]
        ).astype(np.uint8)
        np.save(
            paths["similarity_matrices"] / f"similarityMatrix_{index}.npy",
            equality,
        )
    return {
        "sample_index": index,
        "mode": plan.mode,
        "text1": plan.line1.text,
        "text2": plan.line2.text,
        "shared_text": plan.shared,
        "primary_font": plan.primary_font.name,
        "secondary_font": plan.secondary_font.name,
        "line1_segments": meta1,
        "line2_segments": meta2,
    }


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
    parser.add_argument("--font-size", type=int, default=90)
    parser.add_argument("--segment-gap-min", type=int, default=4)
    parser.add_argument("--segment-gap-max", type=int, default=10)
    parser.add_argument("--outer-margin", type=int, default=8)
    parser.add_argument("--vertical-margin", type=int, default=5)
    parser.add_argument("--target-fill-ratio", type=float, default=0.94)
    parser.add_argument("--max-text-chars", type=int, default=63)
    parser.add_argument("--original-ratio", type=float, default=0.25)
    parser.add_argument("--cross-injection-ratio", type=float, default=0.45)
    parser.add_argument("--aligned-unaligned-ratio", type=float, default=0.30)
    parser.add_argument("--mixed-font-injection-prob", type=float, default=0.30)
    parser.add_argument("--skip-matrices", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_font <= 0:
        raise ValueError("--samples-per-font must be positive")
    if not 0 < args.target_fill_ratio <= 1:
        raise ValueError("--target-fill-ratio must be in (0, 1]")
    if args.segment_gap_min < 0 or args.segment_gap_max < args.segment_gap_min:
        raise ValueError("Invalid segment gap range")
    root = args.output_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {root}; pass --overwrite"
        )
    root.mkdir(parents=True, exist_ok=True)
    paths = output_dirs(root)
    fonts = discover_fonts(args.font_dir, args.fonts, args.font_count)
    external = read_corpus(args.corpus)
    base_pool = tuple(dict.fromkeys((*BASE_PHRASES, *external)))
    context_pool = tuple(dict.fromkeys((*CONTEXT_PHRASES, *external)))
    rng = random.Random(args.seed)
    total = len(fonts) * args.samples_per_font
    counts = {"original": 0, "cross_injection": 0, "aligned_unaligned": 0}
    print("Selected fonts:", *(f"\n  - {font}" for font in fonts), sep="")
    with (root / "metadata.jsonl").open("w", encoding="utf-8") as manifest:
        index = 1
        for font_index, primary in enumerate(fonts):
            secondary = fonts[(font_index + 1) % len(fonts)]
            for _ in tqdm(range(args.samples_per_font), desc=primary.name):
                mode = select_mode(rng, args)
                plan = build_plan(
                    mode,
                    rng,
                    base_pool,
                    context_pool,
                    fonts,
                    primary,
                    secondary,
                    args.mixed_font_injection_prob,
                    args.max_text_chars,
                )
                record = save(
                    index,
                    plan,
                    render_line(plan.line1, rng, args),
                    render_line(plan.line2, rng, args),
                    paths,
                    args.skip_matrices,
                )
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[mode] += 1
                index += 1
    summary = {
        "total_samples": total,
        "samples_per_font": args.samples_per_font,
        "fonts": [font.name for font in fonts],
        "mode_counts": counts,
        "exact_transcripts": True,
        "segment_gap_range": [args.segment_gap_min, args.segment_gap_max],
        "target_fill_ratio": args.target_fill_ratio,
    }
    (root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
