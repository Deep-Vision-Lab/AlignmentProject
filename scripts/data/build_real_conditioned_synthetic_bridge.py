#!/usr/bin/env python3
"""Build the offline real-conditioned synthetic bridge dataset.

For every leakage-safe real manuscript line used as an anchor, this builder writes:

* one composite POSITIVE synthetic line containing 1..N exact shared islands from
  the real transcript, in their original order, separated by guaranteed-unrelated
  synthetic distractor text; and
* K NEGATIVE synthetic lines whose normalized text shares neither a complete word
  nor a configured character n-gram with the real anchor.

The positive synthetic line also gets a grayscale alignment mask with the same
``width x height`` as the rendered line. Pixel columns occupied by shared islands
are white (255); distractors, gaps, and the background are black (0). Exact shared
boxes and segment metadata are persisted in the manifest for audit/debugging.

All Arabic shaping/font rendering happens once here on CPU. No synthetic rendering
is performed in the GPU training loop. Validation/test real pages are excluded
before generation to avoid handwriting leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from real_unique_line_training import _all_unique_records, _positive_eval_pages


_ALEF_MAP = str.maketrans(
    {
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي",
        "ؤ": "و", "ئ": "ي", "ة": "ه", "ـ": "",
    }
)


def clean_render_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("ـ", "")
    return " ".join(text.split()).strip()


def normalize_match_text(text: str) -> str:
    text = clean_render_text(text).translate(_ALEF_MAP)
    text = re.sub(r"[^\u0600-\u06FF0-9 ]+", " ", text)
    return " ".join(text.split()).strip()


def compact(text: str) -> str:
    return normalize_match_text(text).replace(" ", "")


def ngrams(text: str, n: int) -> set[str]:
    value = compact(text)
    if n <= 0 or len(value) < n:
        return set()
    return {value[i:i+n] for i in range(len(value) - n + 1)}


def meaningful_words(text: str, min_chars: int) -> set[str]:
    return {word for word in normalize_match_text(text).split() if len(word) >= int(min_chars)}


def safe_negative(anchor_text: str, candidate_text: str, *, negative_ngram: int, min_overlap_word_chars: int) -> bool:
    """True only when candidate has no legitimate word/sequence alignment with anchor."""
    anchor_norm = normalize_match_text(anchor_text)
    candidate_norm = normalize_match_text(candidate_text)
    if not anchor_norm or not candidate_norm or candidate_norm == anchor_norm:
        return False
    if meaningful_words(anchor_norm, min_overlap_word_chars) & meaningful_words(candidate_norm, min_overlap_word_chars):
        return False
    if ngrams(anchor_norm, negative_ngram) & ngrams(candidate_norm, negative_ngram):
        return False
    return True


def candidate_span_records(text: str, *, min_chars: int, max_chars: int, max_words: int) -> list[dict]:
    """Exact contiguous anchor spans with logical word coordinates."""
    clean = clean_render_text(text)
    words = clean.split()
    records: list[dict] = []
    for span_width in range(1, max(1, int(max_words)) + 1):
        for start in range(0, max(0, len(words) - span_width + 1)):
            phrase = " ".join(words[start:start + span_width]).strip()
            length = len(compact(phrase))
            if int(min_chars) <= length <= int(max_chars):
                records.append({"text": phrase, "word_start": start, "word_end": start + span_width})
    if records:
        records.sort(key=lambda item: (item["word_end"] - item["word_start"], len(compact(item["text"]))), reverse=True)
        return records
    chars = clean.replace(" ", "")
    if len(chars) >= int(min_chars):
        width = min(int(max_chars), len(chars))
        return [{"text": chars[:width], "word_start": 0, "word_end": 1}]
    return []


def candidate_spans(text: str, **kwargs) -> list[str]:
    return [item["text"] for item in candidate_span_records(text, **kwargs)]


def choose_nonoverlapping_shared(span_records: list[dict], rng: random.Random, requested_count: int) -> list[dict]:
    candidates = list(span_records)
    rng.shuffle(candidates)
    chosen: list[dict] = []
    for candidate in candidates:
        start, end = int(candidate["word_start"]), int(candidate["word_end"])
        if any(start < int(item["word_end"]) and int(item["word_start"]) < end for item in chosen):
            continue
        chosen.append(dict(candidate))
        if len(chosen) >= max(1, int(requested_count)):
            break
    if not chosen:
        chosen = [dict(span_records[0])]
    chosen.sort(key=lambda item: (int(item["word_start"]), int(item["word_end"])))
    return chosen


def _font_candidates(fonts_arg: str) -> list[Path]:
    if fonts_arg.strip():
        paths = []
        for raw in fonts_arg.split(","):
            path = Path(raw.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_DIR / path
            if not path.is_file():
                raise FileNotFoundError(f"Requested font does not exist: {path}")
            paths.append(path.resolve())
        if paths:
            return paths
    fonts_dir = PROJECT_DIR / "Fonts"
    discovered = sorted([*fonts_dir.glob("*.ttf"), *fonts_dir.glob("*.otf"), *fonts_dir.glob("*.TTF")])
    if not discovered:
        raise FileNotFoundError(f"No Arabic fonts found under {fonts_dir}; pass --fonts path1.ttf,path2.ttf")
    return [path.resolve() for path in discovered]


def _shape(text: str) -> str:
    clean = clean_render_text(text)
    reshaper = arabic_reshaper.ArabicReshaper(configuration={
        "delete_harakat": True,
        "support_zwj": False,
        "delete_at_sign": True,
        "use_unshaped_instead_of_isolated": True,
    })
    visual = get_display(reshaper.reshape(clean))
    visual = "".join(ch for ch in visual if ch.isprintable() and ord(ch) != 0x200C)
    if not visual:
        raise ValueError(f"Arabic renderer produced empty output for {text!r}")
    return visual


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    visual = _shape(text)
    left, top, right, bottom = draw.textbbox((0, 0), visual, font=font)
    return visual, left, top, max(1, right - left), max(1, bottom - top)


def render_arabic_line(text: str, font_path: Path, output_path: Path, *, width: int, height: int, font_size: int, padding: int) -> None:
    visual = _shape(text)
    size = int(font_size)
    while True:
        font = ImageFont.truetype(str(font_path), size)
        probe = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(probe)
        left, top, right, bottom = draw.textbbox((0, 0), visual, font=font)
        text_width, text_height = max(1, right - left), max(1, bottom - top)
        if (text_width <= width - 2 * padding and text_height <= height - 2 * padding) or size <= 16:
            break
        size -= 2
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    x = max(padding, width - padding - text_width - left)
    y = max(padding, (height - text_height) // 2 - top)
    draw.text((x, y), visual, font=font, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_segmented_positive(
    segments: list[dict], font_path: Path, output_path: Path, mask_path: Path,
    *, width: int, height: int, font_size: int, padding: int,
    gap_min_px: int, gap_max_px: int, rng: random.Random,
) -> tuple[list[dict], int]:
    """Render logical RTL segments plus a full-height shared-region mask."""
    if not segments:
        raise ValueError("Cannot render an empty segmented positive")
    size = int(font_size)
    gap_min_px = max(2, int(gap_min_px))
    gap_max_px = max(gap_min_px, int(gap_max_px))
    gaps = [rng.randint(gap_min_px, gap_max_px) for _ in range(max(0, len(segments) - 1))]
    probe = Image.new("L", (width, height), 255)
    probe_draw = ImageDraw.Draw(probe)
    measurements = []
    while True:
        font = ImageFont.truetype(str(font_path), size)
        measurements = [_measure(probe_draw, segment["text"], font) for segment in segments]
        total_width = sum(item[3] for item in measurements) + sum(gaps)
        tallest = max(item[4] for item in measurements)
        if (total_width <= width - 2 * padding and tallest <= height - 2 * padding) or size <= 16:
            break
        size -= 2
    total_width = sum(item[3] for item in measurements) + sum(gaps)
    if total_width > width - 2 * padding:
        raise RuntimeError(f"Composite positive does not fit {width}px even at font size {size}: {total_width}px")

    image = Image.new("RGB", (width, height), (255, 255, 255))
    mask = Image.new("L", (width, height), 0)
    draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    x_right = width - padding
    rendered: list[dict] = []
    for index, (segment, measured) in enumerate(zip(segments, measurements)):
        visual, left, top, text_width, text_height = measured
        x = int(x_right - text_width - left)
        jitter = rng.randint(-2, 2)
        y = int(max(padding, min(height - padding - text_height - top, (height - text_height) // 2 - top + jitter)))
        draw.text((x, y), visual, font=font, fill=(0, 0, 0))
        x0 = max(0, int(x + left))
        x1 = min(width, int(x0 + text_width))
        bbox = [x0, 0, x1, height]
        if segment["kind"] == "shared":
            mask_draw.rectangle((x0, 0, max(x0, x1 - 1), height - 1), fill=255)
        rendered.append({"kind": str(segment["kind"]), "text": str(segment["text"]), "bbox_px": bbox})
        x_right = x0 - (gaps[index] if index < len(gaps) else 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    mask.save(mask_path)
    return rendered, size


def _read(path: Path) -> str:
    return clean_render_text(path.read_text(encoding="utf-8"))


def _anchor_id(record: dict) -> str:
    payload = f"{record['image_path']}|{record['text_path']}|{record['page_id']}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _side(image_path: str, text_path: str, line_idx: int = -1, mask_path: str | None = None) -> dict:
    result = {"line_image_path": image_path, "text_original_path": text_path, "line_idx": int(line_idx)}
    if mask_path:
        result["alignment_mask_path"] = mask_path
    return result


def _safe_phrase_pool(
    anchor_index: int, anchor_text: str, negative_pool: list[tuple[int, str]], *, rng: random.Random,
    negative_ngram: int, min_overlap_word_chars: int, needed: int,
) -> list[str]:
    selected, seen = [], set()
    if not negative_pool:
        return selected
    start = rng.randrange(len(negative_pool))
    for offset in range(len(negative_pool)):
        source_index, phrase = negative_pool[(start + offset) % len(negative_pool)]
        if int(source_index) == int(anchor_index):
            continue
        key = normalize_match_text(phrase)
        if not key or key in seen:
            continue
        if not safe_negative(anchor_text, phrase, negative_ngram=negative_ngram, min_overlap_word_chars=min_overlap_word_chars):
            continue
        selected.append(phrase)
        seen.add(key)
        if len(selected) >= int(needed):
            break
    return selected


def _positive_layout(shared: list[dict], distractors: list[str], rng: random.Random) -> list[dict]:
    """Shared islands remain in real-line order and are separated by distractors."""
    if not shared:
        raise ValueError("Positive layout requires at least one shared island")
    segments: list[dict] = []
    distractor_index = 0
    if len(shared) == 1:
        distractor = distractors[0]
        pair = [{"kind": "shared", "text": shared[0]["text"]}, {"kind": "distractor", "text": distractor}]
        if rng.random() < 0.5:
            pair.reverse()
        return pair
    if rng.random() < 0.5:
        segments.append({"kind": "distractor", "text": distractors[distractor_index]})
        distractor_index += 1
    for shared_index, item in enumerate(shared):
        segments.append({"kind": "shared", "text": item["text"]})
        if shared_index + 1 < len(shared):
            segments.append({"kind": "distractor", "text": distractors[distractor_index]})
            distractor_index += 1
    if distractor_index < len(distractors) and rng.random() < 0.5:
        segments.append({"kind": "distractor", "text": distractors[distractor_index]})
    return segments


def build(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to rebuild it.")
        shutil.rmtree(output_dir)
    for dirname in ("images", "texts", "masks"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    eval_valid_pages, eval_test_pages = _positive_eval_pages(str(data_dir))
    heldout_pages = set(eval_valid_pages) | set(eval_test_pages)
    records = [record for record in _all_unique_records(str(data_dir)) if str(record["page_id"]) not in heldout_pages]
    records.sort(key=lambda item: str(item["image_path"]))
    if args.max_anchors > 0:
        records = records[:args.max_anchors]
    if not records:
        raise RuntimeError("No leakage-safe real anchors were found.")

    font_paths = _font_candidates(args.fonts)
    rng = random.Random(args.seed)
    usable_records, anchor_texts, span_pools = [], [], []
    for record in records:
        text = _read(Path(record["text_path"]))
        spans = candidate_span_records(text, min_chars=args.min_positive_chars, max_chars=args.max_phrase_chars, max_words=args.max_phrase_words)
        if spans:
            usable_records.append(record)
            anchor_texts.append(text)
            span_pools.append(spans)

    negative_pool: list[tuple[int, str]] = []
    for source_index, spans in enumerate(span_pools):
        for item in spans[:max(6, args.pool_spans_per_anchor)]:
            negative_pool.append((source_index, item["text"]))
    rng.shuffle(negative_pool)

    stats = {
        "dataset_version": 2,
        "anchors_considered": len(records), "anchors_written": 0,
        "positive_rows": 0, "negative_rows": 0,
        "positive_shared_islands_1": 0, "positive_shared_islands_2": 0, "positive_shared_islands_3": 0,
        "max_shared_islands": int(args.max_shared_islands),
        "negative_ngram": int(args.negative_ngram),
        "min_overlap_word_chars": int(args.min_overlap_word_chars),
        "negatives_per_anchor": int(args.negatives_per_anchor),
        "mask_semantics": "white=shared synthetic x-region; black=distractor/gap/background",
        "heldout_page_count": len(heldout_pages), "fonts": [path.name for path in font_paths], "seed": int(args.seed),
    }

    manifest_path = output_dir / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for anchor_index, (record, anchor_text, spans) in enumerate(zip(usable_records, anchor_texts, span_pools)):
            requested = rng.randint(1, max(1, min(3, int(args.max_shared_islands))))
            shared = choose_nonoverlapping_shared(spans, rng, requested)
            shared_count = len(shared)
            distractor_need = max(2, shared_count + 1)
            safe_phrases = _safe_phrase_pool(
                anchor_index, anchor_text, negative_pool, rng=rng,
                negative_ngram=args.negative_ngram, min_overlap_word_chars=args.min_overlap_word_chars,
                needed=distractor_need + args.negatives_per_anchor,
            )
            required = distractor_need + args.negatives_per_anchor
            if len(safe_phrases) < required:
                raise RuntimeError(
                    f"Could not find enough content-clean phrases for {record['image_path']} "
                    f"(found {len(safe_phrases)}/{required}). Do not silently weaken the guarantee."
                )
            segments = _positive_layout(shared, safe_phrases[:distractor_need], rng)
            negative_texts = safe_phrases[distractor_need:distractor_need + args.negatives_per_anchor]
            positive_text = " ".join(segment["text"] for segment in segments).strip()

            anchor_id = _anchor_id(record)
            pair_id = f"bridge_{anchor_id}"
            real_image, real_text = str(Path(record["image_path"]).resolve()), str(Path(record["text_path"]).resolve())
            page_id = str(record["page_id"])
            font = font_paths[anchor_index % len(font_paths)]

            pos_stem = f"{anchor_id}_pos_00"
            pos_image_rel = Path("images") / f"{pos_stem}.png"
            pos_text_rel = Path("texts") / f"{pos_stem}.txt"
            pos_mask_rel = Path("masks") / f"{pos_stem}_mask.png"
            rendered_segments, effective_font_size = render_segmented_positive(
                segments, font, output_dir / pos_image_rel, output_dir / pos_mask_rel,
                width=args.width, height=args.height, font_size=args.font_size, padding=args.padding,
                gap_min_px=args.segment_gap_min_px, gap_max_px=args.segment_gap_max_px, rng=rng,
            )
            (output_dir / pos_text_rel).write_text(clean_render_text(positive_text), encoding="utf-8")
            shared_boxes = [segment["bbox_px"] for segment in rendered_segments if segment["kind"] == "shared"]
            shared_texts = [segment["text"] for segment in rendered_segments if segment["kind"] == "shared"]
            shared_chars = sum(len(compact(text)) for text in shared_texts)
            positive_chars, anchor_chars = max(1, len(compact(positive_text))), max(1, len(compact(anchor_text)))

            pos_row = {
                "pair_id": pair_id, "label_type": "medium_match",
                "A_page_id": page_id, "B_page_id": f"synthetic:{pair_id}",
                "A": _side(real_image, real_text),
                "B": _side(pos_image_rel.as_posix(), pos_text_rel.as_posix(), mask_path=pos_mask_rel.as_posix()),
                "scores": {"text_score": 1.0, "avg_sim": 1.0, "coverage_A": min(1.0, shared_chars / anchor_chars), "coverage_B": min(1.0, shared_chars / positive_chars)},
                "bridge": {
                    "dataset_version": 2, "relation": "positive_multi_island", "anchor_id": anchor_id,
                    "shared_island_count": shared_count, "shared_texts": shared_texts,
                    "shared_boxes_px": shared_boxes, "segments": rendered_segments,
                    "alignment_mask_path": pos_mask_rel.as_posix(), "font": font.name,
                    "font_size": effective_font_size, "negative_ngram_guarantee": args.negative_ngram,
                    "min_overlap_word_chars": args.min_overlap_word_chars,
                },
            }
            manifest.write(json.dumps(pos_row, ensure_ascii=False) + "\n")
            stats["positive_rows"] += 1
            stats[f"positive_shared_islands_{shared_count}"] += 1

            for neg_index, negative_text in enumerate(negative_texts):
                assert safe_negative(anchor_text, negative_text, negative_ngram=args.negative_ngram, min_overlap_word_chars=args.min_overlap_word_chars)
                stem = f"{anchor_id}_neg_{neg_index:02d}"
                image_rel, text_rel = Path("images") / f"{stem}.png", Path("texts") / f"{stem}.txt"
                neg_font = font_paths[(anchor_index + neg_index + 1) % len(font_paths)]
                render_arabic_line(negative_text, neg_font, output_dir / image_rel, width=args.width, height=args.height, font_size=args.font_size, padding=args.padding)
                (output_dir / text_rel).write_text(clean_render_text(negative_text), encoding="utf-8")
                row = {
                    "pair_id": pair_id, "label_type": "no_shared_content",
                    "A_page_id": page_id, "B_page_id": f"synthetic:{pair_id}:neg{neg_index}",
                    "A": _side(real_image, real_text), "B": _side(image_rel.as_posix(), text_rel.as_posix()),
                    "scores": {"text_score": 0.0, "avg_sim": 0.0, "coverage_A": 0.0, "coverage_B": 0.0},
                    "bridge": {
                        "dataset_version": 2, "relation": "negative_no_shared_content", "anchor_id": anchor_id,
                        "negative_text": negative_text, "font": neg_font.name,
                        "negative_ngram_guarantee": args.negative_ngram,
                        "min_overlap_word_chars": args.min_overlap_word_chars,
                    },
                }
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["negative_rows"] += 1
            stats["anchors_written"] += 1
            if (anchor_index + 1) % 100 == 0:
                print(f"bridge_build anchors={anchor_index + 1}/{len(usable_records)} rows={stats['positive_rows'] + stats['negative_rows']}", flush=True)

    (output_dir / "metadata.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=== REAL-CONDITIONED SYNTHETIC BRIDGE V2 ===")
    print(f"output={output_dir}")
    print(f"manifest={manifest_path}")
    for key, value in stats.items():
        print(f"{key}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "DataSet" / "ArabicDataset"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "DataSet" / "RealSyntheticBridge_v2"))
    parser.add_argument("--negatives-per-anchor", type=int, default=4)
    parser.add_argument("--negative-ngram", type=int, default=3)
    parser.add_argument("--min-overlap-word-chars", type=int, default=1)
    parser.add_argument("--max-shared-islands", type=int, default=3)
    parser.add_argument("--min-positive-chars", type=int, default=4)
    parser.add_argument("--max-phrase-chars", type=int, default=24)
    parser.add_argument("--max-phrase-words", type=int, default=2)
    parser.add_argument("--pool-spans-per-anchor", type=int, default=16)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--fonts", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--font-size", type=int, default=58)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--segment-gap-min-px", type=int, default=12)
    parser.add_argument("--segment-gap-max-px", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.negatives_per_anchor <= 0:
        parser.error("--negatives-per-anchor must be positive")
    if args.negative_ngram < 2:
        parser.error("--negative-ngram must be >= 2")
    if not 1 <= args.max_shared_islands <= 3:
        parser.error("--max-shared-islands must be between 1 and 3")
    if args.min_overlap_word_chars < 1:
        parser.error("--min-overlap-word-chars must be >= 1")
    return args


if __name__ == "__main__":
    build(parse_args())
