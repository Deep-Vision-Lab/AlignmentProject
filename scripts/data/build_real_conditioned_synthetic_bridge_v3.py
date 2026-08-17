#!/usr/bin/env python3
"""Build RealSyntheticBridge V3 offline.

V3 keeps one genuine real manuscript line as the anchor. The synthetic positive is
a complete synthetic sentence containing 1..3 exact spans from the real transcript.
Negatives are complete synthetic sentences with no accepted word or configured
character n-gram shared with the anchor.

Rendering policy:
- black background, white Arabic text;
- fonts may vary word/chunk by word/chunk, but a font is used only if its cmap
  supports the exact shaped text that Pillow will draw;
- positive mask is white over shared x-regions and black elsewhere;
- generated lines must occupy most of the available horizontal canvas;
- only non-geometric appearance augmentation: blur/noise/brightness/contrast.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from real_unique_line_training import _all_unique_records, _positive_eval_pages
from scripts.data.build_real_conditioned_synthetic_bridge import (
    _anchor_id,
    _font_candidates,
    _measure,
    _read,
    _shape,
    _side,
    candidate_span_records,
    choose_nonoverlapping_shared,
    clean_render_text,
    compact,
    normalize_match_text,
    safe_negative,
)

DATASET_VERSION = 3
DATASET_REVISION = "v3.2-glyphsafe-fullwidth-8neg"


def has_bad_unicode(text: str) -> bool:
    """Reject replacement/control/private-use characters before they reach a font."""
    for ch in clean_render_text(text):
        if ch == "\ufffd":
            return True
        category = unicodedata.category(ch)
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            return True
    return False


@lru_cache(maxsize=None)
def font_codepoints(font_path: str) -> frozenset[int]:
    """Return all Unicode codepoints mapped by a font's Unicode cmaps."""
    font = TTFont(font_path, lazy=True)
    try:
        codepoints: set[int] = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                codepoints.update(int(cp) for cp in table.cmap.keys())
        return frozenset(codepoints)
    finally:
        font.close()


def font_supports_text(font_path: Path, text: str) -> bool:
    """Check the exact bidi/reshaped string, not just the logical Arabic string."""
    if not clean_render_text(text) or has_bad_unicode(text):
        return False
    try:
        visual = _shape(text)
    except Exception:
        return False
    cmap = font_codepoints(str(font_path.resolve()))
    return all(ch.isspace() or ord(ch) in cmap for ch in visual)


def supported_fonts(text: str, fonts: list[Path]) -> list[Path]:
    return [font for font in fonts if font_supports_text(font, text)]


def sentence_is_renderable(text: str, fonts: list[Path]) -> bool:
    """V3 defaults to per-word chunks, so require every word to have a safe font."""
    words = clean_render_text(text).split()
    return bool(words) and not has_bad_unicode(text) and all(supported_fonts(word, fonts) for word in words)


def chunk_words(words: list[str], rng: random.Random, max_chunk_words: int) -> list[str]:
    out: list[str] = []
    i = 0
    limit = max(1, int(max_chunk_words))
    while i < len(words):
        take = rng.randint(1, min(limit, len(words) - i))
        out.append(" ".join(words[i:i + take]))
        i += take
    return out


def sentence_segments(text: str, rng: random.Random, max_chunk_words: int) -> list[dict]:
    return [
        {"kind": "distractor", "text": chunk}
        for chunk in chunk_words(clean_render_text(text).split(), rng, max_chunk_words)
        if chunk.strip()
    ]


def positive_segments(base_sentence: str, shared: list[dict], rng: random.Random, max_chunk_words: int) -> list[dict]:
    """Keep the entire unrelated base sentence and insert ordered shared islands."""
    words = clean_render_text(base_sentence).split()
    if not shared or len(words) < len(shared):
        raise ValueError("Positive sentence needs a non-empty base and shared islands")
    positions = sorted(rng.sample(range(len(words) + 1), len(shared)))
    out: list[dict] = []
    cursor = 0
    for position, island in zip(positions, shared):
        if position > cursor:
            out.extend(
                {"kind": "distractor", "text": chunk}
                for chunk in chunk_words(words[cursor:position], rng, max_chunk_words)
            )
        out.append({"kind": "shared", "text": island["text"]})
        cursor = position
    if cursor < len(words):
        out.extend(
            {"kind": "distractor", "text": chunk}
            for chunk in chunk_words(words[cursor:], rng, max_chunk_words)
        )
    return out


def assign_fonts(segments: list[dict], fonts: list[Path], rng: random.Random) -> list[Path]:
    """Choose only fonts that can render the exact shaped segment without .notdef."""
    choices: list[list[Path]] = []
    for segment in segments:
        valid = supported_fonts(str(segment["text"]), fonts)
        if not valid:
            raise RuntimeError(
                f"No configured font can safely render segment {segment['text']!r}. "
                "The sentence should have been filtered before rendering."
            )
        choices.append(valid)
    chosen = [rng.choice(valid) for valid in choices]

    # Force actual mixed-font augmentation when there is a safe alternative.
    if len(fonts) > 1 and len(chosen) > 1 and len({p.name for p in chosen}) == 1:
        first = chosen[0]
        candidates = [
            (idx, [path for path in valid if path != first])
            for idx, valid in enumerate(choices)
            if any(path != first for path in valid)
        ]
        if candidates:
            idx, alternatives = rng.choice(candidates)
            chosen[idx] = rng.choice(alternatives)
    return chosen


def appearance_augment(image: Image.Image, rng: random.Random, args: argparse.Namespace) -> tuple[Image.Image, dict]:
    """Non-geometric augmentation only."""
    contrast = rng.uniform(args.contrast_min, args.contrast_max)
    brightness = rng.uniform(args.brightness_min, args.brightness_max)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    blur = 0.0
    if rng.random() < args.blur_prob:
        blur = rng.uniform(0.0, args.blur_max_radius)
        if blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(blur))

    sigma = 0.0
    noise_seed = None
    if rng.random() < args.noise_prob:
        sigma = rng.uniform(0.0, args.noise_sigma_max)
        if sigma > 0:
            noise_seed = rng.randrange(2**32 - 1)
            nrng = np.random.default_rng(noise_seed)
            arr = np.asarray(image, dtype=np.float32)
            arr += nrng.normal(0.0, sigma, arr.shape).astype(np.float32)
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    return image, {
        "geometric_transform": False,
        "contrast": round(contrast, 5),
        "brightness": round(brightness, 5),
        "blur_radius": round(blur, 5),
        "noise_sigma": round(sigma, 5),
        "noise_seed": noise_seed,
    }


def render_segments(
    segments: list[dict], fonts: list[Path], output: Path, mask_output: Path | None,
    rng: random.Random, args: argparse.Namespace,
) -> tuple[list[dict], int, dict]:
    """Render glyph-safe RTL chunks and enforce a mostly-filled horizontal line."""
    if not segments:
        raise ValueError("Cannot render an empty sentence")
    chosen = assign_fonts(segments, fonts, rng)
    gaps = [rng.randint(args.segment_gap_min_px, args.segment_gap_max_px) for _ in range(len(segments) - 1)]
    probe = Image.new("L", (args.width, args.height), 0)
    probe_draw = ImageDraw.Draw(probe)
    content_width = args.width - 2 * args.padding
    content_height = args.height - 2 * args.padding

    def measure_at(size: int):
        loaded_fonts = [ImageFont.truetype(str(path), size) for path in chosen]
        metrics = [_measure(probe_draw, segment["text"], font) for segment, font in zip(segments, loaded_fonts)]
        width = sum(item[3] for item in metrics) + sum(gaps)
        tallest = max(item[4] for item in metrics)
        return loaded_fonts, metrics, width, tallest

    size = min(int(args.font_size), int(args.max_font_size))
    loaded, measured, total_width, tallest = measure_at(size)
    while (total_width > content_width or tallest > content_height) and size > args.min_font_size:
        size -= 1
        loaded, measured, total_width, tallest = measure_at(size)
    if total_width > content_width or tallest > content_height:
        raise RuntimeError(f"Full sentence does not fit {args.width}px even at font size {size}: {total_width}px")

    # If a sentence is visually short, enlarge it until it fills the target width or
    # until the next size would no longer fit. This removes half-empty black lines.
    target_width = float(args.min_line_fill_ratio) * float(content_width)
    while total_width < target_width and size < args.max_font_size:
        next_loaded, next_measured, next_width, next_tallest = measure_at(size + 1)
        if next_width > content_width or next_tallest > content_height:
            break
        size += 1
        loaded, measured, total_width, tallest = next_loaded, next_measured, next_width, next_tallest

    fill_ratio = float(total_width) / float(max(1, content_width))
    if fill_ratio < float(args.min_line_fill_ratio):
        raise RuntimeError(
            f"Generated sentence is too sparse: fill_ratio={fill_ratio:.3f} < "
            f"{args.min_line_fill_ratio:.3f}. Increase sentence length instead of keeping a half-empty line."
        )

    image = Image.new("RGB", (args.width, args.height), (0, 0, 0))
    mask = Image.new("L", (args.width, args.height), 0) if mask_output else None
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask) if mask is not None else None
    x_right = args.width - args.padding
    rendered: list[dict] = []

    for i, (segment, metric, font_path, font) in enumerate(zip(segments, measured, chosen, loaded)):
        visual, left, top, text_width, text_height = metric
        x = int(x_right - text_width - left)
        y = int(max(args.padding, min(args.height - args.padding - text_height - top, (args.height - text_height) // 2 - top)))
        draw.text((x, y), visual, font=font, fill=(255, 255, 255))
        x0 = max(0, int(x + left))
        x1 = min(args.width, x0 + int(text_width))
        bbox = [x0, 0, x1, args.height]
        if segment["kind"] == "shared" and mask_draw is not None:
            mask_draw.rectangle((x0, 0, max(x0, x1 - 1), args.height - 1), fill=255)
        rendered.append({
            "kind": segment["kind"],
            "text": segment["text"],
            "bbox_px": bbox,
            "font": font_path.name,
            "font_size": size,
            "glyph_safe": True,
        })
        x_right = x0 - (gaps[i] if i < len(gaps) else 0)

    image, aug = appearance_augment(image, rng, args)
    aug["line_fill_ratio"] = round(fill_ratio, 5)
    aug["rendered_span_px"] = int(total_width)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    if mask is not None and mask_output is not None:
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_output)
    return rendered, size, aug


def phrase_pool(texts: list[str], fonts: list[Path]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for source_index, text in enumerate(texts):
        words = clean_render_text(text).split()
        for width in (1, 2):
            for start in range(max(0, len(words) - width + 1)):
                phrase = " ".join(words[start:start + width])
                if len(compact(phrase)) >= 2 and sentence_is_renderable(phrase, fonts):
                    out.append((source_index, phrase))
    return out


def safe_full_lines(
    anchor_index: int, anchor_text: str, texts: list[str], fonts: list[Path],
    rng: random.Random, args: argparse.Namespace,
) -> list[str]:
    order = list(range(len(texts)))
    rng.shuffle(order)
    out: list[str] = []
    seen: set[str] = set()
    for idx in order:
        if idx == anchor_index:
            continue
        text = clean_render_text(texts[idx])
        words = text.split()
        key = normalize_match_text(text)
        compact_len = len(compact(text))
        if key in seen or not args.sentence_min_words <= len(words) <= args.sentence_max_words:
            continue
        if compact_len < args.min_sentence_chars or compact_len > args.max_sentence_chars:
            continue
        if not sentence_is_renderable(text, fonts):
            continue
        if safe_negative(anchor_text, text, negative_ngram=args.negative_ngram,
                         min_overlap_word_chars=args.min_overlap_word_chars):
            out.append(text)
            seen.add(key)
    return out


def compose_safe_sentence(
    anchor_index: int, anchor_text: str, pool: list[tuple[int, str]], fonts: list[Path],
    forbidden: set[str], rng: random.Random, args: argparse.Namespace,
) -> str | None:
    """Fallback: compose a long, glyph-safe sentence from content-clean phrases."""
    if not pool:
        return None
    for _ in range(500):
        target = rng.randint(args.sentence_min_words, args.sentence_max_words)
        selected: list[str] = []
        word_count = 0
        start = rng.randrange(len(pool))
        for offset in range(len(pool)):
            source_index, phrase = pool[(start + offset) % len(pool)]
            if source_index == anchor_index or not sentence_is_renderable(phrase, fonts):
                continue
            count = len(phrase.split())
            if word_count + count > target:
                continue
            trial = " ".join([*selected, phrase]).strip()
            if len(compact(trial)) > args.max_sentence_chars:
                continue
            if not safe_negative(anchor_text, trial, negative_ngram=args.negative_ngram,
                                 min_overlap_word_chars=args.min_overlap_word_chars):
                continue
            selected.append(phrase)
            word_count += count
            if word_count >= target and len(compact(trial)) >= args.min_sentence_chars:
                break
        sentence = " ".join(selected).strip()
        key = normalize_match_text(sentence)
        if (
            word_count >= args.sentence_min_words
            and len(compact(sentence)) >= args.min_sentence_chars
            and key and key not in forbidden
            and sentence_is_renderable(sentence, fonts)
            and safe_negative(anchor_text, sentence, negative_ngram=args.negative_ngram,
                              min_overlap_word_chars=args.min_overlap_word_chars)
        ):
            return sentence
    return None


def choose_safe_sentences(
    anchor_index: int, anchor_text: str, texts: list[str], pool: list[tuple[int, str]],
    fonts: list[Path], needed: int, rng: random.Random, args: argparse.Namespace,
) -> list[str]:
    out = safe_full_lines(anchor_index, anchor_text, texts, fonts, rng, args)
    # Prefer longer natural full lines so the final canvas is not half-empty.
    out.sort(key=lambda value: len(compact(value)), reverse=True)
    out = out[:needed]
    seen = {normalize_match_text(x) for x in out}
    while len(out) < needed:
        sentence = compose_safe_sentence(anchor_index, anchor_text, pool, fonts, seen, rng, args)
        if sentence is None:
            break
        out.append(sentence)
        seen.add(normalize_match_text(sentence))
    return out


def build(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    for name in ("images", "texts", "masks"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    fonts = _font_candidates(args.fonts)
    valid_pages, test_pages = _positive_eval_pages(str(data_dir))
    heldout_pages = set(valid_pages) | set(test_pages)
    records = [r for r in _all_unique_records(str(data_dir)) if str(r["page_id"]) not in heldout_pages]
    records.sort(key=lambda r: str(r["image_path"]))
    if args.max_anchors > 0:
        records = records[:args.max_anchors]

    usable, texts, spans = [], [], []
    for record in records:
        text = _read(Path(record["text_path"]))
        candidates = candidate_span_records(
            text,
            min_chars=args.min_positive_chars,
            max_chars=args.max_phrase_chars,
            max_words=args.max_phrase_words,
        )
        candidates = [candidate for candidate in candidates if supported_fonts(candidate["text"], fonts)]
        if candidates:
            usable.append(record)
            texts.append(text)
            spans.append(candidates)
    if not usable:
        raise RuntimeError("No leakage-safe usable anchors with glyph-safe shared spans")

    rng = random.Random(args.seed)
    pool = phrase_pool(texts, fonts)
    rng.shuffle(pool)

    stats = {
        "dataset_version": DATASET_VERSION,
        "dataset_revision": DATASET_REVISION,
        "dataset_semantics": "full_sentence_multi_island_mixed_font_white_on_black_glyphsafe_fullwidth",
        "anchors_considered": len(records), "anchors_written": 0,
        "positive_rows": 0, "negative_rows": 0,
        "positive_shared_islands_1": 0, "positive_shared_islands_2": 0, "positive_shared_islands_3": 0,
        "positive_full_sentence_rows": 0, "mixed_font_positive_rows": 0, "mixed_font_negative_rows": 0,
        "negatives_per_anchor": args.negatives_per_anchor,
        "negative_ngram": args.negative_ngram,
        "min_overlap_word_chars": args.min_overlap_word_chars,
        "sentence_min_words": args.sentence_min_words,
        "sentence_max_words": args.sentence_max_words,
        "min_sentence_chars": args.min_sentence_chars,
        "max_sentence_chars": args.max_sentence_chars,
        "min_line_fill_ratio": args.min_line_fill_ratio,
        "image_polarity": "white_text_on_black_background",
        "font_mixing": "per_segment_glyph_safe",
        "font_validation": "fonttools_unicode_cmap_on_shaped_text",
        "appearance_augmentation": {
            "geometric": False, "blur_prob": args.blur_prob, "blur_max_radius": args.blur_max_radius,
            "noise_prob": args.noise_prob, "noise_sigma_max": args.noise_sigma_max,
            "contrast_range": [args.contrast_min, args.contrast_max],
            "brightness_range": [args.brightness_min, args.brightness_max],
        },
        "mask_semantics": "white=shared synthetic x-region; black=distractor/gap/background",
        "heldout_page_count": len(heldout_pages), "fonts": [p.name for p in fonts], "seed": args.seed,
    }

    manifest_path = output_dir / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for anchor_index, (record, anchor_text, span_records) in enumerate(zip(usable, texts, spans)):
            requested = rng.randint(1, min(3, args.max_shared_islands))
            shared = choose_nonoverlapping_shared(span_records, rng, requested)
            safe_sentences = choose_safe_sentences(
                anchor_index, anchor_text, texts, pool, fonts,
                1 + args.negatives_per_anchor, rng, args,
            )
            if len(safe_sentences) < 1 + args.negatives_per_anchor:
                raise RuntimeError(
                    f"Could not construct enough complete glyph-safe content-clean sentences for {record['image_path']} "
                    f"({len(safe_sentences)}/{1 + args.negatives_per_anchor}); guarantee was not weakened"
                )
            base_sentence, negative_texts = safe_sentences[0], safe_sentences[1:]
            segments = positive_segments(base_sentence, shared, rng, args.max_font_chunk_words)
            positive_text = " ".join(s["text"] for s in segments).strip()

            anchor_id = _anchor_id(record)
            pair_id = f"bridge_{anchor_id}"
            pos_image_rel = Path("images") / f"{anchor_id}_pos_00.png"
            pos_text_rel = Path("texts") / f"{anchor_id}_pos_00.txt"
            pos_mask_rel = Path("masks") / f"{anchor_id}_pos_00_mask.png"
            rendered, effective_size, pos_aug = render_segments(
                segments, fonts, output_dir / pos_image_rel, output_dir / pos_mask_rel, rng, args
            )
            (output_dir / pos_text_rel).write_text(clean_render_text(positive_text), encoding="utf-8")
            shared_boxes = [s["bbox_px"] for s in rendered if s["kind"] == "shared"]
            shared_texts = [s["text"] for s in rendered if s["kind"] == "shared"]
            pos_fonts = sorted({s["font"] for s in rendered})
            shared_chars = sum(len(compact(t)) for t in shared_texts)
            anchor_chars = max(1, len(compact(anchor_text)))
            positive_chars = max(1, len(compact(positive_text)))

            row = {
                "pair_id": pair_id, "label_type": "medium_match",
                "A_page_id": str(record["page_id"]), "B_page_id": f"synthetic:{pair_id}",
                "A": _side(str(Path(record["image_path"]).resolve()), str(Path(record["text_path"]).resolve())),
                "B": _side(pos_image_rel.as_posix(), pos_text_rel.as_posix(), mask_path=pos_mask_rel.as_posix()),
                "scores": {"text_score": 1.0, "avg_sim": 1.0,
                           "coverage_A": min(1.0, shared_chars / anchor_chars),
                           "coverage_B": min(1.0, shared_chars / positive_chars)},
                "bridge": {
                    "dataset_version": DATASET_VERSION, "dataset_revision": DATASET_REVISION,
                    "relation": "positive_full_sentence_multi_island",
                    "anchor_id": anchor_id, "base_sentence": base_sentence,
                    "positive_full_sentence": positive_text, "shared_island_count": len(shared),
                    "shared_texts": shared_texts, "shared_boxes_px": shared_boxes, "segments": rendered,
                    "alignment_mask_path": pos_mask_rel.as_posix(), "fonts": pos_fonts,
                    "font_size": effective_size, "image_polarity": "white_text_on_black_background",
                    "appearance_augmentation": pos_aug, "glyph_safe": True,
                    "negative_ngram_guarantee": args.negative_ngram,
                    "min_overlap_word_chars": args.min_overlap_word_chars,
                },
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["positive_rows"] += 1
            stats["positive_full_sentence_rows"] += 1
            stats[f"positive_shared_islands_{len(shared)}"] += 1
            if len(pos_fonts) > 1:
                stats["mixed_font_positive_rows"] += 1

            for neg_index, negative_text in enumerate(negative_texts):
                neg_segments = sentence_segments(negative_text, rng, args.max_font_chunk_words)
                image_rel = Path("images") / f"{anchor_id}_neg_{neg_index:02d}.png"
                text_rel = Path("texts") / f"{anchor_id}_neg_{neg_index:02d}.txt"
                rendered_neg, neg_size, neg_aug = render_segments(
                    neg_segments, fonts, output_dir / image_rel, None, rng, args
                )
                (output_dir / text_rel).write_text(clean_render_text(negative_text), encoding="utf-8")
                neg_fonts = sorted({s["font"] for s in rendered_neg})
                neg_row = {
                    "pair_id": pair_id, "label_type": "no_shared_content",
                    "A_page_id": str(record["page_id"]), "B_page_id": f"synthetic:{pair_id}:neg{neg_index}",
                    "A": _side(str(Path(record["image_path"]).resolve()), str(Path(record["text_path"]).resolve())),
                    "B": _side(image_rel.as_posix(), text_rel.as_posix()),
                    "scores": {"text_score": 0.0, "avg_sim": 0.0, "coverage_A": 0.0, "coverage_B": 0.0},
                    "bridge": {
                        "dataset_version": DATASET_VERSION, "dataset_revision": DATASET_REVISION,
                        "relation": "negative_full_sentence_no_shared_content",
                        "anchor_id": anchor_id, "negative_text": negative_text, "segments": rendered_neg,
                        "fonts": neg_fonts, "font_size": neg_size,
                        "image_polarity": "white_text_on_black_background",
                        "appearance_augmentation": neg_aug, "glyph_safe": True,
                        "negative_ngram_guarantee": args.negative_ngram,
                        "min_overlap_word_chars": args.min_overlap_word_chars,
                    },
                }
                manifest.write(json.dumps(neg_row, ensure_ascii=False) + "\n")
                stats["negative_rows"] += 1
                if len(neg_fonts) > 1:
                    stats["mixed_font_negative_rows"] += 1

            stats["anchors_written"] += 1
            if (anchor_index + 1) % 100 == 0:
                print(f"bridge_v3 anchors={anchor_index + 1}/{len(usable)}", flush=True)

    (output_dir / "metadata.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=== REAL-SYNTHETIC BRIDGE V3 ===")
    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"output={output_dir}")
    print(f"manifest={manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(PROJECT_DIR / "DataSet" / "ArabicDataset"))
    p.add_argument("--output-dir", default=str(PROJECT_DIR / "DataSet" / "RealSyntheticBridge_v3"))
    p.add_argument("--negatives-per-anchor", type=int, default=8)
    p.add_argument("--negative-ngram", type=int, default=3)
    p.add_argument("--min-overlap-word-chars", type=int, default=1)
    p.add_argument("--max-shared-islands", type=int, default=3)
    p.add_argument("--min-positive-chars", type=int, default=4)
    p.add_argument("--max-phrase-chars", type=int, default=24)
    p.add_argument("--max-phrase-words", type=int, default=2)
    p.add_argument("--sentence-min-words", type=int, default=8)
    p.add_argument("--sentence-max-words", type=int, default=16)
    p.add_argument("--min-sentence-chars", type=int, default=36)
    p.add_argument("--max-sentence-chars", type=int, default=130)
    p.add_argument("--max-font-chunk-words", type=int, default=1)
    p.add_argument("--fonts", default="")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--font-size", type=int, default=56)
    p.add_argument("--min-font-size", type=int, default=14)
    p.add_argument("--max-font-size", type=int, default=88)
    p.add_argument("--min-line-fill-ratio", type=float, default=0.75)
    p.add_argument("--padding", type=int, default=16)
    p.add_argument("--segment-gap-min-px", type=int, default=8)
    p.add_argument("--segment-gap-max-px", type=int, default=20)
    p.add_argument("--blur-prob", type=float, default=0.65)
    p.add_argument("--blur-max-radius", type=float, default=1.15)
    p.add_argument("--noise-prob", type=float, default=0.80)
    p.add_argument("--noise-sigma-max", type=float, default=9.0)
    p.add_argument("--contrast-min", type=float, default=0.88)
    p.add_argument("--contrast-max", type=float, default=1.14)
    p.add_argument("--brightness-min", type=float, default=0.90)
    p.add_argument("--brightness-max", type=float, default=1.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-anchors", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.negatives_per_anchor <= 0:
        p.error("--negatives-per-anchor must be positive")
    if args.negative_ngram < 2:
        p.error("--negative-ngram must be >=2")
    if not 1 <= args.max_shared_islands <= 3:
        p.error("--max-shared-islands must be 1..3")
    if args.sentence_min_words < 2 or args.sentence_max_words < args.sentence_min_words:
        p.error("invalid sentence word range")
    if args.min_sentence_chars < 8 or args.max_sentence_chars < args.min_sentence_chars:
        p.error("invalid sentence character range")
    if not 0.50 <= args.min_line_fill_ratio <= 0.95:
        p.error("--min-line-fill-ratio must be between 0.50 and 0.95")
    if args.max_font_size < args.font_size or args.font_size < args.min_font_size:
        p.error("invalid font-size range")
    if not 0 <= args.blur_prob <= 1 or not 0 <= args.noise_prob <= 1:
        p.error("probabilities must be in [0,1]")
    return args


if __name__ == "__main__":
    build(parse_args())
