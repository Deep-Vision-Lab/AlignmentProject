#!/usr/bin/env python3
"""Build an offline real-conditioned synthetic bridge dataset.

For every leakage-safe real manuscript line used as an anchor, this builder writes:

* one positive synthetic line containing an exact contiguous span from the real
  transcript; and
* K negative synthetic lines whose normalized text is guaranteed not to share a
  word (>= ``--min-overlap-word-chars``) or character n-gram (``--negative-ngram``)
  with the full real anchor transcript.

The expensive Arabic shaping/font rendering happens once here, never in the GPU
training loop.  The output follows the normal real-manifest schema, so existing
image/text preprocessing, Span-DTW code, and pair objectives can reuse it.

Only canonical TRAIN-SAFE real anchors are used. Pages assigned to the existing
positive-pair validation/test split are excluded before any synthetic data is
created, preventing the bridge corpus from leaking evaluation handwriting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Reuse the exact page split and physical-line de-duplication used by R0.
import sys

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from real_unique_line_training import _all_unique_records, _positive_eval_pages


_ALEF_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }
)


def clean_render_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("ـ", "")
    return " ".join(text.split()).strip()


def normalize_match_text(text: str) -> str:
    text = clean_render_text(text).translate(_ALEF_MAP)
    # Keep Arabic letters/digits and spaces only.  Punctuation should not make a
    # negative look artificially distinct from the anchor.
    text = re.sub(r"[^\u0600-\u06FF0-9 ]+", " ", text)
    return " ".join(text.split()).strip()


def compact(text: str) -> str:
    return normalize_match_text(text).replace(" ", "")


def ngrams(text: str, n: int) -> set[str]:
    value = compact(text)
    if n <= 0 or len(value) < n:
        return set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def meaningful_words(text: str, min_chars: int) -> set[str]:
    return {
        word
        for word in normalize_match_text(text).split()
        if len(word) >= int(min_chars)
    }


def candidate_spans(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    max_words: int,
) -> list[str]:
    """Return renderable contiguous word spans, preferring actual word boundaries."""
    clean = clean_render_text(text)
    words = clean.split()
    candidates: list[str] = []
    for width in range(1, max(1, int(max_words)) + 1):
        for start in range(0, max(0, len(words) - width + 1)):
            phrase = " ".join(words[start : start + width]).strip()
            length = len(compact(phrase))
            if int(min_chars) <= length <= int(max_chars):
                candidates.append(phrase)

    if candidates:
        # Stable de-duplication while preserving the longer-span preference below.
        unique = list(dict.fromkeys(candidates))
        unique.sort(key=lambda item: (len(item.split()), len(compact(item))), reverse=True)
        return unique

    # Fallback for pathological one-token transcripts: use a contiguous logical
    # character span, never an invented/reshuffled sequence.
    chars = clean.replace(" ", "")
    if len(chars) >= int(min_chars):
        width = min(int(max_chars), len(chars))
        return [chars[:width]]
    return []


def safe_negative(
    anchor_text: str,
    candidate_text: str,
    *,
    negative_ngram: int,
    min_overlap_word_chars: int,
) -> bool:
    anchor_norm = normalize_match_text(anchor_text)
    candidate_norm = normalize_match_text(candidate_text)
    if not anchor_norm or not candidate_norm:
        return False
    if candidate_norm == anchor_norm:
        return False
    anchor_words = meaningful_words(anchor_norm, min_overlap_word_chars)
    candidate_words = meaningful_words(candidate_norm, min_overlap_word_chars)
    if anchor_words & candidate_words:
        return False
    if ngrams(anchor_norm, negative_ngram) & ngrams(candidate_norm, negative_ngram):
        return False
    return True


def _font_candidates(fonts_arg: str) -> list[Path]:
    if fonts_arg.strip():
        paths = []
        for raw in fonts_arg.split(","):
            path = Path(raw.strip()).expanduser()
            if not path.is_absolute():
                path = PROJECT_DIR / path
            if path.is_file():
                paths.append(path.resolve())
            else:
                raise FileNotFoundError(f"Requested font does not exist: {path}")
        if paths:
            return paths

    fonts_dir = PROJECT_DIR / "Fonts"
    discovered = sorted(
        [*fonts_dir.glob("*.ttf"), *fonts_dir.glob("*.otf"), *fonts_dir.glob("*.TTF")]
    )
    if not discovered:
        raise FileNotFoundError(
            f"No Arabic fonts found under {fonts_dir}; pass --fonts path1.ttf,path2.ttf"
        )
    return [path.resolve() for path in discovered]


def render_arabic_line(
    text: str,
    font_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    font_size: int,
    padding: int,
) -> None:
    """Render black Arabic text on a white fixed-size line canvas."""
    clean = clean_render_text(text)
    reshaper = arabic_reshaper.ArabicReshaper(
        configuration={
            "delete_harakat": True,
            "support_zwj": False,
            "delete_at_sign": True,
            "use_unshaped_instead_of_isolated": True,
        }
    )
    visual = get_display(reshaper.reshape(clean))
    visual = "".join(ch for ch in visual if ch.isprintable() and ord(ch) != 0x200C)
    if not visual:
        raise ValueError(f"Arabic renderer produced empty output for {text!r}")

    # Reduce font size only when necessary.  Keeping the same 128x1024 canvas as
    # real training avoids any rendering work in __getitem__.
    size = int(font_size)
    while True:
        font = ImageFont.truetype(str(font_path), size)
        probe = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(probe)
        left, top, right, bottom = draw.textbbox((0, 0), visual, font=font)
        text_width = max(1, right - left)
        text_height = max(1, bottom - top)
        if (
            text_width <= width - 2 * padding
            and text_height <= height - 2 * padding
        ) or size <= 16:
            break
        size -= 2

    image = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # get_display() produces visual RTL order; right-align it like a manuscript line.
    x = max(padding, width - padding - text_width - left)
    y = max(padding, (height - text_height) // 2 - top)
    draw.text((x, y), visual, font=font, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _read(path: Path) -> str:
    return clean_render_text(path.read_text(encoding="utf-8"))


def _anchor_id(record: dict) -> str:
    payload = f"{record['image_path']}|{record['text_path']}|{record['page_id']}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _side(image_path: str, text_path: str, line_idx: int = -1) -> dict:
    return {
        "line_image_path": image_path,
        "text_original_path": text_path,
        "line_idx": int(line_idx),
    }


def build(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    texts_dir = output_dir / "texts"
    images_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    eval_valid_pages, eval_test_pages = _positive_eval_pages(str(data_dir))
    heldout_pages = set(eval_valid_pages) | set(eval_test_pages)
    records = [
        record
        for record in _all_unique_records(str(data_dir))
        if str(record["page_id"]) not in heldout_pages
    ]
    records.sort(key=lambda item: str(item["image_path"]))
    if args.max_anchors > 0:
        records = records[: args.max_anchors]
    if not records:
        raise RuntimeError("No leakage-safe real anchors were found.")

    font_paths = _font_candidates(args.fonts)
    rng = random.Random(args.seed)

    anchor_texts: list[str] = []
    span_pools: list[list[str]] = []
    usable_records: list[dict] = []
    for record in records:
        text = _read(Path(record["text_path"]))
        spans = candidate_spans(
            text,
            min_chars=args.min_positive_chars,
            max_chars=args.max_phrase_chars,
            max_words=args.max_phrase_words,
        )
        if not spans:
            continue
        usable_records.append(record)
        anchor_texts.append(text)
        span_pools.append(spans)

    # Global negative phrase pool.  Each item remembers which real line it came
    # from so a line can never use one of its own spans as a negative.
    negative_pool: list[tuple[int, str]] = []
    for source_index, spans in enumerate(span_pools):
        # Cap per source to keep candidate search bounded while retaining variety.
        for phrase in spans[: max(4, args.pool_spans_per_anchor)]:
            negative_pool.append((source_index, phrase))
    rng.shuffle(negative_pool)

    manifest_path = output_dir / "dataset_manifest.jsonl"
    stats = {
        "anchors_considered": len(records),
        "anchors_written": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "negative_ngram": int(args.negative_ngram),
        "min_overlap_word_chars": int(args.min_overlap_word_chars),
        "negatives_per_anchor": int(args.negatives_per_anchor),
        "heldout_page_count": len(heldout_pages),
        "fonts": [path.name for path in font_paths],
        "seed": int(args.seed),
    }

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for anchor_index, (record, anchor_text, spans) in enumerate(
            zip(usable_records, anchor_texts, span_pools)
        ):
            # Prefer a longer shared phrase when possible, but vary it deterministically.
            preferred = [span for span in spans if len(span.split()) >= 2] or spans
            positive_text = rng.choice(preferred[: min(len(preferred), 12)])

            negative_texts: list[str] = []
            seen_negatives: set[str] = set()
            # Randomized scan of a bounded pool makes generation deterministic and fast.
            start = rng.randrange(max(1, len(negative_pool)))
            for offset in range(len(negative_pool)):
                source_index, phrase = negative_pool[(start + offset) % len(negative_pool)]
                if source_index == anchor_index:
                    continue
                key = normalize_match_text(phrase)
                if key in seen_negatives:
                    continue
                if not safe_negative(
                    anchor_text,
                    phrase,
                    negative_ngram=args.negative_ngram,
                    min_overlap_word_chars=args.min_overlap_word_chars,
                ):
                    continue
                negative_texts.append(phrase)
                seen_negatives.add(key)
                if len(negative_texts) >= args.negatives_per_anchor:
                    break

            if len(negative_texts) < args.negatives_per_anchor:
                # A hard error is preferable to silently weakening the no-overlap
                # guarantee for difficult real lines.
                raise RuntimeError(
                    "Could not find enough guaranteed negatives for anchor "
                    f"{record['image_path']} (found {len(negative_texts)}/"
                    f"{args.negatives_per_anchor}). Try --negative-ngram 3 or fewer "
                    "--negatives-per-anchor only if the guarantee is still suitable."
                )

            anchor_id = _anchor_id(record)
            pair_id = f"bridge_{anchor_id}"
            real_image = str(Path(record["image_path"]).resolve())
            real_text = str(Path(record["text_path"]).resolve())
            page_id = str(record["page_id"])

            def write_synthetic(kind: str, ordinal: int, text: str) -> tuple[str, str, str]:
                stem = f"{anchor_id}_{kind}_{ordinal:02d}"
                image_rel = Path("images") / f"{stem}.png"
                text_rel = Path("texts") / f"{stem}.txt"
                font = font_paths[(anchor_index + ordinal + (0 if kind == "pos" else 1)) % len(font_paths)]
                render_arabic_line(
                    text,
                    font,
                    output_dir / image_rel,
                    width=args.width,
                    height=args.height,
                    font_size=args.font_size,
                    padding=args.padding,
                )
                (output_dir / text_rel).write_text(clean_render_text(text), encoding="utf-8")
                return image_rel.as_posix(), text_rel.as_posix(), font.name

            pos_image, pos_text_path, pos_font = write_synthetic("pos", 0, positive_text)
            pos_row = {
                "pair_id": pair_id,
                "label_type": "medium_match",
                "A_page_id": page_id,
                "B_page_id": f"synthetic:{pair_id}",
                "A": _side(real_image, real_text),
                "B": _side(pos_image, pos_text_path),
                "scores": {
                    "text_score": 1.0,
                    "avg_sim": 1.0,
                    "coverage_A": min(1.0, len(compact(positive_text)) / max(1, len(compact(anchor_text)))),
                    "coverage_B": 1.0,
                },
                "bridge": {
                    "relation": "positive_shared_span",
                    "anchor_id": anchor_id,
                    "shared_text": positive_text,
                    "font": pos_font,
                    "negative_ngram_guarantee": int(args.negative_ngram),
                },
            }
            manifest.write(json.dumps(pos_row, ensure_ascii=False) + "\n")
            stats["positive_rows"] += 1

            for neg_index, negative_text in enumerate(negative_texts):
                neg_image, neg_text_path, neg_font = write_synthetic(
                    "neg", neg_index, negative_text
                )
                # Defensive re-check immediately before persisting the row.
                assert safe_negative(
                    anchor_text,
                    negative_text,
                    negative_ngram=args.negative_ngram,
                    min_overlap_word_chars=args.min_overlap_word_chars,
                )
                neg_row = {
                    "pair_id": pair_id,
                    "label_type": "no_shared_content",
                    "A_page_id": page_id,
                    "B_page_id": f"synthetic:{pair_id}:neg{neg_index}",
                    "A": _side(real_image, real_text),
                    "B": _side(neg_image, neg_text_path),
                    "scores": {
                        "text_score": 0.0,
                        "avg_sim": 0.0,
                        "coverage_A": 0.0,
                        "coverage_B": 0.0,
                    },
                    "bridge": {
                        "relation": "negative_no_shared_span",
                        "anchor_id": anchor_id,
                        "negative_text": negative_text,
                        "font": neg_font,
                        "negative_ngram_guarantee": int(args.negative_ngram),
                        "min_overlap_word_chars": int(args.min_overlap_word_chars),
                    },
                }
                manifest.write(json.dumps(neg_row, ensure_ascii=False) + "\n")
                stats["negative_rows"] += 1

            stats["anchors_written"] += 1
            if (anchor_index + 1) % 100 == 0:
                print(
                    f"bridge_build anchors={anchor_index + 1}/{len(usable_records)} "
                    f"rows={stats['positive_rows'] + stats['negative_rows']}",
                    flush=True,
                )

    (output_dir / "metadata.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("=== REAL-CONDITIONED SYNTHETIC BRIDGE ===")
    print(f"output={output_dir}")
    print(f"manifest={manifest_path}")
    for key, value in stats.items():
        print(f"{key}={value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_DIR / "DataSet" / "ArabicDataset"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "DataSet" / "RealSyntheticBridge_v1"),
    )
    parser.add_argument("--negatives-per-anchor", type=int, default=4)
    parser.add_argument("--negative-ngram", type=int, default=4)
    parser.add_argument("--min-overlap-word-chars", type=int, default=3)
    parser.add_argument("--min-positive-chars", type=int, default=4)
    parser.add_argument("--max-phrase-chars", type=int, default=28)
    parser.add_argument("--max-phrase-words", type=int, default=3)
    parser.add_argument("--pool-spans-per-anchor", type=int, default=12)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--fonts", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--font-size", type=int, default=58)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.negatives_per_anchor <= 0:
        parser.error("--negatives-per-anchor must be positive")
    if args.negative_ngram < 2:
        parser.error("--negative-ngram must be >= 2")
    return args


if __name__ == "__main__":
    build(parse_args())
