#!/usr/bin/env python3
"""Validate RealSyntheticBridge V3 before any GPU training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.data.build_real_conditioned_synthetic_bridge import clean_render_text, normalize_match_text, safe_negative
from scripts.data.build_real_conditioned_synthetic_bridge_v3 import (
    DATASET_REVISION,
    font_supports_text,
    has_bad_unicode,
)

EXPECTED_LAYOUT_VERSION = 2


def resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def assert_path_group(value: str, category: str, anchor_id: str, *, family: str) -> None:
    parts = Path(str(value)).parts
    expected = (family, category, anchor_id)
    if len(parts) < 3 or tuple(parts[:3]) != expected:
        raise RuntimeError(f"{anchor_id}: expected {family}/{category}/{anchor_id}/..., got {value}")


def assert_mask(pair_id: str, mask: np.ndarray, boxes: list[list[int]]) -> None:
    expected = np.zeros_like(mask, dtype=np.uint8)
    h, w = expected.shape
    for box in boxes:
        if len(box) != 4:
            raise RuntimeError(f"{pair_id}: invalid box {box}")
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise RuntimeError(f"{pair_id}: box outside mask {box}")
        expected[y0:y1, x0:x1] = 255
    if not np.array_equal(mask, expected):
        raise RuntimeError(f"{pair_id}: mask does not match stored shared boxes")


def check_image_quality(pair_id: str, path: Path, min_fill_ratio: float) -> float:
    with Image.open(path) as image:
        if image.size != (1024, 128):
            raise RuntimeError(f"{pair_id}: expected 1024x128, got {image.size}")
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    dark_fraction = float((arr < 64).mean())
    p99 = float(np.percentile(arr, 99))
    if dark_fraction < 0.55:
        raise RuntimeError(f"{pair_id}: image is not predominantly black (dark_fraction={dark_fraction:.3f})")
    if p99 < 120:
        raise RuntimeError(f"{pair_id}: no sufficiently bright foreground text (p99={p99:.1f})")

    # Noise can brighten isolated pixels, so require at least two strong foreground
    # pixels in a column before considering that column part of the text span.
    foreground_columns = np.where((arr > 128).sum(axis=0) >= 2)[0]
    if foreground_columns.size == 0:
        raise RuntimeError(f"{pair_id}: no foreground text columns")
    span_ratio = float(foreground_columns[-1] - foreground_columns[0] + 1) / float(arr.shape[1])
    # The renderer computes fill inside the padded width, while this pixel check uses
    # the whole 1024px canvas. Allow a small tolerance for glyph side bearings.
    required = max(0.50, float(min_fill_ratio) - 0.06)
    if span_ratio < required:
        raise RuntimeError(
            f"{pair_id}: generated line is too horizontally empty "
            f"(foreground_span_ratio={span_ratio:.3f} < {required:.3f})"
        )
    return span_ratio


def font_map(meta: dict) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in meta.get("fonts") or []:
        candidates = list((PROJECT_DIR / "Fonts").rglob(str(name)))
        if candidates:
            found[str(name)] = candidates[0].resolve()
    return found


def validate_segments(pair_id: str, segments: list[dict], fonts_by_name: dict[str, Path]) -> set[str]:
    used: set[str] = set()
    for segment in segments:
        text = str(segment.get("text", ""))
        font_name = str(segment.get("font", ""))
        if not text or not font_name:
            raise RuntimeError(f"{pair_id}: segment missing text/font metadata")
        if has_bad_unicode(text):
            raise RuntimeError(f"{pair_id}: segment contains replacement/control/private-use Unicode: {text!r}")
        font_path = fonts_by_name.get(font_name)
        if font_path is None:
            raise RuntimeError(f"{pair_id}: cannot locate font recorded in manifest: {font_name}")
        if not font_supports_text(font_path, text):
            raise RuntimeError(
                f"{pair_id}: font {font_name} does not support the exact shaped segment {text!r}; "
                "this would render a tofu/question-mark box"
            )
        if segment.get("glyph_safe") is not True:
            raise RuntimeError(f"{pair_id}: segment is missing glyph_safe=true")
        used.add(font_name)
    return used


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--expected-negatives", type=int, default=None)
    p.add_argument("--max-groups", type=int, default=0)
    args = p.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if int(meta.get("dataset_version", 0)) != 3:
        raise RuntimeError(f"Expected dataset_version=3, got {meta.get('dataset_version')}")
    if meta.get("dataset_revision") != DATASET_REVISION:
        raise RuntimeError(
            f"Expected dataset_revision={DATASET_REVISION}, got {meta.get('dataset_revision')}; rebuild V3"
        )
    if int(meta.get("layout_version", 0)) != EXPECTED_LAYOUT_VERSION:
        raise RuntimeError(
            f"Bridge V3 requires layout_version={EXPECTED_LAYOUT_VERSION}; run organizer/rebuild first"
        )
    if meta.get("real_samples_copied") is not True:
        raise RuntimeError("Bridge V3 must contain local real-anchor copies")
    if meta.get("image_polarity") != "white_text_on_black_background":
        raise RuntimeError("V3 requires white text on black background")
    if (meta.get("appearance_augmentation") or {}).get("geometric") is not False:
        raise RuntimeError("V3 must not use geometric augmentation")

    for required in (
        "images/real", "images/positive", "images/negative",
        "texts/real", "texts/positive", "texts/negative",
        "masks/positive", "anchors", "positive", "negative",
    ):
        if not (root / required).is_dir():
            raise RuntimeError(f"Missing organized dataset directory: {required}")
    for required in (
        "README_DATASET.md", "anchor_index.jsonl", "real_lines_index.jsonl",
        "real_lines.csv", "dataset_manifest.jsonl",
    ):
        if not (root / required).is_file():
            raise RuntimeError(f"Missing organized dataset index: {required}")

    expected_negatives = args.expected_negatives if args.expected_negatives is not None else int(meta["negatives_per_anchor"])
    negative_ngram = int(meta["negative_ngram"])
    min_overlap_word_chars = int(meta["min_overlap_word_chars"])
    min_fill_ratio = float(meta.get("min_line_fill_ratio", 0.75))
    fonts_by_name = font_map(meta)
    if len(fonts_by_name) != len(meta.get("fonts") or []):
        missing = sorted(set(meta.get("fonts") or []) - set(fonts_by_name))
        raise RuntimeError(f"Could not locate configured font files: {missing}")

    groups: dict[str, list[dict]] = defaultdict(list)
    with (root / "dataset_manifest.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                groups[str(row["pair_id"])].append(row)
    if not groups:
        raise RuntimeError("empty manifest")

    index_rows = [json.loads(raw) for raw in (root / "anchor_index.jsonl").read_text(encoding="utf-8").splitlines() if raw.strip()]
    real_rows = [json.loads(raw) for raw in (root / "real_lines_index.jsonl").read_text(encoding="utf-8").splitlines() if raw.strip()]
    index_by_anchor = {str(row["anchor_id"]): row for row in index_rows}
    real_by_anchor = {str(row["anchor_id"]): row for row in real_rows}
    if len(real_by_anchor) != len(groups):
        raise RuntimeError(f"real_lines_index count {len(real_by_anchor)} != anchor groups {len(groups)}")

    checked = positives = negatives = mixed_pos = mixed_neg = 0
    islands = {1: 0, 2: 0, 3: 0}
    min_observed_fill = 1.0

    for pair_id in sorted(groups):
        rows = groups[pair_id]
        pos_rows = [r for r in rows if r.get("label_type") == "medium_match"]
        neg_rows = [r for r in rows if r.get("label_type") == "no_shared_content"]
        if len(pos_rows) != 1 or len(neg_rows) != expected_negatives:
            raise RuntimeError(f"{pair_id}: expected 1 positive + {expected_negatives} negatives")

        pos = pos_rows[0]
        bridge = pos.get("bridge") or {}
        anchor_id = str(bridge.get("anchor_id") or "")
        if not anchor_id:
            raise RuntimeError(f"{pair_id}: missing anchor_id")
        if bridge.get("dataset_revision") != DATASET_REVISION:
            raise RuntimeError(f"{pair_id}: stale dataset revision")
        if int(bridge.get("layout_version", 0)) != EXPECTED_LAYOUT_VERSION:
            raise RuntimeError(f"{pair_id}: manifest row was not rewritten to current organized layout")

        for path in (
            root / "anchors" / anchor_id / "anchor.json",
            root / "positive" / anchor_id / "relation.json",
            root / "negative" / anchor_id / "relations.json",
        ):
            if not path.is_file():
                raise RuntimeError(f"{pair_id}: missing per-anchor relationship file {path}")
        if anchor_id not in index_by_anchor or anchor_id not in real_by_anchor:
            raise RuntimeError(f"{pair_id}: anchor missing from human/real scrape indexes")

        assert_path_group(pos["A"]["line_image_path"], "real", anchor_id, family="images")
        assert_path_group(pos["A"]["text_original_path"], "real", anchor_id, family="texts")
        anchor_image_path = resolve(root, pos["A"]["line_image_path"])
        anchor_text_path = resolve(root, pos["A"]["text_original_path"])
        if not anchor_image_path.is_file() or not anchor_text_path.is_file():
            raise RuntimeError(f"{pair_id}: local real anchor copy is missing")
        anchor_text = clean_render_text(anchor_text_path.read_text(encoding="utf-8"))
        anchor_norm = normalize_match_text(anchor_text)

        base_sentence = clean_render_text(bridge.get("base_sentence", ""))
        positive_sentence = clean_render_text(bridge.get("positive_full_sentence", ""))
        if not base_sentence or not positive_sentence:
            raise RuntimeError(f"{pair_id}: missing full-sentence metadata")
        if not safe_negative(anchor_text, base_sentence, negative_ngram=negative_ngram,
                             min_overlap_word_chars=min_overlap_word_chars):
            raise RuntimeError(f"{pair_id}: base sentence is not content-clean")

        segments = list(bridge.get("segments") or [])
        shared_texts = list(bridge.get("shared_texts") or [])
        shared_boxes = list(bridge.get("shared_boxes_px") or [])
        island_count = int(bridge.get("shared_island_count", 0))
        if island_count not in {1, 2, 3} or len(shared_texts) != island_count or len(shared_boxes) != island_count:
            raise RuntimeError(f"{pair_id}: invalid shared-island metadata")
        distractor_text = clean_render_text(" ".join(str(s.get("text", "")) for s in segments if s.get("kind") == "distractor"))
        if normalize_match_text(distractor_text) != normalize_match_text(base_sentence):
            raise RuntimeError(f"{pair_id}: full base sentence was not preserved in the positive")
        for text in shared_texts:
            if normalize_match_text(text) not in anchor_norm:
                raise RuntimeError(f"{pair_id}: shared text is not in anchor: {text!r}")

        pos_fonts = validate_segments(pair_id, segments, fonts_by_name)
        if len(segments) > 1 and len(fonts_by_name) > 1 and len(pos_fonts) < 2:
            raise RuntimeError(f"{pair_id}: positive did not mix fonts despite safe alternatives")
        if len(pos_fonts) > 1:
            mixed_pos += 1
        pos_aug = bridge.get("appearance_augmentation") or {}
        if pos_aug.get("geometric_transform") is not False or bridge.get("glyph_safe") is not True:
            raise RuntimeError(f"{pair_id}: positive augmentation/glyph metadata invalid")
        if float(pos_aug.get("line_fill_ratio", 0.0)) < min_fill_ratio:
            raise RuntimeError(f"{pair_id}: recorded positive line_fill_ratio below required minimum")

        assert_path_group(pos["B"]["line_image_path"], "positive", anchor_id, family="images")
        assert_path_group(pos["B"]["text_original_path"], "positive", anchor_id, family="texts")
        pos_image = resolve(root, pos["B"]["line_image_path"])
        mask_value = pos["B"].get("alignment_mask_path") or bridge["alignment_mask_path"]
        assert_path_group(mask_value, "positive", anchor_id, family="masks")
        pos_mask = resolve(root, mask_value)
        min_observed_fill = min(min_observed_fill, check_image_quality(pair_id, pos_image, min_fill_ratio))
        with Image.open(pos_mask) as image:
            if image.size != (1024, 128):
                raise RuntimeError(f"{pair_id}: bad mask size")
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise RuntimeError(f"{pair_id}: mask is not binary")
        assert_mask(pair_id, mask, shared_boxes)

        for neg in neg_rows:
            b = neg.get("bridge") or {}
            if str(b.get("anchor_id")) != anchor_id or b.get("dataset_revision") != DATASET_REVISION:
                raise RuntimeError(f"{pair_id}: negative identity/revision mismatch")
            assert_path_group(neg["B"]["line_image_path"], "negative", anchor_id, family="images")
            assert_path_group(neg["B"]["text_original_path"], "negative", anchor_id, family="texts")
            text_path = resolve(root, neg["B"]["text_original_path"])
            image_path = resolve(root, neg["B"]["line_image_path"])
            neg_text = clean_render_text(text_path.read_text(encoding="utf-8"))
            if len(neg_text.split()) < int(meta["sentence_min_words"]):
                raise RuntimeError(f"{pair_id}: negative is not a full sentence")
            if len(normalize_match_text(neg_text).replace(" ", "")) < int(meta["min_sentence_chars"]):
                raise RuntimeError(f"{pair_id}: negative is too short")
            if not safe_negative(anchor_text, neg_text, negative_ngram=negative_ngram,
                                 min_overlap_word_chars=min_overlap_word_chars):
                raise RuntimeError(f"{pair_id}: negative violates no-overlap guarantee")
            nsegments = list(b.get("segments") or [])
            nfonts = validate_segments(pair_id, nsegments, fonts_by_name)
            if len(nsegments) > 1 and len(fonts_by_name) > 1 and len(nfonts) < 2:
                raise RuntimeError(f"{pair_id}: negative did not mix fonts despite safe alternatives")
            if len(nfonts) > 1:
                mixed_neg += 1
            naug = b.get("appearance_augmentation") or {}
            if naug.get("geometric_transform") is not False or b.get("glyph_safe") is not True:
                raise RuntimeError(f"{pair_id}: negative augmentation/glyph metadata invalid")
            if float(naug.get("line_fill_ratio", 0.0)) < min_fill_ratio:
                raise RuntimeError(f"{pair_id}: recorded negative line_fill_ratio below required minimum")
            min_observed_fill = min(min_observed_fill, check_image_quality(pair_id, image_path, min_fill_ratio))

        checked += 1
        positives += 1
        negatives += len(neg_rows)
        islands[island_count] += 1
        if args.max_groups > 0 and checked >= args.max_groups:
            break

    print("=== REAL-SYNTHETIC BRIDGE V3 SMOKE TEST ===")
    print(f"data_dir={root}")
    print(f"dataset_revision={DATASET_REVISION}")
    print("layout=anchor_grouped_self_contained_with_real_scrape_index")
    print("relationship_key=anchor_id")
    print(f"groups_checked={checked}")
    print(f"positive_rows_checked={positives}")
    print(f"negative_rows_checked={negatives}")
    print(f"negatives_per_anchor={expected_negatives}")
    print(f"shared_islands_1={islands[1]}")
    print(f"shared_islands_2={islands[2]}")
    print(f"shared_islands_3={islands[3]}")
    print(f"mixed_font_positive_rows={mixed_pos}")
    print(f"mixed_font_negative_rows={mixed_neg}")
    print(f"minimum_observed_foreground_span_ratio={min_observed_fill:.4f}")
    print("glyph_validation=PASS")
    print("image_polarity=white_text_on_black_background")
    print("appearance_augmentation=blur+noise+brightness+contrast; geometric=false")
    print("real_lines_index=present")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
