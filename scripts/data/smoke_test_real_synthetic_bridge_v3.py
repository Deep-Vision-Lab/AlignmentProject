#!/usr/bin/env python3
"""Validate RealSyntheticBridge V3 before any GPU training."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.build_real_conditioned_synthetic_bridge import clean_render_text, normalize_match_text, safe_negative


def resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


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


def check_white_on_black(pair_id: str, path: Path) -> None:
    with Image.open(path) as image:
        if image.size != (1024, 128):
            raise RuntimeError(f"{pair_id}: expected 1024x128, got {image.size}")
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    # Appearance augmentation may add mild gray noise, but the canvas must remain
    # predominantly dark with clearly bright foreground ink.
    dark_fraction = float((arr < 64).mean())
    p99 = float(np.percentile(arr, 99))
    if dark_fraction < 0.55:
        raise RuntimeError(f"{pair_id}: image is not predominantly black (dark_fraction={dark_fraction:.3f})")
    if p99 < 120:
        raise RuntimeError(f"{pair_id}: no sufficiently bright foreground text (p99={p99:.1f})")


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
    if meta.get("image_polarity") != "white_text_on_black_background":
        raise RuntimeError("V3 requires white text on black background")
    if (meta.get("appearance_augmentation") or {}).get("geometric") is not False:
        raise RuntimeError("V3 must not use geometric augmentation")

    expected_negatives = args.expected_negatives if args.expected_negatives is not None else int(meta["negatives_per_anchor"])
    negative_ngram = int(meta["negative_ngram"])
    min_overlap_word_chars = int(meta["min_overlap_word_chars"])

    groups: dict[str, list[dict]] = defaultdict(list)
    with (root / "dataset_manifest.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                row = json.loads(raw)
                groups[str(row["pair_id"])].append(row)
    if not groups:
        raise RuntimeError("empty manifest")

    checked = positives = negatives = mixed_pos = mixed_neg = 0
    islands = {1: 0, 2: 0, 3: 0}
    for pair_id in sorted(groups):
        rows = groups[pair_id]
        pos_rows = [r for r in rows if r.get("label_type") == "medium_match"]
        neg_rows = [r for r in rows if r.get("label_type") == "no_shared_content"]
        if len(pos_rows) != 1 or len(neg_rows) != expected_negatives:
            raise RuntimeError(f"{pair_id}: expected 1 positive + {expected_negatives} negatives")

        pos = pos_rows[0]
        anchor_text_path = resolve(root, pos["A"]["text_original_path"])
        anchor_text = clean_render_text(anchor_text_path.read_text(encoding="utf-8"))
        anchor_norm = normalize_match_text(anchor_text)
        bridge = pos.get("bridge") or {}
        if int(bridge.get("dataset_version", 0)) != 3:
            raise RuntimeError(f"{pair_id}: positive row is not V3")

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
        for s in segments:
            if not s.get("font"):
                raise RuntimeError(f"{pair_id}: segment missing font metadata")
        fonts = {s.get("font") for s in segments}
        if len(segments) > 1 and len(meta.get("fonts") or []) > 1 and len(fonts) < 2:
            raise RuntimeError(f"{pair_id}: positive did not mix fonts")
        if len(fonts) > 1:
            mixed_pos += 1
        aug = bridge.get("appearance_augmentation") or {}
        if aug.get("geometric_transform") is not False:
            raise RuntimeError(f"{pair_id}: positive augmentation metadata allows geometry")

        pos_image = resolve(root, pos["B"]["line_image_path"])
        pos_mask = resolve(root, pos["B"].get("alignment_mask_path") or bridge["alignment_mask_path"])
        check_white_on_black(pair_id, pos_image)
        with Image.open(pos_mask) as image:
            if image.size != (1024, 128):
                raise RuntimeError(f"{pair_id}: bad mask size")
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise RuntimeError(f"{pair_id}: mask is not binary")
        assert_mask(pair_id, mask, shared_boxes)

        for neg in neg_rows:
            b = neg.get("bridge") or {}
            text_path = resolve(root, neg["B"]["text_original_path"])
            image_path = resolve(root, neg["B"]["line_image_path"])
            neg_text = clean_render_text(text_path.read_text(encoding="utf-8"))
            if len(neg_text.split()) < int(meta["sentence_min_words"]):
                raise RuntimeError(f"{pair_id}: negative is not a full sentence")
            if not safe_negative(anchor_text, neg_text, negative_ngram=negative_ngram,
                                 min_overlap_word_chars=min_overlap_word_chars):
                raise RuntimeError(f"{pair_id}: negative violates no-overlap guarantee")
            check_white_on_black(pair_id, image_path)
            nsegments = list(b.get("segments") or [])
            nfonts = {s.get("font") for s in nsegments}
            if len(nsegments) > 1 and len(meta.get("fonts") or []) > 1 and len(nfonts) < 2:
                raise RuntimeError(f"{pair_id}: negative did not mix fonts")
            if len(nfonts) > 1:
                mixed_neg += 1
            if (b.get("appearance_augmentation") or {}).get("geometric_transform") is not False:
                raise RuntimeError(f"{pair_id}: negative augmentation metadata allows geometry")

        checked += 1; positives += 1; negatives += len(neg_rows); islands[island_count] += 1
        if args.max_groups > 0 and checked >= args.max_groups:
            break

    print("=== REAL-SYNTHETIC BRIDGE V3 SMOKE TEST ===")
    print(f"data_dir={root}")
    print(f"groups_checked={checked}")
    print(f"positive_rows_checked={positives}")
    print(f"negative_rows_checked={negatives}")
    print(f"shared_islands_1={islands[1]}")
    print(f"shared_islands_2={islands[2]}")
    print(f"shared_islands_3={islands[3]}")
    print(f"mixed_font_positive_rows={mixed_pos}")
    print(f"mixed_font_negative_rows={mixed_neg}")
    print("image_polarity=white_text_on_black_background")
    print("appearance_augmentation=blur+noise+brightness+contrast; geometric=false")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
