#!/usr/bin/env python3
"""Validate an already-built real-conditioned synthetic bridge V2 corpus.

Checks performed before any GPU job:
- exactly one positive and K negatives per real anchor;
- every synthetic image/text exists and has fixed geometry;
- every positive has 1..3 shared islands plus a same-size binary alignment mask;
- mask white regions agree with persisted shared boxes and remain black elsewhere;
- every declared shared text is truly contained in the real anchor transcript;
- every positive distractor and every negative line satisfies the no-word/no-n-gram
  guarantee against the real anchor.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from build_real_conditioned_synthetic_bridge import (
    clean_render_text,
    normalize_match_text,
    safe_negative,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path)


def _assert_mask_matches_boxes(pair_id: str, mask: np.ndarray, boxes: list[list[int]]) -> None:
    expected = np.zeros_like(mask, dtype=np.uint8)
    height, width = expected.shape
    for box in boxes:
        if len(box) != 4:
            raise RuntimeError(f"{pair_id}: invalid shared bbox {box!r}")
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise RuntimeError(f"{pair_id}: shared bbox outside mask: {box!r}")
        expected[y0:y1, x0:x1] = 255
    if not np.array_equal(mask, expected):
        mismatch = int(np.count_nonzero(mask != expected))
        raise RuntimeError(
            f"{pair_id}: alignment mask disagrees with shared boxes at {mismatch} pixels"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--expected-negatives", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    metadata_path = root / "metadata.json"
    manifest_path = root / "dataset_manifest.jsonl"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Missing metadata/manifest under {root}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("dataset_version", 0)) < 2:
        raise RuntimeError("This smoke test expects RealSyntheticBridge dataset_version >= 2")
    expected_negatives = (
        int(args.expected_negatives)
        if args.expected_negatives is not None
        else int(metadata["negatives_per_anchor"])
    )
    negative_ngram = int(metadata["negative_ngram"])
    min_overlap_word_chars = int(metadata["min_overlap_word_chars"])

    groups: dict[str, list[dict]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {manifest_path}:{lineno}: {exc}") from exc
            groups[str(row.get("pair_id", ""))].append(row)
    if not groups:
        raise RuntimeError("Bridge manifest contains no anchor groups.")

    checked = positives = negatives = 0
    island_histogram = {1: 0, 2: 0, 3: 0}
    for pair_id in sorted(groups):
        rows = groups[pair_id]
        positive_rows = [row for row in rows if row.get("label_type") == "medium_match"]
        negative_rows = [row for row in rows if row.get("label_type") == "no_shared_content"]
        if len(positive_rows) != 1:
            raise RuntimeError(f"{pair_id}: expected 1 positive, found {len(positive_rows)}")
        if len(negative_rows) != expected_negatives:
            raise RuntimeError(
                f"{pair_id}: expected {expected_negatives} negatives, found {len(negative_rows)}"
            )

        positive = positive_rows[0]
        anchor_text_path = _resolve(root, positive["A"]["text_original_path"])
        if not anchor_text_path.is_file():
            raise FileNotFoundError(f"{pair_id}: missing real anchor text {anchor_text_path}")
        anchor_text = clean_render_text(anchor_text_path.read_text(encoding="utf-8"))
        anchor_norm = normalize_match_text(anchor_text)

        bridge = positive.get("bridge") or {}
        island_count = int(bridge.get("shared_island_count", 0))
        shared_texts = list(bridge.get("shared_texts") or [])
        shared_boxes = list(bridge.get("shared_boxes_px") or [])
        segments = list(bridge.get("segments") or [])
        if island_count not in {1, 2, 3}:
            raise RuntimeError(f"{pair_id}: invalid shared_island_count={island_count}")
        if len(shared_texts) != island_count or len(shared_boxes) != island_count:
            raise RuntimeError(
                f"{pair_id}: island metadata mismatch count={island_count} "
                f"texts={len(shared_texts)} boxes={len(shared_boxes)}"
            )
        if not any(segment.get("kind") == "distractor" for segment in segments):
            raise RuntimeError(f"{pair_id}: positive has no distractor region")
        for shared_text in shared_texts:
            shared_norm = normalize_match_text(shared_text)
            if not shared_norm or shared_norm not in anchor_norm:
                raise RuntimeError(
                    f"{pair_id}: declared shared text is not an anchor substring: {shared_text!r}"
                )
        for segment in segments:
            if segment.get("kind") == "distractor" and not safe_negative(
                anchor_text,
                str(segment.get("text", "")),
                negative_ngram=negative_ngram,
                min_overlap_word_chars=min_overlap_word_chars,
            ):
                raise RuntimeError(
                    f"{pair_id}: positive distractor violates no-overlap guarantee: "
                    f"{segment.get('text')!r}"
                )

        pos_image = _resolve(root, positive["B"]["line_image_path"])
        pos_text = _resolve(root, positive["B"]["text_original_path"])
        mask_value = positive["B"].get("alignment_mask_path") or bridge.get("alignment_mask_path")
        if not mask_value:
            raise RuntimeError(f"{pair_id}: positive row has no alignment_mask_path")
        pos_mask = _resolve(root, mask_value)
        for path in (pos_image, pos_text, pos_mask):
            if not path.is_file():
                raise FileNotFoundError(f"{pair_id}: missing positive artifact {path}")
        with Image.open(pos_image) as image:
            if image.size != (1024, 128):
                raise RuntimeError(f"{pair_id}: positive size must be (1024, 128), got {image.size}")
        with Image.open(pos_mask) as image:
            if image.size != (1024, 128):
                raise RuntimeError(f"{pair_id}: mask size must be (1024, 128), got {image.size}")
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}) or 255 not in values or 0 not in values:
            raise RuntimeError(f"{pair_id}: mask must contain both binary values 0/255, got {values}")
        _assert_mask_matches_boxes(pair_id, mask, shared_boxes)

        for row in negative_rows:
            synth_image = _resolve(root, row["B"]["line_image_path"])
            synth_text_path = _resolve(root, row["B"]["text_original_path"])
            if not synth_image.is_file() or not synth_text_path.is_file():
                raise FileNotFoundError(f"{pair_id}: missing negative artifact")
            with Image.open(synth_image) as image:
                if image.size != (1024, 128):
                    raise RuntimeError(
                        f"{pair_id}: negative size must be (1024, 128), got {image.size}"
                    )
            synth_text = clean_render_text(synth_text_path.read_text(encoding="utf-8"))
            if not safe_negative(
                anchor_text,
                synth_text,
                negative_ngram=negative_ngram,
                min_overlap_word_chars=min_overlap_word_chars,
            ):
                raise RuntimeError(
                    f"{pair_id}: persisted negative violates no-overlap guarantee: {synth_text!r}"
                )

        checked += 1
        positives += 1
        negatives += len(negative_rows)
        island_histogram[island_count] += 1
        if args.max_groups > 0 and checked >= args.max_groups:
            break

    print("=== REAL-SYNTHETIC BRIDGE V2 SMOKE TEST ===")
    print(f"data_dir={root}")
    print(f"groups_checked={checked}")
    print(f"positive_rows_checked={positives}")
    print(f"negative_rows_checked={negatives}")
    print(f"shared_islands_1={island_histogram[1]}")
    print(f"shared_islands_2={island_histogram[2]}")
    print(f"shared_islands_3={island_histogram[3]}")
    print(f"negative_ngram={negative_ngram}")
    print(f"min_overlap_word_chars={min_overlap_word_chars}")
    print("mask_semantics=white_shared_black_unaligned")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
