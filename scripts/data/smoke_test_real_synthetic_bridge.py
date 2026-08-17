#!/usr/bin/env python3
"""Validate an already-built real-conditioned synthetic bridge corpus.

This is deliberately cheap: it performs no model inference and no rendering. It
checks the invariants that matter before an expensive GPU job is submitted:
- every anchor group has exactly one positive and K guaranteed negatives;
- every synthetic image/text file exists;
- synthetic images have the configured fixed line geometry; and
- every negative still satisfies the same no-word/no-n-gram guarantee used by the
  builder when compared with its real anchor transcript.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from build_real_conditioned_synthetic_bridge import clean_render_text, safe_negative


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path)


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

        anchor_text_path = _resolve(root, positive_rows[0]["A"]["text_original_path"])
        if not anchor_text_path.is_file():
            raise FileNotFoundError(f"{pair_id}: missing real anchor text {anchor_text_path}")
        anchor_text = clean_render_text(anchor_text_path.read_text(encoding="utf-8"))

        for row in rows:
            synth_image = _resolve(root, row["B"]["line_image_path"])
            synth_text_path = _resolve(root, row["B"]["text_original_path"])
            if not synth_image.is_file():
                raise FileNotFoundError(f"{pair_id}: missing synthetic image {synth_image}")
            if not synth_text_path.is_file():
                raise FileNotFoundError(f"{pair_id}: missing synthetic text {synth_text_path}")
            with Image.open(synth_image) as image:
                if image.size != (1024, 128):
                    raise RuntimeError(
                        f"{pair_id}: expected synthetic size (1024, 128), got {image.size}"
                    )
            synth_text = clean_render_text(synth_text_path.read_text(encoding="utf-8"))
            if row.get("label_type") == "no_shared_content" and not safe_negative(
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
        if args.max_groups > 0 and checked >= args.max_groups:
            break

    print("=== REAL-SYNTHETIC BRIDGE SMOKE TEST ===")
    print(f"data_dir={root}")
    print(f"groups_checked={checked}")
    print(f"positive_rows_checked={positives}")
    print(f"negative_rows_checked={negatives}")
    print(f"negative_ngram={negative_ngram}")
    print(f"min_overlap_word_chars={min_overlap_word_chars}")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
