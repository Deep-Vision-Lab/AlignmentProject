#!/usr/bin/env python3
"""Report real lines that cannot fit the connected-subword Span-DTW lattice."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from connected_subword_mode import connected_units
from RealDataSet import ArabicManifestLinePairDataset


def _labels(value: str):
    value = str(value).strip()
    if value.lower() in {"all", "*", "any"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="DataSet/ArabicDataset",
        help="ArabicDataset directory or dataset_manifest.jsonl path",
    )
    parser.add_argument("--labels", default="high_match,medium_match")
    parser.add_argument("--text-key", default="text_original_path")
    parser.add_argument("--max-windows", type=int, default=125)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    data_path = Path(args.data_dir).expanduser().resolve()
    manifest = (
        data_path
        if data_path.suffix.lower() == ".jsonl"
        else data_path / "dataset_manifest.jsonl"
    )
    dataset = ArabicManifestLinePairDataset(
        manifest,
        transform=None,
        text_key=args.text_key,
        allowed_labels=_labels(args.labels),
        paired=True,
        validate_paths=False,
    )

    rows = []
    counts = Counter()
    for sample_index, sample in enumerate(dataset.samples):
        for side_name in ("A", "B"):
            side = sample[side_name]
            text = dataset._read_text(side[dataset.text_key])
            units = connected_units(text)
            required = len(units)
            counts[required] += 1
            rows.append(
                {
                    "required": required,
                    "sample_index": sample_index,
                    "pair_id": sample.get("pair_id", sample_index),
                    "side": side_name,
                    "line_idx": side.get("line_idx", -1),
                    "image": side.get("line_image_path", ""),
                    "text_path": side.get(dataset.text_key, ""),
                    "characters": len(str(text).strip()),
                    "words": len(str(text).strip().split()),
                    "subwords": sum(unit.kind == "subword" for unit in units),
                    "boundaries": sum(unit.kind == "boundary" for unit in units),
                    "spaces": sum(unit.kind == "space" for unit in units),
                }
            )

    rows.sort(key=lambda item: item["required"], reverse=True)
    offenders = [item for item in rows if item["required"] > args.max_windows]
    print(
        "Connected-subword feasibility report: "
        f"pairs={len(dataset)} lines={len(rows)} max_windows={args.max_windows} "
        f"offending_lines={len(offenders)} max_required={rows[0]['required'] if rows else 0}"
    )
    if rows:
        values = sorted(item["required"] for item in rows)
        for percentile in (50, 90, 95, 99, 100):
            position = min(
                len(values) - 1,
                max(0, int(round((percentile / 100.0) * (len(values) - 1)))),
            )
            print(f"p{percentile}_required={values[position]}")

    for item in offenders[: max(0, args.top)]:
        print(
            "OFFENDER "
            f"required={item['required']} pair_id={item['pair_id']} "
            f"side={item['side']} line_idx={item['line_idx']} "
            f"chars={item['characters']} words={item['words']} "
            f"subwords={item['subwords']} boundaries={item['boundaries']} "
            f"spaces={item['spaces']} image={item['image']} "
            f"text={item['text_path']}"
        )


if __name__ == "__main__":
    main()
