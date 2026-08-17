#!/usr/bin/env python3
"""Reject Bridge V3 datasets containing synthetic lines rendered with tiny text."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--min-font-size", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    manifest = root / "dataset_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    sizes: list[int] = []
    rows_checked = 0
    segments_checked = 0
    failures: list[str] = []

    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            bridge = row.get("bridge") or {}
            relation = str(bridge.get("relation", ""))
            if not relation.startswith(("positive_", "negative_")):
                continue
            rows_checked += 1
            pair_id = str(row.get("pair_id", "?"))
            row_size = int(bridge.get("font_size", 0) or 0)
            sizes.append(row_size)
            if row_size < args.min_font_size:
                failures.append(
                    f"{pair_id}/{row.get('B_page_id')}: row font_size={row_size} < {args.min_font_size}"
                )

            for segment in bridge.get("segments") or []:
                segments_checked += 1
                size = int(segment.get("font_size", 0) or 0)
                if size < args.min_font_size:
                    failures.append(
                        f"{pair_id}/{row.get('B_page_id')}: segment {segment.get('text')!r} "
                        f"font_size={size} < {args.min_font_size}"
                    )

    if rows_checked == 0:
        raise RuntimeError("No synthetic Bridge V3 rows found")
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"READABLE_FONT_TEST=FAIL: {len(failures)} too-small row/segment entries\n{preview}"
        )

    print("=== BRIDGE V3 READABLE FONT TEST ===")
    print(f"data_dir={root}")
    print(f"rows_checked={rows_checked}")
    print(f"segments_checked={segments_checked}")
    print(f"required_min_font_size={args.min_font_size}")
    print(f"observed_min_font_size={min(sizes)}")
    print(f"observed_median_font_size={statistics.median(sizes):.1f}")
    print(f"observed_max_font_size={max(sizes)}")
    print("READABLE_FONT_TEST=PASS")


if __name__ == "__main__":
    main()
