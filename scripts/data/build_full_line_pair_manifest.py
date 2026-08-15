#!/usr/bin/env python3
"""Build a deterministic manifest containing every generated real line-pair row.

The canonical dataset_manifest.jsonl intentionally samples no_shared_content rows.
For late-stage pair discrimination we want the full relationship pool while
keeping the same positive rows and the same ArabicDataset-relative path semantics.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("DataSet/ArabicDataset"))
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or (root / "dataset_manifest_full_pairs.jsonl")).expanduser().resolve()
    source_dir = root / "DatasetPairs" / "line_pairs"
    files = sorted(source_dir.glob("pair_*/pairs_lines_full.jsonl"))
    if not files:
        raise SystemExit(f"ERROR: no pairs_lines_full.jsonl files under {source_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows = 0
    with output.open("w", encoding="utf-8") as dst:
        for path in files:
            with path.open("r", encoding="utf-8") as src:
                for lineno, raw in enumerate(src, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
                    counts[str(row.get("label_type", ""))] += 1
                    rows += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== FULL REAL LINE-PAIR MANIFEST ===")
    print(f"source_files={len(files)}")
    print(f"rows={rows}")
    print("labels=" + json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    print(f"output={output}")


if __name__ == "__main__":
    main()
