#!/usr/bin/env python3
"""Audit which real Arabic line crops are used by manifest/pair training.

Reports three pools:
1) lines referenced by dataset_manifest.jsonl (the sampled training entry point),
2) lines referenced by full DatasetPairs/line_pairs/*/pairs_lines_full.jsonl files,
3) all line crops physically present under DatasetPairs/page_pairs/*/{A,B}/linesImages.

This makes it possible to distinguish:
- sampled manifest lines,
- full-pair lines omitted by manifest sampling,
- line crops that never appear in any line-pair row (singleton/no-partner candidates).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("DataSet/ArabicDataset"))
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--write-singletons", type=Path, default=None,
                   help="Optional JSONL output for line crops not present in any full line-pair row.")
    return p.parse_args()


def norm(root: Path, value) -> str | None:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = root / p
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p.resolve())


def iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc


def row_line_paths(root: Path, row: dict) -> set[str]:
    out = set()
    for side_name in ("A", "B"):
        side = row.get(side_name)
        if isinstance(side, dict):
            value = norm(root, side.get("line_image_path"))
            if value:
                out.add(value)
    return out


def expected_text_for_image(image_path: Path) -> Path:
    side_dir = image_path.parent.parent
    return side_dir / "text" / "final" / "original" / f"{image_path.stem}.txt"


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = (args.manifest or (root / "dataset_manifest.jsonl")).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"ERROR: missing dataset root: {root}")
    if not manifest.is_file():
        raise SystemExit(f"ERROR: missing manifest: {manifest}")

    manifest_rows = list(iter_jsonl(manifest))
    label_counts = Counter(str(row.get("label_type", "")) for row in manifest_rows)
    manifest_lines = set()
    manifest_lines_by_label: dict[str, set[str]] = {}
    for row in manifest_rows:
        label = str(row.get("label_type", ""))
        paths = row_line_paths(root, row)
        manifest_lines.update(paths)
        manifest_lines_by_label.setdefault(label, set()).update(paths)

    full_pair_files = sorted((root / "DatasetPairs" / "line_pairs").glob("pair_*/pairs_lines_full.jsonl"))
    full_pair_rows = 0
    full_pair_lines = set()
    full_label_counts = Counter()
    for path in full_pair_files:
        for row in iter_jsonl(path):
            full_pair_rows += 1
            full_label_counts[str(row.get("label_type", ""))] += 1
            full_pair_lines.update(row_line_paths(root, row))

    page_pair_glob = root / "DatasetPairs" / "page_pairs"
    all_crop_paths = sorted(
        p.resolve()
        for p in page_pair_glob.glob("pair_*/*/linesImages/*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    all_crop_lines = {norm(root, p) for p in all_crop_paths}
    all_crop_lines.discard(None)

    full_not_sampled = full_pair_lines - manifest_lines
    singleton_candidates = all_crop_lines - full_pair_lines
    manifest_missing_on_disk = {
        p for p in manifest_lines
        if not (Path(p) if Path(p).is_absolute() else root / p).is_file()
    }
    full_missing_on_disk = {
        p for p in full_pair_lines
        if not (Path(p) if Path(p).is_absolute() else root / p).is_file()
    }

    singleton_records = []
    missing_singleton_text = 0
    for image in all_crop_paths:
        rel = norm(root, image)
        if rel not in singleton_candidates:
            continue
        text_path = expected_text_for_image(image)
        if not text_path.is_file():
            missing_singleton_text += 1
            continue
        try:
            rel_text = str(text_path.resolve().relative_to(root))
        except ValueError:
            rel_text = str(text_path.resolve())
        parts = image.relative_to(root / "DatasetPairs" / "page_pairs").parts
        pair_id = parts[0] if len(parts) > 0 else ""
        side = parts[1] if len(parts) > 1 else ""
        singleton_records.append({
            "sample_type": "singleton_real_line",
            "source_pair_id": pair_id,
            "side": side,
            "line_image_path": rel,
            "text_original_path": rel_text,
            "line_name": image.stem,
        })

    print("=== REAL DATASET COVERAGE AUDIT ===")
    print(f"root={root}")
    print(f"manifest={manifest}")
    print()
    print("[sampled manifest]")
    print(f"rows={len(manifest_rows)}")
    print("labels=" + json.dumps(dict(sorted(label_counts.items())), ensure_ascii=False))
    print(f"unique_line_paths={len(manifest_lines)}")
    for label in sorted(manifest_lines_by_label):
        print(f"  {label}_unique_line_paths={len(manifest_lines_by_label[label])}")
    print()
    print("[full line-pair source]")
    print(f"pair_jsonl_files={len(full_pair_files)}")
    print(f"rows={full_pair_rows}")
    print("labels=" + json.dumps(dict(sorted(full_label_counts.items())), ensure_ascii=False))
    print(f"unique_line_paths={len(full_pair_lines)}")
    print()
    print("[all page-pair line crops]")
    print(f"physical_line_crop_files={len(all_crop_lines)}")
    print()
    print("[unused / expansion pools]")
    print(f"full_pair_lines_not_in_sampled_manifest={len(full_not_sampled)}")
    print(f"line_crops_not_in_any_full_pair_row={len(singleton_candidates)}")
    print(f"singleton_candidates_with_text={len(singleton_records)}")
    print(f"singleton_candidates_missing_text={missing_singleton_text}")
    print()
    print("[path integrity]")
    print(f"manifest_references_missing_on_disk={len(manifest_missing_on_disk)}")
    print(f"full_pair_references_missing_on_disk={len(full_missing_on_disk)}")

    if args.write_singletons is not None:
        output = args.write_singletons.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in singleton_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print()
        print(f"singleton_manifest={output}")
        print(f"singleton_manifest_rows={len(singleton_records)}")


if __name__ == "__main__":
    main()
