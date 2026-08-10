#!/usr/bin/env python3
"""Build ground-truth aligned-region masks for real Arabic line pairs.

For every manifest row, this script:
1. resolves the saved A/B line images;
2. loads their subword bounding boxes using the repository's existing bbox parser;
3. orders boxes in Arabic reading order (right-to-left);
4. aligns bbox texts with an exact, order-preserving LCS after the same Arabic
   normalization used by the real-box evaluator;
5. groups matches into consecutive A/B runs; and
6. paints every run as a full-height white vertical interval on a black mask.

The masks are intentionally line-sized, not model-window-sized.  Consecutive
matched boxes are filled from the left-most to right-most edge of the run, so
small spaces between consecutive aligned subwords are white as well.

By default manifests are not modified.  ``--write-manifests`` writes companion
``*_with_masks.jsonl`` files and leaves the source manifests untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Evaluation.real_subword_box_json import load_json_annotations
from Evaluation.real_subword_box_metrics import BoxAnnotations, SubwordBox, load_line_annotations


_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full-height binary masks for bbox-text-aligned real line pairs."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "DataSet" / "ArabicDatasetRealAug10K",
        help="Root containing dataset_manifest.jsonl/train_manifest.jsonl/etc.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest name/path. Repeat to process more than one manifest.",
    )
    parser.add_argument(
        "--all-manifests",
        action="store_true",
        help="Process dataset_manifest, train_manifest, valid_manifest and test_manifest when present.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Mask output root. Default: <dataset-root>/alignment_masks.",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=None,
        help="Preview image directory. Default: <output-root>/preview.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also save a 4-row A/A-mask/B/B-mask preview image per processed pair.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many rows total. 0 means no limit.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many rows before processing (useful for spot checks).",
    )
    parser.add_argument(
        "--write-manifests",
        action="store_true",
        help="Write companion *_with_masks.jsonl manifests; never overwrites originals.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask/preview files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately when a row cannot produce aligned masks.",
    )
    parser.add_argument(
        "--box-coordinate-space",
        choices=("auto", "original", "normalized"),
        default="auto",
        help="Coordinate system used by bbox annotations. auto treats max x<=1.5 as normalized.",
    )
    parser.add_argument(
        "--text-key",
        default="text_original_path",
        help="Side text path used to fill missing bbox labels.",
    )
    return parser.parse_args()


def _normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _ARABIC_DIACRITICS.sub("", text.replace("ـ", ""))
    return "".join(text.split())


def _reading_order(boxes: Iterable[SubwordBox]) -> list[SubwordBox]:
    # Arabic lines are read right-to-left. This is the same ordering used by
    # Evaluation.real_subword_box_metrics.
    return sorted(boxes, key=lambda box: (-box.center_x, box.y0, box.source_row))


def _resolve_path(dataset_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)

    candidates = (
        dataset_root / path,
        dataset_root.parent / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {value!r}. Tried:\n  - {rendered}")


def _read_units(text_path: Path) -> list[str]:
    text = text_path.read_text(encoding="utf-8").strip()
    try:
        from connected_subword_mode import connected_units

        units = [unit.text for unit in connected_units(text) if unit.kind == "subword"]
        if units:
            return units
    except Exception:
        pass
    return [token for token in text.split() if token]


def _fill_missing_box_text(
    boxes: list[SubwordBox], text_path: Path | None
) -> tuple[list[SubwordBox], str]:
    if boxes and all(_normalise_text(box.text) for box in boxes):
        return boxes, "bbox_labels"
    if text_path is None or not text_path.is_file():
        return boxes, "missing_text"

    units = _read_units(text_path)
    if len(units) != len(boxes):
        return boxes, f"text_count_mismatch:{len(units)}!={len(boxes)}"

    filled = [
        SubwordBox(unit, box.x0, box.y0, box.x1, box.y1, box.source_row)
        for unit, box in zip(units, boxes)
    ]
    return filled, "transcript_units"


def _load_annotations(image_path: Path, width: int, height: int) -> BoxAnnotations:
    annotations = load_json_annotations(image_path)
    if annotations.boxes:
        return annotations
    # Older samples may still keep Excel annotations. Keep that path working so
    # the mask builder can be used on both the original and augmented real data.
    fallback = load_line_annotations(image_path, width, height)
    return fallback if fallback.boxes else annotations


def _lcs_pairs(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    n, m = len(left), len(right)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if left[i] and left[i] == right[j]:
                dp[i, j] = 1 + dp[i + 1, j + 1]
            else:
                dp[i, j] = max(dp[i + 1, j], dp[i, j + 1])

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if left[i] and left[i] == right[j] and dp[i, j] == 1 + dp[i + 1, j + 1]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _consecutive_runs(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    if not pairs:
        return []
    runs = [[pairs[0]]]
    for pair in pairs[1:]:
        previous = runs[-1][-1]
        if pair[0] == previous[0] + 1 and pair[1] == previous[1] + 1:
            runs[-1].append(pair)
        else:
            runs.append([pair])
    return runs


def _coordinate_scale(boxes: list[SubwordBox], width: int, mode: str) -> float:
    if mode == "normalized":
        return float(width)
    if mode == "original":
        return 1.0
    maximum = max((abs(float(box.x1)) for box in boxes), default=0.0)
    return float(width) if maximum <= 1.5 else 1.0


def _run_intervals(
    boxes: list[SubwordBox],
    runs: list[list[tuple[int, int]]],
    side_index: int,
    width: int,
    coordinate_space: str,
) -> list[tuple[int, int]]:
    scale = _coordinate_scale(boxes, width, coordinate_space)
    intervals: list[tuple[int, int]] = []
    for run in runs:
        selected = [boxes[pair[side_index]] for pair in run]
        left = min(float(box.x0) for box in selected) * scale
        right = max(float(box.x1) for box in selected) * scale
        x0 = max(0, min(width, int(math.floor(left))))
        x1 = max(0, min(width, int(math.ceil(right))))
        if x1 > x0:
            intervals.append((x0, x1))
    return intervals


def _mask(width: int, height: int, intervals: list[tuple[int, int]]) -> Image.Image:
    array = np.zeros((height, width), dtype=np.uint8)
    for x0, x1 in intervals:
        array[:, max(0, x0) : min(width, x1)] = 255
    return Image.fromarray(array, mode="L")


def _safe(value: object) -> str:
    rendered = _SAFE.sub("_", str(value or "").strip()).strip("_")
    return rendered[:80] or "sample"


def _sample_key(row: dict, image_a: Path, image_b: Path) -> str:
    pair_id = _safe(row.get("pair_id", "pair"))
    digest = hashlib.sha1(
        f"{row.get('pair_id','')}|{image_a}|{image_b}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{pair_id}__{_safe(image_a.stem)}__{_safe(image_b.stem)}__{digest}"


def _save_image(image: Image.Image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _make_preview(
    image_a: Path,
    mask_a: Image.Image,
    image_b: Path,
    mask_b: Image.Image,
) -> Image.Image:
    with Image.open(image_a) as opened:
        a = opened.convert("RGB")
    with Image.open(image_b) as opened:
        b = opened.convert("RGB")
    rows = [a, mask_a.convert("RGB"), b, mask_b.convert("RGB")]
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "black")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas


def _manifest_paths(args: argparse.Namespace, dataset_root: Path) -> list[Path]:
    values: list[Path] = []
    if args.all_manifests:
        for name in (
            "dataset_manifest.jsonl",
            "train_manifest.jsonl",
            "valid_manifest.jsonl",
            "test_manifest.jsonl",
        ):
            candidate = dataset_root / name
            if candidate.is_file():
                values.append(candidate)
    for value in args.manifest:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        values.append(candidate)
    if not values:
        values.append(dataset_root / "dataset_manifest.jsonl")

    unique: list[Path] = []
    seen = set()
    for path in values:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
    return unique


def _companion_manifest(path: Path) -> Path:
    name = path.name
    if name.endswith(".jsonl"):
        return path.with_name(name[:-6] + "_with_masks.jsonl")
    return path.with_name(name + "_with_masks.jsonl")


def _process_row(
    row: dict,
    dataset_root: Path,
    output_root: Path,
    preview_root: Path,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    updated = deepcopy(row)
    side_a = updated.get("A")
    side_b = updated.get("B")
    if not isinstance(side_a, dict) or not isinstance(side_b, dict):
        raise ValueError("Manifest row must contain dictionary sides A and B")

    image_a = _resolve_path(dataset_root, side_a["line_image_path"])
    image_b = _resolve_path(dataset_root, side_b["line_image_path"])
    text_a = (
        _resolve_path(dataset_root, side_a[args.text_key])
        if side_a.get(args.text_key)
        else None
    )
    text_b = (
        _resolve_path(dataset_root, side_b[args.text_key])
        if side_b.get(args.text_key)
        else None
    )

    with Image.open(image_a) as opened:
        width_a, height_a = opened.size
    with Image.open(image_b) as opened:
        width_b, height_b = opened.size

    ann_a = _load_annotations(image_a, width_a, height_a)
    ann_b = _load_annotations(image_b, width_b, height_b)
    if not ann_a.boxes:
        raise ValueError(
            f"No bbox annotations for A ({image_a.name}); status={ann_a.status} error={ann_a.error}"
        )
    if not ann_b.boxes:
        raise ValueError(
            f"No bbox annotations for B ({image_b.name}); status={ann_b.status} error={ann_b.error}"
        )

    boxes_a, text_source_a = _fill_missing_box_text(_reading_order(ann_a.boxes), text_a)
    boxes_b, text_source_b = _fill_missing_box_text(_reading_order(ann_b.boxes), text_b)
    labels_a = [_normalise_text(box.text) for box in boxes_a]
    labels_b = [_normalise_text(box.text) for box in boxes_b]
    pairs = _lcs_pairs(labels_a, labels_b)
    if not pairs:
        raise ValueError(
            "No order-preserving shared bbox text was found between A and B; "
            f"A_labels={sum(bool(v) for v in labels_a)}/{len(labels_a)} "
            f"B_labels={sum(bool(v) for v in labels_b)}/{len(labels_b)}"
        )

    runs = _consecutive_runs(pairs)
    intervals_a = _run_intervals(
        boxes_a, runs, 0, width_a, args.box_coordinate_space
    )
    intervals_b = _run_intervals(
        boxes_b, runs, 1, width_b, args.box_coordinate_space
    )
    if not intervals_a or not intervals_b:
        raise ValueError("Matched bbox text produced no non-empty pixel intervals")

    mask_a = _mask(width_a, height_a, intervals_a)
    mask_b = _mask(width_b, height_b, intervals_b)
    key = _sample_key(row, image_a, image_b)
    sample_dir = output_root / key
    mask_a_path = sample_dir / "A_mask.png"
    mask_b_path = sample_dir / "B_mask.png"
    _save_image(mask_a, mask_a_path, args.overwrite)
    _save_image(mask_b, mask_b_path, args.overwrite)

    if args.preview:
        preview = _make_preview(image_a, mask_a, image_b, mask_b)
        _save_image(preview, preview_root / f"{key}.png", args.overwrite)

    side_a["alignment_mask_path"] = _relative_or_absolute(mask_a_path, dataset_root)
    side_b["alignment_mask_path"] = _relative_or_absolute(mask_b_path, dataset_root)
    updated["alignment_mask_meta"] = {
        "method": "bbox_text_lcs_consecutive_runs",
        "shared_subword_boxes": len(pairs),
        "consecutive_runs": len(runs),
        "A": {
            "box_count": len(boxes_a),
            "matched_box_count": len({pair[0] for pair in pairs}),
            "intervals_x": [list(interval) for interval in intervals_a],
            "bbox_annotation_path": ann_a.workbook,
            "bbox_annotation_status": ann_a.status,
            "bbox_text_source": text_source_a,
            "mask_size": [width_a, height_a],
        },
        "B": {
            "box_count": len(boxes_b),
            "matched_box_count": len({pair[1] for pair in pairs}),
            "intervals_x": [list(interval) for interval in intervals_b],
            "bbox_annotation_path": ann_b.workbook,
            "bbox_annotation_status": ann_b.status,
            "bbox_text_source": text_source_b,
            "mask_size": [width_b, height_b],
        },
    }

    report = {
        "status": "ok",
        "pair_id": str(row.get("pair_id", "")),
        "sample_key": key,
        "image_A": str(image_a),
        "image_B": str(image_b),
        "boxes_A": len(boxes_a),
        "boxes_B": len(boxes_b),
        "matched_boxes": len(pairs),
        "consecutive_runs": len(runs),
        "intervals_A": intervals_a,
        "intervals_B": intervals_b,
        "mask_A": str(mask_a_path),
        "mask_B": str(mask_b_path),
    }
    return updated, report


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root / "alignment_masks"
    )
    preview_root = (
        args.preview_root.expanduser().resolve()
        if args.preview_root is not None
        else output_root / "preview"
    )
    manifests = _manifest_paths(args, dataset_root)

    if args.write_manifests and args.limit > 0:
        raise SystemExit(
            "Refusing --write-manifests with --limit: a partial companion manifest would be misleading. "
            "Run the preview without --write-manifests, then run the full command."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "alignment_mask_report.jsonl"
    reports: list[dict] = []
    processed_total = 0
    ok_total = 0
    failed_total = 0
    cache: dict[str, tuple[dict, dict]] = {}

    for manifest_path in manifests:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        updated_rows: list[dict] = []
        for row_index, row in enumerate(rows):
            if row_index < max(0, args.start_index):
                updated_rows.append(row)
                continue
            if args.limit > 0 and processed_total >= args.limit:
                updated_rows.append(row)
                continue

            processed_total += 1
            cache_key = json.dumps(
                {
                    "pair_id": row.get("pair_id"),
                    "A": (row.get("A") or {}).get("line_image_path"),
                    "B": (row.get("B") or {}).get("line_image_path"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            try:
                if cache_key in cache:
                    updated, report = deepcopy(cache[cache_key][0]), dict(cache[cache_key][1])
                else:
                    updated, report = _process_row(
                        row, dataset_root, output_root, preview_root, args
                    )
                    cache[cache_key] = (deepcopy(updated), dict(report))
                ok_total += 1
                print(
                    f"OK manifest={manifest_path.name} row={row_index} "
                    f"pair={report['pair_id']} matched={report['matched_boxes']} "
                    f"runs={report['consecutive_runs']} "
                    f"A={report['intervals_A']} B={report['intervals_B']}",
                    flush=True,
                )
            except Exception as exc:
                failed_total += 1
                updated = deepcopy(row)
                updated["alignment_mask_meta"] = {
                    "method": "bbox_text_lcs_consecutive_runs",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report = {
                    "status": "error",
                    "manifest": manifest_path.name,
                    "row": row_index,
                    "pair_id": str(row.get("pair_id", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"ERROR manifest={manifest_path.name} row={row_index} "
                    f"pair={row.get('pair_id','')} {report['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.strict:
                    raise

            report["manifest"] = manifest_path.name
            report["row"] = row_index
            reports.append(report)
            updated_rows.append(updated)

        if args.write_manifests:
            destination = _companion_manifest(manifest_path)
            with destination.open("w", encoding="utf-8") as handle:
                for row in updated_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"WROTE manifest={destination}", flush=True)

    with report_path.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")

    print(
        "SUMMARY "
        f"processed={processed_total} ok={ok_total} failed={failed_total} "
        f"output_root={output_root} report={report_path}",
        flush=True,
    )
    if args.preview:
        print(f"PREVIEW_ROOT {preview_root}", flush=True)
    return 1 if args.strict and failed_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
