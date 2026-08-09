#!/usr/bin/env python3
"""Normalize a standalone real+augmented dataset to one line pair per folder.

The input dataset may contain original page-pair folders with many line images,
while each manifest row refers to exactly one A line and one B line. This script
materializes every manifest row as its own physical pair folder so the entire
output follows one rule:

    one pair folder == one A line + one B line

The existing manifest pairing is authoritative. Lines are never paired by equal
filenames; e.g. an A/line_08 row may legitimately pair with B/line_07.

The output preserves labels, scores, sample_origin, and the existing explicit
train/valid/test assignment. The original manifest pair id is retained as
``source_manifest_pair_id`` so all provenance remains available.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Iterable

from PIL import Image

from Evaluation.real_subword_box_json import load_json_annotations
from Evaluation.real_flat_page_bbox import load_flat_page_line_annotations


POSITIVE_LABELS = {"high_match", "medium_match"}
NEGATIVE_LABEL = "no_shared_content"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _resolve(root: Path, value) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (root / path, root.parent / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _prepare_output(source_root: Path, output_root: Path, overwrite: bool) -> None:
    if source_root == output_root:
        raise ValueError("Input and output dataset directories must be different.")
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output must not be nested inside the source dataset.")

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output is not empty: {output_root}; pass --overwrite"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _copy_text(source: Path | None, fallback: Path, destination: Path) -> None:
    source = source if source is not None and source.is_file() else fallback
    if not source.is_file():
        raise FileNotFoundError(f"Transcript does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _load_boxes(image_path: Path):
    annotations = load_json_annotations(image_path)
    if annotations.status == "ok" and annotations.boxes:
        return annotations
    fallback = load_flat_page_line_annotations(image_path)
    if fallback.status == "ok" and fallback.boxes:
        return fallback
    return None


def _bbox_payload(image_path: Path, text: str, annotations) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size
    boxes = []
    for box in annotations.boxes:
        boxes.append(
            {
                "text": str(box.text),
                "x0": float(box.x0),
                "y0": float(box.y0),
                "x1": float(box.x1),
                "y1": float(box.y1),
                "source_row": int(box.source_row),
            }
        )
    return {
        "image": "line_01.png",
        "width": int(width),
        "height": int(height),
        "reading_order": "rtl",
        "text": text,
        "boxes": boxes,
    }


def _flat_boxes(payload: dict) -> list[dict]:
    output = []
    for box in payload.get("boxes", []):
        x0 = float(box["x0"])
        y0 = float(box["y0"])
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        output.append(
            {
                "x1": round(x0, 3),
                "y1": round(y0, 3),
                "x2": round(x1, 3),
                "y2": round(y1, 3),
                "w": round(x1 - x0, 3),
                "h": round(y1 - y0, 3),
                "cx": round((x0 + x1) / 2.0, 3),
                "cy": round((y0 + y1) / 2.0, 3),
                "text": str(box.get("text", "")),
            }
        )
    return output


def _materialize_side(
    source_root: Path,
    source_side: dict,
    destination_side: Path,
    source_pair_id: str,
    side_name: str,
) -> tuple[dict, bool]:
    image_path = _resolve(source_root, source_side.get("line_image_path"))
    original_text = _resolve(source_root, source_side.get("text_original_path"))
    if image_path is None or not image_path.is_file():
        raise FileNotFoundError(
            f"{source_pair_id} {side_name}: missing line image {image_path}"
        )
    if original_text is None or not original_text.is_file():
        raise FileNotFoundError(
            f"{source_pair_id} {side_name}: missing original text {original_text}"
        )

    image_rel = Path("linesImages/line_01.png")
    original_rel = Path("text/final/original/line_01.txt")
    tashkeel_rel = Path("text/final/tashkeel/line_01.txt")
    raw_rel = Path("text/raw/line_01.txt")
    bbox_rel = Path("bbox.json")
    debug_bbox_rel = Path("debug/bboxes.json")
    source_meta_rel = Path("debug/source.json")

    (destination_side / image_rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination_side / image_rel)
    _copy_text(original_text, original_text, destination_side / original_rel)
    _copy_text(
        _resolve(source_root, source_side.get("text_tashkeel_path")),
        original_text,
        destination_side / tashkeel_rel,
    )
    _copy_text(
        _resolve(source_root, source_side.get("text_raw_path")),
        original_text,
        destination_side / raw_rel,
    )

    text = original_text.read_text(encoding="utf-8").strip()
    annotations = _load_boxes(image_path)
    has_bbox = annotations is not None
    if has_bbox:
        payload = _bbox_payload(image_path, text, annotations)
        (destination_side / bbox_rel).parent.mkdir(parents=True, exist_ok=True)
        (destination_side / bbox_rel).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (destination_side / debug_bbox_rel).parent.mkdir(parents=True, exist_ok=True)
        (destination_side / debug_bbox_rel).write_text(
            json.dumps(_flat_boxes(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    source_meta = {
        "source_manifest_pair_id": source_pair_id,
        "side": side_name,
        "source_line_idx": source_side.get("line_idx"),
        "source_line_image_path": str(source_side.get("line_image_path", "")),
        "source_text_original_path": str(source_side.get("text_original_path", "")),
        "bbox_recovered": bool(has_bbox),
        "bbox_source": annotations.sheet if annotations is not None else "",
    }
    (destination_side / source_meta_rel).parent.mkdir(parents=True, exist_ok=True)
    (destination_side / source_meta_rel).write_text(
        json.dumps(source_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    side = copy.deepcopy(source_side)
    side["source_line_idx"] = source_side.get("line_idx")
    side["source_line_image_path"] = str(source_side.get("line_image_path", ""))
    side["source_text_original_path"] = str(source_side.get("text_original_path", ""))
    side["line_idx"] = 1
    side["line_image_path"] = image_rel.as_posix()
    side["text_original_path"] = original_rel.as_posix()
    side["text_tashkeel_path"] = tashkeel_rel.as_posix()
    side["text_raw_path"] = raw_rel.as_posix()
    side["page_dir"] = "."
    if has_bbox:
        side["bbox_path"] = bbox_rel.as_posix()
    else:
        side.pop("bbox_path", None)
        side["bbox_status"] = "unavailable_after_normalization"
    return side, has_bbox


def _prefix_side_paths(side: dict, prefix: Path) -> None:
    for key in (
        "line_image_path",
        "text_original_path",
        "text_tashkeel_path",
        "text_raw_path",
        "bbox_path",
    ):
        if side.get(key):
            side[key] = (prefix / side[key]).as_posix()
    side["page_dir"] = prefix.as_posix()


def _materialize_row(
    source_root: Path,
    output_root: Path,
    row: dict,
    index: int,
) -> tuple[dict, int]:
    new_pair_id = f"pair_{index:06d}"
    pair_root = output_root / "DatasetPairs" / "page_pairs" / new_pair_id
    if pair_root.exists():
        raise FileExistsError(f"Duplicate output pair: {pair_root}")

    source_pair_id = str(row.get("pair_id", f"source_row_{index}"))
    if not isinstance(row.get("A"), dict) or not isinstance(row.get("B"), dict):
        raise ValueError(f"Manifest row {index} does not contain A/B dictionaries")

    side_a, bbox_a = _materialize_side(
        source_root, row["A"], pair_root / "A", source_pair_id, "A"
    )
    side_b, bbox_b = _materialize_side(
        source_root, row["B"], pair_root / "B", source_pair_id, "B"
    )

    record = copy.deepcopy(row)
    record["source_manifest_pair_id"] = source_pair_id
    record["source_manifest_row"] = index
    record["pair_id"] = new_pair_id
    record["one_line_pair_layout"] = True
    record["A"] = side_a
    record["B"] = side_b

    prefix = Path("DatasetPairs") / "page_pairs" / new_pair_id
    _prefix_side_paths(record["A"], prefix / "A")
    _prefix_side_paths(record["B"], prefix / "B")

    pair_meta = {
        "pair_id": new_pair_id,
        "source_manifest_pair_id": source_pair_id,
        "source_manifest_row": index,
        "split": record.get("split"),
        "label_type": record.get("label_type"),
        "sample_origin": record.get("sample_origin"),
        "source_A_line_idx": row["A"].get("line_idx"),
        "source_B_line_idx": row["B"].get("line_idx"),
    }
    (pair_root / "sample.json").write_text(
        json.dumps(pair_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record, int(bbox_a) + int(bbox_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("DataSet/ArabicDatasetRealAug10K")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DataSet/ArabicDatasetRealAug10KOneLine"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.data_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    manifest_path = source_root / "dataset_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    _prepare_output(source_root, output_root, args.overwrite)
    rows = _read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError("Input dataset manifest is empty")

    normalized: list[dict] = []
    bbox_sides = 0
    origin_counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        record, recovered = _materialize_row(
            source_root, output_root, row, index
        )
        normalized.append(record)
        bbox_sides += recovered
        origin = str(record.get("sample_origin", "unspecified"))
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        if index % 250 == 0 or index == len(rows):
            print(f"materialized {index}/{len(rows)} line-pair folders", flush=True)

    train_rows = [
        row for row in normalized
        if row.get("split") == "train" and str(row.get("label_type", "")) in POSITIVE_LABELS
    ]
    valid_rows = [
        row for row in normalized
        if row.get("split") == "valid" and str(row.get("label_type", "")) in POSITIVE_LABELS
    ]
    test_rows = [
        row for row in normalized
        if row.get("split") == "test" and str(row.get("label_type", "")) in POSITIVE_LABELS
    ]
    negative_rows = [
        row for row in normalized
        if row.get("split") == "negative" or str(row.get("label_type", "")) == NEGATIVE_LABEL
    ]

    _write_jsonl(output_root / "dataset_manifest.jsonl", normalized)
    _write_jsonl(output_root / "train_manifest.jsonl", train_rows)
    _write_jsonl(output_root / "valid_manifest.jsonl", valid_rows)
    _write_jsonl(output_root / "test_manifest.jsonl", test_rows)
    _write_jsonl(output_root / "no_shared_content_manifest.jsonl", negative_rows)

    summary = {
        "source_dataset": str(source_root),
        "output_dataset": str(output_root),
        "layout": "one physical folder per manifest line pair",
        "pairing_rule": "existing manifest A/B paths; never equal-filename pairing",
        "total_pair_folders": len(normalized),
        "train_positive_pairs": len(train_rows),
        "valid_positive_pairs": len(valid_rows),
        "test_positive_pairs": len(test_rows),
        "negative_pairs": len(negative_rows),
        "sample_origin_counts": dict(sorted(origin_counts.items())),
        "bbox_sides_recovered": bbox_sides,
        "bbox_sides_total": 2 * len(normalized),
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Normalized dataset ready: {output_root}", flush=True)


if __name__ == "__main__":
    main()
