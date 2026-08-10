#!/usr/bin/env python3
"""Build full-height ground-truth masks for aligned real Arabic line regions.

The real dataset stores OCR/subword boxes at page level in ``debug/bboxes.json``.
Those records are clustered into page lines using ``page_meta.json``'s
``line_threshold_used``.  The requested ``line_N.png`` then receives the boxes
from line N, ordered right-to-left.  A/B box texts are aligned with an exact
order-preserving LCS; consecutive matched boxes are merged into full-height
white x-intervals on an otherwise black mask.

Source manifests are never overwritten.  ``--write-manifests`` creates sibling
``*_with_masks.jsonl`` files.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

_SCRIPT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
PROJECT_ROOT = _SCRIPT_ROOT if (_SCRIPT_ROOT / "Evaluation").exists() else Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Evaluation.real_subword_box_json import load_json_annotations
    from Evaluation.real_subword_box_metrics import load_line_annotations
except Exception:
    load_json_annotations = None
    load_line_annotations = None

_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_LINE_NUMBER = re.compile(r"(?:line[_\-\s]*)?0*(\d+)", re.IGNORECASE)
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source_row: int = -1

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y0 + self.y1)


@dataclass(frozen=True)
class LineAnnotations:
    boxes: tuple[Box, ...]
    source: str
    status: str
    detail: str = ""
    clustering_method: str = ""
    clustered_lines: int = 0
    expected_lines: int = 0
    threshold: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full-height binary masks for aligned real line pairs.")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "DataSet" / "ArabicDatasetRealAug10K")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--all-manifests", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--preview-root", type=Path, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--write-manifests", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--text-key", default="text_original_path")
    return parser.parse_args()


def _normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _ARABIC_DIACRITICS.sub("", text.replace("ـ", ""))
    return "".join(text.split())


def _line_index(image_path: Path) -> int | None:
    match = _LINE_NUMBER.search(image_path.stem)
    return int(match.group(1)) if match else None


def _side_root(image_path: Path) -> Path:
    for parent in image_path.parents:
        if parent.name in {"A", "B"}:
            return parent
    return image_path.parent


def _resolve_path(dataset_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    candidates = (dataset_root / path, dataset_root.parent / path, PROJECT_ROOT / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {value!r}; tried: " + ", ".join(str(p) for p in candidates))


def _to_float(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _page_box(record: object, row: int) -> Box | None:
    if not isinstance(record, dict):
        return None
    x0 = _to_float(record.get("x1", record.get("x0", record.get("left"))))
    y0 = _to_float(record.get("y1", record.get("y0", record.get("top"))))
    x1 = _to_float(record.get("x2", record.get("right")))
    y1 = _to_float(record.get("y2", record.get("bottom")))
    if x1 is None and x0 is not None:
        width = _to_float(record.get("w", record.get("width")))
        x1 = None if width is None else x0 + width
    if y1 is None and y0 is not None:
        height = _to_float(record.get("h", record.get("height")))
        y1 = None if height is None else y0 + height
    if None in (x0, y0, x1, y1):
        return None
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    if right <= left or bottom <= top:
        return None
    return Box(str(record.get("text", "")).strip(), left, top, right, bottom, row)


def _threshold_clusters(boxes: list[Box], threshold: float) -> list[list[Box]]:
    ordered = sorted(boxes, key=lambda box: (box.center_y, box.center_x))
    clusters: list[list[Box]] = []
    centers: list[float] = []
    for box in ordered:
        if not clusters or abs(box.center_y - centers[-1]) > threshold:
            clusters.append([box]); centers.append(box.center_y)
        else:
            clusters[-1].append(box)
            centers[-1] = float(np.median([item.center_y for item in clusters[-1]]))
    return clusters


def _largest_gap_clusters(boxes: list[Box], expected: int) -> list[list[Box]]:
    ordered = sorted(boxes, key=lambda box: (box.center_y, box.center_x))
    if expected <= 1 or len(ordered) <= 1:
        return [ordered] if ordered else []
    gaps = [(ordered[i + 1].center_y - ordered[i].center_y, i) for i in range(len(ordered) - 1)]
    cut_after = {index for _gap, index in sorted(gaps, reverse=True)[: max(0, expected - 1)]}
    clusters: list[list[Box]] = []
    current: list[Box] = []
    for index, box in enumerate(ordered):
        current.append(box)
        if index in cut_after:
            clusters.append(current); current = []
    if current:
        clusters.append(current)
    return clusters


def _page_width(side_root: Path, boxes: list[Box], line_width: int) -> float:
    original = side_root / "original_image.png"
    if original.is_file():
        try:
            with Image.open(original) as image:
                return float(image.width)
        except Exception:
            pass
    maximum = max((box.x1 for box in boxes), default=float(line_width))
    return max(float(line_width), float(maximum))


def _load_page_level_line_annotations(image_path: Path, line_width: int) -> LineAnnotations:
    side = _side_root(image_path)
    bbox_path = side / "debug" / "bboxes.json"
    meta_path = side / "page_meta.json"
    if not bbox_path.is_file():
        return LineAnnotations((), str(bbox_path), "missing", "debug/bboxes.json not found")
    try:
        payload = json.loads(bbox_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return LineAnnotations((), str(bbox_path), "parse_error", f"{type(exc).__name__}: {exc}")
    if not isinstance(payload, list):
        return LineAnnotations((), str(bbox_path), "bad_schema", f"expected list, got {type(payload).__name__}")

    boxes = [box for row, record in enumerate(payload, start=1) if (box := _page_box(record, row)) is not None]
    if not boxes:
        return LineAnnotations((), str(bbox_path), "no_boxes", "no valid x1/y1/x2/y2 bbox records")

    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    threshold = float(_to_float(meta.get("line_threshold_used"), 42.0) or 42.0)
    expected = int(_to_float(meta.get("num_lines"), 0) or 0)
    clusters = _threshold_clusters(boxes, threshold)
    method = "page_y_threshold"
    if expected > 0 and len(clusters) != expected:
        clusters = _largest_gap_clusters(boxes, expected)
        method = "page_y_largest_gaps_fallback"
    clusters = sorted(clusters, key=lambda group: float(np.mean([box.center_y for box in group])))

    requested = _line_index(image_path)
    if requested is None or requested < 1 or requested > len(clusters):
        return LineAnnotations((), str(bbox_path), "line_not_found", f"requested line={requested}, clustered_lines={len(clusters)}, expected={expected}", method, len(clusters), expected, threshold)

    selected = clusters[requested - 1]
    source_width = _page_width(side, boxes, line_width)
    scale_x = float(line_width) / max(1.0, source_width)
    line_boxes = [Box(box.text, box.x0 * scale_x, 0.0, box.x1 * scale_x, 1.0, box.source_row) for box in selected]
    line_boxes = sorted(line_boxes, key=lambda box: (-box.center_x, box.source_row))
    detail = f"page_boxes={len(boxes)} selected={len(line_boxes)} line={requested} source_width={source_width:.1f} line_width={line_width} scale_x={scale_x:.6f}"
    return LineAnnotations(tuple(line_boxes), str(bbox_path), "ok", detail, method, len(clusters), expected, threshold)


def _convert_external_annotations(annotations, source: str) -> LineAnnotations:
    boxes = tuple(Box(str(box.text), float(box.x0), float(box.y0), float(box.x1), float(box.y1), int(getattr(box, "source_row", -1))) for box in annotations.boxes)
    return LineAnnotations(boxes, source, "ok" if boxes else str(getattr(annotations, "status", "no_boxes")), str(getattr(annotations, "error", "")))


def _load_annotations(image_path: Path, width: int, height: int) -> LineAnnotations:
    if load_json_annotations is not None:
        try:
            annotations = load_json_annotations(image_path)
            if getattr(annotations, "boxes", ()):
                return _convert_external_annotations(annotations, str(getattr(annotations, "workbook", "json")))
        except Exception:
            pass
    page_level = _load_page_level_line_annotations(image_path, width)
    if page_level.boxes:
        return page_level
    if load_line_annotations is not None:
        try:
            annotations = load_line_annotations(image_path, width, height)
            if getattr(annotations, "boxes", ()):
                return _convert_external_annotations(annotations, str(getattr(annotations, "workbook", "excel")))
        except Exception:
            pass
    return page_level


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


def _fill_missing_box_text(boxes: list[Box], text_path: Path | None) -> tuple[list[Box], str]:
    if boxes and all(_normalise_text(box.text) for box in boxes):
        return boxes, "bbox_labels"
    if text_path is None or not text_path.is_file():
        return boxes, "missing_text"
    units = _read_units(text_path)
    if len(units) != len(boxes):
        return boxes, f"text_count_mismatch:{len(units)}!={len(boxes)}"
    return [Box(unit, box.x0, box.y0, box.x1, box.y1, box.source_row) for unit, box in zip(units, boxes)], "transcript_units"


def _lcs_pairs(left: list[str], right: list[str]) -> list[tuple[int, int]]:
    n, m = len(left), len(right)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i, j] = 1 + dp[i + 1, j + 1] if left[i] and left[i] == right[j] else max(dp[i + 1, j], dp[i, j + 1])
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if left[i] and left[i] == right[j] and dp[i, j] == 1 + dp[i + 1, j + 1]:
            result.append((i, j)); i += 1; j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return result


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


def _run_intervals(boxes: list[Box], runs: list[list[tuple[int, int]]], side_index: int, width: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for run in runs:
        selected = [boxes[pair[side_index]] for pair in run]
        x0 = max(0, min(width, int(math.floor(min(box.x0 for box in selected)))))
        x1 = max(0, min(width, int(math.ceil(max(box.x1 for box in selected)))))
        if x1 > x0:
            intervals.append((x0, x1))
    return intervals


def _mask(width: int, height: int, intervals: list[tuple[int, int]]) -> Image.Image:
    array = np.zeros((height, width), dtype=np.uint8)
    for x0, x1 in intervals:
        array[:, x0:x1] = 255
    return Image.fromarray(array, mode="L")


def _safe(value: object) -> str:
    rendered = _SAFE.sub("_", str(value or "").strip()).strip("_")
    return rendered[:80] or "sample"


def _sample_key(row: dict, image_a: Path, image_b: Path) -> str:
    digest = hashlib.sha1(f"{row.get('pair_id','')}|{image_a}|{image_b}".encode("utf-8")).hexdigest()[:10]
    return f"{_safe(row.get('pair_id','pair'))}__{_safe(image_a.stem)}__{_safe(image_b.stem)}__{digest}"


def _save_image(image: Image.Image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _preview(image_a: Path, mask_a: Image.Image, image_b: Path, mask_b: Image.Image) -> Image.Image:
    with Image.open(image_a) as opened:
        a = opened.convert("RGB")
    with Image.open(image_b) as opened:
        b = opened.convert("RGB")
    rows = [a, mask_a.convert("RGB"), b, mask_b.convert("RGB")]
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "black")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y)); y += row.height
    return canvas


def _manifest_paths(args: argparse.Namespace, root: Path) -> list[Path]:
    values: list[Path] = []
    if args.all_manifests:
        for name in ("dataset_manifest.jsonl", "train_manifest.jsonl", "valid_manifest.jsonl", "test_manifest.jsonl"):
            if (root / name).is_file():
                values.append(root / name)
    for value in args.manifest:
        path = Path(value).expanduser(); values.append(path if path.is_absolute() else root / path)
    if not values:
        values = [root / "dataset_manifest.jsonl"]
    unique: list[Path] = []
    seen = set()
    for path in values:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved); unique.append(resolved)
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(path)
    return unique


def _companion(path: Path) -> Path:
    return path.with_name(path.stem + "_with_masks.jsonl")


def _process_row(row: dict, root: Path, output_root: Path, preview_root: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    updated = deepcopy(row)
    side_a, side_b = updated.get("A"), updated.get("B")
    if not isinstance(side_a, dict) or not isinstance(side_b, dict):
        raise ValueError("manifest row must contain dictionary A and B sides")
    image_a = _resolve_path(root, side_a["line_image_path"]); image_b = _resolve_path(root, side_b["line_image_path"])
    text_a = _resolve_path(root, side_a[args.text_key]) if side_a.get(args.text_key) else None
    text_b = _resolve_path(root, side_b[args.text_key]) if side_b.get(args.text_key) else None
    with Image.open(image_a) as image: width_a, height_a = image.size
    with Image.open(image_b) as image: width_b, height_b = image.size
    ann_a = _load_annotations(image_a, width_a, height_a); ann_b = _load_annotations(image_b, width_b, height_b)
    if not ann_a.boxes: raise ValueError(f"No bbox annotations for A ({image_a.name}); status={ann_a.status} detail={ann_a.detail}")
    if not ann_b.boxes: raise ValueError(f"No bbox annotations for B ({image_b.name}); status={ann_b.status} detail={ann_b.detail}")
    boxes_a, text_source_a = _fill_missing_box_text(list(ann_a.boxes), text_a); boxes_b, text_source_b = _fill_missing_box_text(list(ann_b.boxes), text_b)
    labels_a = [_normalise_text(box.text) for box in boxes_a]; labels_b = [_normalise_text(box.text) for box in boxes_b]
    pairs = _lcs_pairs(labels_a, labels_b)
    if not pairs: raise ValueError(f"No order-preserving shared bbox text; A={labels_a} B={labels_b}")
    runs = _consecutive_runs(pairs); intervals_a = _run_intervals(boxes_a, runs, 0, width_a); intervals_b = _run_intervals(boxes_b, runs, 1, width_b)
    if not intervals_a or not intervals_b: raise ValueError("matched boxes produced no non-empty intervals")
    mask_a, mask_b = _mask(width_a, height_a, intervals_a), _mask(width_b, height_b, intervals_b)
    key = _sample_key(row, image_a, image_b); sample_dir = output_root / key
    mask_a_path, mask_b_path = sample_dir / "A_mask.png", sample_dir / "B_mask.png"
    _save_image(mask_a, mask_a_path, args.overwrite); _save_image(mask_b, mask_b_path, args.overwrite)
    if args.preview: _save_image(_preview(image_a, mask_a, image_b, mask_b), preview_root / f"{key}.png", args.overwrite)
    side_a["alignment_mask_path"] = _relative(mask_a_path, root); side_b["alignment_mask_path"] = _relative(mask_b_path, root)
    updated["alignment_mask_meta"] = {
        "method": "page_bbox_line_cluster_text_lcs_consecutive_runs", "shared_subword_boxes": len(pairs), "consecutive_runs": len(runs),
        "A": {"box_count": len(boxes_a), "matched_box_count": len({i for i, _ in pairs}), "intervals_x": [list(v) for v in intervals_a], "bbox_annotation_path": ann_a.source, "bbox_status": ann_a.status, "bbox_detail": ann_a.detail, "bbox_text_source": text_source_a, "clustering_method": ann_a.clustering_method, "clustered_lines": ann_a.clustered_lines, "expected_lines": ann_a.expected_lines, "line_threshold": ann_a.threshold, "mask_size": [width_a, height_a]},
        "B": {"box_count": len(boxes_b), "matched_box_count": len({j for _, j in pairs}), "intervals_x": [list(v) for v in intervals_b], "bbox_annotation_path": ann_b.source, "bbox_status": ann_b.status, "bbox_detail": ann_b.detail, "bbox_text_source": text_source_b, "clustering_method": ann_b.clustering_method, "clustered_lines": ann_b.clustered_lines, "expected_lines": ann_b.expected_lines, "line_threshold": ann_b.threshold, "mask_size": [width_b, height_b]},
    }
    report = {"status": "ok", "pair_id": str(row.get("pair_id", "")), "sample_key": key, "image_A": str(image_a), "image_B": str(image_b), "boxes_A": len(boxes_a), "boxes_B": len(boxes_b), "matched_boxes": len(pairs), "consecutive_runs": len(runs), "intervals_A": intervals_a, "intervals_B": intervals_b, "A_cluster_method": ann_a.clustering_method, "A_clustered_lines": ann_a.clustered_lines, "A_expected_lines": ann_a.expected_lines, "B_cluster_method": ann_b.clustering_method, "B_clustered_lines": ann_b.clustered_lines, "B_expected_lines": ann_b.expected_lines, "mask_A": str(mask_a_path), "mask_B": str(mask_b_path)}
    return updated, report


def main() -> int:
    args = parse_args(); root = args.dataset_root.expanduser().resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    output_root = args.output_root.expanduser().resolve() if args.output_root else root / "alignment_masks"
    preview_root = args.preview_root.expanduser().resolve() if args.preview_root else output_root / "preview"
    manifests = _manifest_paths(args, root)
    if args.write_manifests and args.limit > 0: raise SystemExit("Refusing --write-manifests with --limit; preview first, then run the full manifest.")
    output_root.mkdir(parents=True, exist_ok=True); reports: list[dict] = []; processed = ok = failed = 0; cache: dict[str, tuple[dict, dict]] = {}
    for manifest in manifests:
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]; updated_rows: list[dict] = []
        for row_index, row in enumerate(rows):
            if row_index < max(0, args.start_index) or (args.limit > 0 and processed >= args.limit): updated_rows.append(row); continue
            processed += 1
            cache_key = json.dumps({"pair": row.get("pair_id"), "A": (row.get("A") or {}).get("line_image_path"), "B": (row.get("B") or {}).get("line_image_path")}, sort_keys=True, ensure_ascii=False)
            try:
                if cache_key in cache: updated, report = deepcopy(cache[cache_key][0]), dict(cache[cache_key][1])
                else:
                    updated, report = _process_row(row, root, output_root, preview_root, args); cache[cache_key] = (deepcopy(updated), dict(report))
                ok += 1
                print(f"OK manifest={manifest.name} row={row_index} pair={report['pair_id']} boxes={report['boxes_A']}/{report['boxes_B']} matched={report['matched_boxes']} runs={report['consecutive_runs']} A_lines={report['A_clustered_lines']}/{report['A_expected_lines']} B_lines={report['B_clustered_lines']}/{report['B_expected_lines']} A={report['intervals_A']} B={report['intervals_B']}", flush=True)
            except Exception as exc:
                failed += 1; updated = deepcopy(row); updated["alignment_mask_meta"] = {"method": "page_bbox_line_cluster_text_lcs_consecutive_runs", "status": "error", "error": f"{type(exc).__name__}: {exc}"}; report = {"status": "error", "pair_id": str(row.get("pair_id", "")), "error": f"{type(exc).__name__}: {exc}"}
                print(f"ERROR manifest={manifest.name} row={row_index} pair={row.get('pair_id','')} {report['error']}", file=sys.stderr, flush=True)
                if args.strict: raise
            report.update({"manifest": manifest.name, "row": row_index}); reports.append(report); updated_rows.append(updated)
        if args.write_manifests:
            destination = _companion(manifest); destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated_rows), encoding="utf-8"); print(f"WROTE manifest={destination}", flush=True)
    report_path = output_root / "alignment_mask_report.jsonl"; report_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reports), encoding="utf-8")
    print(f"SUMMARY processed={processed} ok={ok} failed={failed} output_root={output_root} report={report_path}", flush=True)
    if args.preview: print(f"PREVIEW_ROOT {preview_root}", flush=True)
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
