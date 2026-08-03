"""Excel subword-box metrics for the legacy main-branch evaluator."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
from openpyxl import load_workbook


_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_LINE_NUMBER = re.compile(r"(?:line[_\-\s]*)?0*(\d+)", re.IGNORECASE)

_ALIASES = {
    "text": {"subword", "sub_word", "connected_subword", "word", "token", "text", "label", "transcription", "content", "arabic", "string", "نص", "كلمة", "مقطع", "الكلمة"},
    "line": {"line", "line_idx", "line_index", "line_number", "line_no", "row", "row_idx", "row_index", "سطر", "رقم_السطر"},
    "image": {"image", "image_name", "filename", "file_name", "line_image", "image_path", "path", "اسم_الصورة"},
    "bbox": {"bbox", "bounding_box", "boundingbox", "box", "coordinates", "coords", "rectangle", "المربع", "الاحداثيات"},
    "x0": {"x0", "xmin", "x_min", "left", "start_x", "x_start", "min_x", "bbox_x0", "box_x0"},
    "y0": {"y0", "ymin", "y_min", "top", "start_y", "y_start", "min_y", "bbox_y0", "box_y0"},
    "x1": {"x1", "xmax", "x_max", "right", "end_x", "x_end", "max_x", "bbox_x1", "box_x1"},
    "y1": {"y1", "ymax", "y_max", "bottom", "end_y", "y_end", "max_y", "bbox_y1", "box_y1"},
    "x": {"x", "bbox_x", "box_x"},
    "y": {"y", "bbox_y", "box_y"},
    "width": {"w", "width", "bbox_width", "box_width"},
    "height": {"h", "height", "bbox_height", "box_height"},
}


@dataclass(frozen=True)
class SubwordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source_row: int = -1

    @property
    def width(self):
        return max(0.0, self.x1 - self.x0)

    @property
    def center_x(self):
        return 0.5 * (self.x0 + self.x1)


@dataclass(frozen=True)
class BoxAnnotations:
    boxes: tuple[SubwordBox, ...]
    workbook: str = ""
    sheet: str = ""
    status: str = "missing"
    error: str = ""


def normalize_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("ـ", "")
    return "".join(_DIACRITICS.sub("", text).split())


def _header(value) -> str:
    value = re.sub(r"[\s\-/\\]+", "_", str(value or "").strip().lower())
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE)


def _canonical(value):
    value = _header(value)
    for key, aliases in _ALIASES.items():
        if value in aliases:
            return key
    return None


def _as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _line_index(image_path: Path):
    match = _LINE_NUMBER.search(image_path.stem)
    return int(match.group(1)) if match else None


def _side_root(image_path: Path):
    for parent in image_path.parents:
        if parent.name in {"A", "B"}:
            return parent
    return image_path.parent.parent


def _candidate_score(path: Path, image_path: Path):
    rendered = str(path).lower()
    score = 100 if image_path.stem.lower() in path.stem.lower() else 0
    for token, weight in (("subword", 50), ("bbox", 45), ("bounding", 40), ("box", 35), ("annotation", 30), ("label", 20)):
        if token in rendered:
            score += weight
    return (-score, len(path.parts), str(path))


@lru_cache(maxsize=4096)
def discover_workbook(image_value: str, annotation_root: str = ""):
    image_path = Path(image_value)
    roots = []
    if annotation_root:
        roots.append(Path(annotation_root).expanduser())
    roots.append(_side_root(image_path))
    names = (
        f"{image_path.stem}.xlsx", f"{image_path.stem}.xlsm",
        "subword_boxes.xlsx", "bounding_boxes.xlsx", "boxes.xlsx",
        "annotations.xlsx", "labels.xlsx",
    )
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for directory in (root, root / "annotations", root / "Annotations", root / "bounding_boxes", root / "boxes", root / "excel", root / "Excel", root / "labels"):
            for name in names:
                path = directory / name
                if path.is_file():
                    candidates.append(path.resolve())
        if not candidates:
            candidates.extend(path.resolve() for path in root.rglob("*.xlsx"))
            candidates.extend(path.resolve() for path in root.rglob("*.xlsm"))
    unique = sorted(set(candidates), key=lambda path: _candidate_score(path, image_path))
    return unique[0] if unique else None


def _mapping(row):
    result = {}
    for index, value in enumerate(row):
        key = _canonical(value)
        if key and key not in result:
            result[key] = index
    if "bbox" in result or {"x0", "x1"}.issubset(result) or {"x", "width"}.issubset(result):
        return result
    return {}


def _value(row, mapping, key):
    index = mapping.get(key)
    return row[index] if index is not None and index < len(row) else None


def _bbox_numbers(value):
    values = [float(item) for item in _NUMBER.findall(str(value or ""))]
    if len(values) < 4:
        return None
    x0, y0, third, fourth = values[:4]
    if os.environ.get("REAL_BOX_BBOX_FORMAT", "xyxy").strip().lower() == "xywh":
        return x0, y0, x0 + third, y0 + fourth
    return x0, y0, third, fourth


def _sheet_matches(sheet_name: str, line_index):
    rendered = str(sheet_name).strip().lower()
    if line_index is None or ("line" not in rendered and "سطر" not in rendered):
        return True
    match = _LINE_NUMBER.search(rendered)
    return match is None or int(match.group(1)) == int(line_index)


def _parse_rows(rows, image_path: Path):
    mapping = {}
    header_index = -1
    for index, row in enumerate(rows[:20]):
        mapping = _mapping(row)
        if mapping:
            header_index = index
            break
    if not mapping:
        return []
    line_index = _line_index(image_path)
    boxes = []
    for row_number, row in enumerate(rows[header_index + 1:], start=header_index + 2):
        image_value = _value(row, mapping, "image")
        if image_value not in (None, ""):
            stem = Path(str(image_value)).stem.lower()
            if image_path.stem.lower() not in stem:
                continue
        if "line" in mapping and line_index is not None:
            row_line = _as_float(_value(row, mapping, "line"))
            if row_line is not None and int(row_line) != line_index:
                continue
        if "bbox" in mapping:
            parsed = _bbox_numbers(_value(row, mapping, "bbox"))
            if parsed is None:
                continue
            x0, y0, x1, y1 = parsed
        elif {"x0", "x1"}.issubset(mapping):
            x0 = _as_float(_value(row, mapping, "x0")); x1 = _as_float(_value(row, mapping, "x1"))
            y0 = _as_float(_value(row, mapping, "y0")); y1 = _as_float(_value(row, mapping, "y1"))
        else:
            x0 = _as_float(_value(row, mapping, "x")); y0 = _as_float(_value(row, mapping, "y"))
            width = _as_float(_value(row, mapping, "width")); height = _as_float(_value(row, mapping, "height"))
            x1 = None if x0 is None or width is None else x0 + width
            y1 = None if y0 is None or height is None else y0 + height
        if x0 is None or x1 is None:
            continue
        y0 = 0.0 if y0 is None else y0
        y1 = 1.0 if y1 is None else y1
        left, right = sorted((float(x0), float(x1)))
        top, bottom = sorted((float(y0), float(y1)))
        if right > left:
            boxes.append(SubwordBox(str(_value(row, mapping, "text") or "").strip(), left, top, right, bottom, row_number))
    return sorted(boxes, key=lambda box: (-box.center_x, box.y0, box.source_row))


def load_annotations(image_path, annotation_root=""):
    image_path = Path(image_path)
    workbook = discover_workbook(str(image_path), str(annotation_root or ""))
    if workbook is None:
        return BoxAnnotations((), "", "", "missing", "No Excel workbook found")
    try:
        book = load_workbook(workbook, read_only=True, data_only=True)
        parsed = []
        try:
            for sheet in book.worksheets:
                if not _sheet_matches(sheet.title, _line_index(image_path)):
                    continue
                boxes = _parse_rows([tuple(row) for row in sheet.iter_rows(values_only=True)], image_path)
                if boxes:
                    parsed.append((sheet.title, boxes))
        finally:
            book.close()
    except Exception as exc:
        return BoxAnnotations((), str(workbook), "", "parse_error", f"{type(exc).__name__}: {exc}")
    if not parsed:
        return BoxAnnotations((), str(workbook), "", "no_boxes", "No compatible box table found")
    sheet, boxes = max(parsed, key=lambda item: len(item[1]))
    status = "ok" if all(normalize_text(box.text) for box in boxes) else "missing_text"
    error = "" if status == "ok" else "Every box needs a subword text value"
    return BoxAnnotations(tuple(boxes), str(workbook), sheet, status, error)


def lcs_pairs(left, right):
    n, m = len(left), len(right)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i, j] = 1 + dp[i + 1, j + 1] if left[i] and left[i] == right[j] else max(dp[i + 1, j], dp[i, j + 1])
    result = []
    i = j = 0
    while i < n and j < m:
        if left[i] and left[i] == right[j] and dp[i, j] == 1 + dp[i + 1, j + 1]:
            result.append((i, j)); i += 1; j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return result


def _ratio(a, b, empty=0.0):
    return float(a / b) if b else float(empty)


def binary_metrics(tp, fp, fn, tn):
    precision = _ratio(tp, tp + fp, 1.0 if tp + fn == 0 else 0.0)
    recall = _ratio(tp, tp + fn, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "specificity": _ratio(tn, tn + fp, 1.0),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }


def line_metrics(prefix, annotations, gt_indices, interval):
    boxes = annotations.boxes
    rule = os.environ.get("REAL_BOX_IN_MASK_RULE", "center").strip().lower()
    minimum = max(0.0, min(1.0, float(os.environ.get("REAL_BOX_MIN_COVERAGE", "0.50"))))
    predicted = set()
    coverages = []
    for index, box in enumerate(boxes):
        overlap = 0.0 if interval is None else max(0.0, min(box.x1, interval[1]) - max(box.x0, interval[0]))
        coverage = _ratio(overlap, box.width)
        coverages.append(coverage)
        center_inside = bool(interval and interval[0] <= box.center_x <= interval[1])
        inside = center_inside if rule == "center" else coverage >= minimum
        if rule in {"center_or_coverage", "either"}:
            inside = center_inside or coverage >= minimum
        if inside:
            predicted.add(index)
    all_indices = set(range(len(boxes)))
    tp = len(predicted & gt_indices); fp = len(predicted - gt_indices)
    fn = len(gt_indices - predicted); tn = len(all_indices - predicted - gt_indices)
    metrics = binary_metrics(tp, fp, fn, tn)
    gt_interval = None if not gt_indices else (min(boxes[i].x0 for i in gt_indices), max(boxes[i].x1 for i in gt_indices))
    interval_iou = None
    if interval is not None and gt_interval is not None:
        intersection = max(0.0, min(interval[1], gt_interval[1]) - max(interval[0], gt_interval[0]))
        union = max(interval[1], gt_interval[1]) - min(interval[0], gt_interval[0])
        interval_iou = _ratio(intersection, union)
    return {
        f"{prefix}_box_annotation_path": annotations.workbook,
        f"{prefix}_box_annotation_sheet": annotations.sheet,
        f"{prefix}_box_annotation_status": annotations.status,
        f"{prefix}_box_annotation_error": annotations.error,
        f"{prefix}_box_count": len(boxes),
        f"{prefix}_shared_gt_boxes": len(gt_indices),
        f"{prefix}_predicted_mask_boxes": len(predicted),
        f"{prefix}_box_tp": tp, f"{prefix}_box_fp": fp, f"{prefix}_box_fn": fn, f"{prefix}_box_tn": tn,
        f"{prefix}_box_precision": metrics["precision"], f"{prefix}_box_recall": metrics["recall"], f"{prefix}_box_f1": metrics["f1"],
        f"{prefix}_box_specificity": metrics["specificity"], f"{prefix}_box_accuracy": metrics["accuracy"],
        f"{prefix}_shared_box_mask_coverage": float(np.mean([coverages[i] for i in gt_indices])) if gt_indices else None,
        f"{prefix}_box_interval_iou": interval_iou,
        f"{prefix}_pred_start_px": None if interval is None else interval[0],
        f"{prefix}_pred_end_px": None if interval is None else interval[1],
        f"{prefix}_gt_start_px": None if gt_interval is None else gt_interval[0],
        f"{prefix}_gt_end_px": None if gt_interval is None else gt_interval[1],
    }


def aggregate(rows):
    valid = [row for row in rows if row.get("real_box_evaluated")]
    tp = sum(int(row.get("pair_box_tp", 0)) for row in valid)
    fp = sum(int(row.get("pair_box_fp", 0)) for row in valid)
    fn = sum(int(row.get("pair_box_fn", 0)) for row in valid)
    tn = sum(int(row.get("pair_box_tn", 0)) for row in valid)
    micro = binary_metrics(tp, fp, fn, tn)
    def mean(key):
        values = [float(row[key]) for row in valid if row.get(key) not in (None, "")]
        return float(np.mean(values)) if values else None
    return {
        "samples": len(rows),
        "real_box_samples": len(valid),
        "real_box_missing_or_unusable_samples": len(rows) - len(valid),
        "real_box_status_counts": dict(Counter(str(row.get("real_box_status", "not_evaluated")) for row in rows)),
        "box_micro_tp": tp, "box_micro_fp": fp, "box_micro_fn": fn, "box_micro_tn": tn,
        "box_micro_precision": micro["precision"], "box_micro_recall": micro["recall"], "box_micro_f1": micro["f1"],
        "box_micro_specificity": micro["specificity"], "box_micro_accuracy": micro["accuracy"],
        "box_macro_precision": mean("pair_box_precision"), "box_macro_recall": mean("pair_box_recall"), "box_macro_f1": mean("pair_box_f1"),
        "mean_box_interval_iou": mean("mean_box_interval_iou"),
        "mean_shared_subword_matches": mean("shared_subword_matches"),
    }
