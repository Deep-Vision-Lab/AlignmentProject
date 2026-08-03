"""Map real subword boxes through the exact evaluation crop/pad geometry."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps

from Evaluation.real_subword_box_metrics import (
    BoxAnnotations,
    SubwordBox,
    _fill_missing_text,
    _read_workbook,
    _reading_order,
)
from zero_shot_preprocessing import _ink_mask


_SPREADSHEET_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
_GENERIC_METADATA = {
    "a", "b", "image", "images", "line", "lines", "page", "pages",
    "dataset", "arabicdataset", "quran", "original", "final", "text",
}


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _foreground_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    gray_image = ImageOps.autocontrast(image.convert("L"))
    gray = np.asarray(gray_image, dtype=np.uint8)
    mask = _ink_mask(gray)
    ys, xs = np.nonzero(mask)
    if xs.size < 4 or ys.size < 4:
        return 0, 0, image.width, image.height
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int(round((x1 - x0) * 0.025)))
    pad_y = max(2, int(round((y1 - y0) * 0.15)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image.width, x1 + pad_x),
        min(image.height, y1 + pad_y),
    )


def _geometry(image: Image.Image, target_width: int, target_height: int):
    zero_shot = _flag("ZERO_SHOT_PREPROCESS", True)
    crop = zero_shot and _flag("ZERO_SHOT_FOREGROUND_CROP", True)
    preserve = zero_shot and _flag("ZERO_SHOT_PRESERVE_ASPECT", True)
    crop_box = _foreground_bounds(image) if crop else (0, 0, image.width, image.height)
    x0, y0, x1, y1 = crop_box
    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)
    if preserve:
        desired_height = max(
            8,
            int(round(target_height * _number("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72))),
        )
        scale = min(desired_height / crop_height, target_width / crop_width)
        new_width = max(1, min(target_width, int(round(crop_width * scale))))
        new_height = max(1, min(target_height, int(round(crop_height * scale))))
        offset_x = max(0, target_width - new_width) // 2
        offset_y = max(0, target_height - new_height) // 2
        scale_x = new_width / crop_width
        scale_y = new_height / crop_height
    else:
        offset_x = offset_y = 0
        scale_x = target_width / crop_width
        scale_y = target_height / crop_height
    return crop_box, scale_x, scale_y, float(offset_x), float(offset_y)


def _side_root(image_path: Path) -> Path:
    for parent in image_path.parents:
        if parent.name in {"A", "B"}:
            return parent
    return image_path.parent


def _dataset_root(image_path: Path) -> Path:
    explicit = os.environ.get("REAL_BOX_ANNOTATIONS_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    for parent in (image_path, *image_path.parents):
        if parent.name == "ArabicDataset":
            return parent
    return _side_root(image_path)


@lru_cache(maxsize=16)
def _all_spreadsheets(root_value: str) -> tuple[Path, ...]:
    root = Path(root_value)
    if not root.exists():
        return ()
    if root.is_file():
        return (root.resolve(),) if root.suffix.lower() in _SPREADSHEET_SUFFIXES else ()
    return tuple(
        sorted(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _SPREADSHEET_SUFFIXES
            and not path.name.startswith("~$")
        )
    )


def _metadata_files(image_path: Path) -> list[Path]:
    side_root = _side_root(image_path)
    pair_root = side_root.parent
    return [
        side_root / "page_meta.json",
        pair_root / "pair_meta.json",
    ]


def _metadata_values(value, key: str = ""):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _metadata_values(child_value, str(child_key))
    elif isinstance(value, list):
        for item in value:
            yield from _metadata_values(item, key)
    elif value not in (None, ""):
        lowered = key.lower()
        if any(token in lowered for token in ("writer", "page", "source", "image", "file", "path", "id")):
            yield str(value)


def _candidate_tokens(image_path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in image_path.parts:
        lowered = part.lower()
        if re.fullmatch(r"pair_\d+", lowered):
            tokens.add(lowered)
    tokens.add(image_path.stem.lower())

    for metadata_path in _metadata_files(image_path):
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for raw_value in _metadata_values(payload):
            value_path = Path(raw_value)
            candidates = [raw_value, value_path.name, value_path.stem]
            for candidate in candidates:
                for token in re.findall(r"[\w.-]+", str(candidate).lower()):
                    token = token.strip("._-")
                    if len(token) >= 3 and token not in _GENERIC_METADATA:
                        tokens.add(token)
    return tokens


def _candidate_score(workbook: Path, image_path: Path, tokens: set[str]) -> tuple[int, int, str]:
    rendered = str(workbook).lower()
    parts = {part.lower() for part in workbook.parts}
    score = 0
    side_root = _side_root(image_path)
    try:
        workbook.relative_to(side_root)
        score += 2000
    except ValueError:
        pass

    pair_ids = [part.lower() for part in image_path.parts if re.fullmatch(r"pair_\d+", part.lower())]
    for pair_id in pair_ids:
        if pair_id in rendered:
            score += 1000

    side_name = side_root.name.lower()
    if side_name in {"a", "b"} and side_name in parts:
        score += 100

    if image_path.stem.lower() in workbook.stem.lower():
        score += 250

    for token in tokens:
        if token in rendered:
            score += 40 if len(token) >= 5 else 15

    for keyword, weight in (
        ("subword", 100),
        ("bounding", 80),
        ("bbox", 80),
        ("annotation", 60),
        ("label", 30),
        ("box", 25),
    ):
        if keyword in rendered:
            score += weight
    return (-score, len(workbook.parts), rendered)


def candidate_workbooks(image_path: Path) -> list[Path]:
    root = _dataset_root(image_path)
    spreadsheets = _all_spreadsheets(str(root))
    tokens = _candidate_tokens(image_path)
    return sorted(
        spreadsheets,
        key=lambda workbook: _candidate_score(workbook, image_path, tokens),
    )


def _load_best_workbook(image_path: Path) -> BoxAnnotations:
    candidates = candidate_workbooks(image_path)
    if not candidates:
        root = _dataset_root(image_path)
        return BoxAnnotations(
            (),
            "",
            "",
            "missing",
            f"No Excel workbook found under annotation root: {root}",
        )

    parse_errors = []
    for workbook in candidates:
        annotations = _read_workbook(workbook, image_path)
        if annotations.boxes:
            return annotations
        if annotations.status == "parse_error":
            parse_errors.append(f"{workbook}: {annotations.error}")

    preview = "; ".join(str(path) for path in candidates[:5])
    error = (
        f"Found {len(candidates)} Excel workbook(s), but none contained a compatible "
        f"table for {image_path.name}. First candidates: {preview}"
    )
    if parse_errors:
        error += f". Parse errors: {'; '.join(parse_errors[:3])}"
    return BoxAnnotations((), str(candidates[0]), "", "no_boxes", error)


def load_line_annotations(image_path, target_width: int, target_height: int) -> BoxAnnotations:
    image_path = Path(image_path)
    annotations = _load_best_workbook(image_path)
    if not annotations.boxes:
        return annotations

    coordinate_space = os.environ.get("REAL_BOX_COORDINATE_SPACE", "original").strip().lower()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        source_width, source_height = image.size
        crop_box, scale_x, scale_y, offset_x, offset_y = _geometry(
            image, int(target_width), int(target_height)
        )
    crop_x0, crop_y0, _crop_x1, _crop_y1 = crop_box
    maximum = max(max(box.x1, box.y1) for box in annotations.boxes)
    if coordinate_space == "auto":
        coordinate_space = "normalized" if maximum <= 1.5 else "original"

    transformed = []
    for box in annotations.boxes:
        if coordinate_space == "normalized":
            source_x0, source_x1 = box.x0 * source_width, box.x1 * source_width
            source_y0, source_y1 = box.y0 * source_height, box.y1 * source_height
        elif coordinate_space == "model":
            mapped_x0, mapped_x1 = box.x0, box.x1
            mapped_y0, mapped_y1 = box.y0, box.y1
            source_x0 = source_x1 = source_y0 = source_y1 = None
        else:
            source_x0, source_x1 = box.x0, box.x1
            source_y0, source_y1 = box.y0, box.y1

        if coordinate_space != "model":
            mapped_x0 = (source_x0 - crop_x0) * scale_x + offset_x
            mapped_x1 = (source_x1 - crop_x0) * scale_x + offset_x
            mapped_y0 = (source_y0 - crop_y0) * scale_y + offset_y
            mapped_y1 = (source_y1 - crop_y0) * scale_y + offset_y

        x0 = max(0.0, min(float(target_width), mapped_x0))
        x1 = max(0.0, min(float(target_width), mapped_x1))
        y0 = max(0.0, min(float(target_height), mapped_y0))
        y1 = max(0.0, min(float(target_height), mapped_y1))
        if y1 <= y0:
            y0, y1 = 0.0, float(target_height)
        if x1 > x0:
            transformed.append(SubwordBox(box.text, x0, y0, x1, y1, box.source_row))

    transformed = _fill_missing_text(_reading_order(transformed), image_path)
    status, error = annotations.status, annotations.error
    if transformed and any(not box.text.strip() for box in transformed):
        status = "missing_text"
        error = "Excel text is missing and transcript subword count does not match box count"
    return BoxAnnotations(tuple(transformed), annotations.workbook, annotations.sheet, status, error)
