"""Discover and parse bbox.json annotations for quantitative evaluation.

The loader accepts either one global JSON file or per-page/per-side bbox.json
files.  It supports common keyed-image, nested-list, and COCO-like layouts.
The parsed result uses the existing BoxAnnotations/SubwordBox interface so the
TP/FP/FN, precision/recall/F1 and IoU scoring code remains unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

from Evaluation.real_subword_box_metrics import BoxAnnotations, SubwordBox


_JSON_NAMES = ("bbox.json", "bboxes.json", "bounding_boxes.json")
_IMAGE_KEYS = {
    "image", "image_name", "imagename", "file", "file_name", "filename",
    "image_path", "path", "line_image", "line_image_path", "name",
}
_TEXT_KEYS = (
    "subword", "connected_subword", "text", "word", "token", "label",
    "transcription", "content", "arabic", "string",
)
_LINE_KEYS = ("line", "line_idx", "line_index", "line_number", "line_no", "row")
_BOX_CONTAINER_KEYS = {
    "boxes", "bboxes", "bounding_boxes", "annotations", "subwords", "words",
    "tokens", "regions", "components", "items", "objects",
}
_X0_KEYS = ("x0", "xmin", "x_min", "left", "start_x", "x_start", "min_x")
_Y0_KEYS = ("y0", "ymin", "y_min", "top", "start_y", "y_start", "min_y")
_X1_KEYS = ("x1", "xmax", "x_max", "right", "end_x", "x_end", "max_x")
_Y1_KEYS = ("y1", "ymax", "y_max", "bottom", "end_y", "y_end", "max_y")
_X_KEYS = ("x", "bbox_x", "box_x")
_Y_KEYS = ("y", "bbox_y", "box_y")
_W_KEYS = ("w", "width", "bbox_width", "box_width")
_H_KEYS = ("h", "height", "bbox_height", "box_height")
_LINE_NUMBER = re.compile(r"(?:line[_\-\s]*)?0*(\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^\w]+", "_", str(value or "").strip().lower()).strip("_")


def _mapping(record: dict) -> dict[str, Any]:
    return {_normalise_key(key): value for key, value in record.items()}


def _first(mapping: dict[str, Any], keys: Iterable[str]):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _line_index(image_path: Path) -> int | None:
    match = _LINE_NUMBER.search(image_path.stem)
    return int(match.group(1)) if match else None


def _same_image(value: Any, image_path: Path) -> bool:
    rendered = str(value or "").strip().replace("\\", "/")
    if not rendered:
        return False
    candidate = Path(rendered)
    expected_name = image_path.name.lower()
    expected_stem = image_path.stem.lower()
    return (
        candidate.name.lower() == expected_name
        or candidate.stem.lower() == expected_stem
        or rendered.lower().rstrip("/").endswith("/" + expected_name)
    )


def _record_image_match(record: dict, image_path: Path) -> bool | None:
    normalised = _mapping(record)
    values = [normalised[key] for key in _IMAGE_KEYS if key in normalised]
    if values:
        return any(_same_image(value, image_path) for value in values)

    requested_line = _line_index(image_path)
    if requested_line is not None:
        raw_line = _first(normalised, _LINE_KEYS)
        parsed_line = _as_float(raw_line)
        if parsed_line is not None:
            return int(parsed_line) == requested_line
    return None


def _bbox_from_value(value: Any, *, force_xywh: bool = False):
    if isinstance(value, dict):
        return _bbox_from_record(value)
    if isinstance(value, (list, tuple)):
        numbers = [_as_float(item) for item in value[:4]]
    else:
        numbers = [_as_float(item) for item in _NUMBER.findall(str(value or ""))[:4]]
    if len(numbers) < 4 or any(item is None for item in numbers):
        return None
    x0, y0, third, fourth = (float(item) for item in numbers)
    bbox_format = os.environ.get("REAL_BOX_BBOX_FORMAT", "auto").strip().lower()
    use_xywh = force_xywh or bbox_format == "xywh"
    if bbox_format == "auto" and not force_xywh:
        use_xywh = False
    return (x0, y0, x0 + third, y0 + fourth) if use_xywh else (x0, y0, third, fourth)


def _bbox_from_record(record: dict):
    values = _mapping(record)
    for key in ("bbox", "bounding_box", "boundingbox", "box", "coordinates", "coords", "rectangle"):
        if key in values:
            parsed = _bbox_from_value(values[key])
            if parsed is not None:
                return parsed

    x0 = _as_float(_first(values, _X0_KEYS))
    x1 = _as_float(_first(values, _X1_KEYS))
    y0 = _as_float(_first(values, _Y0_KEYS))
    y1 = _as_float(_first(values, _Y1_KEYS))
    if x0 is not None and x1 is not None:
        return x0, 0.0 if y0 is None else y0, x1, 1.0 if y1 is None else y1

    x = _as_float(_first(values, _X_KEYS))
    y = _as_float(_first(values, _Y_KEYS))
    width = _as_float(_first(values, _W_KEYS))
    height = _as_float(_first(values, _H_KEYS))
    if x is not None and width is not None:
        y = 0.0 if y is None else y
        height = 1.0 if height is None else height
        return x, y, x + width, y + height
    return None


def _text_from_record(record: dict) -> str:
    values = _mapping(record)
    value = _first(values, _TEXT_KEYS)
    return str(value or "").strip()


def _box_from_record(record: dict, source_row: int) -> SubwordBox | None:
    parsed = _bbox_from_record(record)
    if parsed is None:
        return None
    x0, y0, x1, y1 = parsed
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    if right <= left:
        return None
    return SubwordBox(_text_from_record(record), left, top, right, bottom, source_row)


def _walk_matching_nodes(value: Any, image_path: Path):
    """Yield subtrees explicitly associated with the requested image."""
    if isinstance(value, dict):
        for key, child in value.items():
            if _same_image(key, image_path):
                yield child

        match = _record_image_match(value, image_path)
        if match is True:
            yielded_container = False
            for key, child in value.items():
                if _normalise_key(key) in _BOX_CONTAINER_KEYS:
                    yielded_container = True
                    yield child
            if not yielded_container:
                yield value
            return
        if match is False:
            return
        for child in value.values():
            yield from _walk_matching_nodes(child, image_path)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_matching_nodes(child, image_path)


def _collect_boxes(value: Any, image_path: Path, output: list[SubwordBox]) -> None:
    if isinstance(value, dict):
        match = _record_image_match(value, image_path)
        if match is False:
            return
        parsed = _box_from_record(value, len(output) + 1)
        if parsed is not None:
            output.append(parsed)
            return
        for child in value.values():
            _collect_boxes(child, image_path, output)
    elif isinstance(value, list):
        for child in value:
            _collect_boxes(child, image_path, output)


def _coco_boxes(payload: dict, image_path: Path) -> list[SubwordBox]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return []

    image_ids = set()
    for item in images:
        if not isinstance(item, dict):
            continue
        values = _mapping(item)
        image_value = _first(values, _IMAGE_KEYS)
        if image_value is not None and _same_image(image_value, image_path):
            image_ids.add(item.get("id"))
    if not image_ids:
        return []

    result = []
    categories = {
        item.get("id"): str(item.get("name", ""))
        for item in payload.get("categories", [])
        if isinstance(item, dict)
    }
    for row, item in enumerate(annotations, start=1):
        if not isinstance(item, dict) or item.get("image_id") not in image_ids:
            continue
        parsed = _bbox_from_value(item.get("bbox"), force_xywh=True)
        if parsed is None:
            parsed = _bbox_from_record(item)
        if parsed is None:
            continue
        x0, y0, x1, y1 = parsed
        text = _text_from_record(item) or categories.get(item.get("category_id"), "")
        left, right = sorted((float(x0), float(x1)))
        top, bottom = sorted((float(y0), float(y1)))
        if right > left:
            result.append(SubwordBox(text, left, top, right, bottom, row))
    return result


@lru_cache(maxsize=128)
def _load_json(path_value: str):
    path = Path(path_value)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _side_root(image_path: Path) -> Path:
    for parent in image_path.parents:
        if parent.name in {"A", "B"}:
            return parent
    return image_path.parent


def _dataset_root(image_path: Path) -> Path:
    for parent in (image_path, *image_path.parents):
        if parent.name in {"ArabicDataset", "Synthetic_Arabic", "Synthetic_Arabic_100000"}:
            return parent
    return _side_root(image_path)


@lru_cache(maxsize=32)
def _json_files_under(root_value: str) -> tuple[Path, ...]:
    root = Path(root_value).expanduser()
    if root.is_file():
        return (root.resolve(),) if root.name.lower() in _JSON_NAMES else ()
    if not root.exists():
        return ()
    return tuple(
        sorted(
            path.resolve()
            for name in _JSON_NAMES
            for path in root.rglob(name)
            if path.is_file()
        )
    )


def candidate_json_files(image_path: Path) -> list[Path]:
    explicit = os.environ.get("REAL_BOX_JSON", "").strip()
    explicit_root = os.environ.get("REAL_BOX_ANNOTATIONS_ROOT", "").strip()
    roots = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    if explicit_root:
        roots.append(Path(explicit_root).expanduser())
    roots.extend((_side_root(image_path), _side_root(image_path).parent, _dataset_root(image_path)))

    candidates = []
    for root in roots:
        candidates.extend(_json_files_under(str(root)))
    unique = list(dict.fromkeys(path.resolve() for path in candidates))

    side_root = _side_root(image_path)
    def rank(path: Path):
        score = 0
        try:
            path.relative_to(side_root)
            score += 1000
        except ValueError:
            pass
        rendered = str(path).lower()
        if image_path.stem.lower() in rendered:
            score += 200
        for part in image_path.parts:
            if re.fullmatch(r"pair_\d+", part.lower()) and part.lower() in rendered:
                score += 300
        return (-score, len(path.parts), rendered)
    return sorted(unique, key=rank)


def _parse_file(path: Path, image_path: Path) -> list[SubwordBox]:
    payload = _load_json(str(path))
    if isinstance(payload, dict):
        coco = _coco_boxes(payload, image_path)
        if coco:
            return coco

    matched_nodes = list(_walk_matching_nodes(payload, image_path))
    boxes: list[SubwordBox] = []
    for node in matched_nodes:
        _collect_boxes(node, image_path, boxes)
    if boxes:
        return boxes

    # A bbox.json stored next to one line/page often has no explicit image key.
    near_image = path.parent in {
        image_path.parent,
        _side_root(image_path),
        _side_root(image_path) / "annotations",
        _side_root(image_path) / "labels",
    }
    if near_image:
        _collect_boxes(payload, image_path, boxes)
    return boxes


def load_json_annotations(image_path: str | Path) -> BoxAnnotations:
    image_path = Path(image_path)
    candidates = candidate_json_files(image_path)
    if not candidates:
        return BoxAnnotations((), "", "", "missing", "No bbox.json annotation file was found")

    errors = []
    for path in candidates:
        try:
            boxes = _parse_file(path, image_path)
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if boxes:
            return BoxAnnotations(tuple(boxes), str(path), "json", "ok", "")

    detail = f"Found {len(candidates)} bbox JSON file(s), but none contained boxes for {image_path.name}"
    if errors:
        detail += ". " + "; ".join(errors[:3])
    return BoxAnnotations((), str(candidates[0]), "json", "no_boxes", detail)
