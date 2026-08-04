"""Discover and parse bbox.json annotations for quantitative evaluation.

The loader accepts one global JSON file or per-page/per-side bbox.json files.
It supports keyed-image, keyed-line, raw-array, nested-list, parallel
boxes/labels, and COCO-like layouts. Parsed boxes use the existing
BoxAnnotations/SubwordBox interface so quantitative scoring is unchanged.
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
    "transcription", "content", "arabic", "string", "value",
)
_TEXT_LIST_KEYS = ("texts", "labels", "subword_labels", "words_text", "tokens_text")
_LINE_KEYS = ("line", "line_idx", "line_index", "line_number", "line_no", "row")
_BOX_CONTAINER_KEYS = {
    "boxes", "bboxes", "bounding_boxes", "annotations", "subwords", "words",
    "tokens", "regions", "components", "items", "objects", "lines",
    "line_boxes", "line_bboxes",
}
_X0_KEYS = ("x0", "xmin", "x_min", "left", "start_x", "x_start", "min_x")
_Y0_KEYS = ("y0", "ymin", "y_min", "top", "start_y", "y_start", "min_y")
_X1_KEYS = ("x1", "x2", "xmax", "x_max", "right", "end_x", "x_end", "max_x")
_Y1_KEYS = ("y1", "y2", "ymax", "y_max", "bottom", "end_y", "y_end", "max_y")
_X_KEYS = ("x", "bbox_x", "box_x")
_Y_KEYS = ("y", "bbox_y", "box_y")
_W_KEYS = ("w", "width", "bbox_width", "box_width")
_H_KEYS = ("h", "height", "bbox_height", "box_height")
_LINE_NUMBER = re.compile(r"(?:line[_\-\s]*)?0*(\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_GENERIC_KEYS = _BOX_CONTAINER_KEYS | _IMAGE_KEYS | {
    "bbox", "bounding_box", "boundingbox", "box", "coordinates", "coords",
    "rectangle", "metadata", "meta", "data", "result", "results", "pages",
    "page", "width", "height", "image_width", "image_height",
}


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


def _same_line_key(value: Any, image_path: Path) -> bool:
    if _same_image(value, image_path):
        return True
    requested = _line_index(image_path)
    if requested is None:
        return False
    rendered = _normalise_key(value)
    if rendered.isdigit():
        return int(rendered) == requested
    match = _LINE_NUMBER.fullmatch(rendered)
    return match is not None and int(match.group(1)) == requested


def _looks_like_line_key(value: Any) -> bool:
    rendered = _normalise_key(value)
    return rendered.isdigit() or _LINE_NUMBER.fullmatch(rendered) is not None


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


def _numeric_sequence(value: Any, minimum: int = 4) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        return None
    numbers = [_as_float(item) for item in value[:minimum]]
    if any(item is None for item in numbers):
        return None
    return [float(item) for item in numbers]


def _point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        values = _mapping(value)
        x = _as_float(_first(values, ("x", "left", "x0", "x1")))
        y = _as_float(_first(values, ("y", "top", "y0", "y1")))
        return None if x is None or y is None else (x, y)
    numbers = _numeric_sequence(value, minimum=2)
    return None if numbers is None else (numbers[0], numbers[1])


def _bbox_from_value(value: Any, *, force_xywh: bool = False):
    if isinstance(value, dict):
        return _bbox_from_record(value)
    numbers = _numeric_sequence(value)
    if numbers is None:
        numbers = [_as_float(item) for item in _NUMBER.findall(str(value or ""))[:4]]
        if len(numbers) < 4 or any(item is None for item in numbers):
            return None
        numbers = [float(item) for item in numbers]
    x0, y0, third, fourth = numbers[:4]
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

    for first_key, second_key in (
        ("top_left", "bottom_right"),
        ("upper_left", "lower_right"),
        ("start", "end"),
        ("min", "max"),
    ):
        if first_key in values and second_key in values:
            first = _point(values[first_key])
            second = _point(values[second_key])
            if first is not None and second is not None:
                return first[0], first[1], second[0], second[1]

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
    value = _first(_mapping(record), _TEXT_KEYS)
    return str(value or "").strip()


def _make_box(parsed, text: str, source_row: int) -> SubwordBox | None:
    if parsed is None:
        return None
    x0, y0, x1, y1 = parsed
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    if right <= left:
        return None
    return SubwordBox(str(text or "").strip(), left, top, right, bottom, source_row)


def _box_from_record(record: dict, source_row: int, inherited_text: str = "") -> SubwordBox | None:
    return _make_box(
        _bbox_from_record(record),
        _text_from_record(record) or inherited_text,
        source_row,
    )


def _box_from_sequence(value: list | tuple, source_row: int, inherited_text: str = "") -> SubwordBox | None:
    numbers = _numeric_sequence(value)
    if numbers is not None:
        return _make_box(_bbox_from_value(numbers), inherited_text, source_row)
    if len(value) >= 5 and isinstance(value[0], str):
        return _make_box(_bbox_from_value(value[1:5]), value[0], source_row)
    if len(value) >= 5 and isinstance(value[4], str):
        return _make_box(_bbox_from_value(value[:4]), value[4], source_row)
    return None


def _walk_matching_nodes(value: Any, image_path: Path):
    """Yield subtrees explicitly associated with the requested image/line."""
    if isinstance(value, dict):
        for key, child in value.items():
            if _same_line_key(key, image_path):
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
        for key, child in value.items():
            if _looks_like_line_key(key) and not _same_line_key(key, image_path):
                continue
            yield from _walk_matching_nodes(child, image_path)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_matching_nodes(child, image_path)


def _parallel_boxes(record: dict, output: list[SubwordBox]) -> bool:
    values = _mapping(record)
    raw_boxes = _first(values, ("boxes", "bboxes", "bounding_boxes"))
    raw_texts = _first(values, _TEXT_LIST_KEYS)
    if not isinstance(raw_boxes, list) or not isinstance(raw_texts, list):
        return False
    if len(raw_boxes) != len(raw_texts):
        return False
    added = False
    for raw_box, raw_text in zip(raw_boxes, raw_texts):
        box = _make_box(_bbox_from_value(raw_box), str(raw_text or ""), len(output) + 1)
        if box is not None:
            output.append(box)
            added = True
    return added


def _collect_boxes(
    value: Any,
    image_path: Path,
    output: list[SubwordBox],
    inherited_text: str = "",
) -> None:
    if isinstance(value, dict):
        match = _record_image_match(value, image_path)
        if match is False:
            return
        if _parallel_boxes(value, output):
            return
        parsed = _box_from_record(value, len(output) + 1, inherited_text)
        if parsed is not None:
            output.append(parsed)
            return

        for key, child in value.items():
            normalised = _normalise_key(key)
            if _looks_like_line_key(key) and not _same_line_key(key, image_path):
                continue
            child_text = inherited_text
            if normalised not in _GENERIC_KEYS and not _same_line_key(key, image_path):
                if _bbox_from_value(child) is not None:
                    child_text = str(key)
            _collect_boxes(child, image_path, output, child_text)
        return

    if isinstance(value, (list, tuple)):
        parsed = _box_from_sequence(value, len(output) + 1, inherited_text)
        if parsed is not None:
            output.append(parsed)
            return
        for child in value:
            _collect_boxes(child, image_path, output, inherited_text)


def _coco_boxes(payload: dict, image_path: Path) -> list[SubwordBox]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return []

    image_ids = set()
    for item in images:
        if not isinstance(item, dict):
            continue
        image_value = _first(_mapping(item), _IMAGE_KEYS)
        if image_value is not None and _same_image(image_value, image_path):
            image_ids.add(item.get("id"))
    if not image_ids:
        return []

    categories = {
        item.get("id"): str(item.get("name", ""))
        for item in payload.get("categories", [])
        if isinstance(item, dict)
    }
    result = []
    for row, item in enumerate(annotations, start=1):
        if not isinstance(item, dict) or item.get("image_id") not in image_ids:
            continue
        parsed = _bbox_from_value(item.get("bbox"), force_xywh=True)
        if parsed is None:
            parsed = _bbox_from_record(item)
        text = _text_from_record(item) or categories.get(item.get("category_id"), "")
        box = _make_box(parsed, text, row)
        if box is not None:
            result.append(box)
    return result


@lru_cache(maxsize=128)
def _load_json(path_value: str):
    return json.loads(Path(path_value).read_text(encoding="utf-8-sig"))


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
    local = []
    for path in unique:
        try:
            path.relative_to(side_root)
            local.append(path)
        except ValueError:
            pass
    # The real dataset has one bbox.json per A/B side. Avoid accidentally
    # matching another page's line_01 when a side-local annotation exists.
    if local:
        return sorted(local, key=lambda path: (len(path.parts), str(path)))

    def rank(path: Path):
        score = 0
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

    boxes: list[SubwordBox] = []
    for node in _walk_matching_nodes(payload, image_path):
        _collect_boxes(node, image_path, boxes)
    if boxes:
        return boxes

    near_image = path.is_relative_to(_side_root(image_path))
    if near_image:
        _collect_boxes(payload, image_path, boxes)
    return boxes


def _schema_summary(path: Path) -> str:
    try:
        payload = _load_json(str(path))
    except Exception as exc:
        return f"{path}: unreadable ({type(exc).__name__}: {exc})"
    if isinstance(payload, dict):
        keys = list(payload)[:15]
        return f"{path}: object keys={keys}"
    if isinstance(payload, list):
        first_type = type(payload[0]).__name__ if payload else "empty"
        return f"{path}: list len={len(payload)} first_type={first_type}"
    return f"{path}: top-level type={type(payload).__name__}"


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

    summaries = "; ".join(_schema_summary(path) for path in candidates[:3])
    detail = (
        f"Found {len(candidates)} relevant bbox JSON file(s), but none contained "
        f"boxes for {image_path.name}. Schema: {summaries}"
    )
    if errors:
        detail += ". Parse errors: " + "; ".join(errors[:3])
    return BoxAnnotations((), str(candidates[0]), "json", "no_boxes", detail)
