"""Prefer canonical per-line/per-side ``bbox.json`` annotations.

The ArabicDataset also contains unrelated diagnostic files named
``debug/bboxes.json``.  Those files describe page/line extraction geometry, not
connected-subword ground-truth boxes.  The generic JSON loader historically
accepted both names and could therefore select a nearby debug file before the
canonical annotation supplied by the user.

By default this patch discovers only files named exactly ``bbox.json`` and
prefers files scoped to the current page-pair side.  Alternate names can be
re-enabled explicitly with ``REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES=1``.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

from Evaluation import real_subword_box_json as json_loader


_PAIR = re.compile(r"pair_\d+", re.IGNORECASE)
_INSTALLED = False


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _pair_token(path: Path) -> str:
    for part in path.parts:
        if _PAIR.fullmatch(part):
            return part.lower()
    return ""


def _same_pair_and_side(path: Path, image_path: Path) -> bool:
    requested_pair = _pair_token(image_path)
    if not requested_pair or requested_pair not in {part.lower() for part in path.parts}:
        return False
    side_root = json_loader._side_root(image_path)
    side = side_root.name.upper() if side_root.name.upper() in {"A", "B"} else ""
    if not side:
        return True
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.lower() == requested_pair:
            return any(candidate.upper() == side for candidate in parts[index + 1 :])
    return False


def _rank(path: Path, image_path: Path) -> tuple[int, int, int, str]:
    side_root = json_loader._side_root(image_path)
    local = 0 if _under(path, side_root) else 1
    scoped = 0 if _same_pair_and_side(path, image_path) else 1
    debug = 1 if any(part.lower() == "debug" for part in path.parts) else 0
    return local, scoped, debug, str(path)


def canonical_candidate_json_files(image_path: str | Path) -> list[Path]:
    image_path = Path(image_path)
    explicit_value = os.environ.get("REAL_BOX_JSON", "").strip()
    explicit_root_value = os.environ.get("REAL_BOX_ANNOTATIONS_ROOT", "").strip()

    # An explicitly supplied file is authoritative.  Reject an accidental
    # plural/debug filename unless the caller opts into alternate names.
    if explicit_value:
        explicit = Path(explicit_value).expanduser()
        if explicit.is_file():
            if explicit.name.lower() == "bbox.json" or _flag(
                "REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES", False
            ):
                return [explicit.resolve()]
            return []

    roots: list[Path] = []
    if explicit_value:
        roots.append(Path(explicit_value).expanduser())
    if explicit_root_value:
        roots.append(Path(explicit_root_value).expanduser())
    side_root = json_loader._side_root(image_path)
    roots.extend((side_root, side_root.parent, json_loader._dataset_root(image_path)))

    discovered: list[Path] = []
    for root in roots:
        discovered.extend(json_loader._json_files_under(str(root)))
    unique = list(dict.fromkeys(path.resolve() for path in discovered))

    canonical = [path for path in unique if path.name.lower() == "bbox.json"]
    canonical.sort(key=lambda path: _rank(path, image_path))

    local = [path for path in canonical if _under(path, side_root)]
    if local:
        return local

    scoped = [path for path in canonical if _same_pair_and_side(path, image_path)]
    if scoped:
        return scoped

    # A single global bbox.json is a valid supported layout.  Multiple global
    # files are also returned because the parser checks explicit image/line keys.
    if canonical:
        return canonical

    if _flag("REAL_BOX_ALLOW_ALTERNATE_JSON_NAMES", False):
        fallback = list(unique)
        fallback.sort(key=lambda path: _rank(path, image_path))
        return fallback
    return []


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    json_loader.candidate_json_files = canonical_candidate_json_files
    _INSTALLED = True
