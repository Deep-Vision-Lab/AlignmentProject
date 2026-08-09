#!/usr/bin/env python3
"""Run bbox-strip injection with flat page-level bbox recovery enabled."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Evaluation.real_subword_box_json as standard_bbox  # noqa: E402
from Evaluation.real_flat_page_bbox import load_flat_page_line_annotations  # noqa: E402


_standard_loader = standard_bbox.load_json_annotations


def _load_with_flat_page_fallback(image_path):
    result = _standard_loader(image_path)
    if result.status == "ok" and result.boxes:
        return result

    fallback = load_flat_page_line_annotations(image_path)
    if fallback.status == "ok" and fallback.boxes:
        print(
            f"[real-bbox] recovered {len(fallback.boxes)} subword boxes for "
            f"{Path(image_path).name} via {fallback.sheet}",
            file=sys.stderr,
        )
        return fallback

    # Preserve both diagnostics when neither parser works.
    detail = str(result.error or "")
    fallback_detail = str(fallback.error or "")
    if fallback_detail:
        detail = f"{detail}; flat-page fallback: {fallback_detail}" if detail else fallback_detail
    return type(result)(result.boxes, result.workbook, result.sheet, result.status, detail)


standard_bbox.load_json_annotations = _load_with_flat_page_fallback

GENERATOR = ROOT / "scripts" / "data" / "augment_real_bbox_strip_injection.py"
sys.argv[0] = str(GENERATOR)
runpy.run_path(str(GENERATOR), run_name="__main__")
