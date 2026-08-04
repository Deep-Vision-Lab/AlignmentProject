#!/usr/bin/env python3
"""Run the legacy quantitative evaluator with bbox.json annotations."""
from __future__ import annotations

import os

from Evaluation import eval_real_subword_boxes as legacy
from Evaluation.real_subword_box_json import load_json_annotations


def _load_annotations(image_path, annotation_root=""):
    source = str(annotation_root or "").strip()
    if source:
        os.environ["REAL_BOX_JSON"] = source
    return load_json_annotations(image_path)


# Preserve the legacy model, SW path and metric aggregation while replacing
# only the obsolete Excel annotation source.
legacy.load_annotations = _load_annotations


if __name__ == "__main__":
    legacy.main()
