#!/usr/bin/env python3
"""Stage-1 fixed63 synthetic training wrapper.

The shared branch-aware runtime owns the actual trainer/loss stack. This wrapper
only records an explicit Stage-1 marker and keeps the logical AraBERT model id in
checkpoint metadata when Transformers loads from a resolved offline snapshot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training_runtime import entrypoint as branch_runtime

optimized = branch_runtime.optimized
_original_model_config = optimized.base.model_config


def _stage1_model_config(stride, args):
    config = dict(_original_model_config(stride, args))
    logical_id = os.environ.get("ARABIC_TEXT_MODEL_ID", "").strip()
    if logical_id:
        config["arabic_text_model_name"] = logical_id
    config.update(
        {
            "stage1_synthetic_fixed63": True,
            "stage1_data_domain": "synthetic",
            "stage1_target_domain": "synthetic",
            "stage1_arabic_text_model_resolved_snapshot": os.environ.get(
                "ARABIC_TEXT_MODEL_RESOLVED_PATH", ""
            ),
        }
    )
    return config


optimized.base.model_config = _stage1_model_config


if __name__ == "__main__":
    optimized.main()
