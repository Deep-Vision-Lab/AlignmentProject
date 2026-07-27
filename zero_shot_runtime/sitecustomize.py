"""Automatically install zero-shot runtime patches in every torchrun rank."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def _enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


if _enabled("ZERO_SHOT_PROFILE") and _enabled("ZERO_SHOT_SOURCE_GEOMETRY", True):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from zero_shot_geometry import install_source_compatible_geometry

    installed = install_source_compatible_geometry()
    if installed and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            "zero_shot_geometry enabled: fixed ink height + horizontal-only "
            "compression for long lines",
            flush=True,
        )
