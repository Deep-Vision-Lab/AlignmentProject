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


def _is_training_process() -> bool:
    return any(Path(argument).name == "train_optimized.py" for argument in sys.argv)


# PYTHONPATH is inherited by launcher helper commands such as `conda info`.
# Install heavy project patches only inside the actual training Python process.
if _is_training_process() and _enabled("ZERO_SHOT_PROFILE"):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if _enabled("ZERO_SHOT_SOURCE_GEOMETRY", True):
        from zero_shot_geometry import install_source_compatible_geometry

        geometry_installed = install_source_compatible_geometry()
        if geometry_installed and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "zero_shot_geometry enabled: fixed ink height + horizontal-only "
                "compression for long lines",
                flush=True,
            )

    if os.environ.get("VISUAL_ENCODER_TYPE", "cnn_bilstm").strip().lower() == "vit":
        from vit_training_runtime import install_vit_training_runtime

        vit_installed = install_vit_training_runtime()
        if vit_installed and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "visual_encoder enabled: full-height window patch projection + ViT; "
                "CNN and BiLSTM disabled",
                flush=True,
            )
