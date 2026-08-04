#!/usr/bin/env python3
"""Run the branch trainer with dense Arabic 1-3 augmentation."""
from __future__ import annotations

import os
import sys

import DataLoader as original_data_loader
import SyntheticAugmentedDataLoader as augmented_data_loader
import DenseSyntheticAugmentation  # noqa: F401  # patches augmented_data_loader


sys.modules["DataLoader"] = augmented_data_loader
os.environ.setdefault("DATASET_TYPE", "synthetic")
os.environ.setdefault(
    "SYNTHETIC_DATASET_FOLDERS",
    "Synthetic_Arabic_1,Synthetic_Arabic_2,Synthetic_Arabic_3",
)
os.environ.setdefault("SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER", "3000")
os.environ.setdefault("SYNTHETIC_AUGMENT_COPIES_PER_SAMPLE", "2")

# 35% cross-line, 30% two-region, 25% aligned+unaligned, 10% full-line.
os.environ.setdefault("SYNTHETIC_INJECTION_PROB", "0.35")
os.environ.setdefault("SYNTHETIC_TWO_REGION_PROB", "0.30")
os.environ.setdefault("SYNTHETIC_ALIGNED_UNALIGNED_PROB", "0.25")

# Wider fragments and smaller gaps produce denser merged lines.
os.environ.setdefault("SYNTHETIC_FRAGMENT_MIN_FRACTION", "0.20")
os.environ.setdefault("SYNTHETIC_FRAGMENT_MAX_FRACTION", "0.40")
os.environ.setdefault("SYNTHETIC_SOURCE_GAP_FRACTION", "0.04")
os.environ.setdefault("SYNTHETIC_CANVAS_GAP_FRACTION", "0.08")
os.environ.setdefault("SYNTHETIC_MISMATCH_SPAN_DISTANCE", "0.12")

# Mild appearance changes only.
os.environ.setdefault("SYNTHETIC_SCALE_MIN", "0.90")
os.environ.setdefault("SYNTHETIC_SCALE_MAX", "1.00")
os.environ.setdefault("SYNTHETIC_TRANSLATE_PCT", "0.04")
os.environ.setdefault("SYNTHETIC_CONTRAST", "0.10")

import train


def main() -> None:
    try:
        train.main()
    finally:
        sys.modules["DataLoader"] = original_data_loader


if __name__ == "__main__":
    main()
