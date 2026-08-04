#!/usr/bin/env python3
"""Run the standard trainer with lightweight Arabic 1-3 augmentation."""
from __future__ import annotations

import os
import sys

import DataLoader as original_data_loader
import SyntheticAugmentedDataLoader as augmented_data_loader

# train.select_dataloaders imports DataLoader lazily. Replacing that module only
# in this dedicated entry point leaves train.py and real-data launchers unchanged.
sys.modules["DataLoader"] = augmented_data_loader
os.environ.setdefault("DATASET_TYPE", "synthetic")
os.environ.setdefault(
    "SYNTHETIC_DATASET_FOLDERS",
    "Synthetic_Arabic_1,Synthetic_Arabic_2,Synthetic_Arabic_3",
)
os.environ.setdefault("SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER", "3000")

# 9,000 unique raw samples x 2 virtual copies = 18,000 samples per epoch.
os.environ.setdefault("SYNTHETIC_AUGMENT_COPIES_PER_SAMPLE", "2")

# Alignment-focused mode mixture: 50% donor injection, 35% two-region,
# and 15% lightweight full-line scale/translation/contrast augmentation.
os.environ.setdefault("SYNTHETIC_INJECTION_PROB", "0.50")
os.environ.setdefault("SYNTHETIC_TWO_REGION_PROB", "0.35")
os.environ.setdefault("SYNTHETIC_SCALE_MIN", "0.90")
os.environ.setdefault("SYNTHETIC_SCALE_MAX", "1.00")
os.environ.setdefault("SYNTHETIC_TRANSLATE_PCT", "0.04")
os.environ.setdefault("SYNTHETIC_CONTRAST", "0.10")

# Disable less relevant or potentially alignment-damaging transforms by default.
os.environ.setdefault("SYNTHETIC_ROTATION_DEGREES", "0")
os.environ.setdefault("SYNTHETIC_SHEAR_DEGREES", "0")
os.environ.setdefault("SYNTHETIC_BRIGHTNESS", "0")
os.environ.setdefault("SYNTHETIC_SHARPNESS", "0")
os.environ.setdefault("SYNTHETIC_BLUR_PROB", "0")
os.environ.setdefault("SYNTHETIC_NOISE_PROB", "0")
os.environ.setdefault("SYNTHETIC_MORPHOLOGY_PROB", "0")
os.environ.setdefault("SYNTHETIC_ERASE_PROB", "0")

import train


def main() -> None:
    try:
        train.main()
    finally:
        sys.modules["DataLoader"] = original_data_loader


if __name__ == "__main__":
    main()
