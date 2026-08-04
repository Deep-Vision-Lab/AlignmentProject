#!/usr/bin/env python3
"""Run the standard trainer with the balanced 4-folder synthetic loader."""
from __future__ import annotations

import os
import sys

# Import the original module first so the augmented loader can reuse its
# collate function, worker settings, transforms, and negative generation.
import DataLoader as original_data_loader
import SyntheticAugmentedDataLoader as augmented_data_loader

# train.select_dataloaders imports DataLoader lazily. Replacing that module only
# for this entry point keeps train.py and all real-dataset launchers unchanged.
sys.modules["DataLoader"] = augmented_data_loader
os.environ.setdefault("DATASET_TYPE", "synthetic")
os.environ.setdefault("SYNTHETIC_AUGMENT", "1")
os.environ.setdefault("SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER", "3000")

import train


def main() -> None:
    try:
        train.main()
    finally:
        sys.modules["DataLoader"] = original_data_loader


if __name__ == "__main__":
    main()
