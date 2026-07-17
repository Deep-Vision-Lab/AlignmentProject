#!/usr/bin/env python3
"""Full-quality training with the opt-in augmented real-data loader."""

# Importing this module installs the full-quality compositional and differentiable
# order-loss patches without starting training.
import train_full_quality  # noqa: F401
import train as base
from AugmentedRealDataLoader import build_dataloaders

# train.py imported its original function with ``from DataLoader import ...``.
# Replace that bound reference before main() selects the datasets.
base.build_dataloaders = build_dataloaders


if __name__ == "__main__":
    base.main()
