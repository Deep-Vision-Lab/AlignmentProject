"""Epoch-wise random subset sampling for real-data training.

When REAL_TRAIN_SAMPLES_PER_EPOCH is smaller than the available training pool,
keep the full dataset resident but draw a fresh deterministic subset every epoch.
The same global subset is reconstructed on every DDP rank and then sharded
without overlap between ranks.
"""
from __future__ import annotations

import math
import os

import torch
from torch.utils.data import Sampler


class EpochRandomSubsetSampler(Sampler[int]):
    """Sample a fresh global subset each epoch and shard it across DDP ranks."""

    def __init__(
        self,
        dataset_size: int,
        target_size: int,
        *,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_size = int(dataset_size)
        self.target_size = int(target_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.epoch = 0

        if self.dataset_size <= 0:
            raise ValueError("EpochRandomSubsetSampler requires a non-empty dataset.")
        if self.target_size <= 0:
            raise ValueError("target_size must be positive.")
        if self.target_size > self.dataset_size:
            raise ValueError(
                f"target_size={self.target_size} exceeds dataset_size={self.dataset_size}."
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank={self.rank} is invalid for world_size={self.world_size}."
            )

        self.num_samples = int(math.ceil(self.target_size / self.world_size))
        self.total_size = self.num_samples * self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        selected = torch.randperm(
            self.dataset_size, generator=generator
        ).tolist()[: self.target_size]

        # Equal-length DDP ranks are required so every rank reaches collective
        # operations together. Padding is only needed when target_size is not
        # divisible by world_size. For the canonical 6000/2 configuration this
        # branch is not used, so all 6000 examples are unique within the epoch.
        if self.total_size > self.target_size:
            needed = self.total_size - self.target_size
            selected.extend(selected[:needed])

        return iter(selected[self.rank : self.total_size : self.world_size])


def install_epoch_subset_sampling(train_module) -> None:
    """Patch select_dataloaders to honor a target smaller than the train pool."""
    if getattr(train_module, "_epoch_subset_sampling_installed", False):
        return

    original_select = train_module.select_dataloaders

    def select_dataloaders(args):
        train_loader, valid_loader, test_loader, train_sampler = original_select(args)
        if str(getattr(args, "dataset_type", "")).lower() != "real":
            return train_loader, valid_loader, test_loader, train_sampler

        try:
            target = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
        except ValueError:
            target = 0
        pool_size = len(train_loader.dataset)
        if target <= 0 or target >= pool_size:
            return train_loader, valid_loader, test_loader, train_sampler

        seed = int(os.environ.get("DATASET_SPLIT_SEED", "42"))
        sampler = EpochRandomSubsetSampler(
            pool_size,
            target,
            seed=seed,
            rank=train_module.CTX.rank,
            world_size=train_module.CTX.world_size,
        )
        train_loader = train_module._rebuild_loader(train_loader, sampler)

        if train_module.CTX.is_main:
            print(
                "Epoch-random real subset sampling enabled: "
                f"pool={pool_size} samples_per_epoch={target} "
                f"per_rank={len(sampler)} world_size={train_module.CTX.world_size} "
                f"seed={seed} refresh=each_epoch",
                flush=True,
            )
        return train_loader, valid_loader, test_loader, sampler

    train_module.select_dataloaders = select_dataloaders
    train_module._epoch_subset_sampling_installed = True
