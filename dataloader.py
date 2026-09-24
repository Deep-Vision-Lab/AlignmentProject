"""Seeded, group-safe splits and dedicated augmentation-free validation views."""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AlignmentDataset


def collate_samples(samples):
    return {key: torch.stack(values) if all(torch.is_tensor(v) for v in values) else values
            for key in samples[0] for values in [[s[key] for s in samples]]}


def _seed_worker(_worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def create_dataloaders(root, dataset_type='auto', train_ratio=.8, val_ratio=.1,
                       test_ratio=.1, split_mode='group', seed=42, batch_size=32,
                       num_workers=4, augment=True, *, split_ids=None, **dataset_options):
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r < 0 or not math.isfinite(r) for r in ratios) or not math.isclose(sum(ratios), 1., abs_tol=1e-9):
        raise ValueError('Split ratios must be nonnegative and sum to 1')
    if split_mode not in {'random', 'group', 'predefined'}:
        raise ValueError('split_mode must be random, group or predefined')
    dataset = AlignmentDataset(root, dataset_type, augment=False, **dataset_options)
    names = ('train', 'val', 'test')
    records = {name: [] for name in names}
    if split_ids is not None:
        by_id = {r['sample_id']: r for r in dataset.records}
        if len(by_id) != len(dataset):
            raise ValueError('Duplicate sample IDs')
        flat = [i for name in names for i in split_ids[name]]
        if len(flat) != len(set(flat)) or set(flat) != set(by_id):
            raise ValueError('Saved split IDs disagree with dataset population')
        records = {name: [by_id[i] for i in split_ids[name]] for name in names}
    elif split_mode == 'predefined':
        for r in dataset.records:
            if r['split'] not in names:
                raise ValueError(f'Missing/invalid predefined split: {r["sample_id"]}')
            records[r['split']].append(r)
    else:
        # Even random mode keeps repeated group IDs atomic. Random independent
        # line splitting must not separate the two sides of a synthetic sample.
        groups = sorted({r['group_id'] for r in dataset.records})
        random.Random(seed).shuffle(groups)
        first, second = int(len(groups) * train_ratio), int(len(groups) * (train_ratio + val_ratio))
        assigned = {g: name for name, gs in zip(names, (groups[:first], groups[first:second], groups[second:])) for g in gs}
        for r in dataset.records:
            records[assigned[r['group_id']]].append(r)
    group_sets = [{r['group_id'] for r in records[name]} for name in names]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Data contract failure: group leakage across saved/predefined splits')
    loaders = []
    for index, name in enumerate(names):
        view = copy.copy(dataset)
        view.records = records[name]
        view.augment = bool(augment and name == 'train')
        loader = DataLoader(view, batch_size=batch_size, shuffle=name == 'train' and len(view) > 0,
                            num_workers=num_workers, collate_fn=collate_samples,
                            worker_init_fn=_seed_worker,
                            generator=torch.Generator().manual_seed(seed + index))
        loaders.append(loader)
    return tuple(loaders)
