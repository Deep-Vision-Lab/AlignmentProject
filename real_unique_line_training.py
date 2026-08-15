"""Leakage-safe single-line real adaptation for the Stage-1 visual encoder.

This mode deliberately ignores A/B partner supervision.  It deduplicates every
physical real line referenced by the full line-pair manifest, keeps validation
and test pages held out according to the canonical positive-pair 80/10/10 split,
and trains only the ordinary single image<->text Span-DTW/local-hard-negative
path already supported by train.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import DataLoader as base_loader
import extra_real_training as legacy
from real_span_feasibility import minimum_required_spans


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _resolve(root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path, root.parent / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    rendered = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {value!r}. Tried:\n  - {rendered}")


class UniqueRealLineDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform):
        self.root = root.resolve()
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[int(index)]
        image_path = Path(record["image_path"])
        text_path = Path(record["text_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.copy()
        with text_path.open("r", encoding="utf-8") as handle:
            text = " " + handle.read().strip() + " "
        return text, image


def _positive_eval_pages(data_dir):
    # Reuse exactly the same positive-pair split implementation as the joint-real
    # curriculum.  Only the page sets are needed here; partner losses stay off.
    from joint_real_training_v5 import _group_split

    dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    _train, valid, test = _group_split(dataset)
    valid_pages = legacy._sample_page_ids(dataset, (valid,))
    test_pages = legacy._sample_page_ids(dataset, (test,))
    # If a page participates in both pair-id groups, keep it in test only so it
    # never appears in training/validation.
    valid_pages -= test_pages
    return valid_pages, test_pages


def _all_unique_records(data_dir) -> list[dict]:
    manifest = base_loader._real_manifest_path(data_dir)
    root = manifest.parent.resolve()
    text_key = os.environ.get("REAL_TEXT_KEY", "text_original_path")
    records: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {manifest}:{lineno}: {exc}") from exc
            for side_name in ("A", "B"):
                side = row.get(side_name) or {}
                if not side.get("line_image_path") or not side.get(text_key):
                    continue
                image_path = _resolve(root, side["line_image_path"])
                text_path = _resolve(root, side[text_key])
                key = str(image_path)
                if key in records:
                    continue
                page_id = row.get(f"{side_name}_page_id") or side.get("page_id")
                if page_id is None:
                    page_id = f"{row.get('pair_id', 'unknown')}:{side_name}"
                records[key] = {
                    "image_path": str(image_path),
                    "text_path": str(text_path),
                    "page_id": str(page_id),
                }
    return list(records.values())


def _filter_feasible(records: list[dict], split_name: str) -> tuple[list[dict], int]:
    max_windows = _env_int("REAL_MAX_ALIGNMENT_WINDOWS", 125)
    max_span_chars = _env_int("MAX_TEXT_SPAN_CHARS", 2)
    kept = []
    removed = 0
    for record in records:
        with Path(record["text_path"]).open("r", encoding="utf-8") as handle:
            text = handle.read().strip()
        required = minimum_required_spans(text, max_span_chars=max_span_chars)
        if required <= max_windows:
            kept.append(record)
        else:
            removed += 1
    if not kept:
        raise RuntimeError(f"Unique-line feasibility removed every {split_name} sample.")
    return kept, removed


def _make_loader(dataset, shuffle: bool):
    kwargs = dict(
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        collate_fn=base_loader.custom_collate_fn,
        num_workers=base_loader._num_workers,
        pin_memory=base_loader._pin_memory,
        persistent_workers=base_loader._persistent_workers,
        drop_last=False,
    )
    if base_loader._prefetch_factor is not None:
        kwargs["prefetch_factor"] = base_loader._prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_unique_line_dataloaders(data_dir):
    root = Path(data_dir).expanduser().resolve()
    valid_pages, test_pages = _positive_eval_pages(data_dir)
    records = _all_unique_records(data_dir)

    train_records, valid_records, test_records = [], [], []
    for record in records:
        page_id = record["page_id"]
        if page_id in test_pages:
            test_records.append(record)
        elif page_id in valid_pages:
            valid_records.append(record)
        else:
            train_records.append(record)

    train_records, train_removed = _filter_feasible(train_records, "train")
    valid_records, valid_removed = _filter_feasible(valid_records, "valid")
    test_records, test_removed = _filter_feasible(test_records, "test")

    transform = base_loader.real_transform
    train = UniqueRealLineDataset(root, train_records, transform)
    valid = UniqueRealLineDataset(root, valid_records, transform)
    test = UniqueRealLineDataset(root, test_records, transform)

    print(
        "Unique real image-text adaptation dataset: "
        f"manifest={base_loader._real_manifest_path(data_dir)} "
        f"unique_total={len(records)} train={len(train)} valid={len(valid)} test={len(test)} "
        f"feasibility_removed={train_removed}/{valid_removed}/{test_removed} "
        "partner_supervision=False",
        flush=True,
    )
    return _make_loader(train, True), _make_loader(valid, False), _make_loader(test, False)


def install(base) -> None:
    original_select = base.select_dataloaders

    def select_dataloaders(args):
        if str(getattr(args, "dataset_type", "")).lower() != "real":
            return original_select(args)
        train_loader, valid_loader, test_loader = build_unique_line_dataloaders(args.data_dir)
        if not base.CTX.enabled:
            return train_loader, valid_loader, test_loader, None

        seed = base._env_int("DATASET_SPLIT_SEED", 42)
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=base.CTX.world_size,
            rank=base.CTX.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        valid_sampler = base.DistributedEvalSampler(valid_loader.dataset, base.CTX.rank, base.CTX.world_size)
        test_sampler = base.DistributedEvalSampler(test_loader.dataset, base.CTX.rank, base.CTX.world_size)
        return (
            base._rebuild_loader(train_loader, train_sampler),
            base._rebuild_loader(valid_loader, valid_sampler),
            base._rebuild_loader(test_loader, test_sampler),
            train_sampler,
        )

    base.select_dataloaders = select_dataloaders
    previous_model_config = base.model_config

    def model_config(stride, args):
        config = dict(previous_model_config(stride, args))
        config.update(
            {
                "real_adaptation_stage": "R0_unique_image_text",
                "unique_real_line_adaptation": True,
                "partner_supervision": False,
            }
        )
        return config

    base.model_config = model_config
    if getattr(base.CTX, "is_main", True):
        print(
            "Unique real image-text adaptation installed: partner/image-pair/sequence "
            "supervision is disabled; only single-line image-text objectives remain.",
            flush=True,
        )
