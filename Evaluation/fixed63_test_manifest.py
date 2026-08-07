#!/usr/bin/env python3
"""Create the exact held-out fixed-63 synthetic test manifest used by training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=27000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-start", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=20, help="0 means all remaining test pairs")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--indices", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_dir).expanduser().resolve()
    total = int(args.num_samples)
    start = int(args.test_start) - 1
    if total <= 0 or start < 0 or args.n_samples < 0:
        raise SystemExit("num-samples/test-start/n-samples are invalid")

    train_size = int(0.6 * total)
    valid_size = int(0.2 * total)
    permutation = torch.randperm(
        total,
        generator=torch.Generator().manual_seed(int(args.split_seed)),
    ).tolist()
    test_zero_based = permutation[train_size + valid_size :]
    selected = test_zero_based[start:] if args.n_samples == 0 else test_zero_based[start : start + args.n_samples]
    if args.n_samples and len(selected) != args.n_samples:
        raise SystemExit(
            f"Requested {args.n_samples} test pairs from position {args.test_start}, "
            f"but only {len(selected)} are available."
        )
    if not selected:
        raise SystemExit("No held-out test pairs selected")

    records = []
    dataset_indices = []
    for output_index, zero_based in enumerate(selected, start=1):
        dataset_index = int(zero_based) + 1
        paths = {
            "image1": root / "images" / f"img1_{dataset_index}.png",
            "image2": root / "images" / f"img2_{dataset_index}.png",
            "text1": root / "texts" / f"text1_{dataset_index}.txt",
            "text2": root / "texts" / f"text2_{dataset_index}.txt",
            "mask1": root / "masks" / f"mask1_{dataset_index}.png",
            "mask2": root / "masks" / f"mask2_{dataset_index}.png",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise SystemExit("Missing synthetic sample files: " + ", ".join(missing))
        lengths = (
            len(paths["text1"].read_text(encoding="utf-8").strip()),
            len(paths["text2"].read_text(encoding="utf-8").strip()),
        )
        if lengths != (63, 63):
            raise SystemExit(
                f"Dataset index {dataset_index} has transcript lengths {lengths}, expected (63, 63)."
            )
        records.append(
            {
                "index": output_index,
                "pair_id": f"synthetic_dataset_index_{dataset_index}",
                "label_type": "synthetic_test",
                "dataset_index": dataset_index,
                "image1": str(paths["image1"]),
                "image2": str(paths["image2"]),
            }
        )
        dataset_indices.append(dataset_index)

    manifest = Path(args.manifest)
    indices = Path(args.indices)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    indices.write_text(
        json.dumps(
            {
                "split": "test",
                "split_seed": int(args.split_seed),
                "total_samples": total,
                "test_start": int(args.test_start),
                "selected_count": len(records),
                "selected_dataset_indices": dataset_indices,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(len(records))


if __name__ == "__main__":
    main()
