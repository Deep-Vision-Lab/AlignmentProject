#!/usr/bin/env python3
"""Re-evaluate saved train/validation memberships; never select or access test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from Evaluation._eval_utils import load_evaluation_models
from epoch_monitoring import evaluate_split, assert_disjoint


class SavedLines(Dataset):
    def __init__(self, rows, contract):
        self.rows, self.contract = rows, contract

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        root = Path(row["_root"])
        image, _ = self.contract.prepare_line(root / row["line_image_path"])
        text = (root / row["text_path"]).read_text(encoding="utf-8")
        return self.contract.tensor_transform()(image), text, []


def collate(lines):
    images, texts, negatives = zip(*lines)
    return torch.stack(images), list(texts), list(negatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True, help="Monitoring split_manifest.json, not evaluated_ids.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    data = Path(args.manifest).read_bytes()
    if hashlib.sha256(data).hexdigest() != models.config.get("split_manifest_sha256"):
        raise ValueError("Saved split manifest identity does not match checkpoint")
    manifests = json.loads(data)
    if set(manifests) != {"train_eval", "val_eval"}:
        raise ValueError("Only saved train_eval/val_eval memberships are accepted")
    assert_disjoint(manifests)
    protocol = models.config["monitoring"]
    P = SimpleNamespace(**dict(models.config,
        positive_letter_dtw_gamma=protocol["gamma"],
        restoration_contrastive_weight=protocol["contrastive_weight"],
        num_negatives=models.config.get("negative_transcripts", 0)))
    if protocol["negative_active"]:
        raise ValueError("Standalone saved-line evaluation requires negative transcripts inactive; use the trainer's collated monitoring pass otherwise")
    results = {}
    for split, rows in manifests.items():
        loader = DataLoader(SavedLines(rows, models.contract), batch_size=protocol["batch_size"],
            shuffle=False, drop_last=False, num_workers=0, collate_fn=collate)
        results[split] = evaluate_split(models.image_model, models.text_model, loader, P,
            seed=protocol["seed"], device=models.device, spatial_records=rows, config=models.config)
    report = dict(checkpoint=str(Path(args.weights).resolve()), population="full-split",
        protocol=dict(protocol, population="full-split", max_records=0),
        checkpoint_monitoring_protocol=protocol,
        evaluation_contract=models.contract.metadata(args.weights), results=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
