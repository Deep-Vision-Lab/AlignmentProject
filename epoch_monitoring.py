"""Deterministic end-of-epoch measurements, separate from optimization.

Rank zero evaluates unwrapped weights, without distributed loss collectives.
DTW is averaged over eligible lines. SIGReg is a fixed-sketch, fixed-batch
population estimator: token-count weighted batch statistics, NOT a line loss.
"""
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vlm_restoration_positive_dtw import (
    _clean_letters, positive_letter_dtw_loss, strong_sigreg_loss,
    letter_dtw_cost_matrix, negative_letter_dtw_margin_loss,
)


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def broadcast_monitor_result(payload):
    """All ranks participate, including on a rank-zero diagnostic failure."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError("Epoch monitoring failed on rank zero:\n" + payload[0]["error"])
    return payload[0]


@contextmanager
def deterministic_evaluation(model, text, seed):
    modes = [(m, m.training) for root in (model, text) for m in root.modules()]
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model.eval()
            text.eval()
            with torch.no_grad():
                yield
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(py)
            np.random.set_state(np_state)


def clean_view(dataset):
    """Use original membership, unwrapping online repeats/augmentation only."""
    name = type(dataset).__name__
    if name == "RepeatToLengthDataset":
        return clean_view(dataset.dataset)
    if name in {"AllPageLinesScanAugmentedSubset", "AugmentedRealSubset"}:
        return Subset(clean_view(dataset.base_dataset), list(dataset.indices))
    if isinstance(dataset, Subset):
        return Subset(clean_view(dataset.dataset), list(dataset.indices))
    return copy.copy(dataset)


def records(dataset):
    if isinstance(dataset, Subset):
        base = records(dataset.dataset)
        return [base[int(i)] for i in dataset.indices]
    if not hasattr(dataset, "samples"):
        raise ValueError("Monitoring requires recoverable manifest/sample identities; no regenerated split fallback")
    root = Path(getattr(dataset, "root", getattr(dataset, "dataset_root", ".")))
    result = []
    for sample in dataset.samples:
        row = dict(sample)
        row["_root"] = str(root)
        row["record_id"] = hashlib.sha256(json.dumps(sample, sort_keys=True, default=str).encode()).hexdigest()
        result.append(row)
    return result


def identity_keys(row):
    keys = {("record", row["record_id"]), ("line_identity", row["record_id"])}
    for name in ("pair_id", "source_line_id", "parent_id", "augmentation_parent_id", "original_parent_id"):
        if row.get(name):
            keys.add((name, str(row[name])))
            if name != "pair_id":
                keys.add(("line_identity", str(row[name])))
    for name in ("line_image_path", "image_path", "image1_path", "image2_path"):
        if row.get(name):
            keys.add(("image", str(Path(row["_root"], row[name]).resolve())))
    for name in ("A", "B"):
        if isinstance(row.get(name), dict):
            keys |= identity_keys(dict(row[name], _root=row["_root"], record_id=row["record_id"]))
    return keys


def assert_disjoint(manifests):
    identities = {split: set().union(*(identity_keys(r) for r in rows)) for split, rows in manifests.items()}
    names = list(identities)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = identities[first] & identities[second]
            if shared:
                raise ValueError(f"Data-contract leakage: {first}/{second}: {sorted(shared)[:5]}")


def batch_lines(batch):
    if isinstance(batch, dict):
        return [(batch[f"images{s}"], batch[f"texts{s}"], batch.get(f"neg_texts{s}")) for s in (1, 2)]
    images, texts, negatives = batch
    return [(images, texts, negatives)]


def sigreg_settings(P):
    return {key: getattr(P, "sigreg_" + key) for key in
            ("sketch_dim", "num_knots", "t_min", "t_max", "min_samples", "slice_chunk")}


def evaluate_split(model, text, loader, P, *, seed, device, preview=None, spatial_records=None, config=None):
    started = time.perf_counter()
    costs, windows, letters = [], [], []
    skipped_records = []
    xml_spatial = []
    evaluated = skipped = invalid = total_tokens = batches = 0
    reasons = {}
    sig_sum = sig_tokens = neg_sum = neg_count = 0
    record_offset = 0
    with deterministic_evaluation(model, text, seed):
        for batch in loader:
            sides = batch_lines(batch)
            for side, (images, transcripts, negatives) in enumerate(sides):
                # Same FP32 policy for both splits, independent of online AMP.
                bundle = model(images.to(device), return_training_bundle=True)
                valid = bundle["token_valid"]
                count = int(valid.sum())
                total_tokens += count
                batches += 1
                if float(P.sigreg_weight) > 0:
                    # Identical random directions for every eval batch/epoch/split.
                    with torch.random.fork_rng():
                        torch.manual_seed(seed)
                        sig, _ = strong_sigreg_loss(bundle["fused_pre_l2"], valid,
                            distributed_statistics=False, **sigreg_settings(P))
                    if not torch.isfinite(sig):
                        raise FloatingPointError("Nonfinite evaluation SIGReg")
                    sig_sum += float(sig) * count
                    sig_tokens += count
                for i, transcript in enumerate(transcripts):
                    n_letters = len(_clean_letters(transcript))
                    n_windows = int(valid[i].sum())
                    reason = "empty_cleaned_transcript" if n_letters == 0 else ("no_valid_windows" if n_windows == 0 else None)
                    if reason:
                        skipped += 1
                        reasons[reason] = reasons.get(reason, 0) + 1
                        skipped_records.append(dict(record_index=record_offset+i, side=side+1, reason=reason))
                        continue
                    loss, _ = positive_letter_dtw_loss(P, text, bundle["semantic"][i:i+1], valid[i:i+1], [transcript])
                    if not torch.isfinite(loss):
                        invalid += 1
                        reasons["nonfinite_dtw"] = reasons.get("nonfinite_dtw", 0) + 1
                        continue
                    evaluated += 1
                    costs.append(float(loss))
                    windows.append(n_windows)
                    letters.append(n_letters)
                    if spatial_records is not None:
                        from monitoring_spatial import score_xml_transcript
                        xml_spatial.append(score_xml_transcript(spatial_records[record_offset+i], transcript,
                            bundle, i, text, P, config))
                    if float(P.restoration_contrastive_weight) > 0 and int(P.num_negatives) > 0:
                        neg, _ = negative_letter_dtw_margin_loss(P, text, bundle["semantic"][i:i+1], valid[i:i+1], [transcript], [negatives[i]])
                        neg_sum += float(neg)
                        neg_count += 1
                    if preview is not None:
                        preview(record_offset+i, side, transcript, images[i], bundle, i)
            record_offset += len(sides[0][1])
    dtw = float(np.mean(costs)) if costs else None
    sigreg = sig_sum / sig_tokens if sig_tokens else None
    negative = neg_sum / neg_count if neg_count else None
    weighted = float(P.sigreg_weight) * sigreg if sigreg is not None else 0.0
    total = None if dtw is None else float(P.positive_letter_dtw_weight) * dtw + weighted + float(P.restoration_contrastive_weight) * (negative or 0.0)
    return dict(positive_dtw=dtw, sigreg_raw=sigreg, sigreg_weighted=weighted,
                negative=negative, total=total, evaluated=evaluated, skipped=skipped,
                invalid=invalid, reasons=reasons, skipped_records=skipped_records, valid_tokens=total_tokens,
                sigreg_populations=batches, letters_mean=float(np.mean(letters)) if letters else None,
                windows_mean=float(np.mean(windows)) if windows else None,
                xml_part_localization=xml_spatial, seconds=time.perf_counter()-started)


class EpochMonitor:
    def __init__(self, module, train_loader, valid_loader, config, job_id):
        self.module, self.config = module, config
        self.seed = int(os.environ.get("MONITOR_SEED", "31415"))
        self.batch_size = int(os.environ.get("MONITOR_BATCH_SIZE", "16"))
        self.limit = int(os.environ.get("MONITOR_MAX_RECORDS", "0"))
        self.preview_count = int(os.environ.get("MONITOR_PREVIEWS", "3"))
        if self.limit < 0 or self.preview_count < 0 or self.batch_size < 1:
            raise ValueError("Monitor subset/preview counts must be nonnegative and batch size positive")
        self.root = Path(os.environ.get("MONITOR_OUTPUT", f"Results/Monitoring/{job_id}"))
        self.P = SimpleNamespace(**{k: getattr(module.P, k) for k in dir(module.P) if not k.startswith("_")})
        self.P.positive_letter_dtw_gamma = float(os.environ.get("MONITOR_GAMMA", str(module.P.positive_letter_dtw_gamma_end)))
        self.loaders, self.rows = {}, {}
        manifests = {}
        for split, loader in (("train_eval", train_loader), ("val_eval", valid_loader)):
            view = clean_view(loader.dataset)
            rows = records(view)
            manifests[split] = rows
            if self.limit:
                indices = sorted(random.Random(self.seed).sample(range(len(view)), min(self.limit, len(view))))
                view = Subset(view, indices)
                rows = [rows[i] for i in indices]
            self.rows[split] = rows
            self.loaders[split] = DataLoader(view, batch_size=self.batch_size, shuffle=False,
                drop_last=False, num_workers=0, collate_fn=loader.collate_fn,
                generator=torch.Generator().manual_seed(self.seed))
        assert_disjoint(manifests)
        manifest_json = json.dumps(manifests, sort_keys=True, default=str)
        config["split_manifest_sha256"] = hashlib.sha256(manifest_json.encode()).hexdigest()
        config["monitoring"] = dict(gamma=self.P.positive_letter_dtw_gamma, precision="float32",
            seed=self.seed, batch_size=self.batch_size, max_records=self.limit,
            population="fixed-subset" if self.limit else "full-split",
            sigreg_estimator="token-weighted fixed-batch population statistic; identical seeded sketches; local collectives disabled",
            negative_active=float(self.P.restoration_contrastive_weight)>0 and int(self.P.num_negatives)>0,
            positive_weight=float(self.P.positive_letter_dtw_weight), sigreg_weight=float(self.P.sigreg_weight),
            contrastive_weight=float(self.P.restoration_contrastive_weight),
            source_provenance="online augmentation removed; pre-rendered manifest records retained, not claimed clean originals")
        config["monitoring"]["online_max_batches"] = int(os.environ.get("PROFILE_MAX_BATCHES", "0"))
        config["monitoring"]["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        config["monitoring"]["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        setup = [None]
        if module.CTX.is_main:
            try:
                self.root.mkdir(parents=True, exist_ok=False)
                (self.root / "split_manifest.json").write_text(manifest_json)
                (self.root / "evaluated_ids.json").write_text(json.dumps(self.rows, default=str, ensure_ascii=False, indent=2))
                (self.root / "preview_ids.json").write_text(json.dumps({s: rows[:self.preview_count] for s, rows in self.rows.items()}, default=str, ensure_ascii=False, indent=2))
                print("MONITORING_CONTRACT " + json.dumps(dict(config["monitoring"],
                    split_sizes={s: len(rows) for s, rows in manifests.items()},
                    monitored_sizes={s: len(rows) for s, rows in self.rows.items()},
                    split_manifest_sha256=config["split_manifest_sha256"],
                    transformer_positions=config.get("position_mode", "learned"),
                    dtw_position_prior=self.P.positive_letter_dtw_position_prior), sort_keys=True), flush=True)
                setup[0] = {"ready": True}
            except Exception as exc:
                setup[0] = {"error": str(exc)}
        broadcast_monitor_result(setup)
        self.history = []
        self.best = float("inf")

    def run(self, model, text, epoch, train_loss, train_stats, lr):
        payload = [None]
        if self.module.CTX.is_main:
            try:
                raw = self.module._unwrap_model(model)
                results = {}
                for split, loader in self.loaders.items():
                    from Evaluation.eval_point3_training_paths import monitoring_preview
                    def preview(index, side, transcript, image, bundle, i):
                        if index < self.preview_count:
                            monitoring_preview(self.root / f"epoch_{epoch:03d}" / split / f"record_{index:04d}_side{side+1}",
                                self.rows[split][index], transcript, image, bundle, i, raw, text, self.P, self.config)
                    results[split] = evaluate_split(raw, text, loader, self.P, seed=self.seed,
                        device=self.module.P.device, preview=preview, spatial_records=self.rows[split], config=self.config)
                from monitoring_spatial import evaluate_spatial_splits
                spatial = evaluate_spatial_splits(raw, self.rows, self.config, self.root / f"epoch_{epoch:03d}" / "spatial")
                for split in results:
                    observations = results[split]["xml_part_localization"]
                    regions = [region for item in observations if item["status"] == "available" for region in item["regions"]]
                    spatial[split]["xml_part_localization"] = dict(
                        eligible_lines=sum(item["status"] == "available" for item in observations),
                        eligible_regions=len(regions),
                        unavailable_reasons={reason: sum(item.get("reason") == reason for item in observations)
                            for reason in sorted({item["reason"] for item in observations if "reason" in item})},
                        metrics={k: float(np.mean([r[k] for r in regions if r[k] is not None]))
                            if any(r[k] is not None for r in regions) else None
                            for k in ("interval_iou", "precision", "recall", "f1", "center_error_px", "boundary_error_px", "success_iou_and_support")})
                    spatial[split]["character_localization"] = "XML text-fragment localization is reported separately only on exact identity matches; multi-letter boxes are not per-letter ground truth"
                (self.root / f"epoch_{epoch:03d}" / "spatial" / "summary.json").write_text(json.dumps(spatial, indent=2))
                gap = None
                if all(results[s]["positive_dtw"] is not None for s in results):
                    gap = results["val_eval"]["positive_dtw"] - results["train_eval"]["positive_dtw"]
                row = dict(run_id=self.root.name, epoch=epoch, checkpoint=f"model_epoch_{epoch:03d}.pth",
                    optimizer_steps_epoch=train_stats.get("line_optimizer_steps", 0) / self.module.CTX.world_size,
                    learning_rate=lr, online_gamma=float(self.module.P.positive_letter_dtw_gamma),
                    fixed_reference_gamma=self.P.positive_letter_dtw_gamma, generalization_gap=gap,
                    protocol=self.config["monitoring"], spatial=spatial, **results,
                    train_online=dict(total=train_loss, positive_dtw=train_stats.get("positive_letter_dtw"),
                        evaluated=train_stats.get("line_evaluated_count"), skipped=train_stats.get("line_skipped_count"),
                        invalid=train_stats.get("line_invalid_count", 0),
                        sigreg_raw=train_stats.get("sigreg_loss"), sigreg_weighted=train_stats.get("sigreg_weighted"),
                        aggregation="optimization-batch weighted; weights change; augmented/dropout-active",
                        stats=train_stats))
                row = json_finite(row)
                self.history.append(row)
                self.write_history()
                val = results["val_eval"]["positive_dtw"]
                improved = val is not None and results["val_eval"]["invalid"] == 0 and val < self.best
                if improved:
                    self.best = val
                payload[0] = dict(row=row, improved=improved)
            except Exception as exc:
                import traceback
                payload[0] = dict(error=traceback.format_exc())
        result = broadcast_monitor_result(payload)
        if result["improved"]:
            self.config["best_validation_dtw"] = dict(metric="val_eval.positive_dtw", value=result["row"]["val_eval"]["positive_dtw"], epoch=epoch,
                fixed_reference_gamma=self.P.positive_letter_dtw_gamma, population=self.config["monitoring"]["population"])
        self.config["evaluated_epoch"] = epoch
        return result

    def write_history(self):
        (self.root / "history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=2, allow_nan=False))
        flat = []
        for row in self.history:
            for split in ("train_online", "train_eval", "val_eval"):
                flat.append({"epoch": row["epoch"], "split": split, "run_id": row["run_id"],
                    "checkpoint": row["checkpoint"], "optimizer_steps_epoch": row["optimizer_steps_epoch"],
                    "population": row["protocol"]["population"],
                    "positive_weight": self.P.positive_letter_dtw_weight, "sigreg_weight": self.P.sigreg_weight,
                    "negative_active": row["protocol"]["negative_active"],
                    "competition_temperature": self.P.positive_letter_dtw_competition_temperature,
                    "dtw_position_prior": self.P.positive_letter_dtw_position_prior,
                    "gamma": row["online_gamma"] if split == "train_online" else row["fixed_reference_gamma"],
                    "generalization_gap": row["generalization_gap"], "learning_rate": row["learning_rate"],
                    **{k: v for k, v in row[split].items() if not isinstance(v, (dict, list))}})
        fields = list(dict.fromkeys(k for r in flat for k in r))
        with (self.root / "history.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = {
            "clean_dtw": [(s, "positive_dtw") for s in ("train_eval", "val_eval")],
            "total_objectives": [(s, "total") for s in ("train_online", "train_eval", "val_eval")],
            "sigreg_raw": [(s, "sigreg_raw") for s in ("train_online", "train_eval", "val_eval")],
            "sigreg_weighted": [(s, "sigreg_weighted") for s in ("train_online", "train_eval", "val_eval")],
        }
        for name, series in groups.items():
            fig, ax = plt.subplots()
            for split, key in series:
                ax.plot([r["epoch"] for r in self.history], [r[split].get(key) for r in self.history], marker="o", label=split)
            ax.set(xlabel="Completed epoch", ylabel=name, title=self.config["monitoring"]["population"])
            ax.legend()
            fig.savefig(self.root / f"{name}.png")
            plt.close(fig)
        for metric in ("interval_iou", "precision", "recall", "f1", "center_error_px", "boundary_error_px", "success_iou_and_support"):
            if not any(r["spatial"][s]["xml_part_localization"]["metrics"][metric] is not None
                       for r in self.history for s in ("train_eval", "val_eval")):
                continue
            fig, ax = plt.subplots()
            for split in ("train_eval", "val_eval"):
                ax.plot([r["epoch"] for r in self.history], [r["spatial"][split]["xml_part_localization"]["metrics"][metric] for r in self.history], label=split)
            ax.set(xlabel="Completed epoch", ylabel=metric, title="Exact-identity XML fragments only")
            ax.legend()
            fig.savefig(self.root / f"xml_fragment_{metric}.png")
            plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot([r["epoch"] for r in self.history], [r["generalization_gap"] for r in self.history], marker="o")
        ax.set(xlabel="Completed epoch", ylabel="val_eval DTW - train_eval DTW")
        fig.savefig(self.root / "generalization_gap.png")
        plt.close(fig)
        for metric in ("iou", "precision", "recall", "f1", "center_error_px", "region_success_rate"):
            if not any(r["spatial"][s]["region_metrics"][metric] is not None
                       for r in self.history for s in ("train_eval", "val_eval")):
                continue
            fig, ax = plt.subplots()
            for split in ("train_eval", "val_eval"):
                ax.plot([r["epoch"] for r in self.history], [r["spatial"][split]["region_metrics"][metric] for r in self.history], label=split)
            ax.set(xlabel="Completed epoch", ylabel=metric)
            ax.legend()
            fig.savefig(self.root / f"spatial_{metric}.png")
            plt.close(fig)
