"""Runtime integration for real Excel subword-box quantitative evaluation."""
from __future__ import annotations

from Evaluation import real_subword_box_metrics as box_metrics
from Evaluation.real_subword_box_geometry import load_line_annotations


# The metric module resolves this global at evaluation time. Replace its simple
# scaling fallback with the exact foreground-crop, aspect-resize and pad map.
box_metrics.load_line_annotations = load_line_annotations


def install(sw_runner_module) -> None:
    if getattr(sw_runner_module, "_real_subword_box_patch_installed", False):
        return

    original_evaluate = sw_runner_module.evaluate_sample
    original_aggregate = sw_runner_module.aggregate
    original_fieldnames = sw_runner_module.batch_fieldnames

    def evaluate_sample(*args, **kwargs):
        row = original_evaluate(*args, **kwargs)
        dataset_type = kwargs.get("dataset_type")
        if dataset_type is None and len(args) >= 3:
            dataset_type = args[2]
        if str(dataset_type or row.get("dataset_type", "")).lower() != "real":
            return row

        models = kwargs.get("models") if "models" in kwargs else (args[0] if args else None)
        pair = kwargs.get("pair") if "pair" in kwargs else (args[1] if len(args) > 1 else None)
        if models is None or pair is None:
            return row

        metrics = box_metrics.metrics_from_evaluation_row(row, pair, models)
        row.update(metrics)
        print(
            f"[{row.get('index')}] real-box status={row.get('real_box_status')} "
            f"shared={row.get('shared_subword_matches')} "
            f"precision={row.get('pair_box_precision')} "
            f"recall={row.get('pair_box_recall')} "
            f"f1={row.get('pair_box_f1')} "
            f"interval_iou={row.get('mean_box_interval_iou')}",
            flush=True,
        )
        return row

    def aggregate(rows):
        summary = original_aggregate(rows)
        summary.update(box_metrics.aggregate(rows))
        return summary

    def batch_fieldnames():
        names = list(original_fieldnames())
        for name in box_metrics.fieldnames():
            if name not in names:
                names.append(name)
        return names

    sw_runner_module.evaluate_sample = evaluate_sample
    sw_runner_module._evaluate_sample = evaluate_sample
    sw_runner_module.aggregate = aggregate
    sw_runner_module._aggregate = aggregate
    sw_runner_module.batch_fieldnames = batch_fieldnames
    sw_runner_module._batch_fieldnames = batch_fieldnames
    sw_runner_module._real_subword_box_patch_installed = True
