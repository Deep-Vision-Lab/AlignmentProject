"""Runtime integration for bbox.json subword-box quantitative evaluation."""
from __future__ import annotations

import os

from Evaluation import real_subword_box_metrics as box_metrics
from Evaluation.real_subword_bbox_json_priority import install as install_bbox_json_priority
from Evaluation.real_subword_box_geometry import load_line_annotations


# Install JSON discovery before quantitative metrics resolve annotations.
install_bbox_json_priority()

# The metric module resolves this global at evaluation time. Replace its legacy
# workbook fallback with bbox JSON loading plus the exact crop/resize/pad mapping.
box_metrics.load_line_annotations = load_line_annotations


_PAIR_UNDEFINED_FIELDS = (
    "pair_box_precision",
    "pair_box_recall",
    "pair_box_f1",
    "pair_box_specificity",
    "pair_box_accuracy",
    "mean_box_interval_iou",
    "mean_box_pixel_iou",
    "mean_region_iou",
)
_LINE_UNDEFINED_SUFFIXES = (
    "box_precision",
    "box_recall",
    "box_f1",
    "box_specificity",
    "box_accuracy",
    "mean_box_mask_coverage",
    "shared_box_mask_coverage",
    "box_interval_iou",
    "box_pixel_precision",
    "box_pixel_recall",
    "box_pixel_f1",
    "box_pixel_iou",
    "region_iou",
    "start_error_px",
    "end_error_px",
)
_SUMMARY_UNDEFINED_FIELDS = (
    "box_micro_precision",
    "box_micro_recall",
    "box_micro_f1",
    "box_micro_specificity",
    "box_micro_accuracy",
    "box_macro_precision",
    "box_macro_recall",
    "box_macro_f1",
    "mean_box_interval_iou",
    "mean_box_pixel_iou",
    "mean_shared_subword_matches",
)


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_unavailable_metrics(metrics: dict) -> dict:
    """Use null, not a vacuous perfect score, when GT cannot be evaluated."""
    status = str(metrics.get("real_box_status") or "")
    if metrics.get("real_box_evaluated") and status == "ok":
        return metrics

    for field in _PAIR_UNDEFINED_FIELDS:
        metrics[field] = None
    for prefix in ("line1", "line2"):
        for suffix in _LINE_UNDEFINED_SUFFIXES:
            metrics[f"{prefix}_{suffix}"] = None
    return metrics


def _annotation_detail(row: dict, prefix: str) -> str:
    return (
        f"{prefix}: status={row.get(prefix + '_box_annotation_status')} "
        f"source={row.get(prefix + '_box_annotation_path') or '<none>'} "
        f"error={row.get(prefix + '_box_annotation_error') or '<none>'}"
    )


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

        try:
            metrics = box_metrics.metrics_from_evaluation_row(row, pair, models)
        except FileNotFoundError as exc:
            if _flag("REAL_REQUIRE_BOX_ANNOTATIONS", False):
                raise SystemExit(
                    "Strict bbox JSON evaluation aborted on the first unresolved "
                    f"annotation: {exc}"
                ) from exc
            raise

        row.update(_sanitize_unavailable_metrics(metrics))
        print(
            f"[{row.get('index')}] real-box status={row.get('real_box_status')} "
            f"shared={row.get('shared_subword_matches')} "
            f"precision={row.get('pair_box_precision')} "
            f"recall={row.get('pair_box_recall')} "
            f"f1={row.get('pair_box_f1')} "
            f"interval_iou={row.get('mean_box_interval_iou')}",
            flush=True,
        )
        if row.get("real_box_status") != "ok":
            print(
                f"[{row.get('index')}] bbox annotation detail: "
                f"{_annotation_detail(row, 'line1')}; "
                f"{_annotation_detail(row, 'line2')}",
                flush=True,
            )
        return row

    def aggregate(rows):
        summary = original_aggregate(rows)

        # Only fully resolved, shared-subword annotation rows may contribute to
        # quantitative localization scores. Keep all rows for status accounting.
        aggregate_rows = []
        for row in rows:
            if str(row.get("real_box_status") or "") == "ok":
                aggregate_rows.append(row)
            else:
                unavailable = dict(row)
                unavailable["real_box_evaluated"] = False
                aggregate_rows.append(unavailable)

        box_summary = box_metrics.aggregate(aggregate_rows)
        if int(box_summary.get("real_box_samples") or 0) == 0:
            for field in _SUMMARY_UNDEFINED_FIELDS:
                box_summary[field] = None
        summary.update(box_summary)
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
