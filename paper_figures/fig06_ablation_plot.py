"""
fig06_ablation_plot.py
======================
Paper-style ablation bar chart from a CSV or JSON results file.

Input CSV format:
    method,metric,value
    CNN only,Top-1 Retrieval,0.42
    CNN + BiLSTM,Top-1 Retrieval,0.55
    ...

Input JSON format:
    [{"method": "CNN only", "metric": "Top-1 Retrieval", "value": 0.42}, ...]

Output:
  fig06_ablation_{metric}.png / .pdf
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from utils.plotting import setup_paper_style, save_figure


def _load_results(path):
    """Return list of {"method", "metric", "value"} dicts."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        import json
        with open(path) as f:
            return json.load(f)
    # CSV (default)
    rows = []
    with open(path) as f:
        header = [h.strip() for h in f.readline().split(",")]
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            rows.append({
                "method": parts[0],
                "metric": parts[1],
                "value":  float(parts[2]),
            })
    return rows


def draw_ablation(results_path, metric, output_dir, show_values=True):
    setup_paper_style()

    records = _load_results(results_path)
    records = [r for r in records if r["metric"] == metric]

    if not records:
        print(f"  [fig06] No rows found for metric='{metric}' in {results_path}")
        return

    methods = [r["method"] for r in records]
    values  = [r["value"]  for r in records]
    x       = np.arange(len(methods))

    # Colour: progressively deeper blue for higher values
    norm_vals = np.array(values)
    norm_vals = (norm_vals - norm_vals.min()) / max(norm_vals.max() - norm_vals.min(), 1e-8)
    colors    = plt.cm.Blues(0.35 + 0.55 * norm_vals)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.4), 5))

    bars = ax.bar(x, values, color=colors, edgecolor="#333333",
                  linewidth=0.7, width=0.6)

    if show_values:
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title(f"Ablation Study — {metric}", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    safe_metric = metric.replace(" ", "_").replace("/", "-")
    save_figure(fig, output_dir, f"fig06_ablation_{safe_metric}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study bar chart (fig06)."
    )
    parser.add_argument("--results_csv",  required=True,
                        help="Path to ablation CSV or JSON file")
    parser.add_argument("--metric",       default="Top-1 Retrieval",
                        help="Metric column value to plot")
    parser.add_argument("--output_dir",   default="paper_figures/outputs")
    parser.add_argument("--no_values",    action="store_true",
                        help="Hide value labels above bars")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_ablation(args.results_csv, args.metric, args.output_dir,
                  show_values=not args.no_values)


if __name__ == "__main__":
    main()
