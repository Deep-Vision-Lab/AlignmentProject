#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value, digits=5):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()

    tiny = load_json(root / "real_tiny_overfit" / "metrics.json")

    def line(stage, side):
        return load_json(root / f"{stage}_side{side}" / "summary.json")

    def pair(stage):
        return load_json(root / f"{stage}_pair" / "metrics.json")

    datasets = {
        "stage_a_1": line("stage_a", 1),
        "stage_a_2": line("stage_a", 2),
        "stage_b_1": line("stage_b", 1),
        "stage_b_2": line("stage_b", 2),
        "legacy_1": line("legacy", 1),
        "legacy_2": line("legacy", 2),
        "stage_a_pair": pair("stage_a"),
        "stage_b_pair": pair("stage_b"),
        "legacy_pair": pair("legacy"),
    }

    warnings = []
    conclusions = []

    def warn(stage, message, evidence):
        warnings.append({"stage": stage, "message": message, "evidence": evidence})

    if tiny:
        if tiny.get("passed"):
            conclusions.append(
                "Local capacity gate PASSED: the CNN+decoder can memorize 8 real synthetic windows."
            )
        else:
            conclusions.append(
                "Local capacity gate FAILED before Transformer/fusion/DTW."
            )
            warn(
                "local-capacity",
                "The local CNN+decoder could not overfit 8 real synthetic windows.",
                tiny,
            )

    def inspect_line(label, data):
        if not data:
            return
        target_div = data.get("mean_target_pixel_std_across_windows")
        out_div = data.get("mean_reconstruction_pixel_std_across_windows")
        rank = data.get("primitive_effective_rank")
        structure = data.get("reconstruction_target_structure_correlation")
        decoded_cos = data.get("mean_reconstruction_pairwise_cosine")
        target_cos = data.get("mean_target_pairwise_cosine")

        if target_div and out_div is not None and target_div > 1e-8:
            ratio = out_div / target_div
            if ratio < 0.35:
                warn(
                    label,
                    "Reconstruction diversity is much lower than target diversity.",
                    {"output_target_diversity_ratio": ratio},
                )
        if rank is not None and rank < 3.0:
            warn(
                label,
                "Primitive effective rank is very low.",
                {"primitive_effective_rank": rank},
            )
        if structure is not None and structure < 0.20:
            warn(
                label,
                "Decoded pairwise structure poorly follows target-window structure.",
                {"decode_structure_correlation": structure},
            )
        if (
            decoded_cos is not None
            and target_cos is not None
            and decoded_cos - target_cos > 0.25
        ):
            warn(
                label,
                "Decoded windows are substantially more mutually similar than targets.",
                {
                    "decoded_pairwise_cosine": decoded_cos,
                    "target_pairwise_cosine": target_cos,
                },
            )

    for key in ("stage_a_1", "stage_a_2", "stage_b_1", "stage_b_2", "legacy_1", "legacy_2"):
        inspect_line(key.replace("_", "-"), datasets[key])

    a1 = datasets["stage_a_1"]
    b1 = datasets["stage_b_1"]
    if a1 and b1:
        a_mae = a1.get("mean_restoration_mae")
        b_mae = b1.get("mean_restoration_mae")
        a_rank = a1.get("primitive_effective_rank")
        b_rank = b1.get("primitive_effective_rank")
        a_div = a1.get("mean_reconstruction_pixel_std_across_windows")
        b_div = b1.get("mean_reconstruction_pixel_std_across_windows")

        if a_mae and b_mae and b_mae > a_mae * 1.25:
            warn(
                "stage-b",
                "Restoration became markedly worse after Stage-B alignment training.",
                {"stage_a_mae": a_mae, "stage_b_mae": b_mae},
            )
            conclusions.append(
                "Stage B degraded restoration relative to Stage A: alignment training may be causing catastrophic forgetting."
            )
        if a_rank and b_rank and b_rank < a_rank * 0.70:
            warn(
                "stage-b",
                "Primitive rank dropped strongly from Stage A to Stage B.",
                {"stage_a_rank": a_rank, "stage_b_rank": b_rank},
            )
        if a_div and b_div and b_div < a_div * 0.60:
            warn(
                "stage-b",
                "Reconstruction diversity dropped strongly from Stage A to Stage B.",
                {"stage_a_diversity": a_div, "stage_b_diversity": b_div},
            )

    def inspect_pair(label, data):
        if not data:
            return
        local = data.get("local", {})
        context = data.get("context", {})
        fused = data.get("fused", {})
        dtw = data.get("dtw", {})
        fusion = data.get("fusion_ablation", {})
        restoration = data.get("restoration", {})

        lr = local.get("side1_effective_rank")
        cr = context.get("side1_effective_rank")
        fr = fused.get("side1_effective_rank")
        if lr and cr and cr < lr * 0.65:
            warn(label, "Transformer context reduces effective rank strongly.", {"local_rank": lr, "context_rank": cr})
        if lr and fr and fr < lr * 0.65:
            warn(label, "Fusion reduces effective rank strongly.", {"local_rank": lr, "fused_rank": fr})

        lnw = local.get("nw_normalized_score")
        cnw = context.get("nw_normalized_score")
        fnw = fused.get("nw_normalized_score")
        if lnw is not None and cnw is not None and cnw < lnw - 0.05:
            warn(label, "Transformer context hurts same-pair image-image alignment.", {"local_nw": lnw, "context_nw": cnw})
        if lnw is not None and fnw is not None and fnw < lnw - 0.05:
            warn(label, "Fusion hurts same-pair image-image alignment.", {"local_nw": lnw, "fused_nw": fnw})

        margin = dtw.get("soft_negative_minus_positive")
        if margin is not None and margin <= 0:
            warn(
                label,
                "Negative transcript is not more expensive than the positive transcript.",
                {
                    "positive_soft_dtw": dtw.get("positive_soft_cost"),
                    "negative_soft_dtw": dtw.get("negative_soft_cost"),
                    "negative_minus_positive": margin,
                },
            )

        lc = fusion.get("local_contribution_change")
        cc = fusion.get("context_contribution_change")
        if lc is not None and lc < 1e-3:
            warn(label, "Fusion is almost insensitive to the local vector.", {"local_contribution_change": lc})
        if cc is not None and cc < 1e-3:
            warn(label, "Fusion is almost insensitive to context.", {"context_contribution_change": cc})

        od = restoration.get("side1_output_diversity")
        td = restoration.get("side1_target_diversity")
        if od is not None and td and od / td < 0.35:
            warn(label, "Trained decoder output diversity is far below target diversity.", {"output_diversity": od, "target_diversity": td})

    for key in ("stage_a_pair", "stage_b_pair", "legacy_pair"):
        inspect_pair(key.replace("_", "-"), datasets[key])

    stage_b_pair = datasets["stage_b_pair"]
    if stage_b_pair:
        lnw = stage_b_pair.get("local", {}).get("nw_normalized_score")
        fnw = stage_b_pair.get("fused", {}).get("nw_normalized_score")
        margin = stage_b_pair.get("dtw", {}).get("soft_negative_minus_positive")
        if lnw is not None and fnw is not None and fnw >= lnw - 0.02:
            conclusions.append(
                "Stage-B fusion does not obviously degrade same-pair synthetic alignment relative to local features."
            )
        if margin is not None and margin > 0:
            conclusions.append(
                "Positive/negative text-DTW ordering is correct on the inspected synthetic line."
            )

    if not conclusions:
        conclusions.append(
            "No complete trained-checkpoint result was available; inspect preprocessing and structural outputs."
        )

    lines = [
        "# Synthetic restoration diagnostic report",
        "",
        "This report uses actual synthetic lines/pairs. Structural algorithm tests are listed in statuses.tsv.",
        "",
        "## First conclusions",
        "",
    ]
    lines.extend(f"- {x}" for x in conclusions)

    lines += ["", "## Warnings", ""]
    if warnings:
        for w in warnings:
            lines.append(
                f"- **{w['stage']}** — {w['message']} Evidence: {json.dumps(w['evidence'])}"
            )
    else:
        lines.append("- No automatic warning threshold fired. Inspect the PNG outputs manually.")

    if tiny:
        lines += [
            "",
            "## Real synthetic 8-window capacity gate",
            "",
            f"- status: **{'PASS' if tiny.get('passed') else 'FAIL'}**",
            f"- initial L1: {fmt(tiny.get('initial_l1'))}",
            f"- final L1: {fmt(tiny.get('final_l1'))}",
            f"- loss ratio: {fmt(tiny.get('loss_ratio'))}",
            f"- target diversity: {fmt(tiny.get('target_diversity'))}",
            f"- output diversity: {fmt(tiny.get('output_diversity'))}",
            f"- swap delta: {fmt(tiny.get('swap_delta'))}",
        ]

    line_rows = []
    for label, data in [
        ("Stage A side1", datasets["stage_a_1"]),
        ("Stage A side2", datasets["stage_a_2"]),
        ("Stage B side1", datasets["stage_b_1"]),
        ("Stage B side2", datasets["stage_b_2"]),
        ("Legacy side1", datasets["legacy_1"]),
        ("Legacy side2", datasets["legacy_2"]),
    ]:
        if data:
            line_rows.append(
                (
                    label,
                    fmt(data.get("mean_restoration_mae")),
                    fmt(data.get("primitive_effective_rank"), 3),
                    fmt(data.get("semantic_effective_rank"), 3),
                    fmt(data.get("mean_reconstruction_pairwise_cosine"), 3),
                    fmt(data.get("mean_target_pairwise_cosine"), 3),
                    fmt(data.get("reconstruction_target_structure_correlation"), 3),
                )
            )
    if line_rows:
        lines += [
            "",
            "## Per-line checkpoint diagnostics",
            "",
            "| checkpoint | restoration MAE | primitive rank | fused rank | decoded pair cosine | target pair cosine | decode-structure corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in line_rows)

    pair_rows = []
    for label, data in [
        ("Stage A", datasets["stage_a_pair"]),
        ("Stage B", datasets["stage_b_pair"]),
        ("Legacy", datasets["legacy_pair"]),
    ]:
        if data:
            pair_rows.append(
                (
                    label,
                    fmt(data.get("local", {}).get("nw_normalized_score")),
                    fmt(data.get("context", {}).get("nw_normalized_score")),
                    fmt(data.get("fused", {}).get("nw_normalized_score")),
                    fmt(data.get("dtw", {}).get("positive_soft_cost")),
                    fmt(data.get("dtw", {}).get("negative_soft_cost")),
                    fmt(data.get("dtw", {}).get("soft_negative_minus_positive")),
                )
            )
    if pair_rows:
        lines += [
            "",
            "## Same-pair image-image and text-DTW diagnostics",
            "",
            "| checkpoint | local NW | context NW | fused NW | positive soft DTW | negative soft DTW | neg-pos margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in pair_rows)

    lines += [
        "",
        "## Find the first failing stage",
        "",
        "1. real_tiny_overfit fails -> local encoder/decoder capacity or optimization.",
        "2. tiny-overfit passes but Stage A collapses -> Stage-A training/data/loss problem.",
        "3. Stage A is good but Stage B worsens -> DTW/alignment training is damaging the local representation.",
        "4. local is healthy but context rank/NW drops -> Transformer context problem.",
        "5. context is healthy but fused rank/NW drops -> fusion-head problem.",
        "6. image features are healthy but negative DTW is not costlier than positive -> text grounding / contrastive-DTW problem.",
        "7. text DTW is healthy but fused image-image NW is poor -> insufficient cross-image invariance in the training objective.",
        "",
        "The thresholds above are diagnostic heuristics, not paper metrics. Confirm them by opening the generated images.",
    ]

    report = "\n".join(lines) + "\n"
    (root / "DIAGNOSIS.md").write_text(report, encoding="utf-8")
    (root / "diagnosis.json").write_text(
        json.dumps(
            {"warnings": warnings, "conclusions": conclusions},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"Saved: {root / 'DIAGNOSIS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
