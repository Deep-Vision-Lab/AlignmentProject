#!/usr/bin/env python3
"""Automatic human-readable experiment tracker for SLURM research runs.

The JSON file is machine state; a sibling Markdown file is regenerated after every
change and is the file researchers should read. Jobs update themselves through
run_tracked_stage.sh. `refresh` is a recovery path for timeouts/node failures.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def md_path(json_path: Path) -> Path:
    return json_path.with_suffix(".md")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"run": {}, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_locked(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_markdown(state, md_path(path))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def esc(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def code(value: str | None) -> str:
    return f"`{value}`" if value else "—"


def status_icon(status: str) -> str:
    return {
        "PENDING": "⬜ PENDING",
        "RUNNING": "⏳ RUNNING",
        "COMPLETED": "✅ COMPLETED",
        "FAILED": "❌ FAILED",
        "BLOCKED": "⛔ BLOCKED",
        "CANCELLED": "🚫 CANCELLED",
    }.get(status, esc(status))


def render_markdown(state: dict[str, Any], path: Path) -> None:
    run = state.get("run", {})
    stages = state.get("stages", {})
    lines = [
        f"# Experiment Tracker — {run.get('run_prefix', 'unnamed')}",
        "",
        f"Last updated: **{now()}**",
        "",
        f"- Branch: {code(run.get('branch'))}",
        f"- Commit: {code(run.get('commit'))}",
        f"- Backend: {code(run.get('backend'))}",
        f"- Frozen Bridge V2: {code(run.get('bridge_dataset'))}",
        f"- Results root: {code(run.get('results_root'))}",
        "",
        "Technical status and scientific verdict are intentionally separate. A job may be ✅ COMPLETED but scientifically FAIL.",
        "",
        "| V | Stage | Job name | SLURM ID | Technical | Scientific verdict | Log | Main artifact/result |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for key, item in stages.items():
        tech = item.get("technical_status", "PENDING")
        v = "✅" if tech == "COMPLETED" else ("❌" if tech in {"FAILED", "CANCELLED"} else "")
        artifact = item.get("result_root") or item.get("checkpoint") or item.get("artifact")
        lines.append(
            "| {v} | {stage} — {desc} | {job} | {jid} | {tech} | {sci} | {log} | {artifact} |".format(
                v=v,
                stage=esc(key),
                desc=esc(item.get("description", "")),
                job=esc(item.get("job_name", "")),
                jid=esc(item.get("job_id", "")),
                tech=status_icon(tech),
                sci=esc(item.get("scientific_verdict", "—")),
                log=code(item.get("log_path")),
                artifact=code(artifact),
            )
        )

    lines += ["", "## Stage details", ""]
    for key, item in stages.items():
        lines += [
            f"### {key} — {item.get('description', '')}",
            "",
            f"- Technical: **{status_icon(item.get('technical_status', 'PENDING'))}**",
            f"- Scientific verdict: **{item.get('scientific_verdict', '—')}**",
            f"- Why: {item.get('reason') or 'Pending.'}",
            f"- SLURM: job `{item.get('job_name', '')}` / ID `{item.get('job_id', '')}` / dependency `{item.get('dependency', '') or 'none'}`",
            f"- Log: {code(item.get('log_path'))}",
            f"- Checkpoint: {code(item.get('checkpoint'))}",
            f"- Results: {code(item.get('result_root'))}",
            f"- Artifact/data: {code(item.get('artifact'))}",
        ]
        metrics = item.get("metrics") or {}
        if metrics:
            lines += ["- Key results:"]
            for mk, mv in metrics.items():
                if isinstance(mv, float):
                    lines.append(f"  - `{mk}` = **{mv:.4f}**")
                else:
                    lines.append(f"  - `{mk}` = **{mv}**")
        lines.append("")

    lines += [
        "## Useful commands",
        "",
        "```bash",
        "squeue -u \"$USER\" -o \"%.18i %.38j %.2t %.10M %.55R\"",
        f"python scripts/pipeline/experiment_tracker.py refresh --tracker {path.with_suffix('.json')}",
        "```",
        "",
        "`refresh` is useful after SLURM time limits/node failures where a process may be killed before its EXIT trap can update this file.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def rank_auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    data = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    data.sort(key=lambda z: z[0])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(data):
        j = i + 1
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = rank
        i = j
    rs = sum(r for r, z in zip(ranks, data) if z[1] == 1)
    return (rs - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def average_precision(pos: list[float], neg: list[float]) -> float | None:
    if not pos:
        return None
    data = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg], reverse=True)
    tp = 0
    score = 0.0
    for rank, (_, y) in enumerate(data, start=1):
        if y:
            tp += 1
            score += tp / rank
    return score / len(pos)


def csv_values(path: Path, key: str) -> list[float]:
    if not path.is_file():
        return []
    out: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            value = safe_float(row.get(key))
            if value is not None:
                out.append(value)
    return out


def discrimination_metrics(root: Path) -> dict[str, Any]:
    rows = []
    for threshold in ("0.40", "0.50", "0.60", "0.65", "0.70"):
        pp = root / f"raw_t{threshold}_positive" / "samples.csv"
        np = root / f"raw_t{threshold}_negative" / "samples.csv"
        if not pp.is_file() or not np.is_file():
            continue
        ps = csv_values(pp, "score"); ns = csv_values(np, "score")
        pst = csv_values(pp, "path_steps"); nst = csv_values(np, "path_steps")
        pm = csv_values(pp, "line1_matched_fraction"); nm = csv_values(np, "line1_matched_fraction")
        row = {
            "threshold": float(threshold),
            "score_auc": rank_auc(ps, ns),
            "steps_auc": rank_auc(pst, nst),
            "matched_auc": rank_auc(pm, nm),
            "average_precision": average_precision(ps, ns),
            "positive_steps": sum(pst) / len(pst) if pst else None,
            "negative_steps": sum(nst) / len(nst) if nst else None,
            "positive_score": sum(ps) / len(ps) if ps else None,
            "negative_score": sum(ns) / len(ns) if ns else None,
        }
        vals = [x for x in (row["steps_auc"], row["matched_auc"]) if x is not None]
        row["structural_auc"] = max(vals) if vals else None
        rows.append(row)
    if not rows:
        return {}
    best = max(rows, key=lambda r: (r.get("structural_auc") or -1.0, r["threshold"]))
    return {"best": best, "thresholds": rows}


def flatten_numeric_json(path: Path, prefix: str = "") -> dict[str, float]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, float] = {}
    def walk(value: Any, key: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{key}.{k}" if key else str(k))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            x = safe_float(value)
            if x is not None:
                out[(prefix + key) if prefix else key] = x
    walk(obj, "")
    return out


def summarize_stage(item: dict[str, Any], exit_code: int) -> tuple[str, str, dict[str, Any]]:
    kind = item.get("kind", "other")
    root = Path(item["result_root"]) if item.get("result_root") else None
    artifact = Path(item["artifact"]) if item.get("artifact") else None
    if exit_code != 0:
        return "FAIL", f"SLURM stage exited with code {exit_code}; inspect {item.get('log_path') or 'the job log'}.", {}
    if kind == "dataset":
        meta = artifact / "metadata.json" if artifact else Path("__missing__")
        values = flatten_numeric_json(meta)
        metrics = {k: v for k, v in values.items() if k in {"anchors_written", "positive_rows", "negative_rows", "positive_shared_islands_1", "positive_shared_islands_2", "positive_shared_islands_3"}}
        return "PASS", "Dataset build and smoke validation completed successfully.", metrics
    if kind == "train":
        checkpoint = Path(item.get("checkpoint") or "")
        exists = checkpoint.is_file() if str(checkpoint) else False
        return ("PASS" if exists else "FAIL"), ("Training completed and expected checkpoint exists." if exists else "Training exited successfully but expected checkpoint was not found."), {"checkpoint_exists": int(exists)}
    if kind == "qualitative":
        pngs = len(list(root.rglob("*.png"))) if root and root.exists() else 0
        return ("VISUAL REVIEW" if pngs else "FAIL"), (f"Qualitative evaluation completed and produced {pngs} PNG artifact(s); model-quality judgment requires visual review." if pngs else "No qualitative PNG artifacts were found."), {"png_artifacts": pngs}
    if kind == "quantitative":
        disc = discrimination_metrics(root / "discrimination") if root else {}; best = disc.get("best") or {}
        if not best: return "FAIL", "Quantitative job completed but discrimination CSV outputs were not found.", {}
        structural = best.get("structural_auc"); pos_steps = best.get("positive_steps"); neg_steps = best.get("negative_steps"); gap = (pos_steps-neg_steps) if pos_steps is not None and neg_steps is not None else None
        min_auc=float(os.environ.get("TRACKER_MIN_STRUCTURAL_AUC","0.65")); min_pos=float(os.environ.get("TRACKER_MIN_POS_STEPS","8")); min_gap=float(os.environ.get("TRACKER_MIN_STEP_GAP","2"))
        passed=structural is not None and structural>=min_auc and pos_steps is not None and pos_steps>=min_pos and gap is not None and gap>=min_gap
        metrics={"best_threshold":best.get("threshold"),"score_auc":best.get("score_auc"),"steps_auc":best.get("steps_auc"),"matched_auc":best.get("matched_auc"),"structural_auc":structural,"average_precision":best.get("average_precision"),"positive_steps":pos_steps,"negative_steps":neg_steps,"step_gap":gap}
        reason=(f"PASS: structural AUC={structural:.4f} >= {min_auc:.2f}, positive steps={pos_steps:.2f} >= {min_pos:.1f}, step gap={gap:.2f} >= {min_gap:.1f}." if passed else f"FAIL: requires structural AUC>={min_auc:.2f}, positive steps>={min_pos:.1f}, and positive-negative step gap>={min_gap:.1f}; got structural={structural}, positive_steps={pos_steps}, gap={gap}.")
        return "PASS" if passed else "FAIL", reason, metrics
    if kind == "bridge_eval":
        summary=root/"bridge_summary.json" if root else Path("__missing__")
        if not summary.is_file(): return "FAIL", "Bridge evaluation completed but bridge_summary.json was not found.", {}
        obj=json.loads(summary.read_text(encoding="utf-8")); sv=[safe_float(obj.get(k)) for k in ("path_steps_auc","line1_matched_fraction_auc","line2_matched_fraction_auc")]; sv=[x for x in sv if x is not None]; structural=max(sv) if sv else None
        pos=safe_float(obj.get("path_steps_positive_mean")); neg=safe_float(obj.get("path_steps_negative_mean")); gap=pos-neg if pos is not None and neg is not None else None; min_auc=float(os.environ.get("TRACKER_BRIDGE_MIN_AUC","0.65")); passed=structural is not None and structural>=min_auc and gap is not None and gap>0
        metrics={"score_auc":safe_float(obj.get("score_auc")),"path_steps_auc":safe_float(obj.get("path_steps_auc")),"matched_line1_auc":safe_float(obj.get("line1_matched_fraction_auc")),"matched_line2_auc":safe_float(obj.get("line2_matched_fraction_auc")),"structural_auc":structural,"positive_steps":pos,"negative_steps":neg,"step_gap":gap}
        return "PASS" if passed else "FAIL", f"{'PASS' if passed else 'FAIL'}: Bridge requires structural AUC>={min_auc:.2f} and positive path steps > negative path steps; got structural={structural}, gap={gap}.", metrics
    if kind == "final":
        summary=root/"final_summary.json" if root else Path("__missing__")
        if not summary.is_file(): return "FAIL", "Final evaluation completed but final_summary.json was not found.", {}
        obj=json.loads(summary.read_text(encoding="utf-8")); binary=obj.get("binary",{}); score=binary.get("score",{}); steps=binary.get("path_steps",{}); m1=binary.get("line1_matched_fraction",{}); m2=binary.get("line2_matched_fraction",{})
        sv=[safe_float(x.get("roc_auc")) for x in (steps,m1,m2)]; sv=[x for x in sv if x is not None]; structural=max(sv) if sv else None; pos=safe_float(steps.get("positive_mean")); neg=safe_float(steps.get("negative_mean")); gap=pos-neg if pos is not None and neg is not None else None
        min_auc=float(os.environ.get("TRACKER_MIN_STRUCTURAL_AUC","0.65")); min_pos=float(os.environ.get("TRACKER_MIN_POS_STEPS","8")); min_gap=float(os.environ.get("TRACKER_MIN_STEP_GAP","2")); passed=structural is not None and structural>=min_auc and pos is not None and pos>=min_pos and gap is not None and gap>=min_gap
        labels=obj.get("labels",{}); metrics={"score_auc":safe_float(score.get("roc_auc")),"score_ap":safe_float(score.get("average_precision")),"steps_auc":safe_float(steps.get("roc_auc")),"matched_line1_auc":safe_float(m1.get("roc_auc")),"matched_line2_auc":safe_float(m2.get("roc_auc")),"structural_auc":structural,"positive_steps":pos,"negative_steps":neg,"step_gap":gap,"high_samples":(labels.get("high_match",{}) or {}).get("samples"),"medium_samples":(labels.get("medium_match",{}) or {}).get("samples"),"low_samples":(labels.get("low_match",{}) or {}).get("samples"),"negative_samples":(labels.get("no_shared_content",{}) or {}).get("samples")}
        return "PASS" if passed else "FAIL", f"{'PASS' if passed else 'FAIL'}: final frozen gate requires structural AUC>={min_auc:.2f}, positive steps>={min_pos:.1f}, gap>={min_gap:.1f}; got structural={structural}, positive_steps={pos}, gap={gap}.", metrics
    return "PASS", "Stage completed successfully.", {}


def cmd_init(args):
    state={"run":{"run_prefix":args.run_prefix,"branch":args.branch,"commit":args.commit,"backend":args.backend,"bridge_dataset":args.bridge_dataset or "","results_root":args.results_root or "","created_at":now()},"stages":{}}; save_locked(Path(args.tracker),state); print(md_path(Path(args.tracker)))
def cmd_register(args):
    path=Path(args.tracker); state=load_json(path); old=state.setdefault("stages",{}).get(args.stage,{}); item=dict(old); item.update({"description":args.description or old.get("description",""),"kind":args.kind or old.get("kind","other"),"job_id":args.job_id or old.get("job_id",""),"job_name":args.job_name or old.get("job_name",""),"dependency":args.dependency or old.get("dependency",""),"log_path":args.log_path or old.get("log_path",""),"artifact":args.artifact or old.get("artifact",""),"checkpoint":args.checkpoint or old.get("checkpoint",""),"result_root":args.result_root or old.get("result_root",""),"technical_status":old.get("technical_status","PENDING"),"scientific_verdict":old.get("scientific_verdict","—"),"reason":old.get("reason","Waiting for job to run.")}); state["stages"][args.stage]=item; save_locked(path,state)
def cmd_running(args):
    path=Path(args.tracker); state=load_json(path); item=state.setdefault("stages",{}).setdefault(args.stage,{}); item.update({"technical_status":"RUNNING","started_at":now()});
    if args.job_id: item["job_id"]=args.job_id
    if args.log_path: item["log_path"]=args.log_path
    item["reason"]="Job is currently running."; save_locked(path,state)
def cmd_finish(args):
    path=Path(args.tracker); state=load_json(path); item=state.setdefault("stages",{}).setdefault(args.stage,{}); rc=int(args.exit_code); item["technical_status"]="COMPLETED" if rc==0 else "FAILED"; item["exit_code"]=rc; item["completed_at"]=now(); verdict,reason,metrics=summarize_stage(item,rc); item["scientific_verdict"]=verdict; item["reason"]=reason; item["metrics"]=metrics; save_locked(path,state)
def slurm_state(job_id):
    if not job_id: return None
    try: out=subprocess.check_output(["sacct","-j",job_id,"-X","-n","-P","-o","State,ExitCode"],text=True,stderr=subprocess.DEVNULL)
    except Exception: return None
    for line in out.splitlines():
        if line.strip():
            parts=line.split("|"); return parts[0].split()[0].upper(), parts[1] if len(parts)>1 else ""
    return None
def cmd_refresh(args):
    path=Path(args.tracker); state=load_json(path)
    for _,item in state.get("stages",{}).items():
        status=slurm_state(str(item.get("job_id","")))
        if not status: continue
        slurm,exit_code=status
        if slurm in {"RUNNING","COMPLETING"}: item["technical_status"]="RUNNING"
        elif slurm in {"PENDING","CONFIGURING"}: item["technical_status"]="PENDING"
        elif slurm=="COMPLETED": item["technical_status"]="COMPLETED"; v,r,m=summarize_stage(item,0); item["scientific_verdict"]=v; item["reason"]=r; item["metrics"]=m
        elif slurm in {"CANCELLED","TIMEOUT","FAILED","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED"}: item["technical_status"]="CANCELLED" if slurm=="CANCELLED" else "FAILED"; item["scientific_verdict"]="FAIL"; item["reason"]=f"SLURM state={slurm}, ExitCode={exit_code}. Inspect {item.get('log_path') or 'the job log'}."
    save_locked(path,state); print(md_path(path))
def parser():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("init"); a.add_argument("--tracker",required=True); a.add_argument("--run-prefix",required=True); a.add_argument("--branch",required=True); a.add_argument("--commit",required=True); a.add_argument("--backend",required=True); a.add_argument("--bridge-dataset",default=""); a.add_argument("--results-root",default=""); a.set_defaults(func=cmd_init)
    a=sub.add_parser("register"); a.add_argument("--tracker",required=True); a.add_argument("--stage",required=True); a.add_argument("--description",default=""); a.add_argument("--kind",default="other"); a.add_argument("--job-id",default=""); a.add_argument("--job-name",default=""); a.add_argument("--dependency",default=""); a.add_argument("--log-path",default=""); a.add_argument("--artifact",default=""); a.add_argument("--checkpoint",default=""); a.add_argument("--result-root",default=""); a.set_defaults(func=cmd_register)
    a=sub.add_parser("running"); a.add_argument("--tracker",required=True); a.add_argument("--stage",required=True); a.add_argument("--job-id",default=""); a.add_argument("--log-path",default=""); a.set_defaults(func=cmd_running)
    a=sub.add_parser("finish"); a.add_argument("--tracker",required=True); a.add_argument("--stage",required=True); a.add_argument("--exit-code",required=True,type=int); a.set_defaults(func=cmd_finish)
    a=sub.add_parser("refresh"); a.add_argument("--tracker",required=True); a.set_defaults(func=cmd_refresh)
    return p
if __name__=="__main__":
    ns=parser().parse_args(); ns.func(ns)
