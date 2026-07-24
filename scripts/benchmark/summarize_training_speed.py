#!/usr/bin/env python3
"""Summarize baseline log timing and optimized JSON timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics


BASELINE_PATTERN = re.compile(
    r"global_batch=(?P<batch>\d+).*?time=(?P<seconds>[0-9.]+)s"
)


def baseline_summary(path: Path):
    times = []
    global_batches = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = BASELINE_PATTERN.search(line)
        if not match:
            continue
        times.append(float(match.group("seconds")))
        global_batches.append(int(match.group("batch")))
    if not times:
        raise ValueError(f"No baseline batch timing lines found in {path}")
    throughputs = [batch / seconds for batch, seconds in zip(global_batches, times)]
    return {
        "batches": len(times),
        "mean_batch_seconds": statistics.mean(times),
        "median_batch_seconds": statistics.median(times),
        "mean_samples_per_second": statistics.mean(throughputs),
    }


def optimized_summary(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("train_stats", {})
    return {
        "epoch_seconds": payload.get("epoch_seconds"),
        "mean_samples_per_second": stats.get("samples_per_second"),
        "surface_cache_hit_rate": stats.get("surface_cache_hit_rate"),
        "jax_batched_calls": stats.get("jax_batched_calls"),
        "jax_batched_items": stats.get("jax_batched_items"),
        "profile": payload.get("profile", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--optimized-json", type=Path, required=True)
    args = parser.parse_args()

    baseline = baseline_summary(args.baseline_log)
    optimized = optimized_summary(args.optimized_json)
    baseline_rate = float(baseline["mean_samples_per_second"])
    optimized_rate = float(optimized.get("mean_samples_per_second") or 0.0)
    speedup = optimized_rate / baseline_rate if baseline_rate > 0 else float("nan")
    result = {
        "baseline": baseline,
        "optimized": optimized,
        "throughput_speedup": speedup,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
