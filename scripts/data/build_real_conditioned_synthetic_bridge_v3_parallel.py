#!/usr/bin/env python3
"""Parallel offline launcher for dense/resilient Bridge V3 generation.

The expensive synthetic rendering step is split into deterministic anchor shards.
Each shard is a separate Python process, so CPU-heavy Python/Pillow work is not
serialized by the GIL.  Every shard still loads the full eligible transcript pool;
only the anchors it writes are sharded.  This preserves the negative-sampling and
no-overlap semantics of the single-process builder.

Worker count defaults to BRIDGE_BUILD_WORKERS, then SLURM_CPUS_PER_TASK, then the
machine CPU count.  BLAS/OpenMP libraries are pinned to one thread inside each worker
to avoid N x N oversubscription.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DENSE_BUILDER = PROJECT_DIR / "scripts" / "data" / "build_real_conditioned_synthetic_bridge_v3_dense.py"

SUM_KEYS = {
    "anchors_considered",
    "anchors_written",
    "positive_rows",
    "negative_rows",
    "positive_shared_islands_1",
    "positive_shared_islands_2",
    "positive_shared_islands_3",
    "positive_full_sentence_rows",
    "mixed_font_positive_rows",
    "mixed_font_negative_rows",
    "positive_render_attempts",
    "negative_render_attempts",
    "negative_compose_attempts",
}


def _parse_launcher_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    known, _ = parser.parse_known_args(argv)
    return known


def _replace_option(argv: list[str], name: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(argv):
        token = argv[i]
        if token == name:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value after {name}")
            out.extend([name, value])
            i += 2
            replaced = True
            continue
        if token.startswith(name + "="):
            out.append(f"{name}={value}")
            i += 1
            replaced = True
            continue
        out.append(token)
        i += 1
    if not replaced:
        out.extend([name, value])
    return out


def _worker_count(max_anchors: int) -> int:
    requested = int(
        os.environ.get(
            "BRIDGE_BUILD_WORKERS",
            os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)),
        )
    )
    requested = max(1, requested)
    if max_anchors > 0:
        requested = min(requested, max_anchors)
    return requested


def _run_shard(index: int, workers: int, base_args: list[str], shard_root: Path) -> tuple[int, Path]:
    shard_dir = shard_root / f"shard_{index:03d}"
    log_path = shard_root / "logs" / f"shard_{index:03d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = _replace_option(base_args, "--output-dir", str(shard_dir))
    if "--overwrite" not in args:
        args.append("--overwrite")

    env = os.environ.copy()
    env.update(
        {
            "BRIDGE_NUM_SHARDS": str(workers),
            "BRIDGE_SHARD_INDEX": str(index),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    command = [sys.executable, str(DENSE_BUILDER), *args]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"worker={index + 1}/{workers}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(result.returncode), log_path


def _copy_tree_contents(source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Parallel shard output collision: {destination}")
        shutil.copy2(item, destination)


def _merge(shard_root: Path, output_dir: Path, workers: int) -> None:
    metas: list[dict] = []
    rows: list[dict] = []
    for index in range(workers):
        shard = shard_root / f"shard_{index:03d}"
        metadata_path = shard / "metadata.json"
        manifest_path = shard / "dataset_manifest.jsonl"
        if not metadata_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"Shard {index} finished without metadata/manifest: {shard}")
        metas.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        for raw in manifest_path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                rows.append(json.loads(raw))

    if not metas:
        raise RuntimeError("No shard metadata was produced")

    output_dir.mkdir(parents=True, exist_ok=False)
    for family in ("images", "texts", "masks"):
        for index in range(workers):
            _copy_tree_contents(
                shard_root / f"shard_{index:03d}" / family,
                output_dir / family,
            )

    rows.sort(
        key=lambda row: (
            str(row.get("pair_id", "")),
            0 if row.get("label_type") == "medium_match" else 1,
            str(row.get("B_page_id", "")),
        )
    )
    with (output_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    merged = dict(metas[0])
    for key in SUM_KEYS:
        merged[key] = sum(int(meta.get(key, 0)) for meta in metas)
    merged["anchors_available_full_pool"] = max(
        int(meta.get("anchors_available_full_pool", 0)) for meta in metas
    )
    merged.pop("parallel_shard_index", None)
    merged.pop("parallel_num_shards", None)
    merged["parallel_build"] = {
        "enabled": True,
        "worker_processes": workers,
        "worker_source": "BRIDGE_BUILD_WORKERS/SLURM_CPUS_PER_TASK",
        "full_sentence_pool_per_worker": True,
        "omp_threads_per_worker": 1,
    }

    anchors = int(merged.get("anchors_written", 0))
    negatives = int(merged.get("negative_rows", 0))
    expected_negatives = anchors * int(merged.get("negatives_per_anchor", 0))
    if anchors <= 0:
        raise RuntimeError("Parallel build produced zero anchors")
    if int(merged.get("positive_rows", 0)) != anchors:
        raise RuntimeError("Parallel merge positive-row count does not match anchors")
    if negatives != expected_negatives:
        raise RuntimeError(
            f"Parallel merge negative-row count mismatch: {negatives} != {expected_negatives}"
        )

    (output_dir / "metadata.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logs_target = output_dir / "parallel_build_logs"
    logs_target.mkdir(parents=True, exist_ok=True)
    for log in sorted((shard_root / "logs").glob("*.log")):
        shutil.copy2(log, logs_target / log.name)

    print("=== BRIDGE V3 PARALLEL MERGE ===")
    print(f"workers={workers}")
    print(f"anchors_written={anchors}")
    print(f"positive_rows={merged['positive_rows']}")
    print(f"negative_rows={merged['negative_rows']}")
    print(f"output={output_dir}")
    print("PARALLEL_BUILD=PASS")


def main() -> None:
    base_args = sys.argv[1:]
    known = _parse_launcher_args(base_args)
    output_dir = Path(known.output_dir).expanduser().resolve()
    workers = _worker_count(int(known.max_anchors))
    shard_root = output_dir.parent / f".{output_dir.name}_parallel_shards"

    if output_dir.exists():
        if not known.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)

    print("=== BRIDGE V3 PARALLEL BUILD ===")
    print(f"workers={workers}")
    print(f"slurm_cpus_per_task={os.environ.get('SLURM_CPUS_PER_TASK', '<unset>')}")
    print(f"shard_root={shard_root}")
    print("parallel_backend=process_shards")

    failures: list[tuple[int, Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_shard, index, workers, base_args, shard_root): index
            for index in range(workers)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            return_code, log_path = future.result()
            if return_code != 0:
                failures.append((index, log_path))
            else:
                print(f"worker_done={index + 1}/{workers}", flush=True)

    if failures:
        messages = []
        for index, log_path in failures:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
            messages.append(f"--- shard {index} failed: {log_path} ---\n{tail}")
        raise RuntimeError("Parallel Bridge V3 shard failure(s):\n" + "\n".join(messages))

    _merge(shard_root, output_dir, workers)
    if os.environ.get("KEEP_BRIDGE_SHARDS", "0") != "1":
        shutil.rmtree(shard_root)


if __name__ == "__main__":
    main()
