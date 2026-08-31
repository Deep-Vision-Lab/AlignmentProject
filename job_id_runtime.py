"""Resolve one stable output folder name for training runs.

Priority:
1. explicit JOB_NAME environment override;
2. SLURM_JOB_NAME for batch jobs;
3. legacy JOB_ID environment override;
4. the normal local experiment-name mode suffix.
"""
from __future__ import annotations

import os


def resolve_training_job_id(experiment_name: str, *, finetune: bool) -> str:
    explicit_name = os.environ.get("JOB_NAME", "").strip()
    if explicit_name:
        return explicit_name

    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "").strip()
    if slurm_job_name:
        return slurm_job_name

    legacy_job_id = os.environ.get("JOB_ID", "").strip()
    if legacy_job_id:
        return legacy_job_id

    mode = "finetune" if finetune else "scratch"
    return f"{experiment_name}_{mode}"
