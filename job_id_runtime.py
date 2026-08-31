"""Resolve one stable output job id for training runs.

Priority:
1. explicit JOB_ID environment override;
2. SLURM_JOB_ID for batch jobs;
3. the normal local experiment-name mode suffix.
"""
from __future__ import annotations

import os


def resolve_training_job_id(experiment_name: str, *, finetune: bool) -> str:
    explicit = os.environ.get("JOB_ID", "").strip()
    if explicit:
        return explicit

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if slurm_job_id:
        return slurm_job_id

    mode = "finetune" if finetune else "scratch"
    return f"{experiment_name}_{mode}"
