# Automatic Experiment Tracker

Every offline dataset build and full model pipeline creates `logs/experiments/<RUN_PREFIX>.md`, with a sibling JSON machine-state file. The Markdown tracker records stage/job names, SLURM IDs, dependencies, exact logs, checkpoint/result folders, technical status, scientific verdict/reason, and quantitative metrics.

A job can be `✅ COMPLETED` technically but `FAIL` scientifically. Canonical real quantitative/final defaults are structural AUC >= 0.65, positive mean path steps >= 8, and positive-negative step gap >= 2. Bridge evaluation defaults to structural AUC >= 0.65 with positive path steps greater than negatives. Qualitative stages are marked `VISUAL REVIEW`.

Dataset: `bash scripts/slurm/submit_bridge_v2_dataset.sh`

Model: `RUN_PREFIX=my_run bash scripts/slurm/submit_full_research_pipeline.sh`

Read: `cat logs/experiments/my_run.md`

Recovery after a hard SLURM kill:

```bash
python scripts/pipeline/experiment_tracker.py refresh --tracker logs/experiments/my_run.json
```

Quantitative summaries are rebuilt from actual `samples.csv`, `bridge_summary.json`, and `final_summary.json` files.
