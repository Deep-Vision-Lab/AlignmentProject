# Automatic Experiment Tracker

Every offline dataset build and every full model pipeline creates a human-readable Markdown tracker under:

```text
logs/experiments/<RUN_PREFIX>.md
```

A sibling JSON file is machine state used by `scripts/pipeline/experiment_tracker.py`; normally read the Markdown file.

The tracker records stage/job names, SLURM IDs, dependencies, exact `out/` log paths, checkpoint/result folders, technical status, a separate scientific verdict with reason, and quantitative metrics. A job can therefore be `✅ COMPLETED` technically but `FAIL` scientifically.

Canonical real quantitative/final defaults are structural AUC >= 0.65, positive mean path steps >= 8, and positive-negative step gap >= 2. Bridge evaluation defaults to structural AUC >= 0.65 with positive path steps greater than negatives. Qualitative stages are marked `VISUAL REVIEW` because inventing a numeric PASS/FAIL for images would be misleading.

Dataset submission:

```bash
bash scripts/slurm/submit_bridge_v2_dataset.sh
```

Model submission:

```bash
RUN_PREFIX=my_run bash scripts/slurm/submit_full_research_pipeline.sh
```

Read the tracker:

```bash
cat logs/experiments/my_run.md
```

If a hard SLURM kill prevented automatic update:

```bash
python scripts/pipeline/experiment_tracker.py refresh \
  --tracker logs/experiments/my_run.json
```

Quantitative summaries are rebuilt from the actual result files (`samples.csv`, `bridge_summary.json`, and `final_summary.json`) rather than copied from console text.
