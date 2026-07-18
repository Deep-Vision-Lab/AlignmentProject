# Synthetic and real dataset evaluation

This directory evaluates the current AlignmentProject checkpoint on either the synthetic dataset or the real Arabic manifest dataset.

The same evaluator and retrieval metrics are used for both datasets:

- Recall@1, Recall@5, and Recall@10;
- mean reciprocal rank (MRR);
- mean and median rank;
- mean positive alignment cost;
- mean hardest-negative alignment cost;
- percentage of samples whose positive text beats the hardest unrelated text.

The default score is hard D3TW: every image window is consumed while the text position either advances or remains on the same token. Use `SCORE_MODE=mean` for a fast smoke test.

## First: evaluate the synthetic dataset

```bash
cd /home/ahmedmas/BGU-Lab/AlignmentProject

git fetch origin
git checkout dataset_text_repair
git pull origin dataset_text_repair

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate manucripts_align

bash Evaluation/run_synthetic_evaluation.sh \
  Weights/YOUR_JOB/model_latest.pth \
  DataSet/Synthetic_Arabic
```

Replace `Weights/YOUR_JOB/model_latest.pth` with the checkpoint you want to evaluate.

A quick 16-sample smoke test is:

```bash
N_SAMPLES=16 SCORE_MODE=mean \
  bash Evaluation/run_synthetic_evaluation.sh \
  Weights/YOUR_JOB/model_latest.pth \
  DataSet/Synthetic_Arabic
```

A more meaningful D3TW run is:

```bash
N_SAMPLES=100 SCORE_MODE=d3tw \
  bash Evaluation/run_synthetic_evaluation.sh \
  Weights/YOUR_JOB/model_latest.pth \
  DataSet/Synthetic_Arabic
```

By default, evaluation uses:

```text
split=test
sides=first
samples=64
batch_size=8
score_mode=d3tw
device=auto
```

Override them with environment variables:

```bash
SPLIT=valid \
SIDES=both \
N_SAMPLES=128 \
BATCH_SIZE=4 \
NUM_WORKERS=0 \
SCORE_MODE=d3tw \
DEVICE=cuda:0 \
  bash Evaluation/run_synthetic_evaluation.sh \
  Weights/YOUR_JOB/model_latest.pth \
  DataSet/Synthetic_Arabic
```

Use `SIDES=first` initially. `SIDES=both` adds `img2/text2` when the synthetic dataset and training configuration expose paired lines.

## Outputs

Synthetic results are written to:

```text
Results/Evaluation/synthetic_test_d3tw_retrieval.json
Results/Evaluation/synthetic_test_d3tw_retrieval.csv
```

The JSON contains the model/dataset configuration and aggregate metrics. The CSV contains the rank, positive cost, hardest-negative cost, and top-ranked transcript for every query image.

## Evaluate the real dataset afterward

```bash
bash Evaluation/run_real_evaluation.sh \
  Weights/YOUR_JOB/model_latest.pth \
  DataSet/ArabicDataset
```

The real runner uses exactly the same metrics. It validates manifest paths by default and writes results under `Results/Evaluation/real_*`.

## Direct Python interface

```bash
python Evaluation/evaluate_retrieval.py \
  --dataset-type synthetic \
  --data-dir DataSet/Synthetic_Arabic \
  --weights Weights/YOUR_JOB/model_latest.pth \
  --split test \
  --sides first \
  --n-samples 64 \
  --score-mode d3tw
```

Run `python Evaluation/evaluate_retrieval.py --help` for all options.

## Checkpoint requirement

The checkpoint must include both the image-model state and the text-encoder state. The current `model_latest.pth` and `checkpoint_latest.pth` formats contain both. Evaluating an image-only checkpoint against a newly initialized character embedding would produce invalid retrieval results, so the evaluator refuses that by default.
