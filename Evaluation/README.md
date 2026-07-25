# Evaluation for the optimized Span-DTW model

These scripts reconstruct the visual architecture and Arabic text encoder from the checkpoint's `model_config`. They support:

- `image_model_state_dict` and `model_state_dict` checkpoints;
- the window width and stride used during training;
- Arabic right-to-left patch reversal;
- gated local three-window grouping;
- BiLSTM contextual features;
- ImageNet normalization and synthetic/real preprocessing;
- the trained Arabic span projection, `<SPACE>`, and `<BLANK>` embeddings.

Run all commands from the project root.

## Needleman-Wunsch word evaluation and paired masks

The evaluator performs these steps:

1. localize every word independently in each line with the trained blank-aware image-text Span-DTW path;
2. pool one visual embedding for every localized word;
3. align the two visual word sequences with Needleman-Wunsch;
4. compare predicted index pairs with a transcript-derived reference alignment;
5. draw the same translucent mask color over each predicted word pair in both lines.

The cross-line prediction is image-only. Transcripts provide word boundaries and evaluation labels, not the predicted word similarity matrix.

```bash
python Evaluation/eval_needleman_wunsch_words.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --index 1 \
  --feature local \
  --output Results/Evaluation/NW/pair_1.png
```

Batch evaluation:

```bash
python Evaluation/eval_needleman_wunsch_words.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --batch --start-index 1 --n-samples 200 \
  --output-dir Results/Evaluation/NW
```

Outputs include per-pair PNGs, `samples.csv`, and `summary.json` with:

- NW score and normalized score;
- pair precision, recall, and F1;
- exact-word accuracy;
- mean matched cosine;
- line-1 and line-2 word coverage;
- patch-level NW score.

`Evaluation/eval_image_to_image.py` remains as a compatibility alias for this evaluator.

## Smith-Waterman local image alignment

```bash
python Evaluation/eval_img_align_sw.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --index 1 \
  --feature contextual \
  --output Results/Evaluation/SW/pair_1.png
```

## Image-pair retrieval

```bash
python Evaluation/eval_retrieval.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --n-samples 200
```

## Alignment MAE

```bash
python Evaluation/eval_alignment_mae.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --n-samples 200
```

## Blank-aware Span-DTW heatmap

```bash
python Evaluation/viz_heatmap_dtw.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --index 1 --line 1 \
  --output Results/Evaluation/span_dtw_1.png
```

## Important options

- `--feature local`: raw CNN window representation, recommended for word masks.
- `--feature grouped`: gated three-window local representation.
- `--feature contextual`: BiLSTM contextual sequence representation.
- `--word-gap`: Needleman-Wunsch gap score; it should be negative.
- `--similarity-offset`: value subtracted from each cosine match score. Increasing it makes gaps more competitive.
- `--min-similarity`: do not draw predicted pairs below this cosine value.
- `--dataset-type real`: applies resize, binarization, polarity correction, and normalization compatible with real-data training.
