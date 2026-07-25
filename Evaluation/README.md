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

## Window-level Needleman-Wunsch evaluation

`eval_needleman_wunsch_windows.py` runs Needleman-Wunsch directly on the full
window-to-window visual matrix:

1. encode every window in line 1 and line 2;
2. build an `[S1,S2]` raw cosine matrix;
3. remove broad per-row and per-column cosine bias with the default `mutual-z` score;
4. run global Needleman-Wunsch over those discriminative window match scores;
5. mask matched windows with the same colors in both lines;
6. show raw cosine and the actual NW match-score heatmap side by side;
7. optionally use the trained Span-DTW path only to annotate/evaluate each window token.

The NW prediction is image-only. Transcript tokens do not affect the visual
similarity matrix, normalized score matrix, or NW path.

Why normalization is needed: contextual window embeddings can have a broad
positive cosine background. Running NW directly on raw cosine can reward an
ordinary diagonal even when a brighter displaced ridge is more distinctive.
`mutual-z` rewards a cell only when it is unusually strong for both its line-1
window and its line-2 window.

```bash
python Evaluation/eval_needleman_wunsch_windows.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --index 1 \
  --feature contextual \
  --score-mode mutual-z \
  --score-clip 4.0 \
  --gap -0.35 \
  --similarity-offset 0.0 \
  --min-similarity 0.30 \
  --output Results/Evaluation/NW/window_pair_1_fixed.png
```

Batch evaluation:

```bash
python Evaluation/eval_needleman_wunsch_windows.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --batch --start-index 1 --n-samples 200 \
  --feature contextual \
  --score-mode mutual-z \
  --gap -0.35 \
  --output-dir Results/Evaluation/NW/windows_200
```

Outputs include per-pair PNGs, `samples.csv`, and `summary.json` with:

- NW score and normalized score;
- matched window pairs and gap counts;
- mean matched raw cosine;
- line-1 and line-2 window coverage;
- token agreement for matched windows (annotation metric only).

`Evaluation/eval_image_to_image.py` and the deprecated
`Evaluation/eval_needleman_wunsch_words.py` both delegate to this window-level
evaluator.

## Visual clustering of windows

`viz_window_clusters.py` combines the windows from both lines and clusters their
visual embeddings with deterministic cosine K-means. PCA is used only to draw the
2-D dots.

For every cluster the figure shows:

- one representative window as a small fixed-size corner thumbnail;
- a star at the representative's actual PCA position;
- all other cluster members as dots;
- an arrow between the thumbnail and representative star;
- the majority aligned token for the cluster;
- cluster size and token purity;
- the representative line/window index.

The thumbnail is placed in the corner opposite the representative and is first
downsampled to `--representative-height`, so it cannot scale with output DPI and
hide most cluster dots.

Text does not affect the clustering. The cluster token is assigned after
clustering from the trained blank-aware Span-DTW labels.

```bash
python Evaluation/viz_window_clusters.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --index 1 \
  --feature local \
  --clusters 10 \
  --min-ink 0.01 \
  --representative-height 34 \
  --representative-zoom 1.0 \
  --output Results/Evaluation/Clusters/window_clusters_1_fixed.png \
  --output-dir Results/Evaluation/Clusters
```

The script also writes `window_clusters_<index>.csv` and
`window_clusters_<index>.json` with every member, representative flag, token,
cluster token, purity, ink ratio, and PCA coordinates.

## Feature choices

- `--feature local`: raw CNN window representation; recommended for visual clusters.
- `--feature grouped`: gated three-window local representation.
- `--feature contextual`: BiLSTM contextual representation; recommended for window NW sequence alignment.

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

## Important NW and clustering options

- `--score-mode mutual-z`: recommended NW score; removes row/column cosine bias.
- `--score-mode raw`: reproduces the old raw-cosine behavior for comparison.
- `--score-mode centered`: subtracts row/column means without variance scaling.
- `--score-clip`: protects mutual-z scoring from extremely flat rows or columns.
- `--gap`: NW gap score; it should be negative.
- `--similarity-offset`: subtracted from the chosen NW match score.
- `--min-similarity`: only draw matched window pairs above this raw cosine value.
- `--max-drawn-pairs`: limits connector clutter; it does not alter the NW path.
- `--clusters`: number of visual K-means clusters.
- `--min-ink`: removes nearly empty windows before clustering.
- `--representative-height`: fixed thumbnail height before zoom.
- `--representative-zoom`: final multiplier applied to the fixed-size thumbnail.
