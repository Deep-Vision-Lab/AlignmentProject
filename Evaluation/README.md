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
3. transform that matrix according to `--score-mode`;
4. run global Needleman-Wunsch over the selected score matrix;
5. keep the prediction and traceback at window resolution;
6. merge nearby monotonic matches into phrase/sentence-like visualization groups;
7. draw one continuous mask per merged group with the same color in both lines;
8. show one heatmap: the exact score matrix used by Needleman-Wunsch;
9. write every score value in its heatmap cell and overlay the full traceback;
10. optionally use the trained Span-DTW path only to annotate/evaluate each window token.

The visualization grouping does not modify Needleman-Wunsch. Matches above
`--merge-min-similarity` act as anchors. Weak matches and short gap transitions
may remain inside one group when the number of intervening traceback steps is at
most `--merge-gap-tolerance-windows` and the next anchor advances by no more
than `--merge-max-jump-windows` in either line. The continuous mask spans all
windows between the first and last anchor, including the tolerated interruption.
A new color starts only after a long bridge, a large jump, or a non-monotonic move.
No arrows or connectors are drawn between the lines.

The NW prediction is image-only. Transcript tokens do not affect the visual
similarity matrix, selected score matrix, NW path, or phrase grouping.

### Score modes

The score mode selects which cell value Needleman-Wunsch uses for a diagonal
window match. `--similarity-offset` is subtracted afterward.

- `--score-mode raw`: use raw cosine similarity directly. It is easy to interpret,
  but a broad positive cosine background can pull the global path toward a plain
  diagonal.
- `--score-mode centered`: subtract the row and column baselines and average the
  two centered values. It rewards a cell that is above the usual similarity of
  either participating window, without variance normalization.
- `--score-mode mutual-z`: subtract row/column baselines, normalize both by their
  row/column variation, and average the two standardized values. A positive score
  means the pair is unusually strong for both windows; this is the recommended
  default for the contextual encoder. `--score-clip` limits extreme standardized
  values from very flat rows or columns.

The single heatmap always displays the matrix selected by `--score-mode`, because
that is the matrix actually optimized by Needleman-Wunsch. Raw cosine values are
still retained for reporting `mean_matched_cosine` and for phrase-anchor filtering
with `--merge-min-similarity`.

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
  --merge-gap-tolerance-windows 2 \
  --merge-max-jump-windows 4 \
  --merge-min-similarity 0.30 \
  --max-drawn-groups 0 \
  --heatmap-value-decimals 2 \
  --heatmap-annotation-fontsize 5 \
  --output Results/Evaluation/NW/window_pair_1_phrase_groups.png
```

Every heatmap cell is annotated by default. For very large batch figures, disable
cell labels with `--no-annotate-heatmap-values`.

Batch evaluation:

```bash
python Evaluation/eval_needleman_wunsch_windows.py \
  --weights Weights/<run>/model_latest.pth \
  --data-dir DataSet/Synthetic_Arabic \
  --batch --start-index 1 --n-samples 200 \
  --feature contextual \
  --score-mode mutual-z \
  --gap -0.35 \
  --merge-gap-tolerance-windows 2 \
  --merge-max-jump-windows 4 \
  --merge-min-similarity 0.30 \
  --no-annotate-heatmap-values \
  --output-dir Results/Evaluation/NW/windows_200
```

Outputs include per-pair PNGs, `samples.csv`, and `summary.json` with:

- NW score and normalized score;
- matched window pairs and gap counts;
- strict consecutive-block metrics for low-level diagnostics;
- phrase-group count, anchor count, span length, and mean group cosine;
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

- `--score-mode mutual-z`: recommended NW score; row/column standardized mutual distinctiveness.
- `--score-mode raw`: raw cosine similarity.
- `--score-mode centered`: row/column centered cosine without variance scaling.
- `--score-clip`: protects mutual-z scoring from extremely flat rows or columns.
- `--gap`: NW gap score; it should be negative.
- `--similarity-offset`: subtracted from each selected score-mode cell before NW compares transitions.
- `--merge-gap-tolerance-windows`: tolerated weak/gap traceback steps between phrase anchors.
- `--merge-max-jump-windows`: largest anchor advance allowed in either line before starting a new color.
- `--merge-min-similarity`: minimum raw cosine for a match to act as a phrase anchor.
- `--max-drawn-groups`: limits colored phrase groups; `0` draws all and does not alter the NW path.
- `--heatmap-value-decimals`: decimal precision shown in every heatmap cell.
- `--heatmap-annotation-fontsize`: font size for cell values.
- `--no-annotate-heatmap-values`: omit cell values for faster/smaller batch figures.
- `--clusters`: number of visual K-means clusters.
- `--min-ink`: removes nearly empty windows before clustering.
- `--representative-height`: fixed thumbnail height before zoom.
- `--representative-zoom`: final multiplier applied to the fixed-size thumbnail.
