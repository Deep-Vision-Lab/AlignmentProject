# Paper Figures — Weakly Supervised Arabic Transcript-Image Alignment

Scripts to generate all publication-quality figures for the research paper.
Each script saves outputs as both high-resolution PNG (300 DPI) and PDF.

---

## Directory Structure

```
paper_figures/
├── outputs/                   ← all generated figures go here
├── utils/
│   ├── model_loading.py       ← load model, text embedder, dataset samples
│   ├── similarity.py          ← compute embeddings and similarity matrices
│   ├── alignment.py           ← hard-DTW path extraction (viz only)
│   ├── plotting.py            ← matplotlib helpers, paper style, save helpers
│   └── negatives.py           ← hard-negative generation (mirrors newDataLoader)
├── fig01_architecture.py      ← conceptual architecture diagram
├── fig02_sliding_windows.py   ← sliding-window decomposition figure
├── fig03_similarity_before_after.py  ← heatmap before vs after training
├── fig04_d3tw_path_overlay.py ← similarity heatmap + D3TW path
├── fig05_pos_neg_costs.py     ← positive vs negative DTW cost distributions
├── fig06_ablation_plot.py     ← ablation bar chart (requires CSV input)
├── fig07_cnn_vs_bilstm.py     ← CNN-only vs CNN+BiLSTM heatmap comparison
├── fig08_hard_negative_case.py ← case study with cost-ranked transcripts
├── fig09_embedding_space.py   ← t-SNE / UMAP / PCA embedding visualization
├── fig10_image_only_alignment.py ← image-image alignment (no text)
├── fig11_success_failure_cases.py ← success / partial / failure grid
└── generate_all_figures.py    ← master script (runs all figures)
```

---

## Dependencies

Core (required):
```bash
pip install torch torchvision matplotlib numpy pillow
```

Optional (for specific figures):
```bash
pip install scikit-learn   # fig09 t-SNE / PCA
pip install umap-learn     # fig09 UMAP (faster, cleaner than t-SNE)
pip install arabic-reshaper python-bidi  # Arabic text labels in plots
```

All scripts fall back gracefully when optional packages are missing.

---

## Common Arguments

| Argument | Description | Default |
|---|---|---|
| `--checkpoint` | Trained model weights (.pth) | required for figs 3–11 |
| `--data_dir` | Dataset root directory | `DataSet/Synthetic_Arabic_100000` |
| `--output_dir` | Where to save figures | `paper_figures/outputs` |
| `--device` | `cuda` or `cpu` | auto-detected |
| `--sample_idx` | 0-based sample index | `0` |
| `--num_samples` | Number of samples (figs 5, 9) | `20` |

Run scripts from the **project root** (`AlignmentProject_clone/`).

---

## Individual Figure Commands

### fig01 — Architecture Diagram
```bash
python paper_figures/fig01_architecture.py \
    --output_dir paper_figures/outputs
```
No checkpoint or data required.
Output: `fig01_architecture.png` / `.pdf`

---

### fig02 — Sliding Window Decomposition
```bash
python paper_figures/fig02_sliding_windows.py \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_idx 0 \
    --output_dir paper_figures/outputs
```
Output: `fig02_sliding_windows_sample_0.png` / `.pdf`

---

### fig03 — Similarity Matrix Before vs After Training
```bash
python paper_figures/fig03_similarity_before_after.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_idx 0 \
    --output_dir paper_figures/outputs \
    --device cuda
```
Shows two side-by-side heatmaps on the same colour scale.
Output: `fig03_similarity_before_after_sample_0.png` / `.pdf`

---

### fig04 — D3TW Alignment Path Overlay
```bash
python paper_figures/fig04_d3tw_path_overlay.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_idx 0 \
    --output_dir paper_figures/outputs
```
Also saves a JSON mapping: `fig04_alignment_mapping_sample_0.json`
Output: `fig04_d3tw_path_overlay_sample_0.png` / `.pdf`

---

### fig05 — Positive vs Negative Cost Distribution
```bash
python paper_figures/fig05_pos_neg_costs.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --num_samples 50 \
    --num_negatives 10 \
    --output_dir paper_figures/outputs
```
Output: `fig05_pos_neg_cost_distribution.png` / `.pdf` + `fig05_pos_neg_cost_stats.json`

---

### fig06 — Ablation Study Bar Chart
```bash
python paper_figures/fig06_ablation_plot.py \
    --results_csv path/to/ablation_results.csv \
    --metric "Top-1 Retrieval" \
    --output_dir paper_figures/outputs
```
**Requires** an ablation CSV or JSON file. Example CSV format:
```csv
method,metric,value
CNN only,Top-1 Retrieval,0.42
CNN + BiLSTM,Top-1 Retrieval,0.55
CNN + BiLSTM + D3TW,Top-1 Retrieval,0.63
CNN + BiLSTM + D3TW + Hard Negatives,Top-1 Retrieval,0.71
CNN + BiLSTM + D3TW + FastText,Top-1 Retrieval,0.75
```
Output: `fig06_ablation_Top-1_Retrieval.png` / `.pdf`

---

### fig07 — CNN-only vs CNN+BiLSTM Heatmap
```bash
# With trained CNN-only checkpoint:
python paper_figures/fig07_cnn_vs_bilstm.py \
    --checkpoint_cnn_only Weights/cnn_only/model_latest.pth \
    --checkpoint_bilstm Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_idx 0

# Without CNN-only checkpoint (uses random init):
python paper_figures/fig07_cnn_vs_bilstm.py \
    --checkpoint_bilstm Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000
```
> **Note:** If `--checkpoint_cnn_only` is omitted, panel (a) shows a randomly
> initialised CNN-only model — useful for illustrating the effect of BiLSTM context
> even without a separately trained CNN-only checkpoint.

Output: `fig07_cnn_vs_bilstm_sample_0.png` / `.pdf`

---

### fig08 — Hard Negative Case Study
```bash
python paper_figures/fig08_hard_negative_case.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_idx 0 \
    --num_negatives 5 \
    --output_dir paper_figures/outputs
```
Output: `fig08_hard_negative_case_sample_0.png` / `.pdf` + `.json`

---

### fig09 — Embedding Space (t-SNE / UMAP / PCA)
```bash
python paper_figures/fig09_embedding_space.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --num_samples 20 \
    --method tsne \
    --output_dir paper_figures/outputs
```
Requires `scikit-learn` (t-SNE / PCA) or `umap-learn` (UMAP).
Output: `fig09_embedding_space_before_after.png` / `.pdf`

---

### fig10 — Image-Only Alignment
```bash
python paper_figures/fig10_image_only_alignment.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_idx_a 0 \
    --sample_idx_b 1 \
    --output_dir paper_figures/outputs
```
**No text embeddings used** — this is the evaluation-phase figure.
Output: `fig10_image_only_alignment_0_1.png` / `.pdf`

---

### fig11 — Success / Partial / Failure Cases
```bash
python paper_figures/fig11_success_failure_cases.py \
    --checkpoint Weights/localRun/model_latest.pth \
    --data_dir DataSet/Synthetic_Arabic_100000 \
    --sample_indices 0 1 2 \
    --case_labels success partial failure \
    --notes "Good monotonic alignment" "Minor ambiguity near dots" "Degraded image" \
    --output_dir paper_figures/outputs
```
Output: `fig11_success_failure_cases.png` / `.pdf`

---

## Master Script (All Figures)

```bash
python paper_figures/generate_all_figures.py \
    --checkpoint  Weights/localRun/model_latest.pth \
    --data_dir    DataSet/Synthetic_Arabic_100000 \
    --output_dir  paper_figures/outputs \
    --device      cuda \
    --sample_idx  0 \
    --num_samples 20
```

With optional extras:
```bash
python paper_figures/generate_all_figures.py \
    --checkpoint        Weights/UniScale/model_latest.pth \
    --data_dir          DataSet/Synthetic_Arabic_100000 \
    --output_dir        paper_figures/outputs \
    --device            cuda \
    --sample_idx        0 \
    --sample_idx_b      1 \
    --sample_indices    0 1 2 \
    --num_samples       50 \
    --num_negatives     10 \
    --embedding_method  tsne \
    --ablation_csv      experiments/ablation_results.csv \
    --checkpoint_cnn_only  Weights/cnn_only/model_latest.pth
```

---

## Important Notes

1. **Text branch is frozen and used only for training-style figures** (fig03, fig04,
   fig05, fig07, fig08, fig11). It provides character embeddings to compute
   text-image similarity matrices.

2. **fig10 uses only the visual encoder** — no text embeddings at all. This
   demonstrates the final evaluation setting where alignment is purely image-based.

3. **Arabic windows are right-to-left**: the model sets `use_flip=True` for
   Arabic, so window index 0 corresponds to the rightmost patch (first character).
   All scripts respect this by loading `lang` from `Parameters.py`.

4. **All figures respect Parameters.py**: window_size, stride_ratio, vector_size,
   bilstm_hidden_dim, etc. are read from the same config used during training.

5. **fig09 requires scikit-learn or umap-learn**:
   - `pip install scikit-learn` for t-SNE / PCA
   - `pip install umap-learn` for UMAP

6. **fig06 requires an external ablation CSV** that you produce by running your
   ablation experiments and recording metrics.

---

## Output Files Reference

| Figure | PNG | PDF | Extra |
|--------|-----|-----|-------|
| fig01 | `fig01_architecture.png` | `.pdf` | — |
| fig02 | `fig02_sliding_windows_sample_{N}.png` | `.pdf` | — |
| fig03 | `fig03_similarity_before_after_sample_{N}.png` | `.pdf` | — |
| fig04 | `fig04_d3tw_path_overlay_sample_{N}.png` | `.pdf` | `fig04_alignment_mapping_sample_{N}.json` |
| fig05 | `fig05_pos_neg_cost_distribution.png` | `.pdf` | `fig05_pos_neg_cost_stats.json` |
| fig06 | `fig06_ablation_{metric}.png` | `.pdf` | — |
| fig07 | `fig07_cnn_vs_bilstm_sample_{N}.png` | `.pdf` | — |
| fig08 | `fig08_hard_negative_case_sample_{N}.png` | `.pdf` | `fig08_hard_negative_case_sample_{N}.json` |
| fig09 | `fig09_embedding_space_before_after.png` | `.pdf` | — |
| fig10 | `fig10_image_only_alignment_{A}_{B}.png` | `.pdf` | — |
| fig11 | `fig11_success_failure_cases.png` | `.pdf` | — |
