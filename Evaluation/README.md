# Evaluation Scripts

All scripts must be run from the **project root** (one level above this folder).

```
AlignmentProject/
├── Evaluation/
│   ├── _eval_utils.py              ← shared model/data utilities
│   ├── eval_retrieval.py           ← Recall@K, MRR, ACG
│   ├── eval_alignment_mae.py       ← Local alignment MAE
│   ├── viz_heatmap_dtw.py          ← Similarity heatmap + DTW path
│   ├── viz_image_to_text_mapping.py← Connector-line mapping figure
│   └── viz_distance_histogram.py   ← Pos vs neg cost distributions
```

---

## 1 · Global Retrieval  (`eval_retrieval.py`)

**Metrics:** Recall@1, Recall@5, Recall@10, MRR, ACG

```bash
python Evaluation/eval_retrieval.py \
    --weights   model_epoch_80.pth \
    --data-dir  DataSet/Synthetic_Arabic \
    --n-samples 500
```

| Argument | Default | Description |
|---|---|---|
| `--weights` | `model_epoch_80.pth` | Trained model checkpoint |
| `--data-dir` | `DataSet/Synthetic_Arabic` | Dataset root |
| `--n-samples` | `500` | Number of test pairs to evaluate |
| `--gap` | `-0.5` | Hard-DTW gap penalty |

---

## 2 · Local Alignment MAE  (`eval_alignment_mae.py`)

**Metric:** Mean Absolute Error of per-character frame prediction (in patch units).

```bash
python Evaluation/eval_alignment_mae.py \
    --weights   model_epoch_80.pth \
    --data-dir  DataSet/Synthetic_Arabic \
    --n-samples 200
```

---

## 3 · Heatmap + DTW Path  (`viz_heatmap_dtw.py`)

Produces a figure with: the original image, the cosine-similarity heatmap
(text chars × image patches), and the red DTW alignment path overlaid on it.

```bash
python Evaluation/viz_heatmap_dtw.py \
    --weights   model_epoch_80.pth \
    --image     DataSet/Synthetic_Arabic/images/img1_1.png \
    --output    Results/Evaluation/heatmap_dtw.png
```

The `--text` / `--text-file` arguments are optional; the script auto-infers
the text file from the image path if omitted.

---

## 4 · Image-to-Text Mapping  (`viz_image_to_text_mapping.py`)

The "show-off" graphic: coloured connector lines linking each Arabic character
to the image region the DTW path assigned it.

```bash
python Evaluation/viz_image_to_text_mapping.py \
    --weights   model_epoch_80.pth \
    --image     DataSet/Synthetic_Arabic/images/img1_1.png \
    --output    Results/Evaluation/image_to_text_map.png
```

---

## 5 · Distance Distribution Histograms  (`viz_distance_histogram.py`)

Two overlapping histograms of DTW cost for correct (blue) vs mismatched
(red) pairs. Separated curves prove the embedding space is well-structured.

```bash
python Evaluation/viz_distance_histogram.py \
    --weights        model_epoch_80.pth \
    --data-dir       DataSet/Synthetic_Arabic \
    --n-samples      500 \
    --negs-per-sample 3 \
    --output         Results/Evaluation/distance_histogram.png
```
