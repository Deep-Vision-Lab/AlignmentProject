# Real quantitative alignment evaluation

- Checkpoint: `/home/ahmedmas/BGU-Lab/AlignmentProject/Weights/cnn_bilstm_real_aug/model_best.pth`
- Backend: `cnn_bilstm`
- Split: `test`
- Labels: `high_match,medium_match`

## Automatic crop localization

- Examples: 240
- Mean interval IoU: 0.8818
- Success@IoU 0.50: 0.9958
- Mean boundary MAE: 22.2998 px

## Retrieval and pair discrimination

- Queries: 80
- Recall@1: 0.3875
- Recall@5: 0.6500
- MRR: 0.5205
- Pair AUROC: 0.7796
- Pair average precision: 0.2131
- Thresholded F1: 0.3577

## Interpretation

Crop localization has exact targets but uses a crop from the same real line. Retrieval/discrimination tests real-to-real matching. Sparse intervals, when provided, are the strongest direct localization measure.
