# Synthetic alignment quantitative report

## Protocol

- Checkpoint: `/home/ahmedmas/BGU-Lab/AlignmentProject/Weights/cnn_bilstm_original_scale_ws32_e50_bs64_gpu2/model_latest.pth`
- Git commit: `e98afa63ffdbbdf54a2dd338d986e21ed0be3043`
- Exact held-out split: 60/20/20, seed **42**
- Test samples evaluated: **1600**
- Random negatives per query: **9**

## Localization

| Metric | Result |
|---|---:|
| Mean pair window IoU | 0.9220 |
| 95% bootstrap CI | [0.9175, 0.9262] |
| Mean pair pixel IoU | 0.9221 |
| Mean pair window F1 | 0.9566 |
| Both lines IoU >= 0.50 | 98.81% |
| Both lines IoU >= 0.75 | 93.00% |
| Both lines IoU >= 0.90 | 78.19% |
| All four boundaries within one stride | 30.69% |
| Mean boundary error | 23.63px |
| Full-line baseline IoU | 0.4815 |
| Random-location oracle-length baseline IoU | 0.4591 |

## Pair discrimination

| Metric | Result |
|---|---:|
| AUROC: true vs random pairs | 0.9918 |
| Retrieval top-1 | 93.31% |
| Mean reciprocal rank | 0.9661 |
| Mean true-minus-best-negative margin | 8.1864 |

The primary metric is mean pair window IoU. The strict IoU >= 0.75 rate requires both line regions to be accurate. Pair AUROC and retrieval accuracy test whether the model identifies the correct partner instead of merely finding generic similar Arabic strokes.
