# Synthetic alignment quantitative report

## Protocol

- Checkpoint: `/home/ahmedmas/BGU-Lab/AlignmentProject/Weights/ViT/model_latest.pth`
- Git commit: `57aa5ba7ab881ce85b515295f3adfb952bbf715e`
- Exact held-out split: 60/20/20, seed **42**
- Test samples evaluated: **1600**
- Random negatives per query: **9**

## Localization

| Metric | Result |
|---|---:|
| Mean pair window IoU | 0.7391 |
| 95% bootstrap CI | [0.7310, 0.7469] |
| Mean pair pixel IoU | 0.7403 |
| Mean pair window F1 | 0.8385 |
| Both lines IoU >= 0.50 | 90.56% |
| Both lines IoU >= 0.75 | 43.69% |
| Both lines IoU >= 0.90 | 24.25% |
| All four boundaries within one stride | 12.19% |
| Mean boundary error | 103.08px |
| Full-line baseline IoU | 0.4815 |
| Random-location oracle-length baseline IoU | 0.4591 |

## Pair discrimination

| Metric | Result |
|---|---:|
| AUROC: true vs random pairs | 0.9907 |
| Retrieval top-1 | 92.75% |
| Mean reciprocal rank | 0.9630 |
| Mean true-minus-best-negative margin | 5.9859 |

The primary metric is mean pair window IoU. The strict IoU >= 0.75 rate requires both line regions to be accurate. Pair AUROC and retrieval accuracy test whether the model identifies the correct partner instead of merely finding generic similar Arabic strokes.
