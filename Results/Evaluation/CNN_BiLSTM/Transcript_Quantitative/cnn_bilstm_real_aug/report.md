# Transcript-supervised real-data evaluation

- Checkpoint: `/home/ahmedmas/BGU-Lab/AlignmentProject/Weights/cnn_bilstm_real_aug/model_best.pth`
- Backend: `cnn_bilstm`
- Transcript key: `text_original_path`
- Positive reference: transcript min-coverage >= 0.5 with at least 2 shared words
- Negative reference: transcript min-coverage <= 0.1

## Pair classification on the held-out test split

- Precision: 0.8966
- Recall: 0.4127
- F1 / pair-set Dice: 0.5652
- Pair-set IoU: 0.3939
- AUROC: 0.7585
- Average precision: 0.5911

## Transcript-defined retrieval

- Queries: 63
- Recall@1: 0.5238
- Recall@5: 0.7143
- Recall@10: 0.8413
- MRR: 0.6229
- mAP: 0.6229

## Word-correspondence proxy

- Word metrics were unavailable for this checkpoint or sample selection.

## Interpretation limitation

These metrics use transcripts as supervision. Pair and retrieval metrics are valid image-pair matching metrics. Word Dice/IoU are transcript-supervised proxy metrics because word-to-window locations are forced-aligned by the checkpoint. They are not pixel-level mask Dice or spatial interval IoU.
