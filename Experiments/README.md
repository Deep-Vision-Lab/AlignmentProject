# Image-Based Alignment of Arabic Manuscripts: 4-Phase Experiment Roadmap

This directory contains the testing schedule and scripts required to prove the effectiveness of the Contrastive Soft-DTW architecture for manuscript alignment.

## Phase 1: Ablation Studies (Proving Your Choices)
*Located in `Phase1_Ablation/`*
To prove each architectural choice was necessary, we run the following ablations:
- **Experiment 1 (The Baseline):** CNN + Soft-DTW (No Bi-LSTM, Single-Scale)
- **Experiment 2 (The Sequence Test):** CNN + Bi-LSTM + Soft-DTW (Single-Scale)
- **Experiment 3 (The Multi-Scale Test):** CNN + Bi-LSTM + Soft-D3TW (Multi-Scale Windowing, w=8, w=16)
- **Experiment 4 (The Final Model):** Full architecture (CNN + Bi-LSTM + Transformer + Multi-Scale D3TW + Dropout)

## Phase 2: Quantitative Inference (The Hard Numbers)
*Located in `Phase2_Quantitative/`*
Metrics for Image Retrieval and Alignment:
- **Recall@1, Recall@5, Recall@10:** Percentage of times the correct target sequence is within the top-K closest D3TW distances.
- **Mean Reciprocal Rank (MRR):** Average of the reciprocal ranks of the correct alignments.
- **Average Diagonal Deviation (ADD):** Deviation of the predicted alignment path from the perfect chronological diagonal.

## Phase 3: Qualitative Visualization (The Eye Test)
*Located in `Phase3_Qualitative/`*
- **Distance Matrix Heatmap:** Plotting the N×M distance matrix showing the clear, dark "staircase" representing the alignment path.
- **Visual Bounding Box Mapping:** Physically drawing lines connecting the Arabic visual ink patches to corresponding English ones.

## Phase 4: Stress Testing (Finding the Limits)
*Located in `Phase4_StressTesting/`*
- **Length Discrepancy:** Testing on pairs where the sentence length ratio is highly unbalanced (e.g., 50 patches vs 150 patches).
- **Degraded Ink:** Manually adding noise/artificial fading to the test images to see how Recall@1 and path deviation are affected, proving robustness.
