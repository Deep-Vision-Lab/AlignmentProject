#!/bin/bash
# Phase 1: Ablation Studies (Proving Choices)
# This script runs the ablation experiments to prove every architectural part is necessary.
# NOTE: You may need to adapt `Parameters.py` reading to parse these environment variables.

echo "Running Phase 1 Ablation Studies"

# Experiment 1 (The Baseline): ResNet34 + basic Soft-DTW (No Bi-LSTM, no multi-scale)
echo "Submitting Experiment 1: Baseline"
sbatch \
  --job-name=ablation_baseline \
  --export=ALL,USE_BILSTM=False,MULTI_SCALE_ENABLED=False,JOB_ID=ablation_baseline,env=manucripts_align,model_dir=AlignmentProject \
  ablation_sbatch_template.sbatch

# Experiment 2 (The Sequence Test): ResNet34 + Bi-LSTM (No multi-scale)
echo "Submitting Experiment 2: Sequence Test"
sbatch \
  --job-name=ablation_sequence \
  --export=ALL,USE_BILSTM=True,MULTI_SCALE_ENABLED=False,JOB_ID=ablation_sequence,env=manucripts_align,model_dir=AlignmentProject \
  ablation_sbatch_template.sbatch

# Experiment 3 (The Multi-Scale Test): ResNet34 + Bi-LSTM + Multi-Scale Windowing
echo "Submitting Experiment 3: Multi-Scale Test"
sbatch \
  --job-name=ablation_multiscale \
  --export=ALL,USE_BILSTM=True,MULTI_SCALE_ENABLED=True,JOB_ID=ablation_multiscale,env=manucripts_align,model_dir=AlignmentProject \
  ablation_sbatch_template.sbatch

# Experiment 4 (Your Final Model): Full Architecture
echo "Submitting Experiment 4: Final Model"
sbatch \
  --job-name=ablation_final \
  --export=ALL,USE_BILSTM=True,MULTI_SCALE_ENABLED=True,JOB_ID=ablation_final,env=manucripts_align,model_dir=AlignmentProject \
  ablation_sbatch_template.sbatch

echo "Phase 1 Ablation Studies Submitted!"
