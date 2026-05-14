#!/bin/bash
# Phase 1: Ablation Studies (Proving Choices)
# This script runs the ablation experiments to prove every architectural part is necessary.
# NOTE: You may need to adapt `Parameters.py` reading to parse these environment variables.

echo "Running Phase 1 Ablation Studies"


# Experiment 1 (The Sequence Test): ResNet34 + Bi-LSTM (No multi-scale)
echo "Submitting Experiment 1: Sequence Test"
sbatch \
  --job-name=UniScale \
  --export=ALL,MULTI_SCALE_ENABLED=False,JOB_ID=UniScale,env=manucripts_align,model_dir=AlignmentProject \
  ablation_sbatch_template.sbatch

# Experiment 2 (The Multi-Scale Test): ResNet34 + Bi-LSTM + Multi-Scale Windowing
echo "Submitting Experiment 2: Multi-Scale Test"
sbatch \
  --job-name=MultiScale \
  --export=ALL,MULTI_SCALE_ENABLED=True,JOB_ID=MultiScale,env=manucripts_align,model_dir=AlignmentProject \
  ablation_sbatch_template.sbatch


echo "Phase 1 Ablation Studies Submitted!"
