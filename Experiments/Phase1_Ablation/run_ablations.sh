#!/bin/bash
# Phase 1: Ablation Studies (Proving Choices)
# This script runs the ablation experiments to prove every architectural part is necessary.
# NOTE: You may need to adapt `Parameters.py` reading to parse these environment variables.

echo "Running Phase 1 Ablation Studies"


# Experiment 1 (The Sequence Test): ResNet34 + Bi-LSTM (No multi-scale)
echo "Submitting Experiment 1: Sequence Test"
sbatch \
  --job-name=UniScale \
  --export=ALL,MULTI_SCALE_ENABLED=False,JOB_ID=UniScale,env=manucripts_align,model_dir=AlignmentProject_clone \
  ablation_sbatch_template.sbatch

# Experiment 2 (The Multi-Scale Test): ResNet34 + Bi-LSTM + Multi-Scale Windowing
echo "Submitting Experiment 2: Multi-Scale Test"
sbatch \
  --job-name=MultiScale \
  --export=ALL,MULTI_SCALE_ENABLED=True,JOB_ID=MultiScale,env=manucripts_align,model_dir=AlignmentProject_clone \
  ablation_sbatch_template.sbatch


# ============================================================================
# Fine-tuning runs (uncomment / adapt as needed)
# ============================================================================
# Setting FINETUNE=True makes train.py use finetune_data_dir, finetune_learning_rate,
# and finetune_epochs from Parameters.py. Override DATA_DIR to point at a different
# dataset. PRETRAINED_WEIGHTS is the .pth from the source run.

# Experiment 3 (Finetune UniScale on the secondary dataset)
# echo "Submitting Experiment 3: Finetune UniScale"
# sbatch \
#   --job-name=UniScale_FT \
#   --export=ALL,MULTI_SCALE_ENABLED=False,JOB_ID=UniScale_FT,env=manucripts_align,model_dir=AlignmentProject_clone,FINETUNE=True,PRETRAINED_WEIGHTS=Weights/UniScale/model_latest.pth \
#   ablation_sbatch_template.sbatch

# Experiment 4 (Finetune MultiScale on the secondary dataset)
# echo "Submitting Experiment 4: Finetune MultiScale"
# sbatch \
#   --job-name=MultiScale_FT \
#   --export=ALL,MULTI_SCALE_ENABLED=True,JOB_ID=MultiScale_FT,env=manucripts_align,model_dir=AlignmentProject_clone,FINETUNE=True,PRETRAINED_WEIGHTS=Weights/MultiScale/model_latest.pth \
#   ablation_sbatch_template.sbatch

# Resume a crashed finetune run (no PRETRAINED_WEIGHTS — RESUME restores everything)
# echo "Resuming UniScale_FT"
# sbatch \
#   --job-name=UniScale_FT \
#   --export=ALL,MULTI_SCALE_ENABLED=False,JOB_ID=UniScale_FT,env=manucripts_align,model_dir=AlignmentProject_clone,FINETUNE=True,RESUME=Weights/UniScale_FT/checkpoint_latest.pth \
#   ablation_sbatch_template.sbatch


echo "Phase 1 Ablation Studies Submitted!"
