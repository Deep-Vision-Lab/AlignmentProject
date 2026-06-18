#!/bin/bash
# Arabic alignment ablation launcher.
# Runs controlled experiments over text target space, window size, negative sampling,
# loss type, and multi-scale settings.

set -euo pipefail

# ----------------------------- User settings ---------------------------------
ENV_NAME=${ENV_NAME:-manucripts_align}
MODEL_DIR=${MODEL_DIR:-AlignmentProject_clone}
DATA_DIR=${DATA_DIR:-DataSet/Synthetic_Arabic_100000}
SBATCH_TEMPLATE=${SBATCH_TEMPLATE:-ablation_sbatch_template.sbatch}

# Training defaults. Override from shell if needed, e.g. EPOCHS=100 ./run_ablations.sh
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}
NUM_NEG=${NUM_NEG:-10}
GENERATE_FIGURES=${GENERATE_FIGURES:-False}

# Set DRY_RUN=1 to print sbatch commands without submitting.
DRY_RUN=${DRY_RUN:-0}

mkdir -p out

echo "============================================================"
echo "Arabic Alignment Ablation Launcher"
echo "ENV_NAME:         $ENV_NAME"
echo "MODEL_DIR:        $MODEL_DIR"
echo "DATA_DIR:         $DATA_DIR"
echo "SBATCH_TEMPLATE:  $SBATCH_TEMPLATE"
echo "EPOCHS:           $EPOCHS"
echo "LR:               $LR"
echo "NUM_NEG:          $NUM_NEG"
echo "GENERATE_FIGURES: $GENERATE_FIGURES"
echo "DRY_RUN:          $DRY_RUN"
echo "============================================================"

submit_exp() {
  local job_id="$1"
  local text_embedder="$2"
  local window_size="$3"
  local stride_ratio="$4"
  local negative_mode="$5"
  local multi_scale="$6"
  local ms_windows="$7"
  local loss_type="$8"
  local dtw_norm="$9"
  local blank_token="${10}"

  echo
  echo "Submitting: $job_id"
  echo "  text_embedder=$text_embedder window_size=$window_size stride=$stride_ratio neg=$negative_mode multi_scale=$multi_scale ms=[$ms_windows] loss=$loss_type norm=$dtw_norm blank=$blank_token"

  export_arg="ALL,"
  export_arg+="JOB_ID=${job_id},"
  export_arg+="env=${ENV_NAME},"
  export_arg+="model_dir=${MODEL_DIR},"
  export_arg+="DATA_DIR=${DATA_DIR},"
  export_arg+="TEXT_EMBEDDER_TYPE=${text_embedder},"
  export_arg+="WINDOW_SIZE=${window_size},"
  export_arg+="STRIDE_RATIO=${stride_ratio},"
  export_arg+="NEGATIVE_MODE=${negative_mode},"
  export_arg+="NUM_NEGATIVES=${NUM_NEG},"
  export_arg+="EPOCHS=${EPOCHS},"
  export_arg+="LEARNING_RATE=${LR},"
  export_arg+="MULTI_SCALE_ENABLED=${multi_scale},"
  export_arg+="MULTI_SCALE_WINDOW_SIZES=${ms_windows},"
  export_arg+="CONTRASTIVE_LOSS_TYPE=${loss_type},"
  export_arg+="DTW_COST_NORMALIZATION=${dtw_norm},"
  export_arg+="USE_BLANK_TOKEN=${blank_token},"
  export_arg+="GENERATE_FIGURES=${GENERATE_FIGURES}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo sbatch --job-name="$job_id" --export="$export_arg" "$SBATCH_TEMPLATE"
  else
    sbatch --job-name="$job_id" --export="$export_arg" "$SBATCH_TEMPLATE"
  fi
}

# ----------------------------- Experiments -----------------------------------
# E0: Original-ish baseline. This tells us whether the old setup reproduces the failure.
submit_exp \
  "E0_fasttext_w16_mixed_margin" \
  "fasttext" \
  "16" \
  "0.5" \
  "mixed" \
  "False" \
  "" \
  "margin" \
  "max_len" \
  "False"

# E1: Text target ablation: orthogonal frozen character targets, fair negatives.
submit_exp \
  "E1_orthogonal_w16_lenctrl_margin" \
  "orthogonal_char" \
  "16" \
  "0.5" \
  "length_controlled" \
  "False" \
  "" \
  "margin" \
  "max_len" \
  "False"

# E2: Larger window. Tests whether 16px windows were too ambiguous.
submit_exp \
  "E2_orthogonal_w32_lenctrl_margin" \
  "orthogonal_char" \
  "32" \
  "0.5" \
  "length_controlled" \
  "False" \
  "" \
  "margin" \
  "max_len" \
  "False"

# E3: Arabic dot-confusion negatives. Tests fine character discrimination.
submit_exp \
  "E3_orthogonal_w32_dotconf_margin" \
  "orthogonal_char" \
  "32" \
  "0.5" \
  "dot_confusion" \
  "False" \
  "" \
  "margin" \
  "max_len" \
  "False"

# E4: InfoNCE ranking over DTW costs. Tests whether ranking loss stabilizes pos-vs-neg.
submit_exp \
  "E4_orthogonal_w32_dotconf_infonce" \
  "orthogonal_char" \
  "32" \
  "0.5" \
  "dot_confusion" \
  "False" \
  "" \
  "infonce" \
  "max_len" \
  "False"

# E5: Multi-scale. Tests macro body + micro dot/stroke details.
submit_exp \
  "E5_orthogonal_ms32_16_dotconf_margin" \
  "orthogonal_char" \
  "32" \
  "0.5" \
  "dot_confusion" \
  "True" \
  "32 16" \
  "margin" \
  "max_len" \
  "False"

# E6: Blank token optional. Tests whether ambiguous/background windows stop hurting alignment.
submit_exp \
  "E6_orthogonal_w32_dotconf_blank_margin" \
  "orthogonal_char" \
  "32" \
  "0.5" \
  "dot_confusion" \
  "False" \
  "" \
  "margin" \
  "max_len" \
  "True"

# E7: Random frozen char target control. If it matches orthogonal, FastText is not helping.
submit_exp \
  "E7_random_w32_lenctrl_margin" \
  "random_frozen_char" \
  "32" \
  "0.5" \
  "length_controlled" \
  "False" \
  "" \
  "margin" \
  "max_len" \
  "False"

echo
echo "All ablation jobs submitted."
