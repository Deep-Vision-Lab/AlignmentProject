#!/usr/bin/env bash
# Resolve an offline Hugging Face cache that contains the requested model.
#
# Usage:
#   bash scripts/train/resolve_hf_cache_home.sh PROJECT_DIR MODEL_NAME

set -euo pipefail

PROJECT_DIR="${1:?PROJECT_DIR is required}"
MODEL_NAME="${2:-aubmindlab/bert-base-arabertv02}"

# An explicit local model snapshot does not need a particular cache home.
if [[ -d "${MODEL_NAME}" && -f "${MODEL_NAME}/config.json" ]]; then
  printf '%s\n' "${HF_HOME:-${PROJECT_DIR}/.hf_cache}"
  exit 0
fi

MODEL_SLUG="models--${MODEL_NAME//\//--}"

snapshot_is_complete() {
  local snapshot="$1"
  [[ -f "${snapshot}/config.json" ]] || return 1

  compgen -G "${snapshot}/model*.safetensors" >/dev/null \
    || compgen -G "${snapshot}/pytorch_model*.bin" >/dev/null \
    || return 1

  [[ -f "${snapshot}/tokenizer.json" \
     || -f "${snapshot}/tokenizer_config.json" \
     || -f "${snapshot}/vocab.txt" \
     || -f "${snapshot}/sentencepiece.bpe.model" ]] \
    || return 1
}

cache_contains_model() {
  local cache_home="$1"
  local layout snapshots_dir snapshot

  [[ -n "${cache_home}" && -d "${cache_home}" ]] || return 1

  for layout in "${cache_home}" "${cache_home}/hub"; do
    snapshots_dir="${layout}/${MODEL_SLUG}/snapshots"
    [[ -d "${snapshots_dir}" ]] || continue

    while IFS= read -r -d '' snapshot; do
      if snapshot_is_complete "${snapshot}"; then
        return 0
      fi
    done < <(find "${snapshots_dir}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
  done

  return 1
}

CANDIDATES=(
  "${HF_HOME:-}"
  "${PROJECT_DIR}/.hf_cache"
  "${PROJECT_DIR}_clone/.hf_cache"
  "${HOME}/.cache/huggingface"
  "${TRANSFORMERS_CACHE:-}"
)

for candidate in "${CANDIDATES[@]}"; do
  [[ -n "${candidate}" ]] || continue
  if cache_contains_model "${candidate}"; then
    printf '%s\n' "${candidate}"
    exit 0
  fi
done

cat >&2 <<EOF
ERROR: Could not find an offline cache for ${MODEL_NAME}.
Checked:
  ${HF_HOME:-<HF_HOME unset>}
  ${PROJECT_DIR}/.hf_cache
  ${PROJECT_DIR}_clone/.hf_cache
  ${HOME}/.cache/huggingface
  ${TRANSFORMERS_CACHE:-<TRANSFORMERS_CACHE unset>}

Copy the known cache into this checkout with:
  mkdir -p "${PROJECT_DIR}/.hf_cache"
  rsync -avh --progress "${PROJECT_DIR}_clone/.hf_cache/" "${PROJECT_DIR}/.hf_cache/"
EOF
exit 1
