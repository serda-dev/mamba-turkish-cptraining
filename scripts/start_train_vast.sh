#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/mamba-cpt-tr}"
CONFIG_PATH="${CONFIG_PATH:-configs/train.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
AUTO_RESUME="${AUTO_RESUME:-1}"
HF_DATASET_ID="${HF_DATASET_ID:-}"

cd "${PROJECT_DIR}"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf-cache/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p output dataset "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf_dataset_id)
      if [[ $# -lt 2 ]]; then
        echo "Error: --hf_dataset_id requires a value" >&2
        exit 1
      fi
      HF_DATASET_ID="$2"
      shift 2
      ;;
    --hf_dataset_id=*)
      HF_DATASET_ID="${1#*=}"
      shift
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

RESOLVED_CONFIG_PATH="${CONFIG_PATH}"
TEMP_CONFIG_PATH=""

if [[ -n "${HF_DATASET_ID}" ]]; then
  TEMP_CONFIG_PATH="$(mktemp /tmp/mamba-cpt-tr.XXXXXX.yaml)"
  export HF_DATASET_ID TEMP_CONFIG_PATH CONFIG_PATH
  python - <<'PY'
from pathlib import Path
import os
import yaml

config_path = Path(os.environ["CONFIG_PATH"])
tmp_config_path = Path(os.environ["TEMP_CONFIG_PATH"])
hf_dataset_id = os.environ["HF_DATASET_ID"]

config = yaml.safe_load(config_path.read_text())
config.setdefault("data", {})["hf_dataset_id"] = hf_dataset_id
tmp_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
  RESOLVED_CONFIG_PATH="${TEMP_CONFIG_PATH}"
  trap 'rm -f "${TEMP_CONFIG_PATH}"' EXIT
  echo "==> Overriding HF dataset: ${HF_DATASET_ID}"
fi

resolve_latest_checkpoint() {
  if [[ ! -d "${OUTPUT_DIR}/checkpoints" ]]; then
    return 1
  fi

  local latest_checkpoint
  latest_checkpoint="$(
    find "${OUTPUT_DIR}/checkpoints" -maxdepth 1 -mindepth 1 -type d \
      \( -name 'final' -o -name 'step_*' \) \
      -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "${latest_checkpoint}" ]]; then
    echo "${latest_checkpoint}"
    return 0
  fi

  return 1
}

TRAIN_CMD=(python train.py --config "${RESOLVED_CONFIG_PATH}")

if [[ "${AUTO_RESUME}" == "1" ]]; then
  if latest_checkpoint="$(resolve_latest_checkpoint)"; then
    echo "==> Auto-resume enabled"
    echo "==> Resuming from ${latest_checkpoint}"
    TRAIN_CMD+=(--resume "${latest_checkpoint}")
  else
    echo "==> No checkpoint found, starting fresh"
  fi
fi

TRAIN_CMD+=("${TRAIN_ARGS[@]}")
exec "${TRAIN_CMD[@]}"
