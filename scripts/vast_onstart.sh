#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/mamba-cpt-tr}"
cd "${PROJECT_DIR}"

echo "==> Vast on-start script"
echo "PROJECT_DIR=${PROJECT_DIR}"

mkdir -p /workspace/hf-cache
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf-cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/hf-cache/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AUTO_RESUME="${AUTO_RESUME:-1}"

if [[ "${AUTO_INSTALL:-1}" == "1" ]]; then
  bash scripts/install_env.sh
fi

if [[ "${AUTO_ENV_CHECK:-1}" == "1" ]]; then
  python -m src.utils.env_check || true
fi

echo "==> Vast startup complete"
echo "Repo: ${PROJECT_DIR}"
echo "HF cache: ${HF_HOME}"

if [[ "${AUTO_TRAIN:-0}" == "1" ]]; then
  echo "==> AUTO_TRAIN=1, starting training"
  exec bash scripts/start_train_vast.sh
fi
