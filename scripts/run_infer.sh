#!/bin/bash
# =============================================================================
# Inference Script for Jamba2 CPT Models
# =============================================================================
# Usage:
#   ./scripts/run_infer.sh
#   ./scripts/run_infer.sh serda-dev/Jamba2-3B-Turkish
#   ./scripts/run_infer.sh ./output/checkpoints/final
#   PROMPT="Merhaba" ./scripts/run_infer.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Jamba2 Inference"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "========================================"

MODEL_SOURCE="${1:-serda-dev/Jamba2-3B-Turkish}"

if [ -d "$MODEL_SOURCE" ]; then
    echo "[INFO] Using local checkpoint: $MODEL_SOURCE"
else
    echo "[INFO] Using Hugging Face model repo: $MODEL_SOURCE"
fi

CMD=(python -m src.eval.infer --model_name_or_path "$MODEL_SOURCE")

if [ -n "${PROMPT:-}" ]; then
    CMD+=(--prompt "$PROMPT")
fi

if [ -n "${PROMPTS_FILE:-}" ]; then
    CMD+=(--prompts_file "$PROMPTS_FILE")
fi

if [ -n "${SYSTEM_PROMPT:-}" ]; then
    CMD+=(--system_prompt "$SYSTEM_PROMPT")
fi

if [ -n "${MAX_NEW_TOKENS:-}" ]; then
    CMD+=(--max_new_tokens "$MAX_NEW_TOKENS")
fi

if [ -n "${TEMPERATURE:-}" ]; then
    CMD+=(--temperature "$TEMPERATURE")
fi

if [ -n "${TOP_P:-}" ]; then
    CMD+=(--top_p "$TOP_P")
fi

if [ -n "${TOP_K:-}" ]; then
    CMD+=(--top_k "$TOP_K")
fi

if [ -n "${REPETITION_PENALTY:-}" ]; then
    CMD+=(--repetition_penalty "$REPETITION_PENALTY")
fi

if [ -n "${TORCH_DTYPE:-}" ]; then
    CMD+=(--torch_dtype "$TORCH_DTYPE")
fi

if [ -n "${ATTN_IMPLEMENTATION:-}" ]; then
    CMD+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi

if [ -n "${DEVICE:-}" ]; then
    CMD+=(--device "$DEVICE")
fi

if [ -n "${DEVICE_MAP:-}" ]; then
    CMD+=(--device_map "$DEVICE_MAP")
fi

if [ "${NO_SAMPLE:-0}" = "1" ]; then
    CMD+=(--no_sample)
fi

echo "[INFO] Running:"
printf ' %q' "${CMD[@]}"
printf '\n'
echo "========================================"
"${CMD[@]}"
