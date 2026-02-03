#!/bin/bash
# =============================================================================
# Inference Script for Mamba CPT Model
# =============================================================================
# Usage:
#   ./scripts/run_infer.sh                      # Run with defaults
#   ./scripts/run_infer.sh ./output/custom      # Custom checkpoint
#   PROMPT="Merhaba" ./scripts/run_infer.sh     # Custom prompt
# =============================================================================

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Mamba CPT Inference"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "========================================"

# Checkpoint directory (can be overridden via first argument)
CHECKPOINT_DIR="${1:-./output/checkpoints/final}"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/model.safetensors" ]; then
    echo "[ERROR] model.safetensors not found in: $CHECKPOINT_DIR"
    exit 1
fi

echo "[INFO] Using checkpoint: $CHECKPOINT_DIR"

# Build command
CMD="python -m src.eval.infer --checkpoint_dir $CHECKPOINT_DIR"

# Add optional prompt if PROMPT env var is set
if [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt \"$PROMPT\""
fi

# Add optional prompts file if PROMPTS_FILE env var is set
if [ -n "$PROMPTS_FILE" ]; then
    CMD="$CMD --prompts_file $PROMPTS_FILE"
fi

# Add optional generation parameters
if [ -n "$MAX_NEW_TOKENS" ]; then
    CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
fi

if [ -n "$TEMPERATURE" ]; then
    CMD="$CMD --temperature $TEMPERATURE"
fi

if [ -n "$TOP_P" ]; then
    CMD="$CMD --top_p $TOP_P"
fi

# Run inference
echo "[INFO] Running: $CMD"
echo "========================================"
eval $CMD
