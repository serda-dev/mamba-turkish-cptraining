#!/bin/bash
# =============================================================================
# Perplexity Evaluation Script for Mamba CPT Model
# =============================================================================
# Usage:
#   ./scripts/run_ppl.sh                           # Run with defaults
#   ./scripts/run_ppl.sh ./output/custom           # Custom checkpoint
#   TEXTS_FILE=./eval.txt ./scripts/run_ppl.sh     # Custom texts
# =============================================================================

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Mamba CPT Perplexity Evaluation"
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
CMD="python -m src.eval.perplexity --checkpoint_dir $CHECKPOINT_DIR"

# Add optional texts file if TEXTS_FILE env var is set
if [ -n "$TEXTS_FILE" ]; then
    CMD="$CMD --texts_file $TEXTS_FILE"
fi

# Add optional max length
if [ -n "$MAX_LENGTH" ]; then
    CMD="$CMD --max_length $MAX_LENGTH"
fi

# Run evaluation
echo "[INFO] Running: $CMD"
echo "========================================"
eval $CMD
