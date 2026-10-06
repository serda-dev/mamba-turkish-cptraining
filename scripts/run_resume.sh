#!/bin/bash
# Resume training from a checkpoint
# Usage: ./scripts/run_resume.sh [checkpoint_path] [extra_args...]
# Example: ./scripts/run_resume.sh output/checkpoints/step_005000 --max_steps 10000

set -e

# Default to latest step checkpoint if not specified
CHECKPOINT_PATH="${1:-output/checkpoints/step_005000}"
shift 2>/dev/null || true

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    ls -la output/checkpoints/ 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

echo "=================================================="
echo "Resuming Jamba2 CPT Training"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Extra args: $@"
echo "=================================================="

# Activate conda environment
CONDA_ENV="${CONDA_ENV:-tr_jamba2_cpt}"
echo "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Run training with resume
python train.py \
    --resume "$CHECKPOINT_PATH" \
    "$@"

echo "Resume training complete!"
