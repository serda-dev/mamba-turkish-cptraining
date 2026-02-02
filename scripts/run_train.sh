#!/bin/bash
# Mamba CPT Training Script
# Usage: ./scripts/run_train.sh [config_path] [extra_args...]

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Default config
CONFIG="${1:-configs/train.yaml}"
shift 2>/dev/null || true

echo "========================================"
echo "Mamba CPT Pipeline - Turkish"
echo "========================================"
echo "Config: $CONFIG"
echo "Extra args: $@"
echo "========================================"

# Activate conda environment
CONDA_ENV="${CONDA_ENV:-tr_mamba_cpt}"
echo "Activating conda environment: $CONDA_ENV"

# Initialize conda for script usage
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Check Python
python --version

# Run training
python train.py --config "$CONFIG" "$@"

echo "========================================"
echo "Training complete!"
echo "========================================"
