#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing Jamba2 CPT environment"

python -m pip install --upgrade pip setuptools wheel packaging ninja

TORCH_VERSION_EXPECTED="2.5.1+cu121"
TORCH_VERSION_ACTUAL="$(python - <<'PY'
try:
    import torch
    print(torch.__version__)
except Exception:
    print("missing")
PY
)"

if [[ "${TORCH_VERSION_ACTUAL}" != "${TORCH_VERSION_EXPECTED}" ]]; then
  echo "==> Installing torch ${TORCH_VERSION_EXPECTED}"
  python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==${TORCH_VERSION_EXPECTED}"
else
  echo "==> Torch already matches ${TORCH_VERSION_EXPECTED}"
fi

PYTHON_TAG="$(python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"

CXX11_ABI="$(python - <<'PY'
import torch
print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
PY
)"

echo "==> Wheel selectors: python=${PYTHON_TAG}, cxx11abi=${CXX11_ABI}"

echo "==> Installing Python dependencies"
python -m pip install --no-cache-dir \
  "transformers==4.56.1" \
  "einops==0.8.2" \
  "bitsandbytes==0.48.1" \
  "numpy>=1.24.0" \
  "pyyaml>=6.0" \
  "tqdm>=4.65.0" \
  "psutil>=5.9.0"

echo "==> Installing CUDA extension wheels"
MAMBA_WHEEL_URL="https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abi${CXX11_ABI}-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"
CAUSAL_CONV_WHEEL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abi${CXX11_ABI}-${PYTHON_TAG}-${PYTHON_TAG}-linux_x86_64.whl"

python -m pip install --no-cache-dir \
  "${MAMBA_WHEEL_URL}" \
  "${CAUSAL_CONV_WHEEL_URL}"

echo "==> Running environment check"
python -m src.utils.env_check || true
