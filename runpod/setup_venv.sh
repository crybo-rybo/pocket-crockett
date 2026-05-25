#!/usr/bin/env bash
# Create/update the RunPod training venv without replacing the image's CUDA PyTorch.
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT}/.env"
  set +a
fi
ROOT="${REPO_ROOT:-$ROOT}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/pocket-crockett-training}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INCLUDE_BIOCLIP="${INCLUDE_BIOCLIP:-0}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"

usage() {
  cat <<'EOF'
Usage: runpod/setup_venv.sh

Creates a Python virtual environment for RunPod training. By default it uses
--system-site-packages so CUDA-enabled torch/torchvision from the PyTorch pod
image remain visible inside the venv instead of being reinstalled by pip.

Environment variables:
  REPO_ROOT                 Path to cloned pocket-crockett repo
  VENV_DIR                  Venv path (default: /workspace/venvs/pocket-crockett-training)
  PYTHON_BIN                Python used to create the venv (default: python3)
  INCLUDE_BIOCLIP           If 1, install training/requirements-bioclip.txt
  USE_SYSTEM_SITE_PACKAGES  If 1, expose pod image packages inside the venv (default: 1)

Examples:
  ./runpod/setup_venv.sh
  INCLUDE_BIOCLIP=1 ./runpod/setup_venv.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TRAINING_DIR="${ROOT}/training"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv: $VENV_DIR"
  if [[ "$USE_SYSTEM_SITE_PACKAGES" == "1" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

check_cuda_torch() {
  python - <<'PY'
import sys

try:
    import torch
    import torchvision
except ImportError as exc:
    raise SystemExit(
        "error: torch/torchvision are not importable inside the venv. "
        "Start from a CUDA PyTorch RunPod image, or install a torch build that "
        "matches the pod's CUDA runtime before training."
    ) from exc

if not torch.cuda.is_available():
    raise SystemExit(
        "error: torch imports, but CUDA is not available. Use a GPU pod with a "
        "CUDA-enabled PyTorch image before running training."
    )

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda={torch.version.cuda}")
PY
}

check_cuda_torch
python -m pip install --upgrade pip
python -m pip install -r "${TRAINING_DIR}/requirements.txt"
if [[ "$INCLUDE_BIOCLIP" == "1" ]]; then
  python -m pip install -r "${TRAINING_DIR}/requirements-bioclip.txt"
fi
check_cuda_torch

echo "RunPod training venv ready: $VENV_DIR"
