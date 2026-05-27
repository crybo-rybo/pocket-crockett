#!/usr/bin/env bash
# Run a training stage on RunPod after bootstrap.sh has populated DATA_ROOT.
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT}/.env"
  set +a
fi
ROOT="${REPO_ROOT:-$ROOT}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs}"
STAGE="${1:-train}"
CONFIG="${CONFIG:-}"

usage() {
  cat <<'EOF'
Usage: runpod/train.sh STAGE [OPTIONS]

Stages:
  smoke      Short run using training/configs/smoke_test.yaml
  train      Full baseline fine-tune (training/configs/baseline_convnext_tiny.yaml)
  calibrate  Fit temperature scaling on the calibration split
  eval       Evaluate best checkpoint on val/test splits
  upload     Upload run artifacts from OUTPUT_DIR/run to R2

Environment variables:
  REPO_ROOT   Path to cloned pocket-crockett repo (default: parent of runpod/)
  DATA_ROOT   Unpacked dataset root from bootstrap.sh
  OUTPUT_DIR  Directory for checkpoints and reports
  CONFIG      Override config YAML path for train/smoke stages
  RUN_NAME    Subdirectory under OUTPUT_DIR (default: STAGE timestamp)
  VENV_DIR    Training venv path (default: /workspace/venvs/pocket-crockett-training)
  USE_VENV    If 1, use runpod/setup_venv.sh and venv Python (default: 1)
  REFRESH_VENV If 1, rerun setup even when the venv already exists
  PYTHON      Fallback Python interpreter when USE_VENV=0 (default: python3)

Examples:
  ./runpod/train.sh smoke
  RUN_NAME=baseline-v1 ./runpod/train.sh train
  RUN_NAME=baseline-v1 ./runpod/train.sh calibrate
  RUN_NAME=baseline-v1 ./runpod/train.sh eval
  RUN_NAME=baseline-v1 ./runpod/train.sh upload
  RUN_NAME=bioclip-direct-v2 ./runpod/train.sh bioclip-finetune
  PRETRAIN_CHECKPOINT=/workspace/runs/bioclip-pretrain-v2/checkpoint-best.pt RUN_NAME=bioclip-v2 ./runpod/train.sh bioclip-finetune
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON="${PYTHON:-python3}"
USE_VENV="${USE_VENV:-1}"
REFRESH_VENV="${REFRESH_VENV:-0}"
VENV_DIR="${VENV_DIR:-/workspace/venvs/pocket-crockett-training}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-${STAGE}-${TIMESTAMP}}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
TRAINING_DIR="${ROOT}/training"

setup_python_env() {
  if [[ "$USE_VENV" == "1" ]]; then
    local include_bioclip=0
    case "$STAGE" in
      bioclip-pretrain|bioclip-finetune) include_bioclip=1 ;;
      *) ;;
    esac

    if [[ ! -x "${VENV_DIR}/bin/python" || "$REFRESH_VENV" == "1" || "$include_bioclip" == "1" ]]; then
      INCLUDE_BIOCLIP="$include_bioclip" VENV_DIR="$VENV_DIR" PYTHON_BIN="$PYTHON" "${ROOT}/runpod/setup_venv.sh"
    fi
    PYTHON="${VENV_DIR}/bin/python"
  fi
}

resolve_config() {
  case "$STAGE" in
    smoke) CONFIG="${CONFIG:-${TRAINING_DIR}/configs/smoke_test.yaml}" ;;
    train) CONFIG="${CONFIG:-${TRAINING_DIR}/configs/baseline_convnext_tiny.yaml}" ;;
    bioclip-pretrain) CONFIG="${CONFIG:-${TRAINING_DIR}/configs/bioclip_two_stage.yaml}" ;;
    bioclip-finetune) CONFIG="${CONFIG:-${TRAINING_DIR}/configs/bioclip_two_stage.yaml}" ;;
    *) ;;
  esac
}

case "$STAGE" in
  smoke|train|calibrate|eval|upload|bioclip-pretrain|bioclip-finetune) ;;
  *)
    echo "error: unknown stage: $STAGE" >&2
    usage
    exit 2
    ;;
esac

setup_python_env
mkdir -p "$RUN_DIR"
resolve_config

echo "=== Pocket Crockett train stage: $STAGE ==="
echo "RUN_DIR=$RUN_DIR"
echo "DATA_ROOT=$DATA_ROOT"

case "$STAGE" in
  smoke|train)
    "$PYTHON" "${TRAINING_DIR}/train.py" \
      --config "$CONFIG" \
      --data-root "$DATA_ROOT" \
      --output-dir "$RUN_DIR"
    ;;
  bioclip-pretrain)
    "$PYTHON" "${TRAINING_DIR}/train.py" \
      --config "$CONFIG" \
      --data-root "$DATA_ROOT" \
      --output-dir "$RUN_DIR" \
      --stage pretrain
    ;;
  bioclip-finetune)
    PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-${CHECKPOINT:-}}"
    CMD=(
      "$PYTHON" "${TRAINING_DIR}/train.py"
      --config "$CONFIG"
      --data-root "$DATA_ROOT"
      --output-dir "$RUN_DIR"
      --stage finetune
    )
    if [[ -n "$PRETRAIN_CHECKPOINT" ]]; then
      CMD+=(--checkpoint "$PRETRAIN_CHECKPOINT")
    else
      echo "No PRETRAIN_CHECKPOINT set; fine-tuning from the base BioCLIP weights."
    fi
    "${CMD[@]}"
    ;;
  calibrate)
    "$PYTHON" "${TRAINING_DIR}/calibrate.py" \
      --run-dir "$RUN_DIR" \
      --data-root "$DATA_ROOT"
    ;;
  eval)
    "$PYTHON" "${TRAINING_DIR}/evaluate.py" \
      --run-dir "$RUN_DIR" \
      --data-root "$DATA_ROOT" \
      --splits val test
    ;;
  upload)
    "$PYTHON" "${ROOT}/tools/upload_checkpoints_to_r2.py" \
      --run-dir "$RUN_DIR"
    ;;
esac

echo "Stage complete: $STAGE -> $RUN_DIR"
