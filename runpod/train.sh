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
SPLITS_DIR="${SPLITS_DIR:-${ROOT}/vision/splits}"

usage() {
  cat <<'EOF'
Usage: runpod/train.sh STAGE [OPTIONS]

Stages:
  smoke      Short run using training/configs/smoke_test.yaml
  train      Full baseline fine-tune (training/configs/baseline_convnext_tiny.yaml)
  calibrate  Fit temperature scaling on the calibration split
  ood        Fit embedding-distance OOD guard after calibration
  eval       Evaluate best checkpoint on val/test splits
  upload     Upload run artifacts from OUTPUT_DIR/run to R2

Environment variables:
  REPO_ROOT   Path to cloned pocket-crockett repo (default: parent of runpod/)
  DATA_ROOT   Unpacked dataset root from bootstrap.sh
  OUTPUT_DIR  Directory for checkpoints and reports
  SPLITS_DIR   Split CSV directory (default: $REPO_ROOT/vision/splits)
  CONFIG      Override config YAML path for train/smoke stages
  RUN_NAME    Subdirectory under OUTPUT_DIR (default: STAGE timestamp)
  OOD_NEGATIVES_CSV Optional CSV of extra OOD validation negatives
  FAIL_ON_RELEASE_GATE If 1, eval exits non-zero when danger/OOD release gates fail (default: 1)
  VENV_DIR    Training venv path (default: /workspace/venvs/pocket-crockett-training)
  USE_VENV    If 1, use runpod/setup_venv.sh and venv Python (default: 1)
  REFRESH_VENV If 1, rerun setup even when the venv already exists
  PYTHON      Fallback Python interpreter when USE_VENV=0 (default: python3)

Examples:
  ./runpod/train.sh smoke
  RUN_NAME=baseline-v1 ./runpod/train.sh train
  RUN_NAME=baseline-v1 ./runpod/train.sh calibrate
  RUN_NAME=baseline-v1 ./runpod/train.sh ood
  RUN_NAME=baseline-v1 ./runpod/train.sh eval
  RUN_NAME=baseline-v1 ./runpod/train.sh upload
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
FAIL_ON_RELEASE_GATE="${FAIL_ON_RELEASE_GATE:-1}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-${STAGE}-${TIMESTAMP}}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
TRAINING_DIR="${ROOT}/training"

setup_python_env() {
  if [[ "$USE_VENV" == "1" ]]; then
    local include_bioclip="${INCLUDE_BIOCLIP:-0}"
    case "$STAGE" in
      bioclip-pretrain|bioclip-finetune) include_bioclip=1 ;;
      ood)
        if [[ -f "${RUN_DIR}/config.resolved.json" ]] && grep -q '"type": "bioclip"' "${RUN_DIR}/config.resolved.json"; then
          include_bioclip=1
        fi
        ;;
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
  smoke|train|calibrate|ood|eval|upload|bioclip-pretrain|bioclip-finetune) ;;
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
    if [[ -z "$PRETRAIN_CHECKPOINT" ]]; then
      echo "error: set PRETRAIN_CHECKPOINT to the bioclip-pretrain checkpoint-best.pt path" >&2
      exit 2
    fi
    "$PYTHON" "${TRAINING_DIR}/train.py" \
      --config "$CONFIG" \
      --data-root "$DATA_ROOT" \
      --output-dir "$RUN_DIR" \
      --stage finetune \
      --checkpoint "$PRETRAIN_CHECKPOINT"
    ;;
  calibrate)
    "$PYTHON" "${TRAINING_DIR}/calibrate.py" \
      --run-dir "$RUN_DIR" \
      --data-root "$DATA_ROOT" \
      --splits-dir "$SPLITS_DIR"
    ;;
  ood)
    ood_args=(
      fit
      --run-dir "$RUN_DIR"
      --data-root "$DATA_ROOT"
      --splits-dir "$SPLITS_DIR"
    )
    if [[ -n "${OOD_NEGATIVES_CSV:-}" ]]; then
      ood_args+=(--ood-negatives-csv "$OOD_NEGATIVES_CSV")
    fi
    "$PYTHON" "${TRAINING_DIR}/ood.py" "${ood_args[@]}"
    ;;
  eval)
    eval_args=(
      --run-dir "$RUN_DIR"
      --data-root "$DATA_ROOT"
      --splits-dir "$SPLITS_DIR"
      --splits val test
    )
    if [[ "$FAIL_ON_RELEASE_GATE" == "1" ]]; then
      eval_args+=(--fail-on-release-gate)
    fi
    "$PYTHON" "${TRAINING_DIR}/evaluate.py" "${eval_args[@]}"
    ;;
  upload)
    "$PYTHON" "${ROOT}/tools/upload_checkpoints_to_r2.py" \
      --run-dir "$RUN_DIR"
    ;;
esac

echo "Stage complete: $STAGE -> $RUN_DIR"
