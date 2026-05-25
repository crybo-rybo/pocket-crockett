#!/usr/bin/env bash
# Pull vision tar shards from Cloudflare R2, verify checksums, and unpack once
# onto a RunPod Network Volume for training.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHARD_DIR="${SHARD_DIR:-/workspace/shards}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
CHECKSUMS_FILE="${CHECKSUMS_FILE:-$ROOT/vision/checksums.sha256}"
R2_REMOTE="${R2_REMOTE:-r2}"
R2_BUCKET="${R2_BUCKET:-pocket-crockett-vision}"
R2_PREFIX="${R2_PREFIX:-vision-shards/}"
INCLUDE_PRETRAINING="${INCLUDE_PRETRAINING:-0}"
SKIP_PULL="${SKIP_PULL:-0}"
SKIP_UNPACK="${SKIP_UNPACK:-0}"

usage() {
  cat <<'EOF'
Usage: runpod/bootstrap.sh [OPTIONS]

Environment variables:
  SHARD_DIR            Local directory for downloaded .tar files (default: /workspace/shards)
  DATA_ROOT            Unpacked dataset root; images land under DATA_ROOT/images/... (default: /workspace/data)
  R2_REMOTE            rclone remote name (default: r2)
  R2_BUCKET            R2 bucket name (default: pocket-crockett-vision)
  R2_PREFIX            Key prefix inside the bucket (default: vision-shards/)
  INCLUDE_PRETRAINING  If 1, also pull/unpack pretraining-*.tar using checksums_pretraining.sha256
  SKIP_PULL            If 1, skip rclone and assume shards already exist in SHARD_DIR
  SKIP_UNPACK          If 1, skip tar extraction (shards only)
  SHARDS               Space-separated shard names to process (default: all output-class shards)

Examples:
  # Output-class shards only (first training campaign).
  ./runpod/bootstrap.sh

  # Smoke test: pull and unpack only train-000 and val-000.
  SHARDS="train-000.tar val-000.tar" ./runpod/bootstrap.sh

  # Include PlantNet pretraining pool for a later campaign.
  INCLUDE_PRETRAINING=1 ./runpod/bootstrap.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEFAULT_SHARDS=(
  train-000.tar train-001.tar train-002.tar train-003.tar
  val-000.tar test-000.tar calibration-000.tar
)
if [[ -n "${SHARDS:-}" ]]; then
  # shellcheck disable=SC2206
  SELECTED_SHARDS=($SHARDS)
else
  SELECTED_SHARDS=("${DEFAULT_SHARDS[@]}")
  if [[ "$INCLUDE_PRETRAINING" == "1" ]]; then
    SELECTED_SHARDS+=(
      pretraining-000.tar pretraining-001.tar pretraining-002.tar pretraining-003.tar
      pretraining-004.tar pretraining-005.tar pretraining-006.tar pretraining-007.tar
      pretraining-008.tar pretraining-009.tar
    )
  fi
fi

if [[ "$INCLUDE_PRETRAINING" == "1" && -z "${SHARDS:-}" ]]; then
  CHECKSUMS_FILES=("$CHECKSUMS_FILE" "$ROOT/vision/checksums_pretraining.sha256")
else
  CHECKSUMS_FILES=("$CHECKSUMS_FILE")
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 2
  fi
}

ensure_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    return
  fi
  echo "Installing rclone..."
  require_cmd curl
  curl -fsSL https://rclone.org/install.sh | bash
}

verify_selected_shards() {
  local checksums="$1"
  local tmp
  tmp="$(mktemp)"
  for shard in "${SELECTED_SHARDS[@]}"; do
    grep -F "  shards/$shard" "$checksums" >>"$tmp" || {
      echo "error: $shard not listed in $checksums" >&2
      rm -f "$tmp"
      exit 2
    }
  done
  # checksums.sha256 paths are repo-relative (shards/foo.tar); verify inside SHARD_DIR.
  sed 's#  shards/#  #' "$tmp" >"${tmp}.paths"
  echo "Verifying shard checksums from $(basename "$checksums")..."
  (cd "$SHARD_DIR" && shasum -a 256 -c "${tmp}.paths")
  rm -f "$tmp" "${tmp}.paths"
}

pull_shards() {
  ensure_rclone
  mkdir -p "$SHARD_DIR"
  local remote_path="${R2_REMOTE}:${R2_BUCKET}/${R2_PREFIX}"
  echo "Pulling ${#SELECTED_SHARDS[@]} shard(s) from ${remote_path} -> ${SHARD_DIR}"
  for shard in "${SELECTED_SHARDS[@]}"; do
    rclone copyto \
      "${remote_path}${shard}" \
      "${SHARD_DIR}/${shard}" \
      --progress \
      --checksum
  done
}

unpack_shards() {
  mkdir -p "$DATA_ROOT"
  for shard in "${SELECTED_SHARDS[@]}"; do
    local tar_path="${SHARD_DIR}/${shard}"
    if [[ ! -f "$tar_path" ]]; then
      echo "error: missing shard: $tar_path" >&2
      exit 2
    fi
    echo "Unpacking ${shard} -> ${DATA_ROOT}"
    tar -xf "$tar_path" -C "$DATA_ROOT" --exclude manifest.json
  done
}

echo "=== Pocket Crockett vision bootstrap ==="
echo "ROOT=$ROOT"
echo "SHARD_DIR=$SHARD_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "Selected shards: ${SELECTED_SHARDS[*]}"

if [[ "$SKIP_PULL" != "1" ]]; then
  pull_shards
else
  echo "Skipping rclone pull (SKIP_PULL=1)"
fi

mkdir -p "$SHARD_DIR"
for checksums in "${CHECKSUMS_FILES[@]}"; do
  if [[ ! -f "$checksums" ]]; then
    echo "error: checksums file not found: $checksums" >&2
    exit 2
  fi
  verify_selected_shards "$checksums"
done

if [[ "$SKIP_UNPACK" != "1" ]]; then
  unpack_shards
else
  echo "Skipping unpack (SKIP_UNPACK=1)"
fi

echo
echo "Disk usage:"
du -sh "$SHARD_DIR" "$DATA_ROOT" 2>/dev/null || true
echo "Bootstrap complete."
