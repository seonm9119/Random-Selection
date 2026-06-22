#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
STATUS_FILE="$PROJECT_DIR/scripts/run_rscl_batch1024_search_when_ready.status"
LOG_FILE="$PROJECT_DIR/logs/run_rscl_batch1024_search_when_ready.log"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results/smoke"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "runner_started_at=$(date -Is)" >"$STATUS_FILE"
echo "mode=wait_for_gpu_memory" >>"$STATUS_FILE"
echo "required_free_memory_mb=30000" >>"$STATUS_FILE"

run_search() {
  local dataset="$1"
  local output_path="$2"

  echo "running=${dataset}_batch1024_search started_at=$(date -Is)" >>"$STATUS_FILE"
  "$PYTHON_BIN" -B scripts/rscl_smoke_hparam_search.py \
    --dataset "$dataset" \
    --batch-sizes 1024 \
    --warmup-epochs 10 \
    --observe-epochs 50 \
    --val-every-epochs 5 \
    --val-steps 1 \
    --num-workers 0 \
    --device cuda \
    --amp \
    --wait-for-gpu-memory \
    --gpu-memory-poll-seconds 120 \
    --output-path "$output_path" \
    >>"$LOG_FILE" 2>&1
  echo "completed=${dataset}_batch1024_search finished_at=$(date -Is)" >>"$STATUS_FILE"
}

run_search cifar10 results/smoke/rscl_cifar10_batch1024_hparam_search_warmup10_observe50_val5.json
run_search cifar100 results/smoke/rscl_cifar100_batch1024_hparam_search_warmup10_observe50_val5.json

echo "runner_finished_at=$(date -Is)" >>"$STATUS_FILE"
