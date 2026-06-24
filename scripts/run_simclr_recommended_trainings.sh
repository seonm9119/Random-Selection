#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
TRAIN_SCRIPT="$PROJECT_DIR/simclr/train.py"
OUTPUT_DIR="$PROJECT_DIR/simclr/pretrained"
STATUS_FILE="$PROJECT_DIR/scripts/run_simclr_recommended_trainings.status"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "runner_started_at=$(date -Is)" >"$STATUS_FILE"

run_training() {
  local dataset="$1"
  local batch_size="$2"
  local learning_rate="$3"
  local temperature="$4"
  local basename="${dataset}_batch_${batch_size}_best"

  echo "running=$basename started_at=$(date -Is)" >>"$STATUS_FILE"

  set +e
  "$PYTHON_BIN" -B "$TRAIN_SCRIPT" \
    --dataset "$dataset" \
    --epochs 500 \
    --batch-size "$batch_size" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$learning_rate" \
    --temperature "$temperature" \
    --weight-decay 1e-6 \
    --warmup-epochs 10 \
    --num-workers 4 \
    --device cuda \
    --amp \
    --suppress-external-progress
  local exit_code=$?
  set -e

  if [ "$exit_code" -ne 0 ]; then
    echo "failed=$basename exit_code=$exit_code failed_at=$(date -Is)" >>"$STATUS_FILE"
    exit "$exit_code"
  fi

  echo "completed=$basename finished_at=$(date -Is)" >>"$STATUS_FILE"
}

run_training cifar10 1024 1.5 0.5
run_training cifar10 512 1.5 1.0
run_training cifar10 256 0.5 1.0
run_training cifar100 1024 1.5 0.5
run_training cifar100 512 0.5 1.0
run_training cifar100 256 0.5 1.0

echo "runner_finished_at=$(date -Is)" >>"$STATUS_FILE"
