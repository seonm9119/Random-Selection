#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/Random-Selection}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
BRANCH="${BRANCH:-main}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/.git/auto_commit_push_loop.log}"
LOCK_DIR="$PROJECT_DIR/.git/auto_commit_push_loop.lock"

cd "$PROJECT_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "$(date -Is) auto commit loop is already running" >> "$LOG_FILE"
  exit 0
fi

cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

log() {
  echo "$(date -Is) $*" >> "$LOG_FILE"
}

push_with_token() {
  if [[ ! -f token.txt ]]; then
    log "skip push: token.txt not found"
    return 1
  fi

  local token basic_auth
  token="$(tr -d '\r\n' < token.txt)"
  basic_auth="$(printf 'x-access-token:%s' "$token" | base64 -w 0)"
  git -c http.extraHeader="Authorization: Basic $basic_auth" push origin "$BRANCH"
}

stage_allowed_changes() {
  git add -A -- . \
    ':!token.txt' \
    ':!code_style.txt' \
    ':!scripts/results/**/*.pt' \
    ':!benchmark/*/pretrained/*' \
    ':!simclr/pretrained/*' \
    ':!rscl/pretrained/*'
}

make_commit_message() {
  local staged_paths
  staged_paths="$(git diff --cached --name-only)"

  if grep -q '^scripts/results/smoke/rscl_.*batch512' <<< "$staged_paths"; then
    printf 'chore: sync rscl batch512 smoke results'
  elif grep -q '^scripts/results/' <<< "$staged_paths"; then
    printf 'chore: sync experiment results'
  elif grep -q '^scripts/' <<< "$staged_paths"; then
    printf 'chore: sync experiment scripts'
  else
    printf 'chore: sync project updates'
  fi
}

sync_once() {
  git status --short >/dev/null
  stage_allowed_changes

  if git diff --cached --quiet; then
    log "no changes to commit"
    return 0
  fi

  local message
  message="$(make_commit_message)"
  git commit -m "$message"
  log "committed: $message"

  if push_with_token; then
    log "pushed $BRANCH"
  else
    log "push failed"
    return 1
  fi
}

log "auto commit loop started interval=${INTERVAL_SECONDS}s branch=$BRANCH"

while true; do
  sync_once || true
  sleep "$INTERVAL_SECONDS"
done
