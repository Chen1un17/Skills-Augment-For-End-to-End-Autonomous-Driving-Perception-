#!/usr/bin/env bash

set -u

trap 'exit 130' INT TERM

RUN_ID="${RUN_ID:-run-dtpqa-real-cat1-kimi512-pilot}"
SUBSET="${SUBSET:-real}"
QUESTION_TYPE="${QUESTION_TYPE:-category_1}"
EDGE_MODEL="${EDGE_MODEL:-Qwen/Qwen3.5-9B}"
CLOUD_MODEL="${CLOUD_MODEL:-Pro/moonshotai/Kimi-K2.5}"
JUDGE_MODEL="${JUDGE_MODEL:-Pro/moonshotai/Kimi-K2.5}"
EXECUTION_MODE="${EXECUTION_MODE:-edge_only}"
EDGE_MAX_COMPLETION_TOKENS="${EDGE_MAX_COMPLETION_TOKENS:-512}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-1}"
DTPQA_ROOT="${DTPQA_ROOT:-data/dtpqa}"
SKILL_STORE_DIR="${SKILL_STORE_DIR:-/tmp/dtpqa_skills_empty}"
BATCH_SLEEP_SECONDS="${BATCH_SLEEP_SECONDS:-0}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 offset [offset ...]" >&2
  exit 1
fi

successes=0
failures=0

for offset in "$@"; do
  printf '[%s] replay offset=%s run_id=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$offset" "$RUN_ID"

  REQUEST_TIMEOUT_SECONDS="$REQUEST_TIMEOUT_SECONDS" \
  MAX_RETRIES="$MAX_RETRIES" \
  EDGE_MODEL="$EDGE_MODEL" \
  CLOUD_MODEL="$CLOUD_MODEL" \
  JUDGE_MODEL="$JUDGE_MODEL" \
  EDGE_MAX_COMPLETION_TOKENS="$EDGE_MAX_COMPLETION_TOKENS" \
  DTPQA_ROOT="$DTPQA_ROOT" \
  SKILL_STORE_DIR="$SKILL_STORE_DIR" \
  .venv/bin/ad-replay-dtpqa \
    --subset "$SUBSET" \
    --question-type "$QUESTION_TYPE" \
    --offset "$offset" \
    --limit 1 \
    --run-id "$RUN_ID" \
    --execution-mode "$EXECUTION_MODE" \
    --append

  status=$?
  if [ "$status" -eq 0 ]; then
    successes=$((successes + 1))
  else
    failures=$((failures + 1))
  fi

  printf '[%s] replay offset=%s status=%s successes=%s failures=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "$offset" \
    "$status" \
    "$successes" \
    "$failures"

  if [ "$BATCH_SLEEP_SECONDS" -gt 0 ]; then
    sleep "$BATCH_SLEEP_SECONDS"
  fi
done

printf '[%s] batch complete run_id=%s successes=%s failures=%s\n' \
  "$(date '+%Y-%m-%d %H:%M:%S')" \
  "$RUN_ID" \
  "$successes" \
  "$failures"
