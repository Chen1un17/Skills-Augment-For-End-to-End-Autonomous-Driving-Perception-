#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
REAL_PLAN="${REAL_PLAN:-$ROOT/experiments/dtpqa-integration/results/real_cat1_completed52_plan.json}"
REAL_PREFIX="${REAL_PREFIX:-dtpqa_real52_${TIMESTAMP}}"
SYNTH_PREFIX="${SYNTH_PREFIX:-dtpqa_synth200_${TIMESTAMP}}"
RESULTS_PREFIX="${RESULTS_PREFIX:-$ROOT/experiments/dtpqa-integration/results}"
MCP_PORT="${MCP_PORT:-8003}"
MCP_LOG="${MCP_LOG:-$ROOT/mcp_campaign_${TIMESTAMP}.log}"

REAL_EDGE_RUN_ID="${REAL_EDGE_RUN_ID:-${REAL_PREFIX}_edge}"
REAL_CLOUD_RUN_ID="${REAL_CLOUD_RUN_ID:-${REAL_PREFIX}_cloud}"
REAL_HYBRID_RUN_ID="${REAL_HYBRID_RUN_ID:-${REAL_PREFIX}_hybrid}"
SYNTH_EDGE_RUN_ID="${SYNTH_EDGE_RUN_ID:-${SYNTH_PREFIX}_edge}"
SYNTH_CLOUD_RUN_ID="${SYNTH_CLOUD_RUN_ID:-${SYNTH_PREFIX}_cloud}"
SYNTH_HYBRID_RUN_ID="${SYNTH_HYBRID_RUN_ID:-${SYNTH_PREFIX}_hybrid}"

REAL_COMPARE_PREFIX="${RESULTS_PREFIX}/real52_three_way_${TIMESTAMP}"
SYNTH_COMPARE_PREFIX="${RESULTS_PREFIX}/synth200_three_way_${TIMESTAMP}"
CAMPAIGN_SUMMARY="${RESULTS_PREFIX}/campaign_${TIMESTAMP}.json"

export EDGE_MODEL="${EDGE_MODEL:-Qwen/Qwen3.5-9B}"
export CLOUD_MODEL="${CLOUD_MODEL:-Pro/moonshotai/Kimi-K2.5}"
export JUDGE_MODEL="${JUDGE_MODEL:-Pro/moonshotai/Kimi-K2.5}"
export MCP_SERVER_PORT="$MCP_PORT"
export REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-300}"
export MAX_RETRIES="${MAX_RETRIES:-3}"

mkdir -p "$RESULTS_PREFIX"

ensure_mcp() {
  if ! bash -lc "unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; curl -sS -o /dev/null http://127.0.0.1:${MCP_PORT}/mcp -X POST -d '{}'"; then
    echo "[campaign] starting MCP server on ${MCP_PORT}"
    nohup "$ROOT/.venv/bin/ad-mcp-server" >"$MCP_LOG" 2>&1 &
    sleep 8
  fi
}

run_real_mode() {
  local mode="$1"
  local run_id="$2"
  echo "[campaign] real52 mode=${mode} run_id=${run_id}"
  "$ROOT/.venv/bin/python" "$ROOT/experiments/dtpqa-integration/code/run_real_cat1_plan.py" \
    --plan "$REAL_PLAN" \
    --run-id "$run_id" \
    --execution-mode "$mode" \
    --max-passes 3 \
    --retry-sleep-seconds 15 \
    --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS" \
    --max-retries "$MAX_RETRIES" \
    --sleep-seconds 1
  "$ROOT/.venv/bin/ad-eval-dtpqa" --run-id "$run_id" || true
}

run_synth_mode() {
  local mode="$1"
  local run_id="$2"
  local skill_store="/tmp/${run_id}_skills"
  echo "[campaign] synth200 mode=${mode} run_id=${run_id}"
  RUN_ID="$run_id" \
  LIMIT=200 \
  EXECUTION_MODE="$mode" \
  SKILL_STORE_DIR="$skill_store" \
  EDGE_MODEL="$EDGE_MODEL" \
  CLOUD_MODEL="$CLOUD_MODEL" \
  JUDGE_MODEL="$JUDGE_MODEL" \
  MCP_SERVER_PORT="$MCP_PORT" \
  REQUEST_TIMEOUT_SECONDS="$REQUEST_TIMEOUT_SECONDS" \
  MAX_RETRIES="$MAX_RETRIES" \
  bash "$ROOT/run_large_scale_200.sh"
}

ensure_mcp

run_real_mode edge_only "$REAL_EDGE_RUN_ID"
run_real_mode cloud_only "$REAL_CLOUD_RUN_ID"
run_real_mode hybrid "$REAL_HYBRID_RUN_ID"

"$ROOT/.venv/bin/python" "$ROOT/experiments/dtpqa-integration/code/compare_dtpqa_three_way_runs.py" \
  --edge-run-id "$REAL_EDGE_RUN_ID" \
  --cloud-run-id "$REAL_CLOUD_RUN_ID" \
  --hybrid-run-id "$REAL_HYBRID_RUN_ID" \
  --output-prefix "$REAL_COMPARE_PREFIX"

run_synth_mode edge_only "$SYNTH_EDGE_RUN_ID"
run_synth_mode cloud_only "$SYNTH_CLOUD_RUN_ID"
run_synth_mode hybrid "$SYNTH_HYBRID_RUN_ID"

"$ROOT/.venv/bin/python" "$ROOT/experiments/dtpqa-integration/code/compare_dtpqa_three_way_runs.py" \
  --edge-run-id "$SYNTH_EDGE_RUN_ID" \
  --cloud-run-id "$SYNTH_CLOUD_RUN_ID" \
  --hybrid-run-id "$SYNTH_HYBRID_RUN_ID" \
  --output-prefix "$SYNTH_COMPARE_PREFIX"

cat >"$CAMPAIGN_SUMMARY" <<JSON
{
  "timestamp": "$TIMESTAMP",
  "mcp_log": "$MCP_LOG",
  "real": {
    "edge_only": "$REAL_EDGE_RUN_ID",
    "cloud_only": "$REAL_CLOUD_RUN_ID",
    "hybrid": "$REAL_HYBRID_RUN_ID",
    "comparison_prefix": "$REAL_COMPARE_PREFIX"
  },
  "synth": {
    "edge_only": "$SYNTH_EDGE_RUN_ID",
    "cloud_only": "$SYNTH_CLOUD_RUN_ID",
    "hybrid": "$SYNTH_HYBRID_RUN_ID",
    "comparison_prefix": "$SYNTH_COMPARE_PREFIX"
  }
}
JSON

echo "[campaign] complete"
echo "[campaign] summary: $CAMPAIGN_SUMMARY"
