#!/bin/bash
# Large-scale experiment with 200+ samples and judge evaluation

set -euo pipefail

cd "$(dirname "$0")"

# Clear proxy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

RUN_ID="${RUN_ID:-dtpqa_synth_200_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-200}"
OFFSET="${OFFSET:-0}"
EXECUTION_MODE="${EXECUTION_MODE:-edge_only}"
SKILL_STORE_DIR="${SKILL_STORE_DIR:-/tmp/dtpqa_skills_empty}"
MCP_SERVER_PORT="${MCP_SERVER_PORT:-8003}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://127.0.0.1:${MCP_SERVER_PORT}/mcp}"
RETRIES_PER_CASE="${RETRIES_PER_CASE:-2}"
START_FRESH_SERVER="${START_FRESH_SERVER:-1}"
export EDGE_MODEL="${EDGE_MODEL:-Qwen/Qwen3.5-9B}"
export CLOUD_MODEL="${CLOUD_MODEL:-Pro/moonshotai/Kimi-K2.5}"
export JUDGE_MODEL="${JUDGE_MODEL:-Pro/moonshotai/Kimi-K2.5}"
export MCP_SERVER_PORT
export MCP_SERVER_URL
export SKILL_STORE_DIR

RUN_DIR="data/artifacts/${RUN_ID}"
FAILURES_PATH="${RUN_DIR}/failures.jsonl"
SERVER_LOG_PATH="${RUN_DIR}/mcp_server.log"
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Large-Scale Experiment: $LIMIT samples"
echo "Run ID: $RUN_ID"
echo "=============================================="
echo ""

# Clean skill store
rm -rf "$SKILL_STORE_DIR"
mkdir -p "$SKILL_STORE_DIR"

MCP_SERVER_PID=""
cleanup() {
    if [ -n "${MCP_SERVER_PID}" ] && kill -0 "${MCP_SERVER_PID}" >/dev/null 2>&1; then
        kill "${MCP_SERVER_PID}" >/dev/null 2>&1 || true
        wait "${MCP_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

mcp_is_healthy() {
    local status
    status="$(curl -s -o /dev/null -w '%{http_code}' "${MCP_SERVER_URL}" || true)"
    [ "${status}" = "200" ] || [ "${status}" = "406" ]
}

# Ensure MCP server is running against the same clean skill store used by the client.
if [ "$EXECUTION_MODE" = "hybrid" ]; then
    if [ "$START_FRESH_SERVER" = "1" ]; then
        if lsof -ti tcp:"${MCP_SERVER_PORT}" >/dev/null 2>&1; then
            echo "Stopping existing MCP server on port ${MCP_SERVER_PORT}..."
            lsof -ti tcp:"${MCP_SERVER_PORT}" | xargs kill >/dev/null 2>&1 || true
            sleep 1
        fi
        echo "Starting clean MCP server at ${MCP_SERVER_URL}..."
        .venv/bin/ad-mcp-server >"${SERVER_LOG_PATH}" 2>&1 &
        MCP_SERVER_PID=$!
        for _ in $(seq 1 30); do
            if mcp_is_healthy; then
                break
            fi
            sleep 1
        done
    fi

    if ! mcp_is_healthy; then
        echo "MCP server failed to become healthy at ${MCP_SERVER_URL}" >&2
        exit 1
    fi
fi

# Run 200 samples sequentially
echo "Starting data collection..."
for ((i=OFFSET; i<OFFSET+LIMIT; i++)); do
    progress_index=$((i - OFFSET))
    if [ $((progress_index % 10)) -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Progress: ${progress_index}/$LIMIT ($((progress_index*100/LIMIT))%)"
    fi

    CMD=(
        .venv/bin/ad-replay-dtpqa
        --subset synth
        --question-type category_1
        --offset "$i"
        --limit 1
        --run-id "$RUN_ID"
        --execution-mode "$EXECUTION_MODE"
    )
    if [ "$EXECUTION_MODE" = "hybrid" ]; then
        CMD+=(--server-url "$MCP_SERVER_URL")
    fi
    if [ -f "${RUN_DIR}/predictions.jsonl" ]; then
        CMD+=(--append)
    fi

    attempt=1
    success=0
    case_log="${RUN_DIR}/case_${i}.log"
    while [ "$attempt" -le "$RETRIES_PER_CASE" ]; do
        if "${CMD[@]}" >"${case_log}" 2>&1; then
            tail -n 1 "${case_log}"
            success=1
            break
        fi
        echo "Case offset ${i} failed on attempt ${attempt}. See ${case_log}" >&2
        attempt=$((attempt + 1))
        sleep 1
    done

    if [ "$success" -ne 1 ]; then
        export FAILED_OFFSET="${i}"
        export FAILED_CASE_LOG="${case_log}"
        .venv/bin/python >>"${FAILURES_PATH}" <<'PY'
import json
import os
from pathlib import Path
payload = {
    "offset": int(os.environ["FAILED_OFFSET"]),
    "log_path": os.environ["FAILED_CASE_LOG"],
    "tail": Path(os.environ["FAILED_CASE_LOG"]).read_text(encoding="utf-8", errors="ignore").splitlines()[-20:],
}
print(json.dumps(payload, ensure_ascii=False))
PY
        unset FAILED_OFFSET
        unset FAILED_CASE_LOG
    fi
done

echo ""
echo "=============================================="
echo "Data Collection Complete"
echo "=============================================="
echo ""

# Analyze results
echo "Analyzing results..."
uv run python analyze_results.py "$RUN_ID"

echo ""
echo "=============================================="
echo "Experiment Complete: $RUN_ID"
echo "Results: data/artifacts/$RUN_ID"
echo "=============================================="
