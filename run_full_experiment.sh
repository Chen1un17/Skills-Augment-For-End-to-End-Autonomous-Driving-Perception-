#!/bin/bash
# Full automated experiment runner

set -e

cd "$(dirname "$0")"

# Clear proxy settings
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

LIMIT="${1:-20}"
RUN_ID="dtpqa_synth_full_$(date +%Y%m%d_%H%M%S)"
EXECUTION_MODE="${EXECUTION_MODE:-hybrid}"
export EDGE_MODEL="${EDGE_MODEL:-Qwen/Qwen3.5-9B}"
export CLOUD_MODEL="${CLOUD_MODEL:-Pro/moonshotai/Kimi-K2.5}"
export JUDGE_MODEL="${JUDGE_MODEL:-Pro/moonshotai/Kimi-K2.5}"

echo "=============================================="
echo "DTPQA Synthetic Full Experiment"
echo "Run ID: $RUN_ID"
echo "Samples: $LIMIT"
echo "=============================================="
echo ""

# Clean skill store
rm -rf /tmp/dtpqa_skills_empty/*
mkdir -p /tmp/dtpqa_skills_empty

# Run samples sequentially
for ((i=0; i<$LIMIT; i++)); do
    echo "[$(date '+%H:%M:%S')] Running sample $((i+1))/$LIMIT"

    if [ $i -eq 0 ]; then
        uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $i \
            --limit 1 \
            --run-id "$RUN_ID" \
            --execution-mode "$EXECUTION_MODE" 2>&1 | tail -3
    else
        uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $i \
            --limit 1 \
            --run-id "$RUN_ID" \
            --execution-mode "$EXECUTION_MODE" \
            --append 2>&1 | tail -2
    fi
done

echo ""
echo "=============================================="
echo "Analyzing Results"
echo "=============================================="
uv run python analyze_results.py "$RUN_ID"

echo ""
echo "=============================================="
echo "Experiment Complete: $RUN_ID"
echo "Results: data/artifacts/$RUN_ID"
echo "=============================================="
