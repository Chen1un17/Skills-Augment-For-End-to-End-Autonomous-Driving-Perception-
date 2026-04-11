#!/bin/bash
# Batch runner for DTPQA synth experiments

set -e

cd "$(dirname "$0")"

# Clear proxy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

RUN_ID="dtpqa_synth_baseline_$(date +%Y%m%d_%H%M%S)"
LIMIT="${1:-50}"
export EDGE_MODEL="${EDGE_MODEL:-Qwen/Qwen3.5-9B}"
export CLOUD_MODEL="${CLOUD_MODEL:-Pro/moonshotai/Kimi-K2.5}"
export JUDGE_MODEL="${JUDGE_MODEL:-Pro/moonshotai/Kimi-K2.5}"

echo "=============================================="
echo "Running DTPQA Synth Baseline Experiment"
echo "Run ID: $RUN_ID"
echo "Limit: $LIMIT samples"
echo "=============================================="
echo ""

# Clean skill store
rm -rf /tmp/dtpqa_skills_empty/*
mkdir -p /tmp/dtpqa_skills_empty

# Run samples one by one
for ((i=0; i<$LIMIT; i++)); do
    echo ""
    echo "[$(date '+%H:%M:%S')] Running sample $i/$LIMIT"

    if [ $i -eq 0 ]; then
        # First sample - no append
        uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $i \
            --limit 1 \
            --run-id $RUN_ID \
            --execution-mode edge_only 2>&1 | tail -5
    else
        # Subsequent samples - append
        uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $i \
            --limit 1 \
            --run-id $RUN_ID \
            --execution-mode edge_only \
            --append 2>&1 | tail -3
    fi

    sleep 1
done

echo ""
echo "=============================================="
echo "Running Judge Evaluation"
echo "=============================================="
uv run ad-eval-dtpqa --run-id $RUN_ID

echo ""
echo "=============================================="
echo "Experiment Complete: $RUN_ID"
echo "=============================================="
echo "Results in: data/artifacts/$RUN_ID"
