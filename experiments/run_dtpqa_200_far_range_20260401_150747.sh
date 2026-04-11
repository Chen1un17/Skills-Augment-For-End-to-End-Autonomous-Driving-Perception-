#!/bin/bash
# 200-Sample Far-Range Skill Generation Experiment
# Generated: 2026-04-01T15:07:47.169085

set -e

echo "=============================================="
echo "200-Sample Far-Range Skill Generation"
echo "Run ID: dtpqa_200_far_range_20260401_150747"
echo "=============================================="

# Clear proxy settings
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# Environment
export EDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="/tmp/dtpqa_far_range_skills_20260401_150747"
export ARTIFACTS_DIR="data/artifacts"

# Lower entropy threshold for far-range sensitivity
export UNCERTAINTY_ENTROPY_THRESHOLD="0.5"
export REQUEST_TIMEOUT_SECONDS="300"
export MAX_RETRIES="3"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"

# Enable DTPQA reflection
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

echo ""
echo "Configuration:"
echo "  - Samples: 200"
echo "  - Entropy threshold: 0.5 (lowered for far-range)"
echo "  - Skill store: /tmp/dtpqa_far_range_skills_20260401_150747"
echo "  - Far-range: Force reflection enabled"
echo ""

# Run in batches of 10 for checkpointing
BATCH_SIZE=10
TOTAL=200

for ((offset=0; offset<TOTAL; offset+=BATCH_SIZE)); do
    remaining=$((TOTAL - offset))
    current_batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))

    echo ""
    echo "[$(date '+%H:%M:%S')] Processing batch: offset=$offset, limit=$current_batch"

    uv run ad-replay-dtpqa \
        --subset synth \
        --question-type category_1 \
        --offset $offset \
        --limit $current_batch \
        --run-id dtpqa_200_far_range_20260401_150747 \
        --append || true

    echo "[$(date '+%H:%M:%S')] Batch complete. Progress: $((offset + current_batch))/$TOTAL"

    # Check for new skills
    skill_count=$(ls -1 /tmp/dtpqa_far_range_skills_20260401_150747/*.json 2>/dev/null | wc -l)
    echo "  Skills generated so far: $skill_count"

    # Small delay between batches
    sleep 2
done

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results: data/artifacts/dtpqa_200_far_range_20260401_150747/"
echo "Skills: /tmp/dtpqa_far_range_skills_20260401_150747/"
echo ""
echo "To evaluate: ./run_judge.sh dtpqa_200_far_range_20260401_150747"
echo "To analyze: python3 analyze_results.py dtpqa_200_far_range_20260401_150747"
