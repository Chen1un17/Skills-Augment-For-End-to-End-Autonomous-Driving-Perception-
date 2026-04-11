#!/bin/bash
# 200-Sample Adaptive Experiment - Fixed

set -e

RUN_ID="dtpqa_200_adaptive_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_adaptive_skills_$(date +%Y%m%d%H%M%S)"

echo "=============================================="
echo "200-Sample Adaptive Experiment"
echo "Run ID: ${RUN_ID}"
echo "Started: $(date)"
echo "=============================================="

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="${SKILL_STORE}"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export REQUEST_TIMEOUT_SECONDS="300"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

mkdir -p "${SKILL_STORE}"

echo ""
echo "Configuration:"
echo "  Entropy threshold: 0.4"
echo "  Far-range: Smart reflection (not forced)"
echo "  Batch size: 5"
echo "  Total: 200 samples"
echo ""

# Process 200 samples in batches
offset=0
batch_size=5
total=200
batch_num=1

while [ $offset -lt $total ]; do
    remaining=$((total - offset))
    if [ $remaining -lt $batch_size ]; then
        batch=$remaining
    else
        batch=$batch_size
    fi
    
    progress=$((offset * 100 / total))
    
    echo ""
    echo "[$(date '+%H:%M:%S')] =========================================="
    echo "[$(date '+%H:%M:%S')] Batch ${batch_num}/40 | Offset: ${offset} | Progress: ${progress}%"
    
    # Run the batch
    uv run ad-replay-dtpqa \
        --subset synth \
        --question-type category_1 \
        --offset ${offset} \
        --limit ${batch} \
        --run-id ${RUN_ID} \
        --execution-mode hybrid \
        --append || echo "Batch had errors, continuing..."
    
    echo "[$(date '+%H:%M:%S')] ✓ Batch ${batch_num} complete"
    
    # Count skills
    skill_count=$(ls "${SKILL_STORE}"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] Skills generated: ${skill_count}"
    
    # Move to next batch
    offset=$((offset + batch))
    batch_num=$((batch_num + 1))
    
    sleep 1
done

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE!"
echo "=============================================="
echo "Run ID: ${RUN_ID}"
echo "Skills: ${SKILL_STORE}"
echo "Artifacts: data/artifacts/${RUN_ID}/"
echo "Completed: $(date)"
echo "=============================================="
