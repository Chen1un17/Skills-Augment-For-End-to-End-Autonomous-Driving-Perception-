#!/bin/bash
# 200-Sample Adaptive Experiment - Smart Reflection
# Auto Research: Model learns when to ask for help

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="dtpqa_200_adaptive_${TIMESTAMP}"
SKILL_STORE="/tmp/dtpqa_adaptive_skills_${TIMESTAMP}"

echo "=============================================="
echo "200-Sample Adaptive Skill Learning"
echo "Run ID: ${RUN_ID}"
echo "=============================================="

# Clear proxy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# Environment
export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="${SKILL_STORE}"
export ARTIFACTS_DIR="data/artifacts"

# ADAPTIVE SETTINGS - Lower thresholds for learning
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export REQUEST_TIMEOUT_SECONDS="300"
export MAX_RETRIES="3"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

mkdir -p "${SKILL_STORE}"

echo ""
echo "Adaptive Policy Configuration:"
echo "  Entropy threshold: 0.4 (sensitive)"
echo "  Far-range strategy: Smart (NOT forced)"
echo "  Goal: Model learns its own uncertainty"
echo "  Expected skills: 15-30"
echo ""

# Stats tracking
TOTAL=200
BATCH_SIZE=5
SKILL_COUNT=0
START_TIME=$(date +%s)

for ((offset=0; offset<TOTAL; offset+=BATCH_SIZE)); do
    remaining=$((TOTAL - offset))
    batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))
    
    echo ""
    echo "[$(date '+%H:%M:%S')] =========================================="
    echo "[$(date '+%H:%M:%S')] Batch $((offset/BATCH_SIZE + 1))/40"
    echo "[$(date '+%H:%M:%S')] Offset: ${offset}, Limit: ${batch}"
    echo "[$(date '+%H:%M:%S')] Progress: ${offset}/${TOTAL} ($((${offset * 100 / TOTAL}))%)"
    
    # Run batch
    if uv run ad-replay-dtpqa \
        --subset synth \
        --question-type category_1 \
        --offset ${offset} \
        --limit ${batch} \
        --run-id ${RUN_ID} \
        --execution-mode hybrid \
        --append 2>&1; then
        echo "[$(date '+%H:%M:%S')] ✓ Batch complete"
    else
        echo "[$(date '+%H:%M:%S')] ⚠ Batch had errors, continuing..."
    fi
    
    # Check skills
    NEW_COUNT=$(ls "${SKILL_STORE}"/*.json 2>/dev/null | wc -l)
    if [ "$NEW_COUNT" -ne "$SKILL_COUNT" ]; then
        echo "[$(date '+%H:%M:%S')] 📚 New skills: ${SKILL_COUNT} → ${NEW_COUNT}"
        SKILL_COUNT=$NEW_COUNT
    fi
    
    # Estimate remaining time
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ $offset -gt 0 ]; then
        RATE=$((ELAPSED * 1000 / offset))  # ms per sample
        REMAINING=$(( (TOTAL - offset) * RATE / 1000 ))
        MINS=$((REMAINING / 60))
        echo "[$(date '+%H:%M:%S')] ⏱  Estimated remaining: ~${MINS} min"
    fi
    
    sleep 1
done

echo ""
echo "=============================================="
echo "EXPERIMENT COMPLETE"
echo "=============================================="
echo "Run ID: ${RUN_ID}"
echo "Skills generated: ${SKILL_COUNT}"
echo "Skill store: ${SKILL_STORE}"
echo "Artifacts: data/artifacts/${RUN_ID}/"
echo ""
echo "Next steps:"
echo "  1. Run judge:  ./run_judge.sh ${RUN_ID}"
echo "  2. Analyze:    python3 analyze_results.py ${RUN_ID}"
echo "  3. View skills: ls ${SKILL_STORE}/"
echo "=============================================="
