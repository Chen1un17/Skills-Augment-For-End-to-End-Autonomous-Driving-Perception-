#!/bin/bash
# 修复代理问题的实验脚本

# 1. 清除代理
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

echo "代理已清除"

# 2. 环境设置
export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="/tmp/dtpqa_fixed_$(date +%s)"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export REQUEST_TIMEOUT_SECONDS="180"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8000"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

mkdir -p "$SKILL_STORE_DIR"

RUN_ID="dtpqa_200_fixed_$(date +%Y%m%d_%H%M%S)"

echo "Run ID: $RUN_ID"
echo "开始实验..."

# 3. 运行200样本
for offset in 0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195; do
    echo "[$(date +%H:%M:%S)] Processing offset=$offset"
    
    uv run ad-replay-dtpqa \
        --subset synth \
        --question-type category_1 \
        --offset $offset \
        --limit 5 \
        --run-id $RUN_ID \
        --execution-mode hybrid \
        --append 2>&1 | tail -5
    
    # 显示进度
    if [ -f "data/artifacts/$RUN_ID/predictions.jsonl" ]; then
        count=$(wc -l < "data/artifacts/$RUN_ID/predictions.jsonl")
        echo "  Progress: $count/200"
    fi
done

echo "Experiment complete: $RUN_ID"
