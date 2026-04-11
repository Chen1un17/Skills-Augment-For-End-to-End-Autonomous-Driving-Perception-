#!/bin/bash
# 修复代理问题并重新运行实验

echo "=============================================="
echo "修复代理问题并重启实验"
echo "=============================================="

# 1. 彻底清除所有代理设置
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
export NO_PROXY="127.0.0.1,localhost,*.siliconflow.cn"

echo ""
echo "✓ 代理已清除:"
echo "  http_proxy: [$http_proxy]"
echo "  https_proxy: [$https_proxy]"
echo "  all_proxy: [$all_proxy]"

# 2. 测试直接连接（不走代理）
echo ""
echo "--- 测试API直接连接 ---"
START_TIME=$(date +%s.%N)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --connect-timeout 10 \
    --max-time 30 \
    https://api.siliconflow.cn/v1/models \
    -H "Authorization: Bearer ${SILICONFLOW_API_KEY}" 2>/dev/null)
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "0")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "401" ]; then
    echo "✓ API连接正常 (HTTP $HTTP_CODE, ${ELAPSED}s)"
else
    echo "✗ API连接失败 (HTTP $HTTP_CODE)"
fi

# 3. 创建新的实验脚本（彻底无代理）
RUN_ID="dtpqa_200_fixed_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_fixed_skills_$(date +%Y%m%d%H%M%S)"

echo ""
echo "--- 启动新实验 ---"
echo "Run ID: $RUN_ID"
echo "策略: 无代理直连 + 自适应反射"
echo ""

# 设置环境（无代理）
export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="$SKILL_STORE"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export REQUEST_TIMEOUT_SECONDS="180"  # 增加到3分钟
export MAX_RETRIES="3"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

mkdir -p "$SKILL_STORE"

# 运行批次（带详细日志）
BATCH_SIZE=5
TOTAL=200

for ((offset=0; offset<TOTAL; offset+=BATCH_SIZE)); do
    remaining=$((TOTAL - offset))
    batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))
    progress=$((offset * 100 / TOTAL))
    
    echo "[$(date '+%H:%M:%S')] Batch $((offset/BATCH_SIZE + 1))/40 | Offset: $offset | $progress%"
    
    # 运行并计时
    BATCH_START=$(date +%s)
    
    if uv run ad-replay-dtpqa \
        --subset synth \
        --question-type category_1 \
        --offset $offset \
        --limit $batch \
        --run-id $RUN_ID \
        --execution-mode hybrid \
        --append 2>&1; then
        
        BATCH_END=$(date +%s)
        BATCH_TIME=$((BATCH_END - BATCH_START))
        echo "  ✓ 完成 (${BATCH_TIME}s)"
    else
        echo "  ✗ 失败，继续..."
    fi
    
    # 统计
    skill_count=$(ls "$SKILL_STORE"/*.json 2>/dev/null | wc -l)
    pred_count=$(wc -l < "data/artifacts/$RUN_ID/predictions.jsonl" 2>/dev/null || echo "0")
    echo "  进度: $pred_count/$TOTAL 样本 | $skill_count 技能"
    
    sleep 1
done

echo ""
echo "=============================================="
echo "实验完成!"
echo "Run ID: $RUN_ID"
echo "技能: $SKILL_STORE"
echo "=============================================="
