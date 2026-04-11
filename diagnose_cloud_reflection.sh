#!/bin/bash
# 云端反射问题诊断脚本

echo "=============================================="
echo "🔍 云端反射问题诊断"
echo "=============================================="

# 1. 检查MCP服务器状态
echo ""
echo "[1/5] 检查MCP服务器..."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

if curl -s http://127.0.0.1:8000/mcp -X POST -d '{}' > /dev/null 2>&1; then
    echo "✓ MCP服务器端口8000响应正常"
else
    echo "✗ MCP服务器端口8000无响应"
fi

# 2. 检查环境变量
echo ""
echo "[2/5] 检查环境变量..."
echo "  REQUEST_TIMEOUT_SECONDS: ${REQUEST_TIMEOUT_SECONDS:-未设置}"
echo "  MCP_SERVER_HOST: ${MCP_SERVER_HOST:-未设置}"
echo "  MCP_SERVER_PORT: ${MCP_SERVER_PORT:-未设置}"
echo "  ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER: ${ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER:-未设置}"

# 3. 测试单样本（带详细日志）
echo ""
echo "[3/5] 测试单样本反射（预计3-5分钟）..."
export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="/tmp/diagnose_skills"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.1"  # 极低阈值，强制反射
export REQUEST_TIMEOUT_SECONDS="300"  # 5分钟
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8000"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

mkdir -p "$SKILL_STORE_DIR"

RUN_ID="diagnose_$(date +%s)"

echo "  Run ID: $RUN_ID"
echo "  测试样本: offset=0, limit=1"
echo "  阈值: 0.1 (强制反射)"
echo "  超时: 300秒"
echo ""
echo "  ⏳ 正在运行，请等待..."

# 运行并记录时间
START_TIME=$(date +%s)
uv run ad-replay-dtpqa \
    --subset synth \
    --question-type category_1 \
    --offset 0 \
    --limit 1 \
    --run-id $RUN_ID \
    --execution-mode hybrid \
    2>&1 | tee /tmp/diagnose_log.txt | grep -E "(Running|Finished|ERROR|timeout|reflection)" | tail -20

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "  耗时: ${DURATION}秒"

# 4. 分析结果
echo ""
echo "[4/5] 分析结果..."
PRED_FILE="data/artifacts/$RUN_ID/predictions.jsonl"
if [ -f "$PRED_FILE" ]; then
    echo "✓ 预测文件存在"
    # 检查是否有反射
    if grep -q '"reflection_result":' "$PRED_FILE"; then
        echo "✓ 反射成功执行"
        grep '"reflection_result":' "$PRED_FILE" | head -1
    else
        echo "✗ 未找到反射结果"
    fi
    
    # 显示文件内容
    echo ""
    echo "  预测内容预览:"
    head -c 500 "$PRED_FILE" | jq . 2>/dev/null || head -c 500 "$PRED_FILE"
else
    echo "✗ 预测文件不存在"
fi

# 5. 检查日志
echo ""
echo "[5/5] 错误分析..."
if grep -q "timeout" /tmp/diagnose_log.txt 2>/dev/null; then
    echo "⚠ 发现超时错误:"
    grep "timeout" /tmp/diagnose_log.txt | tail -3
elif grep -q "ERROR" /tmp/diagnose_log.txt 2>/dev/null; then
    echo "⚠ 发现错误:"
    grep "ERROR" /tmp/diagnose_log.txt | tail -3
else
    echo "✓ 未发现明显错误"
fi

echo ""
echo "=============================================="
echo "诊断完成"
echo "详细日志: /tmp/diagnose_log.txt"
echo "Run ID: $RUN_ID"
echo "=============================================="
