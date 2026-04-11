#!/bin/bash
# 200样本实验 - 修复超时版本
# 确保客户端和服务器都使用300秒超时

set -e

echo "=============================================="
echo "🚀 200样本实验（修复超时版）"
echo "=============================================="

# 1. 彻底清除代理
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# 2. 设置超时（客户端和服务器都需要）
export REQUEST_TIMEOUT_SECONDS="300"  # 5分钟，与服务器一致

# 3. 检查MCP服务器
echo ""
echo "[检查] MCP服务器..."
if ! curl -s http://127.0.0.1:8000/mcp -X POST -d '{}' > /dev/null 2>&1; then
    echo "✗ MCP服务器未运行"
    echo "请先运行: ./fix_mcp_and_restart.sh"
    exit 1
fi
echo "✓ MCP服务器正常"

# 4. 配置实验
RUN_ID="dtpqa_200_final_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_final_skills_$(date +%s)"
mkdir -p "$SKILL_STORE"

export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="$SKILL_STORE"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8000"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"
export MAX_RETRIES="3"

LOG_FILE="${RUN_ID}.log"
TOTAL=200
START_TIME=$(date +%s)

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

clear_screen() {
    printf "\033[2J\033[H"
}

calc_percent() {
    local part=$1
    local total=$2
    if [ "$total" -gt 0 ]; then
        echo $((part * 100 / total))
    else
        echo "0"
    fi
}

show_dashboard() {
    local current=$1
    local status=$2
    
    IFS='|' read -r pred_count reflection_count skill_count <<< "$(get_stats)"
    local progress=$(calc_percent $pred_count $TOTAL)
    local reflection_rate=$(calc_percent $reflection_count $pred_count)
    
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    local remain_min="??"
    local remain_sec="??"
    if [ "$pred_count" -gt 0 ]; then
        local avg_time=$((elapsed / pred_count))
        local remaining=$(((TOTAL - pred_count) * avg_time))
        remain_min=$((remaining / 60))
        remain_sec=$((remaining % 60))
    fi
    
    clear_screen
    printf "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${CYAN}║${NC}     🚀 200样本实验（修复超时版）- 实时监控                   ${CYAN}║${NC}\n"
    printf "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""
    printf " ${BLUE}配置:${NC} 超时=${YELLOW}300秒${NC} | 阈值=${YELLOW}0.4${NC} | 反射=${YELLOW}启用${NC}\n"
    echo ""
    printf " ${BLUE}▸ 实验信息${NC}\n"
    printf "   Run ID:   ${CYAN}%s${NC}\n" "$RUN_ID"
    printf "   已运行:   ${YELLOW}%02d:%02d${NC}\n" $elapsed_min $elapsed_sec
    printf "   预计剩余: ${GREEN}%s分%s秒${NC}\n" "$remain_min" "$remain_sec"
    echo ""
    printf " ${BLUE}▸ 进度${NC} ${WHITE}%d/%d (%d%%)${NC}\n" $pred_count $TOTAL $progress
    printf "   "
    local filled=$((progress / 2))
    for ((i=0; i<50; i++)); do
        if [ $i -lt $filled ]; then printf "${GREEN}█${NC}"; else printf "${RED}░${NC}"; fi
    done
    printf "\n\n"
    printf " ${BLUE}▸ 指标${NC}\n"
    printf "   完成样本: ${GREEN}%d${NC} | 云端反射: ${CYAN}%d${NC} (%d%%) | 技能: ${YELLOW}%d${NC}\n" $pred_count $reflection_count $reflection_rate $skill_count
    echo ""
    printf " ${BLUE}▸ 状态${NC}\n"
    printf "   %s\n" "$status"
    echo ""
    printf "${CYAN}────────────────────────────────────────────────────────────────${NC}\n"
    printf "   按 Ctrl+C 停止 | 日志: ${LOG_FILE}\n"
}

get_stats() {
    local pred_file="data/artifacts/$RUN_ID/predictions.jsonl"
    local count=0
    local reflections=0
    local skills=0
    
    if [ -f "$pred_file" ]; then
        count=$(wc -l < "$pred_file" | tr -d ' ')
        reflections=$(grep -c '"reflection_result":' "$pred_file" 2>/dev/null || echo "0")
    fi
    skills=$(ls "$SKILL_STORE"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "${count}|${reflections}|${skills}"
}

# 主循环
main() {
    log "实验启动: $RUN_ID (超时=300s)"
    
    for ((offset=0; offset<TOTAL; offset+=5)); do
        show_dashboard $offset "运行中: offset=$offset"
        log "Batch: offset=$offset"
        
        if uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $offset \
            --limit 5 \
            --run-id $RUN_ID \
            --execution-mode hybrid \
            --append >> "$LOG_FILE" 2>&1; then
            log "✓ Batch complete"
        else
            log "✗ Batch error"
        fi
        sleep 1
    done
    
    show_dashboard $TOTAL "完成!"
    echo ""
    echo "✅ 实验完成!"
    echo "Run ID: $RUN_ID"
    echo "分析: python3 analyze_results.py $RUN_ID"
}

trap 'echo ""; exit 0' INT
main
