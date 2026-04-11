#!/bin/bash
# 200样本实验 - 纯边缘模式（禁用反射，避免超时）
# 策略: 先测试边缘准确率，再针对性优化

set -e

# 初始化
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

RUN_ID="dtpqa_200_edge_only_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_edge_skills_$(date +%s)"
mkdir -p "$SKILL_STORE"

export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="$SKILL_STORE"
export ARTIFACTS_DIR="data/artifacts"
# 禁用反射 - 避免超时问题
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="0"
export UNCERTAINTY_ENTROPY_THRESHOLD="999"  # 设置极高，永不触发

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

show_dashboard() {
    local current=$1
    local total=$2
    local pred_file="data/artifacts/$RUN_ID/predictions.jsonl"
    
    local count=0
    if [ -f "$pred_file" ]; then
        count=$(wc -l < "$pred_file" | tr -d ' ')
    fi
    
    local progress=$((count * 100 / total))
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local elapsed_min=$((elapsed / 60))
    
    clear_screen
    printf "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${CYAN}║${NC}     🔬 200样本实验 - 纯边缘模式（无反射，避免超时）          ${CYAN}║${NC}\n"
    printf "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""
    printf " ${BLUE}Run ID:${NC}    $RUN_ID\n"
    printf " ${BLUE}策略:${NC}      纯边缘推理（禁用云端反射）\n"
    printf " ${BLUE}目的:${NC}      测试基线准确率，避免超时\n"
    echo ""
    printf " ${YELLOW}进度:${NC}      ${GREEN}%d/%d${NC} (%d%%)\n" $count $total $progress
    printf "   "
    local filled=$((progress / 2))
    for ((i=0; i<50; i++)); do
        if [ $i -lt $filled ]; then printf "${GREEN}█${NC}"; else printf "${RED}░${NC}"; fi
    done
    printf "\n\n"
    printf " ${YELLOW}已运行:${NC}    %02d分钟\n" $elapsed_min
    printf " ${YELLOW}预计剩余:${NC}  ~%d分钟\n" $(((total - count) * elapsed / count / 60))
    echo ""
    printf " ${CYAN}────────────────────────────────────────────────────────────${NC}\n"
    printf "   按 Ctrl+C 停止\n"
}

# 主循环
main() {
    log "实验启动: $RUN_ID (纯边缘模式)"
    
    for ((offset=0; offset<TOTAL; offset+=5)); do
        show_dashboard $offset $TOTAL
        log "Starting batch: offset=$offset"
        
        if uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $offset \
            --limit 5 \
            --run-id $RUN_ID \
            --execution-mode edge_only \
            --append >> "$LOG_FILE" 2>&1; then
            log "✓ Batch complete: offset=$offset"
        else
            log "✗ Batch error: offset=$offset"
        fi
        sleep 1
    done
    
    show_dashboard $TOTAL $TOTAL
    echo ""
    echo "✅ 实验完成!"
    echo "Run ID: $RUN_ID"
    echo "结果: data/artifacts/$RUN_ID/"
}

trap 'echo ""; exit 0' INT
main
