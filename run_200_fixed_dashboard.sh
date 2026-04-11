#!/bin/bash
# 200样本实验 - 修复版实时监控

set -e

# 初始化
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

RUN_ID="dtpqa_200_fixed_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_fixed_skills_$(date +%s)"
mkdir -p "$SKILL_STORE"

export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="$SKILL_STORE"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export REQUEST_TIMEOUT_SECONDS="180"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8000"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

LOG_FILE="${RUN_ID}.log"
TOTAL=200
START_TIME=$(date +%s)

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

clear_screen() {
    printf "\033[2J\033[H"
}

# 获取统计
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

# 计算百分比（修复除零错误）
calc_percent() {
    local part=$1
    local total=$2
    if [ "$total" -gt 0 ]; then
        echo $((part * 100 / total))
    else
        echo "0"
    fi
}

# 显示仪表板
show_dashboard() {
    local current_batch=$1
    local status=$2
    
    # 获取统计
    IFS='|' read -r pred_count reflection_count skill_count <<< "$(get_stats)"
    
    # 计算百分比（使用修复后的函数）
    local progress=$(calc_percent $pred_count $TOTAL)
    local reflection_rate=$(calc_percent $reflection_count $pred_count)
    
    # 时间计算
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    # 计算剩余时间
    local remain_min="??"
    local remain_sec="??"
    if [ "$pred_count" -gt 0 ]; then
        local avg_time=$((elapsed / pred_count))
        local remaining=$(((TOTAL - pred_count) * avg_time))
        remain_min=$((remaining / 60))
        remain_sec=$((remaining % 60))
    fi
    
    clear_screen
    
    # 标题
    printf "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${CYAN}║${NC}     🔬 边缘-云VLM自适应实验 - 实时监控仪表板                 ${CYAN}║${NC}\n"
    printf "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""
    
    # 信息
    printf " ${BLUE}▸ 实验信息${NC}\n"
    printf "   Run ID:   ${CYAN}%s${NC}\n" "$RUN_ID"
    printf "   开始时间: $(date -r $START_TIME '+%H:%M:%S')\n"
    printf "   已运行:   ${YELLOW}%02d:%02d${NC}\n" $elapsed_min $elapsed_sec
    printf "   预计剩余: ${GREEN}%s分%s秒${NC}\n" "$remain_min" "$remain_sec"
    echo ""
    
    # 进度条
    printf " ${BLUE}▸ 总体进度${NC} ${WHITE}%d/%d (%d%%)${NC}\n" $pred_count $TOTAL $progress
    printf "   "
    local filled=$((progress / 2))
    for ((i=0; i<50; i++)); do
        if [ $i -lt $filled ]; then
            printf "${GREEN}█${NC}"
        else
            printf "${RED}░${NC}"
        fi
    done
    printf "\n\n"
    
    # 关键指标
    printf " ${BLUE}▸ 关键指标${NC}\n"
    printf "   ┌────────────────────────────────────────────────────┐\n"
    printf "   │ ${YELLOW}完成样本${NC}  │ ${GREEN}%3d${NC}/${WHITE}%d${NC}  │ 进度: %d%%          │\n" $pred_count $TOTAL $progress
    printf "   │ ${YELLOW}云端反射${NC}  │ ${CYAN}%3d${NC}次    │ 触发率: ${MAGENTA}%d%%${NC}        │\n" $reflection_count $reflection_rate
    printf "   │ ${YELLOW}技能生成${NC}  │ ${YELLOW}%3d${NC}个    │                     │\n" $skill_count
    printf "   └────────────────────────────────────────────────────┘\n"
    echo ""
    
    # 当前状态
    printf " ${BLUE}▸ 当前状态${NC}\n"
    case "$status" in
        "running")
            printf "   ${GREEN}● 运行中${NC} Batch %d/40\n" $current_batch
            ;;
        "complete")
            printf "   ${GREEN}✓ 完成${NC} Batch %d/40\n" $current_batch
            ;;
        "error")
            printf "   ${RED}✗ 错误${NC} Batch %d/40\n" $current_batch
            ;;
        *)
            printf "   ${BLUE}⏸ 等待中${NC}\n"
            ;;
    esac
    echo ""
    
    # 最近日志
    printf " ${BLUE}▸ 最近日志${NC}\n"
    if [ -f "$LOG_FILE" ]; then
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/   /'
    else
        printf "   (等待日志...)\n"
    fi
    echo ""
    
    printf "${CYAN}────────────────────────────────────────────────────────────────${NC}\n"
    printf "  按 Ctrl+C 停止实验 | 日志: ${LOG_FILE}\n"
    printf "${CYAN}────────────────────────────────────────────────────────────────${NC}\n"
}

# 最终报告
show_final() {
    clear_screen
    IFS='|' read -r pred_count reflection_count skill_count <<< "$(get_stats)"
    local now=$(date +%s)
    local total_min=$(((now - START_TIME) / 60))
    
    printf "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${GREEN}║${NC}                     🎉 实验完成!                             ${GREEN}║${NC}\n"
    printf "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""
    printf " ${CYAN}Run ID:${NC}      $RUN_ID\n"
    printf " ${CYAN}总耗时:${NC}      ${YELLOW}%d分钟${NC}\n" $total_min
    printf " ${CYAN}完成样本:${NC}    ${GREEN}%d/%d${NC}\n" $pred_count $TOTAL
    printf " ${CYAN}云端反射:${NC}    ${CYAN}%d次${NC} (%d%%)\n" $reflection_count $(calc_percent $reflection_count $pred_count)
    printf " ${CYAN}技能生成:${NC}    ${YELLOW}%d个${NC}\n" $skill_count
    echo ""
    printf " ${BLUE}结果位置:${NC}\n"
    printf "   预测结果: data/artifacts/$RUN_ID/predictions.jsonl\n"
    printf "   技能目录: $SKILL_STORE\n"
    printf "   详细日志: $LOG_FILE\n"
    echo ""
}

# 主函数
main() {
    log "实验启动: $RUN_ID"
    show_dashboard 0 "waiting"
    sleep 2
    
    local batch_num=1
    for ((offset=0; offset<TOTAL; offset+=5)); do
        show_dashboard $batch_num "running"
        log "Starting batch $batch_num: offset=$offset"
        
        if uv run ad-replay-dtpqa \
            --subset synth \
            --question-type category_1 \
            --offset $offset \
            --limit 5 \
            --run-id $RUN_ID \
            --execution-mode hybrid \
            --append >> "$LOG_FILE" 2>&1; then
            log "✓ Batch $batch_num complete"
        else
            log "✗ Batch $batch_num error"
        fi
        
        batch_num=$((batch_num + 1))
        sleep 1
    done
    
    show_final
    log "实验完成"
}

trap 'echo ""; echo "${YELLOW}停止实验...${NC}"; exit 0' INT

main
