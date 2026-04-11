#!/bin/bash
# 200样本实验 - 高级实时监控
# 功能: 进度条、彩色指标、实时图表、详细统计

set -e

# ============================================
# 初始化
# ============================================
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

RUN_ID="dtpqa_200_advanced_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_advanced_skills_$(date +%s)"
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

LOG_FILE="${RUN_ID}_detailed.log"
DASHBOARD_FILE="/tmp/dashboard_${RUN_ID}.txt"

TOTAL=200
START_TIME=$(date +%s)

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# ============================================
# 工具函数
# ============================================
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

clear_screen() {
    printf "\033[2J\033[H"
}

# ============================================
# 获取实时统计
# ============================================
get_stats() {
    local pred_file="data/artifacts/$RUN_ID/predictions.jsonl"
    local stats=""
    
    if [ -f "$pred_file" ]; then
        local count=$(wc -l < "$pred_file" | tr -d ' ')
        local reflections=$(grep -c '"reflection_result":' "$pred_file" 2>/dev/null || echo "0")
        local skills=$(ls "$SKILL_STORE"/*.json 2>/dev/null | wc -l | tr -d ' ')
        
        # 简单准确率统计（如果有ground truth）
        local correct=$(grep -c '"is_correct": true' "$pred_file" 2>/dev/null || echo "0")
        
        stats="${count}|${reflections}|${skills}|${correct}"
    else
        stats="0|0|0|0"
    fi
    
    echo "$stats"
}

# ============================================
# 绘制ASCII图表
# ============================================
draw_mini_chart() {
    local value=$1
    local max=$2
    local width=20
    local filled=$((value * width / max))
    
    printf "["
    for ((i=0; i<filled; i++)); do printf "${GREEN}█${NC}"; done
    for ((i=filled; i<width; i++)); do printf "${RED}░${NC}"; done
    printf "]"
}

# ============================================
# 主仪表板
# ============================================
show_dashboard() {
    local current_batch=$1
    local batch_status=$2
    
    # 获取统计
    IFS='|' read -r pred_count reflection_count skill_count correct_count <<< "$(get_stats)"
    
    # 计算百分比
    local progress=$((pred_count * 100 / TOTAL))
    local reflection_rate=$((pred_count > 0 ? reflection_count * 100 / pred_count : 0))
    local accuracy=$((pred_count > 0 ? correct_count * 100 / pred_count : 0))
    
    # 时间计算
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    if [ $pred_count -gt 0 ]; then
        local avg_time=$((elapsed / pred_count))
        local remaining=$(((TOTAL - pred_count) * avg_time))
        local remain_min=$((remaining / 60))
        local remain_sec=$((remaining % 60))
    else
        local remain_min="??"
        local remain_sec="??"
    fi
    
    clear_screen
    
    # 标题
    printf "${CYAN}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${NC}\n"
    printf "${CYAN}┃${NC}  ${WHITE}🔬 边缘-云VLM自适应实验 - 实时监控仪表板${NC}                     ${CYAN}┃${NC}\n"
    printf "${CYAN}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${NC}\n"
    echo ""
    
    # 基本信息
    printf " ${BLUE}▸ 实验信息${NC}\n"
    printf "   Run ID:     ${CYAN}%s${NC}\n" "$RUN_ID"
    printf "   开始时间:   $(date -r $START_TIME '+%H:%M:%S')\n"
    printf "   已运行:     ${YELLOW}%02d:%02d${NC}\n" $elapsed_min $elapsed_sec
    printf "   预计剩余:   ${GREEN}%s分%s秒${NC}\n" "$remain_min" "$remain_sec"
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
    echo ""
    echo ""
    
    # 关键指标
    printf " ${BLUE}▸ 关键指标${NC}\n"
    printf "   ┌─────────────────────────────────────────────────────────┐\n"
    printf "   │ ${YELLOW}完成样本${NC}    │ ${GREEN}%3d${NC}/${WHITE}%d${NC}    │ " $pred_count $TOTAL
    draw_mini_chart $pred_count $TOTAL
    printf " │\n"
    printf "   │ ${YELLOW}云端反射${NC}    │ ${CYAN}%3d${NC}次      │ 触发率: ${MAGENTA}%d%%${NC}          │\n" $reflection_count $reflection_rate
    printf "   │ ${YELLOW}技能生成${NC}    │ ${YELLOW}%3d${NC}个      │ 新技能学习           │\n" $skill_count
    printf "   │ ${YELLOW}当前准确率${NC}  │ ${GREEN}%3d${NC}%%       │ 基于已完成样本       │\n" $accuracy
    printf "   └─────────────────────────────────────────────────────────┘\n"
    echo ""
    
    # 当前批次状态
    printf " ${BLUE}▸ 当前批次${NC}\n"
    if [ "$batch_status" = "processing" ]; then
        printf "   ${GREEN}● 正在处理${NC} Batch %d/40 (offset=%d)\n" $current_batch $((current_batch * 5))
        printf "   ${CYAN}⟳${NC} 等待API响应...\n"
    elif [ "$batch_status" = "complete" ]; then
        printf "   ${GREEN}✓ 已完成${NC} Batch %d/40\n" $current_batch
    elif [ "$batch_status" = "error" ]; then
        printf "   ${RED}✗ 错误${NC} Batch %d/40\n" $current_batch
    else
        printf "   ${BLUE}⏸ 等待中${NC}\n"
    fi
    echo ""
    
    # 最近日志
    printf " ${BLUE}▸ 最近日志${NC}\n"
    if [ -f "$LOG_FILE" ]; then
        tail -6 "$LOG_FILE" | sed 's/^/   /' | tail -5
    else
        printf "   (等待日志...)\n"
    fi
    echo ""
    
    # 底部提示
    printf "${CYAN}────────────────────────────────────────────────────────────────${NC}\n"
    printf "  ${WHITE}提示:${NC} 按 Ctrl+C 优雅停止实验 | 日志: ${LOG_FILE}\n"
    printf "${CYAN}────────────────────────────────────────────────────────────────${NC}\n"
}

# ============================================
# 运行实验批次
# ============================================
run_batch() {
    local offset=$1
    local batch_num=$2
    
    show_dashboard $batch_num "processing"
    log "Starting batch $batch_num: offset=$offset, limit=5"
    
    if uv run ad-replay-dtpqa \
        --subset synth \
        --question-type category_1 \
        --offset $offset \
        --limit 5 \
        --run-id $RUN_ID \
        --execution-mode hybrid \
        --append >> "$LOG_FILE" 2>&1; then
        log "✓ Batch $batch_num complete"
        show_dashboard $batch_num "complete"
    else
        log "✗ Batch $batch_num error"
        show_dashboard $batch_num "error"
    fi
}

# ============================================
# 最终报告
# ============================================
show_final_report() {
    clear_screen
    
    IFS='|' read -r pred_count reflection_count skill_count correct_count <<< "$(get_stats)"
    
    local now=$(date +%s)
    local total_time=$((now - START_TIME))
    local total_min=$((total_time / 60))
    
    printf "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${GREEN}║${NC}                     🎉 实验完成!                               ${GREEN}║${NC}\n"
    printf "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""
    printf " ${CYAN}Run ID:${NC}        $RUN_ID\n"
    printf " ${CYAN}总耗时:${NC}        ${YELLOW}%d分钟${NC}\n" $total_min
    printf " ${CYAN}完成样本:${NC}      ${GREEN}%d/%d${NC}\n" $pred_count $TOTAL
    printf " ${CYAN}云端反射:${NC}      ${CYAN}%d次${NC} (%d%%)\n" $reflection_count $((pred_count > 0 ? reflection_count * 100 / pred_count : 0))
    printf " ${CYAN}技能生成:${NC}      ${YELLOW}%d个${NC}\n" $skill_count
    echo ""
    printf " ${BLUE}结果位置:${NC}\n"
    printf "   预测结果: data/artifacts/$RUN_ID/predictions.jsonl\n"
    printf "   技能目录: $SKILL_STORE\n"
    printf "   详细日志: $LOG_FILE\n"
    echo ""
    printf " ${BLUE}下一步:${NC}\n"
    printf "   1. 分析结果: python3 analyze_results.py $RUN_ID\n"
    printf "   2. 法官评估: ./run_judge.sh $RUN_ID\n"
    printf "   3. 查看技能: ls $SKILL_STORE\n"
    echo ""
}

# ============================================
# 主函数
# ============================================
main() {
    log "实验启动: $RUN_ID"
    
    # 显示初始界面
    show_dashboard 0 "waiting"
    sleep 2
    
    # 运行所有批次
    local batch_num=1
    for ((offset=0; offset<TOTAL; offset+=5)); do
        run_batch $offset $batch_num
        batch_num=$((batch_num + 1))
        sleep 1
    done
    
    # 显示最终报告
    show_final_report
    log "实验完成"
}

# 捕获中断信号
trap 'echo ""; echo "${YELLOW}正在停止实验...${NC}"; exit 0' INT

# 启动
main
