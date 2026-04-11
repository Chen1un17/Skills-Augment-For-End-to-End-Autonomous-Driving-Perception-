#!/bin/bash
# 200样本实验 - 实时监控版
# 带进度条、实时指标、彩色输出

# ============================================
# 设置
# ============================================
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

RUN_ID="dtpqa_200_live_$(date +%Y%m%d_%H%M%S)"
SKILL_STORE="/tmp/dtpqa_live_skills_$(date +%s)"
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

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

TOTAL=200
BATCH_SIZE=5
START_TIME=$(date +%s)

# ============================================
# 清屏函数
# ============================================
clear_screen() {
    printf "\033[2J\033[H"
}

# ============================================
# 绘制进度条
# ============================================
draw_progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "]"
}

# ============================================
# 显示仪表板
# ============================================
show_dashboard() {
    local current_offset=$1
    local status=$2
    local message=$3
    
    clear_screen
    
    # 计算统计
    local completed=$((current_offset > TOTAL ? TOTAL : current_offset))
    local progress=$((completed * 100 / TOTAL))
    
    # 从结果文件读取统计
    local pred_file="data/artifacts/$RUN_ID/predictions.jsonl"
    local pred_count=0
    local reflection_count=0
    local skill_count=0
    
    if [ -f "$pred_file" ]; then
        pred_count=$(wc -l < "$pred_file" | tr -d ' ')
        # 简单统计反射（如果有jq会更快，但用grep兼容）
        reflection_count=$(grep -c '"reflection_result":' "$pred_file" 2>/dev/null || echo "0")
    fi
    
    skill_count=$(ls "$SKILL_STORE"/*.json 2>/dev/null | wc -l | tr -d ' ')
    
    # 计算时间
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local elapsed_min=$((elapsed / 60))
    local elapsed_sec=$((elapsed % 60))
    
    if [ $completed -gt 0 ]; then
        local rate=$((elapsed / completed))
        local remaining=$(((TOTAL - completed) * rate))
        local remaining_min=$((remaining / 60))
        local remaining_sec=$((remaining % 60))
    else
        local remaining_min="--"
        local remaining_sec="--"
    fi
    
    # 显示仪表板
    echo ""
    printf "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${CYAN}║${NC}               🚀 200样本自适应实验 - 实时监控                   ${CYAN}║${NC}\n"
    printf "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
    echo ""
    
    printf " ${BLUE}Run ID:${NC}     $RUN_ID\n"
    printf " ${BLUE}开始时间:${NC}   $(date -r $START_TIME '+%H:%M:%S')\n"
    printf " ${BLUE}当前时间:${NC}   $(date '+%H:%M:%S')\n"
    echo ""
    
    printf " ${YELLOW}▶ 总体进度${NC}\n"
    printf "   "
    draw_progress_bar $completed $TOTAL
    printf " ${GREEN}%d/%d (%d%%)${NC}\n" $completed $TOTAL $progress
    echo ""
    
    printf " ${YELLOW}▶ 实时指标${NC}\n"
    printf "   ├─ 完成样本:    ${GREEN}%d${NC}\n" $pred_count
    printf "   ├─ 云端反射:    ${CYAN}%d${NC} (触发率: %s%%)\n" $reflection_count "$(($pred_count > 0 ? reflection_count * 100 / pred_count : 0))"
    printf "   ├─ 技能生成:    ${YELLOW}%d${NC}\n" $skill_count
    printf "   └─ 当前批次:    ${BLUE}Offset %d${NC}\n" $current_offset
    echo ""
    
    printf " ${YELLOW}▶ 时间统计${NC}\n"
    printf "   ├─ 已运行:      %02d:%02d\n" $elapsed_min $elapsed_sec
    printf "   └─ 预计剩余:    %02d:%02d\n" $remaining_min $remaining_sec
    echo ""
    
    printf " ${YELLOW}▶ 当前状态${NC}\n"
    if [ "$status" = "running" ]; then
        printf "   ${GREEN}●${NC} %s\n" "$message"
    elif [ "$status" = "error" ]; then
        printf "   ${RED}●${NC} %s\n" "$message"
    else
        printf "   ${BLUE}●${NC} %s\n" "$message"
    fi
    echo ""
    
    printf "${CYAN}════════════════════════════════════════════════════════════════════${NC}\n"
    printf " 最近日志 | Ctrl+C 停止实验 | 日志保存到: ${RUN_ID}.log\n"
    printf "${CYAN}════════════════════════════════════════════════════════════════════${NC}\n"
}

# ============================================
# 显示最近日志
# ============================================
show_recent_logs() {
    local log_file="${RUN_ID}.log"
    if [ -f "$log_file" ]; then
        echo ""
        tail -8 "$log_file" | sed 's/^/  /'
    fi
}

# ============================================
# 主循环
# ============================================
main() {
    # 清屏并显示初始界面
    clear_screen
    show_dashboard 0 "ready" "准备启动实验..."
    sleep 2
    
    # 启动后台实验进程并将输出重定向到日志
    {
        for ((offset=0; offset<TOTAL; offset+=BATCH_SIZE)); do
            local remaining=$((TOTAL - offset))
            local batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))
            
            # 更新状态
            show_dashboard $offset "running" "处理批次: offset=$offset, limit=$batch"
            
            # 运行批次（后台执行）
            echo "[$(date '+%H:%M:%S')] Starting batch: offset=$offset, limit=$batch" >> "${RUN_ID}.log"
            
            if uv run ad-replay-dtpqa \
                --subset synth \
                --question-type category_1 \
                --offset $offset \
                --limit $batch \
                --run-id $RUN_ID \
                --execution-mode hybrid \
                --append >> "${RUN_ID}.log" 2>&1; then
                echo "[$(date '+%H:%M:%S')] ✓ Batch complete: offset=$offset" >> "${RUN_ID}.log"
            else
                echo "[$(date '+%H:%M:%S')] ✗ Batch error: offset=$offset" >> "${RUN_ID}.log"
            fi
            
            # 短暂停顿让屏幕刷新
            sleep 0.5
        done
        
        # 完成
        show_dashboard $TOTAL "complete" "实验完成!"
        echo "" >> "${RUN_ID}.log"
        echo "=== EXPERIMENT COMPLETE ===" >> "${RUN_ID}.log"
        echo "Run ID: $RUN_ID" >> "${RUN_ID}.log"
        echo "Skills: $SKILL_STORE" >> "${RUN_ID}.log"
        
    } &
    
    # 获取后台进程ID
    BG_PID=$!
    
    # 实时刷新循环
    while kill -0 $BG_PID 2>/dev/null; do
        local current_offset=$(grep -o "offset=[0-9]*" "${RUN_ID}.log" 2>/dev/null | tail -1 | cut -d= -f2 || echo "0")
        show_dashboard ${current_offset:-0} "running" "实验运行中..."
        show_recent_logs
        sleep 3
    done
    
    # 最终显示
    show_dashboard $TOTAL "complete" "实验已完成!"
    echo ""
    echo "📊 最终结果:"
    echo "  Run ID: $RUN_ID"
    echo "  日志: ${RUN_ID}.log"
    echo "  技能: $SKILL_STORE"
    echo "  结果: data/artifacts/$RUN_ID/"
    echo ""
    echo "分析命令:"
    echo "  python3 analyze_results.py $RUN_ID"
}

# 捕获Ctrl+C
trap 'echo ""; echo "停止实验..."; kill $BG_PID 2>/dev/null; exit 0' INT

# 启动
main
