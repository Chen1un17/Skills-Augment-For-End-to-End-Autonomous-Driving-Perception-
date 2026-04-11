#!/bin/bash
# 200样本实验监控脚本

LOG_FILE="200_exp.log"
SKILL_DIR=$(ls -td /tmp/dtpqa_adaptive_skills_* 2>/dev/null | head -1)

echo "=============================================="
echo "200样本实验监控"
echo "时间: $(date '+%H:%M:%S')"
echo "=============================================="

# 1. 检查进程
PID=$(pgrep -f "run_dtpqa_200_adaptive" | head -1)
if [ -n "$PID" ]; then
    echo "✓ 实验运行中 (PID: $PID)"
    RUNTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
    echo "  运行时长: $RUNTIME"
else
    echo "✗ 实验未运行"
fi

# 2. 解析进度
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "--- 进度 ---"
    tail -5 "$LOG_FILE" | grep -E "(Batch|Progress|complete)" | tail -3
    
    # 计算完成率
    BATCHES=$(grep -c "Batch" "$LOG_FILE" 2>/dev/null)
    if [ "$BATCHES" -gt 0 ]; then
        PROGRESS=$((BATCHES * 100 / 40))
        echo "  总体进度: ~${PROGRESS}% (${BATCHES}/40 批次)"
    fi
fi

# 3. 技能统计
if [ -n "$SKILL_DIR" ]; then
    echo ""
    echo "--- 技能生成 ---"
    SKILL_COUNT=$(ls "$SKILL_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "  技能数量: $SKILL_COUNT"
    
    if [ "$SKILL_COUNT" -gt 0 ]; then
        echo "  技能目录: $SKILL_DIR"
        # 显示最新技能
        echo "  最新技能:"
        ls -lt "$SKILL_DIR"/*.json 2>/dev/null | head -3 | awk '{print "    " $9}'
    fi
else
    echo ""
    echo "--- 技能生成 ---"
    echo "  技能目录尚未创建"
fi

# 4. 检查是否有错误
if [ -f "$LOG_FILE" ]; then
    ERRORS=$(grep -c "ERROR\|Error\|failed" "$LOG_FILE" 2>/dev/null)
    if [ "$ERRORS" -gt 0 ]; then
        echo ""
        echo "⚠ 发现 $ERRORS 个错误:"
        grep -E "ERROR|Error|failed" "$LOG_FILE" | tail -3
    fi
fi

# 5. 预估剩余时间
if [ -f "$LOG_FILE" ] && [ "$BATCHES" -gt 1 ]; then
    echo ""
    echo "--- 预估 ---"
    # 简单估算（假设每批次5样本，每样本120秒）
    REMAINING=$(( (40 - BATCHES) * 5 * 2 ))  # 分钟
    if [ $REMAINING -gt 60 ]; then
        REMAINING_HOUR=$((REMAINING / 60))
        REMAINING_MIN=$((REMAINING % 60))
        echo "  预计剩余: ${REMAINING_HOUR}小时${REMAINING_MIN}分"
    else
        echo "  预计剩余: ${REMAINING}分钟"
    fi
fi

echo ""
echo "=============================================="
echo "刷新: watch -n 30 ./monitor_200_exp.sh"
echo "日志: tail -f 200_exp.log"
echo "=============================================="
