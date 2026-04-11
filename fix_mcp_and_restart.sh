#!/bin/bash
# 修复MCP服务器超时问题并重启

echo "=============================================="
echo "🔧 修复MCP服务器超时问题"
echo "=============================================="

# 1. 停止现有MCP服务器
echo ""
echo "[1/3] 停止现有MCP服务器..."
pkill -f "ad-mcp-server" 2>/dev/null
sleep 2
echo "✓ MCP服务器已停止"

# 2. 设置正确的环境变量
echo ""
echo "[2/3] 配置环境变量..."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# 关键：增加超时时间到5分钟
export REQUEST_TIMEOUT_SECONDS="300"
export SKILL_STORE_DIR="/tmp/dtpqa_mcp_skills_$(date +%s)"
mkdir -p "$SKILL_STORE_DIR"

echo "  REQUEST_TIMEOUT_SECONDS: $REQUEST_TIMEOUT_SECONDS"
echo "  SKILL_STORE_DIR: $SKILL_STORE_DIR"

# 3. 启动新的MCP服务器
echo ""
echo "[3/3] 启动MCP服务器（5分钟超时）..."
nohup uv run ad-mcp-server > mcp_server_fixed.log 2>&1 &
echo "  MCP服务器PID: $!"
sleep 3

# 4. 验证
echo ""
echo "[4/3] 验证MCP服务器..."
if curl -s http://127.0.0.1:8000/mcp -X POST -d '{}' > /dev/null 2>&1; then
    echo "✓ MCP服务器响应正常"
else
    echo "✗ MCP服务器无响应"
    exit 1
fi

echo ""
echo "=============================================="
echo "✅ MCP服务器修复完成！"
echo "=============================================="
echo ""
echo "当前配置："
echo "  - 超时时间: 300秒 (5分钟)"
echo "  - 端口: 8000"
echo "  - 日志: mcp_server_fixed.log"
echo ""
echo "现在可以运行实验:"
echo "  ./run_200_fixed_dashboard.sh"
