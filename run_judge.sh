#!/bin/bash
# Judge evaluation runner

set -e

RUN_ID="${1:-}"

if [ -z "$RUN_ID" ]; then
    echo "Usage: $0 <run_id>"
    echo "Available runs:"
    ls -t data/artifacts/ | head -10
    exit 1
fi

echo "=============================================="
echo "Judge Evaluation: $RUN_ID"
echo "=============================================="

# Clear proxy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# Run judge evaluation
DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA" \
JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5" \
MCP_SERVER_URL="http://127.0.0.1:8003/mcp" \
NO_PROXY="127.0.0.1" \
uv run ad-eval-dtpqa --run-id "$RUN_ID" 2>&1

echo ""
echo "Evaluation complete!"
