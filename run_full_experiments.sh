#!/bin/bash
# Full automated experiment runner for DTPQA synth dataset

set -e

cd "$(dirname "$0")"

# Clear proxy
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

echo "=============================================="
echo "Starting Full Automated Research Pipeline"
echo "Dataset: DTPQA Synthetic"
echo "=============================================="
echo ""

# Check MCP server
if ! curl -s http://127.0.0.1:8003/mcp > /dev/null 2>&1; then
    echo "Starting MCP server..."
    uv run ad-mcp-server &
    MCP_PID=$!
    sleep 5
fi

# Clean skill store
rm -rf /tmp/dtpqa_skills_empty/*
mkdir -p /tmp/dtpqa_skills_empty

# Run comprehensive experiments
echo "Running comprehensive experiment suite..."
python experiments/dtpqa-integration/code/launch_full_research.py --auto --limit 100

echo ""
echo "=============================================="
echo "Experiments Complete!"
echo "=============================================="
