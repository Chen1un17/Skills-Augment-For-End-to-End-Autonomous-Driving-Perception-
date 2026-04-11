#!/bin/bash
#
# Master script to run DTPQA real dataset automated research
#

set -e

echo "=============================================="
echo "DTPQA Real Dataset Automated Research Runner"
echo "=============================================="
echo ""

# Check Python environment
echo "Checking environment..."
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Please install uv."
    exit 1
fi

# Check if MCP server is running
MCP_PORT=${MCP_PORT:-8003}
if ! curl -s "http://127.0.0.1:${MCP_PORT}/mcp" > /dev/null 2>&1; then
    echo "WARNING: MCP server not running on port ${MCP_PORT}"
    echo "Please start it with: uv run ad-mcp-server"
    echo ""
    read -p "Start MCP server now in background? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting MCP server..."
        uv run ad-mcp-server &
        MCP_PID=$!
        sleep 5
        if ! curl -s "http://127.0.0.1:${MCP_PORT}/mcp" > /dev/null 2>&1; then
            echo "ERROR: Failed to start MCP server"
            exit 1
        fi
        echo "MCP server started (PID: $MCP_PID)"
    else
        exit 1
    fi
else
    echo "✓ MCP server is running on port ${MCP_PORT}"
fi

# Check data directory
if [ ! -d "data/dtpqa" ]; then
    echo "ERROR: DTPQA data not found at data/dtpqa"
    exit 1
fi
echo "✓ DTPQA data found"

# Clean skill store
SKILL_STORE="/tmp/dtpqa_skills_empty"
if [ -d "$SKILL_STORE" ]; then
    echo "Cleaning skill store..."
    rm -rf "$SKILL_STORE"/*
fi
mkdir -p "$SKILL_STORE"
echo "✓ Skill store ready"

echo ""
echo "=============================================="
echo "Starting Automated Research Pipeline"
echo "=============================================="
echo ""

# Parse arguments
MODE="${1:-auto}"
LIMIT="${2:-}"

# Run the research
CMD="uv run python experiments/dtpqa-integration/code/launch_full_research.py --${MODE}"
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "=============================================="
echo "Research Complete!"
echo "=============================================="
echo ""
echo "Check results in:"
echo "  - experiments/dtpqa-integration/report/"
echo "  - experiments/dtpqa-integration/results/"
echo "  - data/artifacts/"
echo ""
