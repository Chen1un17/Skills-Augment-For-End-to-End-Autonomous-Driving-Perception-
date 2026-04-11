#!/bin/bash
#
# Quick test script for DTPQA real dataset experiments
#

set -e

echo "========================================"
echo "Quick Test: DTPQA Real Dataset"
echo "========================================"
echo ""

# Test with just 10 samples
echo "Testing baseline with 10 samples..."
uv run python experiments/dtpqa-integration/code/launch_full_research.py --baseline --limit 10

echo ""
echo "Test complete! Check data/artifacts/ for results."
