#!/bin/bash
# Launch script for far-range skill generation experiment
# Generated: 2026-04-01T14:49:23.761498

set -e

echo "=============================================="
echo "Far-Range Skill Generation Experiment"
echo "Run ID: dtpqa_far_range_skill_gen_20260401_144923"
echo "Samples: 200"
echo "Entropy Threshold: 0.6 (lowered for far-range)"
echo "=============================================="

# Clear all proxy settings
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# Environment settings
export EDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="/tmp/dtpqa_far_range_skills"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.6"
export REQUEST_TIMEOUT_SECONDS="300.0"
export MAX_RETRIES="3"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"

# Enable DTPQA reflection trigger
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

echo ""
echo "Starting experiment with settings:"
echo "  - Samples: 200"
echo "  - Batch size: 5"
echo "  - Entropy threshold: 0.6"
echo "  - Skill store: /tmp/dtpqa_far_range_skills"
echo ""

# Run experiment
uv run python3 -c "
import sys
sys.path.insert(0, 'src')
from ad_cornercase.experiments.config import ExperimentConfig
from ad_cornercase.experiments.runner import ExperimentRunner

config = ExperimentConfig.load('experiments/configs/dtpqa_far_range_skill_gen_20260401_144923.json')
runner = ExperimentRunner(config)
status = runner.run(resume=True, batch_size=5)

print(f'\nExperiment completed with state: {status.state}')
print(f'Completed: {status.completed_cases}/{status.total_cases}')
"

echo ""
echo "Experiment complete!"
echo "Results: data/artifacts/dtpqa_far_range_skill_gen_20260401_144923/"
echo ""
echo "To run judge evaluation:"
echo "  ./run_judge.sh dtpqa_far_range_skill_gen_20260401_144923"
