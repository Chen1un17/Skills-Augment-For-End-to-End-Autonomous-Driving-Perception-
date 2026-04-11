#!/usr/bin/env python3
"""Launch 200-sample experiment with far-range skill generation focus."""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ad_cornercase.experiments.config import ExperimentConfig, ModelConfig, DatasetConfig
from ad_cornercase.experiments.runner import ExperimentRunner

def main():
    # Create experiment configuration optimized for far-range skill generation
    config = ExperimentConfig(
        name="dtpqa-far-range-skill-gen",
        description="200-sample experiment targeting far-range skill generation with lower entropy threshold",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
            cloud_model="Pro/moonshotai/Kimi-K2.5",
            judge_model="Pro/moonshotai/Kimi-K2.5",
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="synth",  # Using synth for larger sample size
            question_type="category_1",
            limit=200,  # 200 samples as requested
            offset=0,
        ),
        batch_size=5,  # Process 5 samples at a time for efficiency
        max_retries=3,
        request_timeout_seconds=300.0,
        batch_sleep_seconds=5,  # Small delay between batches
        skill_store_dir=Path("/tmp/dtpqa_far_range_skills"),
        clean_skill_store=True,  # Start fresh
        execution_mode="hybrid",
        enable_reflection=True,
        entropy_threshold=0.6,  # LOWER threshold for far-range sensitivity
        enable_dtpqa_people_reflection=True,
        enable_judge=True,
        dtpqa_root=Path("data/Distance-Annotated Traffic Perception Question Ans/DTPQA"),
        artifacts_dir=Path("data/artifacts"),
    )

    # Save configuration
    config_path = Path(f"experiments/configs/{config.run_id}.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)
    print(f"[INFO] Configuration saved to: {config_path}")

    # Create launch script with proper proxy settings
    launch_script = f'''#!/bin/bash
# Launch script for far-range skill generation experiment
# Generated: {datetime.now().isoformat()}

set -e

echo "=============================================="
echo "Far-Range Skill Generation Experiment"
echo "Run ID: {config.run_id}"
echo "Samples: 200"
echo "Entropy Threshold: 0.6 (lowered for far-range)"
echo "=============================================="

# Clear all proxy settings
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# Environment settings
export EDGE_MODEL="{config.models.edge_model}"
export EDGE_MAX_COMPLETION_TOKENS="{config.models.edge_max_completion_tokens}"
export CLOUD_MODEL="{config.models.cloud_model}"
export JUDGE_MODEL="{config.models.judge_model}"
export DTPQA_ROOT="{config.dtpqa_root}"
export SKILL_STORE_DIR="{config.skill_store_dir}"
export ARTIFACTS_DIR="{config.artifacts_dir}"
export UNCERTAINTY_ENTROPY_THRESHOLD="{config.entropy_threshold}"
export REQUEST_TIMEOUT_SECONDS="{config.request_timeout_seconds}"
export MAX_RETRIES="{config.max_retries}"
export MCP_SERVER_HOST="{config.mcp_server_host}"
export MCP_SERVER_PORT="{config.mcp_server_port}"

# Enable DTPQA reflection trigger
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

echo ""
echo "Starting experiment with settings:"
echo "  - Samples: 200"
echo "  - Batch size: {config.batch_size}"
echo "  - Entropy threshold: {config.entropy_threshold}"
echo "  - Skill store: {config.skill_store_dir}"
echo ""

# Run experiment
uv run python3 -c "
import sys
sys.path.insert(0, 'src')
from ad_cornercase.experiments.config import ExperimentConfig
from ad_cornercase.experiments.runner import ExperimentRunner

config = ExperimentConfig.load('{config_path}')
runner = ExperimentRunner(config)
status = runner.run(resume=True, batch_size={config.batch_size})

print(f'\\nExperiment completed with state: {{status.state}}')
print(f'Completed: {{status.completed_cases}}/{{status.total_cases}}')
"

echo ""
echo "Experiment complete!"
echo "Results: data/artifacts/{config.run_id}/"
echo ""
echo "To run judge evaluation:"
echo "  ./run_judge.sh {config.run_id}"
'''

    script_path = Path(f"experiments/run_{config.run_id}.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(launch_script)
    script_path.chmod(0o755)
    print(f"[INFO] Launch script saved to: {script_path}")

    # Print summary
    print("\n" + "="*60)
    print("Far-Range Skill Generation Experiment Configuration")
    print("="*60)
    print(f"Run ID: {config.run_id}")
    print(f"Samples: 200")
    print(f"Subset: synth")
    print(f"Entropy Threshold: 0.6 (lowered from 1.0)")
    print(f"Batch Size: {config.batch_size}")
    print(f"Skill Store: {config.skill_store_dir}")
    print(f"Clean Skill Store: Yes")
    print("")
    print("Strategy:")
    print("  1. Lower entropy threshold (0.6) to capture more far-range uncertainty")
    print("  2. 200 samples for comprehensive coverage")
    print("  3. Batch size 5 for efficiency with checkpointing")
    print("  4. Clean skill store to start fresh skill learning")
    print("")
    print(f"To launch: ./{script_path}")
    print("="*60)

    return config.run_id

if __name__ == "__main__":
    run_id = main()
    print(f"\n[INFO] Run ID: {run_id}")
