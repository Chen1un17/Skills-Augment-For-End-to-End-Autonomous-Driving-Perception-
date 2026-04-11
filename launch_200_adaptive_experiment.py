#!/usr/bin/env python3
"""Launch 200-sample adaptive experiment with smart reflection.

Auto Research Principles:
- Model learns its own uncertainty boundaries
- Hybrid triggers (entropy + confidence + skill mismatch)
- No forced reflection - let system adapt organically
- Generate skills from learned corrections
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')

from ad_cornercase.universal_framework import (
    ReflectionPolicy, ReflectionTrigger, SkillLibrary
)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"dtpqa_200_adaptive_{timestamp}"
    skill_store = f"/tmp/dtpqa_adaptive_skills_{timestamp}"
    
    # Adaptive policy - no forced reflection
    policy = ReflectionPolicy(
        name="adaptive_smart",
        trigger=ReflectionTrigger.HYBRID,
        entropy_threshold=0.4,  # Lower for sensitivity
        confidence_threshold=0.65,
        distance_ranges={
            "far": {
                "min": 30, "max": float("inf"),
                "force_reflect": False,  # Smart, not forced
                "confidence_threshold": 0.5,
            }
        }
    )
    
    # Create launch script
    script = f'''#!/bin/bash
# 200-Sample Adaptive Experiment
set -e

echo "=============================================="
echo "200-Sample Adaptive Skill Learning"
echo "Run ID: {run_id}"
echo "=============================================="

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="{skill_store}"
export ARTIFACTS_DIR="data/artifacts"
export UNCERTAINTY_ENTROPY_THRESHOLD="0.4"
export REQUEST_TIMEOUT_SECONDS="300"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

mkdir -p {skill_store}

echo ""
echo "Policy: Adaptive Smart Reflection"
echo "  - Entropy threshold: 0.4"
echo "  - Far-range: NO forced reflection"
echo "  - Strategy: Learn when to ask for help"
echo ""

# Run 200 samples in batches
BATCH_SIZE=5
for ((offset=0; offset<200; offset+=BATCH_SIZE)); do
    remaining=$((200 - offset))
    batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))
    
    echo "[$(date +%H:%M:%S)] Batch $((offset/batch+1))/40: offset=$offset"
    
    uv run ad-replay-dtpqa \\
        --subset synth \\
        --question-type category_1 \\
        --offset $offset \\
        --limit $batch \\
        --run-id {run_id} \\
        --execution-mode hybrid \\
        --append 2>&1 | tail -20
    
    # Progress stats
    completed=$((offset + batch))
    skills=$(ls {skill_store}/*.json 2>/dev/null | wc -l)
    echo "  Progress: $completed/200 | Skills: $skills"
    
    sleep 1
done

echo ""
echo "=============================================="
echo "Complete!"
echo "Run: {run_id}"
echo "Skills: {skill_store}"
echo "=============================================="
'''
    
    script_path = f"experiments/run_{run_id}.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    Path(script_path).chmod(0o755)
    
    print("="*70)
    print("ADAPTIVE 200-SAMPLE EXPERIMENT")
    print("="*70)
    print(f"\nRun ID: {run_id}")
    print(f"Policy: Adaptive Smart Reflection")
    print(f"  - Entropy threshold: 0.4")
    print(f"  - Far-range: NO forced reflection")
    print(f"  - Strategy: Model learns when to ask for help")
    print(f"\nLaunch: ./{script_path}")
    print("="*70)
    
    return run_id, script_path

if __name__ == "__main__":
    run_id, script = main()
    print(f"\n[INFO] Ready to launch: ./{script}")
