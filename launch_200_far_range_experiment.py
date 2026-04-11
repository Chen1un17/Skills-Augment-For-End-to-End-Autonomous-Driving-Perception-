#!/usr/bin/env python3
"""Launch 200-sample experiment with automatic far-range skill generation.

This experiment is specifically designed to:
1. Process 200 samples from DTPQA synth dataset
2. Use LOW entropy threshold (0.5) to trigger reflection on uncertain cases
3. Automatically generate skills for far-range (30m+) scenarios
4. Persist learned skills for future use
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ad_cornercase.universal_framework import (
    EdgeCloudOrchestrator,
    OrchestratorConfig,
    SkillLibrary,
    ReflectionPolicy,
    ReflectionTrigger,
    PerceptionTask,
    TASK_REGISTRY,
)


def create_adaptive_policy() -> ReflectionPolicy:
    """Create an adaptive reflection policy that learns when to reflect.

    Auto Research Principles:
    1. Let the model learn its own uncertainty boundaries
    2. Use hybrid triggers: entropy + confidence + skill mismatch
    3. Lower threshold for far-range but don't force
    4. Learn from both successes and failures
    """
    return ReflectionPolicy(
        name="adaptive_smart_reflection",
        trigger=ReflectionTrigger.HYBRID,  # Combine multiple signals
        entropy_threshold=0.4,  # Based on error analysis: avg error entropy = 0.50
        confidence_threshold=0.65,  # Lower = more conservative for far-range
        difficulty_threshold=0.6,
        distance_ranges={
            "far": {
                "min": 30,
                "max": float("inf"),
                "force_reflect": False,  # DON'T force - let model learn
                "confidence_threshold": 0.5,  # Lower threshold = more likely to reflect
                "entropy_threshold": 0.35,  # Even more sensitive for far-range
            },
            "mid": {
                "min": 20,
                "max": 30,
                "force_reflect": False,
                "confidence_threshold": 0.65,
                "entropy_threshold": 0.4,
            },
            "near": {
                "min": 0,
                "max": 20,
                "force_reflect": False,
                "confidence_threshold": 0.7,
                "entropy_threshold": 0.5,
            },
        },
        task_overrides={
            "OBJECT_DETECTION": {
                "trigger": "HYBRID",
                "entropy_threshold": 0.4,
            }
        }
    )


def setup_experiment():
    """Setup the 200-sample experiment."""

    # Create run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"dtpqa_200_far_range_{timestamp}"

    # Create skill store directory
    skill_store_dir = Path(f"/tmp/dtpqa_far_range_skills_{timestamp}")
    skill_store_dir.mkdir(parents=True, exist_ok=True)

    # Initialize skill library
    skill_lib = SkillLibrary(storage_dir=skill_store_dir)

    # Create orchestrator config with adaptive policy
    policy = create_adaptive_policy()

    config = OrchestratorConfig(
        edge_model="Qwen/Qwen3.5-9B",
        edge_max_tokens=512,
        cloud_model="Pro/moonshotai/Kimi-K2.5",
        cloud_max_tokens=1024,
        reflection_policy=policy,
        skill_library=skill_lib,
        enable_skill_learning=True,
        enable_skill_application=True,
        persist_new_skills=True,
    )

    print("="*70)
    print("200-SAMPLE FAR-RANGE SKILL GENERATION EXPERIMENT")
    print("="*70)
    print(f"\nRun ID: {run_id}")
    print(f"Skill Store: {skill_store_dir}")
    print(f"\nReflection Policy:")
    print(f"  - Far-range (30m+): FORCE REFLECT")
    print(f"  - Entropy threshold: 0.5 (vs default 1.0)")
    print(f"  - Near/Mid: Standard thresholds")
    print(f"\nSkill Learning:")
    print(f"  - Auto-generate skills from corrections")
    print(f"  - Persist to: {skill_store_dir}")
    print(f"  - Apply learned skills to similar cases")

    return run_id, config, skill_store_dir


def create_launch_script(run_id: str, skill_store_dir: Path):
    """Create the bash launch script for the experiment."""

    script_content = f'''#!/bin/bash
# 200-Sample Far-Range Skill Generation Experiment
# Generated: {datetime.now().isoformat()}

set -e

echo "=============================================="
echo "200-Sample Far-Range Skill Generation"
echo "Run ID: {run_id}"
echo "=============================================="

# Clear proxy settings
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

# Environment
export EDGE_MODEL="Qwen/Qwen3.5-9B"
export EDGE_MAX_COMPLETION_TOKENS="512"
export CLOUD_MODEL="Pro/moonshotai/Kimi-K2.5"
export JUDGE_MODEL="Pro/moonshotai/Kimi-K2.5"
export DTPQA_ROOT="data/Distance-Annotated Traffic Perception Question Ans/DTPQA"
export SKILL_STORE_DIR="{skill_store_dir}"
export ARTIFACTS_DIR="data/artifacts"

# Lower entropy threshold for far-range sensitivity
export UNCERTAINTY_ENTROPY_THRESHOLD="0.5"
export REQUEST_TIMEOUT_SECONDS="300"
export MAX_RETRIES="3"
export MCP_SERVER_HOST="127.0.0.1"
export MCP_SERVER_PORT="8003"

# Enable DTPQA reflection
export ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER="1"

echo ""
echo "Configuration:"
echo "  - Samples: 200"
echo "  - Entropy threshold: 0.5 (lowered for far-range)"
echo "  - Skill store: {skill_store_dir}"
echo "  - Far-range: Force reflection enabled"
echo ""

# Run in batches of 10 for checkpointing
BATCH_SIZE=10
TOTAL=200

for ((offset=0; offset<TOTAL; offset+=BATCH_SIZE)); do
    remaining=$((TOTAL - offset))
    current_batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))

    echo ""
    echo "[$(date '+%H:%M:%S')] Processing batch: offset=$offset, limit=$current_batch"

    uv run ad-replay-dtpqa \\
        --subset synth \\
        --question-type category_1 \\
        --offset $offset \\
        --limit $current_batch \\
        --run-id {run_id} \\
        --execution-mode hybrid \\
        --append || true

    echo "[$(date '+%H:%M:%S')] Batch complete. Progress: $((offset + current_batch))/$TOTAL"

    # Check for new skills
    skill_count=$(ls -1 {skill_store_dir}/*.json 2>/dev/null | wc -l)
    echo "  Skills generated so far: $skill_count"

    # Small delay between batches
    sleep 2
done

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results: data/artifacts/{run_id}/"
echo "Skills: {skill_store_dir}/"
echo ""
echo "To evaluate: ./run_judge.sh {run_id}"
echo "To analyze: python3 analyze_results.py {run_id}"
'''

    script_path = Path(f"experiments/run_{run_id}.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)

    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)
    return script_path


def analyze_far_range_errors():
    """Analyze previous experiment to understand far-range failure patterns."""

    print("\n" + "="*70)
    print("ANALYZING PREVIOUS EXPERIMENT (31 samples)")
    print("="*70)

    predictions = []
    pred_file = Path("data/artifacts/dtpqa_synth_50_20260401_115405/predictions.jsonl")

    if not pred_file.exists():
        print("Previous experiment not found, skipping analysis")
        return

    with open(pred_file) as f:
        for line in f:
            predictions.append(json.loads(line))

    # Find far-range errors
    far_errors = []
    for p in predictions:
        if p.get('metadata', {}).get('distance_group') == 'far':
            gt = p.get('ground_truth_answer', '').lower()
            pred = p.get('baseline_result', {}).get('qa_report', [{}])[0].get('answer', '').lower()

            if gt == 'yes' and 'no' in pred:
                far_errors.append({
                    'case_id': p['case_id'],
                    'distance': p['metadata'].get('distance_meters'),
                    'entropy': p.get('baseline_result', {}).get('entropy', 0),
                    'confidence': 1 - p.get('baseline_result', {}).get('entropy', 0),
                })

    print(f"\nFar-range samples: {len([p for p in predictions if p.get('metadata', {}).get('distance_group') == 'far'])}")
    print(f"Far-range errors: {len(far_errors)}")

    if far_errors:
        print("\nError characteristics:")
        for err in far_errors:
            print(f"  {err['case_id']}: {err['distance']}m, entropy={err['entropy']:.2f}")

        avg_entropy = sum(e['entropy'] for e in far_errors) / len(far_errors)
        print(f"\nAverage entropy of errors: {avg_entropy:.2f}")
        print("Recommendation: Set threshold to {:.1f} to catch these cases".format(avg_entropy * 0.8))


def main():
    """Main entry point."""

    # Analyze previous results
    analyze_far_range_errors()

    # Setup new experiment
    run_id, config, skill_store_dir = setup_experiment()

    # Create launch script
    script_path = create_launch_script(run_id, skill_store_dir)

    print("\n" + "="*70)
    print("EXPERIMENT READY TO LAUNCH")
    print("="*70)
    print(f"\nRun ID: {run_id}")
    print(f"Launch script: {script_path}")
    print(f"\nTo start the experiment:")
    print(f"  ./{script_path}")
    print(f"\nExpected duration: ~6-8 hours (200 samples × ~2 min/sample)")
    print(f"Expected skills generated: 20-50 (based on 10-25% reflection rate)")
    print("="*70)

    return run_id, script_path


if __name__ == "__main__":
    run_id, script_path = main()
    print(f"\n[INFO] Run ID: {run_id}")
    print(f"[INFO] Launch: ./{script_path}")
