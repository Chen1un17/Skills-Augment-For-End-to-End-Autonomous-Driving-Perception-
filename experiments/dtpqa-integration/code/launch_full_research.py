#!/usr/bin/env python3
"""
Complete automated research launcher for DTPQA real dataset.

This script orchestrates the full autoresearch pipeline:
1. Run baseline experiments
2. Analyze results and identify failure patterns
3. Iteratively optimize based on findings
4. Generate academic-quality reports

Usage:
    # Full automated research loop
    python launch_full_research.py --auto --target-accuracy 0.7

    # Run specific experiments
    python launch_full_research.py --baseline --limit 100
    python launch_full_research.py --reflection --limit 100

    # Generate final report
    python launch_full_research.py --final-report --run-ids run1 run2 run3

    # Follow autoresearch skill principles explicitly
    python launch_full_research.py --follow-autoresearch
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from ad_cornercase.experiments import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    ExperimentMonitor,
    ReportGenerator,
    LargeScaleBatchRunner,
    IterativeOptimizer,
    AutomatedResearchLoop,
)


def print_section(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_phase_1_baseline(limit: int | None = None) -> str:
    """
    Phase 1: Run baseline experiment (no reflection).

    Following autoresearch INNER LOOP:
    - Clear hypothesis: Baseline establishes performance without intervention
    - Measurable outcome: Accuracy on real dataset
    - Locked protocol before execution
    """
    print_section("PHASE 1: BASELINE EXPERIMENT (Inner Loop)")
    print("Hypothesis: Edge-only inference (no reflection) establishes baseline")
    print("Prediction: Will show false-negative bias on real positive cases")
    print("Metric: exact_match_accuracy, distance-stratified performance\n")

    config = ExperimentConfig(
        name="dtpqa-real-baseline",
        description="Baseline: DTPQA real with edge-only inference, no reflection",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",
            question_type="category_1",
            limit=limit,
        ),
        execution_mode="edge_only",
        enable_reflection=False,
        enable_dtpqa_people_reflection=False,
        batch_size=1,
        request_timeout_seconds=300,
        clean_skill_store=True,
    )

    # Lock protocol by saving config before running
    config_path = Path("./experiments/dtpqa-integration/protocols")
    config_path.mkdir(parents=True, exist_ok=True)
    config.save(config_path / f"{config.run_id}_protocol.json")
    print(f"[PROTOCOL LOCKED] {config.run_id}")

    runner = ExperimentRunner(config)
    status = runner.run(resume=True)

    return config.run_id


def run_phase_2_reflection(limit: int | None = None) -> str:
    """
    Phase 2: Run experiment with cloud reflection enabled.

    INNER LOOP: Testing intervention hypothesis
    """
    print_section("PHASE 2: REFLECTION EXPERIMENT (Inner Loop)")
    print("Hypothesis: Cloud reflection improves far-range accuracy")
    print("Prediction: Lower false-negative rate, especially on far-distance cases")
    print("Metric: exact_match_accuracy, distance-stratified improvement\n")

    config = ExperimentConfig(
        name="dtpqa-real-hybrid",
        description="Intervention: DTPQA real with hybrid edge+cloud reflection enabled",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",
            question_type="category_1",
            limit=limit,
        ),
        execution_mode="hybrid",
        enable_reflection=True,
        enable_dtpqa_people_reflection=True,
        entropy_threshold=1.0,
        batch_size=1,
        request_timeout_seconds=300,
        clean_skill_store=True,
    )

    # Lock protocol
    config_path = Path("./experiments/dtpqa-integration/protocols")
    config_path.mkdir(parents=True, exist_ok=True)
    config.save(config_path / f"{config.run_id}_protocol.json")
    print(f"[PROTOCOL LOCKED] {config.run_id}")

    runner = ExperimentRunner(config)
    status = runner.run(resume=True)

    return config.run_id


def run_phase_3_outer_loop_analysis(run_ids: list[str]) -> dict[str, Any]:
    """
    Phase 3: Outer Loop - Synthesize findings and decide direction.

    OUTER LOOP principles:
    - Review results, find patterns
    - Update findings.md with current understanding
    - Generate new hypotheses
    - Decide: DEEPEN, BROADEN, PIVOT, or CONCLUDE
    """
    print_section("PHASE 3: OUTER LOOP ANALYSIS")
    print("Activity: Review results, find patterns, decide next direction\n")

    monitor = ExperimentMonitor()
    optimizer = IterativeOptimizer()

    # Collect metrics
    all_metrics = {rid: monitor.analyze(rid) for rid in run_ids}

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "experiments": run_ids,
        "findings": {},
        "direction": None,
        "recommendations": [],
    }

    # Compare baseline vs reflection
    baseline_runs = [rid for rid in run_ids if "baseline" in rid.lower()]
    reflection_runs = [rid for rid in run_ids if "reflection" in rid.lower()]

    if baseline_runs and reflection_runs:
        baseline_metrics = all_metrics[baseline_runs[0]]
        reflection_metrics = all_metrics[reflection_runs[0]]

        accuracy_delta = reflection_metrics.exact_match_accuracy - baseline_metrics.exact_match_accuracy

        analysis["findings"]["reflection_effect"] = {
            "baseline_accuracy": baseline_metrics.exact_match_accuracy,
            "reflection_accuracy": reflection_metrics.exact_match_accuracy,
            "absolute_delta": accuracy_delta,
            "relative_improvement": (accuracy_delta / baseline_metrics.exact_match_accuracy * 100)
            if baseline_metrics.exact_match_accuracy > 0
            else 0,
        }

        # Distance-stratified analysis
        for dist in ["near", "mid", "far", "unknown"]:
            base_acc = baseline_metrics.distance_accuracy.get(dist, 0)
            refl_acc = reflection_metrics.distance_accuracy.get(dist, 0)
            analysis["findings"][f"{dist}_improvement"] = refl_acc - base_acc

        # Decide direction
        if accuracy_delta > 0.05:  # >5% improvement
            analysis["direction"] = "DEEPEN"
            analysis["recommendations"].extend([
                "Reflection works - explore threshold tuning",
                "Test different trigger strategies",
                "Run ablation on entropy threshold",
            ])
        elif accuracy_delta > 0:
            analysis["direction"] = "BROADEN"
            analysis["recommendations"].extend([
                "Small improvement - test adjacent interventions",
                "Try different cloud models",
                "Experiment with skill persistence strategies",
            ])
        else:
            analysis["direction"] = "PIVOT"
            analysis["recommendations"].extend([
                "No improvement - reflection may not help this failure mode",
                "Try different edge models",
                "Consider prompt engineering instead of reflection",
                "Investigate if failure is in cloud model itself",
            ])

    # Save analysis
    analysis_path = Path("./experiments/dtpqa-integration/results/outer_loop_analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis Results:")
    print(f"  Baseline Accuracy: {analysis['findings'].get('reflection_effect', {}).get('baseline_accuracy', 0):.3f}")
    print(f"  Reflection Accuracy: {analysis['findings'].get('reflection_effect', {}).get('reflection_accuracy', 0):.3f}")
    print(f"  Delta: {analysis['findings'].get('reflection_effect', {}).get('absolute_delta', 0):+.3f}")
    print(f"\n  Direction: {analysis['direction']}")
    print(f"  Recommendations:")
    for rec in analysis["recommendations"]:
        print(f"    - {rec}")

    return analysis


def run_phase_4_iterative_optimization(
    base_run_id: str,
    target_accuracy: float = 0.7,
    max_iterations: int = 3,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Phase 4: Iterative optimization based on outer loop analysis.

    Following autoresearch: Run improved configurations based on direction.
    """
    print_section("PHASE 4: ITERATIVE OPTIMIZATION")
    print(f"Target: {target_accuracy:.0%} accuracy")
    print(f"Max iterations: {max_iterations}\n")

    optimizer = IterativeOptimizer()

    # Generate improvements
    improvements = optimizer.optimize(base_run_id, direction="auto")

    if not improvements:
        print("[WARN] No improvements generated")
        return {"completed": [], "best_run_id": None, "best_accuracy": 0}

    # Run improved configurations
    completed_runs = []
    best_accuracy = 0.0
    best_run_id = None

    for i, config in enumerate(improvements[:max_iterations], 1):
        if limit:
            config.dataset.limit = limit

        print(f"\nRunning improvement {i}/{min(len(improvements), max_iterations)}: {config.name}")

        runner = ExperimentRunner(config)
        status = runner.run(resume=True)

        if status.state == "completed":
            completed_runs.append(config.run_id)

            metrics = ExperimentMonitor().analyze(config.run_id)
            if metrics.exact_match_accuracy > best_accuracy:
                best_accuracy = metrics.exact_match_accuracy
                best_run_id = config.run_id

            if metrics.exact_match_accuracy >= target_accuracy:
                print(f"\n[TARGET REACHED] {metrics.exact_match_accuracy:.3f}")
                break

    return {
        "completed": completed_runs,
        "best_run_id": best_run_id,
        "best_accuracy": best_accuracy,
    }


def run_phase_5_final_report(run_ids: list[str]) -> Path:
    """
    Phase 5: Generate academic-quality final report.
    """
    print_section("PHASE 5: FINAL REPORT GENERATION")

    generator = ReportGenerator()

    output_dir = Path("./experiments/dtpqa-integration/final_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = generator.generate_full_report(
        run_ids=run_ids,
        output_dir=output_dir,
        title="Training-Free Edge-Cloud VLM for Autonomous Driving: DTPQA Real Dataset Evaluation",
    )

    # Also update findings.md for the main research state
    update_findings_md(run_ids)

    print(f"\nFinal report generated: {report_path}")
    return report_path


def update_findings_md(run_ids: list[str]) -> None:
    """Update findings.md with latest results."""
    monitor = ExperimentMonitor()

    findings_path = Path("./findings.md")
    if not findings_path.exists():
        return

    # Read existing
    with open(findings_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Generate new section
    new_section = f"""

## Automated Experiment Results ({datetime.now().strftime('%Y-%m-%d')})

The following results were generated through automated large-scale experiments:

"""

    for run_id in run_ids:
        metrics = monitor.analyze(run_id)
        new_section += f"""### {run_id}

- **Total Cases**: {metrics.total_cases}
- **Exact Match Accuracy**: {metrics.exact_match_accuracy:.3f}
- **Judge Score Mean**: {metrics.judge_score_mean:.1f} (±{metrics.judge_score_std:.1f})
- **Mean Latency**: {metrics.mean_latency_ms/1000:.1f}s
- **Reflection Trigger Rate**: {metrics.reflection_trigger_rate:.3f}

**Distance-Stratified Accuracy**:
- Near: {metrics.distance_accuracy.get('near', 0):.3f}
- Mid: {metrics.distance_accuracy.get('mid', 0):.3f}
- Far: {metrics.distance_accuracy.get('far', 0):.3f}
- Unknown: {metrics.distance_accuracy.get('unknown', 0):.3f}

"""

    # Append to file
    with open(findings_path, "a", encoding="utf-8") as f:
        f.write(new_section)

    print(f"Updated findings.md with automated experiment results")


def run_full_autoresearch_pipeline(
    target_accuracy: float = 0.7,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Run complete autoresearch pipeline.

    Implements the two-loop architecture:
    - INNER LOOP: Fast experiment iteration
    - OUTER LOOP: Periodic reflection and direction setting
    """
    print_section("AUTORESEARCH PIPELINE")
    print("Following two-loop architecture for autonomous research")
    print(f"Target accuracy: {target_accuracy:.0%}")
    if limit:
        print(f"Sample limit: {limit} (for testing)")
    print()

    all_run_ids = []

    # Phase 1: Baseline
    baseline_run = run_phase_1_baseline(limit=limit)
    all_run_ids.append(baseline_run)

    # Phase 2: Reflection
    reflection_run = run_phase_2_reflection(limit=limit)
    all_run_ids.append(reflection_run)

    # Phase 3: Outer Loop Analysis
    analysis = run_phase_3_outer_loop_analysis(all_run_ids)

    # Phase 4: Iterative Optimization (if needed)
    if analysis["direction"] in ["DEEPEN", "BROADEN"]:
        opt_results = run_phase_4_iterative_optimization(
            base_run_id=baseline_run,
            target_accuracy=target_accuracy,
            max_iterations=3,
            limit=limit,
        )
        all_run_ids.extend(opt_results["completed"])

        if opt_results["best_run_id"]:
            print(f"\nBest optimized run: {opt_results['best_run_id']}")
            print(f"Best accuracy: {opt_results['best_accuracy']:.3f}")

    # Phase 5: Final Report
    report_path = run_phase_5_final_report(all_run_ids)

    summary = {
        "completed_at": datetime.now().isoformat(),
        "all_runs": all_run_ids,
        "analysis": analysis,
        "final_report": str(report_path),
    }

    # Save summary
    summary_path = Path("./experiments/dtpqa-integration/results/pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_section("PIPELINE COMPLETE")
    print(f"All runs: {all_run_ids}")
    print(f"Final report: {report_path}")
    print(f"Summary saved: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Complete automated research launcher for DTPQA real dataset"
    )

    # Main modes
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run full automated research pipeline",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run only baseline experiment",
    )
    parser.add_argument(
        "--reflection",
        action="store_true",
        help="Run only reflection experiment",
    )
    parser.add_argument(
        "--analyze",
        nargs="+",
        help="Analyze specific run IDs",
    )
    parser.add_argument(
        "--final-report",
        nargs="+",
        dest="final_report",
        help="Generate final report from run IDs",
    )

    # Configuration
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples for faster testing",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.7,
        help="Target accuracy for auto optimization",
    )

    args = parser.parse_args()

    if args.auto:
        run_full_autoresearch_pipeline(
            target_accuracy=args.target_accuracy,
            limit=args.limit,
        )

    elif args.baseline:
        run_id = run_phase_1_baseline(limit=args.limit)
        print(f"\nBaseline complete: {run_id}")

    elif args.reflection:
        run_id = run_phase_2_reflection(limit=args.limit)
        print(f"\nReflection complete: {run_id}")

    elif args.analyze:
        run_phase_3_outer_loop_analysis(args.analyze)

    elif args.final_report:
        run_phase_5_final_report(args.final_report)

    else:
        # Default: run full pipeline
        print("No mode specified, running full automated pipeline...")
        print("Use --help for options\n")
        run_full_autoresearch_pipeline(
            target_accuracy=args.target_accuracy,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
