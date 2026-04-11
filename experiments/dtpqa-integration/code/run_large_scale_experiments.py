#!/usr/bin/env python3
"""
Large-scale automated experiment runner for DTPQA real dataset.

This script implements the autoresearch two-loop architecture:
- INNER LOOP: Fast experiment iteration with clear measurable outcomes
- OUTER LOOP: Periodic reflection on results and direction adjustment

Usage:
    # Run full baseline experiment on DTPQA real
    python run_large_scale_experiments.py --mode baseline

    # Run with reflection enabled
    python run_large_scale_experiments.py --mode reflection

    # Run ablation studies
    python run_large_scale_experiments.py --mode ablation

    # Run comprehensive batch (baseline + reflection + ablations)
    python run_large_scale_experiments.py --mode comprehensive

    # Monitor running experiment
    python run_large_scale_experiments.py --monitor <run_id>

    # Generate report from completed runs
    python run_large_scale_experiments.py --report <run_id1> <run_id2> ...
"""

from __future__ import annotations

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from ad_cornercase.experiments import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    ExperimentMonitor,
    ReportGenerator,
)
from ad_cornercase.experiments.batch_runner import LargeScaleBatchRunner


def run_baseline_experiment(limit: int | None = None, resume: bool = True) -> str:
    """Run baseline experiment on DTPQA real dataset (no reflection)."""
    print("\n" + "="*70)
    print("INNER LOOP: Running BASELINE experiment")
    print("Prediction: No reflection will show baseline accuracy on real data")
    print("="*70 + "\n")

    config = ExperimentConfig(
        name="dtpqa-real-baseline",
        description="Baseline experiment on DTPQA real dataset without reflection",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",  # ONLY real dataset as requested
            question_type="category_1",
            limit=limit,
        ),
        execution_mode="edge_only",
        enable_reflection=False,
        batch_size=1,
        request_timeout_seconds=300,
        clean_skill_store=True,
    )

    # Save config
    config_dir = Path("./experiments/dtpqa-integration/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config.save(config_dir / f"{config.run_id}_config.json")

    runner = ExperimentRunner(config)
    status = runner.run(resume=resume)

    print(f"\n{'='*70}")
    print(f"BASELINE Complete: {status.run_id}")
    print(f"State: {status.state}")
    print(f"Completed: {status.completed_cases}/{status.total_cases}")
    print(f"{'='*70}\n")

    return config.run_id


def run_reflection_experiment(limit: int | None = None, resume: bool = True) -> str:
    """Run experiment with cloud reflection enabled."""
    print("\n" + "="*70)
    print("INNER LOOP: Running REFLECTION experiment")
    print("Prediction: Cloud reflection will improve far-range accuracy")
    print("="*70 + "\n")

    config = ExperimentConfig(
        name="dtpqa-real-reflection",
        description="DTPQA real dataset with cloud reflection enabled",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",  # ONLY real dataset
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

    # Save config
    config_dir = Path("./experiments/dtpqa-integration/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config.save(config_dir / f"{config.run_id}_config.json")

    runner = ExperimentRunner(config)
    status = runner.run(resume=resume)

    print(f"\n{'='*70}")
    print(f"REFLECTION Complete: {status.run_id}")
    print(f"State: {status.state}")
    print(f"Completed: {status.completed_cases}/{status.total_cases}")
    print(f"{'='*70}\n")

    return config.run_id


def run_cloud_only_experiment(limit: int | None = None, resume: bool = True) -> str:
    """Run cloud-only experiment using the cloud model with the edge prompt."""
    print("\n" + "="*70)
    print("INNER LOOP: Running CLOUD-ONLY experiment")
    print("Prediction: Single-pass cloud model gives the upper single-model baseline")
    print("="*70 + "\n")

    config = ExperimentConfig(
        name="dtpqa-real-cloud-only",
        description="Cloud-only experiment on DTPQA real dataset",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
            cloud_model="Pro/moonshotai/Kimi-K2.5",
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",
            question_type="category_1",
            limit=limit,
        ),
        execution_mode="cloud_only",
        enable_reflection=False,
        enable_dtpqa_people_reflection=False,
        batch_size=1,
        request_timeout_seconds=300,
        clean_skill_store=True,
    )

    config_dir = Path("./experiments/dtpqa-integration/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config.save(config_dir / f"{config.run_id}_config.json")

    runner = ExperimentRunner(config)
    status = runner.run(resume=resume)

    print(f"\n{'='*70}")
    print(f"CLOUD-ONLY Complete: {status.run_id}")
    print(f"State: {status.state}")
    print(f"Completed: {status.completed_cases}/{status.total_cases}")
    print(f"{'='*70}\n")

    return config.run_id


def run_ablation_studies() -> list[str]:
    """Run ablation studies for hyperparameter tuning."""
    print("\n" + "="*70)
    print("INNER LOOP: Running ABLATION studies")
    print("Systematically varying key hyperparameters")
    print("="*70 + "\n")

    runner = LargeScaleBatchRunner(
        output_dir=Path("./experiments/dtpqa-integration/results"),
        max_parallel=1,
    )

    experiments = runner.create_ablation_studies()

    batch_status = runner.run_batch(
        experiments=experiments,
        batch_id=f"dtpqa_real_ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    return [
        exp_id for exp_id, exp in batch_status.experiment_status.items()
        if exp.get("state") == "completed"
    ]


def run_comprehensive_batch() -> list[str]:
    """Run comprehensive batch including baseline, reflection, and ablations."""
    print("\n" + "="*70)
    print("AUTORESEARCH: Running COMPREHENSIVE batch")
    print("Following two-loop architecture for systematic investigation")
    print("="*70 + "\n")

    runner = LargeScaleBatchRunner(
        output_dir=Path("./experiments/dtpqa-integration/results"),
        max_parallel=1,
    )

    # Create all experiments
    experiments = []

    # Main experiments
    experiments.extend(runner.create_dtpqa_real_experiments(
        model_variants=["Qwen/Qwen3.5-9B"],
        execution_modes=["edge_only", "cloud_only", "hybrid"],
        sample_limits=[None],  # Full dataset
    ))

    # Ablation studies (smaller sample for speed)
    # experiments.extend(runner.create_ablation_studies())

    batch_status = runner.run_batch(
        experiments=experiments,
        batch_id=f"dtpqa_real_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    return [
        exp_id for exp_id, exp in batch_status.experiment_status.items()
        if exp.get("state") == "completed"
    ]


def monitor_experiment(run_id: str):
    """Monitor a running experiment."""
    monitor = ExperimentMonitor()
    monitor.watch(run_id, interval_seconds=10.0)


def generate_report(run_ids: list[str], output_dir: Path | None = None):
    """Generate academic report from completed runs."""
    print("\n" + "="*70)
    print("OUTER LOOP: Generating research report")
    print("Synthesizing results into academic narrative")
    print("="*70 + "\n")

    generator = ReportGenerator()

    output_dir = output_dir or Path("./experiments/dtpqa-integration/report")
    report_path = generator.generate_full_report(
        run_ids=run_ids,
        output_dir=output_dir,
        title="DTPQA Real Dataset: Training-Free Edge-Cloud VLM Evaluation",
    )

    print(f"\nReport generated: {report_path}")

    # Also generate markdown summary
    summary_path = output_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"""# DTPQA Real Dataset Experiment Summary

Generated: {datetime.now().isoformat()}

## Experiments Included

""")
        for run_id in run_ids:
            f.write(f"- `{run_id}`\n")

        f.write("\n## Quick Results\n\n")

        monitor = ExperimentMonitor()
        for run_id in run_ids:
            metrics = monitor.analyze(run_id)
            f.write(f"\n### {run_id}\n\n")
            f.write(f"- Accuracy: {metrics.exact_match_accuracy:.3f}\n")
            f.write(f"- Judge Score: {metrics.judge_score_mean:.1f}\n")
            f.write(f"- Latency: {metrics.mean_latency_ms/1000:.1f}s\n")

    print(f"Summary saved: {summary_path}")


def outer_loop_analysis(run_ids: list[str]) -> dict:
    """
    Perform outer loop analysis on completed experiments.

    This is where research synthesis happens - finding patterns,
    updating findings, and deciding next direction.
    """
    print("\n" + "="*70)
    print("OUTER LOOP: Analysis and Synthesis")
    print("Reviewing results, finding patterns, updating understanding")
    print("="*70 + "\n")

    monitor = ExperimentMonitor()

    # Collect all metrics
    all_metrics = {rid: monitor.analyze(rid) for rid in run_ids}

    # Find patterns
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "experiments_analyzed": len(run_ids),
        "patterns": {},
        "recommendations": [],
    }

    # Compare baseline vs reflection
    baseline_runs = [rid for rid, metrics in all_metrics.items() if metrics.execution_mode == "edge_only"]
    cloud_only_runs = [rid for rid, metrics in all_metrics.items() if metrics.execution_mode == "cloud_only"]
    reflection_runs = [rid for rid, metrics in all_metrics.items() if metrics.execution_mode == "hybrid"]

    if baseline_runs and reflection_runs:
        baseline_acc = all_metrics[baseline_runs[0]].exact_match_accuracy
        reflection_acc = all_metrics[reflection_runs[0]].exact_match_accuracy

        analysis["patterns"]["reflection_effect"] = {
            "baseline_accuracy": baseline_acc,
            "reflection_accuracy": reflection_acc,
            "delta": reflection_acc - baseline_acc,
            "percent_change": ((reflection_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0,
        }

        if reflection_acc > baseline_acc:
            analysis["recommendations"].append(
                "Reflection improves accuracy - consider DEEPENing with more granular trigger conditions"
            )
        else:
            analysis["recommendations"].append(
                "Reflection did not improve accuracy - consider PIVOT to different intervention strategy"
            )

    if baseline_runs and cloud_only_runs:
        baseline_acc = all_metrics[baseline_runs[0]].exact_match_accuracy
        cloud_acc = all_metrics[cloud_only_runs[0]].exact_match_accuracy
        analysis["patterns"]["cloud_only_effect"] = {
            "edge_accuracy": baseline_acc,
            "cloud_accuracy": cloud_acc,
            "delta": cloud_acc - baseline_acc,
        }

    # Distance-stratified analysis
    far_accuracies = {
        rid: m.distance_accuracy.get("far", 0)
        for rid, m in all_metrics.items()
    }
    if far_accuracies:
        best_far_run = max(far_accuracies, key=far_accuracies.get)
        analysis["patterns"]["far_range_performance"] = {
            "best_run": best_far_run,
            "best_accuracy": far_accuracies[best_far_run],
        }

        if far_accuracies[best_far_run] < 0.5:
            analysis["recommendations"].append(
                "Far-range accuracy remains low (<50%) - need BROADENing with specialized far-range interventions"
            )

    # Save analysis
    analysis_path = Path("./experiments/dtpqa-integration/results/outer_loop_analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved: {analysis_path}")
    print(f"\nKey Findings:")
    for pattern_name, pattern_data in analysis["patterns"].items():
        print(f"  - {pattern_name}: {pattern_data}")
    print(f"\nRecommendations:")
    for rec in analysis["recommendations"]:
        print(f"  - {rec}")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Large-scale automated experiment runner for DTPQA real dataset"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "cloud_only", "reflection", "ablation", "comprehensive", "monitor", "report", "analyze"],
        default="baseline",
        help="Experiment mode to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Run IDs for report generation or analysis",
    )
    parser.add_argument(
        "--watch",
        type=str,
        help="Watch/monitor a specific run ID",
    )

    args = parser.parse_args()

    if args.watch:
        monitor_experiment(args.watch)
        return

    if args.mode == "baseline":
        run_id = run_baseline_experiment(limit=args.limit, resume=args.resume)
        print(f"\nTo monitor: python {__file__} --watch {run_id}")

    elif args.mode == "reflection":
        run_id = run_reflection_experiment(limit=args.limit, resume=args.resume)
        print(f"\nTo monitor: python {__file__} --watch {run_id}")

    elif args.mode == "cloud_only":
        run_id = run_cloud_only_experiment(limit=args.limit, resume=args.resume)
        print(f"\nTo monitor: python {__file__} --watch {run_id}")

    elif args.mode == "ablation":
        completed = run_ablation_studies()
        print(f"\nCompleted ablations: {completed}")

    elif args.mode == "comprehensive":
        completed = run_comprehensive_batch()
        print(f"\nCompleted experiments: {completed}")

        # Auto-generate report
        if completed:
            generate_report(completed)
            outer_loop_analysis(completed)

    elif args.mode == "monitor":
        if not args.run_ids:
            print("Error: --run-ids required for monitoring")
            sys.exit(1)
        for run_id in args.run_ids:
            monitor_experiment(run_id)

    elif args.mode == "report":
        if not args.run_ids:
            print("Error: --run-ids required for report generation")
            sys.exit(1)
        generate_report(args.run_ids)

    elif args.mode == "analyze":
        if not args.run_ids:
            print("Error: --run-ids required for analysis")
            sys.exit(1)
        outer_loop_analysis(args.run_ids)


if __name__ == "__main__":
    main()
