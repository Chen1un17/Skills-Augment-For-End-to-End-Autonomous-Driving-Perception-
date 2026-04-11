"""Iterative optimizer for model improvement based on experiment results.

This implements the autoresearch outer loop - taking experiment results,
analyzing failures, and generating improved configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy

from .config import ExperimentConfig, ModelConfig
from .monitor import ExperimentMonitor, ExperimentMetrics
from ad_cornercase.evaluation.metrics import canonicalize_answer, exact_match
from ad_cornercase.schemas.evaluation import CasePredictionRecord


@dataclass
class OptimizationIteration:
    """Record of a single optimization iteration."""
    iteration: int
    parent_run_id: str | None
    config: ExperimentConfig
    results: ExperimentMetrics | None = None
    improvements: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[str] = field(default_factory=list)


class IterativeOptimizer:
    """
    Iterative optimizer that improves model configuration based on results.

    Following autoresearch principles:
    - DEEPEN: Supported results raise follow-up questions
    - BROADEN: Adjacent questions are tested
    - PIVOT: When assumptions break, change direction
    """

    def __init__(self, output_dir: Path = Path("./experiments/dtpqa-integration/optimization")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = ExperimentMonitor()
        self.iterations: list[OptimizationIteration] = []

    def analyze_failures(self, run_id: str) -> dict[str, Any]:
        """Analyze failure patterns in experiment results."""
        metrics = self.monitor.analyze(run_id)

        # Load detailed results
        artifacts_dir = self.monitor.artifacts_dir
        replay_path = artifacts_dir / run_id / "predictions.jsonl"

        failures = {
            "by_distance": {"near": [], "mid": [], "far": [], "unknown": []},
            "by_answer_type": {"false_negative": 0, "false_positive": 0, "other": 0},
            "common_patterns": [],
        }

        if not replay_path.exists():
            return failures

        with open(replay_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = CasePredictionRecord.model_validate_json(line)
                if not exact_match(record):
                    distance = str(record.metadata.get("distance_group") or "unknown")
                    predicted = ""
                    if record.final_result.qa_report:
                        predicted = record.final_result.qa_report[0].answer.lower()
                    ground_truth = record.ground_truth_answer.lower()
                    answer_options = record.metadata.get("answer_options")
                    normalized_ground_truth = canonicalize_answer(
                        record.ground_truth_answer,
                        answer_options if isinstance(answer_options, list) else None,
                    )

                    failures["by_distance"][distance].append(record.model_dump(mode="json"))

                    # Classify error type
                    gt_yes = normalized_ground_truth == "yes" or "true" in ground_truth
                    pred_no = predicted.startswith("no") or "false" in predicted

                    if gt_yes and pred_no:
                        failures["by_answer_type"]["false_negative"] += 1
                    elif not gt_yes and not pred_no:
                        failures["by_answer_type"]["false_positive"] += 1
                    else:
                        failures["by_answer_type"]["other"] += 1

        # Identify patterns
        total_failures = sum(len(v) for v in failures["by_distance"].values())

        # Check for distance-specific issues
        for dist in ["near", "mid", "far", "unknown"]:
            dist_failures = len(failures["by_distance"][dist])
            if dist_failures > 0:
                dist_total = metrics.distance_counts.get(dist, 0)
                if dist_total > 0:
                    failure_rate = dist_failures / dist_total
                    if failure_rate > 0.5:
                        failures["common_patterns"].append(
                            f"High failure rate in {dist} distance: {failure_rate:.1%}"
                        )

        # Check for false negative bias (the main issue found in research)
        fn_rate = failures["by_answer_type"]["false_negative"] / max(total_failures, 1)
        if fn_rate > 0.5:
            failures["common_patterns"].append(
                f"Strong false-negative bias: {fn_rate:.1%} of failures are false negatives"
            )

        return failures

    def generate_hypotheses(self, failures: dict[str, Any]) -> list[str]:
        """Generate hypotheses based on failure analysis."""
        hypotheses = []

        for pattern in failures.get("common_patterns", []):
            if "far distance" in pattern.lower():
                hypotheses.extend([
                    "H_far_1: Small objects at far distance need higher-resolution crops",
                    "H_far_2: Reflection trigger threshold is too high for far-range cases",
                    "H_far_3: Edge model lacks detail for distant objects - need cloud intervention",
                ])

            if "false-negative" in pattern.lower():
                hypotheses.extend([
                    "H_fn_1: Edge model is over-conservative - adjust prompting",
                    "H_fn_2: Reflection not triggering on confident wrong answers",
                    "H_fn_3: Need broader category_1 reflection trigger",
                ])

            if "near distance" in pattern.lower():
                hypotheses.extend([
                    "H_near_1: Near objects may be too large - need different crop strategy",
                    "H_near_2: Occlusion handling needed for close objects",
                ])

        return hypotheses

    def propose_improvements(self, base_config: ExperimentConfig,
                            failures: dict[str, Any],
                            hypotheses: list[str]) -> list[ExperimentConfig]:
        """Propose improved configurations based on analysis."""
        improvements = []

        # Strategy 1: Lower reflection threshold for far-range
        if any("far" in p.lower() for p in failures.get("common_patterns", [])):
            config = deepcopy(base_config)
            config.entropy_threshold = 0.5  # Lower threshold
            config.name = f"{base_config.name}_low_threshold"
            improvements.append(config)

        # Strategy 2: Increase edge model capacity
        if any("false-negative" in p.lower() for p in failures.get("common_patterns", [])):
            config = deepcopy(base_config)
            config.models.edge_max_completion_tokens = 1024  # More tokens
            config.name = f"{base_config.name}_more_tokens"
            improvements.append(config)

        # Strategy 3: Enable aggressive reflection
        if failures["by_answer_type"]["false_negative"] > failures["by_answer_type"]["false_positive"]:
            config = deepcopy(base_config)
            config.enable_reflection = True
            config.enable_dtpqa_people_reflection = True
            config.name = f"{base_config.name}_aggressive_reflection"
            improvements.append(config)

        # Strategy 4: Combined approach
        if len(failures["common_patterns"]) >= 2:
            config = deepcopy(base_config)
            config.entropy_threshold = 0.5
            config.models.edge_max_completion_tokens = 1024
            config.enable_reflection = True
            config.name = f"{base_config.name}_combined_improvements"
            improvements.append(config)

        return improvements

    def optimize(self, run_id: str, direction: str = "auto") -> list[ExperimentConfig]:
        """
        Optimize based on experiment results.

        Args:
            run_id: Base experiment to optimize from
            direction: "deepen", "broaden", "pivot", or "auto"

        Returns:
            List of improved configurations to try
        """
        print(f"\n{'='*70}")
        print(f"OUTER LOOP: Optimization Iteration")
        print(f"Analyzing: {run_id}")
        print(f"Direction: {direction}")
        print(f"{'='*70}\n")

        # Analyze results
        metrics = self.monitor.analyze(run_id)
        failures = self.analyze_failures(run_id)
        hypotheses = self.generate_hypotheses(failures)

        # Record this iteration
        iteration = len(self.iterations) + 1
        iter_record = OptimizationIteration(
            iteration=iteration,
            parent_run_id=run_id,
            config=ExperimentConfig(name=f"iteration_{iteration}"),
            results=metrics,
            hypotheses=hypotheses,
        )
        self.iterations.append(iter_record)

        # Print analysis
        print("Failure Analysis:")
        print(f"  Total failures by distance:")
        for dist in ["near", "mid", "far", "unknown"]:
            count = len(failures["by_distance"][dist])
            print(f"    {dist}: {count}")

        print(f"\nError Types:")
        for error_type, count in failures["by_answer_type"].items():
            print(f"  {error_type}: {count}")

        print(f"\nIdentified Patterns:")
        for pattern in failures["common_patterns"]:
            print(f"  - {pattern}")

        print(f"\nGenerated Hypotheses:")
        for hypothesis in hypotheses:
            print(f"  - {hypothesis}")

        # Determine direction if auto
        if direction == "auto":
            if metrics.exact_match_accuracy < 0.5:
                direction = "pivot"  # Need major change
            elif len(failures["common_patterns"]) > 0:
                direction = "deepen"  # Follow up on specific issues
            else:
                direction = "broaden"  # Test variations

        print(f"\nSelected Direction: {direction.upper()}")

        # Generate improvements
        base_config = ExperimentConfig(
            name="optimized",
            models=ModelConfig(edge_model="Qwen/Qwen3.5-9B", edge_max_completion_tokens=512),
        )

        improvements = self.propose_improvements(base_config, failures, hypotheses)

        print(f"\nProposed Improvements ({len(improvements)} configs):")
        for i, config in enumerate(improvements, 1):
            print(f"  {i}. {config.name}")
            print(f"     - reflection: {config.enable_reflection}")
            print(f"     - threshold: {config.entropy_threshold}")
            print(f"     - tokens: {config.models.edge_max_completion_tokens}")

        # Save optimization record
        self._save_optimization_record()

        return improvements

    def _save_optimization_record(self) -> None:
        """Save optimization history."""
        record_path = self.output_dir / "optimization_history.json"

        record = {
            "iterations": [
                {
                    "iteration": i.iteration,
                    "parent_run_id": i.parent_run_id,
                    "config_name": i.config.name,
                    "results": i.results.to_dict() if i.results else None,
                    "hypotheses": i.hypotheses,
                }
                for i in self.iterations
            ]
        }

        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print(f"\nOptimization record saved: {record_path}")


class AutomatedResearchLoop:
    """
    Fully automated research loop following autoresearch principles.

    This runs the two-loop architecture autonomously:
    - Inner loop: Run experiments
    - Outer loop: Analyze and optimize
    """

    def __init__(
        self,
        max_iterations: int = 5,
        accuracy_target: float = 0.7,
        output_dir: Path = Path("./experiments/dtpqa-integration/auto_research"),
    ):
        self.max_iterations = max_iterations
        self.accuracy_target = accuracy_target
        self.output_dir = output_dir
        self.optimizer = IterativeOptimizer(output_dir)
        self.completed_runs: list[str] = []

    def run(
        self,
        initial_config: ExperimentConfig | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Run automated research loop until target or max iterations.

        Args:
            initial_config: Starting configuration
            limit: Sample limit for faster iteration

        Returns:
            Summary of research outcomes
        """
        from .runner import ExperimentRunner

        initial_config = initial_config or ExperimentConfig(
            name="auto_research_baseline",
            dataset=ExperimentConfig().dataset,
        )
        initial_config.dataset.limit = limit

        best_accuracy = 0.0
        best_run_id = None

        print(f"\n{'='*70}")
        print(f"AUTOMATED RESEARCH LOOP")
        print(f"Target accuracy: {self.accuracy_target:.1%}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"{'='*70}\n")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*70}\n")

            # Run experiment
            runner = ExperimentRunner(initial_config)
            status = runner.run(resume=True)

            if status.state != "completed":
                print(f"[WARN] Experiment did not complete: {status.state}")
                continue

            self.completed_runs.append(initial_config.run_id)

            # Analyze results
            metrics = self.optimizer.monitor.analyze(initial_config.run_id)

            print(f"\nResults:")
            print(f"  Accuracy: {metrics.exact_match_accuracy:.3f}")
            print(f"  Judge Score: {metrics.judge_score_mean:.1f}")

            # Check target
            if metrics.exact_match_accuracy >= self.accuracy_target:
                print(f"\n{'='*70}")
                print(f"TARGET REACHED! Accuracy: {metrics.exact_match_accuracy:.3f}")
                print(f"{'='*70}")
                best_run_id = initial_config.run_id
                best_accuracy = metrics.exact_match_accuracy
                break

            # Track best
            if metrics.exact_match_accuracy > best_accuracy:
                best_accuracy = metrics.exact_match_accuracy
                best_run_id = initial_config.run_id

            # Optimize for next iteration
            if iteration < self.max_iterations:
                improvements = self.optimizer.optimize(
                    initial_config.run_id,
                    direction="auto"
                )

                if improvements:
                    initial_config = improvements[0]  # Try first improvement
                    initial_config.dataset.limit = limit
                else:
                    print("[WARN] No improvements generated, stopping")
                    break

        # Summary
        summary = {
            "iterations": len(self.completed_runs),
            "best_run_id": best_run_id,
            "best_accuracy": best_accuracy,
            "target_reached": best_accuracy >= self.accuracy_target,
            "all_runs": self.completed_runs,
        }

        # Save summary
        summary_path = self.output_dir / "research_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"RESEARCH COMPLETE")
        print(f"Best accuracy: {best_accuracy:.3f}")
        print(f"Best run: {best_run_id}")
        print(f"Target reached: {summary['target_reached']}")
        print(f"{'='*70}")

        return summary
