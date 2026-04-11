"""Experiment monitoring and metrics tracking."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field, asdict

from ad_cornercase.evaluation.metrics import compute_skill_success_rate, exact_match
from ad_cornercase.schemas.evaluation import CasePredictionRecord


@dataclass
class DistanceMetrics:
    """Metrics grouped by distance."""
    near: dict[str, float] = field(default_factory=dict)
    mid: dict[str, float] = field(default_factory=dict)
    far: dict[str, float] = field(default_factory=dict)
    unknown: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    """Aggregated experiment metrics."""
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_mode: str = "unknown"

    # Overall metrics
    total_cases: int = 0
    exact_match_accuracy: float = 0.0
    judge_score_mean: float = 0.0
    judge_score_std: float = 0.0
    mean_latency_ms: float = 0.0

    # Distance-stratified metrics
    distance_accuracy: dict[str, float] = field(default_factory=dict)
    distance_judge_mean: dict[str, float] = field(default_factory=dict)
    distance_latency_mean: dict[str, float] = field(default_factory=dict)
    distance_counts: dict[str, int] = field(default_factory=dict)

    # Reflection metrics
    reflection_trigger_rate: float = 0.0
    skill_success_rate: float = 0.0

    # Answer distribution
    answer_distribution: dict[str, int] = field(default_factory=dict)
    ground_truth_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentMonitor:
    """Monitor and analyze experiment results."""

    def __init__(self, artifacts_dir: Path = Path("./data/artifacts")):
        self.artifacts_dir = artifacts_dir

    def _load_records(self, run_id: str) -> list[CasePredictionRecord]:
        """Load prediction records from artifact directory."""
        predictions_path = self.artifacts_dir / run_id / "predictions.jsonl"
        if not predictions_path.exists():
            return []

        records: list[CasePredictionRecord] = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(CasePredictionRecord.model_validate_json(line))
        return records

    def _load_metrics(self, run_id: str) -> dict[str, Any] | None:
        """Load evaluation metrics if available."""
        metrics_path = self.artifacts_dir / run_id / "metrics.json"
        if not metrics_path.exists():
            return None

        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def analyze(self, run_id: str) -> ExperimentMetrics:
        """Analyze experiment results and return metrics."""
        records = self._load_records(run_id)
        metrics_payload = self._load_metrics(run_id)

        metrics = ExperimentMetrics(run_id=run_id)

        if not records:
            return metrics

        metrics.total_cases = len(records)

        execution_modes = {str(record.metadata.get("execution_mode") or "unknown") for record in records}
        if len(execution_modes) == 1:
            metrics.execution_mode = next(iter(execution_modes))

        # Calculate overall accuracy
        exact_values = [exact_match(record) for record in records]
        metrics.exact_match_accuracy = sum(exact_values) / len(records)

        # Calculate latency
        latencies = [
            float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0))
            for record in records
        ]
        if latencies:
            metrics.mean_latency_ms = sum(latencies) / len(latencies)

        # Distance-stratified analysis
        distance_groups: dict[str, list[CasePredictionRecord]] = {
            "near": [], "mid": [], "far": [], "unknown": []
        }

        for record in records:
            distance_group = str(record.metadata.get("distance_group") or "unknown")
            if distance_group not in distance_groups:
                distance_group = "unknown"
            distance_groups[distance_group].append(record)

        for group, cases in distance_groups.items():
            if not cases:
                continue

            metrics.distance_counts[group] = len(cases)

            # Accuracy by distance
            correct = sum(exact_match(case) for case in cases)
            metrics.distance_accuracy[group] = correct / len(cases)

            # Latency by distance
            latencies = [
                float(case.metadata.get("pipeline_latency_ms", case.final_result.latency_ms or 0.0))
                for case in cases
            ]
            if latencies:
                metrics.distance_latency_mean[group] = sum(latencies) / len(latencies)

        judged_scores = [record.judge_score for record in records if record.judge_score is not None]
        if judged_scores:
            metrics.judge_score_mean = sum(judged_scores) / len(judged_scores)
            variance = sum((score - metrics.judge_score_mean) ** 2 for score in judged_scores) / len(judged_scores)
            metrics.judge_score_std = variance ** 0.5
        elif metrics_payload:
            metrics.judge_score_mean = metrics_payload.get("judge_score_mean", 0.0)

        for group, cases in distance_groups.items():
            judged_group_scores = [case.judge_score for case in cases if case.judge_score is not None]
            if judged_group_scores:
                metrics.distance_judge_mean[group] = sum(judged_group_scores) / len(judged_group_scores)

        metrics.reflection_trigger_rate = (
            sum(record.reflection_result is not None for record in records) / len(records)
        )
        metrics.skill_success_rate = compute_skill_success_rate(records, 70.0)

        # Answer distributions
        pred_answers: dict[str, int] = {}
        gt_answers: dict[str, int] = {}
        for record in records:
            pred = (
                record.final_result.qa_report[0].answer
                if record.final_result.qa_report
                else "unknown"
            )[:20]
            gt = record.ground_truth_answer[:20]
            pred_answers[pred] = pred_answers.get(pred, 0) + 1
            gt_answers[gt] = gt_answers.get(gt, 0) + 1

        metrics.answer_distribution = pred_answers
        metrics.ground_truth_distribution = gt_answers

        return metrics

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Compare multiple experiment runs."""
        comparison = {
            "runs": [],
            "summary": {},
        }

        all_metrics = []
        for run_id in run_ids:
            metrics = self.analyze(run_id)
            all_metrics.append(metrics)
            comparison["runs"].append({
                "run_id": run_id,
                "metrics": metrics.to_dict(),
            })

        # Summary statistics
        if all_metrics:
            comparison["summary"] = {
                "best_accuracy_run": max(all_metrics, key=lambda m: m.exact_match_accuracy).run_id,
                "best_judge_run": max(all_metrics, key=lambda m: m.judge_score_mean).run_id,
                "fastest_run": min(all_metrics, key=lambda m: m.mean_latency_ms).run_id,
                "accuracy_range": {
                    "min": min(m.exact_match_accuracy for m in all_metrics),
                    "max": max(m.exact_match_accuracy for m in all_metrics),
                    "mean": sum(m.exact_match_accuracy for m in all_metrics) / len(all_metrics),
                },
            }

        return comparison

    def watch(self, run_id: str, interval_seconds: float = 10.0) -> None:
        """Watch experiment progress in real-time."""
        import time

        print(f"[WATCH] Monitoring run: {run_id}")
        print(f"[WATCH] Press Ctrl+C to stop watching\n")

        last_completed = 0
        try:
            while True:
                metrics = self.analyze(run_id)
                status_path = self.artifacts_dir / run_id / "experiment_status.json"

                status = {"state": "unknown"}
                if status_path.exists():
                    with open(status_path, "r", encoding="utf-8") as f:
                        status = json.load(f)

                # Only print if there's new progress
                if metrics.total_cases > last_completed or status.get("state") != "running":
                    last_completed = metrics.total_cases
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status.get('state', 'unknown')}")
                    print(f"  Progress: {status.get('progress_pct', 0):.1f}% ({metrics.total_cases} cases)")
                    print(f"  Accuracy: {metrics.exact_match_accuracy:.3f}")
                    print(f"  Judge Mean: {metrics.judge_score_mean:.2f}")
                    print(f"  Latency: {metrics.mean_latency_ms/1000:.1f}s")

                    if metrics.distance_accuracy:
                        print(f"  Distance Accuracy: near={metrics.distance_accuracy.get('near', 0):.2f}, "
                              f"mid={metrics.distance_accuracy.get('mid', 0):.2f}, "
                              f"far={metrics.distance_accuracy.get('far', 0):.2f}")

                if status.get("state") in ["completed", "failed", "paused"]:
                    print(f"\n[WATCH] Experiment {status['state']}")
                    break

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n[WATCH] Stopped watching")

    def export_metrics(self, run_id: str, output_path: Path) -> None:
        """Export metrics to JSON file."""
        metrics = self.analyze(run_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)

        print(f"[INFO] Exported metrics to {output_path}")
