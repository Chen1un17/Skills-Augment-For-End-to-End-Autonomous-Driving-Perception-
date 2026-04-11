"""Universal evaluator that works across benchmarks.

Provides standardized metrics and evaluation procedures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any
import json
from datetime import datetime


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1_SCORE = auto()
    EXACT_MATCH = auto()
    SEMANTIC_SIMILARITY = auto()
    JUDGE_SCORE = auto()
    LATENCY = auto()
    COST = auto()


@dataclass
class MetricResult:
    """Result for a single metric."""
    metric_type: MetricType
    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    benchmark_name: str
    num_samples: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    stratified_results: dict[str, dict[str, MetricResult]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
            "metrics": {
                name: {
                    "value": metric.value,
                    "type": metric.metric_type.name,
                    "details": metric.details,
                }
                for name, metric in self.metrics.items()
            },
            "stratified_results": self.stratified_results,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Save results to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class UniversalEvaluator:
    """Universal evaluator for perception tasks.

    Works across different benchmarks with configurable metrics.
    """

    def __init__(
        self,
        metrics: list[MetricType] | None = None,
        stratify_by: list[str] | None = None,
    ):
        self.metrics = metrics or [MetricType.ACCURACY, MetricType.LATENCY]
        self.stratify_by = stratify_by or []

    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        benchmark_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate a set of predictions.

        Args:
            predictions: List of prediction results
            benchmark_name: Name of the benchmark

        Returns:
            Evaluation results
        """
        result = EvaluationResult(
            benchmark_name=benchmark_name,
            num_samples=len(predictions),
        )

        # Compute each metric
        for metric_type in self.metrics:
            metric_result = self._compute_metric(predictions, metric_type)
            result.metrics[metric_result.name] = metric_result

        # Compute stratified results
        for stratify_key in self.stratify_by:
            stratified = self._stratify_and_evaluate(predictions, stratify_key)
            result.stratified_results[stratify_key] = stratified

        return result

    def _compute_metric(
        self,
        predictions: list[dict[str, Any]],
        metric_type: MetricType,
    ) -> MetricResult:
        """Compute a single metric."""
        if metric_type == MetricType.ACCURACY:
            return self._compute_accuracy(predictions)
        elif metric_type == MetricType.EXACT_MATCH:
            return self._compute_exact_match(predictions)
        elif metric_type == MetricType.LATENCY:
            return self._compute_latency(predictions)
        elif metric_type == MetricType.PRECISION:
            return self._compute_precision(predictions)
        elif metric_type == MetricType.RECALL:
            return self._compute_recall(predictions)
        else:
            return MetricResult(
                metric_type=metric_type,
                name=metric_type.name.lower(),
                value=0.0,
                details={"error": "Not implemented"},
            )

    def _compute_accuracy(self, predictions: list[dict[str, Any]]) -> MetricResult:
        """Compute accuracy."""
        correct = 0
        total = len(predictions)

        for pred in predictions:
            gt = pred.get("ground_truth", {}).get("answer", "").lower()
            prediction = pred.get("prediction", {}).get("answer", "").lower()

            # Simple match - can be customized
            if gt and prediction:
                if gt in prediction or prediction in gt:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return MetricResult(
            metric_type=MetricType.ACCURACY,
            name="accuracy",
            value=accuracy,
            details={"correct": correct, "total": total},
        )

    def _compute_exact_match(self, predictions: list[dict[str, Any]]) -> MetricResult:
        """Compute exact match accuracy."""
        correct = 0
        total = len(predictions)

        for pred in predictions:
            gt = pred.get("ground_truth", {}).get("answer", "").strip().lower()
            prediction = pred.get("prediction", {}).get("answer", "").strip().lower()

            if gt == prediction:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return MetricResult(
            metric_type=MetricType.EXACT_MATCH,
            name="exact_match",
            value=accuracy,
            details={"correct": correct, "total": total},
        )

    def _compute_latency(self, predictions: list[dict[str, Any]]) -> MetricResult:
        """Compute latency statistics."""
        latencies = [
            pred.get("latency_ms", 0)
            for pred in predictions
            if "latency_ms" in pred
        ]

        if not latencies:
            return MetricResult(
                metric_type=MetricType.LATENCY,
                name="latency_ms",
                value=0.0,
                details={"error": "No latency data"},
            )

        import numpy as np

        return MetricResult(
            metric_type=MetricType.LATENCY,
            name="latency_ms",
            value=float(np.mean(latencies)),
            details={
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "std": float(np.std(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
            },
        )

    def _compute_precision(self, predictions: list[dict[str, Any]]) -> MetricResult:
        """Compute precision for binary classification."""
        tp = fp = 0

        for pred in predictions:
            gt = pred.get("ground_truth", {}).get("answer", "").lower()
            prediction = pred.get("prediction", {}).get("answer", "").lower()

            # Assume binary: yes/no
            if "yes" in prediction:
                if "yes" in gt:
                    tp += 1
                else:
                    fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return MetricResult(
            metric_type=MetricType.PRECISION,
            name="precision",
            value=precision,
            details={"true_positives": tp, "false_positives": fp},
        )

    def _compute_recall(self, predictions: list[dict[str, Any]]) -> MetricResult:
        """Compute recall for binary classification."""
        tp = fn = 0

        for pred in predictions:
            gt = pred.get("ground_truth", {}).get("answer", "").lower()
            prediction = pred.get("prediction", {}).get("answer", "").lower()

            if "yes" in gt:
                if "yes" in prediction:
                    tp += 1
                else:
                    fn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return MetricResult(
            metric_type=MetricType.RECALL,
            name="recall",
            value=recall,
            details={"true_positives": tp, "false_negatives": fn},
        )

    def _stratify_and_evaluate(
        self,
        predictions: list[dict[str, Any]],
        stratify_key: str,
    ) -> dict[str, MetricResult]:
        """Stratify predictions and compute metrics per group."""
        groups: dict[str, list[dict[str, Any]]] = {}

        for pred in predictions:
            value = pred.get("metadata", {}).get(stratify_key, "unknown")
            if value not in groups:
                groups[value] = []
            groups[value].append(pred)

        results = {}
        for group_name, group_preds in groups.items():
            # Compute accuracy for this group
            metric = self._compute_accuracy(group_preds)
            results[group_name] = metric

        return results
