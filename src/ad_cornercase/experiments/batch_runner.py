"""Batch experiment runner for large-scale automated research."""

from __future__ import annotations

import os
import json
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Iterator, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config import ExperimentConfig, ModelConfig, DatasetConfig


@dataclass
class BatchExperiment:
    """Single experiment configuration in a batch."""
    name: str
    config: ExperimentConfig
    priority: int = 0  # Higher = run first


@dataclass
class BatchStatus:
    """Status of batch experiment run."""
    batch_id: str
    total_experiments: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0
    pending: int = 0
    start_time: str | None = None
    end_time: str | None = None
    experiment_status: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total_experiments": self.total_experiments,
            "completed": self.completed,
            "failed": self.failed,
            "running": self.running,
            "pending": self.pending,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "experiment_status": self.experiment_status,
        }


class LargeScaleBatchRunner:
    """Runner for large-scale batch experiments with real-time monitoring."""

    def __init__(
        self,
        output_dir: Path = Path("./experiments/dtpqa-integration/results"),
        max_parallel: int = 1,  # Sequential by default for API rate limits
    ):
        self.output_dir = output_dir
        self.max_parallel = max_parallel
        self._stop_requested = False
        self._lock = threading.Lock()

    def create_dtpqa_real_experiments(
        self,
        model_variants: list[str] | None = None,
        execution_modes: list[str] | None = None,
        sample_limits: list[int | None] | None = None,
    ) -> list[BatchExperiment]:
        """
        Create comprehensive DTPQA real dataset experiments.

        Args:
            model_variants: List of edge models to test
            execution_modes: Execution modes to test
            sample_limits: List of sample limits for testing (None = full dataset)
        """
        model_variants = model_variants or [
            "Qwen/Qwen3.5-9B",
        ]
        execution_modes = execution_modes or ["edge_only", "cloud_only", "hybrid"]
        sample_limits = sample_limits or [None]  # Full dataset by default

        experiments = []

        for model in model_variants:
            for execution_mode in execution_modes:
                for limit in sample_limits:
                    # Build experiment name
                    model_short = model.split("/")[-1].lower().replace("-", "_")
                    mode_tag = execution_mode.replace("_", "-")
                    limit_tag = f"n{limit}" if limit else "full"

                    name = f"dtpqa_real_{model_short}_{mode_tag}_{limit_tag}"

                    config = ExperimentConfig(
                        name=name,
                        description=f"DTPQA real dataset with {model}, mode={execution_mode}",
                        models=ModelConfig(
                            edge_model=model,
                            edge_max_completion_tokens=512,
                        ),
                        dataset=DatasetConfig(
                            benchmark="dtpqa",
                            subset="real",
                            question_type="category_1",
                            limit=limit,
                        ),
                        execution_mode=execution_mode,
                        enable_reflection=execution_mode == "hybrid",
                        enable_dtpqa_people_reflection=execution_mode == "hybrid",
                        batch_size=1,
                        request_timeout_seconds=300,
                    )

                    experiments.append(BatchExperiment(
                        name=name,
                        config=config,
                        priority=2 if execution_mode == "edge_only" else 1 if execution_mode == "cloud_only" else 0,
                    ))

        return experiments

    def create_ablation_studies(self) -> list[BatchExperiment]:
        """Create ablation study experiments for DTPQA synth."""
        experiments = []
        base_model = "Qwen/Qwen3.5-9B"

        # Ablation 1: Reflection trigger threshold
        for threshold in [0.5, 1.0, 1.5]:
            name = f"dtpqa_synth_ablation_threshold_{threshold}"
            config = ExperimentConfig(
                name=name,
                description=f"Ablation: entropy threshold = {threshold}",
                models=ModelConfig(edge_model=base_model, edge_max_completion_tokens=512),
                dataset=DatasetConfig(benchmark="dtpqa", subset="synth", question_type="category_1", limit=100),
                execution_mode="hybrid",
                enable_reflection=True,
                entropy_threshold=threshold,
            )
            experiments.append(BatchExperiment(name=name, config=config, priority=2))

        # Ablation 2: Edge model token limits
        for tokens in [256, 512, 1024]:
            name = f"dtpqa_synth_ablation_tokens_{tokens}"
            config = ExperimentConfig(
                name=name,
                description=f"Ablation: edge max tokens = {tokens}",
                models=ModelConfig(edge_model=base_model, edge_max_completion_tokens=tokens),
                dataset=DatasetConfig(benchmark="dtpqa", subset="synth", question_type="category_1", limit=100),
                execution_mode="edge_only",
                enable_reflection=False,
            )
            experiments.append(BatchExperiment(name=name, config=config, priority=2))

        # Ablation 3: Cloud model variants
        for cloud_model in ["Pro/moonshotai/Kimi-K2.5"]:
            name = f"dtpqa_synth_ablation_cloud_{cloud_model.split('/')[-1].lower()}"
            config = ExperimentConfig(
                name=name,
                description=f"Ablation: cloud model = {cloud_model}",
                models=ModelConfig(
                    edge_model=base_model,
                    edge_max_completion_tokens=512,
                    cloud_model=cloud_model,
                ),
                dataset=DatasetConfig(benchmark="dtpqa", subset="synth", question_type="category_1", limit=100),
                enable_reflection=True,
            )
            experiments.append(BatchExperiment(name=name, config=config, priority=2))

        return experiments

    def _run_single_experiment(
        self,
        experiment: BatchExperiment,
        batch_status: BatchStatus,
    ) -> dict[str, Any]:
        """Run a single experiment and return results."""
        from .runner import ExperimentRunner

        run_id = experiment.config.run_id

        with self._lock:
            batch_status.experiment_status[run_id] = {
                "state": "running",
                "start_time": datetime.now().isoformat(),
            }
            batch_status.running += 1
            batch_status.pending -= 1

        print(f"\n[{'='*70}]")
        print(f"[BATCH] Starting: {experiment.name}")
        print(f"[BATCH] Run ID: {run_id}")
        print(f"[{'='*70}]")

        try:
            runner = ExperimentRunner(experiment.config)
            status = runner.run(resume=True)

            result = {
                "success": status.state == "completed",
                "state": status.state,
                "completed_cases": status.completed_cases,
                "failed_cases": status.failed_cases,
                "metrics": status.metrics,
            }

            with self._lock:
                batch_status.experiment_status[run_id].update({
                    "state": status.state,
                    "end_time": datetime.now().isoformat(),
                    "result": result,
                })
                batch_status.running -= 1
                if status.state == "completed":
                    batch_status.completed += 1
                else:
                    batch_status.failed += 1

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Experiment {experiment.name} failed: {error_msg}")

            with self._lock:
                batch_status.experiment_status[run_id].update({
                    "state": "failed",
                    "end_time": datetime.now().isoformat(),
                    "error": error_msg,
                })
                batch_status.running -= 1
                batch_status.failed += 1

            return {"success": False, "error": error_msg}

    def run_batch(
        self,
        experiments: list[BatchExperiment],
        batch_id: str | None = None,
        monitor_interval: float = 30.0,
    ) -> BatchStatus:
        """
        Run a batch of experiments with monitoring.

        Args:
            experiments: List of experiments to run
            batch_id: Optional batch identifier
            monitor_interval: Seconds between progress updates
        """
        batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_status = BatchStatus(
            batch_id=batch_id,
            total_experiments=len(experiments),
            pending=len(experiments),
        )
        batch_status.start_time = datetime.now().isoformat()

        # Sort by priority (higher first)
        experiments = sorted(experiments, key=lambda e: -e.priority)

        print(f"\n[{'='*70}]")
        print(f"[BATCH] Starting batch: {batch_id}")
        print(f"[BATCH] Total experiments: {len(experiments)}")
        print(f"[BATCH] Max parallel: {self.max_parallel}")
        print(f"[{'='*70}]\n")

        # Save batch configuration
        batch_config_path = self.output_dir / f"{batch_id}_config.json"
        batch_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(batch_config_path, "w", encoding="utf-8") as f:
            json.dump({
                "batch_id": batch_id,
                "experiments": [
                    {
                        "name": e.name,
                        "run_id": e.config.run_id,
                        "priority": e.priority,
                    }
                    for e in experiments
                ],
            }, f, indent=2)

        # Run experiments
        if self.max_parallel == 1:
            # Sequential execution
            for exp in experiments:
                if self._stop_requested:
                    print("[BATCH] Stop requested, pausing batch")
                    break
                self._run_single_experiment(exp, batch_status)
                self._save_batch_status(batch_status)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                futures = {
                    executor.submit(self._run_single_experiment, exp, batch_status): exp
                    for exp in experiments
                }

                for future in as_completed(futures):
                    if self._stop_requested:
                        print("[BATCH] Stop requested, waiting for running experiments...")
                        break

                    exp = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Experiment {exp.name} raised: {e}")

                    self._save_batch_status(batch_status)

        # Complete
        batch_status.end_time = datetime.now().isoformat()
        self._save_batch_status(batch_status)

        print(f"\n[{'='*70}]")
        print(f"[BATCH] Batch complete: {batch_id}")
        print(f"[BATCH] Completed: {batch_status.completed}/{batch_status.total_experiments}")
        print(f"[BATCH] Failed: {batch_status.failed}/{batch_status.total_experiments}")
        print(f"[{'='*70}]")

        return batch_status

    def _save_batch_status(self, batch_status: BatchStatus) -> None:
        """Save batch status to file."""
        status_path = self.output_dir / f"{batch_status.batch_id}_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(batch_status.to_dict(), f, indent=2)

    def stop(self) -> None:
        """Request graceful stop."""
        self._stop_requested = True

    def generate_batch_report(self, batch_id: str) -> Path:
        """Generate comprehensive batch report."""
        from .report import ReportGenerator

        status_path = self.output_dir / f"{batch_id}_status.json"
        if not status_path.exists():
            raise FileNotFoundError(f"Batch status not found: {status_path}")

        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)

        run_ids = [
            exp["result"].get("run_id", exp_id)
            for exp_id, exp in status["experiment_status"].items()
            if exp.get("state") == "completed"
        ]

        if not run_ids:
            print("[WARN] No completed experiments to report")
            return None

        report_dir = self.output_dir / f"{batch_id}_report"
        generator = ReportGenerator()
        report_path = generator.generate_full_report(
            run_ids=run_ids,
            output_dir=report_dir,
            title=f"Batch Report: {batch_id}",
        )

        return report_path
