"""Automated experiment runner with checkpointing and monitoring."""

from __future__ import annotations

import os
import json
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

from .config import ExperimentConfig


@dataclass
class ExperimentStatus:
    """Status of an experiment run."""
    run_id: str
    state: str  # pending, running, paused, completed, failed
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0
    current_offset: int = 0
    start_time: str | None = None
    end_time: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "total_cases": self.total_cases,
            "completed_cases": self.completed_cases,
            "failed_cases": self.failed_cases,
            "current_offset": self.current_offset,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metrics": self.metrics,
            "progress_pct": self.progress_percentage,
        }

    @property
    def progress_percentage(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return (self.completed_cases / self.total_cases) * 100


class ExperimentRunner:
    """Automated experiment runner with checkpointing."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.status = ExperimentStatus(run_id=config.run_id, state="pending")
        self._stop_requested = False

    def _prepare_skill_store(self) -> None:
        """Prepare clean skill store if needed."""
        if not self.config.clean_skill_store:
            return
        if self.config.skill_store_dir.exists():
            shutil.rmtree(self.config.skill_store_dir)
        self.config.skill_store_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Cleaned skill store: {self.config.skill_store_dir}")

    def _get_artifact_path(self) -> Path:
        """Get path to experiment artifacts."""
        return self.config.artifacts_dir / self.config.run_id

    def _get_status_path(self) -> Path:
        """Get path to status file."""
        return self._get_artifact_path() / "experiment_status.json"

    def _save_status(self) -> None:
        """Save current status to file."""
        status_path = self._get_status_path()
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(self.status.to_dict(), f, indent=2)

    def _load_existing_status(self) -> bool:
        """Load existing status if resuming."""
        status_path = self._get_status_path()
        if status_path.exists():
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.status.state = data.get("state", "pending")
            self.status.completed_cases = data.get("completed_cases", 0)
            self.status.failed_cases = data.get("failed_cases", 0)
            self.status.current_offset = data.get("current_offset", 0)
            print(f"[INFO] Resuming from offset {self.status.current_offset}")
            return True
        return False

    def _estimate_total_cases(self) -> int:
        """Estimate total number of cases based on dataset."""
        if self.config.dataset.limit:
            return self.config.dataset.limit

        # Load dataset to count
        from ad_cornercase.datasets.dtpqa import DTPQADatasetLoader

        try:
            loader = DTPQADatasetLoader(self.config.dtpqa_root)
            cases = loader.load(
                subset=self.config.dataset.subset,
                question_type=self.config.dataset.question_type,
                limit=None,
                offset=0,
            )
            return len(cases)
        except Exception as e:
            print(f"[WARN] Could not estimate total cases: {e}")
            return 0

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for subprocess."""
        env = os.environ.copy()

        # Clear proxy settings
        for key in ['ALL_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            env.pop(key, None)

        # Model settings
        env["EDGE_MODEL"] = self.config.models.edge_model
        env["EDGE_MAX_COMPLETION_TOKENS"] = str(self.config.models.edge_max_completion_tokens)
        env["CLOUD_MODEL"] = self.config.models.cloud_model
        env["JUDGE_MODEL"] = self.config.models.judge_model
        env["EMBEDDING_MODEL"] = self.config.models.embedding_model

        # Paths
        env["DTPQA_ROOT"] = str(self.config.dtpqa_root)
        env["SKILL_STORE_DIR"] = str(self.config.skill_store_dir)
        env["ARTIFACTS_DIR"] = str(self.config.artifacts_dir)

        # Timeouts and retries
        env["REQUEST_TIMEOUT_SECONDS"] = str(self.config.request_timeout_seconds)
        env["MAX_RETRIES"] = str(self.config.max_retries)

        # MCP server
        env["MCP_SERVER_HOST"] = self.config.mcp_server_host
        env["MCP_SERVER_PORT"] = str(self.config.mcp_server_port)

        # Reflection settings
        env["UNCERTAINTY_ENTROPY_THRESHOLD"] = str(self.config.entropy_threshold)
        if self.config.execution_mode == "hybrid" and self.config.enable_reflection:
            env["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] = (
                "1" if self.config.enable_dtpqa_people_reflection else "0"
            )
        else:
            env["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] = "0"

        return env

    def _run_single_batch(
        self,
        offset: int,
        limit: int,
        append: bool = True
    ) -> tuple[bool, dict[str, Any]]:
        """Run a single batch of experiments."""
        env = self._build_env()

        cmd = [
            "uv", "run", "ad-replay-dtpqa",
            "--subset", self.config.dataset.subset,
            "--question-type", self.config.dataset.question_type,
            "--offset", str(offset),
            "--limit", str(limit),
            "--run-id", self.config.run_id,
            "--execution-mode", self.config.execution_mode,
        ]

        if append:
            cmd.append("--append")

        if self.config.dataset.annotation_glob:
            cmd.extend(["--annotation-glob", self.config.dataset.annotation_glob])

        print(f"[INFO] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=int(self.config.request_timeout_seconds * limit + 60),
            )

            success = result.returncode == 0

            # Parse output for metrics
            metrics = {
                "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
                "returncode": result.returncode,
            }

            if not success:
                print(f"[ERROR] Batch failed: {result.stderr[:500]}")

            return success, metrics

        except subprocess.TimeoutExpired:
            print(f"[ERROR] Batch timed out at offset {offset}")
            return False, {"error": "timeout"}
        except Exception as e:
            print(f"[ERROR] Batch exception: {e}")
            return False, {"error": str(e)}

    def _run_judge_evaluation(self) -> bool:
        """Run judge evaluation on completed replay."""
        if not self.config.enable_judge:
            return True

        env = self._build_env()

        cmd = [
            "uv", "run", "ad-eval-dtpqa",
            "--run-id", self.config.run_id,
        ]

        print(f"[INFO] Running judge evaluation: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for judge
            )

            success = result.returncode == 0
            if not success:
                print(f"[WARN] Judge evaluation failed: {result.stderr[:500]}")

            return success

        except Exception as e:
            print(f"[WARN] Judge evaluation exception: {e}")
            return False

    def run(
        self,
        resume: bool = True,
        batch_size: int | None = None,
    ) -> ExperimentStatus:
        """
        Run the complete experiment.

        Args:
            resume: Whether to resume from previous checkpoint
            batch_size: Override batch size from config
        """
        batch_size = batch_size or self.config.batch_size

        resumed = False
        if resume:
            resumed = self._load_existing_status()

        if not resumed:
            self._prepare_skill_store()

        # Estimate total
        if self.status.total_cases == 0:
            self.status.total_cases = self._estimate_total_cases()
            if self.config.dataset.limit:
                self.status.total_cases = min(self.status.total_cases, self.config.dataset.limit)

        # Start
        self.status.state = "running"
        self.status.start_time = datetime.now().isoformat()
        self._save_status()

        print(f"[INFO] Starting experiment: {self.config.run_id}")
        print(f"[INFO] Total cases: {self.status.total_cases}")
        print(f"[INFO] Starting from offset: {self.status.current_offset}")

        try:
            # Run batches
            while self.status.current_offset < self.status.total_cases:
                if self._stop_requested:
                    print("[INFO] Stop requested, pausing experiment")
                    self.status.state = "paused"
                    self._save_status()
                    return self.status

                remaining = self.status.total_cases - self.status.current_offset
                current_batch_size = min(batch_size, remaining)

                print(f"\n[{'='*60}]")
                print(f"[INFO] Batch: offset={self.status.current_offset}, limit={current_batch_size}")
                print(f"[INFO] Progress: {self.status.progress_percentage:.1f}%")
                print(f"[{'='*60}]")

                success, metrics = self._run_single_batch(
                    offset=self.status.current_offset,
                    limit=current_batch_size,
                    append=self.status.current_offset > 0,
                )

                if success:
                    self.status.completed_cases += current_batch_size
                else:
                    self.status.failed_cases += current_batch_size
                    # Continue on failure - don't stop the whole experiment

                self.status.current_offset += current_batch_size
                self._save_status()

                # Sleep between batches if configured
                if self.config.batch_sleep_seconds > 0:
                    time.sleep(self.config.batch_sleep_seconds)

            # Run judge evaluation
            print("\n[INFO] Running judge evaluation...")
            self._run_judge_evaluation()

            # Complete
            self.status.state = "completed"
            self.status.end_time = datetime.now().isoformat()
            self._save_status()

            print(f"\n[INFO] Experiment completed: {self.config.run_id}")
            print(f"[INFO] Completed: {self.status.completed_cases}")
            print(f"[INFO] Failed: {self.status.failed_cases}")

        except Exception as e:
            self.status.state = "failed"
            self.status.end_time = datetime.now().isoformat()
            self._save_status()
            print(f"[ERROR] Experiment failed: {e}")

        return self.status

    def stop(self) -> None:
        """Request graceful stop."""
        self._stop_requested = True

    def get_results(self) -> dict[str, Any] | None:
        """Get experiment results if available."""
        results_path = self._get_artifact_path() / "metrics.json"
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
