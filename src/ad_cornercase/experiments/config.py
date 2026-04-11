"""Experiment configuration management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for VLM models."""
    edge_model: str = "Qwen/Qwen3.5-9B"
    edge_max_completion_tokens: int = 512
    cloud_model: str = "Pro/moonshotai/Kimi-K2.5"
    judge_model: str = "Pro/moonshotai/Kimi-K2.5"
    embedding_model: str = "BAAI/bge-m3"


@dataclass
class DatasetConfig:
    """Configuration for dataset selection."""
    benchmark: str = "dtpqa"  # dtpqa, coda_lm, drivelm
    subset: str = "real"  # real, synth, all
    question_type: str = "category_1"
    limit: int | None = None
    offset: int = 0
    annotation_glob: str | None = None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""

    # Model settings
    models: ModelConfig = field(default_factory=ModelConfig)

    # Dataset settings
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Experiment settings
    batch_size: int = 1
    max_retries: int = 3
    request_timeout_seconds: float = 300.0
    batch_sleep_seconds: float = 0

    # Skill store
    skill_store_dir: Path = field(default_factory=lambda: Path("/tmp/dtpqa_skills_empty"))
    clean_skill_store: bool = True

    # MCP server
    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8003

    # Evaluation
    enable_judge: bool = True
    judge_only_missing: bool = True

    # Reflection settings
    enable_reflection: bool = True
    entropy_threshold: float = 1.0
    enable_dtpqa_people_reflection: bool = True
    execution_mode: str = "hybrid"  # edge_only, cloud_only, hybrid

    # Paths
    dtpqa_root: Path = field(default_factory=lambda: Path("./data/dtpqa"))
    artifacts_dir: Path = field(default_factory=lambda: Path("./data/artifacts"))

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str | None = None

    def __post_init__(self):
        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.name.replace(" ", "_").replace("-", "_").lower()
            self.run_id = f"{safe_name}_{timestamp}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Convert Path objects to strings
        for key in ["skill_store_dir", "dtpqa_root", "artifacts_dir"]:
            if key in d and isinstance(d[key], Path):
                d[key] = str(d[key])
        return d

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert strings back to Path objects
        for key in ["skill_store_dir", "dtpqa_root", "artifacts_dir"]:
            if key in data and isinstance(data[key], str):
                data[key] = Path(data[key])

        # Reconstruct nested configs
        if "models" in data:
            data["models"] = ModelConfig(**data["models"])
        if "dataset" in data:
            data["dataset"] = DatasetConfig(**data["dataset"])

        return cls(**data)


# Predefined experiment configurations
EXPERIMENT_PRESETS = {
    "dtpqa_real_baseline": ExperimentConfig(
        name="dtpqa-real-baseline",
        description="Baseline experiment on DTPQA real dataset with Qwen edge model",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",
            question_type="category_1",
        ),
        execution_mode="edge_only",
        enable_reflection=False,
    ),

    "dtpqa_real_cloud_only": ExperimentConfig(
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
        ),
        execution_mode="cloud_only",
        enable_reflection=False,
        enable_dtpqa_people_reflection=False,
    ),

    "dtpqa_real_with_reflection": ExperimentConfig(
        name="dtpqa-real-reflection",
        description="DTPQA real dataset with cloud reflection enabled",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="real",
            question_type="category_1",
        ),
        execution_mode="hybrid",
        enable_reflection=True,
        enable_dtpqa_people_reflection=True,
    ),

    "dtpqa_synth_baseline": ExperimentConfig(
        name="dtpqa-synth-baseline",
        description="Baseline experiment on DTPQA synthetic dataset",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="synth",
            question_type="category_1",
        ),
        execution_mode="edge_only",
        enable_reflection=False,
    ),

    "dtpqa_qwen_baseline": ExperimentConfig(
        name="dtpqa-qwen-baseline",
        description="Baseline with Qwen3.5-9B edge model",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="synth",  # Using synth dataset
            question_type="category_1",
            limit=50,  # Small batch for testing
        ),
        enable_reflection=False,
    ),

    "dtpqa_full_scale": ExperimentConfig(
        name="dtpqa-full-scale",
        description="Full-scale experiment on all DTPQA data",
        models=ModelConfig(
            edge_model="Qwen/Qwen3.5-9B",
            edge_max_completion_tokens=512,
        ),
        dataset=DatasetConfig(
            benchmark="dtpqa",
            subset="all",
            question_type="category_1",
        ),
        execution_mode="hybrid",
        enable_reflection=True,
        enable_judge=True,
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """Get a preset configuration by name."""
    if name not in EXPERIMENT_PRESETS:
        available = ", ".join(EXPERIMENT_PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    # Return a copy to avoid modifying the original
    import copy
    return copy.deepcopy(EXPERIMENT_PRESETS[name])
