"""Universal perception task definition.

This module provides a generic, schema-based approach to defining perception tasks
that works across different datasets (DTPQA, DriveLM, CODA-LM, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any
import uuid
import json


class TaskType(Enum):
    """Types of perception tasks - extensible for new benchmarks."""
    OBJECT_DETECTION = auto()
    SCENE_UNDERSTANDING = auto()
    REGIONAL_GROUNDING = auto()
    TEMPORAL_REASONING = auto()
    CAUSAL_REASONING = auto()
    PLANNING = auto()
    OPEN_VOCABULARY_QA = auto()


class InputModality(Enum):
    """Input modalities supported."""
    IMAGE = auto()
    VIDEO = auto()
    MULTI_IMAGE = auto()
    TEXT_ONLY = auto()


@dataclass
class TaskSchema:
    """Schema defining the structure of a perception task.

    This allows the system to understand what inputs are expected
    and what outputs should be produced, without hardcoding task-specific logic.
    """
    name: str
    description: str
    task_type: TaskType
    input_modality: InputModality

    # Input specification
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)

    # Output specification
    output_schema: dict[str, Any] = field(default_factory=dict)

    # Metadata
    benchmark_source: str | None = None
    version: str = "1.0"

    def validate_input(self, inputs: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate that inputs match the schema."""
        missing = [key for key in self.required_inputs if key not in inputs]
        return len(missing) == 0, missing


@dataclass
class PerceptionTask:
    """A single perception task instance.

    This is a universal representation that works across all benchmarks.
    """
    # Identity
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema: TaskSchema | None = None

    # Inputs
    inputs: dict[str, Any] = field(default_factory=dict)

    # Expected output (ground truth for training/eval)
    ground_truth: dict[str, Any] | None = None

    # Metadata for analysis
    metadata: dict[str, Any] = field(default_factory=dict)

    # System outputs (filled during execution)
    edge_output: dict[str, Any] | None = None
    cloud_output: dict[str, Any] | None = None
    final_output: dict[str, Any] | None = None

    # Execution tracking
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    @property
    def task_category(self) -> str:
        """Get task category based on schema and metadata."""
        if self.schema:
            return f"{self.schema.benchmark_source}:{self.schema.task_type.name}"
        return "unknown"

    @property
    def difficulty_score(self) -> float:
        """Calculate estimated difficulty (0-1) based on metadata."""
        score = 0.5  # Default medium difficulty

        # Adjust based on available metadata
        if "distance_meters" in self.metadata:
            dist = self.metadata["distance_meters"]
            if dist is not None:
                # Farther distances are harder
                score += min(dist / 100.0, 0.3)

        if "occlusion_level" in self.metadata:
            score += self.metadata["occlusion_level"] * 0.2

        if "lighting_condition" in self.metadata:
            lighting = self.metadata["lighting_condition"]
            if lighting in ["night", "low_light", "backlit"]:
                score += 0.2

        return min(score, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "schema": {
                "name": self.schema.name if self.schema else None,
                "task_type": self.schema.task_type.name if self.schema else None,
                "benchmark_source": self.schema.benchmark_source if self.schema else None,
            },
            "inputs": {k: str(v)[:100] for k, v in self.inputs.items()},
            "metadata": self.metadata,
            "difficulty_score": self.difficulty_score,
            "task_category": self.task_category,
        }


class TaskRegistry:
    """Registry for task schemas - enables adding new benchmarks without code changes."""

    def __init__(self):
        self._schemas: dict[str, TaskSchema] = {}

    def register(self, schema: TaskSchema) -> None:
        """Register a new task schema."""
        self._schemas[schema.name] = schema
        print(f"[TaskRegistry] Registered schema: {schema.name}")

    def get(self, name: str) -> TaskSchema | None:
        """Get a schema by name."""
        return self._schemas.get(name)

    def list_schemas(self) -> list[str]:
        """List all registered schema names."""
        return list(self._schemas.keys())

    def load_from_file(self, path: Path) -> None:
        """Load task schemas from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        for schema_data in data.get("schemas", []):
            schema = TaskSchema(
                name=schema_data["name"],
                description=schema_data["description"],
                task_type=TaskType[schema_data["task_type"]],
                input_modality=InputModality[schema_data["input_modality"]],
                required_inputs=schema_data.get("required_inputs", []),
                optional_inputs=schema_data.get("optional_inputs", []),
                output_schema=schema_data.get("output_schema", {}),
                benchmark_source=schema_data.get("benchmark_source"),
                version=schema_data.get("version", "1.0"),
            )
            self.register(schema)


# Global registry instance
TASK_REGISTRY = TaskRegistry()


def register_builtin_schemas():
    """Register built-in schemas for common benchmarks."""

    # DTPQA Category 1: Pedestrian Detection
    TASK_REGISTRY.register(TaskSchema(
        name="dtpqa_pedestrian_detection",
        description="Detect if pedestrians are crossing the road",
        task_type=TaskType.OBJECT_DETECTION,
        input_modality=InputModality.IMAGE,
        required_inputs=["image_path", "question"],
        optional_inputs=["distance_hint", "weather_condition"],
        output_schema={
            "answer": "str: Yes/No with explanation",
            "confidence": "float: 0-1",
        },
        benchmark_source="dtpqa",
    ))

    # DriveLM: Multi-round QA
    TASK_REGISTRY.register(TaskSchema(
        name="drivelm_qa",
        description="Answer driving-related questions with temporal context",
        task_type=TaskType.TEMPORAL_REASONING,
        input_modality=InputModality.MULTI_IMAGE,
        required_inputs=["image_paths", "question", " QA history"],
        optional_inputs=["ego_vehicle_state"],
        output_schema={
            "answer": "str: Natural language answer",
            "grounded_objects": "list: Referenced objects",
        },
        benchmark_source="drivelm",
    ))

    # CODA-LM: Corner case detection
    TASK_REGISTRY.register(TaskSchema(
        name="coda_lm_anomaly_detection",
        description="Detect and explain unusual driving scenarios",
        task_type=TaskType.SCENE_UNDERSTANDING,
        input_modality=InputModality.IMAGE,
        required_inputs=["image_path"],
        optional_inputs=["scene_context", "location"],
        output_schema={
            "anomaly_detected": "bool",
            "anomaly_type": "str: Category of anomaly",
            "explanation": "str: Why this is unusual",
        },
        benchmark_source="coda_lm",
    ))

    print(f"[TaskRegistry] Registered {len(TASK_REGISTRY.list_schemas())} built-in schemas")


# Register on import
register_builtin_schemas()
