"""Universal edge-cloud orchestrator with configurable reflection policies.

This module provides a flexible, generalizable approach to edge-cloud
collaboration that works across different perception tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable
import asyncio

from .perception_task import PerceptionTask, TaskType
from .skill import SkillLibrary, SkillMatcher, Skill, SkillPattern, SkillAction


class ReflectionTrigger(Enum):
    """Different triggers that can activate cloud reflection."""
    ENTROPY_THRESHOLD = auto()
    LOW_CONFIDENCE = auto()
    SKILL_MISMATCH = auto()
    DISTANCE_BASED = auto()
    DIFFICULTY_BASED = auto()
    ALWAYS = auto()
    NEVER = auto()


@dataclass
class ReflectionPolicy:
    """Policy for when to trigger cloud reflection.

    This is fully configurable and extensible for different scenarios.
    """
    name: str = "default"

    # Primary trigger
    trigger: ReflectionTrigger = ReflectionTrigger.ENTROPY_THRESHOLD

    # Threshold values
    entropy_threshold: float = 1.0
    confidence_threshold: float = 0.7
    difficulty_threshold: float = 0.7

    # Distance-based settings
    distance_ranges: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Task-specific overrides
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def should_reflect(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        matched_skills: list[tuple[Skill, float]],
    ) -> tuple[bool, str]:
        """Determine if cloud reflection should be triggered.

        Returns (should_reflect, reason)
        """
        # Check task-specific overrides
        task_type = task.schema.task_type.name if task.schema else None
        if task_type and task_type in self.task_overrides:
            override = self.task_overrides[task_type]
            trigger = ReflectionTrigger[override.get("trigger", self.trigger.name)]
        else:
            trigger = self.trigger

        # Evaluate based on trigger type
        if trigger == ReflectionTrigger.NEVER:
            return False, "reflection_disabled"

        if trigger == ReflectionTrigger.ALWAYS:
            return True, "always_reflect"

        if trigger == ReflectionTrigger.ENTROPY_THRESHOLD:
            entropy = edge_output.get("entropy", 0.0)
            if entropy > self.entropy_threshold:
                return True, f"entropy_{entropy:.2f}>threshold"

        if trigger == ReflectionTrigger.LOW_CONFIDENCE:
            confidence = edge_output.get("confidence", 1.0)
            if confidence < self.confidence_threshold:
                return True, f"low_confidence_{confidence:.2f}"

        if trigger == ReflectionTrigger.DIFFICULTY_BASED:
            difficulty = task.difficulty_score
            if difficulty > self.difficulty_threshold:
                return True, f"high_difficulty_{difficulty:.2f}"

        if trigger == ReflectionTrigger.DISTANCE_BASED:
            distance = task.metadata.get("distance_meters")
            if distance is not None:
                for range_name, range_config in self.distance_ranges.items():
                    min_dist = range_config.get("min", 0)
                    max_dist = range_config.get("max", float("inf"))
                    if min_dist <= distance < max_dist:
                        # Check if we should reflect for this range
                        if range_config.get("force_reflect", False):
                            return True, f"distance_range_{range_name}"
                        # Or check if confidence is below range-specific threshold
                        confidence = edge_output.get("confidence", 1.0)
                        range_threshold = range_config.get("confidence_threshold", self.confidence_threshold)
                        if confidence < range_threshold:
                            return True, f"distance_range_{range_name}_low_confidence"

        if trigger == ReflectionTrigger.SKILL_MISMATCH:
            # Trigger if no good skill match
            if not matched_skills:
                return True, "no_skill_match"
            best_score = matched_skills[0][1] if matched_skills else 0
            if best_score < 0.5:
                return True, f"poor_skill_match_{best_score:.2f}"

        return False, "no_trigger_met"


@dataclass
class OrchestratorConfig:
    """Configuration for the edge-cloud orchestrator."""
    # Edge model settings
    edge_model: str = "Qwen/Qwen3.5-9B"
    edge_max_tokens: int = 512

    # Cloud model settings
    cloud_model: str = "Pro/moonshotai/Kimi-K2.5"
    cloud_max_tokens: int = 1024

    # Reflection policy
    reflection_policy: ReflectionPolicy = field(default_factory=ReflectionPolicy)

    # Skill library
    skill_library: SkillLibrary | None = None

    # Execution settings
    enable_skill_learning: bool = True
    enable_skill_application: bool = True
    persist_new_skills: bool = True


class EdgeCloudOrchestrator:
    """Universal orchestrator for edge-cloud perception.

    This is the main entry point that handles:
    1. Edge inference
    2. Skill matching and application
    3. Reflection triggering decision
    4. Cloud inference (when needed)
    5. Skill learning from results
    """

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.skill_matcher = SkillMatcher(config.skill_library) if config.skill_library else None

        # Track statistics
        self.stats = {
            "total_tasks": 0,
            "edge_only": 0,
            "with_reflection": 0,
            "skills_applied": 0,
            "new_skills_learned": 0,
        }

    async def process(
        self,
        task: PerceptionTask,
        edge_inference_fn: Callable[[PerceptionTask], dict[str, Any]],
        cloud_inference_fn: Callable[[PerceptionTask, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Process a perception task through the edge-cloud pipeline.

        Args:
            task: The perception task to process
            edge_inference_fn: Function to run edge inference
            cloud_inference_fn: Optional function to run cloud inference

        Returns:
            Result dictionary with outputs and metadata
        """
        self.stats["total_tasks"] += 1
        start_time = datetime.now()

        # Step 1: Edge inference
        edge_output = await self._run_edge_inference(task, edge_inference_fn)
        task.edge_output = edge_output

        # Step 2: Skill matching and application
        matched_skills = []
        if self.config.enable_skill_application and self.skill_matcher:
            edge_output, applied_skills = self.skill_matcher.find_and_apply(task, edge_output)
            matched_skills = [(self.config.skill_library.get(sid), 1.0) for sid in applied_skills]
            if applied_skills:
                self.stats["skills_applied"] += 1

        # Step 3: Decide if reflection is needed
        should_reflect, reason = self.config.reflection_policy.should_reflect(
            task, edge_output, matched_skills
        )

        result = {
            "task_id": task.task_id,
            "edge_output": edge_output,
            "skills_applied": [s[0].skill_id for s in matched_skills] if matched_skills else [],
            "reflection_triggered": should_reflect,
            "reflection_reason": reason,
        }

        if should_reflect and cloud_inference_fn:
            # Step 4: Cloud reflection
            self.stats["with_reflection"] += 1
            cloud_output = await self._run_cloud_reflection(task, edge_output, cloud_inference_fn)
            task.cloud_output = cloud_output
            result["cloud_output"] = cloud_output
            result["final_output"] = cloud_output

            # Step 5: Learn from this experience
            if self.config.enable_skill_learning:
                new_skill = self._learn_skill(task, edge_output, cloud_output)
                if new_skill and self.config.persist_new_skills:
                    self.config.skill_library.add(new_skill)
                    self.stats["new_skills_learned"] += 1
                    result["new_skill_id"] = new_skill.skill_id
        else:
            self.stats["edge_only"] += 1
            result["final_output"] = edge_output

        # Calculate latency
        end_time = datetime.now()
        result["latency_ms"] = (end_time - start_time).total_seconds() * 1000
        task.completed_at = end_time.isoformat()

        return result

    async def _run_edge_inference(
        self,
        task: PerceptionTask,
        inference_fn: Callable[[PerceptionTask], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run edge inference."""
        # Run in thread pool if function is synchronous
        if asyncio.iscoroutinefunction(inference_fn):
            return await inference_fn(task)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, inference_fn, task)

    async def _run_cloud_reflection(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        inference_fn: Callable[[PerceptionTask, dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Run cloud reflection."""
        if asyncio.iscoroutinefunction(inference_fn):
            return await inference_fn(task, edge_output)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, inference_fn, task, edge_output)

    def _learn_skill(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        cloud_output: dict[str, Any],
    ) -> Skill | None:
        """Learn a new skill from the edge-cloud difference."""
        # Extract what was learned
        edge_answer = edge_output.get("answer", "")
        cloud_answer = cloud_output.get("answer", "")

        if edge_answer == cloud_answer:
            # No difference to learn from
            return None

        # Create a skill that captures this improvement
        skill_id = f"skill_{task.task_id[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Build pattern from task metadata
        pattern = SkillPattern(
            task_type=task.schema.task_type.name if task.schema else None,
            scene_type=task.metadata.get("scene_type"),
            lighting_condition=task.metadata.get("lighting_condition"),
            weather_condition=task.metadata.get("weather_condition"),
            object_categories=task.metadata.get("object_categories", []),
            distance_range=(
                task.metadata.get("distance_meters", 0) * 0.8,
                task.metadata.get("distance_meters", 0) * 1.2,
            ) if task.metadata.get("distance_meters") else (None, None),
            difficulty_range=(task.difficulty_score * 0.8, min(task.difficulty_score * 1.2, 1.0)),
        )

        # Build action based on the correction
        action = SkillAction(
            action_type="apply_label",
            parameters={
                "correction": {
                    "from": edge_answer,
                    "to": cloud_answer,
                },
                "confidence_boost": 0.2,
            },
            reasoning=f"Learned from reflection: {cloud_output.get('reasoning', 'Cloud correction')}",
        )

        skill = Skill(
            skill_id=skill_id,
            name=f"Auto-skill for {task.task_category}",
            pattern=pattern,
            action=action,
            source_task_id=task.task_id,
        )

        return skill

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        stats = self.stats.copy()
        if stats["total_tasks"] > 0:
            stats["edge_only_ratio"] = stats["edge_only"] / stats["total_tasks"]
            stats["reflection_ratio"] = stats["with_reflection"] / stats["total_tasks"]
        return stats


# Predefined policies for common scenarios

POLICY_CONSERVATIVE = ReflectionPolicy(
    name="conservative",
    trigger=ReflectionTrigger.ENTROPY_THRESHOLD,
    entropy_threshold=1.2,  # Only reflect when very uncertain
)

POLICY_AGGRESSIVE = ReflectionPolicy(
    name="aggressive",
    trigger=ReflectionTrigger.LOW_CONFIDENCE,
    confidence_threshold=0.8,  # Reflect on anything less than 80% confident
)

POLICY_DISTANCE_AWARE = ReflectionPolicy(
    name="distance_aware",
    trigger=ReflectionTrigger.DISTANCE_BASED,
    distance_ranges={
        "far": {
            "min": 30,
            "max": float("inf"),
            "force_reflect": False,
            "confidence_threshold": 0.6,  # Lower threshold for far objects
        },
        "mid": {
            "min": 20,
            "max": 30,
            "force_reflect": False,
            "confidence_threshold": 0.7,
        },
        "near": {
            "min": 0,
            "max": 20,
            "force_reflect": False,
            "confidence_threshold": 0.8,
        },
    },
)

POLICY_SKILL_DRIVEN = ReflectionPolicy(
    name="skill_driven",
    trigger=ReflectionTrigger.SKILL_MISMATCH,
)
