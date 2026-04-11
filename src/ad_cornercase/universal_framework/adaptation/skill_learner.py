"""Skill learning and adaptation strategies.

This module provides different strategies for learning skills from
edge-cloud interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ..core.skill import Skill, SkillPattern, SkillAction
from ..core.perception_task import PerceptionTask


class AdaptationStrategy(Enum):
    """Strategies for adapting to new scenarios."""
    CORRECTION_BASED = auto()  # Learn from edge-cloud differences
    CONFIDENCE_BASED = auto()  # Learn from confidence patterns
    ERROR_DRIVEN = auto()  # Learn from mistakes
    HYBRID = auto()  # Combine multiple strategies


@dataclass
class SkillLearner:
    """Learns skills from task execution results.

    This is a generic skill learner that works across benchmarks.
    """

    strategy: AdaptationStrategy = AdaptationStrategy.HYBRID
    min_samples: int = 3  # Minimum samples before creating a skill
    confidence_threshold: float = 0.8

    def learn_from_result(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        cloud_output: dict[str, Any] | None,
        was_correct: bool,
    ) -> Skill | None:
        """Learn a skill from a single task result.

        Args:
            task: The perception task
            edge_output: Output from edge model
            cloud_output: Output from cloud model (if reflection occurred)
            was_correct: Whether the final result was correct

        Returns:
            New skill if learning occurred, None otherwise
        """
        if self.strategy == AdaptationStrategy.CORRECTION_BASED:
            return self._learn_from_correction(task, edge_output, cloud_output)

        elif self.strategy == AdaptationStrategy.CONFIDENCE_BASED:
            return self._learn_from_confidence(task, edge_output, was_correct)

        elif self.strategy == AdaptationStrategy.ERROR_DRIVEN:
            if not was_correct:
                return self._learn_from_error(task, edge_output, cloud_output)
            return None

        elif self.strategy == AdaptationStrategy.HYBRID:
            return self._learn_hybrid(task, edge_output, cloud_output, was_correct)

        return None

    def _learn_from_correction(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        cloud_output: dict[str, Any] | None,
    ) -> Skill | None:
        """Learn from edge-cloud correction."""
        if not cloud_output:
            return None

        edge_answer = edge_output.get("answer", "")
        cloud_answer = cloud_output.get("answer", "")

        if edge_answer == cloud_answer:
            return None  # No correction to learn from

        # Create skill capturing this correction pattern
        return self._create_correction_skill(task, edge_output, cloud_output)

    def _learn_from_confidence(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        was_correct: bool,
    ) -> Skill | None:
        """Learn from confidence patterns."""
        confidence = edge_output.get("confidence", 0.5)

        # If confidence was low but answer was correct,
        # learn to boost confidence in similar cases
        if confidence < self.confidence_threshold and was_correct:
            return self._create_confidence_skill(task, edge_output, boost=True)

        # If confidence was high but answer was wrong,
        # learn to be more cautious
        if confidence > self.confidence_threshold and not was_correct:
            return self._create_confidence_skill(task, edge_output, boost=False)

        return None

    def _learn_from_error(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        cloud_output: dict[str, Any] | None,
    ) -> Skill | None:
        """Learn from errors."""
        # Similar to correction-based but specifically for errors
        if cloud_output:
            return self._create_correction_skill(task, edge_output, cloud_output)
        return None

    def _learn_hybrid(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        cloud_output: dict[str, Any] | None,
        was_correct: bool,
    ) -> Skill | None:
        """Combine multiple learning strategies."""
        # Priority: corrections > confidence patterns
        if cloud_output:
            skill = self._learn_from_correction(task, edge_output, cloud_output)
            if skill:
                return skill

        return self._learn_from_confidence(task, edge_output, was_correct)

    def _create_correction_skill(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        cloud_output: dict[str, Any],
    ) -> Skill:
        """Create a skill from edge-cloud correction."""
        import uuid
        from datetime import datetime

        skill_id = f"correction_{uuid.uuid4().hex[:8]}"

        # Build pattern from task features
        pattern = SkillPattern(
            task_type=task.schema.task_type.name if task.schema else None,
            scene_type=task.metadata.get("scene_type"),
            lighting_condition=task.metadata.get("lighting_condition"),
            weather_condition=task.metadata.get("weather_condition"),
            object_categories=task.metadata.get("object_categories", []),
            distance_range=(
                task.metadata.get("distance_meters", 0) * 0.9,
                task.metadata.get("distance_meters", 0) * 1.1,
            ) if task.metadata.get("distance_meters") else (None, None),
            difficulty_range=(task.difficulty_score * 0.9, min(task.difficulty_score * 1.1, 1.0)),
        )

        action = SkillAction(
            action_type="correct_answer",
            parameters={
                "from_answer": edge_output.get("answer"),
                "to_answer": cloud_output.get("answer"),
                "confidence_adjustment": 0.1,
            },
            reasoning=f"Learned correction: {edge_output.get('answer')} → {cloud_output.get('answer')}",
        )

        return Skill(
            skill_id=skill_id,
            name=f"Auto-correction for {task.task_category}",
            pattern=pattern,
            action=action,
            source_task_id=task.task_id,
        )

    def _create_confidence_skill(
        self,
        task: PerceptionTask,
        edge_output: dict[str, Any],
        boost: bool,
    ) -> Skill:
        """Create a skill for confidence adjustment."""
        import uuid
        from datetime import datetime

        skill_id = f"confidence_{uuid.uuid4().hex[:8]}"

        pattern = SkillPattern(
            task_type=task.schema.task_type.name if task.schema else None,
            scene_type=task.metadata.get("scene_type"),
            difficulty_range=(task.difficulty_score * 0.9, min(task.difficulty_score * 1.1, 1.0)),
        )

        action = SkillAction(
            action_type="adjust_confidence",
            parameters={
                "multiplier": 1.2 if boost else 0.8,
                "reason": "boost_low_confidence" if boost else "reduce_overconfidence",
            },
            reasoning="Adjust confidence based on observed patterns",
        )

        return Skill(
            skill_id=skill_id,
            name=f"Confidence adjustment for {task.task_category}",
            pattern=pattern,
            action=action,
            source_task_id=task.task_id,
        )
