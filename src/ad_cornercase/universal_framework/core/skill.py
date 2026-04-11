"""Universal skill representation and matching.

Skills are structured knowledge that can be learned and applied across
different perception tasks and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import numpy as np


@dataclass
class SkillPattern:
    """A pattern that represents when a skill applies.

    This uses structured features rather than text embeddings,
    making it more interpretable and generalizable.
    """
    # Visual features
    visual_keywords: list[str] = field(default_factory=list)
    scene_type: str | None = None
    lighting_condition: str | None = None
    weather_condition: str | None = None

    # Task features
    task_type: str | None = None
    object_categories: list[str] = field(default_factory=list)
    difficulty_range: tuple[float, float] = (0.0, 1.0)

    # Spatial features
    distance_range: tuple[float | None, float | None] = (None, None)
    region_of_interest: str | None = None

    def matches(self, task_features: dict[str, Any]) -> float:
        """Calculate match score (0-1) against task features."""
        scores = []

        # Scene type match
        if self.scene_type and task_features.get("scene_type") == self.scene_type:
            scores.append(1.0)

        # Lighting match
        if self.lighting_condition and task_features.get("lighting") == self.lighting_condition:
            scores.append(1.0)

        # Weather match
        if self.weather_condition and task_features.get("weather") == self.weather_condition:
            scores.append(1.0)

        # Task type match
        if self.task_type and task_features.get("task_type") == self.task_type:
            scores.append(1.0)

        # Object category overlap
        if self.object_categories:
            task_objects = set(task_features.get("object_categories", []))
            skill_objects = set(self.object_categories)
            if task_objects and skill_objects:
                overlap = len(task_objects & skill_objects) / len(task_objects | skill_objects)
                scores.append(overlap)

        # Distance range match
        if self.distance_range[0] is not None and self.distance_range[1] is not None:
            task_distance = task_features.get("distance_meters")
            if task_distance is not None:
                if self.distance_range[0] <= task_distance <= self.distance_range[1]:
                    scores.append(1.0)
                else:
                    # Calculate distance to nearest bound
                    if task_distance < self.distance_range[0]:
                        ratio = task_distance / self.distance_range[0] if self.distance_range[0] > 0 else 0
                    else:
                        ratio = self.distance_range[1] / task_distance if task_distance > 0 else 0
                    scores.append(max(0, ratio))

        # Difficulty range match
        task_difficulty = task_features.get("difficulty", 0.5)
        if self.difficulty_range[0] <= task_difficulty <= self.difficulty_range[1]:
            scores.append(1.0)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)


@dataclass
class SkillAction:
    """The action/recommendation part of a skill.

    This is what the skill suggests doing when applied.
    """
    action_type: str  # e.g., "apply_label", "adjust_confidence", "add_context"
    parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class Skill:
    """A universal skill that can be applied to perception tasks.

    Skills are structured knowledge learned from experience that
    helps improve perception accuracy without model fine-tuning.
    """
    # Identity
    skill_id: str
    name: str
    version: str = "1.0"

    # Matching pattern
    pattern: SkillPattern = field(default_factory=SkillPattern)

    # Action to take when applied
    action: SkillAction = field(default_factory=lambda: SkillAction(action_type="unknown"))

    # Metadata
    source_task_id: str | None = None
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_applied: str | None = None

    # Performance tracking
    application_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate of this skill."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown
        return self.success_count / total

    @property
    def confidence(self) -> float:
        """Calculate confidence score based on success rate and usage."""
        # Base confidence from success rate
        base = self.success_rate

        # Adjust based on number of applications (more data = more confident)
        total = self.success_count + self.failure_count
        reliability = min(total / 10.0, 1.0)  # Max out at 10 applications

        return base * 0.7 + reliability * 0.3

    def record_application(self, task_id: str, success: bool, details: dict[str, Any] | None = None):
        """Record an application of this skill."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.last_applied = datetime.now().isoformat()
        self.application_history.append({
            "task_id": task_id,
            "success": success,
            "timestamp": self.last_applied,
            "details": details or {},
        })

        # Keep history manageable
        if len(self.application_history) > 100:
            self.application_history = self.application_history[-50:]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "pattern": {
                "visual_keywords": self.pattern.visual_keywords,
                "scene_type": self.pattern.scene_type,
                "lighting_condition": self.pattern.lighting_condition,
                "weather_condition": self.pattern.weather_condition,
                "task_type": self.pattern.task_type,
                "object_categories": self.pattern.object_categories,
                "distance_range": self.pattern.distance_range,
                "difficulty_range": self.pattern.difficulty_range,
            },
            "action": {
                "action_type": self.action.action_type,
                "parameters": self.action.parameters,
                "reasoning": self.action.reasoning,
            },
            "performance": {
                "success_rate": self.success_rate,
                "confidence": self.confidence,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "total_applications": len(self.application_history),
            },
            "metadata": {
                "source_task_id": self.source_task_id,
                "created_at": self.created_at,
                "last_applied": self.last_applied,
            },
        }


class SkillLibrary:
    """A library of skills that can be queried and updated."""

    def __init__(self, storage_dir: Path | None = None):
        self.skills: dict[str, Skill] = {}
        self.storage_dir = storage_dir

        if storage_dir and storage_dir.exists():
            self._load_from_disk()

    def add(self, skill: Skill) -> None:
        """Add a skill to the library."""
        self.skills[skill.skill_id] = skill
        if self.storage_dir:
            self._save_skill(skill)

    def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        return self.skills.get(skill_id)

    def query(
        self,
        task_features: dict[str, Any],
        min_confidence: float = 0.5,
        top_k: int = 3,
    ) -> list[tuple[Skill, float]]:
        """Query for skills matching task features.

        Returns list of (skill, match_score) tuples sorted by score.
        """
        matches = []

        for skill in self.skills.values():
            if skill.confidence < min_confidence:
                continue

            match_score = skill.pattern.matches(task_features)
            if match_score > 0.3:  # Minimum threshold
                matches.append((skill, match_score * skill.confidence))

        # Sort by combined score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def _save_skill(self, skill: Skill) -> None:
        """Save skill to disk."""
        if not self.storage_dir:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        skill_path = self.storage_dir / f"{skill.skill_id}.json"

        with open(skill_path, "w") as f:
            json.dump(skill.to_dict(), f, indent=2)

    def _load_from_disk(self) -> None:
        """Load skills from disk."""
        if not self.storage_dir:
            return

        for skill_file in self.storage_dir.glob("*.json"):
            try:
                with open(skill_file, "r") as f:
                    data = json.load(f)

                skill = self._deserialize_skill(data)
                self.skills[skill.skill_id] = skill
            except Exception as e:
                print(f"[SkillLibrary] Failed to load {skill_file}: {e}")

        print(f"[SkillLibrary] Loaded {len(self.skills)} skills from {self.storage_dir}")

    def _deserialize_skill(self, data: dict[str, Any]) -> Skill:
        """Deserialize skill from dictionary."""
        pattern_data = data.get("pattern", {})
        pattern = SkillPattern(
            visual_keywords=pattern_data.get("visual_keywords", []),
            scene_type=pattern_data.get("scene_type"),
            lighting_condition=pattern_data.get("lighting_condition"),
            weather_condition=pattern_data.get("weather_condition"),
            task_type=pattern_data.get("task_type"),
            object_categories=pattern_data.get("object_categories", []),
            distance_range=tuple(pattern_data.get("distance_range", [None, None])),
            difficulty_range=tuple(pattern_data.get("difficulty_range", [0.0, 1.0])),
        )

        action_data = data.get("action", {})
        action = SkillAction(
            action_type=action_data.get("action_type", "unknown"),
            parameters=action_data.get("parameters", {}),
            reasoning=action_data.get("reasoning", ""),
        )

        return Skill(
            skill_id=data["skill_id"],
            name=data["name"],
            version=data.get("version", "1.0"),
            pattern=pattern,
            action=action,
            source_task_id=data.get("metadata", {}).get("source_task_id"),
        )


class SkillMatcher:
    """Matches skills to tasks and applies them."""

    def __init__(self, skill_library: SkillLibrary):
        self.library = skill_library

    def find_and_apply(
        self,
        task: Any,  # PerceptionTask
        edge_output: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Find matching skills and apply them to edge output.

        Returns:
            (modified_output, applied_skill_ids)
        """
        # Extract task features
        task_features = self._extract_features(task)

        # Query matching skills
        matches = self.library.query(task_features, min_confidence=0.5, top_k=3)

        if not matches:
            return edge_output, []

        # Apply skills
        modified_output = edge_output.copy()
        applied_skills = []

        for skill, score in matches:
            if score > 0.7:  # High confidence match
                modified_output = self._apply_skill(skill, modified_output)
                applied_skills.append(skill.skill_id)

        return modified_output, applied_skills

    def _extract_features(self, task: Any) -> dict[str, Any]:
        """Extract features from task for skill matching."""
        features = {
            "task_type": task.schema.task_type.name if task.schema else None,
            "difficulty": task.difficulty_score,
        }

        # Add metadata features
        if task.metadata:
            features.update({
                "distance_meters": task.metadata.get("distance_meters"),
                "lighting": task.metadata.get("lighting_condition"),
                "weather": task.metadata.get("weather_condition"),
                "scene_type": task.metadata.get("scene_type"),
                "object_categories": task.metadata.get("object_categories", []),
            })

        return features

    def _apply_skill(self, skill: Skill, output: dict[str, Any]) -> dict[str, Any]:
        """Apply a skill to modify the output."""
        modified = output.copy()

        if skill.action.action_type == "adjust_confidence":
            # Adjust confidence scores
            params = skill.action.parameters
            multiplier = params.get("multiplier", 1.0)
            if "confidence" in modified:
                modified["confidence"] = min(modified["confidence"] * multiplier, 1.0)

        elif skill.action.action_type == "add_context":
            # Add contextual information
            params = skill.action.parameters
            context = params.get("context", "")
            if "explanation" in modified:
                modified["explanation"] = f"{context} {modified['explanation']}"

        elif skill.action.action_type == "apply_label":
            # Override or adjust label
            params = skill.action.parameters
            label_adjustment = params.get("label_adjustment", {})
            modified.update(label_adjustment)

        return modified
