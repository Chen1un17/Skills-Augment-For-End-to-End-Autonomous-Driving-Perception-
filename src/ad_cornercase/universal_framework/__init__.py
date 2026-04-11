"""Universal Hierarchical Perception Framework.

A generalizable edge-cloud VLM system that adapts to diverse perception tasks
through skill-based learning without fine-tuning.
"""

from .core.perception_task import PerceptionTask, TaskSchema, TaskType, InputModality, TASK_REGISTRY
from .core.skill import Skill, SkillLibrary, SkillMatcher
from .core.edge_cloud_orchestrator import (
    EdgeCloudOrchestrator,
    OrchestratorConfig,
    ReflectionPolicy,
    ReflectionTrigger,
    POLICY_CONSERVATIVE,
    POLICY_AGGRESSIVE,
    POLICY_DISTANCE_AWARE,
    POLICY_SKILL_DRIVEN,
)
from .adaptation.skill_learner import SkillLearner, AdaptationStrategy
from .evaluation.universal_evaluator import UniversalEvaluator, MetricType

__all__ = [
    "PerceptionTask",
    "TaskSchema",
    "TaskType",
    "InputModality",
    "TASK_REGISTRY",
    "Skill",
    "SkillLibrary",
    "SkillMatcher",
    "EdgeCloudOrchestrator",
    "OrchestratorConfig",
    "ReflectionPolicy",
    "ReflectionTrigger",
    "POLICY_CONSERVATIVE",
    "POLICY_AGGRESSIVE",
    "POLICY_DISTANCE_AWARE",
    "POLICY_SKILL_DRIVEN",
    "SkillLearner",
    "AdaptationStrategy",
    "UniversalEvaluator",
    "MetricType",
]
