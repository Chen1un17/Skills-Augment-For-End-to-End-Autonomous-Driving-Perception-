"""Schemas for skills and matching."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class SkillStatus(str, Enum):
    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"


class SkillAction(str, Enum):
    CREATE_NEW = "create_new"
    UPDATE_EXISTING = "update_existing"
    MERGE_WITH = "merge_with"
    SKIP = "skip"


class ReflectionDecision(BaseModel):
    action: SkillAction
    target_skill_id: str | None = None
    merge_candidate_ids: list[str] = Field(default_factory=list)
    reason: str
    confidence: float


class SkillManifest(BaseModel):
    skill_id: str
    name: str
    version: str = "0.1.0"
    trigger_tags: list[str] = Field(default_factory=list)
    trigger_embedding_text: str
    focus_region: str
    dynamic_question_tree: list[str] = Field(default_factory=list)
    output_constraints: list[str] = Field(default_factory=list)
    fallback_label: str
    source_case_id: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Dynamic skill management fields
    parent_skill_id: str | None = None
    family_id: str | None = None
    effectiveness_score: float = 0.0
    usage_count: int = 0
    last_used_at: datetime | None = None
    status: SkillStatus = SkillStatus.ACTIVE


class SkillBundle(BaseModel):
    manifest: SkillManifest
    skill_markdown: str


class SkillMatch(BaseModel):
    skill_id: str
    score: float
    prompt_patch: str
    manifest: SkillManifest


class SkillMatchRequest(BaseModel):
    case_id: str
    sensor_context: str
    weather_tags: list[str] = Field(default_factory=list)
    top_k_labels: list[str] = Field(default_factory=list)
    entropy: float
    trigger_text: str


class SkillMatchResult(BaseModel):
    matches: list[SkillMatch] = Field(default_factory=list)


class SkillUpdate(BaseModel):
    """Update payload for an existing skill."""
    name: str | None = None
    trigger_tags: list[str] | None = None
    trigger_embedding_text: str | None = None
    focus_region: str | None = None
    dynamic_question_tree: list[str] | None = None
    output_constraints: list[str] | None = None
    effectiveness_score: float | None = None


class SkillActionResult(BaseModel):
    """Result of a skill action (create/update/merge)."""
    action: SkillAction
    skill_id: str | None = None
    success: bool
    message: str
