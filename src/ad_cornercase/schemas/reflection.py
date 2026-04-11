"""Schemas for cloud reflection."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult, SceneGraphTriplet
from ad_cornercase.schemas.skill import SkillManifest


class ReflectionRequest(BaseModel):
    anomaly_case: AnomalyCase
    baseline_result: EdgePerceptionResult
    applied_skill_ids: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    corrected_label: str
    corrected_triplets: list[SceneGraphTriplet] = Field(default_factory=list)
    reflection_summary: str
    new_skill: SkillManifest | None = None
    skill_markdown: str | None = None
    new_skill_id: str | None = None
    should_persist_skill: bool = False
