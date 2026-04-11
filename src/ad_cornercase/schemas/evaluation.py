"""Schemas for evaluation artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ad_cornercase.schemas.reflection import ReflectionResult
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


class CasePredictionRecord(BaseModel):
    case_id: str
    question: str
    ground_truth_answer: str
    baseline_result: EdgePerceptionResult
    final_result: EdgePerceptionResult
    matched_skill_ids: list[str] = Field(default_factory=list)
    reflection_result: ReflectionResult | None = None
    judge_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgementScore(BaseModel):
    score: float
    rationale: str
    hallucination_risk: str


class EvaluationSummary(BaseModel):
    run_id: str
    total_cases: int
    judge_score_mean: float
    regional_triplet_recall: float
    skill_success_rate: float
    latency_delta_ms: float
    vision_token_delta: float
    exact_match_accuracy: float | None = None
    distance_bin_accuracy: dict[str, float] = Field(default_factory=dict)
    distance_bin_counts: dict[str, int] = Field(default_factory=dict)
    distance_group_accuracy: dict[str, float] = Field(default_factory=dict)
    distance_group_judge_score_mean: dict[str, float] = Field(default_factory=dict)
