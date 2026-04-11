"""Schemas for anomaly and replay inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ad_cornercase.schemas.common import BoundingBox, CandidateLabel


class AnomalyCase(BaseModel):
    case_id: str
    frame_id: str
    image_path: Path
    question: str
    ground_truth_answer: str
    crop_bbox: BoundingBox | None = None
    weather_tags: list[str] = Field(default_factory=list)
    sensor_context: str = "front_camera"
    baseline_triplets: list["SceneGraphTriplet"] = Field(default_factory=list)
    top_k_candidates: list[CandidateLabel] = Field(default_factory=list)
    entropy: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth_triplets: list["SceneGraphTriplet"] = Field(default_factory=list)


from ad_cornercase.schemas.scene_graph import SceneGraphTriplet
