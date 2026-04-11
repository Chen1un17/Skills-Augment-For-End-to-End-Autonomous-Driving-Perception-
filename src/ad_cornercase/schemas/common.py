"""Common schema fragments."""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class CandidateLabel(BaseModel):
    label: str
    probability: float

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if not isinstance(value, dict):
            return value
        label = value.get("label") or value.get("name") or value.get("candidate") or value.get("class")
        probability = (
            value.get("probability")
            or value.get("score")
            or value.get("confidence")
            or value.get("prob")
            or value.get("likelihood")
        )
        if label is None or probability is None:
            return value
        return {"label": label, "probability": probability}

    @field_validator("probability", mode="before")
    @classmethod
    def normalize_probability(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            had_percent = stripped.endswith("%")
            value = float(stripped.rstrip("%"))
            if had_percent:
                return value / 100.0
        numeric = float(value)
        if 1.0 < numeric <= 100.0:
            return numeric / 100.0
        return numeric


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="before")
    @classmethod
    def normalize_box(cls, value):
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return {
                "x1": int(value[0]),
                "y1": int(value[1]),
                "x2": int(value[2]),
                "y2": int(value[3]),
            }
        if isinstance(value, dict):
            x1 = value.get("x1", value.get("left", value.get("xmin")))
            y1 = value.get("y1", value.get("top", value.get("ymin")))
            x2 = value.get("x2", value.get("right", value.get("xmax")))
            y2 = value.get("y2", value.get("bottom", value.get("ymax")))
            if None not in (x1, y1, x2, y2):
                return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        return value
