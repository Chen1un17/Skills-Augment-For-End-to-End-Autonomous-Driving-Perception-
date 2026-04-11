"""Schemas for scene graph outputs."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ad_cornercase.schemas.common import BoundingBox, CandidateLabel


GENERAL_CATEGORY_KEYS = (
    "vehicles",
    "vulnerable_road_users",
    "traffic_lights",
    "traffic_cones",
    "barriers",
    "other_objects",
)


def _first_present(mapping: dict[str, Any], *keys: str):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _normalize_candidate_entries(value):
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, str) for item in value):
            probability = 1.0 / len(value)
            return [{"label": item, "probability": probability} for item in value]
        return value
    if isinstance(value, dict):
        nested = _first_present(value, "items", "candidates", "top_k_candidates", "labels")
        if nested is not None:
            return _normalize_candidate_entries(nested)
        return [{"label": label, "probability": probability} for label, probability in value.items()]
    return value


def _normalize_qa_entries(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        nested = _first_present(value, "items", "qa_report", "qa", "answers")
        if nested is not None:
            return _normalize_qa_entries(nested)
        answer = _first_present(value, "answer", "label", "prediction", "response", "value", "classification")
        if answer is not None:
            question = _first_present(value, "question", "query", "prompt") or "What is the anomaly?"
            return [{"question": question, "answer": str(answer)}]
    if isinstance(value, str):
        return [{"question": "What is the anomaly?", "answer": value}]
    return value


def _normalize_general_items(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        nested = _first_present(value, "items", "entries", "objects", "detections")
        if nested is not None:
            return _normalize_general_items(nested)

        grouped: dict[str, dict[str, str]] = {}
        for key, item in value.items():
            match = re.match(r"(.+?)_(description|explanation)$", key)
            if match and item is not None:
                prefix, kind = match.groups()
                grouped.setdefault(prefix, {})[kind] = str(item)
        if grouped:
            normalized = []
            for prefix, item in grouped.items():
                normalized.append(
                    {
                        "object_id": prefix,
                        "description": item.get("description", ""),
                        "explanation": item.get("explanation", ""),
                    }
                )
            return normalized

        if "description" in value or "explanation" in value:
            return [value]

        normalized = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                normalized.extend(_normalize_general_items(item))
            elif item is not None:
                normalized.append({"object_id": str(key), "description": str(item), "explanation": ""})
        return normalized
    if isinstance(value, str):
        return [{"description": value, "explanation": ""}]
    return []


def _normalize_regional_items(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        nested = _first_present(value, "items", "entries", "objects", "detections", "regions")
        if nested is not None:
            return _normalize_regional_items(nested)
        if {"description", "explanation", "bbox", "box", "category_name", "category", "label"} & set(value.keys()):
            return [value]
        normalized = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                normalized.extend(_normalize_regional_items(item))
            elif item is not None:
                normalized.append({"object_id": str(key), "description": str(item)})
        return normalized
    if isinstance(value, str):
        return [{"description": value, "category_name": "Unknown_Object"}]
    return []


class GeneralPerceptionItem(BaseModel):
    description: str = ""
    explanation: str = ""
    category_name: str | None = None
    object_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if isinstance(value, str):
            return {"description": value, "explanation": ""}
        if not isinstance(value, dict):
            return value
        description = _first_present(value, "description", "desc", "summary", "text") or ""
        explanation = _first_present(value, "explanation", "reason", "rationale", "analysis") or ""
        category_name = _first_present(value, "category_name", "label", "name", "object", "entity")
        object_id = _first_present(value, "object_id", "id", "key")
        return {
            "description": str(description),
            "explanation": str(explanation),
            "category_name": str(category_name) if category_name is not None else None,
            "object_id": str(object_id) if object_id is not None else None,
        }


class GeneralPerceptionSection(BaseModel):
    vehicles: list[GeneralPerceptionItem] = Field(default_factory=list)
    vulnerable_road_users: list[GeneralPerceptionItem] = Field(default_factory=list)
    traffic_lights: list[GeneralPerceptionItem] = Field(default_factory=list)
    traffic_cones: list[GeneralPerceptionItem] = Field(default_factory=list)
    barriers: list[GeneralPerceptionItem] = Field(default_factory=list)
    other_objects: list[GeneralPerceptionItem] = Field(default_factory=list)
    description_and_explanation: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for key in GENERAL_CATEGORY_KEYS:
            category_value = normalized.get(key)
            if category_value is not None:
                normalized[key] = _normalize_general_items(category_value)
        summary = _first_present(
            normalized,
            "description_and_explanation",
            "overall_description",
            "overall_summary",
            "summary",
        )
        if summary is not None:
            normalized["description_and_explanation"] = str(summary)
        return normalized


class RegionalPerceptionItem(BaseModel):
    description: str = ""
    explanation: str = ""
    box: BoundingBox | None = None
    category_name: str = "Unknown_Object"
    object_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if isinstance(value, str):
            return {"description": value, "category_name": "Unknown_Object"}
        if not isinstance(value, dict):
            return value
        description = _first_present(value, "description", "desc", "summary", "text") or ""
        explanation = _first_present(value, "explanation", "reason", "rationale", "analysis") or ""
        box = _first_present(value, "box", "bbox", "bounding_box", "coordinates")
        category_name = _first_present(value, "category_name", "category", "label", "name", "object") or "Unknown_Object"
        object_id = _first_present(value, "object_id", "id", "key")
        return {
            "description": str(description),
            "explanation": str(explanation),
            "box": box,
            "category_name": str(category_name),
            "object_id": str(object_id) if object_id is not None else None,
        }


class DrivingSuggestion(BaseModel):
    summary: str = ""
    explanation: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if isinstance(value, str):
            return {"summary": value, "explanation": ""}
        if not isinstance(value, dict):
            return value
        summary = _first_present(value, "summary", "recommended_action", "action", "suggestion", "advice") or ""
        explanation = _first_present(value, "explanation", "reason", "rationale", "description") or ""
        return {"summary": str(summary), "explanation": str(explanation)}


class SceneGraphTriplet(BaseModel):
    subject: str
    relation: str
    object: str

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if not isinstance(value, dict):
            return value
        subject = value.get("subject") or value.get("entity") or value.get("source")
        relation = (
            value.get("relation")
            or value.get("predicate")
            or value.get("action")
            or value.get("attribute")
            or value.get("state")
            or value.get("intent")
            or "related_to"
        )
        object_value = value.get("object") or value.get("target") or value.get("value")
        if subject is None or object_value is None:
            return value
        return {
            "subject": subject,
            "relation": relation,
            "object": object_value,
        }


class QAItem(BaseModel):
    question: str
    answer: str

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if not isinstance(value, dict):
            return value
        question = value.get("question") or value.get("query") or value.get("prompt") or "What is the anomaly?"
        answer = value.get("answer") or value.get("label") or value.get("prediction") or value.get("response")
        if answer is None:
            return value
        return {"question": question, "answer": str(answer)}


class EdgePerceptionResult(BaseModel):
    general_perception: GeneralPerceptionSection = Field(default_factory=GeneralPerceptionSection)
    regional_perception: list[RegionalPerceptionItem] = Field(default_factory=list)
    driving_suggestions: DrivingSuggestion = Field(default_factory=DrivingSuggestion)
    triplets: list[SceneGraphTriplet] = Field(default_factory=list)
    qa_report: list[QAItem] = Field(default_factory=list)
    top_k_candidates: list[CandidateLabel] = Field(default_factory=list)
    entropy: float = 0.0
    recommended_action: str = ""
    latency_ms: float = 0.0
    vision_tokens: int = 0
    applied_skill_ids: list[str] = Field(default_factory=list)
    used_fallback_label: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)

        general = _first_present(normalized, "general_perception", "task1_general_perception", "task1", "general")
        if general is None and any(key in normalized for key in GENERAL_CATEGORY_KEYS):
            general = {key: normalized.get(key) for key in GENERAL_CATEGORY_KEYS if key in normalized}
            summary = _first_present(
                normalized,
                "description_and_explanation",
                "overall_description",
                "overall_summary",
            )
            if summary is not None:
                general["description_and_explanation"] = summary
        if general is not None:
            normalized["general_perception"] = general

        regional = _first_present(
            normalized,
            "regional_perception",
            "task2_regional_perception",
            "task2",
            "regions",
            "detections",
        )
        if regional is not None:
            normalized["regional_perception"] = _normalize_regional_items(regional)

        suggestions = _first_present(
            normalized,
            "driving_suggestions",
            "task3_driving_suggestions",
            "task3",
            "driving_advice",
            "suggestions",
        )
        if suggestions is not None:
            normalized["driving_suggestions"] = suggestions

        triplets = _first_present(
            normalized,
            "triplets",
            "scene_graph",
            "scene_graph_triplets",
            "graph",
            "relationships",
            "relation_triplets",
        )
        if isinstance(triplets, dict):
            triplets = _first_present(triplets, "triplets", "scene_graph", "relations", "relationships")
        if triplets is not None:
            normalized["triplets"] = triplets

        qa_report = _normalize_qa_entries(
            _first_present(normalized, "qa_report", "qa", "qa_items", "question_answers", "reasoning_qa", "answers")
        )
        if qa_report is None:
            qa_report = _normalize_qa_entries(
                _first_present(normalized, "answer", "label", "predicted_label", "prediction", "classification")
            )
        if qa_report is not None:
            normalized["qa_report"] = qa_report

        candidates = _normalize_candidate_entries(
            _first_present(normalized, "top_k_candidates", "candidates", "top_candidates", "candidate_labels", "labels")
        )
        if candidates is None:
            top_label = _first_present(normalized, "label", "predicted_label", "prediction", "classification")
            if isinstance(top_label, str):
                candidates = [{"label": top_label, "probability": 1.0}]
        if candidates is not None:
            normalized["top_k_candidates"] = candidates

        action = _first_present(
            normalized,
            "recommended_action",
            "action",
            "recommendation",
            "decision",
            "next_action",
        )
        if action is not None:
            normalized["recommended_action"] = str(action)

        return normalized

    def _primary_label(self) -> str | None:
        if self.qa_report:
            return self.qa_report[0].answer
        if self.top_k_candidates:
            return self.top_k_candidates[0].label
        return None

    @model_validator(mode="after")
    def ensure_minimum_structure(self):
        if not self.recommended_action and self.driving_suggestions.summary:
            self.recommended_action = self.driving_suggestions.summary

        if not self.driving_suggestions.summary and self.recommended_action:
            self.driving_suggestions = DrivingSuggestion(summary=self.recommended_action, explanation="")

        primary_label = self._primary_label()
        if primary_label and not self.top_k_candidates:
            self.top_k_candidates = [CandidateLabel(label=primary_label, probability=1.0)]

        if primary_label and not self.qa_report:
            self.qa_report = [QAItem(question="What is the anomaly?", answer=primary_label)]

        return self
