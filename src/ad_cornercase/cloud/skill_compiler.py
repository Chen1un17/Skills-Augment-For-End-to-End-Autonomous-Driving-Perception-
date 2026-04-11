"""Skill compilation from reflection outputs."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field, field_validator

from ad_cornercase.config import ProjectSettings
from ad_cornercase.schemas.skill import SkillBundle, SkillManifest
from ad_cornercase.skill_store.manifest_writer import build_skill_markdown


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return cleaned or "skill"


class SkillCompileOutput(BaseModel):
    name: str
    trigger_tags: list[str] = Field(default_factory=list)
    trigger_embedding_text: str
    focus_region: str
    dynamic_question_tree: list[str] = Field(default_factory=list)
    output_constraints: list[str] = Field(default_factory=list)
    skill_markdown: str | None = None

    @field_validator("dynamic_question_tree", "output_constraints", mode="before")
    @classmethod
    def coerce_list_like(cls, value):
        if value is None:
            return []
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    if "question" in item:
                        confidence = item.get("confidence")
                        suffix = f" (confidence={confidence})" if confidence is not None else ""
                        normalized.append(f"{item['question']}{suffix}")
                    else:
                        normalized.append("; ".join(f"{key}: {sub_value}" for key, sub_value in item.items()))
                else:
                    normalized.append(str(item))
            return normalized
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [f"{key}: {item}" for key, item in value.items()]
        return [str(value)]


class SkillCompiler:
    def __init__(self, project_settings: ProjectSettings) -> None:
        self._project_settings = project_settings

    def compile_bundle(
        self,
        *,
        case_id: str,
        output: SkillCompileOutput,
        reflection_summary: str,
    ) -> SkillBundle:
        base = _slugify(f"{output.name}-{case_id}")
        digest = hashlib.sha1(output.trigger_embedding_text.encode("utf-8")).hexdigest()[:8]
        skill_id = f"{base}-{digest}"
        manifest = SkillManifest(
            skill_id=skill_id,
            name=output.name,
            trigger_tags=output.trigger_tags,
            trigger_embedding_text=output.trigger_embedding_text,
            focus_region=output.focus_region or self._project_settings.default_focus_region,
            dynamic_question_tree=output.dynamic_question_tree,
            output_constraints=output.output_constraints or self._project_settings.skill_defaults.get("output_constraints", []),
            fallback_label=self._project_settings.fallback_label,
            source_case_id=case_id,
        )
        markdown = output.skill_markdown or build_skill_markdown(manifest, reflection_summary)
        return SkillBundle(manifest=manifest, skill_markdown=markdown)
