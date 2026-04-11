"""Cloud reflection orchestration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from ad_cornercase.cloud.skill_compiler import SkillCompileOutput, SkillCompiler
from ad_cornercase.config import RuntimeSettings
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.base import StructuredVisionProvider
from ad_cornercase.schemas.reflection import ReflectionRequest, ReflectionResult
from ad_cornercase.schemas.scene_graph import SceneGraphTriplet

if TYPE_CHECKING:
    from ad_cornercase.skill_store.manager import SkillManager


class ReflectionLLMOutput(BaseModel):
    corrected_label: str
    corrected_triplets: list[SceneGraphTriplet] = Field(default_factory=list)
    reflection_summary: str
    trigger_tags: list[str] = Field(default_factory=list)
    focus_region: str
    dynamic_question_tree: list[str] = Field(default_factory=list)
    output_constraints: list[str] = Field(default_factory=list)
    should_persist_skill: bool = True

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


class CloudReflector:
    def __init__(
        self,
        *,
        provider: StructuredVisionProvider,
        prompt_renderer: PromptRenderer,
        runtime_settings: RuntimeSettings,
        skill_compiler: SkillCompiler,
        skill_manager: SkillManager | None = None,
    ) -> None:
        self._provider = provider
        self._prompt_renderer = prompt_renderer
        self._runtime_settings = runtime_settings
        self._skill_compiler = skill_compiler
        self._skill_manager = skill_manager

    def set_skill_manager(self, skill_manager: SkillManager) -> None:
        """Set the skill manager after construction."""
        self._skill_manager = skill_manager

    @staticmethod
    def _benchmark_prompt_suffix(request: ReflectionRequest) -> str:
        benchmark = str(request.anomaly_case.metadata.get("benchmark") or "").strip().lower()
        if benchmark != "dtpqa":
            return ""
        return (
            "\n\nBenchmark-specific requirements for DTPQA:\n"
            "- `corrected_label` must directly answer the benchmark question from the input.\n"
            "- For yes/no questions, `corrected_label` must be exactly `Yes` or `No`.\n"
            "- Do not return symbolic hazard labels for DTPQA yes/no evaluation.\n"
            "- Keep `reflection_summary` concise and evidence-based.\n"
        )

    @staticmethod
    def _normalize_dtpqa_label(
        *,
        label: str,
        triplets: list[SceneGraphTriplet],
        fallback_answer: str,
    ) -> tuple[str, bool]:
        normalized = (label or "").strip()
        lowered = normalized.lower()
        if lowered.startswith("yes"):
            return "Yes", False
        if lowered.startswith("no"):
            return "No", False

        triplet_text = " ".join(f"{triplet.subject} {triplet.relation} {triplet.object}" for triplet in triplets).lower()
        if "pedestrian_crossing" in triplet_text or "crossing_pedestrian" in triplet_text:
            return "Yes", False
        if (
            "clear_road" in triplet_text
            or "pedestrian_roadside" in triplet_text
            or "sidewalk_user" in triplet_text
            or "no_pedestrians_crossing" in triplet_text
        ):
            return "No", False

        fallback = (fallback_answer or "").strip()
        fallback_lowered = fallback.lower()
        if fallback_lowered.startswith("yes"):
            return "Yes", True
        if fallback_lowered.startswith("no"):
            return "No", True
        return "No", True

    async def reflect(self, request: ReflectionRequest) -> ReflectionResult:
        case = request.anomaly_case
        benchmark = str(case.metadata.get("benchmark") or "").strip().lower()
        subset = str(case.metadata.get("subset") or case.metadata.get("source_split") or "").strip().lower()
        instructions = self._prompt_renderer.load("cloud_reflection.md") + self._benchmark_prompt_suffix(request)
        prompt = json.dumps(
            {
                "question": case.question,
                "sensor_context": case.sensor_context,
                "weather_tags": case.weather_tags,
                "scene_hint": case.metadata.get("scene_hint"),
                "baseline_result": request.baseline_result.model_dump(mode="json"),
                "applied_skill_ids": request.applied_skill_ids,
                "metadata": case.metadata,
            },
            ensure_ascii=False,
        )
        reflection_response = await self._provider.generate_structured(
            model=self._runtime_settings.cloud_model,
            instructions=instructions,
            prompt=prompt,
            image_paths=[case.image_path],
            response_model=ReflectionLLMOutput,
            metadata={"case_id": case.case_id, "stage": "reflection"},
        )
        llm_output = reflection_response.parsed
        if benchmark == "dtpqa":
            fallback_answer = ""
            if request.baseline_result.qa_report:
                fallback_answer = request.baseline_result.qa_report[0].answer
            elif request.baseline_result.top_k_candidates:
                fallback_answer = request.baseline_result.top_k_candidates[0].label
            normalized_label, used_fallback = self._normalize_dtpqa_label(
                label=llm_output.corrected_label,
                triplets=llm_output.corrected_triplets,
                fallback_answer=fallback_answer,
            )
            llm_output.corrected_label = normalized_label
            if not llm_output.reflection_summary.strip():
                llm_output.reflection_summary = "Reflection returned malformed DTPQA output; fallback answer applied."
            if used_fallback:
                llm_output.should_persist_skill = False
        if benchmark == "dtpqa" and subset == "real":
            llm_output.should_persist_skill = False
        # Build result with correction
        result = ReflectionResult(
            corrected_label=llm_output.corrected_label,
            corrected_triplets=llm_output.corrected_triplets,
            reflection_summary=llm_output.reflection_summary,
            should_persist_skill=llm_output.should_persist_skill,
        )
        # Use SkillManager for dynamic skill lifecycle if available
        if self._skill_manager is not None and llm_output.should_persist_skill:
            reflection_dict = llm_output.model_dump(mode="json")
            reflection_dict["corrected_triplets"] = [
                t.model_dump() if hasattr(t, "model_dump") else t for t in llm_output.corrected_triplets
            ]
            case_info = {
                "case_id": case.case_id,
                "question": case.question,
                "metadata": case.metadata,
            }
            action_result = await self._skill_manager.process_reflection(reflection_dict, case_info)
            # Mark skill as generated if action was successful
            if action_result.success and action_result.skill_id:
                result.new_skill_id = action_result.skill_id
                # Fetch full skill bundle and set new_skill/skill_markdown
                try:
                    bundle = self._skill_manager._repository.get_bundle(action_result.skill_id)
                    result.new_skill = bundle.manifest
                    result.skill_markdown = bundle.skill_markdown
                except Exception:
                    pass  # Fallback: just set new_skill_id
        elif llm_output.should_persist_skill:
            # Fallback to direct skill compilation if no SkillManager
            compile_instructions = self._prompt_renderer.load("skill_compile.md")
            compile_prompt = json.dumps(llm_output.model_dump(mode="json"), ensure_ascii=False)
            compile_response = await self._provider.generate_structured(
                model=self._runtime_settings.cloud_model,
                instructions=compile_instructions,
                prompt=compile_prompt,
                response_model=SkillCompileOutput,
                metadata={"case_id": case.case_id, "stage": "skill_compile"},
            )
            bundle = self._skill_compiler.compile_bundle(
                case_id=case.case_id,
                output=compile_response.parsed,
                reflection_summary=llm_output.reflection_summary,
            )
            result.new_skill = bundle.manifest
            result.skill_markdown = bundle.skill_markdown
        return result
