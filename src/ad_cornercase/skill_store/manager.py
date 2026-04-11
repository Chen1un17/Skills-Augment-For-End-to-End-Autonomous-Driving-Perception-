"""Dynamic skill lifecycle management."""

from __future__ import annotations

import json

from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.base import EmbeddingProvider, StructuredVisionProvider
from ad_cornercase.cloud.skill_compiler import SkillCompileOutput
from ad_cornercase.schemas.skill import (
    ReflectionDecision,
    SkillAction,
    SkillActionResult,
    SkillBundle,
    SkillManifest,
    SkillUpdate,
)
from ad_cornercase.skill_store.repository import SkillRepository


class SkillManager:
    """Manages skill lifecycle: create, update, merge, archive."""

    def __init__(
        self,
        repository: SkillRepository,
        embedding_provider: EmbeddingProvider,
        llm_provider: StructuredVisionProvider,
        prompt_renderer: PromptRenderer,
        model_name: str = "Pro/moonshotai/Kimi-K2.5",
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._prompt_renderer = prompt_renderer
        self._model_name = model_name

    async def process_reflection(
        self,
        reflection_output: dict,
        case_info: dict,
    ) -> SkillActionResult:
        """Process reflection output and decide what skill action to take."""
        should_persist = reflection_output.get("should_persist_skill", False)
        if not should_persist:
            return SkillActionResult(
                action=SkillAction.SKIP,
                skill_id=None,
                success=True,
                message="Skill persistence disabled for this case",
            )
        # Validate reflection data before attempting skill creation
        reflection_summary = reflection_output.get("reflection_summary", "")
        corrected_label = reflection_output.get("corrected_label", "")
        # Skip skill creation if reflection data is insufficient or invalid
        if not reflection_summary or len(corrected_label) <= 1:
            return SkillActionResult(
                action=SkillAction.SKIP,
                skill_id=None,
                success=True,
                message=f"Skipping skill creation: insufficient reflection data (summary={bool(reflection_summary)}, label={corrected_label!r})",
            )
        # When should_persist=True and data is valid, create a new skill
        return await self._create_new_skill(reflection_output, case_info)

    async def _decide_action(self, reflection_output: dict, active_skills: list[SkillManifest]) -> ReflectionDecision:
        """Use LLM to decide skill action."""
        prompt = self._prompt_renderer.load("skill_decide.md")
        skills_json = json.dumps([self._skill_to_dict(s) for s in active_skills], indent=2)
        input_data = {
            "corrected_label": reflection_output.get("corrected_label", ""),
            "reflection_summary": reflection_output.get("reflection_summary", ""),
            "trigger_tags": reflection_output.get("trigger_tags", []),
            "focus_region": reflection_output.get("focus_region", ""),
            "dynamic_question_tree": reflection_output.get("dynamic_question_tree", []),
            "should_persist_skill": reflection_output.get("should_persist_skill", False),
            "existing_skills": skills_json,
        }
        prompt = prompt.format(**input_data)
        try:
            response = await self._llm_provider.generate_structured(
                model=self._model_name,
                instructions="You are a skill decision engine. Return valid JSON only.",
                prompt=prompt,
                response_model=ReflectionDecision,
            )
            return response.parsed
        except Exception:
            # Fallback to create_new on error
            return ReflectionDecision(
                action=SkillAction.CREATE_NEW,
                target_skill_id=None,
                merge_candidate_ids=[],
                reason="LLM decision failed, defaulting to create_new",
                confidence=0.0,
            )

    async def _create_new_skill(self, reflection_output: dict, case_info: dict) -> SkillActionResult:
        """Create a new skill from reflection."""
        try:
            prompt = self._prompt_renderer.load("skill_compile.md")
            prompt_input = json.dumps(reflection_output, ensure_ascii=False)
            response = await self._llm_provider.generate_structured(
                model=self._model_name,
                instructions=prompt,
                prompt=prompt_input,
                response_model=SkillCompileOutput,
            )
            compiled = response.parsed
            manifest = SkillManifest(
                skill_id=self._generate_skill_id(compiled.name or "skill"),
                name=compiled.name or "Unnamed",
                trigger_tags=compiled.trigger_tags or [],
                trigger_embedding_text=compiled.trigger_embedding_text or "",
                focus_region=compiled.focus_region or "full_frame",
                dynamic_question_tree=compiled.dynamic_question_tree or [],
                output_constraints=compiled.output_constraints or [],
                fallback_label=reflection_output.get("corrected_label", "Unknown"),
                source_case_id=case_info.get("case_id", "unknown"),
                family_id=self._generate_skill_id(compiled.name or "skill"),
            )
            bundle = SkillBundle(
                manifest=manifest,
                skill_markdown=compiled.skill_markdown or "# Skill\n\n",
            )
            embedding = (await self._embedding_provider.embed([manifest.trigger_embedding_text]))[0]
            path = self._repository.save_bundle(bundle, embedding)
            return SkillActionResult(
                action=SkillAction.CREATE_NEW,
                skill_id=manifest.skill_id,
                success=True,
                message=f"Created new skill: {manifest.skill_id}",
            )
        except Exception as e:
            return SkillActionResult(
                action=SkillAction.CREATE_NEW,
                skill_id=None,
                success=False,
                message=f"Failed to create skill: {e}",
            )

    async def _update_skill(self, skill_id: str, reflection_output: dict) -> SkillActionResult:
        """Update an existing skill."""
        try:
            existing = self._repository.get_bundle(skill_id)
            prompt = self._prompt_renderer.load("skill_update.md")
            prompt_input = {
                "skill_id": skill_id,
                "name": existing.manifest.name,
                "trigger_tags": existing.manifest.trigger_tags,
                "focus_region": existing.manifest.focus_region,
                "dynamic_question_tree": existing.manifest.dynamic_question_tree,
                "output_constraints": existing.manifest.output_constraints,
                "version": existing.manifest.version,
                "corrected_label": reflection_output.get("corrected_label", ""),
                "reflection_summary": reflection_output.get("reflection_summary", ""),
                "new_trigger_tags": reflection_output.get("trigger_tags", []),
                "new_focus_region": reflection_output.get("focus_region", ""),
                "new_dynamic_question_tree": reflection_output.get("dynamic_question_tree", []),
            }
            response = await self._llm_provider.generate_structured(
                model=self._model_name,
                instructions="You are a skill update engine. Return valid JSON only.",
                prompt=json.dumps(prompt_input, ensure_ascii=False),
                response_model=dict,
            )
            compiled = response.parsed
            updated_manifest = existing.manifest.model_copy(deep=True)
            updated_manifest.name = compiled.get("name", existing.manifest.name)
            updated_manifest.trigger_tags = compiled.get("trigger_tags", existing.manifest.trigger_tags)
            updated_manifest.trigger_embedding_text = compiled.get(
                "trigger_embedding_text", existing.manifest.trigger_embedding_text
            )
            updated_manifest.focus_region = compiled.get("focus_region", existing.manifest.focus_region)
            updated_manifest.dynamic_question_tree = compiled.get(
                "dynamic_question_tree", existing.manifest.dynamic_question_tree
            )
            updated_manifest.output_constraints = compiled.get(
                "output_constraints", existing.manifest.output_constraints
            )
            updated_bundle = SkillBundle(manifest=updated_manifest, skill_markdown=compiled.get("skill_markdown", existing.skill_markdown))
            embedding = (await self._embedding_provider.embed([updated_manifest.trigger_embedding_text]))[0]
            return self._repository.update_skill(skill_id, updated_bundle, embedding)
        except Exception as e:
            return SkillActionResult(
                action=SkillAction.UPDATE_EXISTING,
                skill_id=skill_id,
                success=False,
                message=f"Failed to update skill: {e}",
            )

    async def _merge_skills(self, skill_ids: list[str], reflection_output: dict) -> SkillActionResult:
        """Merge multiple skills into one."""
        if not skill_ids:
            return SkillActionResult(
                action=SkillAction.MERGE_WITH,
                skill_id=None,
                success=False,
                message="No skills to merge",
            )
        try:
            skills_data = []
            for sid in skill_ids:
                bundle = self._repository.get_bundle(sid)
                skills_data.append(
                    {
                        "skill_id": bundle.manifest.skill_id,
                        "name": bundle.manifest.name,
                        "trigger_tags": bundle.manifest.trigger_tags,
                        "focus_region": bundle.manifest.focus_region,
                        "dynamic_question_tree": bundle.manifest.dynamic_question_tree,
                        "output_constraints": bundle.manifest.output_constraints,
                    }
                )
            prompt = self._prompt_renderer.load("skill_merge.md")
            prompt_input = {"skills_to_merge": json.dumps(skills_data, indent=2)}
            response = await self._llm_provider.generate_structured(
                model=self._model_name,
                instructions="You are a skill merge engine. Return valid JSON only.",
                prompt=prompt.format(**prompt_input),
                response_model=dict,
            )
            compiled = response.parsed
            target_id = skill_ids[0]
            merged_manifest = SkillManifest(
                skill_id=target_id,
                name=compiled.get("name", "MergedSkill"),
                trigger_tags=compiled.get("trigger_tags", []),
                trigger_embedding_text=compiled.get("trigger_embedding_text", ""),
                focus_region=compiled.get("focus_region", "full_frame"),
                dynamic_question_tree=compiled.get("dynamic_question_tree", []),
                output_constraints=compiled.get("output_constraints", []),
                fallback_label=reflection_output.get("corrected_label", "Unknown"),
                source_case_id="merged",
                family_id=target_id,
            )
            merged_bundle = SkillBundle(
                manifest=merged_manifest,
                skill_markdown=compiled.get("skill_markdown", "# Merged Skill\n\n"),
            )
            embedding = (await self._embedding_provider.embed([merged_manifest.trigger_embedding_text]))[0]
            return self._repository.merge_skills(target_id, skill_ids[1:], merged_bundle, embedding)
        except Exception as e:
            return SkillActionResult(
                action=SkillAction.MERGE_WITH,
                skill_id=skill_ids[0] if skill_ids else None,
                success=False,
                message=f"Failed to merge skills: {e}",
            )

    def _generate_skill_id(self, name: str) -> str:
        """Generate a unique skill ID from name."""
        import hashlib
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        hash_suffix = hashlib.sha1(name.encode()).hexdigest()[:8]
        return f"{safe_name}_{hash_suffix}"

    @staticmethod
    def _skill_to_dict(skill: SkillManifest) -> dict:
        return {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "trigger_tags": skill.trigger_tags,
            "focus_region": skill.focus_region,
            "dynamic_question_tree": skill.dynamic_question_tree,
            "effectiveness_score": skill.effectiveness_score,
            "usage_count": skill.usage_count,
        }
