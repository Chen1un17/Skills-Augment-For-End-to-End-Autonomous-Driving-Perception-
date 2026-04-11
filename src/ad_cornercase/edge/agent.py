"""Edge perception agent."""

from __future__ import annotations

import json

from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.edge.uncertainty import normalized_entropy
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.base import StructuredVisionProvider
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult
from ad_cornercase.schemas.skill import SkillMatch


class EdgeAgent:
    def __init__(
        self,
        *,
        provider: StructuredVisionProvider,
        prompt_renderer: PromptRenderer,
        runtime_settings: RuntimeSettings,
        project_settings: ProjectSettings,
    ) -> None:
        self._provider = provider
        self._prompt_renderer = prompt_renderer
        self._runtime_settings = runtime_settings
        self._project_settings = project_settings

    def _benchmark_prompt_suffix(self, case: AnomalyCase) -> str:
        benchmark = str(case.metadata.get("benchmark") or "").strip().lower()
        if benchmark != "dtpqa":
            return ""
        return (
            "\n\nBenchmark-specific requirements for DTPQA:\n"
            "- `qa_report[0].question` must exactly copy the benchmark question from the input.\n"
            "- `qa_report[0].answer` must directly answer that benchmark question instead of only naming a hazard label.\n"
            "- For yes/no questions, the answer must start with exactly `Yes` or `No`.\n"
            "- If answer options are provided, answer with the option text, not the option letter.\n"
            "- Keep the benchmark answer concise and evaluation-friendly."
        )

    def _validate_case_inputs(self, case: AnomalyCase) -> None:
        benchmark = str(case.metadata.get("benchmark") or "").strip().lower()
        if benchmark == "dtpqa" and not case.image_path.exists():
            raise FileNotFoundError(
                f"DTPQA benchmark case {case.case_id} requires a local image, but none was found at "
                f"{case.image_path}. Refusing to continue because the provider would otherwise skip the image "
                "and run text-only, invalidating the benchmark."
            )

    async def perceive(self, case: AnomalyCase, skill_matches: list[SkillMatch] | None = None) -> EdgePerceptionResult:
        self._validate_case_inputs(case)
        matches = skill_matches or []
        if matches:
            instructions = self._prompt_renderer.render(
                "edge_with_skill.md",
                skill_instructions="\n\n".join(match.prompt_patch for match in matches),
            )
        else:
            instructions = self._prompt_renderer.load("edge_scene_graph.md")
        instructions += self._benchmark_prompt_suffix(case)
        prompt = json.dumps(
            {
                "question": case.question,
                "sensor_context": case.sensor_context,
                "weather_tags": case.weather_tags,
                "scene_hint": case.metadata.get("scene_hint"),
                "metadata": case.metadata,
                "fallback_label": self._project_settings.fallback_label,
            },
            ensure_ascii=False,
        )
        response = await self._provider.generate_structured(
            model=self._runtime_settings.edge_model,
            instructions=instructions,
            prompt=prompt,
            image_paths=[case.image_path],
            response_model=EdgePerceptionResult,
            metadata={"case_id": case.case_id, "skill_count": str(len(matches))},
            max_completion_tokens=self._runtime_settings.edge_max_completion_tokens,
        )
        result = response.parsed
        result.latency_ms = response.latency_ms
        result.vision_tokens = response.input_tokens + response.output_tokens
        result.entropy = normalized_entropy(result.top_k_candidates)
        result.applied_skill_ids = [match.skill_id for match in matches]
        primary_label = result.qa_report[0].answer if result.qa_report else (
            result.top_k_candidates[0].label if result.top_k_candidates else ""
        )
        result.used_fallback_label = primary_label == self._project_settings.fallback_label
        return result
