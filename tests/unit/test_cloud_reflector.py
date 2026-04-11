import json
from pathlib import Path

import pytest

from ad_cornercase.cloud.reflector import CloudReflector, ReflectionLLMOutput
from ad_cornercase.cloud.skill_compiler import SkillCompileOutput, SkillCompiler
from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.base import StructuredProviderResult, StructuredVisionProvider
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.reflection import ReflectionRequest
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


class _CaptureReflectionProvider(StructuredVisionProvider):
    def __init__(self) -> None:
        self.reflection_prompt = ""
        self.instructions = ""
        self.call_count = 0
        self.should_persist_skill = False
        self.corrected_label = "Yes"
        self.reflection_summary_text = "A small pedestrian is visible in the image."

    async def generate_structured(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        response_model: type[ReflectionLLMOutput],
        image_paths=(),
        metadata=None,
        max_completion_tokens: int = 2048,
    ) -> StructuredProviderResult[ReflectionLLMOutput]:
        del model, image_paths, metadata, max_completion_tokens
        self.call_count += 1
        self.instructions = instructions
        self.reflection_prompt = prompt
        if response_model is SkillCompileOutput:
            parsed = response_model.model_validate(
                {
                    "name": "Synth Pedestrian Skill",
                    "trigger_tags": ["pedestrian"],
                    "trigger_embedding_text": "pedestrian crossing road synth",
                    "focus_region": "full_frame",
                    "dynamic_question_tree": ["Are there any people in the image?"],
                    "output_constraints": ["Keep the answer concise."],
                    "skill_markdown": "# Synth Pedestrian Skill\n",
                }
            )
        else:
            parsed = response_model.model_validate(
                {
                    "corrected_label": self.corrected_label,
                    "corrected_triplets": [],
                    "reflection_summary": self.reflection_summary_text,
                    "trigger_tags": ["pedestrian"],
                    "focus_region": "full_frame",
                    "dynamic_question_tree": ["Are there any people in the image?"],
                    "output_constraints": ["Keep the answer concise."],
                    "should_persist_skill": self.should_persist_skill,
                }
            )
        return StructuredProviderResult(
            parsed=parsed,
            raw_text=parsed.model_dump_json(),
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
        )


@pytest.mark.asyncio
async def test_cloud_reflector_does_not_include_ground_truth_answer_in_prompt(tmp_path: Path) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=tmp_path / "coda_lm",
        artifacts_dir=tmp_path / "artifacts",
        skill_store_dir=tmp_path / "skills",
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        log_level="ERROR",
    )
    provider = _CaptureReflectionProvider()
    reflector = CloudReflector(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(ProjectSettings()),
    )
    case = AnomalyCase(
        case_id="annotations-9376",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", "subset": "real"},
    )
    baseline_result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": case.question, "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Distant_Pedestrian", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    await reflector.reflect(
        ReflectionRequest(
            anomaly_case=case,
            baseline_result=baseline_result,
            applied_skill_ids=[],
        )
    )

    prompt_payload = json.loads(provider.reflection_prompt)
    assert "ground_truth_answer" not in prompt_payload


@pytest.mark.asyncio
async def test_cloud_reflector_disables_skill_persistence_for_dtpqa(tmp_path: Path) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=tmp_path / "coda_lm",
        artifacts_dir=tmp_path / "artifacts",
        skill_store_dir=tmp_path / "skills",
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        log_level="ERROR",
    )
    provider = _CaptureReflectionProvider()
    provider.should_persist_skill = True
    reflector = CloudReflector(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(ProjectSettings()),
    )
    case = AnomalyCase(
        case_id="annotations-9594",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", "subset": "real"},
    )
    baseline_result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": case.question, "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Distant_Vehicle", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    result = await reflector.reflect(
        ReflectionRequest(
            anomaly_case=case,
            baseline_result=baseline_result,
            applied_skill_ids=[],
        )
    )

    assert result.should_persist_skill is False
    assert result.new_skill is None
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_cloud_reflector_adds_dtpqa_answer_format_guidance(tmp_path: Path) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=tmp_path / "coda_lm",
        artifacts_dir=tmp_path / "artifacts",
        skill_store_dir=tmp_path / "skills",
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        log_level="ERROR",
    )
    provider = _CaptureReflectionProvider()
    reflector = CloudReflector(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(ProjectSettings()),
    )
    case = AnomalyCase(
        case_id="annotations-9594",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", "subset": "real"},
    )
    baseline_result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": case.question, "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Distant_Pedestrian", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    await reflector.reflect(
        ReflectionRequest(
            anomaly_case=case,
            baseline_result=baseline_result,
            applied_skill_ids=[],
        )
    )

    assert "must be exactly `Yes` or `No`" in provider.instructions
    assert "Do not return symbolic hazard labels" in provider.instructions


@pytest.mark.asyncio
async def test_cloud_reflector_allows_skill_persistence_for_dtpqa_synth(tmp_path: Path) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=tmp_path / "coda_lm",
        artifacts_dir=tmp_path / "artifacts",
        skill_store_dir=tmp_path / "skills",
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        log_level="ERROR",
    )
    provider = _CaptureReflectionProvider()
    provider.should_persist_skill = True
    reflector = CloudReflector(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(ProjectSettings()),
    )
    case = AnomalyCase(
        case_id="annotations-synth-1",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", "subset": "synth"},
    )
    baseline_result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": case.question, "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Distant_Pedestrian", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    result = await reflector.reflect(
        ReflectionRequest(
            anomaly_case=case,
            baseline_result=baseline_result,
            applied_skill_ids=[],
        )
    )

    assert result.should_persist_skill is True


@pytest.mark.asyncio
async def test_cloud_reflector_normalizes_malformed_dtpqa_label_to_baseline_answer(tmp_path: Path) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=tmp_path / "coda_lm",
        artifacts_dir=tmp_path / "artifacts",
        skill_store_dir=tmp_path / "skills",
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        log_level="ERROR",
    )
    provider = _CaptureReflectionProvider()
    provider.corrected_label = ","
    provider.reflection_summary_text = ""
    provider.should_persist_skill = True
    reflector = CloudReflector(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(ProjectSettings()),
    )
    case = AnomalyCase(
        case_id="annotations-malformed-1",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="No",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", "subset": "synth"},
    )
    baseline_result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": case.question, "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Distant_Pedestrian", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    result = await reflector.reflect(
        ReflectionRequest(
            anomaly_case=case,
            baseline_result=baseline_result,
            applied_skill_ids=[],
        )
    )

    assert result.corrected_label == "No"
    assert result.reflection_summary == "Reflection returned malformed DTPQA output; fallback answer applied."
    assert result.should_persist_skill is False
