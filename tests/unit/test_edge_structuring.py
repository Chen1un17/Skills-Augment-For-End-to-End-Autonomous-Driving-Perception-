from pathlib import Path

import pytest

from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.edge.agent import EdgeAgent
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.base import StructuredProviderResult, StructuredVisionProvider
from ad_cornercase.providers.openai_responses import FakeStructuredVisionProvider
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


def test_edge_perception_result_normalizes_aliases_and_preserves_triplets() -> None:
    payload = {
        "scene_graph": [{"entity": "obstacle", "predicate": "is", "value": "Overturned Truck"}],
        "qa": {"query": "What is the anomaly?", "label": "Overturned Truck"},
        "candidates": {"Overturned Truck": 0.88, "Construction Debris": 0.12},
        "action": "Decelerate immediately",
    }

    result = EdgePerceptionResult.model_validate(payload)

    assert result.qa_report[0].answer == "Overturned Truck"
    assert result.top_k_candidates[0].label == "Overturned Truck"
    assert result.recommended_action == "Decelerate immediately"
    assert result.triplets
    assert any(triplet.relation == "is" for triplet in result.triplets)


def test_edge_perception_result_accepts_three_stage_prediction_flow() -> None:
    payload = {
        "general_perception": {
            "vehicles": [
                {
                    "description": "A black SUV directly in front of the ego vehicle.",
                    "explanation": "It constrains following distance.",
                    "category_name": "Lead_Vehicle",
                }
            ],
            "vulnerable_road_users": [
                {
                    "description": "A roadside worker near the right curb.",
                    "explanation": "The worker is close to a narrowed roadway edge.",
                    "category_name": "Road_Worker",
                }
            ],
            "traffic_lights": [],
            "traffic_cones": [
                {
                    "description": "An orange traffic cone near the right margin.",
                    "explanation": "It marks the work zone boundary.",
                    "category_name": "traffic_cone",
                }
            ],
            "barriers": [],
            "other_objects": [],
            "description_and_explanation": "Road work is narrowing the right side of the lane.",
        },
        "regional_perception": [
            {
                "description": "Orange traffic cone at the right-front edge.",
                "explanation": "Conical marker for construction area.",
                "box": [267, 567, 330, 719],
                "category_name": "traffic_cone",
            }
        ],
        "driving_suggestions": {
            "summary": "slow_down_and_keep_safe_gap",
            "explanation": "Maintain distance from the lead SUV and respect the work zone.",
        },
    }

    result = EdgePerceptionResult.model_validate(payload)

    assert result.general_perception.vehicles
    assert result.regional_perception
    assert result.driving_suggestions.summary == "slow_down_and_keep_safe_gap"
    assert result.recommended_action == "slow_down_and_keep_safe_gap"
    assert result.qa_report == []
    assert result.top_k_candidates == []
    assert result.triplets == []


@pytest.mark.asyncio
async def test_edge_agent_preserves_model_inference_fields(tmp_path: Path) -> None:
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
    agent = EdgeAgent(
        provider=FakeStructuredVisionProvider(
            {
                "EdgePerceptionResult": lambda prompt, metadata: {
                    "general_perception": {
                        "vehicles": [
                            {
                                "description": "A black SUV directly ahead.",
                                "explanation": "It is the lead vehicle in the current lane.",
                                "category_name": "Lead_Vehicle",
                            }
                        ],
                        "vulnerable_road_users": [],
                        "traffic_lights": [],
                        "traffic_cones": [],
                        "barriers": [],
                        "other_objects": [],
                        "description_and_explanation": "A lead SUV is constraining ego speed.",
                    },
                    "regional_perception": [
                        {
                            "description": "Black SUV in the center lane ahead.",
                            "explanation": "Primary front object.",
                            "box": [100, 100, 300, 280],
                            "category_name": "Lead_Vehicle",
                        }
                    ],
                    "driving_suggestions": {
                        "summary": "slow_down_and_keep_safe_gap",
                        "explanation": "Maintain following distance from the lead SUV.",
                    },
                    "triplets": [
                        {"subject": "lead_vehicle", "relation": "is", "object": "Lead_Vehicle"},
                        {"subject": "ego_vehicle", "relation": "should", "object": "slow_down_and_keep_safe_gap"},
                    ],
                    "qa_report": [
                        {
                            "question": "What is the primary hazard or obstacle?",
                            "answer": "Lead_Vehicle",
                        }
                    ],
                    "top_k_candidates": [
                        {"label": "Lead_Vehicle", "probability": 0.91},
                        {"label": "Traffic_Cone", "probability": 0.09},
                    ],
                }
            }
        ),
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    case = AnomalyCase(
        case_id="image-experiment-1",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Identify the main driving-relevant entities ahead and provide the safest ego-vehicle action.",
        ground_truth_answer="Lead_Vehicle",
        sensor_context="front_camera_daylight_urban_internal_road",
        weather_tags=["daylight", "clear"],
        metadata={
            "scene_hint": (
                "A black SUV is directly ahead in the lane. Pedestrians are walking on the left roadside. "
                "Two sanitation workers stand at the right curb beside cones and a no-entry sign. "
                "The ego vehicle should slow down and keep distance."
            )
        },
    )

    result = await agent.perceive(case)

    assert result.qa_report
    assert result.qa_report[0].answer == "Lead_Vehicle"
    assert result.triplets
    assert any((triplet.subject, triplet.relation, triplet.object) == ("lead_vehicle", "is", "Lead_Vehicle") for triplet in result.triplets)
    assert any((triplet.subject, triplet.relation, triplet.object) == ("ego_vehicle", "should", "slow_down_and_keep_safe_gap") for triplet in result.triplets)
    assert result.general_perception.vehicles
    assert result.regional_perception
    assert result.recommended_action == "slow_down_and_keep_safe_gap"


@pytest.mark.asyncio
async def test_edge_agent_only_marks_fallback_when_primary_label_matches(tmp_path: Path) -> None:
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
    agent = EdgeAgent(
        provider=FakeStructuredVisionProvider(
            {
                "EdgePerceptionResult": lambda prompt, metadata: {
                    "qa_report": [
                        {"question": "Are there any pedestrians crossing the road?", "answer": "Yes"}
                    ],
                    "top_k_candidates": [
                        {"label": "Yes", "probability": 0.9},
                        {"label": "Critical_Unknown_Obstacle", "probability": 0.1},
                    ],
                    "recommended_action": "slow_down",
                }
            }
        ),
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    case = AnomalyCase(
        case_id="fallback-check",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any pedestrians crossing the road?",
        ground_truth_answer="Yes",
    )

    result = await agent.perceive(case)

    assert result.used_fallback_label is False


class _CaptureProvider(StructuredVisionProvider):
    def __init__(self) -> None:
        self.max_completion_tokens: int | None = None

    async def generate_structured(
        self,
        *,
        model: str,
        instructions: str,
        prompt: str,
        response_model: type[EdgePerceptionResult],
        image_paths=(),
        metadata=None,
        max_completion_tokens: int = 2048,
    ) -> StructuredProviderResult[EdgePerceptionResult]:
        del model, instructions, prompt, image_paths, metadata
        self.max_completion_tokens = max_completion_tokens
        parsed = response_model.model_validate(
            {
                "qa_report": [{"question": "Are there any pedestrians crossing the road?", "answer": "Yes"}],
                "top_k_candidates": [{"label": "Yes", "probability": 1.0}],
                "recommended_action": "slow_down",
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
async def test_edge_agent_uses_configured_max_completion_tokens(tmp_path: Path) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        edge_max_completion_tokens=321,
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
    provider = _CaptureProvider()
    agent = EdgeAgent(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    case = AnomalyCase(
        case_id="token-budget-check",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any pedestrians crossing the road?",
        ground_truth_answer="Yes",
    )

    await agent.perceive(case)

    assert provider.max_completion_tokens == 321


@pytest.mark.asyncio
async def test_edge_agent_adds_dtpqa_benchmark_answering_guidance(tmp_path: Path) -> None:
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

    class _InstructionCaptureProvider(StructuredVisionProvider):
        def __init__(self) -> None:
            self.instructions = ""

        async def generate_structured(
            self,
            *,
            model: str,
            instructions: str,
            prompt: str,
            response_model: type[EdgePerceptionResult],
            image_paths=(),
            metadata=None,
            max_completion_tokens: int = 2048,
        ) -> StructuredProviderResult[EdgePerceptionResult]:
            del model, prompt, image_paths, metadata, max_completion_tokens
            self.instructions = instructions
            parsed = response_model.model_validate(
                {
                    "qa_report": [{"question": "Are there any pedestrians crossing the road?", "answer": "Yes"}],
                    "top_k_candidates": [{"label": "Yes", "probability": 1.0}],
                    "recommended_action": "slow_down",
                }
            )
            return StructuredProviderResult(
                parsed=parsed,
                raw_text=parsed.model_dump_json(),
                input_tokens=10,
                output_tokens=5,
                latency_ms=1.0,
            )

    provider = _InstructionCaptureProvider()
    agent = EdgeAgent(
        provider=provider,
        prompt_renderer=PromptRenderer(prompts_dir),
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    case = AnomalyCase(
        case_id="dtpqa-guidance-check",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any pedestrians crossing the road?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa"},
    )

    await agent.perceive(case)

    assert "qa_report[0].question" in provider.instructions
    assert "must start with exactly `Yes` or `No`" in provider.instructions
