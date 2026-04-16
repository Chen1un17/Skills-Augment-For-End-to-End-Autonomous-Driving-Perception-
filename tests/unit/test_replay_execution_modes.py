from pathlib import Path

import pytest

from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.edge.agent import EdgeAgent
from ad_cornercase.edge.replay import ReplayOrchestrator
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.openai_responses import FakeStructuredVisionProvider
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.skill import SkillMatchResult


def _runtime_settings(tmp_path: Path) -> RuntimeSettings:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    return RuntimeSettings(
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


def _case(tmp_path: Path) -> AnomalyCase:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"not-a-real-image-but-exists")
    return AnomalyCase(
        case_id="case-cloud-only",
        frame_id="frame-001",
        image_path=image_path,
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", "distance_group": "far"},
    )


def _payload(label: str):
    def _handler(prompt: str, metadata: dict[str, str]) -> dict:
        del prompt, metadata
        return {
            "qa_report": [{"question": "Are there any people in the image?", "answer": label}],
            "top_k_candidates": [{"label": label, "probability": 1.0}],
            "recommended_action": "slow_down",
        }

    return _handler


def _edge_false_negative_payload(prompt: str, metadata: dict[str, str]) -> dict:
    del prompt, metadata
    return {
        "qa_report": [{"question": "Are there any people in the image?", "answer": "No"}],
        "top_k_candidates": [
            {"label": "Clear_Roadway", "probability": 0.9},
            {"label": "Distant_Pedestrian", "probability": 0.1},
        ],
        "recommended_action": "maintain_current_speed_and_monitor",
    }


@pytest.mark.asyncio
async def test_replay_orchestrator_cloud_only_uses_cloud_agent_without_mcp(tmp_path: Path) -> None:
    runtime_settings = _runtime_settings(tmp_path)
    prompt_renderer = PromptRenderer(runtime_settings.prompts_dir)
    edge_agent = EdgeAgent(
        provider=FakeStructuredVisionProvider({"EdgePerceptionResult": _edge_false_negative_payload}),
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    cloud_agent = EdgeAgent(
        provider=FakeStructuredVisionProvider({"EdgePerceptionResult": _payload("Yes")}),
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    orchestrator = ReplayOrchestrator(
        edge_agent=edge_agent,
        cloud_perception_agent=cloud_agent,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )

    run_dir = await orchestrator.run(
        cases=[_case(tmp_path)],
        server_url=None,
        run_id="run-cloud-only",
        execution_mode="cloud_only",
    )

    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"execution_mode":"cloud_only"' in lines[0]
    assert '"reflection_result":null' in lines[0]
    assert '"answer":"Yes"' in lines[0]


class _NoSkillClient:
    async def match_skills(self, request):  # noqa: ANN001
        del request
        return SkillMatchResult(matches=[])

    async def reflect_anomaly(self, request):  # noqa: ANN001
        del request
        raise AssertionError("DTPQA hybrid path should use direct cloud perception instead of MCP reflection")


class _ReflectionClient:
    async def match_skills(self, request):  # noqa: ANN001
        del request
        return SkillMatchResult(matches=[])

    async def reflect_anomaly(self, request):  # noqa: ANN001
        del request
        from ad_cornercase.schemas.reflection import ReflectionResult

        return ReflectionResult(
            corrected_label="Yes",
            reflection_summary="Reflection path used instead of direct cloud reroute.",
            should_persist_skill=True,
        )


@pytest.mark.asyncio
async def test_replay_orchestrator_hybrid_uses_direct_cloud_perception_for_dtpqa_people_cases(tmp_path: Path) -> None:
    runtime_settings = _runtime_settings(tmp_path)
    prompt_renderer = PromptRenderer(runtime_settings.prompts_dir)
    edge_agent = EdgeAgent(
        provider=FakeStructuredVisionProvider({"EdgePerceptionResult": _edge_false_negative_payload}),
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    cloud_agent = EdgeAgent(
        provider=FakeStructuredVisionProvider({"EdgePerceptionResult": _payload("Yes")}),
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    orchestrator = ReplayOrchestrator(
        edge_agent=edge_agent,
        cloud_perception_agent=cloud_agent,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )

    record = await orchestrator._run_hybrid_case(_case(tmp_path), _NoSkillClient())  # noqa: SLF001

    assert record.final_result.qa_report[0].answer == "Yes"
    assert record.reflection_result is not None
    assert record.reflection_result.corrected_label == "Yes"
    assert record.reflection_result.should_persist_skill is False
    assert record.metadata["hybrid_strategy"] == "direct_cloud_perception"


@pytest.mark.asyncio
async def test_replay_orchestrator_can_disable_direct_cloud_reroute_for_category1(tmp_path: Path) -> None:
    runtime_settings = _runtime_settings(tmp_path)
    runtime_settings.enable_dtpqa_category1_direct_cloud_reroute = False
    prompt_renderer = PromptRenderer(runtime_settings.prompts_dir)
    edge_agent = EdgeAgent(
        provider=FakeStructuredVisionProvider({"EdgePerceptionResult": _edge_false_negative_payload}),
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    cloud_agent = EdgeAgent(
        provider=FakeStructuredVisionProvider({"EdgePerceptionResult": _payload("Yes")}),
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )
    orchestrator = ReplayOrchestrator(
        edge_agent=edge_agent,
        cloud_perception_agent=cloud_agent,
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )

    record = await orchestrator._run_hybrid_case(_case(tmp_path), _ReflectionClient())  # noqa: SLF001

    assert record.reflection_result is not None
    assert record.reflection_result.should_persist_skill is True
    assert record.metadata["hybrid_strategy"] == "mcp_reflection"
