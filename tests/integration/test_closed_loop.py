import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ad_cornercase.cloud.reflector import CloudReflector, ReflectionLLMOutput
from ad_cornercase.cloud.skill_compiler import SkillCompileOutput, SkillCompiler
from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.datasets.coda_lm import CodaLMDatasetLoader
from ad_cornercase.edge.agent import EdgeAgent
from ad_cornercase.edge.replay import ReplayOrchestrator
from ad_cornercase.evaluation.coda_lm_runner import CodaEvaluationRunner
from ad_cornercase.evaluation.judge_runner import JudgeRunner
from ad_cornercase.mcp.client import MCPGatewayClient
from ad_cornercase.mcp.server import create_mcp_server
from ad_cornercase.mcp.tools import CloudReflectionService
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.embeddings import HashEmbeddingProvider
from ad_cornercase.providers.judge import HeuristicJudgeProvider
from ad_cornercase.providers.openai_responses import FakeStructuredVisionProvider
from ad_cornercase.skill_store.matcher import SkillMatcher
from ad_cornercase.skill_store.repository import SkillRepository


def _edge_payload(prompt: str, metadata: dict[str, str]) -> dict:
    del prompt
    skill_count = int(metadata["skill_count"])
    if skill_count == 0:
        label = "Critical_Unknown_Obstacle"
        probabilities = [
            {"label": "Critical_Unknown_Obstacle", "probability": 0.34},
            {"label": "Construction_Debris", "probability": 0.33},
            {"label": "Traffic_Sign", "probability": 0.33},
        ]
    else:
        label = "Overturned_Truck"
        probabilities = [
            {"label": "Overturned_Truck", "probability": 0.92},
            {"label": "Construction_Debris", "probability": 0.05},
            {"label": "Traffic_Sign", "probability": 0.03},
        ]
    return {
        "triplets": [{"subject": "Ego-vehicle", "relation": "Yield_to", "object": "Overturned_Truck_30m_Ahead"}],
        "qa_report": [{"question": "What is the anomaly?", "answer": label}],
        "top_k_candidates": probabilities,
        "entropy": 0.0,
        "recommended_action": "Brake and yield",
        "latency_ms": 0.0,
        "vision_tokens": 0,
        "applied_skill_ids": [],
        "used_fallback_label": label == "Critical_Unknown_Obstacle",
    }


def _reflection_payload(prompt: str, metadata: dict[str, str]) -> dict:
    del prompt, metadata
    return {
        "corrected_label": "Overturned_Truck",
        "corrected_triplets": [{"subject": "Ego-vehicle", "relation": "Yield_to", "object": "Overturned_Truck_30m_Ahead"}],
        "reflection_summary": "Fog plus reflective tilted metal surface indicates an overturned truck.",
        "trigger_tags": ["fog", "reflective", "truck"],
        "focus_region": "lower_center",
        "dynamic_question_tree": [
            "Is there a tilted reflective stripe?",
            "Is there debris near the obstacle?",
        ],
        "output_constraints": ["Return valid JSON.", "Prefer short labels."],
        "should_persist_skill": True,
    }


def _skill_compile_payload(prompt: str, metadata: dict[str, str]) -> dict:
    del prompt, metadata
    return {
        "name": "Fog Reflective Truck",
        "trigger_tags": ["fog", "reflective", "truck"],
        "trigger_embedding_text": "fog reflective truck lower center obstacle",
        "focus_region": "lower_center",
        "dynamic_question_tree": [
            "Is there a tilted reflective stripe?",
            "Is there debris near the obstacle?",
        ],
        "output_constraints": ["Return valid JSON.", "Prefer short labels."],
        "skill_markdown": "# Fog Reflective Truck\n\n- Focus on reflective stripes in the lower center.\n",
    }


@pytest.mark.asyncio
async def test_closed_loop_replay_and_evaluation(tmp_path: Path) -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "coda_lm"
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    artifacts_dir = tmp_path / "artifacts"
    skill_dir = tmp_path / "skills"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=fixture_root,
        artifacts_dir=artifacts_dir,
        skill_store_dir=skill_dir,
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        uncertainty_entropy_threshold=0.8,
        skill_match_threshold=-1.0,
        max_skills_per_call=3,
        mcp_server_host="127.0.0.1",
        mcp_server_port=8000,
        mcp_server_url="http://127.0.0.1:8000/mcp",
        log_level="ERROR",
    )
    project_settings = ProjectSettings()
    prompt_renderer = PromptRenderer(prompts_dir)
    edge_provider = FakeStructuredVisionProvider({"EdgePerceptionResult": _edge_payload})
    cloud_provider = FakeStructuredVisionProvider(
        {
            ReflectionLLMOutput.__name__: _reflection_payload,
            SkillCompileOutput.__name__: _skill_compile_payload,
        }
    )
    repository = SkillRepository(skill_dir)
    embedding_provider = HashEmbeddingProvider()
    matcher = SkillMatcher(
        repository=repository,
        embedding_provider=embedding_provider,
        threshold=-1.0,
        max_matches=3,
    )
    reflector = CloudReflector(
        provider=cloud_provider,
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(project_settings),
    )
    service = CloudReflectionService(
        repository=repository,
        matcher=matcher,
        reflector=reflector,
        embedding_provider=embedding_provider,
    )
    server = create_mcp_server(
        service=service,
        runtime_settings=runtime_settings,
        project_settings=project_settings,
    )
    app = server.streamable_http_app()
    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    async with app.router.lifespan_context(app):
        dataset = CodaLMDatasetLoader(fixture_root)
        cases = dataset.load(split="Mini", task="region_perception")
        edge_agent = EdgeAgent(
            provider=edge_provider,
            prompt_renderer=prompt_renderer,
            runtime_settings=runtime_settings,
            project_settings=project_settings,
        )
        orchestrator = ReplayOrchestrator(
            edge_agent=edge_agent,
            runtime_settings=runtime_settings,
            project_settings=project_settings,
        )
        run_dir = await orchestrator.run_with_client_factory(
            cases=cases,
            server_url=str(runtime_settings.mcp_server_url),
            httpx_client_factory=client_factory,
        )

        predictions_path = run_dir / "predictions.jsonl"
        lines = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2
        assert lines[0]["reflection_result"]["new_skill"]["skill_id"]
        assert lines[1]["matched_skill_ids"]
        assert lines[1]["final_result"]["applied_skill_ids"]

        first_skill_id = lines[0]["reflection_result"]["new_skill"]["skill_id"]
        async with MCPGatewayClient(str(runtime_settings.mcp_server_url), httpx_client_factory=client_factory) as client:
            skill_payload = await client.read_skill(first_skill_id)
        assert skill_payload["manifest"]["skill_id"] == first_skill_id

        evaluation_runner = CodaEvaluationRunner(
            judge_runner=JudgeRunner(
                judge_provider=HeuristicJudgeProvider(),
                prompt_renderer=prompt_renderer,
            ),
            project_settings=project_settings,
        )
        report_path = await evaluation_runner.evaluate_run(run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert report_path.exists()
        assert metrics["total_cases"] == 2
        assert metrics["skill_success_rate"] >= 0.5


@pytest.mark.asyncio
async def test_replay_orchestrator_can_append_without_duplicating_cases(tmp_path: Path) -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "coda_lm"
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    artifacts_dir = tmp_path / "artifacts"
    skill_dir = tmp_path / "skills"
    runtime_settings = RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=fixture_root,
        artifacts_dir=artifacts_dir,
        skill_store_dir=skill_dir,
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        uncertainty_entropy_threshold=0.8,
        skill_match_threshold=-1.0,
        max_skills_per_call=3,
        mcp_server_host="127.0.0.1",
        mcp_server_port=8000,
        mcp_server_url="http://127.0.0.1:8000/mcp",
        log_level="ERROR",
    )
    project_settings = ProjectSettings()
    prompt_renderer = PromptRenderer(prompts_dir)
    edge_provider = FakeStructuredVisionProvider({"EdgePerceptionResult": _edge_payload})
    cloud_provider = FakeStructuredVisionProvider(
        {
            ReflectionLLMOutput.__name__: _reflection_payload,
            SkillCompileOutput.__name__: _skill_compile_payload,
        }
    )
    repository = SkillRepository(skill_dir)
    embedding_provider = HashEmbeddingProvider()
    matcher = SkillMatcher(
        repository=repository,
        embedding_provider=embedding_provider,
        threshold=-1.0,
        max_matches=3,
    )
    reflector = CloudReflector(
        provider=cloud_provider,
        prompt_renderer=prompt_renderer,
        runtime_settings=runtime_settings,
        skill_compiler=SkillCompiler(project_settings),
    )
    service = CloudReflectionService(
        repository=repository,
        matcher=matcher,
        reflector=reflector,
        embedding_provider=embedding_provider,
    )
    server = create_mcp_server(
        service=service,
        runtime_settings=runtime_settings,
        project_settings=project_settings,
    )
    app = server.streamable_http_app()

    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    async with app.router.lifespan_context(app):
        dataset = CodaLMDatasetLoader(fixture_root)
        cases = dataset.load(split="Mini", task="region_perception")
        edge_agent = EdgeAgent(
            provider=edge_provider,
            prompt_renderer=prompt_renderer,
            runtime_settings=runtime_settings,
            project_settings=project_settings,
        )
        orchestrator = ReplayOrchestrator(
            edge_agent=edge_agent,
            runtime_settings=runtime_settings,
            project_settings=project_settings,
        )
        run_dir = await orchestrator.run_with_client_factory(
            cases=cases[:1],
            server_url=str(runtime_settings.mcp_server_url),
            run_id="run-resume",
            httpx_client_factory=client_factory,
        )
        run_dir = await orchestrator.run_with_client_factory(
            cases=cases,
            server_url=str(runtime_settings.mcp_server_url),
            run_id="run-resume",
            append=True,
            httpx_client_factory=client_factory,
        )

        predictions_path = run_dir / "predictions.jsonl"
        lines = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2
        assert [line["case_id"] for line in lines] == [case.case_id for case in cases]
