"""Run a real SiliconFlow-backed demo without binding a local port."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import typer
from openai import AsyncOpenAI

from ad_cornercase.bootstrap import build_cloud_service, build_edge_agent, build_judge_provider, build_prompt_renderer, load_settings
from ad_cornercase.demo_cases import build_experiment_image_cases, build_siliconflow_demo_cases
from ad_cornercase.edge.replay import ReplayOrchestrator
from ad_cornercase.evaluation.coda_lm_runner import CodaEvaluationRunner
from ad_cornercase.evaluation.judge_runner import JudgeRunner
from ad_cornercase.logging import configure_logging
from ad_cornercase.mcp.server import create_mcp_server
from ad_cornercase.skill_store.repository import SkillRepository

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _probe_overrides(model: str) -> dict[str, object]:
    normalized = model.lower()
    if (
        normalized.startswith("qwen/qwen3")
        and "-vl" not in normalized
        and "omni" not in normalized
        and "thinking" not in normalized
    ):
        return {"enable_thinking": False}
    return {}


async def _probe_chat_model(api_key: str, base_url: str, model: str, timeout: float) -> dict:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=64,
            messages=[
                {"role": "system", "content": "Reply in one short sentence."},
                {"role": "user", "content": "Say hello and identify yourself."},
            ],
            extra_body=_probe_overrides(model),
        )
        return {
            "requested_model": model,
            "status": "ok",
            "message": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else None,
        }
    except Exception as exc:
        return {
            "requested_model": model,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def _load_local_skill_payload(skill_store_dir: Path, skill_id: str | None) -> dict | None:
    if not skill_id:
        return None
    try:
        bundle = SkillRepository(skill_store_dir).get_bundle(skill_id)
    except FileNotFoundError:
        return None
    return {
        "manifest": bundle.manifest.model_dump(mode="json"),
        "skill_markdown": bundle.skill_markdown,
    }


async def _run(limit: int | None = None) -> None:
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)
    edge_probe = await _probe_chat_model(
        runtime_settings.openai_api_key,
        runtime_settings.openai_base_url,
        runtime_settings.edge_model,
        timeout=15,
    )
    effective_edge_model = runtime_settings.edge_model
    if edge_probe["status"] != "ok":
        effective_edge_model = runtime_settings.cloud_model

    demo_runtime_settings = runtime_settings.model_copy(
        update={
            "edge_model": effective_edge_model,
            "request_timeout_seconds": max(runtime_settings.request_timeout_seconds, 90.0),
        }
    )

    demo_asset = demo_runtime_settings.project_root() / "data" / "demo" / "placeholder_scene.svg"
    cases = build_siliconflow_demo_cases(demo_asset)
    if limit is not None:
        cases = cases[:limit]

    service = build_cloud_service(demo_runtime_settings, project_settings)
    server = create_mcp_server(
        service=service,
        runtime_settings=demo_runtime_settings,
        project_settings=project_settings,
    )
    asgi_app = server.streamable_http_app()

    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app),
            base_url="http://127.0.0.1:8000",
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    orchestrator = ReplayOrchestrator(
        edge_agent=build_edge_agent(demo_runtime_settings, project_settings),
        runtime_settings=demo_runtime_settings,
        project_settings=project_settings,
    )

    async with asgi_app.router.lifespan_context(asgi_app):
        run_dir = await orchestrator.run_with_client_factory(
            cases=cases,
            server_url="http://127.0.0.1:8000/mcp",
            httpx_client_factory=client_factory,
        )

    predictions_path = run_dir / "predictions.jsonl"
    records = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    first_skill_id = None
    if records:
        first_reflection = records[0].get("reflection_result") or {}
        first_skill_id = ((first_reflection.get("new_skill") or {}).get("skill_id")) or next(
            iter(records[0].get("matched_skill_ids") or []),
            None,
        )

    skill_payload = _load_local_skill_payload(demo_runtime_settings.skill_store_dir, first_skill_id)

    evaluation_runner = CodaEvaluationRunner(
        judge_runner=JudgeRunner(
            judge_provider=build_judge_provider(demo_runtime_settings),
            prompt_renderer=build_prompt_renderer(demo_runtime_settings),
        ),
        project_settings=project_settings,
    )
    report_path = await evaluation_runner.evaluate_run(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    summary = {
        "run_dir": str(run_dir),
        "predictions_path": str(run_dir / "predictions.jsonl"),
        "pretty_predictions_path": str(run_dir / "predictions.pretty.json"),
        "report_path": str(report_path),
        "edge_probe": edge_probe,
        "effective_edge_model": effective_edge_model,
        "requested_edge_model": runtime_settings.edge_model,
        "metrics": metrics,
        "scene_results": [
            {
                "case_id": record["case_id"],
                "question": record["question"],
                "matched_skill_ids": record["matched_skill_ids"],
                "general_perception": record["final_result"].get("general_perception"),
                "regional_perception": record["final_result"].get("regional_perception"),
                "driving_suggestions": record["final_result"].get("driving_suggestions"),
                "final_qa": record["final_result"]["qa_report"],
                "final_triplets": record["final_result"]["triplets"],
                "final_top_k_candidates": record["final_result"]["top_k_candidates"],
                "recommended_action": record["final_result"]["recommended_action"],
                "reflection_summary": (record.get("reflection_result") or {}).get("reflection_summary"),
            }
            for record in records
        ],
        "skill_result": skill_payload,
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def run(
    limit: int | None = typer.Option(None),
    image_path: Path | None = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
    edge_only: bool = typer.Option(False),
) -> None:
    if image_path is not None:
        async def _run_with_image() -> None:
            runtime_settings, project_settings = load_settings()
            configure_logging(runtime_settings.log_level)
            edge_probe = await _probe_chat_model(
                runtime_settings.openai_api_key,
                runtime_settings.openai_base_url,
                runtime_settings.edge_model,
                timeout=15,
            )
            effective_edge_model = runtime_settings.edge_model
            if edge_probe["status"] != "ok":
                effective_edge_model = runtime_settings.cloud_model

            demo_runtime_settings = runtime_settings.model_copy(
                update={
                    "edge_model": effective_edge_model,
                    "max_retries": 2,
                    "request_timeout_seconds": max(runtime_settings.request_timeout_seconds, 180.0),
                }
            )

            cases = build_experiment_image_cases(image_path)
            if limit is not None:
                cases = cases[:limit]

            if edge_only:
                edge_agent = build_edge_agent(demo_runtime_settings, project_settings)
                result = await edge_agent.perceive(cases[0])
                summary = {
                    "requested_edge_model": runtime_settings.edge_model,
                    "effective_edge_model": effective_edge_model,
                    "edge_probe": edge_probe,
                    "experiment_image_path": str(image_path),
                    "case_id": cases[0].case_id,
                    "question": cases[0].question,
                    "general_perception": result.general_perception.model_dump(mode="json"),
                    "regional_perception": [item.model_dump(mode="json") for item in result.regional_perception],
                    "driving_suggestions": result.driving_suggestions.model_dump(mode="json"),
                    "triplets": [triplet.model_dump(mode="json") for triplet in result.triplets],
                    "qa_report": [item.model_dump(mode="json") for item in result.qa_report],
                    "top_k_candidates": [item.model_dump(mode="json") for item in result.top_k_candidates],
                    "recommended_action": result.recommended_action,
                    "latency_ms": result.latency_ms,
                    "vision_tokens": result.vision_tokens,
                }
                typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
                return

            service = build_cloud_service(demo_runtime_settings, project_settings)
            server = create_mcp_server(
                service=service,
                runtime_settings=demo_runtime_settings,
                project_settings=project_settings,
            )
            asgi_app = server.streamable_http_app()

            def client_factory(headers=None, timeout=None, auth=None):
                return httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=asgi_app),
                    base_url="http://127.0.0.1:8000",
                    headers=headers,
                    timeout=timeout,
                    auth=auth,
                )

            orchestrator = ReplayOrchestrator(
                edge_agent=build_edge_agent(demo_runtime_settings, project_settings),
                runtime_settings=demo_runtime_settings,
                project_settings=project_settings,
            )

            async with asgi_app.router.lifespan_context(asgi_app):
                run_dir = await orchestrator.run_with_client_factory(
                    cases=cases,
                    server_url="http://127.0.0.1:8000/mcp",
                    httpx_client_factory=client_factory,
                )

            predictions_path = run_dir / "predictions.jsonl"
            records = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]

            first_skill_id = None
            if records:
                first_reflection = records[0].get("reflection_result") or {}
                first_skill_id = ((first_reflection.get("new_skill") or {}).get("skill_id")) or next(
                    iter(records[0].get("matched_skill_ids") or []),
                    None,
                )

            skill_payload = _load_local_skill_payload(demo_runtime_settings.skill_store_dir, first_skill_id)

            evaluation_runner = CodaEvaluationRunner(
                judge_runner=JudgeRunner(
                    judge_provider=build_judge_provider(demo_runtime_settings),
                    prompt_renderer=build_prompt_renderer(demo_runtime_settings),
                ),
                project_settings=project_settings,
            )
            report_path = await evaluation_runner.evaluate_run(run_dir)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

            summary = {
                "run_dir": str(run_dir),
                "predictions_path": str(run_dir / "predictions.jsonl"),
                "pretty_predictions_path": str(run_dir / "predictions.pretty.json"),
                "report_path": str(report_path),
                "edge_probe": edge_probe,
                "effective_edge_model": effective_edge_model,
                "requested_edge_model": runtime_settings.edge_model,
                "experiment_image_path": str(image_path),
                "metrics": metrics,
                "scene_results": [
                    {
                        "case_id": record["case_id"],
                        "question": record["question"],
                        "matched_skill_ids": record["matched_skill_ids"],
                        "general_perception": record["final_result"].get("general_perception"),
                        "regional_perception": record["final_result"].get("regional_perception"),
                        "driving_suggestions": record["final_result"].get("driving_suggestions"),
                        "final_qa": record["final_result"]["qa_report"],
                        "final_triplets": record["final_result"]["triplets"],
                        "final_top_k_candidates": record["final_result"]["top_k_candidates"],
                        "recommended_action": record["final_result"]["recommended_action"],
                        "reflection_summary": (record.get("reflection_result") or {}).get("reflection_summary"),
                    }
                    for record in records
                ],
                "skill_result": skill_payload,
            }
            typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))

        asyncio.run(_run_with_image())
        return

    asyncio.run(_run(limit))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
