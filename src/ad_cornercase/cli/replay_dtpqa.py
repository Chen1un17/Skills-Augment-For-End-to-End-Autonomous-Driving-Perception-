"""CLI for DTPQA replay."""

from __future__ import annotations

import asyncio

import typer

from ad_cornercase.bootstrap import build_edge_agent, load_settings
from ad_cornercase.datasets.dtpqa import DTPQADatasetLoader
from ad_cornercase.edge.replay import ReplayOrchestrator
from ad_cornercase.logging import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=False)


async def _run(
    subset: str,
    question_type: str | None,
    limit: int | None,
    offset: int,
    annotation_glob: str | None,
    server_url: str | None,
    run_id: str | None,
    append: bool,
    execution_mode: str,
) -> None:
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)
    dataset = DTPQADatasetLoader(runtime_settings.dtpqa_root)
    cases = dataset.load(
        subset=subset,
        question_type=question_type,
        limit=limit,
        offset=offset,
        annotation_glob=annotation_glob,
    )
    orchestrator = ReplayOrchestrator(
        edge_agent=build_edge_agent(runtime_settings, project_settings),
        cloud_perception_agent=build_edge_agent(
            runtime_settings,
            project_settings,
            model_override=runtime_settings.cloud_model,
        ),
        runtime_settings=runtime_settings,
        project_settings=project_settings,
    )
    run_dir = await orchestrator.run(
        cases=cases,
        server_url=(server_url or str(runtime_settings.mcp_server_url)) if execution_mode == "hybrid" else None,
        run_id=run_id,
        append=append,
        execution_mode=execution_mode,
    )
    typer.echo(str(run_dir))


@app.command()
def run(
    subset: str = typer.Option("all", help="Subset name, e.g. all, real, synth."),
    question_type: str | None = typer.Option(None, help="Optional DTPQA question/task type filter."),
    limit: int | None = typer.Option(None, help="Maximum number of cases to replay."),
    offset: int = typer.Option(0, min=0, help="Number of filtered cases to skip before replaying."),
    annotation_glob: str | None = typer.Option(
        None,
        help="Optional glob relative to DTPQA_ROOT for narrowing annotation discovery.",
    ),
    server_url: str | None = typer.Option(None),
    run_id: str | None = typer.Option(None, help="Optional existing or fixed run directory name under data/artifacts."),
    append: bool = typer.Option(
        False,
        help="Append to an existing run directory and skip already-recorded case_ids instead of creating a fresh run.",
    ),
    execution_mode: str = typer.Option(
        "hybrid",
        help="Execution mode: edge_only, cloud_only, or hybrid.",
    ),
) -> None:
    asyncio.run(_run(subset, question_type, limit, offset, annotation_glob, server_url, run_id, append, execution_mode))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
