"""CLI for CODA-LM replay."""

from __future__ import annotations

import asyncio

import typer

from ad_cornercase.bootstrap import build_edge_agent, load_settings
from ad_cornercase.datasets.coda_lm import CodaLMDatasetLoader
from ad_cornercase.edge.replay import ReplayOrchestrator
from ad_cornercase.logging import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=False)


async def _run(split: str, task: str, limit: int | None, server_url: str | None) -> None:
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)
    dataset = CodaLMDatasetLoader(runtime_settings.coda_lm_root)
    cases = dataset.load(split=split, task=task, limit=limit)
    orchestrator = ReplayOrchestrator(
        edge_agent=build_edge_agent(runtime_settings, project_settings),
        runtime_settings=runtime_settings,
        project_settings=project_settings,
    )
    run_dir = await orchestrator.run(
        cases=cases,
        server_url=server_url or str(runtime_settings.mcp_server_url),
    )
    typer.echo(str(run_dir))


@app.command()
def run(
    split: str = typer.Option("Mini"),
    task: str = typer.Option("region_perception"),
    limit: int | None = typer.Option(None),
    server_url: str | None = typer.Option(None),
) -> None:
    asyncio.run(_run(split, task, limit, server_url))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
