"""CLI for CODA-LM evaluation."""

from __future__ import annotations

import asyncio

import typer

from ad_cornercase.bootstrap import build_judge_provider, build_prompt_renderer, load_settings
from ad_cornercase.evaluation.coda_lm_runner import CodaEvaluationRunner
from ad_cornercase.evaluation.judge_runner import JudgeRunner
from ad_cornercase.logging import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=False)


async def _run(run_id: str) -> None:
    runtime_settings, project_settings = load_settings()
    configure_logging(runtime_settings.log_level)
    run_dir = runtime_settings.artifacts_dir / run_id
    runner = CodaEvaluationRunner(
        judge_runner=JudgeRunner(
            judge_provider=build_judge_provider(runtime_settings),
            prompt_renderer=build_prompt_renderer(runtime_settings),
        ),
        project_settings=project_settings,
    )
    report_path = await runner.evaluate_run(run_dir)
    typer.echo(str(report_path))


@app.command()
def run(run_id: str = typer.Option(..., help="Existing run directory name under data/artifacts.")) -> None:
    asyncio.run(_run(run_id))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
