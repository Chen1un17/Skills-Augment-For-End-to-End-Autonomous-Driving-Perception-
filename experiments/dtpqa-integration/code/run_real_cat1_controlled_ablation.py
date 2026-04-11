#!/usr/bin/env python3
"""Run a same-day OFF/ON DTPQA category_1 ablation and summarize the paired result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def _plan_command(
    *,
    plan: Path,
    run_id: str,
    execution_mode: str,
    max_passes: int,
    retry_sleep_seconds: float,
    request_timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "experiments" / "dtpqa-integration" / "code" / "run_real_cat1_plan.py"),
        "--plan",
        str(plan),
        "--run-id",
        run_id,
        "--execution-mode",
        execution_mode,
        "--sleep-seconds",
        str(sleep_seconds),
        "--max-passes",
        str(max_passes),
        "--retry-sleep-seconds",
        str(retry_sleep_seconds),
        "--request-timeout-seconds",
        str(request_timeout_seconds),
        "--max-retries",
        str(max_retries),
    ]


def _compare_command(*, baseline_run_id: str, intervention_run_id: str, output_prefix: Path) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "experiments" / "dtpqa-integration" / "code" / "compare_real_cat1_runs.py"),
        "--baseline-run-id",
        baseline_run_id,
        "--intervention-run-id",
        intervention_run_id,
        "--output-prefix",
        str(output_prefix),
    ]


def _run_checked(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--off-run-id", required=True)
    parser.add_argument("--on-run-id", required=True)
    parser.add_argument("--compare-output-prefix", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--skill-store-dir", required=True)
    parser.add_argument("--edge-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cloud-model", default="Pro/moonshotai/Kimi-K2.5")
    parser.add_argument("--judge-model", default="Pro/moonshotai/Kimi-K2.5")
    parser.add_argument("--off-execution-mode", default="edge_only")
    parser.add_argument("--on-execution-mode", default="hybrid")
    parser.add_argument("--edge-max-completion-tokens", type=int, default=512)
    parser.add_argument("--plan-max-passes", type=int, default=1)
    parser.add_argument("--plan-retry-sleep-seconds", type=float, default=15.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--skip-off", action="store_true")
    parser.add_argument("--skip-on", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    base_env = os.environ.copy()
    base_env.update(
        {
            "MCP_SERVER_URL": args.server_url,
            "SKILL_STORE_DIR": args.skill_store_dir,
            "EDGE_MODEL": args.edge_model,
            "CLOUD_MODEL": args.cloud_model,
            "JUDGE_MODEL": args.judge_model,
            "EDGE_MAX_COMPLETION_TOKENS": str(args.edge_max_completion_tokens),
            "REQUEST_TIMEOUT_SECONDS": str(args.request_timeout_seconds),
            "MAX_RETRIES": str(args.max_retries),
        }
    )

    if not args.skip_off:
        off_env = dict(base_env)
        off_env["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] = "0"
        _run_checked(
            _plan_command(
                plan=args.plan,
                run_id=args.off_run_id,
                execution_mode=args.off_execution_mode,
                max_passes=args.plan_max_passes,
                retry_sleep_seconds=args.plan_retry_sleep_seconds,
                request_timeout_seconds=args.request_timeout_seconds,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            ),
            off_env,
        )

    if not args.skip_on:
        on_env = dict(base_env)
        on_env["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] = "1"
        _run_checked(
            _plan_command(
                plan=args.plan,
                run_id=args.on_run_id,
                execution_mode=args.on_execution_mode,
                max_passes=args.plan_max_passes,
                retry_sleep_seconds=args.plan_retry_sleep_seconds,
                request_timeout_seconds=args.request_timeout_seconds,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            ),
            on_env,
        )

    if not args.skip_compare:
        _run_checked(
            _compare_command(
                baseline_run_id=args.off_run_id,
                intervention_run_id=args.on_run_id,
                output_prefix=args.compare_output_prefix,
            ),
            base_env,
        )


if __name__ == "__main__":
    main()
