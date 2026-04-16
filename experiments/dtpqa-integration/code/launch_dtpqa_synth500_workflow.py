#!/usr/bin/env python3
"""Launch the synth-500 DTPQA workflow in staged steps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "dtpqa-integration" / "results"
PLAN_PATH = RESULTS_DIR / "dtpqa_synth500_balanced_plan.json"
QUESTION_TYPES = ("category_1", "category_2", "category_3", "category_4", "category_5", "category_6")


def _run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _binary(name: str) -> str:
    path = ROOT / ".venv" / "bin" / name
    return str(path) if path.exists() else name


def _run_plan(
    run_id: str,
    selection: str,
    execution_mode: str,
    *,
    question_type: str | None = None,
    skill_store_dir: Path | None = None,
    disable_category1_direct_cloud: bool = False,
    max_retries: int = 1,
    request_timeout_seconds: float = 300.0,
) -> None:
    command = [
        _binary("python"),
        str(ROOT / "experiments" / "dtpqa-integration" / "code" / "run_dtpqa_plan.py"),
        "--plan",
        str(PLAN_PATH),
        "--run-id",
        run_id,
        "--selection",
        selection,
        "--execution-mode",
        execution_mode,
        "--max-retries",
        str(max_retries),
        "--request-timeout-seconds",
        str(request_timeout_seconds),
    ]
    if execution_mode == "hybrid":
        command.append("--start-fresh-server")
    if question_type:
        command.extend(["--question-type", question_type])
    if skill_store_dir is not None:
        command.extend(["--skill-store-dir", str(skill_store_dir)])
    if disable_category1_direct_cloud:
        command.append("--disable-category1-direct-cloud")
    _run(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        choices=(
            "plan",
            "smoke",
            "smoke_report",
            "full",
            "full_report",
            "refinement",
            "refinement_report",
        ),
        required=True,
    )
    parser.add_argument("--tag", default="manual")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "plan" in args.stage:
        _run(
            [
                _binary("python"),
                str(ROOT / "experiments" / "dtpqa-integration" / "code" / "build_dtpqa_synth_balanced_plan.py"),
                "--output",
                str(PLAN_PATH),
            ]
        )

    smoke_edge = f"dtpqa_synth18_{args.tag}_edge_only"
    smoke_cloud = f"dtpqa_synth18_{args.tag}_cloud_only"
    smoke_hybrid = f"dtpqa_synth18_{args.tag}_hybrid"
    full_edge = f"dtpqa_synth500_{args.tag}_edge_only"
    full_cloud = f"dtpqa_synth500_{args.tag}_cloud_only"
    full_hybrid = f"dtpqa_synth500_{args.tag}_hybrid"

    if "smoke" in args.stage:
        _run_plan(
            smoke_edge,
            "smoke",
            "edge_only",
            skill_store_dir=Path("/tmp/dtpqa_synth18_edge_skills"),
            max_retries=args.max_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        _run_plan(
            smoke_cloud,
            "smoke",
            "cloud_only",
            skill_store_dir=Path("/tmp/dtpqa_synth18_cloud_skills"),
            max_retries=args.max_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        _run_plan(
            smoke_hybrid,
            "smoke",
            "hybrid",
            skill_store_dir=Path("/tmp/dtpqa_synth18_hybrid_skills"),
            max_retries=args.max_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )

    if "smoke_report" in args.stage:
        _run(
            [
                _binary("python"),
                str(ROOT / "experiments" / "dtpqa-integration" / "code" / "build_dtpqa_synth500_three_way_report.py"),
                "--plan",
                str(PLAN_PATH),
                "--edge-run-id",
                smoke_edge,
                "--cloud-run-id",
                smoke_cloud,
                "--hybrid-run-id",
                smoke_hybrid,
                "--output-prefix",
                str(RESULTS_DIR / f"dtpqa_synth18_three_way_{args.tag}"),
                "--accuracy-plot",
                str(RESULTS_DIR / "dtpqa_synth500_accuracy_by_category.png"),
                "--latency-plot",
                str(RESULTS_DIR / "dtpqa_synth500_latency_by_category.png"),
            ]
        )

    if "full" in args.stage:
        _run_plan(
            full_edge,
            "full",
            "edge_only",
            skill_store_dir=Path("/tmp/dtpqa_synth500_edge_skills"),
            max_retries=args.max_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        _run_plan(
            full_cloud,
            "full",
            "cloud_only",
            skill_store_dir=Path("/tmp/dtpqa_synth500_cloud_skills"),
            max_retries=args.max_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )
        _run_plan(
            full_hybrid,
            "full",
            "hybrid",
            skill_store_dir=Path("/tmp/dtpqa_synth500_hybrid_skills"),
            max_retries=args.max_retries,
            request_timeout_seconds=args.request_timeout_seconds,
        )

    if "full_report" in args.stage:
        _run(
            [
                _binary("python"),
                str(ROOT / "experiments" / "dtpqa-integration" / "code" / "build_dtpqa_synth500_three_way_report.py"),
                "--plan",
                str(PLAN_PATH),
                "--edge-run-id",
                full_edge,
                "--cloud-run-id",
                full_cloud,
                "--hybrid-run-id",
                full_hybrid,
                "--output-prefix",
                str(RESULTS_DIR / f"dtpqa_synth500_three_way_{args.tag}"),
                "--accuracy-plot",
                str(RESULTS_DIR / "dtpqa_synth500_accuracy_by_category.png"),
                "--latency-plot",
                str(RESULTS_DIR / "dtpqa_synth500_latency_by_category.png"),
            ]
        )

    refinement_manifest_path = RESULTS_DIR / f"dtpqa_synth500_refinement_manifest_{args.tag}.json"
    if "refinement" in args.stage:
        manifest = {"generated_at": None, "question_types": {}}
        for question_type in QUESTION_TYPES:
            run_id = f"dtpqa_synth500_refine_{question_type}_{args.tag}"
            skill_store_dir = Path(f"/tmp/dtpqa_synth500_refine_{question_type}_{args.tag}")
            _run_plan(
                run_id,
                "refinement_adaptation",
                "hybrid",
                question_type=question_type,
                skill_store_dir=skill_store_dir,
                disable_category1_direct_cloud=True,
                max_retries=args.max_retries,
                request_timeout_seconds=args.request_timeout_seconds,
            )
            _run_plan(
                run_id,
                "refinement_holdout",
                "hybrid",
                question_type=question_type,
                skill_store_dir=skill_store_dir,
                disable_category1_direct_cloud=True,
                max_retries=args.max_retries,
                request_timeout_seconds=args.request_timeout_seconds,
            )
            manifest["question_types"][question_type] = {
                "run_id": run_id,
                "skill_store_dir": str(skill_store_dir),
            }
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        refinement_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if "refinement_report" in args.stage:
        _run(
            [
                _binary("python"),
                str(ROOT / "experiments" / "dtpqa-integration" / "code" / "build_dtpqa_skill_refinement_report.py"),
                "--plan",
                str(PLAN_PATH),
                "--manifest",
                str(refinement_manifest_path),
                "--output-prefix",
                str(RESULTS_DIR / f"dtpqa_synth500_skill_refinement_{args.tag}"),
                "--dashboard-plot",
                str(RESULTS_DIR / "dtpqa_synth500_refinement_dashboard.png"),
            ]
        )


if __name__ == "__main__":
    main()
