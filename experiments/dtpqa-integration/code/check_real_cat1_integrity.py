#!/usr/bin/env python3
"""Validate DTPQA real/category_1 plan coverage and paired run integrity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR = ROOT / "data" / "artifacts"

from ad_cornercase.evaluation.metrics import exact_match
from ad_cornercase.schemas.evaluation import CasePredictionRecord


def _load_plan(plan_path: Path) -> Mapping[str, dict[str, Any]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    selections = payload.get("selections", {}) or {}
    cases: dict[str, dict[str, Any]] = {}
    for group_entries in selections.values():
        if not isinstance(group_entries, list):
            continue
        for entry in group_entries:
            case_id = str(entry.get("case_id"))
            if not case_id:
                continue
            cases.setdefault(case_id, entry)
    return cases


def _load_predictions(run_id: str, artifacts_dir: Path) -> dict[str, CasePredictionRecord]:
    path = artifacts_dir / run_id / "predictions.jsonl"
    if not path.exists():
        return {}
    records: dict[str, CasePredictionRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = CasePredictionRecord.model_validate_json(line)
            records[record.case_id] = record
    return records


def _load_status(run_id: str, artifacts_dir: Path) -> list[dict[str, Any]]:
    path = artifacts_dir / run_id / "batch-status.jsonl"
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _summarize_run(run_id: str, expected_case_ids: set[str], artifacts_dir: Path) -> dict[str, Any]:
    records = _load_predictions(run_id, artifacts_dir)
    statuses = _load_status(run_id, artifacts_dir)
    status_by_case = {str(entry.get("case_id")): entry for entry in statuses if entry.get("case_id")}
    existing_case_ids = set(records)
    missing_case_ids = sorted(expected_case_ids - existing_case_ids)
    extra_case_ids = sorted(existing_case_ids - expected_case_ids)
    failed_cases = []
    for case_id, entry in status_by_case.items():
        returncode = entry.get("returncode")
        if isinstance(returncode, int) and returncode != 0:
            failed_cases.append({"case_id": case_id, "returncode": returncode})
    contamination_cases = []
    reflection_cases = []
    exact_by_case: dict[str, float] = {}
    exact_values: list[float] = []
    for case_id, record in records.items():
        value = exact_match(record)
        exact_by_case[case_id] = value
        exact_values.append(value)
        if record.matched_skill_ids:
            contamination_cases.append({"case_id": case_id, "matched_skill_ids": record.matched_skill_ids})
        if record.reflection_result is not None:
            reflection_cases.append(case_id)
    total_cases = len(records)
    accuracy = sum(exact_values) / total_cases if total_cases else 0.0
    status_case_ids = set(status_by_case)
    missing_status_case_ids = sorted(expected_case_ids - status_case_ids)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "plan_case_count": len(expected_case_ids),
        "observed_case_count": total_cases,
        "missing_case_ids": missing_case_ids,
        "extra_case_ids": extra_case_ids,
        "failed_cases": failed_cases,
        "contamination_cases": contamination_cases,
        "reflection_case_ids": reflection_cases,
        "exact_match_accuracy": accuracy,
        "exact_match_by_case": exact_by_case,
        "status_returncodes": {case_id: status_by_case[case_id].get("returncode") for case_id in sorted(status_by_case)},
        "missing_status_case_ids": missing_status_case_ids,
    }
    return summary


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# DTPQA Real/Category 1 Integrity Report", ""]
    if summary.get("plan"):
        plan_info = summary["plan"]
        lines.append("## Plan")
        lines.append(f"- Cases defined: {plan_info['case_count']}")
        lines.append(f"- Case IDs: {', '.join(plan_info['case_ids'])}")
        lines.append("")
    for run_summary in summary["runs"]:
        lines.append(f"## Run {run_summary['run_id']}")
        lines.append(f"- Observed {run_summary['observed_case_count']}/{run_summary['plan_case_count']} cases")
        lines.append(f"- Exact-match accuracy: {run_summary['exact_match_accuracy']:.4f}")
        lines.append(f"- Missing cases: {', '.join(run_summary['missing_case_ids']) or 'none'}")
        lines.append(f"- Extra cases: {', '.join(run_summary['extra_case_ids']) or 'none'}")
        lines.append(f"- Failed cases: {', '.join(item['case_id'] for item in run_summary['failed_cases']) or 'none'}")
        lines.append(f"- Contamination cases: {', '.join(item['case_id'] for item in run_summary['contamination_cases']) or 'none'}")
        lines.append(f"- Reflection cases: {', '.join(run_summary['reflection_case_ids']) or 'none'}")
        lines.append("")
    paired = summary.get("paired")
    if paired:
        lines.append("## Paired Comparison")
        lines.append(f"- Shared cases: {paired['shared_case_count']}")
        lines.append(f"- Exact-match delta: {paired['exact_match_delta']:+.4f}")
        lines.append("- Shared case IDs: " + (", ".join(paired['shared_case_ids']) or "none"))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "experiments" / "dtpqa-integration" / "results" / "real_cat1_integrity")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    expected_cases = set(_load_plan(args.plan).keys())
    plan_summary = {
        "case_count": len(expected_cases),
        "case_ids": sorted(expected_cases),
    }
    runs_summary: list[dict[str, Any]] = []
    for run_id in args.run_id:
        runs_summary.append(_summarize_run(run_id, expected_cases, args.artifacts_dir))

    paired = None
    if len(runs_summary) == 2:
        base, intervention = runs_summary
        shared_ids = sorted(set(base["exact_match_by_case"]) & set(intervention["exact_match_by_case"]))
        base_scores = [base["exact_match_by_case"][case_id] for case_id in shared_ids]
        intervention_scores = [intervention["exact_match_by_case"][case_id] for case_id in shared_ids]
        base_mean = sum(base_scores) / len(base_scores) if base_scores else 0.0
        intervention_mean = sum(intervention_scores) / len(intervention_scores) if intervention_scores else 0.0
        paired = {
            "shared_case_count": len(shared_ids),
            "shared_case_ids": shared_ids,
            "exact_match_delta": intervention_mean - base_mean,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan_summary,
        "runs": runs_summary,
    }
    if paired:
        summary["paired"] = paired

    output_prefix = args.output_prefix
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_build_markdown(summary), encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
