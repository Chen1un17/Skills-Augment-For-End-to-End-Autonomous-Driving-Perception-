#!/usr/bin/env python3
"""Build a shadow skill-refinement report from category-isolated runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.evaluation.metrics import exact_match  # noqa: E402
from ad_cornercase.schemas.evaluation import CasePredictionRecord  # noqa: E402


def _load_records(run_id: str) -> dict[str, CasePredictionRecord]:
    predictions_path = ROOT / "data" / "artifacts" / run_id / "predictions.jsonl"
    if not predictions_path.exists():
        return {}
    records_by_case: dict[str, CasePredictionRecord] = {}
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = CasePredictionRecord.model_validate_json(line)
            records_by_case[record.case_id] = record
    return records_by_case


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _baseline_exact_match(record: CasePredictionRecord) -> float:
    if record.baseline_result.qa_report and record.final_result.qa_report:
        baseline_answer = record.baseline_result.qa_report[0].answer.strip().lower()
        final_answer = record.final_result.qa_report[0].answer.strip().lower()
        ground_truth = record.ground_truth_answer.strip().lower()
        return 1.0 if baseline_answer.startswith(ground_truth) else 0.0
    return 0.0


def _subset_summary(records: list[CasePredictionRecord]) -> dict[str, object]:
    skill_records = [record for record in records if record.matched_skill_ids]
    reflection_records = [record for record in records if record.reflection_result is not None]
    judged_relevant = [record for record in records if (record.matched_skill_ids or record.reflection_result) and record.judge_score is not None]
    return {
        "case_count": len(records),
        "exact_match_accuracy": _mean([exact_match(record) for record in records]),
        "skill_match_rate": len(skill_records) / len(records) if records else 0.0,
        "unique_matched_skill_count": len({skill_id for record in records for skill_id in record.matched_skill_ids}),
        "skill_precision": (
            sum(_baseline_exact_match(record) < exact_match(record) for record in skill_records) / len(skill_records)
            if skill_records
            else 0.0
        ),
        "skill_success_rate": (
            sum((record.judge_score or 0.0) >= 70.0 for record in judged_relevant) / len(judged_relevant)
            if judged_relevant
            else None
        ),
        "reflection_precision": (
            sum(_baseline_exact_match(record) < exact_match(record) for record in reflection_records) / len(reflection_records)
            if reflection_records
            else 0.0
        ),
        "rescue_count": sum(_baseline_exact_match(record) < exact_match(record) for record in records),
        "harm_count": sum(_baseline_exact_match(record) > exact_match(record) for record in records),
        "latency_delta_ms": _mean([
            float((record.baseline_result.latency_ms or 0.0) - (record.final_result.latency_ms or 0.0))
            for record in records
        ]),
        "vision_token_delta": _mean([
            float((record.baseline_result.vision_tokens or 0.0) - (record.final_result.vision_tokens or 0.0))
            for record in records
        ]),
    }


def build_summary(*, plan_path: Path, manifest_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refinement_splits = plan.get("refinement_splits", {})
    question_type_manifest = manifest.get("question_types", {})

    adaptation_summary: dict[str, object] = {}
    holdout_summary: dict[str, object] = {}
    question_type_refinement_summary: dict[str, object] = {}
    skill_store_summary: dict[str, object] = {}
    combined_adaptation_records: list[CasePredictionRecord] = []
    combined_holdout_records: list[CasePredictionRecord] = []

    for question_type, split_payload in sorted(refinement_splits.items()):
        if not isinstance(split_payload, dict):
            continue
        manifest_payload = question_type_manifest.get(question_type, {})
        run_id = manifest_payload.get("run_id")
        if not isinstance(run_id, str):
            continue
        records_by_case = _load_records(run_id)
        adaptation_case_ids = set(split_payload.get("adaptation_case_ids", []))
        holdout_case_ids = set(split_payload.get("holdout_case_ids", []))
        adaptation_records = [records_by_case[case_id] for case_id in adaptation_case_ids if case_id in records_by_case]
        holdout_records = [records_by_case[case_id] for case_id in holdout_case_ids if case_id in records_by_case]
        combined_adaptation_records.extend(adaptation_records)
        combined_holdout_records.extend(holdout_records)

        question_type_refinement_summary[question_type] = {
            "run_id": run_id,
            "adaptation": _subset_summary(adaptation_records),
            "holdout": _subset_summary(holdout_records),
        }

        skill_store_dir = manifest_payload.get("skill_store_dir")
        file_count = 0
        stored_skill_ids: list[str] = []
        if isinstance(skill_store_dir, str):
            skill_path = Path(skill_store_dir)
            if skill_path.exists():
                stored_skill_ids = sorted(path.name for path in skill_path.iterdir())
                file_count = len(stored_skill_ids)
        skill_store_summary[question_type] = {
            "skill_store_dir": skill_store_dir,
            "file_count": file_count,
            "stored_skill_ids": stored_skill_ids,
        }

    adaptation_summary = _subset_summary(combined_adaptation_records)
    holdout_summary = _subset_summary(combined_holdout_records)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "manifest_path": str(manifest_path),
        "adaptation_summary": adaptation_summary,
        "holdout_summary": holdout_summary,
        "question_type_refinement_summary": question_type_refinement_summary,
        "skill_store_summary": skill_store_summary,
    }


def _write_json(summary: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(summary: dict[str, object], path: Path) -> None:
    rows = []
    for question_type, payload in summary["question_type_refinement_summary"].items():
        for split_name in ("adaptation", "holdout"):
            split = payload[split_name]
            rows.append(
                {
                    "question_type": question_type,
                    "split": split_name,
                    "run_id": payload["run_id"],
                    **split,
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# DTPQA Skill Refinement Shadow Report",
        "",
        "## Overall",
        "",
        f"- Adaptation exact match: {summary['adaptation_summary']['exact_match_accuracy']:.4f}",
        f"- Holdout exact match: {summary['holdout_summary']['exact_match_accuracy']:.4f}",
        f"- Holdout skill match rate: {summary['holdout_summary']['skill_match_rate']:.4f}",
        f"- Holdout skill precision: {summary['holdout_summary']['skill_precision']:.4f}",
        f"- Holdout reflection precision: {summary['holdout_summary']['reflection_precision']:.4f}",
        "",
        "## Per Question Type",
        "",
        "| Question Type | Run ID | Adaptation Acc | Holdout Acc | Holdout Skill Match | Holdout Skill Precision | Holdout Reflection Precision |",
        "|---------------|--------|----------------|-------------|---------------------|-------------------------|------------------------------|",
    ]
    for question_type, payload in summary["question_type_refinement_summary"].items():
        holdout = payload["holdout"]
        adaptation = payload["adaptation"]
        lines.append(
            f"| {question_type} | {payload['run_id']} | {adaptation['exact_match_accuracy']:.4f} | "
            f"{holdout['exact_match_accuracy']:.4f} | {holdout['skill_match_rate']:.4f} | "
            f"{holdout['skill_precision']:.4f} | {holdout['reflection_precision']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(summary: dict[str, object], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    question_types = sorted(summary["question_type_refinement_summary"])
    x = np.arange(len(question_types))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    holdout_skill_match = [
        summary["question_type_refinement_summary"][question_type]["holdout"]["skill_match_rate"]
        for question_type in question_types
    ]
    holdout_skill_precision = [
        summary["question_type_refinement_summary"][question_type]["holdout"]["skill_precision"]
        for question_type in question_types
    ]
    holdout_reflection_precision = [
        summary["question_type_refinement_summary"][question_type]["holdout"]["reflection_precision"]
        for question_type in question_types
    ]
    unique_skills = [
        summary["skill_store_summary"][question_type]["file_count"]
        for question_type in question_types
    ]

    axes[0, 0].bar(question_types, holdout_skill_match, color="#1f77b4")
    axes[0, 0].set_title("Holdout Skill Match Rate")
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].bar(x - width / 2, holdout_skill_precision, width, label="skill_precision", color="#2ca02c")
    axes[0, 1].bar(x + width / 2, holdout_reflection_precision, width, label="reflection_precision", color="#ff7f0e")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(question_types)
    axes[0, 1].set_title("Holdout Precision")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    rescue_counts = [
        summary["question_type_refinement_summary"][question_type]["holdout"]["rescue_count"]
        for question_type in question_types
    ]
    harm_counts = [
        summary["question_type_refinement_summary"][question_type]["holdout"]["harm_count"]
        for question_type in question_types
    ]
    axes[1, 0].bar(x - width / 2, rescue_counts, width, label="rescue_count", color="#9467bd")
    axes[1, 0].bar(x + width / 2, harm_counts, width, label="harm_count", color="#d62728")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(question_types)
    axes[1, 0].set_title("Holdout Rescue vs Harm")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(question_types, unique_skills, color="#8c564b")
    axes[1, 1].set_title("Persisted Skill Store Size")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--dashboard-plot", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(plan_path=args.plan, manifest_path=args.manifest)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_json(summary, args.output_prefix.with_suffix(".json"))
    _write_csv(summary, args.output_prefix.with_suffix(".csv"))
    _write_markdown(summary, args.output_prefix.with_suffix(".md"))
    _write_plot(summary, args.dashboard_plot)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
