#!/usr/bin/env python3
"""Verify clean-run integrity for DTPQA real/category_1 experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.evaluation.metrics import canonicalize_answer, exact_match  # noqa: E402
from ad_cornercase.schemas.evaluation import CasePredictionRecord  # noqa: E402


def _load_records(run_ids: list[str]) -> list[CasePredictionRecord]:
    records_by_case: dict[str, CasePredictionRecord] = {}
    for run_id in run_ids:
        predictions_path = ROOT / "data" / "artifacts" / run_id / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = CasePredictionRecord.model_validate_json(line)
                records_by_case[record.case_id] = record
    return sorted(records_by_case.values(), key=lambda record: record.case_id)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _prediction_text(record: CasePredictionRecord) -> str:
    if record.final_result.qa_report:
        return record.final_result.qa_report[0].answer
    return ""


def _is_positive_case(record: CasePredictionRecord) -> bool:
    options = record.metadata.get("answer_options")
    normalized = canonicalize_answer(record.ground_truth_answer, options if isinstance(options, list) else None)
    return normalized == "yes"


def _count_skill_store_files(skill_store_dir: Path | None) -> int | None:
    if skill_store_dir is None:
        return None
    if not skill_store_dir.exists():
        return 0
    return sum(1 for path in skill_store_dir.rglob("*") if path.is_file())


def build_summary(run_ids: list[str], skill_store_dir: Path | None = None) -> dict[str, object]:
    records = _load_records(run_ids)
    exact_values = [exact_match(record) for record in records]
    latencies = [float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0)) for record in records]
    matched_skill_case_ids = [record.case_id for record in records if record.matched_skill_ids]
    reflection_case_ids = [record.case_id for record in records if record.reflection_result is not None]
    positive_records = [record for record in records if _is_positive_case(record)]
    negative_records = [record for record in records if not _is_positive_case(record)]
    distance_group_counts: dict[str, int] = defaultdict(int)
    per_case_rows: list[dict[str, object]] = []

    for record in records:
        group = str(record.metadata.get("distance_group") or "unknown")
        distance_group_counts[group] += 1
        per_case_rows.append(
            {
                "case_id": record.case_id,
                "distance_group": group,
                "ground_truth_answer": record.ground_truth_answer,
                "prediction": _prediction_text(record),
                "exact_match": exact_match(record),
                "latency_ms": float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0)),
                "execution_mode": record.metadata.get("execution_mode", "unknown"),
                "matched_skill_count": len(record.matched_skill_ids),
                "reflection_used": record.reflection_result is not None,
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_ids": run_ids,
        "total_cases": len(records),
        "exact_match_accuracy": _mean(exact_values),
        "positive_case_count": len(positive_records),
        "negative_case_count": len(negative_records),
        "positive_recall": _mean([exact_match(record) for record in positive_records]),
        "negative_specificity": _mean([exact_match(record) for record in negative_records]),
        "latency_mean_ms": _mean(latencies),
        "matched_skill_case_count": len(matched_skill_case_ids),
        "matched_skill_case_ids": matched_skill_case_ids,
        "reflection_case_count": len(reflection_case_ids),
        "reflection_case_ids": reflection_case_ids,
        "all_matched_skill_ids_empty": not matched_skill_case_ids,
        "all_reflection_results_null": not reflection_case_ids,
        "distance_group_counts": dict(sorted(distance_group_counts.items())),
        "execution_modes": sorted({str(record.metadata.get("execution_mode") or "unknown") for record in records}),
        "skill_store_dir": str(skill_store_dir) if skill_store_dir is not None else None,
        "skill_store_file_count": _count_skill_store_files(skill_store_dir),
        "per_case_rows": per_case_rows,
    }
    summary["skill_store_is_empty"] = summary["skill_store_file_count"] in {None, 0}
    return summary


def evaluate_expectations(
    summary: dict[str, object],
    *,
    expect_no_matched_skills: bool,
    expect_no_reflection: bool,
    expect_empty_skill_store: bool,
) -> list[str]:
    failures: list[str] = []
    if expect_no_matched_skills and not summary["all_matched_skill_ids_empty"]:
        failures.append("matched_skill_ids present")
    if expect_no_reflection and not summary["all_reflection_results_null"]:
        failures.append("reflection_result present")
    if expect_empty_skill_store and not summary["skill_store_is_empty"]:
        failures.append("skill store not empty")
    return failures


def write_outputs(summary: dict[str, object], failures: list[str], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    payload = dict(summary)
    payload["expectation_failures"] = failures
    payload["passed"] = not failures
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    per_case_rows = summary["per_case_rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0].keys()) if per_case_rows else [])
        if per_case_rows:
            writer.writeheader()
            writer.writerows(per_case_rows)

    lines = [
        "# Real Category 1 Run Integrity",
        "",
        f"- Run IDs: {', '.join(summary['run_ids'])}",
        f"- Passed: {not failures}",
        f"- Exact match accuracy: {summary['exact_match_accuracy']:.4f}",
        f"- Positive recall: {summary['positive_recall']:.4f} ({summary['positive_case_count']} positives)",
        f"- Negative specificity: {summary['negative_specificity']:.4f} ({summary['negative_case_count']} negatives)",
        f"- Mean latency ms: {summary['latency_mean_ms']:.2f}",
        f"- Matched-skill cases: {summary['matched_skill_case_count']}",
        f"- Reflection cases: {summary['reflection_case_count']}",
        f"- Skill store dir: {summary['skill_store_dir'] or 'N/A'}",
        f"- Skill store file count: {summary['skill_store_file_count']}",
        "",
        "## Expectation Failures",
        "",
    ]
    if failures:
        for failure in failures:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")

    if summary["matched_skill_case_ids"]:
        lines.extend(["", "## Matched Skill Cases", ""])
        for case_id in summary["matched_skill_case_ids"]:
            lines.append(f"- {case_id}")

    if summary["reflection_case_ids"]:
        lines.extend(["", "## Reflection Cases", ""])
        for case_id in summary["reflection_case_ids"]:
            lines.append(f"- {case_id}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--skill-store-dir", type=Path, default=None)
    parser.add_argument("--expect-no-matched-skills", action="store_true")
    parser.add_argument("--expect-no-reflection", action="store_true")
    parser.add_argument("--expect-empty-skill-store", action="store_true")
    args = parser.parse_args()

    summary = build_summary(args.run_id, args.skill_store_dir)
    failures = evaluate_expectations(
        summary,
        expect_no_matched_skills=args.expect_no_matched_skills,
        expect_no_reflection=args.expect_no_reflection,
        expect_empty_skill_store=args.expect_empty_skill_store,
    )
    write_outputs(summary, failures, args.output_prefix)
    if failures:
        raise SystemExit(1)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
