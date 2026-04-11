#!/usr/bin/env python3
"""Summarize one or more real/category_1 runs for paper-style reporting."""

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

from ad_cornercase.evaluation.metrics import exact_match  # noqa: E402
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
    return list(records_by_case.values())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary(run_ids: list[str]) -> dict[str, object]:
    records = _load_records(run_ids)
    records.sort(key=lambda record: record.case_id)

    per_case_rows: list[dict[str, object]] = []
    group_rows: dict[str, dict[str, object]] = {}
    bin_rows: dict[str, dict[str, object]] = {}
    group_latencies: dict[str, list[float]] = defaultdict(list)
    group_exact: dict[str, list[float]] = defaultdict(list)
    group_judge: dict[str, list[float]] = defaultdict(list)
    judged_group_judge: dict[str, list[float]] = defaultdict(list)
    bin_exact: dict[str, list[float]] = defaultdict(list)

    judged_scores: list[float] = []
    latencies: list[float] = []
    exact_values: list[float] = []

    for record in records:
        group = str(record.metadata.get("distance_group") or "unknown")
        distance_bin = str(record.metadata.get("distance_bin") or "unknown")
        pred = record.final_result.qa_report[0].answer if record.final_result.qa_report else ""
        exact_value = exact_match(record)
        latency = float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0))
        judge_score = record.judge_score

        latencies.append(latency)
        exact_values.append(exact_value)
        group_latencies[group].append(latency)
        group_exact[group].append(exact_value)
        group_judge[group].append(judge_score or 0.0)
        bin_exact[distance_bin].append(exact_value)
        if judge_score is not None:
            judged_scores.append(judge_score)
            judged_group_judge[group].append(judge_score)

        per_case_rows.append(
            {
                "case_id": record.case_id,
                "distance_group": group,
                "distance_bin": distance_bin,
                "distance_meters": record.metadata.get("distance_meters"),
                "ground_truth_answer": record.ground_truth_answer,
                "prediction": pred,
                "exact_match": exact_value,
                "judge_score": judge_score,
                "latency_ms": latency,
                "execution_mode": record.metadata.get("execution_mode", "unknown"),
            }
        )

    for group in sorted(group_exact):
        group_rows[group] = {
            "count": len(group_exact[group]),
            "exact_match_accuracy": _mean(group_exact[group]),
            "latency_mean_ms": _mean(group_latencies[group]),
            "judge_score_mean_all_records": _mean(group_judge[group]),
            "judge_score_mean_judged_only": _mean(judged_group_judge[group]),
        }

    for distance_bin in sorted(bin_exact):
        bin_rows[distance_bin] = {
            "count": len(bin_exact[distance_bin]),
            "exact_match_accuracy": _mean(bin_exact[distance_bin]),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_ids": run_ids,
        "total_cases": len(records),
        "judged_cases": len(judged_scores),
        "exact_match_accuracy": _mean(exact_values),
        "judge_score_mean_judged_only": _mean(judged_scores),
        "latency_mean_ms": _mean(latencies),
        "distance_group_summary": group_rows,
        "distance_bin_summary": bin_rows,
        "per_case_rows": per_case_rows,
    }


def write_outputs(summary: dict[str, object], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_case_rows = summary["per_case_rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0].keys()) if per_case_rows else [])
        if per_case_rows:
            writer.writeheader()
            writer.writerows(per_case_rows)

    lines = [
        "# Real Category 1 Summary",
        "",
        f"- Run IDs: {', '.join(summary['run_ids'])}",
        f"- Total cases: {summary['total_cases']}",
        f"- Judged cases: {summary['judged_cases']}",
        f"- Exact match accuracy: {summary['exact_match_accuracy']:.4f}",
        f"- Judge mean (judged only): {summary['judge_score_mean_judged_only']:.2f}",
        f"- Mean latency ms: {summary['latency_mean_ms']:.2f}",
        "",
        "## Distance Group Summary",
        "",
    ]
    for group, row in summary["distance_group_summary"].items():
        lines.append(
            f"- {group}: count={row['count']}, accuracy={row['exact_match_accuracy']:.4f}, "
            f"latency_ms={row['latency_mean_ms']:.2f}, "
            f"judge_mean_judged_only={row['judge_score_mean_judged_only']:.2f}"
        )
    lines.extend(["", "## Distance Bin Summary", ""])
    for distance_bin, row in summary["distance_bin_summary"].items():
        lines.append(f"- {distance_bin}: count={row['count']}, accuracy={row['exact_match_accuracy']:.4f}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "experiments" / "dtpqa-integration" / "results" / "real_cat1_combined_summary",
    )
    args = parser.parse_args()

    summary = build_summary(args.run_id)
    write_outputs(summary, args.output_prefix)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
