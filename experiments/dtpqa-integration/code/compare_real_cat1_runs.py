#!/usr/bin/env python3
"""Build a paired comparison between baseline and intervention real/category_1 runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.evaluation.metrics import canonicalize_answer, exact_match  # noqa: E402
from ad_cornercase.schemas.evaluation import CasePredictionRecord  # noqa: E402


def _load_records(run_ids: list[str]) -> dict[str, CasePredictionRecord]:
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
    return records_by_case


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _prediction_text(record: CasePredictionRecord) -> str:
    if record.final_result.qa_report:
        return record.final_result.qa_report[0].answer
    return ""


def _latency_ms(record: CasePredictionRecord) -> float:
    return float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0))


def _is_positive_case(record: CasePredictionRecord) -> bool:
    options = record.metadata.get("answer_options")
    normalized = canonicalize_answer(record.ground_truth_answer, options if isinstance(options, list) else None)
    return normalized == "yes"


def _build_confusion(records: list[CasePredictionRecord]) -> dict[str, int]:
    confusion: Counter[str] = Counter()
    for record in records:
        ground_truth = record.ground_truth_answer
        prediction = _prediction_text(record)
        confusion[f"{ground_truth} -> {prediction}"] += 1
    return dict(sorted(confusion.items()))


def _conditional_accuracy(records: list[CasePredictionRecord], predicate) -> tuple[float, int]:
    filtered = [record for record in records if predicate(record)]
    if not filtered:
        return 0.0, 0
    return _mean([exact_match(record) for record in filtered]), len(filtered)


def build_summary(
    baseline_run_ids: list[str],
    intervention_run_ids: list[str],
) -> dict[str, object]:
    baseline_records = _load_records(baseline_run_ids)
    intervention_records = _load_records(intervention_run_ids)
    shared_case_ids = sorted(set(baseline_records) & set(intervention_records))

    paired_rows: list[dict[str, object]] = []
    baseline_exact_values: list[float] = []
    intervention_exact_values: list[float] = []
    baseline_judge_values: list[float] = []
    intervention_judge_values: list[float] = []
    baseline_latencies: list[float] = []
    intervention_latencies: list[float] = []
    group_baseline_exact: dict[str, list[float]] = defaultdict(list)
    group_intervention_exact: dict[str, list[float]] = defaultdict(list)
    improved_cases: list[str] = []
    regressed_cases: list[str] = []

    for case_id in shared_case_ids:
        baseline = baseline_records[case_id]
        intervention = intervention_records[case_id]
        baseline_exact = exact_match(baseline)
        intervention_exact = exact_match(intervention)
        baseline_exact_values.append(baseline_exact)
        intervention_exact_values.append(intervention_exact)
        baseline_latency = _latency_ms(baseline)
        intervention_latency = _latency_ms(intervention)
        baseline_latencies.append(baseline_latency)
        intervention_latencies.append(intervention_latency)
        if baseline.judge_score is not None:
            baseline_judge_values.append(baseline.judge_score)
        if intervention.judge_score is not None:
            intervention_judge_values.append(intervention.judge_score)

        group = str(intervention.metadata.get("distance_group") or baseline.metadata.get("distance_group") or "unknown")
        group_baseline_exact[group].append(baseline_exact)
        group_intervention_exact[group].append(intervention_exact)

        if baseline_exact < intervention_exact:
            improved_cases.append(case_id)
        elif baseline_exact > intervention_exact:
            regressed_cases.append(case_id)

        paired_rows.append(
            {
                "case_id": case_id,
                "distance_group": group,
                "distance_bin": intervention.metadata.get("distance_bin") or baseline.metadata.get("distance_bin"),
                "ground_truth_answer": intervention.ground_truth_answer,
                "baseline_prediction": _prediction_text(baseline),
                "intervention_prediction": _prediction_text(intervention),
                "baseline_exact_match": baseline_exact,
                "intervention_exact_match": intervention_exact,
                "delta_exact_match": intervention_exact - baseline_exact,
                "baseline_reflection_used": baseline.reflection_result is not None,
                "intervention_reflection_used": intervention.reflection_result is not None,
                "baseline_judge_score": baseline.judge_score,
                "intervention_judge_score": intervention.judge_score,
                "baseline_latency_ms": baseline_latency,
                "intervention_latency_ms": intervention_latency,
                "latency_delta_ms": intervention_latency - baseline_latency,
                "baseline_matched_skill_ids": baseline.matched_skill_ids,
                "intervention_matched_skill_ids": intervention.matched_skill_ids,
            }
        )

    distance_group_summary: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(group_baseline_exact) | set(group_intervention_exact)):
        baseline_values = group_baseline_exact[group]
        intervention_values = group_intervention_exact[group]
        distance_group_summary[group] = {
            "count": len(intervention_values),
            "baseline_exact_match_accuracy": _mean(baseline_values),
            "intervention_exact_match_accuracy": _mean(intervention_values),
            "accuracy_delta": _mean(intervention_values) - _mean(baseline_values),
        }

    baseline_shared_records = [baseline_records[case_id] for case_id in shared_case_ids]
    intervention_shared_records = [intervention_records[case_id] for case_id in shared_case_ids]
    baseline_positive_recall, positive_case_count = _conditional_accuracy(baseline_shared_records, _is_positive_case)
    intervention_positive_recall, _ = _conditional_accuracy(intervention_shared_records, _is_positive_case)
    baseline_negative_specificity, negative_case_count = _conditional_accuracy(
        baseline_shared_records,
        lambda record: not _is_positive_case(record),
    )
    intervention_negative_specificity, _ = _conditional_accuracy(
        intervention_shared_records,
        lambda record: not _is_positive_case(record),
    )
    baseline_skill_match_case_ids = [record.case_id for record in baseline_shared_records if record.matched_skill_ids]
    intervention_skill_match_case_ids = [record.case_id for record in intervention_shared_records if record.matched_skill_ids]
    baseline_reflection_case_ids = [record.case_id for record in baseline_shared_records if record.reflection_result is not None]
    intervention_reflection_case_ids = [
        record.case_id for record in intervention_shared_records if record.reflection_result is not None
    ]
    baseline_reflection_count = sum(record.reflection_result is not None for record in baseline_shared_records)
    intervention_reflection_count = sum(record.reflection_result is not None for record in intervention_shared_records)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_ids": baseline_run_ids,
        "intervention_run_ids": intervention_run_ids,
        "shared_case_count": len(shared_case_ids),
        "baseline_exact_match_accuracy": _mean(baseline_exact_values),
        "intervention_exact_match_accuracy": _mean(intervention_exact_values),
        "exact_match_delta": _mean(intervention_exact_values) - _mean(baseline_exact_values),
        "positive_case_count": positive_case_count,
        "negative_case_count": negative_case_count,
        "baseline_positive_recall": baseline_positive_recall,
        "intervention_positive_recall": intervention_positive_recall,
        "positive_recall_delta": intervention_positive_recall - baseline_positive_recall,
        "baseline_negative_specificity": baseline_negative_specificity,
        "intervention_negative_specificity": intervention_negative_specificity,
        "negative_specificity_delta": intervention_negative_specificity - baseline_negative_specificity,
        "baseline_judge_score_mean_judged_only": _mean(baseline_judge_values),
        "intervention_judge_score_mean_judged_only": _mean(intervention_judge_values),
        "judge_score_delta": _mean(intervention_judge_values) - _mean(baseline_judge_values),
        "baseline_latency_mean_ms": _mean(baseline_latencies),
        "intervention_latency_mean_ms": _mean(intervention_latencies),
        "latency_delta_ms": _mean(intervention_latencies) - _mean(baseline_latencies),
        "baseline_skill_match_case_count": len(baseline_skill_match_case_ids),
        "intervention_skill_match_case_count": len(intervention_skill_match_case_ids),
        "baseline_skill_match_rate": len(baseline_skill_match_case_ids) / len(shared_case_ids) if shared_case_ids else 0.0,
        "intervention_skill_match_rate": len(intervention_skill_match_case_ids) / len(shared_case_ids)
        if shared_case_ids
        else 0.0,
        "baseline_skill_match_case_ids": baseline_skill_match_case_ids,
        "intervention_skill_match_case_ids": intervention_skill_match_case_ids,
        "baseline_reflection_rate": baseline_reflection_count / len(shared_case_ids) if shared_case_ids else 0.0,
        "intervention_reflection_rate": intervention_reflection_count / len(shared_case_ids) if shared_case_ids else 0.0,
        "baseline_reflection_case_count": baseline_reflection_count,
        "intervention_reflection_case_count": intervention_reflection_count,
        "baseline_reflection_case_ids": baseline_reflection_case_ids,
        "intervention_reflection_case_ids": intervention_reflection_case_ids,
        "improved_case_count": len(improved_cases),
        "regressed_case_count": len(regressed_cases),
        "improved_case_ids": improved_cases,
        "regressed_case_ids": regressed_cases,
        "baseline_confusion": _build_confusion(baseline_shared_records),
        "intervention_confusion": _build_confusion(intervention_shared_records),
        "distance_group_summary": distance_group_summary,
        "paired_rows": paired_rows,
    }


def write_outputs(summary: dict[str, object], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    paired_rows = summary["paired_rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()) if paired_rows else [])
        if paired_rows:
            writer.writeheader()
            writer.writerows(paired_rows)

    lines = [
        "# Real Category 1 Paired Comparison",
        "",
        f"- Baseline run IDs: {', '.join(summary['baseline_run_ids'])}",
        f"- Intervention run IDs: {', '.join(summary['intervention_run_ids'])}",
        f"- Shared cases: {summary['shared_case_count']}",
        f"- Exact match: {summary['baseline_exact_match_accuracy']:.4f} -> {summary['intervention_exact_match_accuracy']:.4f} "
        f"(delta {summary['exact_match_delta']:+.4f})",
        f"- Positive recall: {summary['baseline_positive_recall']:.4f} -> {summary['intervention_positive_recall']:.4f} "
        f"(delta {summary['positive_recall_delta']:+.4f}, positives={summary['positive_case_count']})",
        f"- Negative specificity: {summary['baseline_negative_specificity']:.4f} -> "
        f"{summary['intervention_negative_specificity']:.4f} "
        f"(delta {summary['negative_specificity_delta']:+.4f}, negatives={summary['negative_case_count']})",
        f"- Judge mean (judged only): {summary['baseline_judge_score_mean_judged_only']:.2f} -> "
        f"{summary['intervention_judge_score_mean_judged_only']:.2f} "
        f"(delta {summary['judge_score_delta']:+.2f})",
        f"- Mean latency ms: {summary['baseline_latency_mean_ms']:.2f} -> "
        f"{summary['intervention_latency_mean_ms']:.2f} "
        f"(delta {summary['latency_delta_ms']:+.2f})",
        f"- Skill match rate: {summary['baseline_skill_match_rate']:.4f} -> "
        f"{summary['intervention_skill_match_rate']:.4f}",
        f"- Reflection rate: {summary['baseline_reflection_rate']:.4f} -> {summary['intervention_reflection_rate']:.4f}",
        f"- Improved cases: {summary['improved_case_count']}",
        f"- Regressed cases: {summary['regressed_case_count']}",
        "",
        "## Distance Group Summary",
        "",
    ]
    for group, row in summary["distance_group_summary"].items():
        lines.append(
            f"- {group}: count={row['count']}, "
            f"accuracy={row['baseline_exact_match_accuracy']:.4f}->{row['intervention_exact_match_accuracy']:.4f}, "
            f"delta={row['accuracy_delta']:+.4f}"
        )
    lines.extend(["", "## Confusion", ""])
    lines.append(f"- Baseline: {json.dumps(summary['baseline_confusion'], ensure_ascii=False)}")
    lines.append(f"- Intervention: {json.dumps(summary['intervention_confusion'], ensure_ascii=False)}")
    if summary["improved_case_ids"]:
        lines.extend(["", "## Improved Cases", ""])
        for case_id in summary["improved_case_ids"]:
            lines.append(f"- {case_id}")
    if summary["regressed_case_ids"]:
        lines.extend(["", "## Regressed Cases", ""])
        for case_id in summary["regressed_case_ids"]:
            lines.append(f"- {case_id}")
    if summary["baseline_skill_match_case_ids"] or summary["intervention_skill_match_case_ids"]:
        lines.extend(["", "## Skill Match Cases", ""])
        lines.append(f"- Baseline: {', '.join(summary['baseline_skill_match_case_ids']) or 'None'}")
        lines.append(f"- Intervention: {', '.join(summary['intervention_skill_match_case_ids']) or 'None'}")
    if summary["baseline_reflection_case_ids"] or summary["intervention_reflection_case_ids"]:
        lines.extend(["", "## Reflection Cases", ""])
        lines.append(f"- Baseline: {', '.join(summary['baseline_reflection_case_ids']) or 'None'}")
        lines.append(f"- Intervention: {', '.join(summary['intervention_reflection_case_ids']) or 'None'}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-id", action="append", required=True)
    parser.add_argument("--intervention-run-id", action="append", required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    summary = build_summary(args.baseline_run_id, args.intervention_run_id)
    write_outputs(summary, args.output_prefix)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
