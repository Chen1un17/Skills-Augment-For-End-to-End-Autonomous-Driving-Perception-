#!/usr/bin/env python3
"""Build a multi-category three-way synth DTPQA report from shared plan cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.evaluation.metrics import canonicalize_answer, exact_match  # noqa: E402
from ad_cornercase.schemas.evaluation import CasePredictionRecord  # noqa: E402


MODES = ("edge_only", "cloud_only", "hybrid")
GROUP_ORDER = ("far", "mid", "near", "unknown")
COUNT_WORDS = {
    "zero": 0,
    "none": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


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


def _prediction_text(record: CasePredictionRecord) -> str:
    if record.final_result.qa_report:
        return record.final_result.qa_report[0].answer
    if record.final_result.top_k_candidates:
        return record.final_result.top_k_candidates[0].label
    return ""


def _baseline_prediction_text(record: CasePredictionRecord) -> str:
    if record.baseline_result.qa_report:
        return record.baseline_result.qa_report[0].answer
    if record.baseline_result.top_k_candidates:
        return record.baseline_result.top_k_candidates[0].label
    return ""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_ms(record: CasePredictionRecord) -> float:
    return float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0))


def _canonical_prediction(record: CasePredictionRecord) -> str:
    options = record.metadata.get("answer_options")
    return canonicalize_answer(_prediction_text(record), options if isinstance(options, list) else None)


def _canonical_ground_truth(record: CasePredictionRecord) -> str:
    options = record.metadata.get("answer_options")
    return canonicalize_answer(record.ground_truth_answer, options if isinstance(options, list) else None)


def _baseline_exact_match(record: CasePredictionRecord) -> float:
    options = record.metadata.get("answer_options")
    normalized_prediction = canonicalize_answer(
        _baseline_prediction_text(record),
        options if isinstance(options, list) else None,
    )
    normalized_reference = canonicalize_answer(
        record.ground_truth_answer,
        options if isinstance(options, list) else None,
    )
    if not normalized_prediction or not normalized_reference:
        return 0.0
    if normalized_reference in {"yes", "no"}:
        if normalized_prediction == normalized_reference:
            return 1.0
        if normalized_prediction.startswith(f"{normalized_reference} "):
            return 1.0
    return 1.0 if normalized_prediction == normalized_reference else 0.0


def _is_positive_case(record: CasePredictionRecord) -> bool:
    return _canonical_ground_truth(record) == "yes"


def _parse_count(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in COUNT_WORDS:
        return COUNT_WORDS[normalized]
    if normalized.isdigit():
        return int(normalized)
    return None


def _special_metrics(question_type: str, records: list[CasePredictionRecord]) -> dict[str, object]:
    if not records:
        return {}

    if question_type == "category_1":
        positives = [record for record in records if _is_positive_case(record)]
        negatives = [record for record in records if not _is_positive_case(record)]
        return {
            "yes_recall": _mean([exact_match(record) for record in positives]),
            "no_specificity": _mean([exact_match(record) for record in negatives]),
        }

    if question_type in {"category_2", "category_4", "category_5", "category_6"}:
        confusion = Counter()
        error_answers = Counter()
        for record in records:
            ground_truth = _canonical_ground_truth(record) or record.ground_truth_answer
            prediction = _canonical_prediction(record) or _prediction_text(record)
            confusion[f"{ground_truth} -> {prediction}"] += 1
            if exact_match(record) == 0.0:
                error_answers[prediction or "<empty>"] += 1
        return {
            "confusion_summary": dict(confusion.most_common()),
            "top_error_answers": dict(error_answers.most_common(5)),
        }

    if question_type == "category_3":
        distribution = Counter()
        for record in records:
            if exact_match(record) == 1.0:
                distribution["exact"] += 1
                continue
            ground_truth_count = _parse_count(_canonical_ground_truth(record) or record.ground_truth_answer)
            prediction_count = _parse_count(_canonical_prediction(record) or _prediction_text(record))
            if ground_truth_count is None or prediction_count is None:
                distribution["unparsed_mismatch"] += 1
                continue
            delta = abs(prediction_count - ground_truth_count)
            if delta == 1:
                distribution["off_by_1"] += 1
            else:
                distribution["off_by_2_plus"] += 1
        return {"count_error_distribution": dict(distribution)}

    return {}


def _mode_summary(records: list[CasePredictionRecord], *, include_special_for: str | None = None) -> dict[str, object]:
    latencies = [_latency_ms(record) for record in records]
    reflection_records = [record for record in records if record.reflection_result is not None]
    skill_records = [record for record in records if record.matched_skill_ids]
    distance_summary: dict[str, dict[str, object]] = {}
    by_group: dict[str, list[CasePredictionRecord]] = defaultdict(list)
    for record in records:
        by_group[str(record.metadata.get("distance_group") or "unknown")].append(record)
    for group in GROUP_ORDER:
        group_records = by_group.get(group, [])
        if not group_records:
            continue
        distance_summary[group] = {
            "case_count": len(group_records),
            "exact_match_accuracy": _mean([exact_match(record) for record in group_records]),
        }

    summary: dict[str, object] = {
        "case_count": len(records),
        "exact_match_accuracy": _mean([exact_match(record) for record in records]),
        "mean_latency_ms": _mean(latencies),
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "reflection_rate": len(reflection_records) / len(records) if records else 0.0,
        "skill_match_rate": len(skill_records) / len(records) if records else 0.0,
        "reflection_precision": (
            sum(_baseline_exact_match(record) < exact_match(record) for record in reflection_records) / len(reflection_records)
            if reflection_records
            else 0.0
        ),
        "reflection_rescue_count": sum(_baseline_exact_match(record) < exact_match(record) for record in reflection_records),
        "reflection_harm_count": sum(_baseline_exact_match(record) > exact_match(record) for record in reflection_records),
        "skill_precision": (
            sum(_baseline_exact_match(record) < exact_match(record) for record in skill_records) / len(skill_records)
            if skill_records
            else 0.0
        ),
        "skill_rescue_count": sum(_baseline_exact_match(record) < exact_match(record) for record in skill_records),
        "skill_harm_count": sum(_baseline_exact_match(record) > exact_match(record) for record in skill_records),
        "invalid_answer_count": sum(1 for record in records if not _canonical_prediction(record)),
        "distance_group_accuracy": {
            group: values["exact_match_accuracy"] for group, values in distance_summary.items()
        },
        "distance_group_counts": {
            group: values["case_count"] for group, values in distance_summary.items()
        },
        "hybrid_strategy_counts": dict(
            Counter(
                str(record.metadata.get("hybrid_strategy"))
                for record in records
                if record.metadata.get("hybrid_strategy") is not None
            )
        ),
        "matched_skill_ids": sorted({skill_id for record in records for skill_id in record.matched_skill_ids}),
    }
    if include_special_for is not None:
        summary.update(_special_metrics(include_special_for, records))
    return summary


def build_summary(
    *,
    plan_path: Path,
    edge_run_ids: list[str],
    cloud_run_ids: list[str],
    hybrid_run_ids: list[str],
) -> dict[str, object]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_cases = [item for item in plan.get("cases", []) if isinstance(item, dict)]
    expected_case_ids = {str(item["case_id"]) for item in plan_cases}
    expected_by_question_type: dict[str, set[str]] = defaultdict(set)
    for item in plan_cases:
        expected_by_question_type[str(item["question_type"])].add(str(item["case_id"]))

    mode_records = {
        "edge_only": _load_records(edge_run_ids),
        "cloud_only": _load_records(cloud_run_ids),
        "hybrid": _load_records(hybrid_run_ids),
    }
    shared_case_ids = sorted(expected_case_ids & set(mode_records["edge_only"]) & set(mode_records["cloud_only"]) & set(mode_records["hybrid"]))
    records_by_mode = {
        mode: [mode_records[mode][case_id] for case_id in shared_case_ids]
        for mode in MODES
    }

    overall_summary = {
        mode: _mode_summary(records_by_mode[mode])
        for mode in MODES
    }
    question_type_summary: dict[str, dict[str, object]] = {}
    question_type_distance_summary: dict[str, dict[str, object]] = {}
    for question_type, case_ids in sorted(expected_by_question_type.items()):
        shared_question_case_ids = [case_id for case_id in shared_case_ids if case_id in case_ids]
        question_type_summary[question_type] = {}
        question_type_distance_summary[question_type] = {}
        for mode in MODES:
            subset_records = [mode_records[mode][case_id] for case_id in shared_question_case_ids]
            summary = _mode_summary(subset_records, include_special_for=question_type)
            question_type_summary[question_type][mode] = summary
            question_type_distance_summary[question_type][mode] = summary["distance_group_accuracy"]

    routing_summary = {
        "hybrid_strategy_counts": overall_summary["hybrid"]["hybrid_strategy_counts"],
        "hybrid_reflection_case_count": round(
            overall_summary["hybrid"]["reflection_rate"] * overall_summary["hybrid"]["case_count"]
        ),
        "hybrid_skill_match_rate": overall_summary["hybrid"]["skill_match_rate"],
        "hybrid_reflection_rate": overall_summary["hybrid"]["reflection_rate"],
    }

    per_case_rows: list[dict[str, object]] = []
    for case_id in shared_case_ids:
        edge_record = mode_records["edge_only"][case_id]
        cloud_record = mode_records["cloud_only"][case_id]
        hybrid_record = mode_records["hybrid"][case_id]
        per_case_rows.append(
            {
                "case_id": case_id,
                "question_type": str(hybrid_record.metadata.get("question_type") or edge_record.metadata.get("question_type") or ""),
                "distance_group": str(hybrid_record.metadata.get("distance_group") or "unknown"),
                "ground_truth_answer": hybrid_record.ground_truth_answer,
                "edge_prediction": _prediction_text(edge_record),
                "cloud_prediction": _prediction_text(cloud_record),
                "hybrid_prediction": _prediction_text(hybrid_record),
                "edge_exact_match": exact_match(edge_record),
                "cloud_exact_match": exact_match(cloud_record),
                "hybrid_exact_match": exact_match(hybrid_record),
                "edge_latency_ms": _latency_ms(edge_record),
                "cloud_latency_ms": _latency_ms(cloud_record),
                "hybrid_latency_ms": _latency_ms(hybrid_record),
                "hybrid_reflection_used": hybrid_record.reflection_result is not None,
                "hybrid_skill_match_count": len(hybrid_record.matched_skill_ids),
                "hybrid_strategy": hybrid_record.metadata.get("hybrid_strategy"),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "expected_case_count": len(expected_case_ids),
        "shared_case_count": len(shared_case_ids),
        "edge_run_ids": edge_run_ids,
        "cloud_run_ids": cloud_run_ids,
        "hybrid_run_ids": hybrid_run_ids,
        "overall_summary": overall_summary,
        "question_type_summary": question_type_summary,
        "question_type_distance_summary": question_type_distance_summary,
        "routing_summary": routing_summary,
        "per_case_rows": per_case_rows,
    }


def _write_json(summary: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# DTPQA Synth 500 Three-Way Report",
        "",
        f"- Shared cases: {summary['shared_case_count']}",
        f"- Edge runs: {', '.join(summary['edge_run_ids'])}",
        f"- Cloud runs: {', '.join(summary['cloud_run_ids'])}",
        f"- Hybrid runs: {', '.join(summary['hybrid_run_ids'])}",
        "",
        "## Overall Summary",
        "",
        "| Mode | Cases | Exact Match | Mean Latency (ms) | P50 (ms) | P95 (ms) | Reflection Rate | Skill Match Rate |",
        "|------|-------|-------------|-------------------|----------|----------|-----------------|------------------|",
    ]
    for mode in MODES:
        metrics = summary["overall_summary"][mode]
        lines.append(
            f"| {mode} | {metrics['case_count']} | {metrics['exact_match_accuracy']:.4f} | "
            f"{metrics['mean_latency_ms']:.2f} | {metrics['p50_latency_ms']:.2f} | {metrics['p95_latency_ms']:.2f} | "
            f"{metrics['reflection_rate']:.4f} | {metrics['skill_match_rate']:.4f} |"
        )

    lines.extend(["", "## Per Question Type", ""])
    for question_type, mode_summary in summary["question_type_summary"].items():
        lines.extend(
            [
                f"### {question_type}",
                "",
                "| Mode | Cases | Exact Match | Mean Latency (ms) | Reflection Rate | Skill Match Rate |",
                "|------|-------|-------------|-------------------|-----------------|------------------|",
            ]
        )
        for mode in MODES:
            metrics = mode_summary[mode]
            lines.append(
                f"| {mode} | {metrics['case_count']} | {metrics['exact_match_accuracy']:.4f} | "
                f"{metrics['mean_latency_ms']:.2f} | {metrics['reflection_rate']:.4f} | {metrics['skill_match_rate']:.4f} |"
            )
        if "yes_recall" in mode_summary["edge_only"]:
            lines.append("")
            lines.append(
                "| Mode | Yes Recall | No Specificity |"
            )
            lines.append("|------|------------|----------------|")
            for mode in MODES:
                metrics = mode_summary[mode]
                lines.append(f"| {mode} | {metrics['yes_recall']:.4f} | {metrics['no_specificity']:.4f} |")
        if "count_error_distribution" in mode_summary["edge_only"]:
            lines.append("")
            lines.append("Count error distributions:")
            for mode in MODES:
                metrics = mode_summary[mode]
                lines.append(f"- {mode}: {json.dumps(metrics['count_error_distribution'], ensure_ascii=False)}")
        if "confusion_summary" in mode_summary["edge_only"]:
            lines.append("")
            lines.append("Top error answers:")
            for mode in MODES:
                metrics = mode_summary[mode]
                lines.append(f"- {mode}: {json.dumps(metrics['top_error_answers'], ensure_ascii=False)}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(summary: dict[str, object], *, accuracy_path: Path, latency_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    question_types = sorted(summary["question_type_summary"])
    x = np.arange(len(question_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, mode in enumerate(MODES):
        values = [summary["question_type_summary"][question_type][mode]["exact_match_accuracy"] * 100 for question_type in question_types]
        ax.bar(x + (index - 1) * width, values, width, label=mode)
    ax.set_ylabel("Exact Match Accuracy (%)")
    ax.set_title("DTPQA Synth Accuracy by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(question_types)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, mode in enumerate(MODES):
        values = [summary["question_type_summary"][question_type][mode]["mean_latency_ms"] / 1000.0 for question_type in question_types]
        ax.bar(x + (index - 1) * width, values, width, label=mode)
    ax.set_ylabel("Mean Latency (s)")
    ax.set_title("DTPQA Synth Latency by Category")
    ax.set_xticks(x)
    ax.set_xticklabels(question_types)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    latency_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(latency_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--edge-run-id", action="append", required=True)
    parser.add_argument("--cloud-run-id", action="append", required=True)
    parser.add_argument("--hybrid-run-id", action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--accuracy-plot", type=Path, required=True)
    parser.add_argument("--latency-plot", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(
        plan_path=args.plan,
        edge_run_ids=args.edge_run_id,
        cloud_run_ids=args.cloud_run_id,
        hybrid_run_ids=args.hybrid_run_id,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_json(summary, args.output_prefix.with_suffix(".json"))
    _write_csv(summary["per_case_rows"], args.output_prefix.with_suffix(".csv"))
    _write_markdown(summary, args.output_prefix.with_suffix(".md"))
    _write_plots(summary, accuracy_path=args.accuracy_plot, latency_path=args.latency_plot)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
