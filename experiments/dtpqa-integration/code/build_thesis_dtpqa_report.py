#!/usr/bin/env python3
"""Build a thesis-oriented DTPQA three-way report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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

MODES = ("edge_only", "cloud_only", "hybrid")

# Extracted from the original DTPQA evaluation paper:
# Theodoridis et al., "Evaluating Small Vision-Language Models on
# Distance-Dependent Traffic Perception", arXiv:2510.08352, Table 4 and Table 5.
PAPER_BASELINES = [
    {
        "method": "Human Performance",
        "family": "Human",
        "dtpqa_avg": 84.7,
        "dtp_synth_cat1": 95.7,
        "dtp_real_cat1": 82.6,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 87.0,
    },
    {
        "method": "InternVL3-78B",
        "family": "Large VLM",
        "dtpqa_avg": 66.1,
        "dtp_synth_cat1": 85.9,
        "dtp_real_cat1": 71.2,
        "cat1_synth_negative_specificity": None,
        "cat1_real_negative_specificity": None,
    },
    {
        "method": "Ovis2-2B",
        "family": "Small VLM",
        "dtpqa_avg": 59.4,
        "dtp_synth_cat1": 71.5,
        "dtp_real_cat1": 70.7,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 96.0,
    },
    {
        "method": "Qwen2.5-VL-3B",
        "family": "Small VLM",
        "dtpqa_avg": 50.2,
        "dtp_synth_cat1": 17.3,
        "dtp_real_cat1": 22.9,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 100.0,
    },
    {
        "method": "SAIL-VL-2B",
        "family": "Small VLM",
        "dtpqa_avg": 53.8,
        "dtp_synth_cat1": 43.7,
        "dtp_real_cat1": 46.9,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 100.0,
    },
    {
        "method": "InternVL2.5-2B-MPO",
        "family": "Small VLM",
        "dtpqa_avg": 59.4,
        "dtp_synth_cat1": 71.0,
        "dtp_real_cat1": 68.3,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 95.5,
    },
    {
        "method": "InternVL2.5-2B",
        "family": "Small VLM",
        "dtpqa_avg": 55.8,
        "dtp_synth_cat1": 52.8,
        "dtp_real_cat1": 62.6,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 98.5,
    },
    {
        "method": "Ovis2-1B",
        "family": "Small VLM",
        "dtpqa_avg": 48.1,
        "dtp_synth_cat1": 33.6,
        "dtp_real_cat1": 63.0,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 98.0,
    },
    {
        "method": "Aquila-VL-2B",
        "family": "Small VLM",
        "dtpqa_avg": 54.5,
        "dtp_synth_cat1": 42.9,
        "dtp_real_cat1": 52.2,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 99.5,
    },
    {
        "method": "DeepSeek-VL2-Tiny",
        "family": "Small VLM",
        "dtpqa_avg": 52.9,
        "dtp_synth_cat1": 53.8,
        "dtp_real_cat1": 55.3,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 99.5,
    },
    {
        "method": "Qwen2-VL-2B",
        "family": "Small VLM",
        "dtpqa_avg": 53.6,
        "dtp_synth_cat1": 61.4,
        "dtp_real_cat1": 44.0,
        "cat1_synth_negative_specificity": 100.0,
        "cat1_real_negative_specificity": 100.0,
    },
]


def _load_records(run_ids: list[str]) -> dict[str, CasePredictionRecord]:
    records_by_case: dict[str, CasePredictionRecord] = {}
    for run_id in run_ids:
        predictions_path = ROOT / "data" / "artifacts" / run_id / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = CasePredictionRecord.model_validate_json(line)
                    records_by_case[record.case_id] = record
    return records_by_case


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt_optional(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    return statistics.quantiles(ordered, n=100, method="inclusive")[int(q) - 1]


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


def _latency_ms(record: CasePredictionRecord) -> float:
    return float(record.metadata.get("pipeline_latency_ms", record.final_result.latency_ms or 0.0))


def _is_positive_case(record: CasePredictionRecord) -> bool:
    options = record.metadata.get("answer_options")
    normalized = canonicalize_answer(record.ground_truth_answer, options if isinstance(options, list) else None)
    return normalized == "yes"


def _baseline_exact_match(record: CasePredictionRecord) -> float:
    options = record.metadata.get("answer_options")
    normalized = canonicalize_answer(
        _baseline_prediction_text(record),
        options if isinstance(options, list) else None,
    )
    ground_truth = canonicalize_answer(
        record.ground_truth_answer,
        options if isinstance(options, list) else None,
    )
    if not normalized or not ground_truth:
        return 0.0
    if ground_truth in {"yes", "no"}:
        if normalized == ground_truth:
            return 1.0
        if normalized.startswith(f"{ground_truth} "):
            return 1.0
    return 1.0 if normalized == ground_truth else 0.0


def _mode_metrics(records: list[CasePredictionRecord]) -> dict[str, object]:
    positives = [record for record in records if _is_positive_case(record)]
    negatives = [record for record in records if not _is_positive_case(record)]
    far_records = [record for record in records if str(record.metadata.get("distance_group") or "unknown") == "far"]
    latencies = [_latency_ms(record) for record in records]
    exact_values = [exact_match(record) for record in records]
    baseline_exact_values = [_baseline_exact_match(record) for record in records]
    distance_summary: dict[str, dict[str, float | int]] = {}
    by_group: dict[str, list[CasePredictionRecord]] = defaultdict(list)
    for record in records:
        by_group[str(record.metadata.get("distance_group") or "unknown")].append(record)
    for group, group_records in sorted(by_group.items()):
        distance_summary[group] = {
            "case_count": len(group_records),
            "exact_match_accuracy": _mean([exact_match(record) for record in group_records]),
        }

    reflection_count = sum(record.reflection_result is not None for record in records)
    skill_match_count = sum(bool(record.matched_skill_ids) for record in records)
    total_matched_skill_uses = sum(len(record.matched_skill_ids) for record in records)
    unique_matched_skill_ids = sorted({skill_id for record in records for skill_id in record.matched_skill_ids})
    strategy_counts = Counter(
        str(record.metadata.get("hybrid_strategy"))
        for record in records
        if record.metadata.get("hybrid_strategy") is not None
    )
    reflected_records = [record for record in records if record.reflection_result is not None]
    skill_records = [record for record in records if record.matched_skill_ids]
    reflection_rescue_count = sum(
        _baseline_exact_match(record) < exact_match(record) for record in reflected_records
    )
    reflection_harm_count = sum(
        _baseline_exact_match(record) > exact_match(record) for record in reflected_records
    )
    skill_rescue_count = sum(
        _baseline_exact_match(record) < exact_match(record) for record in skill_records
    )
    skill_harm_count = sum(
        _baseline_exact_match(record) > exact_match(record) for record in skill_records
    )

    return {
        "case_count": len(records),
        "baseline_exact_match_accuracy": _mean(baseline_exact_values),
        "exact_match_accuracy": _mean(exact_values),
        "positive_recall": _mean([exact_match(record) for record in positives]),
        "negative_specificity": _mean([exact_match(record) for record in negatives]),
        "far_accuracy": _mean([exact_match(record) for record in far_records]),
        "mean_latency_ms": _mean(latencies),
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95),
        "reflection_rate": reflection_count / len(records) if records else 0.0,
        "skill_match_rate": skill_match_count / len(records) if records else 0.0,
        "unique_matched_skill_count": len(unique_matched_skill_ids),
        "avg_matched_skills_per_case": total_matched_skill_uses / len(records) if records else 0.0,
        "avg_matched_skills_per_matched_case": (
            total_matched_skill_uses / skill_match_count if skill_match_count else 0.0
        ),
        "reflection_rescue_count": reflection_rescue_count,
        "reflection_harm_count": reflection_harm_count,
        "reflection_precision": (
            reflection_rescue_count / reflection_count if reflection_count else 0.0
        ),
        "skill_rescue_count": skill_rescue_count,
        "skill_harm_count": skill_harm_count,
        "skill_precision": skill_rescue_count / skill_match_count if skill_match_count else 0.0,
        "invalid_answer_count": sum(_prediction_text(record).strip() == "" for record in records),
        "distance_summary": distance_summary,
        "hybrid_strategy_counts": dict(strategy_counts),
    }


def build_summary(
    *,
    edge_run_ids: list[str],
    cloud_run_ids: list[str],
    hybrid_run_ids: list[str],
) -> dict[str, object]:
    mode_records = {
        "edge_only": _load_records(edge_run_ids),
        "cloud_only": _load_records(cloud_run_ids),
        "hybrid": _load_records(hybrid_run_ids),
    }
    shared_case_ids = sorted(set.intersection(*(set(records.keys()) for records in mode_records.values())))
    records_by_mode = {mode: [mode_records[mode][case_id] for case_id in shared_case_ids] for mode in MODES}
    mode_summaries = {mode: _mode_metrics(records_by_mode[mode]) for mode in MODES}

    edge_records = {record.case_id: record for record in records_by_mode["edge_only"]}
    cloud_records = {record.case_id: record for record in records_by_mode["cloud_only"]}
    hybrid_records = {record.case_id: record for record in records_by_mode["hybrid"]}

    improved_case_ids: list[str] = []
    regressed_case_ids: list[str] = []
    cloud_better_than_edge_case_ids: list[str] = []
    hybrid_matches_cloud_case_ids: list[str] = []
    far_rescued_case_ids: list[str] = []
    per_case_rows: list[dict[str, object]] = []

    for case_id in shared_case_ids:
        edge_record = edge_records[case_id]
        cloud_record = cloud_records[case_id]
        hybrid_record = hybrid_records[case_id]
        edge_correct = exact_match(edge_record)
        cloud_correct = exact_match(cloud_record)
        hybrid_correct = exact_match(hybrid_record)
        if edge_correct < hybrid_correct:
            improved_case_ids.append(case_id)
            if str(hybrid_record.metadata.get("distance_group") or "unknown") == "far":
                far_rescued_case_ids.append(case_id)
        elif edge_correct > hybrid_correct:
            regressed_case_ids.append(case_id)
        if edge_correct < cloud_correct:
            cloud_better_than_edge_case_ids.append(case_id)
        if cloud_correct == hybrid_correct == 1.0:
            hybrid_matches_cloud_case_ids.append(case_id)

        per_case_rows.append(
            {
                "case_id": case_id,
                "distance_group": str(hybrid_record.metadata.get("distance_group") or "unknown"),
                "ground_truth_answer": hybrid_record.ground_truth_answer,
                "edge_prediction": _prediction_text(edge_record),
                "cloud_prediction": _prediction_text(cloud_record),
                "hybrid_prediction": _prediction_text(hybrid_record),
                "edge_exact_match": edge_correct,
                "cloud_exact_match": cloud_correct,
                "hybrid_exact_match": hybrid_correct,
                "edge_latency_ms": _latency_ms(edge_record),
                "cloud_latency_ms": _latency_ms(cloud_record),
                "hybrid_latency_ms": _latency_ms(hybrid_record),
                "hybrid_reflection_used": hybrid_record.reflection_result is not None,
                "hybrid_strategy": hybrid_record.metadata.get("hybrid_strategy"),
            }
        )

    edge_correct_total = sum(exact_match(record) for record in records_by_mode["edge_only"])
    cloud_correct_total = sum(exact_match(record) for record in records_by_mode["cloud_only"])
    hybrid_correct_total = sum(exact_match(record) for record in records_by_mode["hybrid"])
    oracle_correct_total = sum(
        max(
            exact_match(edge_records[case_id]),
            exact_match(cloud_records[case_id]),
        )
        for case_id in shared_case_ids
    )

    edge_error_case_ids = [record.case_id for record in records_by_mode["edge_only"] if exact_match(record) == 0.0]
    edge_correct_case_ids = [record.case_id for record in records_by_mode["edge_only"] if exact_match(record) == 1.0]
    hybrid_reflection_case_ids = [
        record.case_id for record in records_by_mode["hybrid"] if record.reflection_result is not None
    ]

    rescue_rate = len(improved_case_ids) / len(edge_error_case_ids) if edge_error_case_ids else 0.0
    harm_rate = len(regressed_case_ids) / len(edge_correct_case_ids) if edge_correct_case_ids else 0.0
    cloud_call_reduction_vs_cloud_only = 1.0 - (
        float(mode_summaries["hybrid"]["reflection_rate"])
    )
    latency_reduction_vs_cloud_only = 1.0 - (
        float(mode_summaries["hybrid"]["mean_latency_ms"]) / float(mode_summaries["cloud_only"]["mean_latency_ms"])
        if float(mode_summaries["cloud_only"]["mean_latency_ms"])
        else 0.0
    )
    cloud_gain_denominator = cloud_correct_total - edge_correct_total
    hybrid_gain_capture_vs_cloud = (
        (hybrid_correct_total - edge_correct_total) / cloud_gain_denominator
        if cloud_gain_denominator > 0
        else 0.0
    )
    hybrid_cloud_calls_per_rescue = (
        len(hybrid_reflection_case_ids) / len(improved_case_ids) if improved_case_ids else 0.0
    )
    hybrid_marginal_latency_per_rescue_ms = (
        (float(mode_summaries["hybrid"]["mean_latency_ms"]) - float(mode_summaries["edge_only"]["mean_latency_ms"]))
        * len(shared_case_ids)
        / len(improved_case_ids)
        if improved_case_ids
        else 0.0
    )
    paper_cat1_best = max(
        row["dtp_synth_cat1"] for row in PAPER_BASELINES if row["family"] == "Small VLM"
    )
    paper_avg_best = max(
        row["dtpqa_avg"] for row in PAPER_BASELINES if row["family"] == "Small VLM"
    )
    paper_cat1_best_method = next(
        row["method"]
        for row in PAPER_BASELINES
        if row["family"] == "Small VLM" and row["dtp_synth_cat1"] == paper_cat1_best
    )
    paper_avg_best_method = next(
        row["method"]
        for row in PAPER_BASELINES
        if row["family"] == "Small VLM" and row["dtpqa_avg"] == paper_avg_best
    )
    paper_cat1_synth_rows = []
    for row in PAPER_BASELINES:
        paper_cat1_synth_rows.append(
            {
                "method": row["method"],
                "family": row["family"],
                "dtp_synth_cat1_accuracy": row["dtp_synth_cat1"],
                "cat1_synth_negative_specificity": row["cat1_synth_negative_specificity"],
                "dtp_real_cat1_accuracy": row["dtp_real_cat1"],
                "cat1_real_negative_specificity": row["cat1_real_negative_specificity"],
                "dtpqa_avg": row["dtpqa_avg"],
            }
        )
    for mode in MODES:
        metrics = mode_summaries[mode]
        paper_cat1_synth_rows.append(
            {
                "method": f"Ours ({mode})",
                "family": "Our system",
                "dtp_synth_cat1_accuracy": round(float(metrics["exact_match_accuracy"]) * 100, 2),
                "cat1_synth_negative_specificity": round(float(metrics["negative_specificity"]) * 100, 2),
                "dtp_real_cat1_accuracy": None,
                "cat1_real_negative_specificity": None,
                "dtpqa_avg": None,
                "mean_latency_ms": round(float(metrics["mean_latency_ms"]), 2),
                "reflection_rate": round(float(metrics["reflection_rate"]) * 100, 2),
                "sample_note": "Our clean 50-sample DTP-Synth/Cat.1 subset",
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_case_count": len(shared_case_ids),
        "edge_run_ids": edge_run_ids,
        "cloud_run_ids": cloud_run_ids,
        "hybrid_run_ids": hybrid_run_ids,
        "mode_summaries": mode_summaries,
        "thesis_metrics": {
            "edge_error_case_count": len(edge_error_case_ids),
            "edge_correct_case_count": len(edge_correct_case_ids),
            "hybrid_improved_case_count": len(improved_case_ids),
            "hybrid_regressed_case_count": len(regressed_case_ids),
            "hybrid_rescue_rate_over_edge_errors": rescue_rate,
            "hybrid_harm_rate_over_edge_correct": harm_rate,
            "far_rescued_case_count": len(far_rescued_case_ids),
            "hybrid_reflection_case_count": len(hybrid_reflection_case_ids),
            "cloud_only_better_than_edge_case_count": len(cloud_better_than_edge_case_ids),
            "hybrid_matches_cloud_on_cloud_wins_case_count": len(hybrid_matches_cloud_case_ids),
            "oracle_edge_cloud_accuracy": oracle_correct_total / len(shared_case_ids) if shared_case_ids else 0.0,
            "hybrid_cloud_call_reduction_vs_cloud_only": cloud_call_reduction_vs_cloud_only,
            "hybrid_latency_reduction_vs_cloud_only": latency_reduction_vs_cloud_only,
            "hybrid_gain_capture_vs_cloud": hybrid_gain_capture_vs_cloud,
            "hybrid_cloud_calls_per_rescue": hybrid_cloud_calls_per_rescue,
            "hybrid_marginal_latency_per_rescue_ms": hybrid_marginal_latency_per_rescue_ms,
            "hybrid_accuracy_delta_vs_edge": (hybrid_correct_total - edge_correct_total) / len(shared_case_ids)
            if shared_case_ids
            else 0.0,
            "hybrid_accuracy_delta_vs_cloud": (hybrid_correct_total - cloud_correct_total) / len(shared_case_ids)
            if shared_case_ids
            else 0.0,
            "paper_best_small_vlm_dtpqa_avg": paper_avg_best,
            "paper_best_small_vlm_dtpqa_avg_method": paper_avg_best_method,
            "paper_best_small_vlm_cat1_synth": paper_cat1_best,
            "paper_best_small_vlm_cat1_synth_method": paper_cat1_best_method,
        },
        "paper_baselines": PAPER_BASELINES,
        "paper_cat1_synth_comparison_rows": paper_cat1_synth_rows,
        "improved_case_ids": improved_case_ids,
        "regressed_case_ids": regressed_case_ids,
        "far_rescued_case_ids": far_rescued_case_ids,
        "per_case_rows": per_case_rows,
    }


def write_outputs(summary: dict[str, object], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = summary["per_case_rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    mode_summaries = summary["mode_summaries"]
    thesis_metrics = summary["thesis_metrics"]
    lines = [
        "# DTPQA Thesis-Oriented Three-Way Report",
        "",
        f"- Shared cases: {summary['shared_case_count']}",
        f"- Edge runs: {', '.join(summary['edge_run_ids'])}",
        f"- Cloud runs: {', '.join(summary['cloud_run_ids'])}",
        f"- Hybrid runs: {', '.join(summary['hybrid_run_ids'])}",
        "",
        "## Core Metrics",
        "",
        "| Mode | Baseline Acc | Final Acc | Yes Recall | No Specificity | Far Accuracy | Mean Latency (ms) | P50 (ms) | P95 (ms) | Reflection Rate |",
        "|------|--------------|-----------|------------|----------------|--------------|-------------------|----------|----------|-----------------|",
    ]
    for mode in MODES:
        metrics = mode_summaries[mode]
        lines.append(
            f"| {mode} | "
            f"{metrics['baseline_exact_match_accuracy']:.4f} | "
            f"{metrics['exact_match_accuracy']:.4f} | "
            f"{metrics['positive_recall']:.4f} | "
            f"{metrics['negative_specificity']:.4f} | "
            f"{metrics['far_accuracy']:.4f} | "
            f"{metrics['mean_latency_ms']:.2f} | "
            f"{metrics['p50_latency_ms']:.2f} | "
            f"{metrics['p95_latency_ms']:.2f} | "
            f"{metrics['reflection_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Thesis-Facing Metrics",
            "",
            f"- Hybrid rescue rate over edge errors: {thesis_metrics['hybrid_rescue_rate_over_edge_errors']:.4f}",
            f"- Hybrid harm rate over edge correct: {thesis_metrics['hybrid_harm_rate_over_edge_correct']:.4f}",
            f"- Far rescued case count: {thesis_metrics['far_rescued_case_count']}",
            f"- Oracle(edge, cloud) accuracy upper bound: {thesis_metrics['oracle_edge_cloud_accuracy']:.4f}",
            f"- Hybrid cloud-call reduction vs cloud-only: {thesis_metrics['hybrid_cloud_call_reduction_vs_cloud_only']:.4f}",
            f"- Hybrid latency reduction vs cloud-only: {thesis_metrics['hybrid_latency_reduction_vs_cloud_only']:.4f}",
            f"- Hybrid gain capture vs cloud gain: {thesis_metrics['hybrid_gain_capture_vs_cloud']:.4f}",
            f"- Hybrid cloud calls per rescued case: {thesis_metrics['hybrid_cloud_calls_per_rescue']:.4f}",
            f"- Hybrid marginal latency per rescued case (ms): {thesis_metrics['hybrid_marginal_latency_per_rescue_ms']:.2f}",
            f"- Hybrid accuracy delta vs edge: {thesis_metrics['hybrid_accuracy_delta_vs_edge']:+.4f}",
            f"- Hybrid accuracy delta vs cloud: {thesis_metrics['hybrid_accuracy_delta_vs_cloud']:+.4f}",
            "",
            "## Skill And Routing Metrics",
            "",
            "| Mode | Skill Match Rate | Unique Matched Skills | Skill Precision | Reflection Precision | Reflection Rescue Count | Reflection Harm Count |",
            "|------|------------------|-----------------------|-----------------|----------------------|-------------------------|-----------------------|",
        ]
    )
    for mode in MODES:
        metrics = mode_summaries[mode]
        lines.append(
            f"| {mode} | "
            f"{metrics['skill_match_rate']:.4f} | "
            f"{metrics['unique_matched_skill_count']} | "
            f"{metrics['skill_precision']:.4f} | "
            f"{metrics['reflection_precision']:.4f} | "
            f"{metrics['reflection_rescue_count']} | "
            f"{metrics['reflection_harm_count']} |"
        )
    lines.extend(
        [
            "",
            "## Published DTPQA Baselines",
            "",
            "Source: Theodoridis et al., `Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception` (arXiv:2510.08352), Table 4 and Table 5.",
            "",
            "| Method | Family | DTPQA Avg | DTP-Synth Cat.1 | DTP-Real Cat.1 | Cat.1-Synth Negative Specificity | Cat.1-Real Negative Specificity |",
            "|--------|--------|-----------|------------------|----------------|----------------------------------|---------------------------------|",
        ]
    )
    for row in summary["paper_baselines"]:
        lines.append(
            f"| {row['method']} | {row['family']} | "
            f"{_fmt_optional(row['dtpqa_avg'])} | "
            f"{_fmt_optional(row['dtp_synth_cat1'])} | "
            f"{_fmt_optional(row['dtp_real_cat1'])} | "
            f"{_fmt_optional(row['cat1_synth_negative_specificity'])} | "
            f"{_fmt_optional(row['cat1_real_negative_specificity'])} |"
        )
    lines.extend(
        [
            "",
            "## Cat.1 Comparison With Our System",
            "",
            "Our rows below are based on the clean 50-sample DTP-Synth/Cat.1 subset and are not directly equivalent to the full-benchmark averages above, but they are useful for thesis-side targeted comparison on the pedestrian-presence task.",
            "",
            "| Method | Family | Cat.1-Synth Accuracy | Cat.1-Synth Negative Specificity | Mean Latency (ms) | Reflection Rate | Note |",
            "|--------|--------|----------------------|----------------------------------|-------------------|-----------------|------|",
        ]
    )
    for row in summary["paper_cat1_synth_comparison_rows"]:
        mean_latency = row.get("mean_latency_ms")
        reflection_rate = row.get("reflection_rate")
        note = row.get("sample_note", "Published DTPQA result")
        cat1_acc = row["dtp_synth_cat1_accuracy"]
        cat1_spec = row["cat1_synth_negative_specificity"]
        lines.append(
            f"| {row['method']} | {row['family']} | "
            f"{_fmt_optional(cat1_acc)} | "
            f"{_fmt_optional(cat1_spec)} | "
            f"{_fmt_optional(mean_latency, digits=2)} | "
            f"{_fmt_optional(reflection_rate, digits=2)} | "
            f"{note} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid Routing",
            "",
        ]
    )
    hybrid_strategy_counts = mode_summaries["hybrid"]["hybrid_strategy_counts"]
    for strategy, count in sorted(hybrid_strategy_counts.items()):
        lines.append(f"- {strategy}: {count}")
    lines.extend(
        [
            "",
            "## Case-Level Changes",
            "",
            f"- Improved case IDs: {', '.join(summary['improved_case_ids']) if summary['improved_case_ids'] else 'None'}",
            f"- Regressed case IDs: {', '.join(summary['regressed_case_ids']) if summary['regressed_case_ids'] else 'None'}",
            f"- Far rescued case IDs: {', '.join(summary['far_rescued_case_ids']) if summary['far_rescued_case_ids'] else 'None'}",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-run-id", action="append", required=True)
    parser.add_argument("--cloud-run-id", action="append", required=True)
    parser.add_argument("--hybrid-run-id", action="append", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(
        edge_run_ids=args.edge_run_id,
        cloud_run_ids=args.cloud_run_id,
        hybrid_run_ids=args.hybrid_run_id,
    )
    write_outputs(summary, args.output_prefix)
    print(args.output_prefix.with_suffix(".json"))


if __name__ == "__main__":
    main()
