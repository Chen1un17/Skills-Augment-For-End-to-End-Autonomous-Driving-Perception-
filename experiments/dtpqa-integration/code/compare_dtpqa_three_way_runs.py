#!/usr/bin/env python3
"""Build a shared-case three-way comparison across edge-only, cloud-only, and hybrid runs."""

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

MODES = ("edge_only", "cloud_only", "hybrid")


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


def _mode_metrics(records: list[CasePredictionRecord]) -> dict[str, object]:
    positives = [record for record in records if _is_positive_case(record)]
    negatives = [record for record in records if not _is_positive_case(record)]
    far_records = [record for record in records if str(record.metadata.get("distance_group") or "unknown") == "far"]
    exact_values = [exact_match(record) for record in records]
    reflection_count = sum(record.reflection_result is not None for record in records)
    skill_match_count = sum(bool(record.matched_skill_ids) for record in records)
    return {
        "case_count": len(records),
        "exact_match_accuracy": _mean(exact_values),
        "positive_recall": _mean([exact_match(record) for record in positives]),
        "negative_specificity": _mean([exact_match(record) for record in negatives]),
        "far_accuracy": _mean([exact_match(record) for record in far_records]),
        "mean_latency_ms": _mean([_latency_ms(record) for record in records]),
        "reflection_rate": reflection_count / len(records) if records else 0.0,
        "skill_match_rate": skill_match_count / len(records) if records else 0.0,
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
    shared_case_ids = sorted(
        set(mode_records["edge_only"])
        & set(mode_records["cloud_only"])
        & set(mode_records["hybrid"])
    )

    records_by_mode = {
        mode: [mode_records[mode][case_id] for case_id in shared_case_ids]
        for mode in MODES
    }
    mode_summaries = {
        mode: _mode_metrics(records_by_mode[mode])
        for mode in MODES
    }

    pairwise = {
        "edge_vs_cloud_exact_match_delta": (
            float(mode_summaries["cloud_only"]["exact_match_accuracy"])
            - float(mode_summaries["edge_only"]["exact_match_accuracy"])
        ),
        "edge_vs_hybrid_exact_match_delta": (
            float(mode_summaries["hybrid"]["exact_match_accuracy"])
            - float(mode_summaries["edge_only"]["exact_match_accuracy"])
        ),
        "cloud_vs_hybrid_exact_match_delta": (
            float(mode_summaries["hybrid"]["exact_match_accuracy"])
            - float(mode_summaries["cloud_only"]["exact_match_accuracy"])
        ),
        "edge_vs_cloud_latency_delta_ms": (
            float(mode_summaries["cloud_only"]["mean_latency_ms"])
            - float(mode_summaries["edge_only"]["mean_latency_ms"])
        ),
        "edge_vs_hybrid_latency_delta_ms": (
            float(mode_summaries["hybrid"]["mean_latency_ms"])
            - float(mode_summaries["edge_only"]["mean_latency_ms"])
        ),
        "cloud_vs_hybrid_latency_delta_ms": (
            float(mode_summaries["hybrid"]["mean_latency_ms"])
            - float(mode_summaries["cloud_only"]["mean_latency_ms"])
        ),
    }

    per_case_rows: list[dict[str, object]] = []
    distance_counts: dict[str, int] = defaultdict(int)
    for case_id in shared_case_ids:
        edge_record = mode_records["edge_only"][case_id]
        cloud_record = mode_records["cloud_only"][case_id]
        hybrid_record = mode_records["hybrid"][case_id]
        distance_group = str(hybrid_record.metadata.get("distance_group") or edge_record.metadata.get("distance_group") or "unknown")
        distance_counts[distance_group] += 1
        per_case_rows.append(
            {
                "case_id": case_id,
                "distance_group": distance_group,
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
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shared_case_count": len(shared_case_ids),
        "edge_run_ids": edge_run_ids,
        "cloud_run_ids": cloud_run_ids,
        "hybrid_run_ids": hybrid_run_ids,
        "distance_group_counts": dict(sorted(distance_counts.items())),
        "mode_summaries": mode_summaries,
        "pairwise_deltas": pairwise,
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
    lines = [
        "# DTPQA Three-Way Comparison",
        "",
        f"- Shared cases: {summary['shared_case_count']}",
        f"- Edge runs: {', '.join(summary['edge_run_ids'])}",
        f"- Cloud runs: {', '.join(summary['cloud_run_ids'])}",
        f"- Hybrid runs: {', '.join(summary['hybrid_run_ids'])}",
        "",
        "| Mode | Exact Match | Positive Recall | Negative Specificity | Far Accuracy | Mean Latency (ms) | Reflection Rate | Skill Match Rate |",
        "|------|-------------|-----------------|----------------------|--------------|-------------------|-----------------|------------------|",
    ]
    for mode in MODES:
        metrics = mode_summaries[mode]
        lines.append(
            f"| {mode} | "
            f"{metrics['exact_match_accuracy']:.4f} | "
            f"{metrics['positive_recall']:.4f} | "
            f"{metrics['negative_specificity']:.4f} | "
            f"{metrics['far_accuracy']:.4f} | "
            f"{metrics['mean_latency_ms']:.2f} | "
            f"{metrics['reflection_rate']:.4f} | "
            f"{metrics['skill_match_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## Pairwise Deltas",
        "",
    ])
    for key, value in summary["pairwise_deltas"].items():
        lines.append(f"- {key}: {value:+.4f}")

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
