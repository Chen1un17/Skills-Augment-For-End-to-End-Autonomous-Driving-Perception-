#!/usr/bin/env python3
"""Build thesis-facing evidence tables from local experiment artifacts.

This script keeps local artifact provenance out of the thesis bibliography while
still giving us a reproducible ledger that maps every core paper claim to a
concrete experiment source.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_ROOT / "data"

SYNTH50_JSON = ROOT / "experiments/dtpqa-integration/results/thesis_synth50_three_way_20260410.json"
REAL_PILOT_JSON = ROOT / "experiments/dtpqa-integration/results/current_real_cat1_pilot_summary.json"
REAL_PARTIAL_JSON = ROOT / "experiments/dtpqa-integration/results/real_cat1_old_vs_intervention_quick_partial.json"

ADAPTATION_RUNS = {
    "skill_pilot_positive": ROOT / "data/artifacts/dtpqa_200_final_20260402_103653/predictions.jsonl",
    "skill_scale_risk": ROOT / "data/artifacts/dtpqa_200_20260402_145332/predictions.jsonl",
    "skill_small_regression": ROOT / "data/artifacts/dtpqa_synth_50_20260401_115405/predictions.jsonl",
}


def _normalize_answer(text: str) -> str:
    return (text or "").strip().lower()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_adaptation_run(path: Path) -> dict:
    total = 0
    baseline_correct = 0
    final_correct = 0
    matched_cases = 0
    reflection_cases = 0
    improved_cases = 0
    harmed_cases = 0
    improved_by_distance: Counter[str] = Counter()
    harmed_by_distance: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            gt = _normalize_answer(row.get("ground_truth_answer", ""))
            baseline = _normalize_answer(((row.get("baseline_result") or {}).get("qa_report") or [{}])[0].get("answer", ""))
            final = _normalize_answer(((row.get("final_result") or {}).get("qa_report") or [{}])[0].get("answer", ""))
            distance_group = (row.get("metadata") or {}).get("distance_group", "unknown")
            baseline_ok = bool(gt) and baseline == gt
            final_ok = bool(gt) and final == gt
            baseline_correct += int(baseline_ok)
            final_correct += int(final_ok)
            matched_cases += int(bool(row.get("matched_skill_ids")))
            reflection_cases += int(row.get("reflection_result") is not None)
            if (not baseline_ok) and final_ok:
                improved_cases += 1
                improved_by_distance[distance_group] += 1
            if baseline_ok and (not final_ok):
                harmed_cases += 1
                harmed_by_distance[distance_group] += 1

    return {
        "run_name": path.parent.name,
        "source_file": str(path.relative_to(ROOT)),
        "total_cases": total,
        "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
        "final_accuracy": round(final_correct / total, 4) if total else 0.0,
        "accuracy_delta": round((final_correct - baseline_correct) / total, 4) if total else 0.0,
        "matched_skill_rate": round(matched_cases / total, 4) if total else 0.0,
        "reflection_rate": round(reflection_cases / total, 4) if total else 0.0,
        "improved_cases": improved_cases,
        "harmed_cases": harmed_cases,
        "improved_by_distance": dict(improved_by_distance),
        "harmed_by_distance": dict(harmed_by_distance),
    }


def build_ledger_rows(synth50: dict, real_pilot: dict, adaptation_summaries: dict[str, dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add_row(
        claim_id: str,
        evidence_tier: str,
        chapter: str,
        section: str,
        metric: str,
        value: str,
        source_file: str,
        figure_or_table: str,
        summary: str,
    ) -> None:
        rows.append(
            {
                "claim_id": claim_id,
                "evidence_tier": evidence_tier,
                "chapter": chapter,
                "section": section,
                "metric": metric,
                "value": value,
                "source_kind": "local_artifact",
                "source_file": source_file,
                "figure_or_table": figure_or_table,
                "summary": summary,
            }
        )

    mode = synth50["mode_summaries"]
    thesis_metrics = synth50["thesis_metrics"]
    add_row(
        "C1",
        "primary",
        "第5章",
        "主实验A",
        "hybrid_exact_match_accuracy",
        f"{mode['hybrid']['exact_match_accuracy']:.4f}",
        "experiments/dtpqa-integration/results/thesis_synth50_three_way_20260410.json",
        "表5-1 / 图5-3",
        "Synth50 clean subset 上，hybrid 获得最高整体准确率。",
    )
    add_row(
        "C2",
        "primary",
        "第5章",
        "主实验A",
        "hybrid_far_accuracy",
        f"{mode['hybrid']['far_accuracy']:.4f}",
        "experiments/dtpqa-integration/results/thesis_synth50_three_way_20260410.json",
        "表5-1 / 图5-3",
        "hybrid 在 far-distance 子集上明显优于 edge_only。",
    )
    add_row(
        "C3",
        "primary",
        "第5章",
        "主实验A",
        "hybrid_cloud_call_reduction_vs_cloud_only",
        f"{thesis_metrics['hybrid_cloud_call_reduction_vs_cloud_only']:.4f}",
        "experiments/dtpqa-integration/results/thesis_synth50_three_way_20260410.json",
        "图5-5",
        "hybrid 相对 cloud_only 明显减少云调用比例。",
    )
    add_row(
        "C4",
        "primary",
        "第5章",
        "主实验A",
        "hybrid_latency_reduction_vs_cloud_only",
        f"{thesis_metrics['hybrid_latency_reduction_vs_cloud_only']:.4f}",
        "experiments/dtpqa-integration/results/thesis_synth50_three_way_20260410.json",
        "图5-4 / 图5-5",
        "hybrid 在保持高准确率的同时显著降低平均时延。",
    )
    add_row(
        "C5",
        "secondary",
        "第5章",
        "局限性分析",
        "real_far_accuracy",
        f"{real_pilot['distance_group_summary']['far']['exact_match_accuracy']:.4f}",
        "experiments/dtpqa-integration/results/current_real_cat1_pilot_summary.json",
        "图5-6",
        "real pilot 在 far 组出现完全崩塌，体现外部效度问题。",
    )
    add_row(
        "C6",
        "secondary",
        "第5章",
        "局限性分析",
        "real_judge_score_mean",
        f"{real_pilot['judge_score_mean_judged_only']:.4f}",
        "experiments/dtpqa-integration/results/current_real_cat1_pilot_summary.json",
        "图5-6",
        "real pilot 的 judge score 与 exact match 同步下滑。",
    )

    for key, summary in adaptation_summaries.items():
        add_row(
            f"SA-{key}",
            "mechanism",
            "第5章",
            "主实验B",
            "final_vs_baseline_accuracy_delta",
            f"{summary['accuracy_delta']:.4f}",
            summary["source_file"],
            "图5-7 / 表5-2",
            f"{summary['run_name']} 用于独立 skill/adaptation 机制实验。",
        )

    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    synth50 = _load_json(SYNTH50_JSON)
    real_pilot = _load_json(REAL_PILOT_JSON)
    real_partial = _load_json(REAL_PARTIAL_JSON)
    adaptation_summaries = {name: summarize_adaptation_run(path) for name, path in ADAPTATION_RUNS.items()}

    ledger_rows = build_ledger_rows(synth50, real_pilot, adaptation_summaries)

    with (DATA_DIR / "evidence_ledger.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "synth50_summary": synth50,
                "real_pilot_summary": real_pilot,
                "real_partial_intervention_summary": real_partial,
                "adaptation_summaries": adaptation_summaries,
                "ledger_rows": ledger_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    with (DATA_DIR / "evidence_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id",
                "evidence_tier",
                "chapter",
                "section",
                "metric",
                "value",
                "source_kind",
                "source_file",
                "figure_or_table",
                "summary",
            ],
        )
        writer.writeheader()
        writer.writerows(ledger_rows)

    with (DATA_DIR / "adaptation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(adaptation_summaries, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {(DATA_DIR / 'evidence_ledger.json').relative_to(ROOT)}")
    print(f"Wrote {(DATA_DIR / 'evidence_ledger.csv').relative_to(ROOT)}")
    print(f"Wrote {(DATA_DIR / 'adaptation_summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
