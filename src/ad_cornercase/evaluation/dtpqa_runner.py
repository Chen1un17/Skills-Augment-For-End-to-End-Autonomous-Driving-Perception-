"""DTPQA evaluation runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ad_cornercase.config import ProjectSettings
from ad_cornercase.evaluation.metrics import summarize_dtpqa_records
from ad_cornercase.evaluation.judge_runner import JudgeRunner
from ad_cornercase.schemas.evaluation import CasePredictionRecord, EvaluationSummary

logger = logging.getLogger(__name__)


class DTPQAEvaluationRunner:
    def __init__(self, *, judge_runner: JudgeRunner, project_settings: ProjectSettings) -> None:
        self._judge_runner = judge_runner
        self._project_settings = project_settings

    def _persist_predictions(
        self,
        *,
        predictions_path: Path,
        pretty_predictions_path: Path,
        records: list[CasePredictionRecord],
    ) -> None:
        predictions_path.write_text(
            "\n".join(record.model_dump_json() for record in records) + "\n",
            encoding="utf-8",
        )
        pretty_predictions_path.write_text(
            json.dumps([record.model_dump(mode="json") for record in records], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    async def evaluate_run(self, run_dir: Path) -> Path:
        predictions_path = run_dir / "predictions.jsonl"
        pretty_predictions_path = run_dir / "predictions.pretty.json"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        records: list[CasePredictionRecord] = []
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(CasePredictionRecord.model_validate_json(line))
        for record in records:
            if record.judge_score is None:
                logger.info("Evaluating DTPQA case %s with judge model", record.case_id)
                record.judge_score = await self._judge_runner.score_record(record)
                self._persist_predictions(
                    predictions_path=predictions_path,
                    pretty_predictions_path=pretty_predictions_path,
                    records=records,
                )
        summary = summarize_dtpqa_records(run_dir.name, records, self._project_settings.judge_score_threshold)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
        report_path = run_dir / "report.md"
        report_path.write_text(self._render_report(summary), encoding="utf-8")
        self._persist_predictions(
            predictions_path=predictions_path,
            pretty_predictions_path=pretty_predictions_path,
            records=records,
        )
        return report_path

    def _render_report(self, summary: EvaluationSummary) -> str:
        lines = [
            f"# DTPQA Evaluation Report: {summary.run_id}",
            "",
            f"- Total cases: {summary.total_cases}",
            f"- Judge score mean: {summary.judge_score_mean:.2f}",
            f"- Exact match accuracy: {(summary.exact_match_accuracy or 0.0):.2f}",
            f"- Skill success rate: {summary.skill_success_rate:.2f}",
            f"- Latency delta ms: {summary.latency_delta_ms:.2f}",
            f"- Vision token delta: {summary.vision_token_delta:.2f}",
            "",
            "## Distance Bin Accuracy",
            "",
        ]
        if summary.distance_bin_accuracy:
            for key in sorted(summary.distance_bin_accuracy):
                count = summary.distance_bin_counts.get(key, 0)
                lines.append(f"- {key}: {summary.distance_bin_accuracy[key]:.2f} ({count} cases)")
        else:
            lines.append("- No distance-bin metadata available.")
        lines.extend(["", "## Distance Group Accuracy", ""])
        if summary.distance_group_accuracy:
            for key in sorted(summary.distance_group_accuracy):
                judge_mean = summary.distance_group_judge_score_mean.get(key, 0.0)
                lines.append(
                    f"- {key}: accuracy={summary.distance_group_accuracy[key]:.2f}, "
                    f"judge_score_mean={judge_mean:.2f}"
                )
        else:
            lines.append("- No distance-group metadata available.")
        lines.append("")
        return "\n".join(lines)
