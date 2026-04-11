"""CODA-LM evaluation runner."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ad_cornercase.config import ProjectSettings
from ad_cornercase.evaluation.metrics import summarize_records
from ad_cornercase.evaluation.judge_runner import JudgeRunner
from ad_cornercase.schemas.evaluation import CasePredictionRecord

logger = logging.getLogger(__name__)


class CodaEvaluationRunner:
    def __init__(self, *, judge_runner: JudgeRunner, project_settings: ProjectSettings) -> None:
        self._judge_runner = judge_runner
        self._project_settings = project_settings

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
                logger.info("Evaluating case %s with judge model", record.case_id)
                record.judge_score = await self._judge_runner.score_record(record)
                logger.info("Judge completed for case %s: %.2f", record.case_id, record.judge_score)
        summary = summarize_records(run_dir.name, records, self._project_settings.judge_score_threshold)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
        report_path = run_dir / "report.md"
        report_path.write_text(self._render_report(summary), encoding="utf-8")
        predictions_path.write_text(
            "\n".join(record.model_dump_json() for record in records) + "\n",
            encoding="utf-8",
        )
        pretty_predictions_path.write_text(
            json.dumps([record.model_dump(mode="json") for record in records], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report_path

    def _render_report(self, summary) -> str:
        return "\n".join(
            [
                f"# Evaluation Report: {summary.run_id}",
                "",
                f"- Total cases: {summary.total_cases}",
                f"- Judge score mean: {summary.judge_score_mean:.2f}",
                f"- Regional triplet recall: {summary.regional_triplet_recall:.2f}",
                f"- Skill success rate: {summary.skill_success_rate:.2f}",
                f"- Latency delta ms: {summary.latency_delta_ms:.2f}",
                f"- Vision token delta: {summary.vision_token_delta:.2f}",
                "",
            ]
        )
