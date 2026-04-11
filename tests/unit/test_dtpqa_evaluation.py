import json
from pathlib import Path

import pytest

from ad_cornercase.config import ProjectSettings
from ad_cornercase.evaluation.dtpqa_runner import DTPQAEvaluationRunner
from ad_cornercase.evaluation.judge_runner import JudgeRunner
from ad_cornercase.evaluation.metrics import exact_match
from ad_cornercase.prompts.renderer import PromptRenderer
from ad_cornercase.providers.judge import HeuristicJudgeProvider
from ad_cornercase.schemas.evaluation import CasePredictionRecord
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


def _result(label: str, action: str, latency_ms: float, vision_tokens: int) -> EdgePerceptionResult:
    return EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "What is the correct answer?", "answer": label}],
            "top_k_candidates": [{"label": label, "probability": 1.0}],
            "recommended_action": action,
            "triplets": [{"subject": "ego_vehicle", "relation": "should", "object": action}],
            "latency_ms": latency_ms,
            "vision_tokens": vision_tokens,
        }
    )


@pytest.mark.asyncio
async def test_dtpqa_runner_summarizes_distance_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-dtpqa"
    run_dir.mkdir()
    predictions_path = run_dir / "predictions.jsonl"
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"

    records = [
        CasePredictionRecord(
            case_id="far-correct",
            question="What is the main hazard ahead?\nOptions:\nA. Overturned truck\nB. Traffic sign",
            ground_truth_answer="Overturned truck",
            baseline_result=_result("Critical_Unknown_Obstacle", "slow_down", 20.0, 200),
            final_result=_result("Overturned truck", "brake", 10.0, 150),
            judge_score=100.0,
            metadata={
                "benchmark": "dtpqa",
                "distance_bin": "30m+",
                "distance_group": "far",
                "answer_options": ["Overturned truck", "Traffic sign"],
            },
        ),
        CasePredictionRecord(
            case_id="near-wrong",
            question="Which road user should ego monitor most carefully?\nOptions:\nA. Pedestrian\nB. Barrier",
            ground_truth_answer="Pedestrian",
            baseline_result=_result("Pedestrian", "slow_down", 18.0, 180),
            final_result=_result("Barrier", "maintain_lane", 12.0, 140),
            judge_score=40.0,
            metadata={
                "benchmark": "dtpqa",
                "distance_bin": "10-20m",
                "distance_group": "near",
                "answer_options": ["Pedestrian", "Barrier"],
            },
        ),
    ]
    predictions_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )

    runner = DTPQAEvaluationRunner(
        judge_runner=JudgeRunner(
            judge_provider=HeuristicJudgeProvider(),
            prompt_renderer=PromptRenderer(prompts_dir),
        ),
        project_settings=ProjectSettings(),
    )
    report_path = await runner.evaluate_run(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert report_path.exists()
    assert metrics["total_cases"] == 2
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["distance_bin_accuracy"]["30m+"] == 1.0
    assert metrics["distance_bin_accuracy"]["10-20m"] == 0.0
    assert metrics["distance_group_accuracy"]["far"] == 1.0
    assert metrics["distance_group_accuracy"]["near"] == 0.0


class _FlakyJudgeRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def score_record(self, record: CasePredictionRecord) -> float:
        self.calls += 1
        if self.calls == 1:
            return 88.0
        raise RuntimeError(f"judge failed for {record.case_id}")


@pytest.mark.asyncio
async def test_dtpqa_runner_persists_partial_judge_progress_on_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-dtpqa-flaky"
    run_dir.mkdir()
    predictions_path = run_dir / "predictions.jsonl"
    records = [
        CasePredictionRecord(
            case_id="first",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="Yes",
            baseline_result=_result("Yes", "slow_down", 10.0, 100),
            final_result=_result("Yes, one pedestrian is crossing.", "slow_down_and_yield", 8.0, 90),
            metadata={"benchmark": "dtpqa", "distance_group": "near"},
        ),
        CasePredictionRecord(
            case_id="second",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="No",
            baseline_result=_result("No", "maintain_speed", 11.0, 110),
            final_result=_result("No, there are no pedestrians crossing the road.", "maintain_speed", 9.0, 95),
            metadata={"benchmark": "dtpqa", "distance_group": "far"},
        ),
    ]
    predictions_path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )
    runner = DTPQAEvaluationRunner(
        judge_runner=_FlakyJudgeRunner(),  # type: ignore[arg-type]
        project_settings=ProjectSettings(),
    )

    with pytest.raises(RuntimeError):
        await runner.evaluate_run(run_dir)

    persisted = [
        CasePredictionRecord.model_validate_json(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert persisted[0].judge_score == 88.0
    assert persisted[1].judge_score is None


def test_exact_match_accepts_yes_no_answers_with_short_explanations() -> None:
    record = CasePredictionRecord(
        case_id="yes-with-rationale",
        question="Are there any pedestrians crossing the road?",
        ground_truth_answer="Yes",
        baseline_result=_result("Yes", "slow_down", 10.0, 100),
        final_result=EdgePerceptionResult.model_validate(
            {
                "qa_report": [
                    {
                        "question": "Are there any pedestrians crossing the road?",
                        "answer": "Yes, one pedestrian is crossing the road from right to left.",
                    }
                ],
                "top_k_candidates": [{"label": "Yes", "probability": 1.0}],
                "recommended_action": "slow_down_and_yield",
            }
        ),
        metadata={"benchmark": "dtpqa", "distance_group": "near"},
    )

    assert exact_match(record) == 1.0
