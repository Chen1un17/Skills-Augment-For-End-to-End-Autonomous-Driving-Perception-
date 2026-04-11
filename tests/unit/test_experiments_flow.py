import json
from pathlib import Path

from ad_cornercase.experiments.batch_runner import LargeScaleBatchRunner
from ad_cornercase.experiments.config import ExperimentConfig
from ad_cornercase.experiments.monitor import ExperimentMonitor
from ad_cornercase.experiments.runner import ExperimentRunner
from ad_cornercase.schemas.evaluation import CasePredictionRecord
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


def _result(label: str, latency_ms: float) -> EdgePerceptionResult:
    return EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Are there any people in the image?", "answer": label}],
            "top_k_candidates": [{"label": label, "probability": 1.0}],
            "recommended_action": "slow_down",
            "latency_ms": latency_ms,
            "vision_tokens": 100,
        }
    )


def test_experiment_runner_resume_preserves_skill_store(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    config = ExperimentConfig(
        name="resume-test",
        skill_store_dir=tmp_path / "skills",
        artifacts_dir=tmp_path / "artifacts",
        clean_skill_store=True,
        enable_judge=False,
    )
    skill_file = config.skill_store_dir / "persisted-skill.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text("keep me", encoding="utf-8")

    status_path = config.artifacts_dir / config.run_id / "experiment_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "completed_cases": 1,
                "failed_cases": 0,
                "current_offset": 1,
            }
        ),
        encoding="utf-8",
    )

    runner = ExperimentRunner(config)
    monkeypatch.setattr(runner, "_estimate_total_cases", lambda: 1)
    monkeypatch.setattr(runner, "_run_judge_evaluation", lambda: True)

    status = runner.run(resume=True)

    assert status.state == "completed"
    assert skill_file.exists()


def test_experiment_runner_build_env_sets_reflection_trigger_by_execution_mode(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="env-test",
        skill_store_dir=tmp_path / "skills",
        artifacts_dir=tmp_path / "artifacts",
    )
    runner = ExperimentRunner(config)

    config.execution_mode = "edge_only"
    config.enable_reflection = False
    assert runner._build_env()["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] == "0"  # noqa: SLF001

    config.execution_mode = "cloud_only"
    config.enable_reflection = False
    assert runner._build_env()["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] == "0"  # noqa: SLF001

    config.execution_mode = "hybrid"
    config.enable_reflection = True
    config.enable_dtpqa_people_reflection = True
    assert runner._build_env()["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] == "1"  # noqa: SLF001


def test_batch_runner_real_experiments_use_real_subset_and_three_modes() -> None:
    runner = LargeScaleBatchRunner()

    experiments = runner.create_dtpqa_real_experiments(sample_limits=[52])

    assert {experiment.config.execution_mode for experiment in experiments} == {"edge_only", "cloud_only", "hybrid"}
    assert {experiment.config.dataset.subset for experiment in experiments} == {"real"}


def test_experiment_monitor_reads_predictions_and_pipeline_latency(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-monitor"
    run_dir.mkdir(parents=True, exist_ok=True)
    record = CasePredictionRecord(
        case_id="case-1",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        baseline_result=_result("No", 10.0),
        final_result=_result("Yes", 25.0),
        metadata={
            "distance_group": "far",
            "execution_mode": "cloud_only",
            "pipeline_latency_ms": 123.0,
        },
    )
    (run_dir / "predictions.jsonl").write_text(record.model_dump_json() + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"judge_score_mean": 0.0}), encoding="utf-8")

    metrics = ExperimentMonitor(tmp_path).analyze("run-monitor")

    assert metrics.total_cases == 1
    assert metrics.execution_mode == "cloud_only"
    assert metrics.mean_latency_ms == 123.0
    assert metrics.distance_accuracy["far"] == 1.0
