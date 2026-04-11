import importlib.util
from pathlib import Path

from ad_cornercase.schemas.evaluation import CasePredictionRecord
from ad_cornercase.schemas.reflection import ReflectionResult
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


ROOT = Path(__file__).resolve().parents[2]


def _load_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(label: str, latency_ms: float) -> EdgePerceptionResult:
    return EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Are there any people in the image?", "answer": label}],
            "top_k_candidates": [{"label": "Pedestrian", "probability": 1.0}],
            "recommended_action": "slow_down",
            "latency_ms": latency_ms,
            "vision_tokens": 100,
        }
    )


def _write_run(tmp_path: Path, run_id: str, records: list[CasePredictionRecord]) -> None:
    run_dir = tmp_path / "data" / "artifacts" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def test_compare_real_cat1_runs_reports_protocol_metrics(tmp_path: Path) -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/compare_real_cat1_runs.py",
        "compare_real_cat1_runs_test",
    )
    module.ROOT = tmp_path

    baseline_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 15.0),
            final_result=_result("No", 15.0),
            metadata={"distance_group": "far", "distance_bin": "30m+"},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 20.0),
            final_result=_result("No", 20.0),
            metadata={"distance_group": "unknown", "distance_bin": "unknown"},
        ),
    ]
    intervention_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 15.0),
            final_result=_result("Yes, a pedestrian is visible.", 40.0),
            reflection_result=ReflectionResult(
                corrected_label="Yes",
                reflection_summary="A small pedestrian is visible.",
            ),
            metadata={"distance_group": "far", "distance_bin": "30m+"},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 20.0),
            final_result=_result("No", 25.0),
            metadata={"distance_group": "unknown", "distance_bin": "unknown"},
        ),
    ]
    _write_run(tmp_path, "baseline-run", baseline_records)
    _write_run(tmp_path, "intervention-run", intervention_records)

    summary = module.build_summary(["baseline-run"], ["intervention-run"])

    assert summary["shared_case_count"] == 2
    assert summary["baseline_exact_match_accuracy"] == 0.5
    assert summary["intervention_exact_match_accuracy"] == 1.0
    assert summary["baseline_positive_recall"] == 0.0
    assert summary["intervention_positive_recall"] == 1.0
    assert summary["baseline_negative_specificity"] == 1.0
    assert summary["intervention_negative_specificity"] == 1.0
    assert summary["baseline_skill_match_case_count"] == 0
    assert summary["intervention_skill_match_case_count"] == 0
    assert summary["baseline_reflection_case_count"] == 0
    assert summary["intervention_reflection_case_count"] == 1
    assert summary["latency_delta_ms"] == 15.0
    assert summary["improved_case_ids"] == ["pos-case"]


def test_verify_real_cat1_run_integrity_enforces_expectations(tmp_path: Path) -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/verify_real_cat1_run_integrity.py",
        "verify_real_cat1_run_integrity_test",
    )
    module.ROOT = tmp_path

    records = [
        CasePredictionRecord(
            case_id="clean-positive",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 12.0),
            final_result=_result("Yes, a pedestrian is visible.", 18.0),
            metadata={"distance_group": "far"},
        ),
        CasePredictionRecord(
            case_id="clean-negative",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 11.0),
            final_result=_result("No", 13.0),
            metadata={"distance_group": "unknown"},
        ),
    ]
    _write_run(tmp_path, "clean-run", records)
    clean_store = tmp_path / "skills-empty"

    summary = module.build_summary(["clean-run"], clean_store)
    failures = module.evaluate_expectations(
        summary,
        expect_no_matched_skills=True,
        expect_no_reflection=True,
        expect_empty_skill_store=True,
    )

    assert summary["all_matched_skill_ids_empty"] is True
    assert summary["all_reflection_results_null"] is True
    assert summary["skill_store_is_empty"] is True
    assert failures == []

    contaminated_records = [
        CasePredictionRecord(
            case_id="dirty-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("Yes", 16.0),
            matched_skill_ids=["skill-1"],
            reflection_result=ReflectionResult(
                corrected_label="Yes",
                reflection_summary="Pedestrian found after reflection.",
            ),
            metadata={"distance_group": "far"},
        )
    ]
    _write_run(tmp_path, "dirty-run", contaminated_records)
    dirty_store = tmp_path / "skills-dirty"
    dirty_store.mkdir(parents=True, exist_ok=True)
    (dirty_store / "skill.md").write_text("persisted skill", encoding="utf-8")

    dirty_summary = module.build_summary(["dirty-run"], dirty_store)
    dirty_failures = module.evaluate_expectations(
        dirty_summary,
        expect_no_matched_skills=True,
        expect_no_reflection=True,
        expect_empty_skill_store=True,
    )

    assert dirty_summary["matched_skill_case_ids"] == ["dirty-case"]
    assert dirty_summary["reflection_case_ids"] == ["dirty-case"]
    assert dirty_summary["skill_store_file_count"] == 1
    assert dirty_failures == [
        "matched_skill_ids present",
        "reflection_result present",
        "skill store not empty",
    ]


def test_compare_dtpqa_three_way_runs_builds_mode_summary(tmp_path: Path) -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/compare_dtpqa_three_way_runs.py",
        "compare_dtpqa_three_way_runs_test",
    )
    module.ROOT = tmp_path

    edge_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("No", 10.0),
            metadata={"distance_group": "far", "execution_mode": "edge_only", "pipeline_latency_ms": 10.0},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 11.0),
            final_result=_result("No", 11.0),
            metadata={"distance_group": "unknown", "execution_mode": "edge_only", "pipeline_latency_ms": 11.0},
        ),
    ]
    cloud_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("Yes", 20.0),
            final_result=_result("Yes", 20.0),
            metadata={"distance_group": "far", "execution_mode": "cloud_only", "pipeline_latency_ms": 20.0},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 21.0),
            final_result=_result("No", 21.0),
            metadata={"distance_group": "unknown", "execution_mode": "cloud_only", "pipeline_latency_ms": 21.0},
        ),
    ]
    hybrid_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("Yes", 35.0),
            reflection_result=ReflectionResult(
                corrected_label="Yes",
                reflection_summary="A small pedestrian is visible.",
            ),
            metadata={"distance_group": "far", "execution_mode": "hybrid", "pipeline_latency_ms": 35.0},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 12.0),
            final_result=_result("No", 30.0),
            metadata={"distance_group": "unknown", "execution_mode": "hybrid", "pipeline_latency_ms": 30.0},
        ),
    ]
    _write_run(tmp_path, "edge-run", edge_records)
    _write_run(tmp_path, "cloud-run", cloud_records)
    _write_run(tmp_path, "hybrid-run", hybrid_records)

    summary = module.build_summary(
        edge_run_ids=["edge-run"],
        cloud_run_ids=["cloud-run"],
        hybrid_run_ids=["hybrid-run"],
    )

    assert summary["shared_case_count"] == 2
    assert summary["mode_summaries"]["edge_only"]["exact_match_accuracy"] == 0.5
    assert summary["mode_summaries"]["cloud_only"]["exact_match_accuracy"] == 1.0
    assert summary["mode_summaries"]["hybrid"]["exact_match_accuracy"] == 1.0
    assert summary["mode_summaries"]["hybrid"]["reflection_rate"] == 0.5
    assert summary["pairwise_deltas"]["edge_vs_cloud_exact_match_delta"] == 0.5


def test_build_thesis_dtpqa_report_includes_paper_baselines(tmp_path: Path) -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/build_thesis_dtpqa_report.py",
        "build_thesis_dtpqa_report_test",
    )
    module.ROOT = tmp_path

    edge_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("No", 10.0),
            metadata={"distance_group": "far", "execution_mode": "edge_only", "pipeline_latency_ms": 10.0},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 9.0),
            final_result=_result("No", 9.0),
            metadata={"distance_group": "unknown", "execution_mode": "edge_only", "pipeline_latency_ms": 9.0},
        ),
    ]
    cloud_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("Yes", 30.0),
            final_result=_result("Yes", 30.0),
            metadata={"distance_group": "far", "execution_mode": "cloud_only", "pipeline_latency_ms": 30.0},
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 31.0),
            final_result=_result("No", 31.0),
            metadata={"distance_group": "unknown", "execution_mode": "cloud_only", "pipeline_latency_ms": 31.0},
        ),
    ]
    hybrid_records = [
        CasePredictionRecord(
            case_id="pos-case",
            question="Are there any people in the image?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("Yes", 18.0),
            reflection_result=ReflectionResult(
                corrected_label="Yes",
                reflection_summary="Direct cloud re-perception used for DTPQA category_1.",
            ),
            metadata={
                "distance_group": "far",
                "execution_mode": "hybrid",
                "pipeline_latency_ms": 18.0,
                "hybrid_strategy": "direct_cloud_perception",
            },
        ),
        CasePredictionRecord(
            case_id="neg-case",
            question="Are there any people in the image?",
            ground_truth_answer="No",
            baseline_result=_result("No", 11.0),
            final_result=_result("No", 12.0),
            metadata={
                "distance_group": "unknown",
                "execution_mode": "hybrid",
                "pipeline_latency_ms": 12.0,
                "hybrid_strategy": "edge_only_passthrough",
            },
        ),
    ]
    _write_run(tmp_path, "edge-run", edge_records)
    _write_run(tmp_path, "cloud-run", cloud_records)
    _write_run(tmp_path, "hybrid-run", hybrid_records)

    summary = module.build_summary(
        edge_run_ids=["edge-run"],
        cloud_run_ids=["cloud-run"],
        hybrid_run_ids=["hybrid-run"],
    )

    assert summary["shared_case_count"] == 2
    assert summary["mode_summaries"]["hybrid"]["baseline_exact_match_accuracy"] == 0.5
    assert summary["mode_summaries"]["hybrid"]["exact_match_accuracy"] == 1.0
    assert summary["thesis_metrics"]["hybrid_gain_capture_vs_cloud"] == 1.0
    assert summary["thesis_metrics"]["paper_best_small_vlm_dtpqa_avg_method"] == "Ovis2-2B"
    assert summary["thesis_metrics"]["paper_best_small_vlm_cat1_synth"] == 71.5
    assert any(row["method"] == "Ours (hybrid)" for row in summary["paper_cat1_synth_comparison_rows"])
