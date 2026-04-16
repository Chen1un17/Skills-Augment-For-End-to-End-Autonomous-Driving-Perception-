import importlib.util
import json
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


def _result(answer: str, latency_ms: float, *, vision_tokens: int = 100) -> EdgePerceptionResult:
    return EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Question?", "answer": answer}],
            "top_k_candidates": [{"label": answer, "probability": 1.0}],
            "recommended_action": "slow_down",
            "latency_ms": latency_ms,
            "vision_tokens": vision_tokens,
        }
    )


def _write_run(tmp_path: Path, run_id: str, records: list[CasePredictionRecord]) -> None:
    run_dir = tmp_path / "data" / "artifacts" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def test_build_dtpqa_synth_balanced_plan_creates_smoke_and_refinement_splits() -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/build_dtpqa_synth_balanced_plan.py",
        "build_dtpqa_synth_balanced_plan_test",
    )

    plan = module.build_plan(
        dtpqa_root=ROOT / "data" / "dtpqa",
        quotas={"category_1": 4, "category_2": 4},
        adaptation_size=2,
        smoke_per_category=1,
    )

    assert plan["subset"] == "synth"
    assert plan["total_cases"] == 8
    assert len(plan["cases"]) == 8
    assert len(plan["smoke_case_ids"]) == 2
    assert len({item["case_id"] for item in plan["cases"]}) == 8
    assert len(plan["refinement_splits"]["category_1"]["adaptation_case_ids"]) == 2
    assert len(plan["refinement_splits"]["category_1"]["holdout_case_ids"]) == 2
    assert all(Path(item["image_path"]).exists() for item in plan["cases"])


def test_run_dtpqa_plan_selection_filters_cases_from_plan() -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/run_dtpqa_plan.py",
        "run_dtpqa_plan_test",
    )

    plan = {
        "smoke_case_ids": ["cat1-a", "cat2-a"],
        "refinement_splits": {
            "category_1": {
                "adaptation_case_ids": ["cat1-a"],
                "holdout_case_ids": ["cat1-b"],
            },
            "category_2": {
                "adaptation_case_ids": ["cat2-a"],
                "holdout_case_ids": ["cat2-b"],
            },
        },
        "cases": [
            {"case_id": "cat1-a", "question_type": "category_1", "subset": "synth", "offset": 0, "distance_group": "far"},
            {"case_id": "cat1-b", "question_type": "category_1", "subset": "synth", "offset": 1, "distance_group": "near"},
            {"case_id": "cat2-a", "question_type": "category_2", "subset": "synth", "offset": 0, "distance_group": "far"},
            {"case_id": "cat2-b", "question_type": "category_2", "subset": "synth", "offset": 1, "distance_group": "near"},
        ],
    }

    smoke_cases = module._select_cases(plan, selection="smoke", question_type=None)  # noqa: SLF001
    holdout_cases = module._select_cases(plan, selection="refinement_holdout", question_type="category_2")  # noqa: SLF001

    assert [item["case_id"] for item in smoke_cases] == ["cat1-a", "cat2-a"]
    assert [item["case_id"] for item in holdout_cases] == ["cat2-b"]


def test_build_dtpqa_synth500_three_way_report_adds_special_metrics(tmp_path: Path) -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/build_dtpqa_synth500_three_way_report.py",
        "build_dtpqa_synth500_three_way_report_test",
    )
    module.ROOT = tmp_path

    plan = {
        "cases": [
            {"case_id": "cat1-pos", "question_type": "category_1"},
            {"case_id": "cat3-count", "question_type": "category_3"},
        ]
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    edge_records = [
        CasePredictionRecord(
            case_id="cat1-pos",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("No", 10.0),
            metadata={"question_type": "category_1", "distance_group": "far"},
        ),
        CasePredictionRecord(
            case_id="cat3-count",
            question="How many pedestrians are crossing the road?",
            ground_truth_answer="Two",
            baseline_result=_result("Three", 12.0),
            final_result=_result("Three", 12.0),
            metadata={"question_type": "category_3", "distance_group": "near"},
        ),
    ]
    cloud_records = [
        CasePredictionRecord(
            case_id="cat1-pos",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="Yes",
            baseline_result=_result("Yes", 25.0),
            final_result=_result("Yes", 25.0),
            metadata={"question_type": "category_1", "distance_group": "far"},
        ),
        CasePredictionRecord(
            case_id="cat3-count",
            question="How many pedestrians are crossing the road?",
            ground_truth_answer="Two",
            baseline_result=_result("Two", 26.0),
            final_result=_result("Two", 26.0),
            metadata={"question_type": "category_3", "distance_group": "near"},
        ),
    ]
    hybrid_records = [
        CasePredictionRecord(
            case_id="cat1-pos",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 10.0),
            final_result=_result("Yes", 30.0),
            reflection_result=ReflectionResult(corrected_label="Yes", reflection_summary="Pedestrian visible."),
            metadata={"question_type": "category_1", "distance_group": "far", "hybrid_strategy": "direct_cloud_perception"},
        ),
        CasePredictionRecord(
            case_id="cat3-count",
            question="How many pedestrians are crossing the road?",
            ground_truth_answer="Two",
            baseline_result=_result("Three", 12.0),
            final_result=_result("Two", 32.0),
            matched_skill_ids=["count-skill"],
            metadata={"question_type": "category_3", "distance_group": "near", "hybrid_strategy": "skill_augmented_edge"},
        ),
    ]
    _write_run(tmp_path, "edge-run", edge_records)
    _write_run(tmp_path, "cloud-run", cloud_records)
    _write_run(tmp_path, "hybrid-run", hybrid_records)

    summary = module.build_summary(
        plan_path=plan_path,
        edge_run_ids=["edge-run"],
        cloud_run_ids=["cloud-run"],
        hybrid_run_ids=["hybrid-run"],
    )

    assert summary["shared_case_count"] == 2
    assert summary["question_type_summary"]["category_1"]["hybrid"]["yes_recall"] == 1.0
    assert summary["question_type_summary"]["category_3"]["edge_only"]["count_error_distribution"]["off_by_1"] == 1
    assert summary["routing_summary"]["hybrid_strategy_counts"]["direct_cloud_perception"] == 1


def test_build_dtpqa_skill_refinement_report_summarizes_holdout_and_skill_store(tmp_path: Path) -> None:
    module = _load_module(
        "experiments/dtpqa-integration/code/build_dtpqa_skill_refinement_report.py",
        "build_dtpqa_skill_refinement_report_test",
    )
    module.ROOT = tmp_path

    plan = {
        "refinement_splits": {
            "category_1": {
                "adaptation_case_ids": ["adapt-1"],
                "holdout_case_ids": ["hold-1"],
            }
        }
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    records = [
        CasePredictionRecord(
            case_id="adapt-1",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 20.0, vision_tokens=120),
            final_result=_result("Yes", 18.0, vision_tokens=90),
            matched_skill_ids=["skill-a"],
            judge_score=88.0,
            metadata={"question_type": "category_1", "distance_group": "far"},
        ),
        CasePredictionRecord(
            case_id="hold-1",
            question="Are there any pedestrians crossing the road?",
            ground_truth_answer="Yes",
            baseline_result=_result("No", 22.0, vision_tokens=120),
            final_result=_result("Yes", 16.0, vision_tokens=80),
            matched_skill_ids=["skill-a"],
            judge_score=91.0,
            metadata={"question_type": "category_1", "distance_group": "far"},
        ),
    ]
    _write_run(tmp_path, "refine-cat1", records)

    skill_store_dir = tmp_path / "skills" / "category_1"
    skill_store_dir.mkdir(parents=True, exist_ok=True)
    (skill_store_dir / "skill-a").write_text("persisted", encoding="utf-8")

    manifest = {
        "question_types": {
            "category_1": {
                "run_id": "refine-cat1",
                "skill_store_dir": str(skill_store_dir),
            }
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = module.build_summary(plan_path=plan_path, manifest_path=manifest_path)

    assert summary["holdout_summary"]["skill_match_rate"] == 1.0
    assert summary["holdout_summary"]["skill_success_rate"] == 1.0
    assert summary["skill_store_summary"]["category_1"]["file_count"] == 1
