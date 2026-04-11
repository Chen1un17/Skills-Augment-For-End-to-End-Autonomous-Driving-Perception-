from pathlib import Path

from ad_cornercase.config import ProjectSettings, RuntimeSettings
from ad_cornercase.edge.replay import ReplayOrchestrator
from ad_cornercase.schemas.anomaly import AnomalyCase
from ad_cornercase.schemas.scene_graph import EdgePerceptionResult


def _runtime_settings(tmp_path: Path) -> RuntimeSettings:
    prompts_dir = Path(__file__).resolve().parents[2] / "configs" / "prompts"
    return RuntimeSettings(
        openai_api_key="test-key",
        openai_base_url="http://test.invalid/v1",
        edge_model="fake-edge",
        cloud_model="fake-cloud",
        judge_model="fake-judge",
        embedding_model="fake-embedding",
        coda_lm_root=tmp_path / "coda_lm",
        artifacts_dir=tmp_path / "artifacts",
        skill_store_dir=tmp_path / "skills",
        prompts_dir=prompts_dir,
        settings_path=Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        uncertainty_entropy_threshold=1.0,
        log_level="ERROR",
    )


def _orchestrator(tmp_path: Path) -> ReplayOrchestrator:
    return ReplayOrchestrator(
        edge_agent=None,  # type: ignore[arg-type]
        runtime_settings=_runtime_settings(tmp_path),
        project_settings=ProjectSettings(),
    )


def _dtpqa_people_case(tmp_path: Path, **metadata) -> AnomalyCase:
    return AnomalyCase(
        case_id="annotations-test",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={"benchmark": "dtpqa", "question_type": "category_1", **metadata},
    )


def _no_answer_with_pedestrian_secondary() -> EdgePerceptionResult:
    return EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Are there any people in the image?", "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Distant_Pedestrian", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "entropy": 0.246,
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )


def test_replay_orchestrator_reflects_on_dtpqa_people_false_negative_pattern(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)

    should_reflect = orchestrator._should_reflect(  # noqa: SLF001
        _dtpqa_people_case(tmp_path),
        _no_answer_with_pedestrian_secondary(),
    )

    assert should_reflect is True


def test_replay_orchestrator_reflects_on_category1_no_answer_even_without_person_like_secondary_candidate(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator(tmp_path)
    result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Are there any people in the image?", "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Empty_Street", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "entropy": 0.246,
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    should_reflect = orchestrator._should_reflect(_dtpqa_people_case(tmp_path), result)  # noqa: SLF001

    assert should_reflect is False


def test_replay_orchestrator_reflects_on_category1_no_answer_with_person_evidence_in_scene_graph(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator(tmp_path)
    result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Are there any people in the image?", "answer": "No"}],
            "general_perception": {
                "vulnerable_road_users": [
                    {
                        "description": "Pedestrian near the right curb.",
                        "explanation": "A person is visible near the travel lane.",
                    }
                ]
            },
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Empty_Street", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "entropy": 0.246,
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    should_reflect = orchestrator._should_reflect(_dtpqa_people_case(tmp_path), result)  # noqa: SLF001

    assert should_reflect is True


def test_replay_orchestrator_does_not_reflect_non_category1_case_without_person_like_secondary_candidate(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator(tmp_path)
    result = EdgePerceptionResult.model_validate(
        {
            "qa_report": [{"question": "Are there any people in the image?", "answer": "No"}],
            "top_k_candidates": [
                {"label": "Clear_Roadway", "probability": 0.95},
                {"label": "Empty_Street", "probability": 0.03},
                {"label": "Critical_Unknown_Obstacle", "probability": 0.02},
            ],
            "entropy": 0.246,
            "recommended_action": "maintain_current_speed_and_monitor",
        }
    )

    should_reflect = orchestrator._should_reflect(  # noqa: SLF001
        _dtpqa_people_case(tmp_path, question_type="category_2"),
        result,
    )

    assert should_reflect is False


def test_replay_orchestrator_does_not_apply_dtpqa_rule_to_non_dtpqa_cases(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    case = AnomalyCase(
        case_id="case-1",
        frame_id="frame-001",
        image_path=tmp_path / "scene.png",
        question="Are there any people in the image?",
        ground_truth_answer="Yes",
        metadata={},
    )

    should_reflect = orchestrator._should_reflect(case, _no_answer_with_pedestrian_secondary())  # noqa: SLF001

    assert should_reflect is False


def test_replay_orchestrator_can_disable_dtpqa_people_trigger_via_runtime_setting(tmp_path: Path) -> None:
    runtime_settings = _runtime_settings(tmp_path)
    runtime_settings.enable_dtpqa_people_reflection_trigger = False
    orchestrator = ReplayOrchestrator(
        edge_agent=None,  # type: ignore[arg-type]
        runtime_settings=runtime_settings,
        project_settings=ProjectSettings(),
    )

    should_reflect = orchestrator._should_reflect(  # noqa: SLF001
        _dtpqa_people_case(tmp_path),
        _no_answer_with_pedestrian_secondary(),
    )

    assert should_reflect is False
