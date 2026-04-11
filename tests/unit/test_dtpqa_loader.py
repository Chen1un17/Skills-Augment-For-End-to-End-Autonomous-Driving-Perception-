from pathlib import Path

from ad_cornercase.datasets.dtpqa import DTPQADatasetLoader


def test_dtpqa_loader_reads_direct_and_nested_annotations() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "dtpqa"
    loader = DTPQADatasetLoader(root)

    cases = loader.load(subset="real")

    assert len(cases) == 2
    cases_by_id = {case.case_id: case for case in cases}

    far_case = cases_by_id["dtpqa-real-1"]
    assert far_case.ground_truth_answer == "Overturned truck"
    assert "Options:" in far_case.question
    assert far_case.metadata["benchmark"] == "dtpqa"
    assert far_case.metadata["distance_bin"] == "30m+"
    assert far_case.metadata["distance_group"] == "far"
    assert far_case.image_path.exists()

    near_case = cases_by_id["dtpqa-real-2"]
    assert near_case.ground_truth_answer == "Pedestrian"
    assert near_case.metadata["distance_bin"] == "10-20m"
    assert near_case.metadata["distance_group"] == "near"
    assert near_case.metadata["question_type"] == "road_user_focus"


def test_dtpqa_loader_filters_question_type() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "dtpqa"
    loader = DTPQADatasetLoader(root)

    cases = loader.load(subset="real", question_type="road_user_focus")

    assert [case.case_id for case in cases] == ["dtpqa-real-2"]


def test_dtpqa_loader_supports_top_level_subset_and_category_groups() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "dtpqa_actual"
    loader = DTPQADatasetLoader(root)

    cases = loader.load(subset="all")

    assert len(cases) == 2
    synth_case = next(case for case in cases if case.metadata["subset"] == "synth")
    real_case = next(case for case in cases if case.metadata["subset"] == "real")
    assert synth_case.metadata["question_type"] == "category_1"
    assert synth_case.metadata["distance_group"] == "mid"
    assert real_case.metadata["question_type"] == "category_2"
    assert real_case.metadata["distance_group"] == "far"


def test_dtpqa_loader_supports_offset_after_filtering() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "dtpqa"
    loader = DTPQADatasetLoader(root)

    cases = loader.load(subset="real", offset=1, limit=1)

    assert [case.case_id for case in cases] == ["dtpqa-real-1"]
