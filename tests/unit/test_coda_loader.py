from pathlib import Path

from ad_cornercase.datasets.coda_lm import CodaLMDatasetLoader


def test_coda_loader_reads_fixture_dataset() -> None:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "coda_lm"
    loader = CodaLMDatasetLoader(root)
    cases = loader.load(split="Mini", task="region_perception")
    assert len(cases) == 2
    assert cases[0].case_id == "case-1"
    assert cases[0].ground_truth_answer == "Overturned_Truck"
    assert cases[0].weather_tags == ["fog", "night"]
    assert cases[0].metadata["benchmark"] == "coda_lm"
