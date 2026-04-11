import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "dtpqa-integration"
    / "code"
    / "run_real_cat1_plan.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_real_cat1_plan_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "selections": {
                    "custom": [
                        {
                            "offset": 10,
                            "case_id": "case-10",
                            "distance_group": "far",
                            "ground_truth_answer": "Yes",
                        },
                        {
                            "offset": 11,
                            "case_id": "case-11",
                            "distance_group": "unknown",
                            "ground_truth_answer": "No",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def test_run_real_cat1_plan_retries_missing_cases_across_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    module.ROOT = tmp_path
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)

    outcomes = {10: [1, 0], 11: [0]}

    def fake_run(command, cwd, env, check):  # noqa: ANN001
        del cwd, env, check
        offset = int(command[command.index("--offset") + 1])
        run_id = command[command.index("--run-id") + 1]
        case_id = f"case-{offset}"
        returncode = outcomes[offset].pop(0)
        if returncode == 0:
            artifact_dir = tmp_path / "data" / "artifacts" / run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = artifact_dir / "predictions.jsonl"
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case_id": case_id}) + "\n")
        return subprocess.CompletedProcess(command, returncode=returncode)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_cat1_plan.py",
            "--plan",
            str(plan_path),
            "--run-id",
            "run-test",
            "--max-passes",
            "2",
            "--retry-sleep-seconds",
            "0",
        ],
    )

    module.main()

    predictions_path = tmp_path / "data" / "artifacts" / "run-test" / "predictions.jsonl"
    batch_status_path = tmp_path / "data" / "artifacts" / "run-test" / "batch-status.jsonl"
    prediction_lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    status_lines = batch_status_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(prediction_lines) == 2
    assert len(status_lines) == 3
    assert '"pass_index": 2' in status_lines[-1]


def test_run_real_cat1_plan_exits_nonzero_when_cases_stay_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    module.ROOT = tmp_path
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)

    outcomes = {10: [1, 1], 11: [0]}

    def fake_run(command, cwd, env, check):  # noqa: ANN001
        del cwd, env, check
        offset = int(command[command.index("--offset") + 1])
        run_id = command[command.index("--run-id") + 1]
        case_id = f"case-{offset}"
        returncode = outcomes[offset].pop(0)
        if returncode == 0:
            artifact_dir = tmp_path / "data" / "artifacts" / run_id
            artifact_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = artifact_dir / "predictions.jsonl"
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case_id": case_id}) + "\n")
        return subprocess.CompletedProcess(command, returncode=returncode)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_cat1_plan.py",
            "--plan",
            str(plan_path),
            "--run-id",
            "run-test",
            "--max-passes",
            "2",
            "--retry-sleep-seconds",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
