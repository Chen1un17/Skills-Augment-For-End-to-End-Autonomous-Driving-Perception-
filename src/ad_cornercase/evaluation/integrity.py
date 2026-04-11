"""Integrity checks for benchmark plans and run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ad_cornercase.datasets.dtpqa import DTPQADatasetLoader


def preflight_dtpqa_plan_cases(
    selected_cases: list[dict[str, object]],
    *,
    dtpqa_root: Path,
    subset: str = "real",
    question_type: str = "category_1",
) -> list[dict[str, object]]:
    """Resolve each planned offset and verify it still points to the expected local image."""

    loader = DTPQADatasetLoader(dtpqa_root)
    issues: list[dict[str, object]] = []
    for item in selected_cases:
        offset = int(item["offset"])
        expected_case_id = str(item.get("case_id") or "")
        try:
            resolved_case = loader.load(
                subset=subset,
                question_type=question_type,
                limit=1,
                offset=offset,
            )[0]
        except Exception as exc:  # pragma: no cover - CLI diagnostic path
            issues.append(
                {
                    "offset": offset,
                    "expected_case_id": expected_case_id,
                    "issue": "load_error",
                    "detail": str(exc),
                }
            )
            continue

        if expected_case_id and resolved_case.case_id != expected_case_id:
            issues.append(
                {
                    "offset": offset,
                    "expected_case_id": expected_case_id,
                    "resolved_case_id": resolved_case.case_id,
                    "issue": "case_id_mismatch",
                }
            )
        if not resolved_case.image_path.exists():
            issues.append(
                {
                    "offset": offset,
                    "expected_case_id": expected_case_id,
                    "resolved_case_id": resolved_case.case_id,
                    "issue": "missing_image",
                    "image_path": str(resolved_case.image_path),
                }
            )
    return issues


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def summarize_plan_execution(selected_cases: list[dict[str, object]], artifact_dir: Path) -> dict[str, object]:
    """Summarize whether a planned run completed every expected case successfully."""

    expected_case_ids = [str(item["case_id"]) for item in selected_cases]
    expected_case_id_set = set(expected_case_ids)

    status_rows = load_jsonl_rows(artifact_dir / "batch-status.jsonl")
    latest_status_by_case: dict[str, dict[str, Any]] = {}
    for row in status_rows:
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id:
            latest_status_by_case[case_id] = row

    prediction_rows = load_jsonl_rows(artifact_dir / "predictions.jsonl")
    prediction_case_ids = {
        case_id
        for row in prediction_rows
        if isinstance((case_id := row.get("case_id")), str) and case_id
    }

    missing_status_case_ids = sorted(expected_case_id_set - set(latest_status_by_case))
    failed_case_ids = sorted(
        case_id
        for case_id, row in latest_status_by_case.items()
        if case_id in expected_case_id_set and int(row.get("returncode", 1)) != 0
    )
    missing_prediction_case_ids = sorted(expected_case_id_set - prediction_case_ids)
    unexpected_prediction_case_ids = sorted(prediction_case_ids - expected_case_id_set)

    return {
        "expected_case_count": len(expected_case_ids),
        "status_case_count": len(latest_status_by_case),
        "prediction_case_count": len(prediction_case_ids),
        "missing_status_case_ids": missing_status_case_ids,
        "failed_case_ids": failed_case_ids,
        "missing_prediction_case_ids": missing_prediction_case_ids,
        "unexpected_prediction_case_ids": unexpected_prediction_case_ids,
        "is_complete": not (
            missing_status_case_ids
            or failed_case_ids
            or missing_prediction_case_ids
            or unexpected_prediction_case_ids
        ),
    }
