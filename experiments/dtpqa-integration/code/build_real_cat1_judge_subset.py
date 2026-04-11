#!/usr/bin/env python3
"""Build a representative judge subset run from one or more real/category_1 runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.schemas.evaluation import CasePredictionRecord  # noqa: E402


def _pick_evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    raw = [round((total - 1) * index / (count - 1)) for index in range(count)]
    chosen: list[int] = []
    used: set[int] = set()
    for candidate in raw:
        index = candidate
        while index < total and index in used:
            index += 1
        if index >= total:
            index = total - 1
            while index in used:
                index -= 1
        chosen.append(index)
        used.add(index)
    return chosen


def _load_records(run_ids: list[str]) -> list[CasePredictionRecord]:
    records_by_case: dict[str, CasePredictionRecord] = {}
    for run_id in run_ids:
        predictions_path = ROOT / "data" / "artifacts" / run_id / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = CasePredictionRecord.model_validate_json(line)
                records_by_case[record.case_id] = record
    return list(records_by_case.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--per-group", type=int, default=4)
    args = parser.parse_args()

    records = _load_records(args.run_id)
    grouped: dict[str, list[CasePredictionRecord]] = defaultdict(list)
    for record in records:
        group = str(record.metadata.get("distance_group") or "unknown")
        record.judge_score = None
        grouped[group].append(record)

    selected: list[CasePredictionRecord] = []
    for group in ("near", "mid", "far", "unknown"):
        rows = sorted(grouped.get(group, []), key=lambda record: record.case_id)
        for index in _pick_evenly_spaced_indices(len(rows), args.per_group):
            selected.append(rows[index])

    run_dir = ROOT / "data" / "artifacts" / args.output_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    pretty_path = run_dir / "predictions.pretty.json"
    manifest_path = run_dir / "subset-manifest.json"

    predictions_path.write_text("\n".join(record.model_dump_json() for record in selected) + "\n", encoding="utf-8")
    pretty_path.write_text(
        json.dumps([record.model_dump(mode="json") for record in selected], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "source_run_ids": args.run_id,
                "per_group": args.per_group,
                "total_cases": len(selected),
                "case_ids": [record.case_id for record in selected],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
