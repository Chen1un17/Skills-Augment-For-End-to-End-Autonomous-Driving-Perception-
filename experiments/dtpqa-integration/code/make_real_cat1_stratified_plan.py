#!/usr/bin/env python3
"""Build a reproducible stratified offset plan for real/category_1 DTPQA replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.datasets.dtpqa import DTPQADatasetLoader  # noqa: E402


@dataclass
class PlannedCase:
    offset: int
    case_id: str
    distance_group: str
    distance_bin: str
    distance_meters: float | None
    ground_truth_answer: str


def _load_existing_case_ids(run_ids: list[str]) -> set[str]:
    case_ids: set[str] = set()
    for run_id in run_ids:
        predictions_path = ROOT / "data" / "artifacts" / run_id / "predictions.jsonl"
        if not predictions_path.exists():
            continue
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                case_id = record.get("case_id")
                if isinstance(case_id, str):
                    case_ids.add(case_id)
    return case_ids


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


def build_plan(
    per_group: int,
    existing_run_ids: list[str],
    excluded_case_ids: set[str],
    excluded_offsets: set[int],
) -> dict[str, object]:
    loader = DTPQADatasetLoader(ROOT / "data" / "dtpqa")
    cases = loader.load(subset="real", question_type="category_1")
    existing_case_ids = _load_existing_case_ids(existing_run_ids)

    grouped: dict[str, list[PlannedCase]] = defaultdict(list)
    total_by_group = Counter()
    available_by_group = Counter()

    for offset, case in enumerate(cases):
        distance_group = str(case.metadata.get("distance_group") or "unknown")
        total_by_group[distance_group] += 1
        if case.case_id in existing_case_ids or case.case_id in excluded_case_ids or offset in excluded_offsets:
            continue
        available_by_group[distance_group] += 1
        grouped[distance_group].append(
            PlannedCase(
                offset=offset,
                case_id=case.case_id,
                distance_group=distance_group,
                distance_bin=str(case.metadata.get("distance_bin") or "unknown"),
                distance_meters=case.metadata.get("distance_meters"),
                ground_truth_answer=case.ground_truth_answer,
            )
        )

    selections: dict[str, list[dict[str, object]]] = {}
    group_order = ("near", "mid", "far", "unknown")
    for group in group_order:
        candidates = grouped.get(group, [])
        indices = _pick_evenly_spaced_indices(len(candidates), per_group)
        selections[group] = [asdict(candidates[index]) for index in indices]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subset": "real",
        "question_type": "category_1",
        "per_group": per_group,
        "existing_run_ids": existing_run_ids,
        "existing_case_count": len(existing_case_ids),
        "excluded_case_ids": sorted(excluded_case_ids),
        "excluded_offsets": sorted(excluded_offsets),
        "group_totals": dict(total_by_group),
        "group_available_after_exclusion": dict(available_by_group),
        "selections": selections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group", type=int, default=16)
    parser.add_argument(
        "--existing-run-id",
        action="append",
        default=[],
        help="Existing run_id to exclude from the new plan. Repeatable.",
    )
    parser.add_argument("--exclude-case-id", action="append", default=[])
    parser.add_argument("--exclude-offset", action="append", type=int, default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "dtpqa-integration" / "results" / "real_cat1_stratified_plan.json",
    )
    args = parser.parse_args()

    plan = build_plan(
        args.per_group,
        args.existing_run_id,
        set(args.exclude_case_id),
        set(args.exclude_offset),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
