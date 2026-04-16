#!/usr/bin/env python3
"""Build a balanced 500-case synth DTPQA plan with smoke and refinement splits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ad_cornercase.datasets.dtpqa import DTPQADatasetLoader  # noqa: E402


GROUP_ORDER = ("far", "mid", "near", "unknown")
DEFAULT_QUOTAS = {
    "category_1": 84,
    "category_2": 84,
    "category_3": 83,
    "category_4": 83,
    "category_5": 83,
    "category_6": 83,
}


@dataclass
class PlannedCase:
    subset: str
    question_type: str
    case_id: str
    offset: int
    image_path: str
    distance_meters: float | None
    distance_group: str
    distance_bin: str
    ground_truth_answer: str
    question: str
    smoke: bool
    refinement_split: str


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
    return sorted(chosen)


def _select_group_indices(grouped_indices: dict[str, list[int]], quota: int) -> list[int]:
    selected_by_group: dict[str, list[int]] = {group: [] for group in GROUP_ORDER}
    remaining_quota = quota

    base = quota // len(GROUP_ORDER)
    remainder = quota % len(GROUP_ORDER)
    group_priority = sorted(
        GROUP_ORDER,
        key=lambda group: (-len(grouped_indices.get(group, [])), GROUP_ORDER.index(group)),
    )
    desired = {group: base for group in GROUP_ORDER}
    for group in group_priority[:remainder]:
        desired[group] += 1

    for group in GROUP_ORDER:
        available = grouped_indices.get(group, [])
        take = min(len(available), desired[group])
        if take > 0:
            chosen = _pick_evenly_spaced_indices(len(available), take)
            selected_by_group[group] = [available[index] for index in chosen]
            remaining_quota -= take

    if remaining_quota > 0:
        selected = {index for values in selected_by_group.values() for index in values}
        leftovers: list[int] = []
        for group in GROUP_ORDER:
            leftovers.extend(index for index in grouped_indices.get(group, []) if index not in selected)
        if remaining_quota > len(leftovers):
            raise ValueError(
                f"Unable to satisfy quota={quota}; only {quota - remaining_quota + len(leftovers)} candidates available."
            )
        chosen_leftovers = _pick_evenly_spaced_indices(len(leftovers), remaining_quota)
        for index in chosen_leftovers:
            selected.add(leftovers[index])

    final_indices = sorted({index for values in selected_by_group.values() for index in values})
    if len(final_indices) < quota:
        selected = set(final_indices)
        leftovers: list[int] = []
        for group in GROUP_ORDER:
            leftovers.extend(index for index in grouped_indices.get(group, []) if index not in selected)
        needed = quota - len(final_indices)
        extra = _pick_evenly_spaced_indices(len(leftovers), needed)
        final_indices.extend(leftovers[index] for index in extra)
        final_indices = sorted(set(final_indices))

    if len(final_indices) != quota:
        raise ValueError(f"Expected {quota} selected indices, found {len(final_indices)}.")
    return final_indices


def build_plan(
    *,
    dtpqa_root: Path,
    quotas: dict[str, int],
    adaptation_size: int = 40,
    smoke_per_category: int = 3,
) -> dict[str, object]:
    loader = DTPQADatasetLoader(dtpqa_root)

    planned_cases: list[dict[str, object]] = []
    smoke_case_ids: list[str] = []
    smoke_by_question_type: dict[str, list[str]] = {}
    refinement_splits: dict[str, dict[str, list[str]]] = {}
    availability: dict[str, dict[str, object]] = {}

    for question_type, quota in quotas.items():
        cases = loader.load(subset="synth", question_type=question_type)
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for offset, case in enumerate(cases):
            grouped_indices[str(case.metadata.get("distance_group") or "unknown")].append(offset)

        selected_offsets = _select_group_indices(grouped_indices, quota)
        smoke_local_indices = set(_pick_evenly_spaced_indices(len(selected_offsets), smoke_per_category))

        selected_case_ids: list[str] = []
        adaptation_case_ids: list[str] = []
        holdout_case_ids: list[str] = []

        for local_index, offset in enumerate(selected_offsets):
            case = cases[offset]
            is_smoke = local_index in smoke_local_indices
            refinement_split = "adaptation" if local_index < adaptation_size else "holdout"
            planned = PlannedCase(
                subset="synth",
                question_type=question_type,
                case_id=case.case_id,
                offset=offset,
                image_path=str(case.image_path),
                distance_meters=case.metadata.get("distance_meters"),
                distance_group=str(case.metadata.get("distance_group") or "unknown"),
                distance_bin=str(case.metadata.get("distance_bin") or "unknown"),
                ground_truth_answer=case.ground_truth_answer,
                question=case.question,
                smoke=is_smoke,
                refinement_split=refinement_split,
            )
            planned_cases.append(asdict(planned))
            selected_case_ids.append(case.case_id)
            if is_smoke:
                smoke_case_ids.append(case.case_id)
            if refinement_split == "adaptation":
                adaptation_case_ids.append(case.case_id)
            else:
                holdout_case_ids.append(case.case_id)

        smoke_by_question_type[question_type] = [case_id for case_id in selected_case_ids if case_id in smoke_case_ids]
        refinement_splits[question_type] = {
            "adaptation_case_ids": adaptation_case_ids,
            "holdout_case_ids": holdout_case_ids,
        }
        availability[question_type] = {
            "available_cases": len(cases),
            "selected_cases": len(selected_case_ids),
            "distance_group_counts": {
                group: len([item for item in planned_cases if item["question_type"] == question_type and item["distance_group"] == group])
                for group in GROUP_ORDER
            },
        }

    if len(planned_cases) != sum(quotas.values()):
        raise ValueError(f"Expected {sum(quotas.values())} planned cases, found {len(planned_cases)}.")
    unique_case_ids = {item["case_id"] for item in planned_cases}
    if len(unique_case_ids) != len(planned_cases):
        raise ValueError("Duplicate case_id detected in plan.")
    if len(smoke_case_ids) != smoke_per_category * len(quotas):
        raise ValueError(f"Expected {smoke_per_category * len(quotas)} smoke cases, found {len(smoke_case_ids)}.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subset": "synth",
        "total_cases": len(planned_cases),
        "adaptation_size_per_question_type": adaptation_size,
        "smoke_cases_per_question_type": smoke_per_category,
        "quotas": quotas,
        "availability": availability,
        "smoke_case_ids": smoke_case_ids,
        "smoke_by_question_type": smoke_by_question_type,
        "refinement_splits": refinement_splits,
        "cases": planned_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "dtpqa-integration" / "results" / "dtpqa_synth500_balanced_plan.json",
    )
    parser.add_argument("--adaptation-size", type=int, default=40)
    parser.add_argument("--smoke-per-category", type=int, default=3)
    args = parser.parse_args()

    plan = build_plan(
        dtpqa_root=ROOT / "data" / "dtpqa",
        quotas=DEFAULT_QUOTAS,
        adaptation_size=args.adaptation_size,
        smoke_per_category=args.smoke_per_category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
