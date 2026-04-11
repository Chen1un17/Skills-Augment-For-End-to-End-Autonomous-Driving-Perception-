#!/usr/bin/env python3
"""Replay a stratified real/category_1 plan into a dedicated run_id."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]


def _iter_selected_cases(plan: dict[str, object], groups: list[str] | None) -> list[dict[str, object]]:
    requested = set(groups or [])
    selections = plan.get("selections", {})
    chosen: list[dict[str, object]] = []
    for group, items in selections.items():
        if requested and group not in requested:
            continue
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    chosen.append(item)
    return chosen


def _command(offset: int, run_id: str, execution_mode: str) -> list[str]:
    return [
        str(ROOT / ".venv" / "bin" / "ad-replay-dtpqa"),
        "--subset",
        "real",
        "--question-type",
        "category_1",
        "--offset",
        str(offset),
        "--limit",
        "1",
        "--run-id",
        run_id,
        "--execution-mode",
        execution_mode,
        "--append",
    ]


def _load_existing_case_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    existing_case_ids: set[str] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = payload.get("case_id")
            if isinstance(case_id, str) and case_id:
                existing_case_ids.add(case_id)
    return existing_case_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--max-passes",
        type=int,
        default=1,
        help="Re-run the plan until all selected case_ids appear in predictions.jsonl, up to this many passes.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=15.0,
        help="Sleep between passes when some selected case_ids are still missing from predictions.jsonl.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--execution-mode", default="hybrid")
    parser.add_argument(
        "--status-log",
        type=Path,
        default=None,
        help="Optional jsonl log path. Defaults to data/artifacts/<run_id>/batch-status.jsonl",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cases = _iter_selected_cases(plan, args.group)
    artifact_dir = ROOT / "data" / "artifacts" / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = artifact_dir / "predictions.jsonl"
    status_log = args.status_log or artifact_dir / "batch-status.jsonl"
    expected_case_ids = {str(item["case_id"]) for item in cases}

    env = os.environ.copy()
    env["REQUEST_TIMEOUT_SECONDS"] = str(args.request_timeout_seconds)
    env["MAX_RETRIES"] = str(args.max_retries)

    try:
        max_passes = max(1, args.max_passes)
        for pass_index in range(1, max_passes + 1):
            existing_case_ids = _load_existing_case_ids(predictions_path)
            pending_cases = [item for item in cases if str(item["case_id"]) not in existing_case_ids]
            if not pending_cases:
                break
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"pass={pass_index}/{max_passes} pending_cases={len(pending_cases)}",
                flush=True,
            )
            for index, item in enumerate(pending_cases, start=1):
                offset = int(item["offset"])
                case_id = str(item["case_id"])
                distance_group = str(item["distance_group"])
                started_at = datetime.now(timezone.utc).isoformat()
                print(
                    f"[{started_at}] pass={pass_index} {index}/{len(pending_cases)} replay offset={offset} "
                    f"case_id={case_id} group={distance_group}",
                    flush=True,
                )
                result = subprocess.run(
                    _command(offset, args.run_id, args.execution_mode),
                    cwd=ROOT,
                    env=env,
                    check=False,
                )
                finished_at = datetime.now(timezone.utc).isoformat()
                status_log.write_text("", encoding="utf-8") if not status_log.exists() else None
                with status_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pass_index": pass_index,
                                "offset": offset,
                                "case_id": case_id,
                                "distance_group": distance_group,
                                "ground_truth_answer": item.get("ground_truth_answer"),
                                "returncode": result.returncode,
                                "started_at": started_at,
                                "finished_at": finished_at,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(
                    f"[{finished_at}] pass={pass_index} offset={offset} case_id={case_id} returncode={result.returncode}",
                    flush=True,
                )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            remaining_case_ids = sorted(expected_case_ids - _load_existing_case_ids(predictions_path))
            if not remaining_case_ids:
                break
            if pass_index < max_passes and args.retry_sleep_seconds > 0:
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] "
                    f"pass={pass_index} incomplete; sleeping {args.retry_sleep_seconds}s before retrying "
                    f"{len(remaining_case_ids)} missing case(s)",
                    flush=True,
                )
                time.sleep(args.retry_sleep_seconds)
    except KeyboardInterrupt:
        print("Interrupted; stopping plan execution.", file=sys.stderr)
        raise SystemExit(130)

    missing_case_ids = sorted(expected_case_ids - _load_existing_case_ids(predictions_path))
    if missing_case_ids:
        print(
            f"Incomplete run {args.run_id}: missing {len(missing_case_ids)}/{len(expected_case_ids)} selected case(s): "
            + ", ".join(missing_case_ids),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
