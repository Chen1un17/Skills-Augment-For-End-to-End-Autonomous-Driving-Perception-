#!/usr/bin/env python3
"""Run a DTPQA synth plan selection in a resumable, plan-driven way."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT / "data" / "artifacts"


def _plan_cases(plan: dict[str, object]) -> list[dict[str, object]]:
    return [item for item in plan.get("cases", []) if isinstance(item, dict)]


def _select_cases(
    plan: dict[str, object],
    *,
    selection: str,
    question_type: str | None,
) -> list[dict[str, object]]:
    chosen: list[dict[str, object]] = []
    refinement_splits = plan.get("refinement_splits", {})
    smoke_case_ids = set(plan.get("smoke_case_ids", []))
    adaptation_case_ids: set[str] = set()
    holdout_case_ids: set[str] = set()
    if isinstance(refinement_splits, dict):
        for split_question_type, split_payload in refinement_splits.items():
            if question_type and split_question_type != question_type:
                continue
            if isinstance(split_payload, dict):
                adaptation_case_ids.update(split_payload.get("adaptation_case_ids", []))
                holdout_case_ids.update(split_payload.get("holdout_case_ids", []))

    for item in _plan_cases(plan):
        item_question_type = str(item.get("question_type") or "")
        if question_type and item_question_type != question_type:
            continue

        case_id = str(item.get("case_id") or "")
        if selection == "smoke" and case_id in smoke_case_ids:
            chosen.append(item)
        elif selection == "full":
            chosen.append(item)
        elif selection == "refinement_adaptation" and case_id in adaptation_case_ids:
            chosen.append(item)
        elif selection == "refinement_holdout" and case_id in holdout_case_ids:
            chosen.append(item)

    return chosen


def _load_existing_case_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    case_ids: set[str] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            case_id = payload.get("case_id")
            if isinstance(case_id, str) and case_id:
                case_ids.add(case_id)
    return case_ids


def _binary_path(name: str) -> str:
    direct = ROOT / ".venv" / "bin" / name
    return str(direct) if direct.exists() else name


def _replay_command(item: dict[str, object], run_id: str, execution_mode: str, server_url: str | None) -> list[str]:
    command = [
        _binary_path("ad-replay-dtpqa"),
        "--subset",
        str(item["subset"]),
        "--question-type",
        str(item["question_type"]),
        "--offset",
        str(item["offset"]),
        "--limit",
        "1",
        "--run-id",
        run_id,
        "--execution-mode",
        execution_mode,
        "--append",
    ]
    if execution_mode == "hybrid" and server_url:
        command.extend(["--server-url", server_url])
    return command


def _health_check(server_url: str) -> bool:
    try:
        with urlopen(server_url, timeout=2.0) as response:
            return response.status in {200, 406}
    except HTTPError as exc:
        return exc.code in {200, 406}
    except URLError:
        return False


def _start_server(
    *,
    env: dict[str, str],
    port: int,
    server_log_path: Path,
) -> subprocess.Popen[str]:
    try:
        existing = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        for raw_pid in existing.stdout.splitlines():
            raw_pid = raw_pid.strip()
            if raw_pid:
                subprocess.run(["kill", raw_pid], cwd=ROOT, check=False)
        if existing.stdout.strip():
            time.sleep(1)
    except FileNotFoundError:
        pass
    if shutil_which := _binary_path("ad-mcp-server"):
        process = subprocess.Popen(
            [shutil_which],
            cwd=ROOT,
            env=env,
            stdout=server_log_path.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(30):
            if _health_check(env["MCP_SERVER_URL"]):
                return process
            time.sleep(1)
        process.terminate()
        raise RuntimeError(f"MCP server failed to become healthy on port {port}")
    raise RuntimeError("Unable to locate ad-mcp-server binary.")


def _status_payload(
    *,
    run_id: str,
    selection: str,
    execution_mode: str,
    total_cases: int,
    completed_cases: int,
    failed_cases: int,
    pending_case_ids: list[str],
    current_case_id: str | None,
    state: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "selection": selection,
        "execution_mode": execution_mode,
        "state": state,
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "failed_cases": failed_cases,
        "pending_case_ids": pending_case_ids,
        "current_case_id": current_case_id,
        "progress_pct": (completed_cases / total_cases * 100.0) if total_cases else 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--selection",
        choices=("smoke", "full", "refinement_adaptation", "refinement_holdout"),
        required=True,
    )
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--execution-mode", choices=("edge_only", "cloud_only", "hybrid"), required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--skill-store-dir", type=Path, default=Path("/tmp/dtpqa_skills_empty"))
    parser.add_argument("--mcp-server-port", type=int, default=8003)
    parser.add_argument("--start-fresh-server", action="store_true")
    parser.add_argument("--disable-category1-direct-cloud", action="store_true")
    parser.add_argument("--disable-dtpqa-people-trigger", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    cases = _select_cases(plan, selection=args.selection, question_type=args.question_type)
    if not cases:
        raise SystemExit("No cases selected from plan.")

    run_dir = ARTIFACTS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    failures_path = run_dir / "failures.jsonl"
    status_path = run_dir / "experiment_status.json"
    batch_status_path = run_dir / "batch-status.jsonl"
    server_log_path = run_dir / "mcp_server.log"

    env = os.environ.copy()
    env["REQUEST_TIMEOUT_SECONDS"] = str(args.request_timeout_seconds)
    env["MAX_RETRIES"] = str(args.max_retries)
    env["SKILL_STORE_DIR"] = str(args.skill_store_dir)
    env["MCP_SERVER_PORT"] = str(args.mcp_server_port)
    env["MCP_SERVER_URL"] = f"http://127.0.0.1:{args.mcp_server_port}/mcp"
    env["ENABLE_DTPQA_PEOPLE_REFLECTION_TRIGGER"] = "0" if args.disable_dtpqa_people_trigger else "1"
    env["ENABLE_DTPQA_CATEGORY1_DIRECT_CLOUD_REROUTE"] = "0" if args.disable_category1_direct_cloud else "1"
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    for key in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    args.skill_store_dir.mkdir(parents=True, exist_ok=True)

    selected_case_ids = {str(item["case_id"]) for item in cases}
    existing_case_ids = _load_existing_case_ids(predictions_path)
    completed_selected_case_ids = existing_case_ids & selected_case_ids
    total_cases = len(cases)
    failed_cases = 0
    server_process: subprocess.Popen[str] | None = None

    try:
        if args.execution_mode == "hybrid" and args.start_fresh_server:
            server_process = _start_server(
                env=env,
                port=args.mcp_server_port,
                server_log_path=server_log_path,
            )

        for index, item in enumerate(cases, start=1):
            case_id = str(item["case_id"])
            if case_id in existing_case_ids:
                continue

            pending_case_ids = [str(entry["case_id"]) for entry in cases if str(entry["case_id"]) not in existing_case_ids]
            status_path.write_text(
                json.dumps(
                    _status_payload(
                        run_id=args.run_id,
                        selection=args.selection,
                        execution_mode=args.execution_mode,
                        total_cases=total_cases,
                        completed_cases=len(completed_selected_case_ids),
                        failed_cases=failed_cases,
                        pending_case_ids=pending_case_ids,
                        current_case_id=case_id,
                        state="running",
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            started_at = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(
                _replay_command(item, args.run_id, args.execution_mode, env["MCP_SERVER_URL"]),
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            finished_at = datetime.now(timezone.utc).isoformat()
            with batch_status_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "case_id": case_id,
                            "question_type": item["question_type"],
                            "distance_group": item["distance_group"],
                            "returncode": result.returncode,
                            "started_at": started_at,
                            "finished_at": finished_at,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if result.returncode == 0:
                existing_case_ids = _load_existing_case_ids(predictions_path)
                completed_selected_case_ids = existing_case_ids & selected_case_ids
            else:
                failed_cases += 1
                with failures_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "case_id": case_id,
                                "question_type": item["question_type"],
                                "distance_group": item["distance_group"],
                                "returncode": result.returncode,
                                "stdout_tail": result.stdout.splitlines()[-20:],
                                "stderr_tail": result.stderr.splitlines()[-20:],
                                "finished_at": finished_at,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        final_existing_case_ids = _load_existing_case_ids(predictions_path)
        completed_selected_case_ids = final_existing_case_ids & selected_case_ids
        missing_case_ids = [str(item["case_id"]) for item in cases if str(item["case_id"]) not in final_existing_case_ids]
        state = "completed" if not missing_case_ids else "failed"
        status_path.write_text(
            json.dumps(
                _status_payload(
                    run_id=args.run_id,
                    selection=args.selection,
                    execution_mode=args.execution_mode,
                    total_cases=total_cases,
                    completed_cases=len(completed_selected_case_ids),
                    failed_cases=failed_cases,
                    pending_case_ids=missing_case_ids,
                    current_case_id=None,
                    state=state,
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if missing_case_ids:
            raise SystemExit(
                f"Incomplete run {args.run_id}: missing {len(missing_case_ids)}/{total_cases} selected case(s)."
            )
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=10)


if __name__ == "__main__":
    main()
