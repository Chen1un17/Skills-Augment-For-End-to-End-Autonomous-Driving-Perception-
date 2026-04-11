#!/usr/bin/env python3
"""
Real-time monitoring dashboard for experiments.

Usage:
    python monitor_dashboard.py --run-id <run_id>
    python monitor_dashboard.py --batch-id <batch_id>
    python monitor_dashboard.py --watch  # Watch all recent experiments
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

try:
    import curses
except ImportError:
    curses = None

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from ad_cornercase.experiments import ExperimentMonitor


def format_duration(start: str | None, end: str | None = None) -> str:
    """Format duration between timestamps."""
    if not start:
        return "N/A"

    try:
        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00')) if end else datetime.now()
        delta = end_dt - start_dt
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except:
        return "N/A"


def format_number(n: float, precision: int = 2) -> str:
    """Format number for display."""
    if n is None:
        return "N/A"
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:.{precision}f}"


def create_text_dashboard(run_ids: list[str], monitor: ExperimentMonitor) -> str:
    """Create text-based dashboard."""
    lines = []
    lines.append("=" * 100)
    lines.append(" " * 35 + "EXPERIMENT MONITOR DASHBOARD")
    lines.append("=" * 100)
    lines.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for run_id in run_ids:
        metrics = monitor.analyze(run_id)
        status_path = monitor.artifacts_dir / run_id / "experiment_status.json"

        status = {"state": "unknown"}
        if status_path.exists():
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)

        # Header
        state_color = {
            "completed": "✓",
            "running": "▶",
            "failed": "✗",
            "paused": "⏸",
        }.get(status.get("state", ""), "?")

        lines.append(f"{state_color} {run_id[:60]}")
        lines.append("-" * 100)

        # Progress
        progress = status.get("progress_pct", 0)
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"  Progress: [{bar}] {progress:.1f}%")

        # Metrics
        lines.append(f"  Cases: {metrics.total_cases} | "
                    f"Accuracy: {format_number(metrics.exact_match_accuracy * 100, 1)}% | "
                    f"Judge: {format_number(metrics.judge_score_mean, 1)} | "
                    f"Latency: {format_number(metrics.mean_latency_ms / 1000, 1)}s")

        # Distance breakdown
        if metrics.distance_accuracy:
            dist_str = "  Distance: "
            for dist in ["near", "mid", "far", "unknown"]:
                acc = metrics.distance_accuracy.get(dist, 0)
                count = metrics.distance_counts.get(dist, 0)
                dist_str += f"{dist.capitalize()}={acc*100:.0f}%({count}) "
            lines.append(dist_str)

        # Timing
        lines.append(f"  Duration: {format_duration(status.get('start_time'), status.get('end_time'))}")
        lines.append("")

    lines.append("=" * 100)
    lines.append("Press Ctrl+C to exit")

    return "\n".join(lines)


def watch_experiments(run_ids: list[str] | None = None, interval: float = 5.0):
    """Watch experiments in real-time."""
    monitor = ExperimentMonitor()

    if run_ids is None:
        # Auto-discover recent runs
        artifacts_dir = Path("./data/artifacts")
        if artifacts_dir.exists():
            runs = sorted(
                [d.name for d in artifacts_dir.iterdir() if d.is_dir()],
                key=lambda x: (artifacts_dir / x).stat().st_mtime,
                reverse=True
            )
            run_ids = runs[:5]  # Last 5 runs

    if not run_ids:
        print("No experiments found to monitor")
        return

    print(f"Monitoring {len(run_ids)} experiment(s)...")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            # Clear screen (cross-platform)
            print("\033[2J\033[H", end="")

            dashboard = create_text_dashboard(run_ids, monitor)
            print(dashboard)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def generate_progress_report(run_id: str, output_path: Path | None = None):
    """Generate detailed progress report."""
    from ad_cornercase.experiments import ReportGenerator

    generator = ReportGenerator()

    output_path = output_path or Path(f"./experiments/dtpqa-integration/report/progress_{run_id}.md")
    generator.generate_progress_report(run_id, output_path)

    print(f"Progress report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment monitoring dashboard")
    parser.add_argument("--run-id", nargs="+", help="Run ID(s) to monitor")
    parser.add_argument("--batch-id", help="Batch ID to monitor")
    parser.add_argument("--watch", action="store_true", help="Watch all recent experiments")
    parser.add_argument("--interval", type=float, default=5.0, help="Update interval in seconds")
    parser.add_argument("--report", help="Generate progress report for run ID")
    parser.add_argument("--output", type=Path, help="Output path for report")

    args = parser.parse_args()

    if args.report:
        generate_progress_report(args.report, args.output)
        return

    if args.watch:
        watch_experiments(args.run_id, args.interval)
        return

    if args.run_id:
        watch_experiments(args.run_id, args.interval)
        return

    if args.batch_id:
        # Load batch status
        batch_status_path = Path(f"./experiments/dtpqa-integration/results/{args.batch_id}_status.json")
        if batch_status_path.exists():
            with open(batch_status_path, "r", encoding="utf-8") as f:
                status = json.load(f)
            run_ids = list(status.get("experiment_status", {}).keys())
            watch_experiments(run_ids, args.interval)
        else:
            print(f"Batch status not found: {batch_status_path}")
        return

    # Default: watch recent runs
    watch_experiments(None, args.interval)


if __name__ == "__main__":
    main()
