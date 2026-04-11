"""Academic report generation for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass

from .monitor import ExperimentMonitor, ExperimentMetrics


@dataclass
class TableConfig:
    """Configuration for table generation."""
    caption: str
    label: str
    columns: list[str]
    rows: list[list[Any]]


class ReportGenerator:
    """Generate academic-quality reports and visualizations."""

    def __init__(self, artifacts_dir: Path = Path("./data/artifacts")):
        self.artifacts_dir = artifacts_dir
        self.monitor = ExperimentMonitor(artifacts_dir)

    def _format_value(self, value: float, precision: int = 2) -> str:
        """Format numeric value for display."""
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return str(value)

    def generate_latex_table(self, run_ids: list[str], title: str = "Results") -> str:
        """Generate LaTeX table comparing runs."""
        metrics_list = [self.monitor.analyze(rid) for rid in run_ids]

        lines = [
            "% Auto-generated LaTeX table",
            f"\\begin{{table}}[t]",
            f"\\centering",
            f"\\caption{{{title}}}",
            f"\\label{{tab:{title.lower().replace(' ', '_')}}}",
            "\\begin{tabular}{l" + "c" * len(run_ids) + "}",
            "\\toprule",
        ]

        # Header
        header = "Metric & " + " & ".join([f"\\texttt{{{m.run_id[:20]}}}" for m in metrics_list]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Rows
        rows = [
            ("Total Cases", [m.total_cases for m in metrics_list], 0),
            ("Exact Match Acc.", [m.exact_match_accuracy for m in metrics_list], 3),
            ("Judge Score Mean", [m.judge_score_mean for m in metrics_list], 2),
            ("Mean Latency (s)", [m.mean_latency_ms / 1000 for m in metrics_list], 1),
        ]

        for name, values, prec in rows:
            row_values = " & ".join([self._format_value(v, prec) for v in values])
            lines.append(f"{name} & {row_values} \\\\")

        # Distance-stratified accuracy
        lines.append("\\midrule")
        lines.append("\\multicolumn{" + str(len(run_ids) + 1) + "}{c}{Distance-Stratified Accuracy} \\\\")
        lines.append("\\midrule")

        for dist in ["near", "mid", "far", "unknown"]:
            values = [m.distance_accuracy.get(dist, 0) for m in metrics_list]
            row_values = " & ".join([self._format_value(v, 3) for v in values])
            lines.append(f"{dist.capitalize()} & {row_values} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    def generate_markdown_table(self, run_ids: list[str], title: str = "Results") -> str:
        """Generate Markdown table comparing runs."""
        metrics_list = [self.monitor.analyze(rid) for rid in run_ids]

        lines = [
            f"## {title}",
            "",
            "| Metric | " + " | ".join([m.run_id[:25] for m in metrics_list]) + " |",
            "|--------|" + "|".join(["--------"] * len(run_ids)) + "|",
        ]

        # Overall metrics
        rows = [
            ("Total Cases", [m.total_cases for m in metrics_list], 0),
            ("Exact Match Acc.", [m.exact_match_accuracy for m in metrics_list], 3),
            ("Judge Score Mean", [m.judge_score_mean for m in metrics_list], 2),
            ("Mean Latency (s)", [m.mean_latency_ms / 1000 for m in metrics_list], 1),
        ]

        for name, values, prec in rows:
            row_values = " | ".join([self._format_value(v, prec) for v in values])
            lines.append(f"| {name} | {row_values} |")

        # Distance-stratified
        lines.extend(["", "### Distance-Stratified Accuracy", ""])
        lines.append("| Distance | " + " | ".join([m.run_id[:25] for m in metrics_list]) + " |")
        lines.append("|----------|" + "|".join(["--------"] * len(run_ids)) + "|")

        for dist in ["near", "mid", "far", "unknown"]:
            values = [m.distance_accuracy.get(dist, 0) for m in metrics_list]
            row_values = " | ".join([self._format_value(v, 3) for v in values])
            lines.append(f"| {dist.capitalize()} | {row_values} |")

        return "\n".join(lines)

    def generate_comparison_plot(self, run_ids: list[str], output_path: Path) -> None:
        """Generate comparison plot using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[WARN] matplotlib not available, skipping plot generation")
            return

        metrics_list = [self.monitor.analyze(rid) for rid in run_ids]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")

        run_labels = [m.run_id[:15] for m in metrics_list]
        x = np.arange(len(run_labels))

        # Plot 1: Overall metrics
        ax = axes[0, 0]
        accuracies = [m.exact_match_accuracy * 100 for m in metrics_list]
        judge_scores = [m.judge_score_mean for m in metrics_list]

        width = 0.35
        ax.bar(x - width/2, accuracies, width, label="Accuracy (%)", alpha=0.8)
        ax.bar(x + width/2, judge_scores, width, label="Judge Score", alpha=0.8)
        ax.set_xlabel("Run")
        ax.set_ylabel("Score")
        ax.set_title("Overall Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Plot 2: Distance-stratified accuracy
        ax = axes[0, 1]
        distances = ["near", "mid", "far", "unknown"]
        dist_data = {dist: [m.distance_accuracy.get(dist, 0) * 100 for m in metrics_list] for dist in distances}

        x_dist = np.arange(len(distances))
        width = 0.15
        for i, run_label in enumerate(run_labels):
            values = [dist_data[dist][i] for dist in distances]
            ax.bar(x_dist + i * width, values, width, label=run_label, alpha=0.8)

        ax.set_xlabel("Distance Group")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Distance-Stratified Accuracy")
        ax.set_xticks(x_dist + width * (len(run_labels) - 1) / 2)
        ax.set_xticklabels(distances)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Plot 3: Latency comparison
        ax = axes[1, 0]
        latencies = [m.mean_latency_ms / 1000 for m in metrics_list]
        colors = ["green" if l < 30 else "orange" if l < 60 else "red" for l in latencies]
        ax.bar(run_labels, latencies, color=colors, alpha=0.7)
        ax.set_xlabel("Run")
        ax.set_ylabel("Latency (seconds)")
        ax.set_title("Mean Latency")
        ax.set_xticklabels(run_labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Plot 4: Sample counts by distance
        ax = axes[1, 1]
        dist_counts = {dist: [m.distance_counts.get(dist, 0) for m in metrics_list] for dist in distances}

        bottom = np.zeros(len(run_labels))
        colors_dist = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]
        for i, dist in enumerate(distances):
            ax.bar(run_labels, dist_counts[dist], bottom=bottom, label=dist, color=colors_dist[i], alpha=0.8)
            bottom += np.array(dist_counts[dist])

        ax.set_xlabel("Run")
        ax.set_ylabel("Sample Count")
        ax.set_title("Samples by Distance Group")
        ax.set_xticklabels(run_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved comparison plot to {output_path}")

    def generate_full_report(
        self,
        run_ids: list[str],
        output_dir: Path,
        title: str = "Experiment Report",
    ) -> Path:
        """Generate full academic report with all tables and plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate markdown report
        report_path = output_dir / f"report_{timestamp}.md"

        lines = [
            f"# {title}",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Runs analyzed:** {len(run_ids)}",
            "",
            "## Summary",
            "",
        ]

        # Add comparison table
        lines.append(self.generate_markdown_table(run_ids, "Experiment Comparison"))
        lines.append("")

        # Individual run details
        lines.append("## Individual Run Details")
        lines.append("")

        for run_id in run_ids:
            metrics = self.monitor.analyze(run_id)
            lines.extend([
                f"### {run_id}",
                "",
                f"- **Total Cases:** {metrics.total_cases}",
                f"- **Exact Match Accuracy:** {metrics.exact_match_accuracy:.3f}",
                f"- **Judge Score Mean:** {metrics.judge_score_mean:.2f} (±{metrics.judge_score_std:.2f})",
                f"- **Mean Latency:** {metrics.mean_latency_ms/1000:.1f}s",
                f"- **Reflection Trigger Rate:** {metrics.reflection_trigger_rate:.3f}",
                f"- **Skill Success Rate:** {metrics.skill_success_rate:.3f}",
                "",
                "#### Distance-Stratified Metrics",
                "",
                "| Distance | Count | Accuracy | Judge Mean | Latency (s) |",
                "|----------|-------|----------|------------|-------------|",
            ])

            for dist in ["near", "mid", "far", "unknown"]:
                count = metrics.distance_counts.get(dist, 0)
                acc = metrics.distance_accuracy.get(dist, 0)
                judge = metrics.distance_judge_mean.get(dist, 0)
                lat = metrics.distance_latency_mean.get(dist, 0) / 1000
                lines.append(f"| {dist.capitalize()} | {count} | {acc:.3f} | {judge:.1f} | {lat:.1f} |")

            lines.append("")

        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[INFO] Saved report to {report_path}")

        # Generate LaTeX tables
        latex_path = output_dir / f"tables_{timestamp}.tex"
        latex_content = self.generate_latex_table(run_ids, title)
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        print(f"[INFO] Saved LaTeX tables to {latex_path}")

        # Generate plots
        plot_path = output_dir / f"comparison_{timestamp}.png"
        self.generate_comparison_plot(run_ids, plot_path)

        # Export raw metrics
        metrics_path = output_dir / f"metrics_{timestamp}.json"
        all_metrics = {rid: self.monitor.analyze(rid).to_dict() for rid in run_ids}
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved raw metrics to {metrics_path}")

        return report_path

    def generate_progress_report(
        self,
        run_id: str,
        output_path: Path,
    ) -> None:
        """Generate interim progress report for a running experiment."""
        metrics = self.monitor.analyze(run_id)
        status_path = self.artifacts_dir / run_id / "experiment_status.json"

        status = {}
        if status_path.exists():
            with open(status_path, "r", encoding="utf-8") as f:
                status = json.load(f)

        lines = [
            f"# Progress Report: {run_id}",
            "",
            f"**Status:** {status.get('state', 'unknown')}",
            f"**Progress:** {status.get('progress_pct', 0):.1f}%",
            f"**Last Updated:** {datetime.now().isoformat()}",
            "",
            "## Current Metrics",
            "",
            f"- **Cases Completed:** {metrics.total_cases} / {status.get('total_cases', 'unknown')}",
            f"- **Exact Match Accuracy:** {metrics.exact_match_accuracy:.3f}",
            f"- **Judge Score Mean:** {metrics.judge_score_mean:.2f}",
            f"- **Mean Latency:** {metrics.mean_latency_ms/1000:.1f}s",
            "",
            "## Distance-Stratified Accuracy",
            "",
            "| Distance | Count | Accuracy |",
            "|----------|-------|----------|",
        ]

        for dist in ["near", "mid", "far", "unknown"]:
            count = metrics.distance_counts.get(dist, 0)
            acc = metrics.distance_accuracy.get(dist, 0)
            lines.append(f"| {dist.capitalize()} | {count} | {acc:.3f} |")

        lines.extend([
            "",
            "## Answer Distribution",
            "",
            "| Answer | Count |",
            "|--------|-------|",
        ])

        for answer, count in sorted(metrics.answer_distribution.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"| {answer[:50]} | {count} |")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[INFO] Saved progress report to {output_path}")
