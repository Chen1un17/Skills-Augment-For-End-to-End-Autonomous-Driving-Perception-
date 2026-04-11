#!/usr/bin/env python3
"""Plot publication-ready summaries for the real/category_1 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


GROUP_ORDER = ["near", "mid", "far", "unknown"]


def load_summary(path: Path) -> dict[str, float | dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def build_bar_data(summary: dict[str, object]) -> tuple[list[str], list[float], list[float]]:
    group_summary = summary["distance_group_summary"]
    labels: list[str] = []
    accuracies: list[float] = []
    judge_scores: list[float] = []
    for group in GROUP_ORDER:
        row = group_summary.get(group)
        if row is None:
            continue
        labels.append(group)
        accuracies.append(row["exact_match_accuracy"])
        judge_scores.append(row["judge_score_mean_judged_only"])
    return labels, accuracies, judge_scores


def plot_summary(summary_path: Path, output_dir: Path) -> None:
    summary = load_summary(summary_path)
    labels, accuracies, judge_scores = build_bar_data(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    palette = ["#2563EB", "#059669", "#D97706", "#7C3AED"]

    axes[0].bar(labels, accuracies, color=palette[: len(labels)])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact Match Accuracy")
    axes[0].set_title("Distance-Group Accuracy (real category_1)")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.6)

    axes[1].bar(labels, judge_scores, color=palette[: len(labels)])
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Judge Score (mean of judged cases)")
    axes[1].set_title("Judge Score by Distance Group")
    axes[1].grid(axis="y", linestyle=":", linewidth=0.6)

    for ax in axes:
        ax.set_xlabel("Distance Group")
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation=0)

    figure_path = output_dir / "real_cat1_distance_group_summary.png"
    fig.savefig(figure_path, dpi=200)
    print(figure_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("experiments/dtpqa-integration/results/current_real_cat1_pilot_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/dtpqa-integration/results"),
    )
    args = parser.parse_args()

    plot_summary(args.summary_json, args.output_dir)


if __name__ == "__main__":
    main()
