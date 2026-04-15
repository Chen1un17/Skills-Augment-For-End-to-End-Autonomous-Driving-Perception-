#!/usr/bin/env python3
"""Fallback workflow figure for the benchmark-faithful and adaptation-oriented split."""

from __future__ import annotations

from matplotlib import patches
import matplotlib.pyplot as plt

from plot_common import save, set_style


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    items = [
        (0.5, "Input Case", "#E8EDF2"),
        (2.2, "Baseline\nPerception", "#FFFFFF"),
        (4.1, "Routing\nDecision", "#FFFFFF"),
        (6.1, "Cloud\nCorrection", "#FFFFFF"),
        (8.2, "Skill\nCompilation", "#FFFFFF"),
        (10.1, "Reuse /\nEvaluation", "#FFFFFF"),
    ]
    for x, label, color in items:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 1.4), 1.4, 1.0, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=color, edgecolor="#CBD5E1"
            )
        )
        ax.text(x + 0.7, 1.9, label, ha="center", va="center", fontsize=9.5, weight="bold")

    for x in [1.9, 3.8, 5.8, 7.9, 9.8]:
        ax.annotate("", xy=(x + 0.3, 1.9), xytext=(x - 0.3, 1.9), arrowprops=dict(arrowstyle="->", lw=1.6, color="#6B7280"))

    ax.add_patch(patches.Rectangle((1.8, 2.8), 4.5, 0.55, facecolor="#E8F2EE", edgecolor="none"))
    ax.text(4.05, 3.08, "Benchmark-faithful：不持久化 benchmark-derived skill，只评估系统效果", ha="center", va="center", fontsize=8.5)
    ax.add_patch(patches.Rectangle((6.0, 2.8), 4.9, 0.55, facecolor="#FFF8F0", edgecolor="none"))
    ax.text(8.45, 3.08, "Adaptation-oriented：独立评估 skill 生成与复用收益", ha="center", va="center", fontsize=8.5)

    save(fig, "fig_workflow")


if __name__ == "__main__":
    main()
