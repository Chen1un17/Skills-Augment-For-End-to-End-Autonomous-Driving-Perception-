#!/usr/bin/env python3
"""Plot independent adaptation-oriented skill experiment summaries."""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt

from plot_common import DATA_DIR, COLORS, save, set_style


def main() -> None:
    set_style()
    with (DATA_DIR / "adaptation_summary.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    runs = list(payload.keys())
    labels = [payload[k]["run_name"] for k in runs]
    base = [payload[k]["baseline_accuracy"] for k in runs]
    final = [payload[k]["final_accuracy"] for k in runs]
    improved = [payload[k]["improved_cases"] for k in runs]
    harmed = [payload[k]["harmed_cases"] for k in runs]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(x - width / 2, base, width, label="baseline", color=COLORS[0])
    axes[0].bar(x + width / 2, final, width, label="final", color=COLORS[4])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["pilot+", "scale-risk", "small-risk"])
    axes[0].set_title("适配实验中的前后准确率")
    axes[0].legend(loc="upper right")

    axes[1].bar(x - width / 2, improved, width, label="improved", color=COLORS[1])
    axes[1].bar(x + width / 2, harmed, width, label="harmed", color=COLORS[4])
    axes[1].set_ylabel("Case count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["pilot+", "scale-risk", "small-risk"])
    axes[1].set_title("适配实验中的修复与伤害样本数")
    axes[1].legend(loc="upper right")

    save(fig, "fig_skill_adaptation")


if __name__ == "__main__":
    main()
